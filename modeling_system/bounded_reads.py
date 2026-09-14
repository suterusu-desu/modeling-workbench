"""Read-only, revision-bound JSON windows. No storage writes or native access."""
import copy
import json
from .store import canonical, digest

MAX_READ_CHARS = 12000
DEFAULT_READ_CHARS = 8000


def json_chars(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-16-le')) // 2


def validate_window(path, offset, limit, max_chars):
    if not isinstance(path, list) or len(path) > 24:
        raise ValueError('Read path must be a list of at most 24 keys/indices')
    for part in path:
        if type(part) is int:
            if part < 0: raise ValueError('Read indices must be nonnegative')
        elif not isinstance(part, str) or len(part) > 512:
            raise ValueError('Read path entries must be keys of at most 512 characters or nonnegative indices')
    if json_chars(path) > 1500: raise ValueError('Read path exceeds 1500 JSON characters')
    for name, value, low, high in [('offset', offset, 0, 10**12), ('limit', limit, 1, 1024),
                                   ('max_chars', max_chars, 2048, MAX_READ_CHARS)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an integer from {low} to {high}')


def select(value, path):
    for part in path:
        if isinstance(value, dict):
            key = sorted(value)[part] if type(part) is int else part
            value = value[key]
        elif isinstance(value, list) and type(part) is int:
            value = value[part]
        else:
            raise ValueError('Read path cannot descend through this value type')
    return value


def read_descriptor(operation, arguments, path=None, **window):
    return {'operation': operation, 'arguments': {**arguments, 'path': path or [], **window}}


def normalize_reads(reads, default_name='detail'):
    """Copy named read descriptors, or wrap one legacy descriptor; never execute it.

    This normalizes descriptor shape only. It preserves arguments and budgets,
    and does not determine an operation's authority, effects or live freshness.
    """
    if not isinstance(default_name,str) or not 1<=len(default_name)<=64:
        raise ValueError('A short nonempty fallback expansion name is required')
    if not isinstance(reads,dict) or len(reads)>32 or json_chars(reads)>16384:
        raise ValueError('Read descriptors must be a bounded object')
    named={default_name:reads} if set(reads)=={'operation','arguments'} and isinstance(reads['operation'],str) else reads
    for name,descriptor in named.items():
        if not isinstance(name,str) or not 1<=len(name)<=64:
            raise ValueError('Expansion names must be short nonempty strings')
        if (not isinstance(descriptor,dict) or set(descriptor)!={'operation','arguments'}
                or not isinstance(descriptor['operation'],str) or not 1<=len(descriptor['operation'])<=128
                or not isinstance(descriptor['arguments'],dict)):
            raise ValueError('Each expansion needs an operation string and arguments object')
    return copy.deepcopy(named)


def page(value, operation, arguments, *, path=None, offset=0, limit=20,
         max_chars=DEFAULT_READ_CHARS, source_revision=None, expected_view=None,
         selection_basis='Exact retained JSON', counts=None, view_revision=None):
    path = [] if path is None else path
    validate_window(path, offset, limit, max_chars)
    if expected_view is not None and (not isinstance(expected_view, str) or len(expected_view) != 64
                                     or any(c not in '0123456789abcdef' for c in expected_view)):
        raise ValueError('expected_view must be a 64-character content fingerprint')
    revision = source_revision or view_revision or digest(canonical(value))
    if expected_view is not None and expected_view != revision:
        return {'status': 'conflicting', 'error': 'changed_read_view', 'expected_view': expected_view,
                'source_revision': revision, 'coverage': 'No rows returned from the changed view',
                'refresh': read_descriptor(operation, arguments, path, offset=0, limit=limit, max_chars=max_chars)}
    selected = select(value, path)
    kind = ('object' if isinstance(selected, dict) else 'array' if isinstance(selected, list)
            else 'string' if isinstance(selected, str) else 'scalar')
    if kind in ('array', 'object') and limit > 64:
        raise ValueError('Container windows permit at most 64 items')
    total = len(selected) if kind != 'scalar' else 1
    if offset > total: raise IndexError('Read offset exceeds the selected value length')
    common = {**arguments, 'expected_view': revision} if source_revision is None else dict(arguments)
    result = {'projection': 'bounded_read', 'source_revision': revision, 'path': path, 'type': kind,
              'selection_basis': selection_basis, 'total': total, 'offset': offset, 'shown': 0,
              'deferred': total, 'content_complete': False, 'max_chars': max_chars,
              'coverage': 'Shown counts selected slots, not recursively expanded content; follow deferred-value reads',
              'items': []}
    if counts is not None: result['counts'] = counts
    # Reserve room for continuation and the public execute() envelope.
    allowance = max_chars - 700 - json_chars(common) - json_chars(path)
    if allowance < json_chars(result) + 120:
        raise ValueError('Read path/metadata leaves insufficient room; increase max_chars or shorten the path')
    if kind == 'scalar':
        if offset == 0: result['items'] = [selected]; result['shown'] = 1
    elif kind == 'string':
        take = min(limit, total-offset)
        while take and json_chars({**result, 'items': [selected[offset:offset+take]]}) > allowance:
            take //= 2
        if total > offset and not take: raise ValueError('String cannot fit the selected read allowance')
        result['items'] = [selected[offset:offset+take]]
        result['shown'] = take
        result['unit'] = 'Unicode code points'
    else:
        keys = sorted(selected) if kind == 'object' else None
        for index in range(offset, min(total, offset+limit)):
            key = keys[index] if keys is not None else index
            child = selected[key]
            child_path = path + [index]  # Object ordinals use sorted keys; no evaluated expressions.
            entry = {'index': index, 'value': child}
            if keys is not None:
                entry.update(key=key if len(key) <= 96 else key[:96], key_length=len(key))
            if json_chars({**result, 'items': result['items'] + [entry]}) > allowance:
                entry['value'] = {'deferred_value': True, 'type': type(child).__name__,
                                  'expand': read_descriptor(operation, common, child_path,
                                      limit=1024 if isinstance(child, str) else 20, max_chars=max_chars)}
            if json_chars({**result, 'items': result['items'] + [entry]}) > allowance:
                break
            result['items'].append(entry); result['shown'] += 1
    result['deferred'] = total-result['shown']
    end = offset+result['shown']
    if end < total:
        result['next'] = read_descriptor(operation, common, path, offset=end, limit=limit, max_chars=max_chars)
    result['content_complete'] = (offset == 0 and end == total and
        not any(isinstance(item, dict) and isinstance(item.get('value'), dict) and
                item['value'].get('deferred_value') is True for item in result['items']))
    result['remaining_after_window'] = total-end
    if total > offset and not result['shown']: raise ValueError('Read window cannot make progress within max_chars')
    if json_chars(result) > max_chars-128: raise ValueError('Read response metadata exceeds max_chars')
    return result


def record(service, key, path, offset, limit, max_chars):
    return page(service.store.get(key), 'read_record', {'record': key}, path=path, offset=offset,
                limit=limit, max_chars=max_chars, source_revision=key,
                selection_basis='Immutable content ID; object integer selectors use sorted key order')
