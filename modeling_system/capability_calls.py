"""Typed arguments for existing queue handlers; no new executor or native access.

An argument's description is public; its exact value and payload destination stay
local. All candidate dependencies are guarded before a selected call is compiled.
"""
from copy import deepcopy
import json

from .controller import fingerprint

DEFAULT = '__default__'
UNKNOWN = '__unknown__'


def validate_call(item):
    spec = item.get('decision', {}).get('arguments', {})
    if not isinstance(spec, dict):
        raise ValueError('Named typed arguments required')
    paths = []
    for name, arg in spec.items():
        path, options = arg.get('path'), arg.get('options')
        if (not isinstance(name, str) or not name or not arg.get('question')
                or not isinstance(path, list) or not path or any(not isinstance(k, str) or not k for k in path)
                or not isinstance(options, list) or not options or len(options) > 253
                or arg.get('type', 'choice') not in ('choice', 'set')):
            raise ValueError('Arguments need meaning, a payload path and qualified closed-set options')
        if any(path[:len(p)] == p or p[:len(path)] == path for p in paths):
            raise ValueError('Argument payload destinations must not overlap')
        paths.append(path)
        ids = [o.get('id') for o in options]
        if (any(not isinstance(k, str) or not k or k in (DEFAULT, UNKNOWN) for k in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('Unique argument option identities required')
        for option in options:
            if not option.get('description') or 'value' not in option:
                raise ValueError('Qualified options need public descriptions and exact local values')
            if any(item['reads'].get(k) != v for k, v in option.get('reads', {}).items()):
                raise ValueError('Every argument option must bind its exact dependencies in the task')
        if 'default' in arg and arg.get('type', 'choice') == 'set':
            raise ValueError('Set arguments use explicit empty-set semantics, not implicit defaults')
    for rule in item.get('decision', {}).get('incompatible', []):
        if not isinstance(rule, dict) or not rule or not set(rule) <= set(spec):
            raise ValueError('Compatibility rules name existing argument choices')
        for name, value in rule.items():
            allowed = {o['id'] for o in spec[name]['options']} | {DEFAULT}
            if value not in allowed:
                raise ValueError('Compatibility rule names an unavailable argument option')
    json.dumps(item.get('decision', {}), allow_nan=False)
    return spec


def argument_questions(action, ordinal):
    """Batch conditional arguments; only the selected operation consumes them."""
    rows, mapping = [], {}
    for index, (name, arg) in enumerate(action.get('decision', {}).get('arguments', {}).items()):
        key = f'call_{ordinal}_arg_{index}'
        question = {'condition': 'Only if the described operation is selected.',
                    'operation': action['public_description'], 'question': arg['question'],
                    'rule': 'Use the supplied current facts and qualified options. Missing support stays unknown.'}
        if arg.get('type', 'choice') == 'set':
            for i, option in enumerate(arg['options']):
                qid = key + '_' + str(i)
                rows.append({'id': qid, 'type': 'noul', 'instructions': {
                    **question, 'candidate': option['description'],
                    'question': 'Should this candidate be included for the stated argument role?',
                    'argument_role': arg['question']}})
                mapping[qid] = {'argument': name, 'member': option['id']}
        else:
            options = [{'id': o['id'], 'description': o['description']} for o in arg['options']]
            if 'default' in arg:
                options.append({'id': DEFAULT, 'description': 'No override is supported or needed; use the declared default.'})
            options.append({'id': UNKNOWN, 'description': 'Required support is missing or contradictory; no qualified binding.'})
            rows.append({'id': key, 'type': 'choice', 'instructions': question, 'options': options})
            mapping[key] = {'argument': name}
    return rows, mapping


def resolve_call(action, mapping, judgments):
    spec = action.get('decision', {}).get('arguments', {})
    selected, values, used = {}, {}, {}
    for key, row in mapping.items():
        answer = judgments[key]
        name = row['argument']
        used[key] = deepcopy(answer)
        if 'member' in row:
            probability = answer['noul']
            if probability == .5:
                return {'status': 'unresolved', 'reason': 'A selected set member has no supported yes/no preference'}
            selected.setdefault(name, [])
            if probability > .5:
                selected[name].append(row['member'])
        else:
            selected[name] = answer['choice']
    for name, arg in spec.items():
        choice = selected[name]
        options = {o['id']: o['value'] for o in arg['options']}
        if choice == UNKNOWN:
            return {'status': 'unresolved', 'reason': 'Selected operation has an unsupported required argument'}
        if isinstance(choice, list):
            if not choice and not arg.get('allow_empty', False):
                return {'status': 'unresolved', 'reason': 'Selected operation requires a nonempty qualified set'}
            values[name] = [deepcopy(options[k]) for k in choice]
        elif choice == DEFAULT and 'default' in arg:
            values[name] = deepcopy(arg['default'])
        elif choice in options:
            values[name] = deepcopy(options[choice])
        else:
            raise ValueError('Argument answer outside qualified options')
    for rule in action.get('decision', {}).get('incompatible', []):
        if all(v in selected[k] if isinstance(selected[k], list) else v == selected[k] for k, v in rule.items()):
            return {'status': 'unresolved', 'reason': 'Selected arguments violate declared compatibility'}
    return {'status': 'bound', 'task': action['id'], 'definition': action['revision'],
            'specification': fingerprint(action.get('decision', {})), 'selected': selected,
            'arguments': values, 'used_judgments': used}


def compile_call(item, call):
    """Materialize an exact handler payload after current dependencies are checked."""
    if (call.get('status') != 'bound' or call['task'] != item['id']
            or call['definition'] != fingerprint(item)
            or call['specification'] != fingerprint(item.get('decision', {}))):
        raise ValueError('Selected call no longer matches the exact task specification')
    spec = validate_call(item)
    if set(spec) != set(call['arguments']):
        raise ValueError('Selected call is incomplete')
    for name, arg in spec.items():
        selected = call['selected'].get(name)
        options = {o['id']: o['value'] for o in arg['options']}
        if isinstance(selected, list):
            expected = [options[k] for k in selected]
        elif selected == DEFAULT and 'default' in arg:
            expected = arg['default']
        else:
            expected = options[selected]
        if fingerprint(expected) != fingerprint(call['arguments'][name]):
            raise ValueError('Compiled argument differs from the qualified selected value')
    result = deepcopy(item)
    for name, arg in spec.items():
        dest = result.setdefault('payload', {})
        for part in arg['path'][:-1]:
            dest = dest.setdefault(part, {})
            if not isinstance(dest, dict):
                raise ValueError('Argument destination is not a payload object')
        dest[arg['path'][-1]] = deepcopy(call['arguments'][name])
    result['workbench']['invocation'] = deepcopy(call)
    return result


class CapabilityCatalog:
    """Decorate an ordinary catalog with typed operation argument specifications."""
    def __init__(self, catalog):
        self.catalog = catalog

    def __call__(self, state, outcomes):
        items = deepcopy(self.catalog(state, outcomes))
        for item in items:
            if item.get('decision', {}).get('arguments'):
                validate_call(item)
                if item.get('required'):
                    raise ValueError('Typed modeling choices cannot be required housekeeping')
                item['select_with_jev'] = True
        return items
