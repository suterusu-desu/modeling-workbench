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
    spec = item.get('parameters', {}).get('arguments', {})
    if not isinstance(spec, dict):
        raise ValueError('Named typed arguments required')
    paths = []
    for name, arg in spec.items():
        path, options = arg.get('path'), arg.get('options')
        if (not isinstance(name, str) or not name or not arg.get('description')
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
    for rule in item.get('parameters', {}).get('incompatible', []):
        if not isinstance(rule, dict) or not rule or not set(rule) <= set(spec):
            raise ValueError('Compatibility rules name existing argument choices')
        for name, value in rule.items():
            allowed = {o['id'] for o in spec[name]['options']} | {DEFAULT}
            if value not in allowed:
                raise ValueError('Compatibility rule names an unavailable argument option')
    json.dumps(item.get('parameters', {}), allow_nan=False)
    return spec






def compile_call(item, call):
    """Materialize an exact handler payload after current dependencies are checked."""
    if (call.get('status') != 'bound' or call['task'] != item['id']
            or call['definition'] != fingerprint(item)
            or call['specification'] != fingerprint(item.get('parameters', {}))):
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




def bind_call(item, selected):
    """Bind explicit owner choices before offering a task for execution.

    Selected values are closed-set option IDs (or a list for a set argument).
    No question is sent to a provider, and omission never means approval.
    """
    spec = validate_call(item)
    if not isinstance(selected, dict) or set(selected) != set(spec):
        raise ValueError('Explicit choices for every argument are required')
    values = {}
    for name, arg in spec.items():
        choice = selected[name]
        options = {o['id']: o['value'] for o in arg['options']}
        if arg.get('type', 'choice') == 'set':
            if (not isinstance(choice, list) or any(not isinstance(k, str) for k in choice)
                    or len(set(choice)) != len(choice) or not set(choice) <= set(options)
                    or not choice and not arg.get('allow_empty', False)):
                raise ValueError('A distinct qualified set is required')
            values[name] = [deepcopy(options[k]) for k in choice]
        elif isinstance(choice, str) and choice in options:
            values[name] = deepcopy(options[choice])
        elif choice == DEFAULT and 'default' in arg:
            values[name] = deepcopy(arg['default'])
        else:
            raise ValueError('Owner choice is outside the qualified options')
    for rule in item.get('parameters', {}).get('incompatible', []):
        if all(v in selected[k] if isinstance(selected[k], list) else v == selected[k] for k,v in rule.items()):
            raise ValueError('Owner choices violate declared compatibility')
    call = {'status': 'bound', 'task': item['id'], 'definition': fingerprint(item),
            'specification': fingerprint(item.get('parameters', {})),
            'selected': deepcopy(selected), 'arguments': values}
    return compile_call(item, call)
