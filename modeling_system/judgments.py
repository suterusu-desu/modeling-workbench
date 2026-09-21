"""Provider-neutral bounded judgments for planning as well as native actions.

Only reviewed public state/questions go on wire. Bindings and identifier maps
stay local. This module does no inference, authorization or native execution.
"""
from copy import deepcopy
import json
import math
from .controller import fingerprint


PLANNING_QUESTIONS = {
    'priority': 'Which offered unresolved issue should be addressed next under the stated objective, observed impact and current evidence?',
    'method': 'Which offered retained method applies to this observed mechanism, including its prerequisites, exclusions and previous failures?',
    'evidence': 'Which offered observation would most directly resolve the stated uncertainty and change the next action? Prefer reusable sufficient evidence.',
    'recovery': 'Which offered recovery procedure matches the observed failure and known effects? Never replay an uncertain effect.',
    'route': 'Which offered handler can resolve this question with the supplied evidence and existing capabilities?',
}


def planning_choice(kind, candidates):
    """Build a reusable planning question with an explicit abstention option.

    Candidates are locally qualified and authorized first. Their descriptions
    contain public applicability facts; an ID is not an instruction to the model.
    """
    if kind not in PLANNING_QUESTIONS:
        raise ValueError('Unknown planning question kind')
    if not candidates or any(c['id'] == 'needs_astra' for c in candidates):
        raise ValueError('Supply actual candidates and reserve needs_astra')
    return {'id': kind, 'type': 'choice',
            'instructions': PLANNING_QUESTIONS[kind] +
                ' Choose none applies when evidence conflicts, context is insufficient, or a new mechanism or visual interpretation is needed. Do not grant authority or accept appearance.',
            'options': deepcopy(candidates) + [{'id': 'needs_astra',
                'description': 'None applies: return the unresolved question and retained evidence to the reasoning owner.'}]}


def _description(value, nullable=False):
    return (nullable and value is None) or (isinstance(value, (str, dict, list)) and bool(value))


def validate_questions(questions):
    if not isinstance(questions, dict) or not 1 <= len(questions) <= 12:
        raise ValueError('One to twelve useful independent questions per batch')
    for key, q in questions.items():
        if not isinstance(key, str) or not key or not isinstance(q, dict):
            raise ValueError('Named typed questions required')
        if set(q) - {'type', 'instructions', 'criteria'} or not _description(q.get('instructions')):
            raise ValueError('Explicit instructions and supported fields required')
        kind, criteria = q.get('type'), q.get('criteria')
        if kind == 'choice':
            valid = (isinstance(criteria, dict) and 2 <= len(criteria) <= 255
                     and all(isinstance(k, str) and k and _description(v, True) for k, v in criteria.items()))
        elif kind == 'score':
            valid = isinstance(criteria, list) and 2 <= len(criteria) <= 10 and all(_description(v) for v in criteria)
        elif kind == 'noul':
            valid = ('criteria' not in q or (isinstance(criteria, dict)
                     and set(criteria) == {'true', 'false'} and all(_description(v) for v in criteria.values())))
        else:
            valid = False
        if not valid:
            raise ValueError('Invalid typed question criteria')
    json.dumps(questions, allow_nan=False)


def prepare_judgments(public_state, decisions, local_binding):
    """Compile declarative private decisions to the existing bounded packet shape."""
    if not isinstance(public_state, dict):
        raise ValueError('Reviewed compact public state object required')
    if not decisions or len({d['id'] for d in decisions}) != len(decisions):
        raise ValueError('Unique local decision IDs required')
    binding = deepcopy(local_binding)
    reads = binding.get('dependencies', {}).get('reads')
    writes = binding.get('dependencies', {}).get('writes')
    if (not binding.get('owner') or not binding.get('authority_revision')
            or not isinstance(reads, dict) or not reads
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in reads.items())
            or not isinstance(writes, list) or not set(writes) <= set(reads)):
        raise ValueError('Owner, authority and complete dependency read/write binding required')
    questions, mapping = {}, {}
    for index, decision in enumerate(decisions):
        key = 'q' + str(index)
        if not isinstance(decision['id'], str) or not decision['id']:
            raise ValueError('Nonempty local decision ID required')
        q = {'type': decision['type'], 'instructions': deepcopy(decision['instructions'])}
        row = {'id': decision['id'], 'type': decision['type']}
        if decision['type'] == 'choice':
            options = decision['options']
            if len({o['id'] for o in options}) != len(options) or any(not isinstance(o['id'], str) or not o['id'] for o in options):
                raise ValueError('Unique nonempty local option IDs required')
            row['options'] = {'o' + str(i): o['id'] for i, o in enumerate(options)}
            q['criteria'] = {'o' + str(i): deepcopy(o['description']) for i, o in enumerate(options)}
        elif 'criteria' in decision:
            q['criteria'] = deepcopy(decision['criteria'])
        questions[key], mapping[key] = q, row
    validate_questions(questions)
    if len(json.dumps({'state': public_state, 'questions': questions}, allow_nan=False).encode()) > 11000:
        raise ValueError('Filter irrelevant state before dispatch; compact batch exceeded')
    return {'state': deepcopy(public_state), 'questions': questions,
            'local_binding': binding, 'dispatch_mapping': mapping}


def _number(value, lower, upper):
    return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper


def validate_answers(questions, answers):
    """Validate full distributions; return values without discarding source answers."""
    validate_questions(questions)
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError('Exact answer set required')
    values = {}
    for key, q in questions.items():
        answer, kind = answers[key], q['type']
        if not isinstance(answer, dict) or answer.get('type') != kind:
            raise ValueError('Answer type mismatch')
        if kind == 'noul':
            value = answer.get('noul')
            if not _number(value, 0, 1):
                raise ValueError('Invalid Noul probability')
        else:
            expected = (q['criteria'] if kind == 'choice' else
                        {str(i): text for i, text in enumerate(q['criteria'])})
            probs = answer.get('probabilities')
            if (not isinstance(probs, dict) or set(probs) != set(expected)
                    or any(not _number(v, 0, 1) for v in probs.values())
                    or abs(sum(probs.values()) - 1) > 1e-5
                    or not _number(answer.get('confidence'), 0, 1)):
                raise ValueError('Invalid complete probability distribution or confidence')
            if kind == 'choice':
                value = answer.get('choice')
                if not isinstance(value, str) or value not in expected or probs[value] < max(probs.values()) - 1e-8:
                    raise ValueError('Choice must be a maximum-probability offered option')
            else:
                value = answer.get('score')
                weighted = sum(int(k) * p for k, p in probs.items())
                if (answer.get('legend') != expected or not _number(value, 0, len(expected) - 1)
                        or abs(value - weighted) > .011):
                    raise ValueError('Score/legend must match ordered levels and their expectation')
        values[key] = value
    return values


def resolve_judgments(packet, answers, current_binding):
    """Read-only decisions, never an appearance approval or permission token."""
    if fingerprint(packet['local_binding']) != fingerprint(current_binding):
        raise ValueError('Planning authority or relevant evidence changed')
    values = validate_answers(packet['questions'], answers)
    result = {}
    for key, value in values.items():
        row = packet['dispatch_mapping'][key]
        record = deepcopy(answers[key])
        if row['type'] == 'choice':
            record['choice'] = row['options'][value]
            record['probabilities'] = {row['options'][k]: v for k, v in record['probabilities'].items()}
        result[row['id']] = record
    return {'judgments': result, 'packet_digest': fingerprint(packet),
            'binding': deepcopy(current_binding), 'native_dispatch': False,
            'appearance_accepted': False, 'confidence_is_correctness': False}
