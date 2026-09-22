"""Conditional method questions in the existing Jev batch; no extra executor."""
from copy import deepcopy


METHOD = {
    'ready': 'Current construction and source-backed lessons support this method under its stated conditions.',
    'missing': 'A named relevant prerequisite lacks the observation or constraint needed by this method.',
    'contradicted': 'Current construction facts contradict a required method condition.',
    'repeated': 'It repeats a demonstrated failed mechanism without addressing the recorded cause.',
    'unknown': 'The supplied evidence cannot determine applicability; do not invent supporting facts.',
}
COVERAGE = {
    'ready': 'The proposed operation covers the relevant established outcomes, regions, relationships and affected motion.',
    'missing': 'A relevant outcome or transition/pose is omitted from the proposed preservation coverage.',
    'contradicted': 'Claimed preservation conflicts with supplied actual dependency or evaluated-output facts.',
    'unknown': 'Evidence cannot establish the relevant coverage; raw coefficients and endpoints alone are insufficient.',
}


def method_questions(action, ordinal, available, descriptions):
    spec = action.get('decision', {}).get('method_checks')
    if not spec: return [], {}
    if not isinstance(spec, dict) or set(spec)-{'method', 'coverage', 'remedies', 'remedy_methods'}:
        raise ValueError('Method checks have named method/coverage questions and qualified remedies')
    questions, mapping = [], {}
    for name, options in (('method', METHOD), ('coverage', COVERAGE)):
        if name not in spec: continue
        if not isinstance(spec[name], str) or not spec[name].strip(): raise ValueError('Meaningful method question required')
        qid = f'method_{ordinal}_{name}'
        questions.append({'id': qid, 'type': 'choice', 'instructions': {
            'question': spec[name], 'operation': action['public_description'],
            'evidence': 'Use required_preservation, current construction and source-backed retained_experience in state.',
            'limits': 'Known fixed checks are enforced by code. Historical successes are conditional. Unknown is not pass; confidence cannot waive requirements. Judge only this operation, not unrelated harmless work.'},
            'options': [{'id': key, 'description': value} for key,value in options.items()]})
        mapping[qid] = name
    offered = {a['id']: a for a in available}
    remedies = spec.get('remedies', [])
    methods = spec.get('remedy_methods', [])
    if (not isinstance(remedies, list) or not isinstance(methods, list)
            or any(not isinstance(k,str) or not k for k in remedies+methods)
            or len(set(methods)) != len(methods) or len(set(remedies)) != len(remedies)
            or any(k not in offered or k == action['id'] for k in remedies)):
        raise ValueError('Remedies must be distinct qualified operations or registered method names')
    # Resolve method-level links only against the final eligible action menu.
    # A completed, stale or blocked diagnostic is never offered as executable;
    # its absence does not invalidate an otherwise useful decision batch.
    remedies = list(dict.fromkeys(remedies + [row['id'] for row in available
        if row.get('method_choice') in methods and row['id'] != action['id']]))
    if remedies:
        qid = f'method_{ordinal}_remedy'
        questions.append({'id': qid, 'type': 'choice', 'instructions': {
            'question': 'Only if this proposed method is missing, contradicted, repeated without repair or unknown: which offered operation most directly resolves its relevant prerequisite?',
            'operation': action['public_description'],
            'limits': 'Choose the specific evidence, preparation or scoped recovery needed. Do not broaden scope or waive the required outcome.'},
            'options': [{'id': k, 'description': descriptions[k]} for k in remedies]
                       + [{'id': '__defer__', 'description': 'No offered remedy resolves this prerequisite.'}]})
        mapping[qid] = 'remedy'
    return questions, mapping


def allowed_remedy(action, candidate):
    spec = action.get('decision', {}).get('method_checks', {})
    return (candidate['id'] in spec.get('remedies', [])
            or candidate.get('method_choice') in spec.get('remedy_methods', []))


def resolve_method(mapping, answers):
    values = {name: deepcopy(answers[key]) for key,name in mapping.items()}
    for name, row in values.items():
        if name != 'remedy' and row.get('choice') not in (METHOD if name == 'method' else COVERAGE):
            raise ValueError('Method judgment outside its typed menu')
    unmet = {name: row for name,row in values.items() if name != 'remedy' and row['choice'] != 'ready'}
    return {'status': 'unmet' if unmet else 'ready', 'checks': values, 'unmet': unmet,
            'remedy': values.get('remedy', {}).get('choice')}
