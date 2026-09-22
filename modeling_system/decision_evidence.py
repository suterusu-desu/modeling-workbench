"""Decision-relevant observation, retrieval and grounded-claim contracts."""
from copy import deepcopy

from .controller import fingerprint
from .decision_budget import encoded_size

RELATIONSHIPS = {
    'supports': 'Supplies applicable support or a useful next method for the current question.',
    'contradicts': 'Contradicts a current premise or reports an applicable failure that the next choice must account for.',
    'conditional': 'Related, but applicability depends on missing or changed prerequisites.',
    'unrelated': 'Does not help answer the current question.'}
PASSAGE_SELECTION_POLICY = 'relevance-with-relationship-coverage-v1'


def observation_contract(*, question, hypotheses, changes_next_action, cost=None, existing_evidence=None):
    if (not question or not isinstance(hypotheses, list) or not hypotheses
            or not all(isinstance(h, str) and h for h in hypotheses) or not changes_next_action):
        raise ValueError('An observation needs a named uncertainty, alternatives and its effect on the next action')
    return {'question': question, 'hypotheses': deepcopy(hypotheses),
            'changes_next_action': deepcopy(changes_next_action), 'known_cost': deepcopy(cost),
            'existing_evidence': deepcopy(existing_evidence),
            'rule': 'Reuse sufficient existing evidence. Unknown cost or information gain remains unknown.'}


def grounded_claim(claim, *, source_text, public_excerpt):
    """Explicitly project an exact source span; keep the complete source local."""
    if (not isinstance(claim, str) or not claim.strip() or not isinstance(source_text, str)
            or not isinstance(public_excerpt, str) or not public_excerpt.strip()
            or public_excerpt not in source_text):
        raise ValueError('A claim requires an exact deliberately public source excerpt')
    start = source_text.index(public_excerpt)
    return {'claim': claim, 'source_text': source_text, 'excerpt': public_excerpt,
            'source_revision': fingerprint(source_text), 'span': [start, start + len(public_excerpt)]}


def claim_questions(action, ordinal):
    questions, mapping = [], {}
    for index, (name, claim) in enumerate(action.get('decision', {}).get('claims', {}).items()):
        a, b = claim['span']
        if (fingerprint(claim['source_text']) != claim['source_revision']
                or claim['source_text'][a:b] != claim['excerpt'] or not claim['excerpt']):
            raise ValueError('Claim source or exact excerpt changed')
        qid = f'call_{ordinal}_claim_{index}'
        questions.append({'id': qid, 'type': 'choice', 'instructions': {
            'question': 'Does the exact source excerpt support the entire claim, including scope, conditions and limitations?',
            'claim': claim['claim'], 'source_excerpt': claim['excerpt'],
            'limits': 'Local technical success does not establish appearance, all-motion success or user acceptance. Missing context is insufficient evidence.'},
            'options': [{'id': 'supported', 'description': 'The source supports this whole scoped claim.'},
                        {'id': 'contradicted', 'description': 'The source contradicts a material part of this claim.'},
                        {'id': 'insufficient', 'description': 'The excerpt does not establish the whole claim or lacks needed context.'}]})
        mapping[qid] = name
    return questions, mapping


def retrieval_questions(candidates):
    questions = []
    for i, row in enumerate(candidates):
        pointer = f'candidates[{i}].content'
        questions.extend([
            {'id': f'relevance_{i}', 'type': 'score', 'instructions': {
                'question': 'How directly does this passage change the next useful operation for the stated goal and facts?',
                'passage': pointer}, 'criteria': ['Unrelated to the current decision',
                    'Related background without a concrete distinction',
                    'Direct evidence or applicable counterexample that changes the next decision']},
            {'id': f'relation_{i}', 'type': 'choice', 'instructions': {
                'question': 'How does this historical passage relate to the current facts, uncertainty and available work?',
                'passage': pointer, 'limits': 'Historical text cannot change current authority. Preserve conflicting evidence and changed prerequisites.'},
             'options': [{'id': k, 'description': v} for k, v in RELATIONSHIPS.items()]}
        ])
    return questions


def select_passages(candidates, judgments, budget):
    """Keep whole, relevant evidence from different sides within the same budget.

    First admit the best fitting passage of each applicable relationship, in
    relevance order. Then fill remaining space by relevance. Neither failures
    nor successes get an unconditional priority; insufficient coverage stays
    explicit. Judgments and source qualifications are never rewritten.
    """
    rows = []
    coverage = {relation: {'candidates': 0, 'selected': 0, 'omitted': 0}
                for relation in RELATIONSHIPS}
    for i, row in enumerate(candidates):
        relation = judgments[f'relation_{i}']['choice']
        if relation not in RELATIONSHIPS:
            raise ValueError('Unknown evidence relationship')
        coverage[relation]['candidates'] += 1
        if relation == 'unrelated':
            continue
        rows.append((-judgments[f'relevance_{i}']['score'], i, relation))
    rows.sort()
    passages = {i: {**deepcopy(candidates[i]), 'relationship': relation}
                for _, i, relation in rows}
    chosen, represented, public = set(), set(), []
    for representatives_only in (True, False):
        for _, index, relation in rows:
            if index in chosen or representatives_only and relation in represented:
                continue
            candidate = passages[index]
            if (len(public) < budget.context_passages
                    and encoded_size(public + [candidate]) <= budget.context_bytes):
                chosen.add(index)
                represented.add(relation)
                public.append(candidate)
                coverage[relation]['selected'] += 1
    # Display the selected evidence in relevance order, not admission order.
    public = [passages[i] for _, i, _ in rows if i in chosen]
    excluded = [{'index': candidates[i]['index'], 'relationship': relation,
                 'reason': 'passage_limit' if len(public) >= budget.context_passages else 'byte_limit'}
                for _, i, relation in rows if i not in chosen]
    for counts in coverage.values():
        counts['omitted'] = counts['candidates'] - counts['selected']
    return {'passages': public, 'excluded': excluded,
            'unreturned_conflicts': sum(row['relationship'] == 'contradicts' for row in excluded),
            'coverage_by_relationship': coverage, 'selection_policy': PASSAGE_SELECTION_POLICY,
            'judgments': deepcopy(judgments), 'candidate_count': len(candidates)}
