"""End-to-end typed choices, evidence stages and outcomes without provider/native IO."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from . import test_operating_session as fixtures
from .capability_calls import DEFAULT, UNKNOWN, compile_call
from .controller import fingerprint, read_json, write_json
from .decision_budget import DecisionBudget
from .decision_evidence import grounded_claim, observation_contract
from .decision_outcomes import composite_scores
from .judgments import prepare_judgments, resolve_judgments
from .operating_session import inspect_session
from .work_queue import LaneSelector


def answer_batch(state, decisions, binding, choose=None):
    packet = prepare_judgments(state, decisions, binding)
    answers = {}
    for key, mapping in packet['dispatch_mapping'].items():
        d = next(d for d in decisions if d['id'] == mapping['id'])
        value = choose(d, state) if choose else None
        if d['type'] == 'choice':
            value = value or d['options'][0]['id']
            selected = next(k for k, v in mapping['options'].items() if v == value)
            answers[key] = {'type': 'choice', 'choice': selected, 'confidence': 1.,
                'probabilities': {k: float(k == selected) for k in mapping['options']}}
        elif d['type'] == 'score':
            value = len(d['criteria']) - 1 if value is None else value
            answers[key] = {'type': 'score', 'score': float(value), 'confidence': 1.,
                'probabilities': {str(i): float(i == value) for i in range(len(d['criteria']))},
                'legend': {str(i): v for i, v in enumerate(d['criteria'])}}
        else:
            answers[key] = {'type': 'noul', 'noul': .8 if value is None else value}
    return resolve_judgments(packet, answers, binding)


class TypedSessionTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.OperatingTests(); self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        task = self.base.item('fit')
        task['payload'] = {'parameters': {'fixed': 'preserved'}}
        task['decision'] = {'arguments': {
            'region': {'question': 'Which supported region contains the dominant observed defect?',
                'path': ['parameters', 'region'], 'options': [
                    {'id': 'left', 'description': 'The first supported region', 'value': 'private-left', 'reads': {'source': 's1'}},
                    {'id': 'right', 'description': 'The other supported region', 'value': 'private-right', 'reads': {'source': 's1'}}]},
            'method': {'question': 'Which qualified mechanism addresses the evidence?', 'path': ['parameters', 'method'],
                'options': [{'id': 'section', 'description': 'Fit corresponding guide sections', 'value': 'section_fit'},
                            {'id': 'surface', 'description': 'Use connected surface constraints', 'value': 'surface_fit'}]},
            'display': {'question': 'Is an override to the existing display justified?', 'path': ['display'],
                'default': False, 'options': [{'id': 'on', 'description': 'Show the nominated comparison', 'value': True}]}
        }}
        self.base.items = [task]
        self.calls, self.executed = [], []
        self.choose = lambda d, s: 'right' if d['id'].endswith('arg_0') else 'surface' if d['id'].endswith('arg_1') else DEFAULT if d['id'].endswith('arg_2') else None

    def session(self):
        def judge(state, decisions, binding):
            self.calls.append((deepcopy(state), deepcopy(decisions), deepcopy(binding)))
            return answer_batch(state, decisions, binding, self.choose)
        def execute(item, context):
            self.executed.append(deepcopy(item))
            return self.base.execute(item, context)
        return self.base.session(execute=execute, judge=judge)

    def test_composed_call_executes_once_preserves_defaults_and_links_review(self):
        session = self.session()
        session.run(max_steps=2)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.executed[0]['payload'], {'parameters': {
            'fixed': 'preserved', 'region': 'private-right', 'method': 'surface_fit'}, 'display': False})
        wire = prepare_judgments(*self.calls[0])
        self.assertNotIn('private-right', json.dumps(wire['questions']))
        self.assertNotIn('"source": "s1"', json.dumps(wire['state']))
        session.record_review('fit', expected_basis=session.review_basis('fit'), judgment={
            'disposition': 'rejected', 'scope': 'local region', 'reason': 'Technical fit did not improve the shape.',
            'next_question': 'Check the correspondence role.'}, evidence=[{'kind': 'file',
                'path': str(self.base.base.image), 'role': 'actual synthetic review'}])
        rows = inspect_session(self.base.service, session.directory)['decision_outcomes']
        self.assertEqual(rows['with_decision'], 1)
        self.assertEqual(rows['reviewed'], 1)
        self.assertEqual(rows['cases'][0]['failure_class'], 'appearance_rejected')
        self.assertEqual(rows['cases'][0]['decision']['call']['arguments']['region'], 'private-right')
        self.session().run(max_steps=2)
        self.assertEqual(len(self.executed), 1)

    def test_unknown_or_incompatible_selected_argument_stops_before_effects(self):
        self.choose = lambda d, s: UNKNOWN if d['id'].endswith('arg_0') else None
        self.assertEqual(self.session().run(max_steps=1)['status'], 'needs_review')
        self.assertEqual(self.executed, [])
        self.base.items[0]['decision']['incompatible'] = [{'region': 'right', 'method': 'surface'}]
        self.choose = lambda d, s: 'right' if d['id'].endswith('arg_0') else 'surface' if d['id'].endswith('arg_1') else None
        self.assertEqual(self.session().run(max_steps=1)['status'], 'needs_review')
        self.assertEqual(self.executed, [])

    def test_unused_argument_uncertainty_does_not_block_the_selected_call(self):
        other = deepcopy(self.base.items[0]); other['id'] = 'other'
        self.base.items.append(other)
        old = self.choose
        self.choose = lambda d, s: UNKNOWN if d['id'].startswith('call_1_arg_') else old(d, s)
        self.session().run(max_steps=1)
        self.assertEqual([r['id'] for r in self.executed], ['fit'])

    def test_changes_during_choice_invalidate_before_execution(self):
        old = self.choose
        def changed(d, state):
            value = old(d, state)
            self.base.state['values']['guide'] = 'later-guide'
            return value
        self.choose = changed
        self.assertEqual(self.session().run(max_steps=1)['status'], 'invalidated')
        self.assertEqual(self.executed, [])

    def test_set_members_and_optional_defaults_are_typed(self):
        self.base.items[0]['decision']['arguments']['views'] = {
            'type': 'set', 'question': 'Which matched views resolve the current distinction?', 'path': ['views'],
            'options': [{'id': 'front', 'value': 'front-camera', 'description': 'Matched front view'},
                        {'id': 'profile', 'value': 'side-camera', 'description': 'Matched profile view'}]}
        old = self.choose
        self.choose = lambda d, s: (.9 if d['id'].endswith('arg_3_1') else .1) if d['type'] == 'noul' else old(d, s)
        self.session().run(max_steps=1)
        self.assertEqual(self.executed[0]['payload']['views'], ['side-camera'])

    def test_claims_are_grounded_and_only_selected_claims_control_execution(self):
        self.base.items[0]['decision']['claims'] = {'benefit': grounded_claim(
            'The complete motion is accepted.', source_text='A local early pose improved. Other motion is unreviewed.',
            public_excerpt='A local early pose improved. Other motion is unreviewed.')}
        old = self.choose
        self.choose = lambda d, s: 'contradicted' if '_claim_' in d['id'] else old(d, s)
        result = self.session().run(max_steps=1)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(self.executed, [])
        self.assertEqual(read_json(self.base.root/'queue/advice/last-batch.json')['claim_checks']['benefit']['choice'], 'contradicted')

    def test_tampered_call_cannot_change_qualified_value(self):
        session = self.session(); session.run(max_steps=1)
        call = read_json(session.selector.directory/'last-batch.json')['call']
        call['arguments']['region'] = 'unqualified'
        with self.assertRaises(ValueError): compile_call(self.base.items[0], call)

    def test_observation_purpose_reaches_the_operation_choice(self):
        self.base.items[0]['decision']['observation'] = observation_contract(
            question='Is the mismatch caused by correspondence or carrier composition?',
            hypotheses=['wrong material pairing', 'duplicate displacement'],
            changes_next_action={'wrong pairing': 'repair correspondence', 'duplicate displacement': 'fix composition'},
            existing_evidence='Recorded paired arrays are sufficient; no new render needed')
        self.session().run(max_steps=1)
        self.assertIn('no new render', json.dumps(self.calls[0][1]))


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = {'owner': 'owner', 'authority_revision': 'a1'}
        self.actions = [{'id': 'edit', 'revision': 'v1', 'lane': 'repair', 'reads': {'mesh': 'm1'}, 'writes': []}]
        self.candidates = [{'index': i, 'role': 'matches', 'content': {'lesson': 'counterexample' if i == 9 else 'background'}} for i in range(10)]
        self.batches = []

    def projection(self, *args):
        return {'state': {'goal': 'Repair the current mechanism', 'retained_experience': {'passages': self.candidates[:4]}},
                'descriptions': {'edit': 'A qualified edit'}, 'experience_candidates': self.candidates}

    def judge(self, state, decisions, binding):
        self.batches.append(deepcopy(state))
        return answer_batch(state, decisions, binding, lambda d, s:
            'contradicts' if d['id'] == 'relation_9' else 'unrelated' if d['id'].startswith('relation_') else
            2 if d['id'] == 'relevance_9' else 0 if d['id'].startswith('relevance_') else None)

    def test_omitted_counterexample_reaches_next_decision_and_restart_reuses_both(self):
        selector = LaneSelector(self.root, judge=self.judge, project=self.projection)
        self.assertEqual(selector(self.snapshot, self.actions, {}), 'edit')
        self.assertEqual(len(self.batches), 2)
        selected = self.batches[1]['retained_experience']['passages']
        self.assertEqual([r['index'] for r in selected], [9])
        self.assertEqual(selected[0]['relationship'], 'contradicts')
        LaneSelector(self.root, judge=self.judge, project=self.projection)(self.snapshot, self.actions, {})
        self.assertEqual(len(self.batches), 2)

    def test_no_matching_source_is_an_empty_context_not_a_forced_winner(self):
        def judge(state, decisions, binding):
            return answer_batch(state, decisions, binding, lambda d, s: 'unrelated' if d['id'].startswith('relation_') else None)
        selector = LaneSelector(self.root, judge=judge, project=self.projection)
        selector(self.snapshot, self.actions, {})
        self.assertEqual(read_json(self.root/'last-batch.json')['retrieval']['passages'], [])

    def test_interrupted_retrieval_answer_is_recovered_without_action_inference(self):
        selector = LaneSelector(self.root, judge=self.judge, project=self.projection)
        batch = selector(self.snapshot, self.actions, {}, prepare_only=True)
        self.assertEqual(batch['phase'], 'experience')
        result = self.judge(batch['packet']['state'], batch['decisions'], batch['binding'])
        selector._complete_retrieval(batch, result)
        before = len(self.batches)
        selector(self.snapshot, self.actions, {})
        self.assertEqual(len(self.batches) - before, 1)


class BudgetAndPolicyTests(unittest.TestCase):
    def test_more_than_twelve_questions_and_larger_state_use_one_consistent_budget(self):
        binding = {'owner': 'owner', 'authority_revision': 'a1', 'dependencies': {'reads': {'source': 'v1'}, 'writes': []},
                   'decision_budget': DecisionBudget().record()}
        decisions = [{'id': str(i), 'type': 'noul', 'instructions': 'Is this named source relevant?'} for i in range(20)]
        packet = prepare_judgments({'reviewed_source': 'a' * 14000}, decisions, binding)
        self.assertEqual(len(packet['questions']), 20)
        binding['decision_budget']['max_questions'] = 12
        with self.assertRaises(ValueError): prepare_judgments({}, decisions, binding)
        with self.assertRaises(ValueError): DecisionBudget(max_request_bytes=64001)

    def test_exact_span_required_and_material_violations_do_not_average_away(self):
        with self.assertRaises(ValueError): grounded_claim('Something improved', source_text='Unknown.', public_excerpt='Accepted.')
        def score(levels, n):
            return {'type': 'score', 'score': n, 'confidence': 1., 'legend': {str(i): v for i, v in enumerate(levels)},
                    'probabilities': {str(i): float(i == n) for i in range(len(levels))}}
        raw = {'a': {'benefit': score(['none', 'local', 'broad'], 2)},
               'b': {'benefit': score(['none', 'broad'], 1)}, 'unknown': {}}
        result = composite_scores(raw, {'benefit': 1}, violations={'a': True})
        self.assertEqual(result[0]['candidate'], 'b')
        self.assertEqual(result[0]['score'], 1)
        self.assertTrue(result[-1]['material_violation'])
        self.assertIsNone(next(r for r in result if r['candidate'] == 'unknown')['score'])


if __name__ == '__main__': unittest.main()
