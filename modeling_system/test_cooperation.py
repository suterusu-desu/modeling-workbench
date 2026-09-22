"""Review overlap and scoped Jev fallback without native or provider effects."""
from copy import deepcopy
import threading
import unittest

from .cooperation import compose_catalogs, request_stop
from .controller import read_json
from . import test_operating_session as operating_fixtures
from . import test_work_queue as queue_fixtures
from .work_queue import DEFER


class CooperativeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = operating_fixtures.OperatingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        f.items = [f.item('candidate', 'appearance_edit'),
            f.item('retain', 'retention', requires={'candidate': ['completed']},
                   visual_review='candidate')]
        self.session = f.session()
        self.session.run(max_steps=1)

    def review(self, session=None, disposition='useful'):
        f = self.fixture
        session = session or self.session
        return session.record_review('candidate', expected_basis=session.review_basis('candidate'),
            judgment={'disposition': disposition, 'scope': 'affected region',
                      'reason': 'Actual synthetic review', 'next_question': 'Resolve remaining shape'},
            evidence=[{'kind': 'file', 'path': str(f.base.image), 'role': 'matched views'}])

    def test_useful_independent_work_runs_while_review_is_pending(self):
        f = self.fixture
        f.items.append(f.item('residuals', lane='diagnosis'))
        handoffs = []
        result = self.session.run(max_steps=4, on_handoff=handoffs.append)
        self.assertEqual(f.ran, ['candidate', 'residuals'])
        self.assertEqual(result['status'], 'needs_review')
        request = handoffs[0]['requests'][0]
        self.assertEqual(request['task'], 'candidate')
        self.assertEqual(request['basis'], self.session.review_basis('candidate'))
        self.assertEqual(request['waiting_tasks'], ['retain'])
        self.assertIn('comparison', request['evidence'])
        self.assertEqual(read_json(self.session.directory/'cooperative-run.json')[
            'operations_completed_while_review_pending'], 1)

    def test_feedback_resumes_same_controller_without_polling_observation(self):
        f = self.fixture
        ready, proceed = threading.Event(), threading.Event()
        calls, errors = [], []
        reviewer = f.session()
        original = self.session.observe_context
        self.session.observe_context = lambda: (calls.append(1), original())[1]
        def submit():
            try:
                self.assertTrue(ready.wait(3))
                count = len(calls)
                # While the mailbox is unchanged, no observation/provider spin.
                self.assertFalse(proceed.wait(.25))
                self.assertEqual(len(calls), count)
                self.review(reviewer)
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=submit)
        thread.start()
        try:
            result = self.session.run(max_steps=5, feedback_timeout=3,
                on_status=lambda s: ready.set() if s['status'] == 'waiting_feedback' else None)
        finally:
            proceed.set(); thread.join(4)
        if errors: raise errors[0]
        self.assertFalse(thread.is_alive())
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(f.ran, ['candidate', 'retain'])
        self.assertGreater(read_json(self.session.directory/'cooperative-run.json')['feedback_wait_seconds'], .1)

    def test_rejection_never_releases_retention_or_waits_for_repeat_review(self):
        self.review(disposition='rejected')
        result = self.session.run(max_steps=3, feedback_timeout=3)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(result['cooperation']['requests'][0]['kind'], 'reasoning')
        self.assertEqual(self.fixture.ran, ['candidate'])
        self.assertEqual(read_json(self.session.directory/'cooperative-run.json')['feedback_wait_seconds'], 0)

    def test_changed_source_rejects_feedback_and_never_promotes(self):
        basis = self.session.review_basis('candidate')
        self.fixture.state['values']['source'] = 'user edit'
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.session.record_review('candidate', expected_basis=basis,
                judgment={}, evidence=[])
        result = self.session.run(max_steps=3, feedback_timeout=3)
        self.assertEqual(result['status'], 'needs_review')
        self.assertFalse(result['cooperation']['requests'][0]['current_inputs'])
        self.assertEqual(self.fixture.ran, ['candidate'])

    def test_stop_while_waiting_keeps_original_receipts(self):
        def status(row):
            if row['status'] == 'waiting_feedback':
                request_stop(self.session.directory, reason='End this supervised interval')
        result = self.session.run(max_steps=3, feedback_timeout=3, on_status=status)
        self.assertEqual(result['status'], 'stopped')
        self.assertEqual(self.fixture.ran, ['candidate'])
        self.assertFalse((self.session.directory/'controller/controller.lock').exists())
        self.assertEqual(read_json(self.session.directory/'cooperative-run.json')['status'], 'closed')
        self.review()
        self.assertEqual(self.session.run(max_steps=3)['status'], 'completed')
        self.assertEqual(self.fixture.ran, ['candidate', 'retain'])

    def test_zero_wait_and_finite_timeout_do_not_poll_inference(self):
        f = self.fixture
        before = len(f.batches)
        result = self.session.run(max_steps=3, feedback_timeout=.15)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(len(f.batches), before)
        self.assertEqual(f.ran, ['candidate'])
        for invalid in (True, -1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                self.session.run(max_steps=3, feedback_timeout=invalid)

    def test_catalog_composition_keeps_existing_identity_and_dependencies(self):
        f = self.fixture
        first = f.items[:1]; other = f.items[1:]
        composed = compose_catalogs(lambda *args: first, lambda *args: other)
        result = composed({}, {})
        self.assertEqual(result, f.items)
        result[0]['description'] = 'caller change'
        self.assertNotEqual(first[0]['description'], 'caller change')
        with self.assertRaisesRegex(ValueError, 'unique'):
            compose_catalogs(lambda *args: first, lambda *args: first)({}, {})


class ScopedFallbackTests(unittest.TestCase):
    def setUp(self):
        self.f = queue_fixtures.LaneSelectorTests(); self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def choose(self, *, global_defer=False, all_defer=False):
        f = self.f; selector = f.selector()
        def judge(state, questions, binding):
            f.calls.append(questions)
            return {'binding': binding, 'judgments': {q['id']: {
                'choice': (DEFER if global_defer else 'diagnosis') if q['id'] == 'next_lane' else
                          DEFER if q['id'] == 'diagnosis' or all_defer else q['options'][0]['id']}
                for q in questions}}
        selector.judge = judge
        return selector(f.snapshot, f.actions, f.plan)

    def test_scoped_deferral_uses_other_real_jev_choice_from_same_batch(self):
        self.assertEqual(self.choose(), 'retrieve')
        self.assertEqual(len(self.f.calls), 1)
        trace = read_json(self.f.root/'last-batch.json')
        self.assertEqual(trace['cooperation']['preferred_lane'], 'diagnosis')
        self.assertEqual(trace['cooperation']['selected_lane'], 'experience')
        self.assertEqual(trace['cooperation']['deferred_lanes'][0]['lane'], 'diagnosis')
        self.assertEqual(self.f.selector()(self.f.snapshot, self.f.actions, self.f.plan), 'retrieve')
        self.assertEqual(len(self.f.calls), 1)

    def test_explicit_global_deferral_is_never_overridden(self):
        self.assertEqual(self.choose(global_defer=True)['status'], 'needs_review')

    def test_no_applicable_choices_yield_one_scoped_handoff(self):
        result = self.choose(all_defer=True)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(len(result['deferred_lanes']), 2)

    def test_fallback_must_pass_its_own_method_and_claim_checks(self):
        from .decision_evidence import grounded_claim
        f = self.f
        f.actions[1]['decision'] = {'method_checks': {'method': 'Is current support sufficient?'},
            'claims': {'support': grounded_claim('Support is complete.',
                source_text='Support is incomplete.', public_excerpt='Support is incomplete.')}}
        for condition in ('method', 'claim'):
            selector = f.selector()
            def judge(state, questions, binding):
                answers = {}
                for q in questions:
                    key = q['id']
                    value = q['options'][0]['id']
                    if key == 'diagnosis': value = DEFER
                    if key == 'next_lane': value = 'diagnosis'
                    if key.endswith('_method'): value = 'missing' if condition == 'method' else 'ready'
                    if '_claim_' in key: value = 'contradicted'
                    answers[key] = {'choice': value}
                return {'binding': binding, 'judgments': answers}
            selector.judge = judge
            f.actions[1]['reads']['support'] = condition
            result = selector(f.snapshot, f.actions, f.plan)
            self.assertEqual(result['status'], 'needs_review')
            self.assertEqual(len(result['deferred_lanes']), 2)


if __name__ == '__main__': unittest.main()
