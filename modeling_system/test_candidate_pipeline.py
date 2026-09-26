from copy import deepcopy
import unittest
from . import test_operating_session as fixtures
from .candidate_pipeline import CandidatePipeline, capability_task
from .controller import read_json


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.OperatingTests(); self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.pipeline = CandidatePipeline(self.base.root/'pipeline', revision='test-v1',
            candidates=lambda state, previous: [self.base.item(k, 'appearance_edit', lane='repair') for k in ('first', 'second')],
            verify=lambda state, previous: [self.base.item('verify', 'verification', lane='verification')],
            retain=lambda state, previous: [self.base.item('retain', 'retention', lane='repair')])

    def session(self):
        session = self.base.session(); session.user_catalog = self.pipeline
        return session

    def review(self, session, key):
        session.record_review(key, expected_basis=session.review_basis(key),
            judgment={'disposition': 'useful', 'scope': 'test region', 'reason': 'Actual image inspected', 'next_question': 'Check motion'},
            evidence=[{'kind': 'file', 'path': str(self.base.base.image), 'role': 'actual synthetic image'}])

    def test_candidate_review_verify_review_retain_resume_without_copying_catalog(self):
        session = self.session()
        self.assertEqual(session.run(max_steps=8)['status'], 'needs_review')
        self.assertEqual(self.base.ran, ['first'])
        original = read_json(self.pipeline.path)['stages']['candidate']
        self.review(session, 'first')
        session = self.session()
        self.assertEqual(session.run(max_steps=8)['status'], 'needs_review')
        self.assertEqual(self.base.ran, ['first', 'verify'])
        self.review(session, 'verify')
        self.assertEqual(session.run(max_steps=8)['status'], 'completed')
        self.assertEqual(self.base.ran, ['first', 'verify', 'retain'])
        self.assertEqual(len(self.base.batches), 1)
        self.assertEqual(read_json(self.pipeline.path)['stages']['candidate'], original)
        self.assertEqual(self.session().run(max_steps=8)['status'], 'completed')
        self.assertEqual(len(self.base.ran), 3)
        metrics = session.metrics()
        self.assertEqual(metrics['retained_tasks'], ['retain'])
        self.assertEqual(metrics['visual_review_submissions'], 2)
        self.assertIsNone(metrics['avoided_owner_turns'])

    def test_failed_candidate_never_runs_other_variant_or_verification(self):
        self.base.results['first'] = {'status': 'failed', 'workbench': {'checks': {}, 'findings': []}}
        self.session().run(max_steps=8)
        self.session().run(max_steps=8)
        self.assertEqual(self.base.ran, ['first'])

    def test_adopt_existing_candidate_preserves_exact_definition_and_does_not_replay(self):
        self.base.items = [self.base.item('existing', 'appearance_edit', lane='repair')]
        session = self.base.session(); session.run(max_steps=1)
        original = deepcopy(session.record['results']['existing'])
        task = deepcopy(original['task'])
        self.assertNotIn('required', task)
        self.pipeline.adopt_completed_candidate(task, original)
        session.user_catalog = self.pipeline
        session.run(max_steps=8)
        self.review(session, 'existing')
        session.run(max_steps=8)
        self.assertEqual(self.base.ran, ['existing', 'verify'])
        self.assertEqual(session.record['results']['existing'], original)
        self.pipeline.adopt_completed_candidate(task, original)  # Idempotent after follow-ups.
        changed = deepcopy(session.record['results']); changed['existing']['result']['extra'] = 'different'
        with self.assertRaises(ValueError): self.pipeline(self.base.state, changed)

    def test_changed_guide_blocks_followup_and_invalidates_visual_review(self):
        session = self.session(); session.run(max_steps=8); self.review(session, 'first')
        self.base.state['values']['guide'] = 'later'
        self.assertEqual(session.run(max_steps=8)['status'], 'needs_review')
        self.assertEqual(self.base.ran, ['first'])

    def test_real_interventions_have_evidence_and_explicit_duration(self):
        session = self.session()
        session.record_intervention(kind='correspondence_preparation', reason='Resolved ambiguous source branch', seconds=12.5,
            evidence=[{'kind': 'file', 'path': str(self.base.base.image), 'role': 'retained evidence'}])
        metrics = session.metrics()
        self.assertEqual(metrics['owner_interventions'], {'correspondence_preparation': 1})
        self.assertEqual(metrics['reported_owner_seconds'], 12.5)

    def test_task_factory_refuses_missing_evidence_or_uncovered_write(self):
        args = dict(key='inspect', revision='1', lane='diagnosis', handler='inspect',
            description='Distinguish attachment causes', completion='Saved section evidence',
            profile='inspection', capability='registered section inspection', method='measured sections',
            bindings={'subject': ['source']}, reads=['source'])
        item = capability_task(self.base.state, **args)
        self.assertEqual(item['reads'], {'source': 's1'})
        with self.assertRaises(ValueError): capability_task(self.base.state, **args, writes=['missing'])
        args['bindings'] = {}
        with self.assertRaises(ValueError): capability_task(self.base.state, **args)


if __name__ == '__main__': unittest.main()
