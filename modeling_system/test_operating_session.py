"""Actual service/episode/queue composition; no provider or native contact."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from . import test_decisions as fixtures
from .controller import fingerprint, read_json, write_json
from .journal import calls
from .operating_session import OperatingSession, PROFILES, protocol
from .work_queue import WorkQueue


class OperatingTests(unittest.TestCase):




    def setUp(self):
        self.base = fixtures.EpisodeTests(); self.base.setUp()
        self.addCleanup(self.base.tearDown)
        self.service, self.root = self.base.s, self.base.root
        self.episode = self.base.episode()['episode']
        self.state = {'owner': 'native owner', 'authority_revision': 'authorized1',
            'values': {'source': 's1', 'guide': 'g1', 'adapter': 'a1'},
            'active_operations': [], 'public_state': {'goal': 'Improve supported form'}}
        self.items = []
        self.ran, self.batches = [], []
        self.results, self.reports = {}, {}

    def item(self, key, profile='analysis', requires=None, lane='diagnosis', **contract):
        bindings = {role: ['guide' if role in ('guide','depth','sections') else
                           'adapter' if role == 'adapter' else 'source']
                    for role in PROFILES[profile]['inputs']}
        return {'id': key, 'revision': '1', 'lane': lane, 'handler': 'run',
            'description': 'Useful ' + key, 'completion_condition': 'Bound checks and actual evidence',
            'reads': deepcopy(self.state['values']), 'writes': [], 'requires': requires or {},
            'workbench': {'profile': profile, 'capability': 'qualified synthetic capability',
                'method': 'synthetic mechanism v1', 'bindings': bindings, **contract}}

    def report(self, item):
        link = {'kind': 'file', 'path': str(self.base.image), 'role': 'synthetic evidence'}
        return {'checks': {key: {'status': 'pass', 'evidence': [link]}
                for key in PROFILES[item['workbench']['profile']]['outputs']},
            'findings': [{'kind': 'measured', 'scope': 'affected region', 'summary': 'Actual returned support was measured.'}]}

    def execute(self, item, context):
        self.ran.append(item['id'])
        return self.results.get(item['id'], {'status': 'completed',
            'workbench': self.reports.get(item['id'], self.report(item))})

    def select(self, state, actions, plan):
        self.batches.append(deepcopy(state['observations']))
        return actions[0]['id']

    def session(self, execute=None, select=None, report=None):
        return OperatingSession(self.root/'queue', service=self.service, episode=self.episode,
            owner='native owner', goal={'objective': 'Useful form'},
            observe_context=lambda: deepcopy(self.state),
            catalog=lambda state, results: deepcopy(self.items), handlers={'run': execute or self.execute},
            select=select or self.select, report=report)






    def test_real_episode_receipts_and_fresh_results_enter_next_decision(self):
        first = self.item('diagnose')
        self.items = [first, self.item('left', requires={'diagnose': ['completed']}),
                      self.item('right', requires={'diagnose': ['completed']})]
        report = self.report(first)
        report['findings'] = [{'kind': 'scope_noop', 'scope': 'first side', 'summary': 'Proposed construction is already present; second side still differs.'}]
        self.reports['diagnose'] = report
        session = self.session(); result = session.run(max_steps=5)
        self.assertEqual(result['status'], 'completed')
        self.assertIn('already present', str(self.batches[0]['workbench']))
        self.assertEqual(self.batches[0]['workbench']['findings'][0]['applicability'], 'current inputs')
        facts = [row for row in calls(self.service.store, self.episode) if row['operation']=='execute_modeling_task']
        self.assertEqual(len(facts), 3)
        fact = self.service.store.get(facts[0]['record'])
        self.assertEqual(fact['intent']['episode'], self.episode)
        self.assertIn('context_record', fact['outcome']['result']['workbench'])
        self.assertTrue(self.service.decision_workspace(self.episode, detail='full')['operations'])
        self.session().run(max_steps=5)
        self.assertEqual(len(self.ran), 3)

    def test_missing_guide_relationship_refuses_before_effect(self):
        item = self.item('fit', profile='appearance_edit', lane='repair')
        del item['workbench']['bindings']['sections']
        self.items = [item]
        result = self.session().run(max_steps=2)
        self.assertEqual(result['status'], 'error')
        self.assertIn('sections', result['reason']); self.assertEqual(self.ran, [])

    def test_saved_candidate_review_and_later_intent_invalidation(self):
        self.items = [self.item('fit', profile='appearance_edit', lane='repair'),
            self.item('retain', profile='retention', lane='verification', requires={'fit': ['completed']},
                      visual_review='fit', consumes={'fit': ['realization', 'preservation']})]
        session = self.session()
        self.assertEqual(session.run(max_steps=4)['status'], 'needs_review')
        self.assertEqual(self.ran, ['fit'])
        basis = session.review_basis('fit')
        session.record_review('fit', expected_basis=basis,
            judgment={'disposition': 'useful', 'scope': 'second-side outline only',
                      'reason': 'Actual matched images show the local gain despite a worse maximum angle.', 'next_question': 'Check remaining corner'},
            evidence=[{'kind': 'file', 'path': str(self.base.image), 'role': 'actual matched views'}])
        self.assertEqual([row['id'] for row in session.observe()['actions']], ['retain'])
        self.state['values']['guide'] = 'g2'
        self.assertEqual(session.observe()['actions'], [])
        with self.assertRaisesRegex(ValueError, 'changed'):
            session.record_review('fit', expected_basis=basis, judgment={}, evidence=[])
        self.state['values']['guide'] = 'g1'
        self.assertEqual(session.run(max_steps=3)['status'], 'completed')
        self.assertEqual(self.ran, ['fit', 'retain'])
        view = self.service.inspect_operating_session(str(session.directory))
        self.assertEqual(view['reviews']['fit']['user_acceptance'], 'not implied')

    def test_technical_completion_does_not_invent_missing_checks(self):
        self.items = [self.item('fit'), self.item('use', requires={'fit': ['completed']}, consumes={'fit': ['analysis']})]
        self.reports['fit'] = {'findings': [], 'checks': {}}
        session = self.session()
        self.assertEqual(session.run(max_steps=4)['status'], 'needs_review')
        self.assertEqual(self.ran, ['fit'])
        self.assertEqual(session.record['results']['fit']['status'], 'completed')
        self.assertEqual(session.record['results']['fit']['result']['workbench']['qualification'], 'needs evidence')

    def test_missing_or_failed_report_preserves_effect_and_blocks_replay(self):
        self.items = [self.item('work')]
        self.reports['work'] = {'findings': [{'kind': 'invented'}]}
        session = self.session()
        self.assertEqual(session.run(max_steps=2)['status'], 'needs_reconciliation')
        entry = session.record['results']['work']
        folder = self.service.store.root/'calls'/entry['operation_handle']
        self.assertEqual(read_json(folder/'capability-result.json')['status'], 'completed')
        with self.assertRaisesRegex(ValueError, 'no known qualified return'):
            session.recover('work')
        self.session().run(max_steps=2)
        self.assertEqual(self.ran, ['work'])

    def test_numpy_results_are_retained_without_report_interruption(self):
        import numpy as np
        item = self.item('work'); self.items = [item]
        self.results['work'] = {'status': 'completed', 'count': np.int32(3),
                                'metrics': np.array([.1, .2]), 'workbench': self.report(item)}
        session = self.session()
        self.assertEqual(session.run(max_steps=2)['status'], 'completed')
        entry = session.record['results']['work']
        raw = read_json(self.service.store.root/'calls'/entry['operation_handle']/'capability-result.json')
        self.assertEqual(raw['count'], 3); self.assertEqual(raw['metrics'], [.1, .2])

    def test_handler_preflight_blocks_invalid_task_before_inference_or_operation(self):
        from .native_recipes import NativeJob
        item = self.item('native', 'inspection', native=True); item['payload'] = {'job': {}}
        item['workbench']['bindings']['adapter'] = ['adapter']
        self.items = [item]
        session = self.session(execute=NativeJob(self.service, self.root/'native'))
        before = list((self.service.store.root/'calls').glob('*/intent.json'))
        self.assertEqual(session.run(max_steps=2)['status'], 'needs_review')
        self.assertEqual(self.batches, [])
        self.assertEqual(list((self.service.store.root/'calls').glob('*/intent.json')), before)
        self.assertIn('blender', session.observe()['observations']['blocked']['native']['preflight'])
        self.assertFalse((self.root/'native').exists())

    def test_oversized_report_keeps_known_effect_and_repairs_without_replay(self):
        first = self.item('work')
        self.items = [first, self.item('use', requires={'work': ['completed']}, consumes={'work': ['analysis']})]
        report = self.report(first); report['findings'][0]['summary'] = 'Complete limits. '*200
        self.reports['work'] = report
        session = self.session()
        self.assertEqual(session.run(max_steps=4)['status'], 'needs_review')
        entry = session.record['results']['work']
        self.assertEqual(entry['status'], 'completed')
        self.assertTrue(entry['result']['workbench']['report_needs_repair'])
        self.assertEqual(self.ran, ['work'])
        folder = self.service.store.root/'calls'/entry['operation_handle']
        retained, = folder.glob('report-*.json')
        self.assertEqual(read_json(retained), report)
        original = (folder/'result.json').read_bytes()
        session.repair_report('work', self.report(first))
        self.assertEqual((folder/'result.json').read_bytes(), original)
        self.assertEqual(session.run(max_steps=4)['status'], 'completed')
        self.assertEqual(self.ran, ['work', 'use'])

    def test_recover_return_after_queue_index_crash_does_not_replay(self):
        self.items = [self.item('work')]
        session = self.session(); session.run(max_steps=2)
        entry = session.record['results']['work']; expected = deepcopy(entry['result'])
        entry['status'] = 'needs_reconciliation'; entry.pop('result')
        write_json(session.path, session.record)
        path = session.directory/'controller/controller.json'
        index = read_json(path); index['attempts'][0]['status'] = 'needs_reconciliation'
        write_json(path, index)
        recovered = session.recover('work')
        self.assertEqual(recovered['workbench'], expected['workbench'])
        self.assertEqual(self.session().run(max_steps=2)['status'], 'completed')
        self.assertEqual(self.ran, ['work'])

    def test_attach_existing_queue_preserves_historical_bytes_and_no_effect_replay(self):
        item = self.item('legacy'); item.pop('workbench'); self.items = [item]
        original = WorkQueue(self.root/'queue', owner='native owner', goal={'objective': 'Useful form'},
            observe_context=lambda: deepcopy(self.state), catalog=lambda *args: deepcopy(self.items),
            handlers={'run': lambda *args: {'status': 'completed'}}, select=None)
        original.run(max_steps=2)
        prior = original.path.read_bytes()
        session = self.session()
        self.assertEqual(session.path.read_bytes(), prior)
        self.assertEqual(session.run(max_steps=2)['status'], 'completed')
        self.assertEqual(self.ran, [])
        self.items.append(self.item('next', requires={'legacy': ['completed']}))
        self.assertEqual(session.run(max_steps=2)['status'], 'completed')
        self.assertEqual(self.ran, ['next'])

    def test_fresh_public_observation_during_selection_invalidates_selection(self):
        self.items = [self.item('one'), self.item('two')]
        def judge(*args):
            result = self.select(*args)
            self.state['public_state']['new'] = 'Scope was contradicted'
            return result
        self.assertEqual(self.session(select=judge).run(max_steps=2)['status'], 'invalidated')
        self.assertEqual(self.ran, [])

    def test_report_repair_preserves_original_failure_and_never_repeats_capability(self):
        self.items = [self.item('work')]
        self.reports['work'] = {'findings': [{'kind': 'invalid'}]}
        session = self.session(); session.run(max_steps=2)
        handle = session.record['results']['work']['operation_handle']
        result_path = self.service.store.root/'calls'/handle/'result.json'
        original = result_path.read_bytes()
        result = session.repair_report('work', self.report(self.items[0]))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result_path.read_bytes(), original)
        self.assertEqual(self.session().run(max_steps=2)['status'], 'completed')
        self.assertEqual(self.ran, ['work'])

    def test_raised_capability_settles_only_from_evidence_backed_reconciliation(self):
        effect = self.root / 'inner-effect.txt'
        def raising(item, context):
            if item['id'] == 'work' and item['revision'] == '1':
                self.ran.append(item['id'])
                effect.write_text('inner effect completed before the wrapper failed')
                raise KeyError('wrapper error after the inner effect')
            return self.execute(item, context)
        self.items = [self.item('work'), self.item('use', requires={'work': ['completed']})]
        session = self.session(execute=raising)
        self.assertEqual(session.run(max_steps=2)['status'], 'needs_reconciliation')
        entry = session.record['results']['work']; handle = entry['operation_handle']
        with self.assertRaisesRegex(ValueError, 'Uncertain queue effect'):
            session.observe()
        with self.assertRaisesRegex(ValueError, 'handle differs'):
            session.settle_reconciled('work', expected_handle='0' * 32)
        with self.assertRaisesRegex(ValueError, 'Reconcile the original operation'):
            session.settle_reconciled('work', expected_handle=handle)
        self.service.reconcile_operation(handle, observed={'effect_status': 'resolved_failed',
            'basis': 'Inner effect inspected; the wrapper raised before any qualified return'}, evidence_paths=[str(effect)])
        settled = session.settle_reconciled('work', expected_handle=handle)
        self.assertEqual(settled['status'], 'failed'); self.assertFalse(settled['effects_replayed'])
        self.assertEqual(settled['reconciliation']['effect_status'], 'resolved_failed')
        self.assertEqual(session.record['results']['work']['status'], 'failed')
        with self.assertRaisesRegex(ValueError, 'not awaiting reconciliation'):
            session.settle_reconciled('work', expected_handle=handle)
        # The failed prerequisite blocks its dependent; nothing is replayed.
        self.assertEqual(self.session(execute=raising).run(max_steps=3)['status'], 'needs_review')
        self.assertEqual(self.ran, ['work'])
        # A corrected revision is new work and runs normally.
        self.items[0]['revision'] = '2'
        self.assertEqual(self.session(execute=raising).run(max_steps=4)['status'], 'completed')
        self.assertEqual(self.ran, ['work', 'work', 'use'])

    def test_confirmed_return_is_not_settled_as_failure(self):
        self.items = [self.item('work')]
        self.reports['work'] = {'findings': [{'kind': 'invalid'}]}
        session = self.session(); session.run(max_steps=2)
        entry = session.record['results']['work']; handle = entry['operation_handle']
        source = self.service.store.root / 'calls' / handle / 'capability-result.json'
        self.service.reconcile_operation(handle, observed={'effect_status': 'confirmed_returned',
            'basis': 'Exact capability return retained'}, evidence_paths=[str(source)])
        with self.assertRaisesRegex(ValueError, 'recover\\(\\) or repair_report'):
            session.settle_reconciled('work', expected_handle=handle)

    def test_feedback_submission_never_overwrites_running_queue(self):
        self.items = [self.item('done')]
        session = self.session(); session.run(max_steps=2)
        self.items.append(self.item('next', requires={'done': ['completed']}))
        session.record['results']['next'] = {'status': 'running', 'definition': 'fixture-running'}
        write_json(session.path, session.record)
        before = session.path.read_bytes()
        reviewer = self.session()
        basis = reviewer.review_basis('done')
        reviewer.record_review('done', expected_basis=basis,
            judgment={'disposition': 'useful', 'scope': 'local', 'reason': 'Actual image gain', 'next_question': 'Remaining defect'},
            evidence=[{'kind': 'file', 'path': str(self.base.image), 'role': 'matched images'}])
        self.assertEqual(session.path.read_bytes(), before)
        self.assertEqual(self.service.inspect_operating_session(str(session.directory))['reviews']['done']['basis'], basis)

    def test_existing_service_preparation_keeps_its_domain_status(self):
        item = self.item('prepare'); item['handler'] = 'service'
        item['payload'] = {'operation': 'capture_operation_context', 'arguments': {'stage': 'review'}}
        self.items = [item]
        session = self.session(report=lambda item, result: self.report(item))
        self.assertEqual(session.run(max_steps=2)['status'], 'completed')
        result = session.record['results']['prepare']['result']
        self.assertIn('context_record', result['service_result'])
        self.assertEqual(len([c for c in calls(self.service.store, self.episode) if c['operation']=='execute_modeling_task']), 1)

    def test_complete_capability_map_and_read_operations(self):
        value = protocol()
        self.assertGreaterEqual(len(value['capabilities']), 30)
        self.assertTrue(all(row['control']['lanes'] for row in value['capabilities']))
        count = len(calls(self.service.store))
        self.service.execute('operating_protocol', {})
        self.assertEqual(len(calls(self.service.store)), count)


    def test_legacy_selection_recovery_requires_exact_identity_and_evidence(self):
        self.items = [self.item('work')]
        session = self.session(); session.run(max_steps=2)
        path = session.directory/'controller/controller.json'
        controller = read_json(path)
        controller['selection'] = {'state': {'source': 'fixture'}, 'actions': [], 'plan': {}}
        write_json(path, controller)
        key = fingerprint(controller['selection'])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            session.recover_selection(expected_selection='stale', observed={}, evidence=[])
        with self.assertRaisesRegex(ValueError, 'no-dispatch evidence'):
            session.recover_selection(expected_selection=key, observed={}, evidence=[])
        self.assertEqual(read_json(path)['selection'], controller['selection'])
        recovered = session.recover_selection(expected_selection=key,
            observed={'effect_status': 'confirmed_not_dispatched', 'basis': 'Synthetic retained preflight refusal and unchanged ledger'},
            evidence=[{'kind': 'file', 'path': str(self.base.image), 'role': 'synthetic refusal evidence'}])
        self.assertFalse(recovered['effects_replayed'])
        self.assertIsNone(read_json(path)['selection'])
        self.assertEqual(self.ran, ['work'])


class MetadataWriteTests(unittest.TestCase):
    def test_windows_sharing_collision_retries_rename_only(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'status.json'
            error = PermissionError('sharing collision'); error.winerror = 5
            real = os.replace
            attempts = []
            def replace(source, destination):
                attempts.append(1)
                if len(attempts) < 3:
                    raise error
                return real(source, destination)
            with patch('modeling_system.controller.os.replace', side_effect=replace), patch('modeling_system.controller.time.sleep'):
                write_json(path, {'status': 'completed'})
            self.assertEqual(read_json(path)['status'], 'completed')
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_nonsharing_error_is_not_retried(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            with patch('modeling_system.controller.os.replace', side_effect=PermissionError('ordinary denial')) as replace:
                with self.assertRaises(PermissionError):
                    write_json(Path(directory)/'status.json', {})
                self.assertEqual(replace.call_count, 1)
            self.assertEqual(read_json(next(Path(directory).glob('.wb-*.tmp'))), {})


if __name__ == '__main__':
    unittest.main()
