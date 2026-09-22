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
    def test_authority_feedback_proof_reconstructs_retained_findings(self):
        session = OperatingSession.__new__(OperatingSession)
        from .decision_budget import DecisionBudget
        session.decision_budget = DecisionBudget()
        session.items = {'old': {'writes': []}}
        session.reviews = {}
        report = {'basis': {'source': 'v1', 'authority': 'old'},
            'findings': ['retained fact'], 'checks': {'analysis': {'status': 'pass'}},
            'qualification': 'technical only'}
        session.record = {'results': {'old': {'result': {'workbench': report}}}}
        before = {'authority_revision': 'old', 'values': {
            'source': 'v1', 'authority': 'old', 'work_queue_scope': 'old-scope'}}
        after = deepcopy(before)
        after['authority_revision'] = 'new'
        after['values'].update(authority='new', work_queue_scope='new-scope')
        for state in (before, after):
            feedback = session._feedback(state)
            state['observations'] = {'workbench': feedback}
            state['values']['operating_feedback'] = fingerprint(feedback)
        proof = session.authority_feedback_rebinding(before, after, ['authority'])
        self.assertEqual(proof['after']['findings'][0]['applicability'], 'historical; inputs changed')
        altered = deepcopy(after); altered['values']['source'] = 'changed'
        with self.assertRaises(ValueError):
            session.authority_feedback_rebinding(before, altered, ['authority'])
        report['findings'] = ['different fact']
        with self.assertRaises(ValueError):
            session.authority_feedback_rebinding(before, after, ['authority'])

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

    def judge(self, state, questions, binding):
        self.batches.append(deepcopy(state))
        return {'binding': binding, 'judgments': {question['id']: {
            'choice': question['options'][0]['id']} for question in questions}}

    def session(self, execute=None, judge=None, report=None):
        return OperatingSession(self.root/'queue', service=self.service, episode=self.episode,
            owner='native owner', goal={'objective': 'Useful form'},
            observe_context=lambda: deepcopy(self.state),
            catalog=lambda state, results: deepcopy(self.items), handlers={'run': execute or self.execute},
            judge=judge or self.judge, report=report, public_projection=lambda snapshot, actions, plan: {
                'state': deepcopy(snapshot['observations']['public']),
                'descriptions': {action['id']: action['description'] for action in actions}})

    def completed_provider_failure(self):
        from .judgments import prepare_judgments
        from .test_provider_recovery import completed_call
        self.items = [self.item('first'), self.item('second')]
        captured = {}
        def interrupted(state, decisions, binding):
            self.batches.append(state)
            packet = prepare_judgments(state, decisions, binding)
            captured['ledger'], captured['call'] = completed_call(self.root, packet)
            raise PermissionError('Completed response metadata interrupted')
        session = self.session(judge=interrupted)
        self.assertEqual(session.run(max_steps=1)['status'], 'needs_reconciliation')
        from .provider_recovery import reconcile_completed_response
        reconcile_completed_response(captured['call'], captured['ledger'], lambda: session.observe())
        call = captured['call']
        selection = read_json(self.root / 'queue/controller/controller.json')['selection']
        args = {'expected_selection': fingerprint(selection), 'packet_path': call / 'owner-packet.json',
            'request_path': call / 'decision-1.request.json', 'response_path': call / 'decision-1.response.json',
            'reconciliation_path': call / 'completed-response-reconciliation.json'}
        return session, args, captured

    def test_completed_provider_recovery_resumes_once_without_new_inference(self):
        session, args, captured = self.completed_provider_failure()
        before = read_json(captured['ledger'] / 'budget.json')
        result = session.recover_completed_selection(**args)
        self.assertEqual(result['provider_calls_added'], 0)
        self.assertEqual(self.ran, [])  # Recovery never performs native work.
        self.assertEqual(session.run(max_steps=1)['status'], 'stopped')
        self.assertEqual(self.ran, ['first'])
        self.assertEqual(len(self.batches), 1)
        self.assertEqual(read_json(captured['ledger'] / 'budget.json'), before)
        self.assertEqual(read_json(self.root / 'queue/controller/controller.json')['selection_recoveries'][0]['selection'], args['expected_selection'])

    def test_completed_provider_recovery_refuses_stale_context(self):
        session, args, _ = self.completed_provider_failure()
        self.state['values']['source'] = 'later edit'
        with self.assertRaises(ValueError): session.recover_completed_selection(**args)
        self.assertIsNotNone(read_json(self.root / 'queue/controller/controller.json')['selection'])
        self.assertEqual(self.ran, [])

    def test_completed_provider_recovery_requires_exact_pending_identity_and_idle_controller(self):
        session, args, _ = self.completed_provider_failure()
        with self.assertRaises(ValueError): session.recover_completed_selection(**{**args, 'expected_selection': 'wrong'})
        write_json(self.root / 'queue/controller/controller.lock', {'active': True})
        with self.assertRaises(FileExistsError): session.recover_completed_selection(**args)
        self.assertEqual(self.ran, [])

    def test_explicit_terminal_retry_keeps_both_calls_and_releases_only_new_answer(self):
        from .judgments import prepare_judgments
        from .provider_recovery import prepare_terminal_retry, reconcile_completed_response
        from .test_provider_recovery import completed_call
        self.items = [self.item('first'), self.item('second')]
        captured = {}
        def terminal(state, decisions, binding):
            packet = prepare_judgments(state, decisions, binding)
            ledger, call = completed_call(self.root, packet)
            receipt = read_json(call/'decision-1.response.json'); receipt.pop('response'); receipt['http'] = 503
            write_json(call/'decision-1.response.json', receipt)
            budget = read_json(ledger/'budget.json'); budget['known_cost_usd'] = 0.
            budget['attempts'][0]['estimated_cost_usd'] = 0.
            write_json(ledger/'budget.json', budget)
            captured.update(ledger=ledger, call=call)
            raise RuntimeError('Terminal HTTP503')
        session = self.session(judge=terminal)
        self.assertEqual(session.run(max_steps=1)['status'], 'needs_reconciliation')
        ledger, original = captured['ledger'], captured['call']
        retry = prepare_terminal_retry(original, ledger, session.observe, reason='One explicit retry')
        retry_root = self.root/'new-attempt'; retry_root.mkdir()
        temporary_ledger, call = completed_call(retry_root, retry['retry_packet'])
        budget = read_json(ledger/'budget.json'); additional = read_json(temporary_ledger/'budget.json')
        budget['attempts'].extend(additional['attempts'])
        budget['known_cost_usd'] += additional['known_cost_usd']
        write_json(ledger/'budget.json', budget)
        reconcile_completed_response(call, ledger, session.observe)
        selection = read_json(self.root/'queue/controller/controller.json')['selection']
        result = session.recover_completed_selection(expected_selection=fingerprint(selection),
            packet_path=call/'owner-packet.json', request_path=call/'decision-1.request.json',
            response_path=call/'decision-1.response.json',
            reconciliation_path=call/'completed-response-reconciliation.json',
            terminal_retry_path=original/'terminal-retry.json')
        self.assertEqual(result['terminal_retry_lineage']['original_http'], 503)
        self.assertEqual(result['terminal_retry_lineage']['ordinal'], 1)
        self.assertEqual(self.ran, [])
        self.assertEqual(read_json(original/'decision-1.response.json')['http'], 503)
        session.run(max_steps=1)
        self.assertEqual(self.ran, ['first'])

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

    def test_fresh_public_observation_during_inference_invalidates_selection(self):
        self.items = [self.item('one'), self.item('two')]
        def judge(*args):
            result = self.judge(*args)
            self.state['public_state']['new'] = 'Scope was contradicted'
            return result
        self.assertEqual(self.session(judge=judge).run(max_steps=2)['status'], 'invalidated')
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

    def test_existing_recipe_keeps_its_own_step_journal_in_same_episode(self):
        from .test_recipes import RecipeTests
        fixture = RecipeTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        self.service, self.episode, self.root = fixture.s, fixture.episode, fixture.root
        recipe = fixture.create()
        item = self.item('diagnostic'); item['handler'] = 'service'
        item['payload'] = {'operation': 'run_recipe_step', 'arguments': {
            'recipe': recipe['recipe'], 'step': 'coverage', 'expected_revision': recipe['revision']}}
        self.items = [item]
        session = self.session(report=lambda item, result: self.report(item))
        self.assertEqual(session.run(max_steps=2)['status'], 'completed')
        records = calls(self.service.store, self.episode)
        self.assertIn('inspect_control_coverage', [row['operation'] for row in records])
        self.assertIn('execute_modeling_task', [row['operation'] for row in records])
        state = self.service.inspect_recipe(recipe['recipe'])
        self.assertEqual(fixture.rows(state)['coverage']['status'], 'reusable')

    def test_complete_capability_map_and_read_operations(self):
        value = protocol()
        self.assertGreaterEqual(len(value['capabilities']), 40)
        self.assertTrue(all(row['control']['lanes'] for row in value['capabilities']))
        count = len(calls(self.service.store))
        self.service.execute('operating_protocol', {})
        self.assertEqual(len(calls(self.service.store)), count)

    def test_projection_refusal_clears_only_known_undispatched_selection(self):
        self.items = [self.item('first'), self.item('second')]
        session = self.session()
        session.projection = lambda *args: {'state': {}, 'operations': {}}
        result = session.run(max_steps=2)
        self.assertEqual(result['status'], 'invalidated')
        self.assertFalse(result['provider_dispatched'])
        self.assertEqual(self.batches, []); self.assertEqual(self.ran, [])
        self.assertIsNone(read_json(session.directory/'controller/controller.json')['selection'])

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
