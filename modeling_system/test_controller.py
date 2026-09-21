from copy import deepcopy
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from .controller import PersistentController, MailboxPlanner, read_json, write_json


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = dict(owner='owner', authority_revision='scope1', stage='repair',
                          values={'mesh': 'm1', 'guide': 'g1'}, active_operations=[], actions=[])
        self.plan = dict(objective='Repair within qualified guide', authority_revision='scope1',
                         stage='repair', reads={'guide': 'g1'})
        self.executed = []

    def action(self, name='measure', **fields):
        return dict(id=name, revision='1', reads={'mesh': 'm1', 'guide': 'g1'}, writes=[], **fields)

    def controller(self, **kwargs):
        args = dict(owner='owner', observe=lambda: deepcopy(self.state),
                    execute=lambda a, c: self.executed.append(a['id']) or {'status': 'completed'},
                    initial_plan=self.plan)
        args.update(kwargs)
        result = PersistentController(self.root, **args)
        self.addCleanup(result.close)
        return result

    def test_native_continues_while_planner_waits_and_runs_on_owner(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def planner(request):
            entered.set()
            release.wait(2)
            return self.plan
        native_threads = []
        self.state['actions'] = [self.action(required=True)]
        controller = self.controller(planner=planner, execute=lambda a, c:
                                     native_threads.append(threading.get_ident()) or {'status': 'completed'})
        result = controller.tick()
        self.assertTrue(entered.wait(1))
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(native_threads, [threading.get_ident()])
        self.assertTrue(result['planner_pending'])

    def test_changed_guide_discards_async_plan(self):
        release = threading.Event()
        self.addCleanup(release.set)
        def planner(_):
            release.wait(2)
            return self.plan
        self.state['actions'] = [self.action(required=True)]
        controller = self.controller(planner=planner)
        controller.tick()
        release.set()
        self.assertTrue(controller.pending['done'].wait(1))
        self.state['values']['guide'] = 'g2'
        controller.tick()
        self.assertIsNone(controller.plan)
        self.assertIn('plan_discarded', (self.root / 'events.jsonl').read_text())

    def test_changed_state_during_selection_prevents_dispatch(self):
        self.state['actions'] = [self.action('a'), self.action('b')]
        def select(*_):
            self.state['values']['mesh'] = 'm2'
            return 'a'
        controller = self.controller(select=select)
        self.assertEqual(controller.tick()['status'], 'invalidated')
        self.assertEqual(self.executed, [])

    def test_selection_executes_directly_in_same_owner_tick(self):
        self.state['actions'] = [self.action('a'), self.action('b')]
        calls = []
        def select(*_):
            calls.append('select')
            return 'b'
        def execute(action, context):
            calls.append('execute:' + action['id'])
            return {'status': 'completed'}
        controller = self.controller(select=select, execute=execute)
        self.assertEqual(controller.tick()['status'], 'completed')
        self.assertEqual(calls, ['select', 'execute:b'])

    def test_scoped_guard_avoids_second_observation_and_reports_time(self):
        calls = []
        self.state['actions'] = [self.action()]
        def observe():
            calls.append('observe')
            return deepcopy(self.state)
        def revalidate(action, expected):
            calls.append('guard')
            return {**deepcopy(self.state), 'action_fingerprint': expected['action_fingerprint']}
        controller = self.controller(observe=observe, revalidate=revalidate)
        result = controller.tick()
        self.assertEqual(calls, ['observe', 'guard'])
        self.assertEqual(self.executed, ['measure'])
        self.assertEqual(set(result['timings_ms']), {'observe', 'revalidate', 'execute', 'cycle'})

    def test_scoped_guard_refuses_missing_dependency_and_changed_action(self):
        self.state['actions'] = [self.action()]
        controller = self.controller(revalidate=lambda *_: dict(self.state, values={'mesh': 'm1'}))
        self.assertEqual(controller.tick()['status'], 'error')
        self.assertEqual(self.executed, [])
        controller.revalidate = lambda *_: dict(self.state, action_fingerprint='replaced')
        self.assertEqual(controller.tick()['status'], 'invalidated')
        self.assertEqual(self.executed, [])

    def test_scoped_guard_refuses_stale_guide(self):
        self.state['actions'] = [self.action()]
        controller = self.controller(revalidate=lambda a, e: dict(self.state,
            values={'mesh': 'm1', 'guide': 'g2'}, action_fingerprint=e['action_fingerprint']))
        self.assertEqual(controller.tick()['status'], 'invalidated')
        self.assertEqual(self.executed, [])

    def test_unknown_native_effect_blocks_restart(self):
        self.state['actions'] = [self.action()]
        def execute(*_):
            raise TimeoutError('Unknown native outcome')
        controller = self.controller(execute=execute)
        self.assertEqual(controller.tick()['status'], 'needs_reconciliation')
        controller.close()
        resumed = self.controller()
        self.assertEqual(resumed.tick()['status'], 'needs_reconciliation')
        self.assertEqual(self.executed, [])

    def test_completed_and_failed_actions_are_suppressed_until_revised(self):
        self.state['actions'] = [self.action()]
        controller = self.controller(execute=lambda *_: {'status': 'no_progress'})
        self.assertEqual(controller.tick()['status'], 'no_progress')
        self.assertEqual(controller.tick()['status'], 'idle')
        self.state['actions'][0]['revision'] = '2'
        self.assertEqual(controller.tick()['status'], 'no_progress')

    def test_required_and_unique_actions_bypass_inference(self):
        self.state['actions'] = [self.action('a', required=True), self.action('b')]
        controller = self.controller(select=lambda *_: self.fail('Unnecessary inference'))
        controller.tick()
        controller.tick()
        self.assertEqual(self.executed, ['a', 'b'])
        self.assertEqual(controller.tick()['status'], 'idle')

    def test_busy_lane_and_second_controller_cannot_execute(self):
        self.state['actions'] = [self.action()]
        self.state['active_operations'] = [{'id': 'existing', 'writes': ['mesh']}]
        controller = self.controller()
        self.assertEqual(controller.tick()['status'], 'waiting_owner')
        with self.assertRaises(FileExistsError):
            self.controller()
        self.assertEqual(self.executed, [])

    def test_invalid_selection_never_retries_provider(self):
        self.state['actions'] = [self.action('a'), self.action('b')]
        calls = []
        controller = self.controller(select=lambda *_: calls.append(1) or 'not-an-action')
        self.assertEqual(controller.tick()['status'], 'needs_reconciliation')
        self.assertEqual(controller.tick()['status'], 'needs_reconciliation')
        self.assertEqual(len(calls), 1)

    def test_status_reports_real_progress_budget_and_stop(self):
        self.state['inference_budget'] = {'remaining_calls': 0, 'remaining_usd': 0}
        self.state['actions'] = [self.action()]
        statuses = []
        def execute(action, context):
            context['progress'](pose='closing', completed_samples=3)
            return {'status': 'completed', 'evidence': ['receipt']}
        controller = self.controller(execute=execute, on_status=statuses.append)
        controller.tick()
        self.assertEqual([s['status'] for s in statuses], ['running', 'running', 'completed'])
        self.assertEqual(statuses[1]['progress']['completed_samples'], 3)
        self.assertEqual(read_json(self.root / 'status.json')['inference_budget']['remaining_calls'], 0)
        controller.cancelled.set()
        self.assertEqual(controller.tick()['status'], 'stopped')

    def test_idle_planning_does_not_repeat_unchanged_request(self):
        calls = []
        controller = self.controller(planner=lambda r: calls.append(r) or self.plan, plan_interval=0)
        controller.tick()
        self.assertTrue(controller.pending['done'].wait(1))
        controller.tick()
        controller.tick()
        self.assertEqual(len(calls), 1)

    def test_close_clears_pending_and_preserves_stop_context(self):
        planner = MailboxPlanner(self.root / 'mailbox')
        self.state['actions'] = [self.action()]
        controller = self.controller(planner=planner)
        result = controller.run(max_steps=1)
        self.assertEqual(result['stage'], 'repair')
        self.assertEqual(result['last_action'], 'measure')
        controller.close()
        self.assertFalse(read_json(self.root / 'status.json')['planner_pending'])

    def test_mailbox_matches_request_identity(self):
        planner = MailboxPlanner(self.root, timeout=2)
        self.addCleanup(planner.close)
        write_json(self.root / 'response.json', {'request_id': 'request1', 'plan': self.plan})
        self.assertEqual(planner({'request_id': 'request1'}), self.plan)

    def test_bounded_windows_replace_retry_preserves_complete_receipt_on_exhaustion(self):
        error = PermissionError('sharing denied'); error.winerror = 32
        path = self.root / 'response.json'
        write_json(path, {'old': True})
        with patch('modeling_system.controller.os.replace', side_effect=error) as replace, \
                patch('modeling_system.controller.time.sleep') as sleep:
            with self.assertRaises(PermissionError): write_json(path, {'complete_response': True})
            self.assertEqual(replace.call_count, 6)
            self.assertEqual(sleep.call_count, 5)
        self.assertEqual(read_json(path), {'old': True})
        self.assertEqual(read_json(next(self.root.glob('response.json.tmp-*'))), {'complete_response': True})

    def test_temporary_windows_sharing_failure_retries_metadata_only(self):
        import os
        real = os.replace
        error = PermissionError('sharing denied'); error.winerror = 5
        attempts = []
        def replace(source, destination):
            attempts.append(1)
            if len(attempts) < 3: raise error
            return real(source, destination)
        with patch('modeling_system.controller.os.replace', side_effect=replace), \
                patch('modeling_system.controller.time.sleep'):
            write_json(self.root / 'response.json', {'complete': True})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(list(self.root.glob('*.tmp-*')), [])

    def test_nonsharing_error_is_not_retried(self):
        with patch('modeling_system.controller.os.replace', side_effect=OSError('disk full')) as replace:
            with self.assertRaises(OSError): write_json(self.root / 'response.json', {'complete': True})
        self.assertEqual(replace.call_count, 1)


if __name__ == '__main__':
    unittest.main()
