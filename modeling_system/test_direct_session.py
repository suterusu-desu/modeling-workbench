"""Direct ownership preserves execution contracts without an inference runtime."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from . import test_operating_session as fixtures
from .capability_calls import bind_call
from .controller import fingerprint, read_json, write_json
from .direct_session import create_session
from .operating_session import inspect_session
from .work_queue import OwnerSelection


class DirectSessionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.OperatingTests(); self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def session(self, **extra):
        f = self.f
        return create_session(f.root/'direct', service=f.service, episode=f.episode,
            owner='native owner', goal={'objective': 'Explicit qualified work'},
            observe_context=lambda: deepcopy(f.state), catalog=lambda s,r: deepcopy(f.items),
            handlers={'run': f.execute}, **extra)

    def test_direct_inspection_needs_no_key_ledger_network_or_selection(self):
        f = self.f
        task = f.item('inspect', 'inspection', native=True)
        task['workbench']['bindings']['adapter'] = ['adapter']
        f.items = [task]
        # A qualified synthetic native handler exercises the same contract; this
        # test does not access Blender. Even an inherited secret path is unused.
        forbidden = f.root/'external-key.txt'; forbidden.write_text('fixture-secret')
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path == forbidden: raise AssertionError('Credential read')
            return original(path, *args, **kwargs)
        with patch.dict(os.environ, {'TYPESAFE_API_KEY_FILE': str(forbidden)}), \
                patch.object(Path, 'read_text', read), \
                patch('socket.create_connection', side_effect=AssertionError('Network call')):
            session = self.session()
            self.assertEqual(session.run(max_steps=3)['status'], 'completed')
            self.assertEqual(session.run(max_steps=3)['status'], 'completed')
        self.assertEqual(f.ran, ['inspect'])
        self.assertFalse((session.directory/'advice').exists())
        self.assertFalse((session.directory/'judgments').exists())
        self.assertEqual(session.metrics()['action_routes'], {'single_executable': 1})

    def test_multiple_operations_require_explicit_order_and_resume_without_replay(self):
        f = self.f; f.items = [f.item('first'), f.item('second')]
        session = self.session()
        self.assertEqual(session.run(max_steps=2)['status'], 'needs_review')
        self.assertEqual(f.ran, [])
        session = self.session(task_order=['second', 'first'])
        self.assertEqual(session.run(max_steps=1)['status'], 'stopped')
        self.assertEqual(f.ran, ['second'])
        session = self.session(task_order=['second', 'first'])
        self.assertEqual(session.run(max_steps=4)['status'], 'completed')
        self.assertEqual(f.ran, ['second', 'first'])
        self.assertEqual(session.metrics()['owner_choices_by_lane'], {'diagnosis': 1})

    def test_order_does_not_bypass_blocked_prerequisites_or_failed_checks(self):
        f = self.f
        f.items = [f.item('inspect'), f.item('fit', requires={'inspect': ['completed']})]
        f.reports['inspect'] = {'checks': {'analysis': {'status': 'fail', 'evidence': [
            {'kind': 'file', 'path': str(f.base.image), 'role': 'Failed support'}]}}, 'findings': []}
        f.items[1]['workbench']['consumes'] = {'inspect': ['analysis']}
        result = self.session(task_order=['fit', 'inspect']).run(max_steps=4)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(f.ran, ['inspect'])

    def test_factory_requires_preservation_for_appearance_even_when_singleton(self):
        f = self.f; f.items = [f.item('edit', 'appearance_edit')]
        result = self.session().run(max_steps=2)
        self.assertEqual(result['status'], 'needs_review')
        self.assertEqual(f.ran, [])

    def test_old_inferred_catalog_is_not_silently_executed(self):
        f = self.f; f.items = [f.item('legacy')]
        f.items[0]['select_with_jev'] = True
        result = self.session().run(max_steps=2)
        self.assertEqual(result['status'], 'error')
        self.assertIn('Legacy inferred', result['reason'])
        self.assertEqual(f.ran, [])

    def test_uncertain_native_effect_stays_stopped_after_restart(self):
        f = self.f; f.items = [f.item('inspect')]
        session = self.session()
        def interrupted(item, context):
            f.ran.append(item['id'])
            raise RuntimeError('Worker disconnected after possible effect')
        session.private_handlers['run'] = interrupted
        self.assertEqual(session.run(max_steps=2)['status'], 'needs_reconciliation')
        self.assertEqual(self.session().run(max_steps=2)['status'], 'needs_reconciliation')
        self.assertEqual(f.ran, ['inspect'])

    def test_historical_decision_receipts_remain_readable_without_transport(self):
        f = self.f; f.items = [f.item('inspect')]
        session = self.session(); session.run(max_steps=2)
        trace = {'choice': 'inspect', 'selection_key': 'historical', 'status': 'retained'}
        write_json(session.directory/'advice/decisions/historical.json', trace)
        record = read_json(session.path)
        record['results']['inspect']['decision'] = {'selection_key': 'historical',
            'trace_revision': fingerprint(trace)}
        write_json(session.path, record)
        before = (session.directory/'advice/decisions/historical.json').read_bytes()
        report = inspect_session(f.service, session.directory)
        self.assertTrue(report['decision_outcomes'])
        self.assertEqual((session.directory/'advice/decisions/historical.json').read_bytes(), before)

    def test_retired_inference_modules_are_not_installed(self):
        for module in ('jev_session','typesafe_transport','provider_dispatch','provider_recovery',
                       'credentials','judgments','planning_batch'):
            self.assertIsNone(importlib.util.find_spec('modeling_system.'+module))

    def test_invalid_order_is_rejected(self):
        for order in (['same', 'same'], [''], 'task'):
            with self.assertRaises(ValueError): OwnerSelection(order)

    def test_typed_tasks_cannot_run_unbound_or_with_a_changed_payload(self):
        f = self.f
        task = f.item('inspect')
        task['parameters'] = {'arguments': {'region': {'description': 'Qualified region',
            'path': ['region'], 'options': [{'id': 'local', 'description': 'Local support',
                'value': [1,2], 'reads': {'source': 's1'}}]}}}
        f.items = [task]
        self.assertEqual(self.session().run(max_steps=1)['status'], 'error')
        f.items = [bind_call(task, {'region': 'local'})]
        f.items[0]['payload']['region'] = [999]
        self.assertEqual(self.session().run(max_steps=1)['status'], 'error')
        self.assertEqual(f.ran, [])
        f.items = [bind_call(task, {'region': 'local'})]
        self.assertEqual(self.session().run(max_steps=2)['status'], 'completed')
        self.assertEqual(f.ran, ['inspect'])


class DirectArgumentTests(unittest.TestCase):
    def item(self):
        return {'id': 'fit', 'reads': {'guide': 'g1'}, 'payload': {}, 'workbench': {},
            'parameters': {'arguments': {
                'support': {'description': 'Qualified guide support', 'path': ['support'],
                    'options': [{'id': 'local', 'description': 'Local support', 'value': [1,2], 'reads': {'guide':'g1'}},
                                {'id': 'whole', 'description': 'Connected support', 'value': [1,2,3]}]},
                'poses': {'description': 'Evaluated phases', 'path': ['poses'], 'type': 'set',
                    'options': [{'id': 'open', 'description': 'Open phase', 'value': 0},
                                {'id': 'closed', 'description': 'Closed phase', 'value': 1}]}
            }, 'incompatible': [{'support': 'local', 'poses': 'closed'}]}}

    def test_explicit_arguments_preserve_qualified_values_and_provenance(self):
        item = self.item(); before = deepcopy(item)
        bound = bind_call(item, {'support':'whole','poses':['open','closed']})
        self.assertEqual(bound['payload'], {'support':[1,2,3], 'poses':[0,1]})
        self.assertEqual(item, before)
        self.assertEqual(bound['workbench']['invocation']['definition'], fingerprint(item))

    def test_missing_incompatible_stale_and_outside_arguments_cannot_bind(self):
        for choices in ({'support':'whole'}, {'support':'invented','poses':['open']},
                        {'support':'local','poses':['closed']}, {'support':'whole','poses':[]},
                        {'support':'whole','poses':['open','open']}):
            with self.assertRaises(ValueError): bind_call(self.item(), choices)
        item = self.item(); item['reads']['guide'] = 'changed'
        with self.assertRaises(ValueError): bind_call(item, {'support':'whole','poses':['open']})


if __name__ == '__main__': unittest.main()
