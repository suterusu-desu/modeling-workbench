from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from .native_bridge import validate_arguments, NativeBridgeError
from . import native_retention as retention


class FakeHooks:
    def __init__(self, args, registry):
        self.args = args; self.calls = []; self.fail = None; self.after = {}
        self.live = {'owner': args['_owner'], 'dirty': False, 'expected_state': args['expected_state'],
            'file': args['source']['path'], 'saved_file': args['source'], 'guide_registry': registry,
            'geometry_state_id': None, 'scene_freshness': {'status': 'dirty'}}

    def step(self, name):
        self.calls.append(name)
        if self.fail == name: raise RuntimeError('Interrupted '+name)
        if name in self.after: self.after[name]()

    def check_dependencies(self): self.step('dependencies')
    def guard(self, args): self.step('guard'); return deepcopy(self.live)
    def rollback(self, before, folder):
        self.step('rollback'); shutil.copy2(before['file'], folder/'rollback.blend')
        return {'rollback': retention.reference(folder/'rollback.blend')}
    def open_candidate(self, candidate, args):
        self.live.update(file=candidate['path'], saved_file=candidate)
        self.step('open'); return {'opened': candidate}
    def restore_runtime(self, candidate, args, folder):
        self.step('restore'); return {'character_content_unchanged': True}
    def set_pose(self, pose): self.live['dirty'] = True; self.step('pose'); return pose
    def set_display(self, display): self.live['dirty'] = True; self.step('display'); return display
    def verify_presentation(self, pose, display):
        self.step('verify'); return {'declared_pose_guide_display_verified': True}
    def save(self, target):
        target.write_bytes(b'retained'); saved = retention.reference(target)
        self.live.update(file=str(target), dirty=False, saved_file=saved, expected_state='after')
        self.step('save'); return saved
    def observe(self): self.step('observe'); return deepcopy(self.live)


class RetentionTransactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); refs = {}
        for name in ('source', 'candidate', 'reopen', 'registry'):
            path = self.root/name; path.write_text(name); refs[name] = retention.reference(path)
        self.args = {k: refs[k] for k in ('source', 'candidate', 'reopen')}
        self.args.update(_owner='owner', expected_state='before', transaction_id='attempt',
            target=str(self.root/'target.blend'), label='Reviewed checkpoint',
            pose={'controls': {'blink': 0}, 'refresh': False}, display={'mode': 'GUIDE_WIRE'})
        self.hooks = FakeHooks(self.args, refs['registry'])

    def run_transaction(self):
        return retention.run_retention(self.args, self.hooks, self.root/'transactions')

    def receipt(self):
        return retention.inspect_receipt(self.root/'transactions', 'owner', 'attempt')

    def test_success_has_fresh_boundaries_and_durable_phases_without_intermediate_scans(self):
        result = self.run_transaction()
        self.assertEqual(result['status'], 'saved')
        self.assertEqual(self.hooks.calls.count('guard'), 1)
        self.assertEqual(self.hooks.calls.count('observe'), 1)
        self.assertEqual(result['saved'], retention.reference(self.args['target']))
        receipt = self.receipt(); self.assertTrue(receipt['historical'])
        self.assertFalse(receipt['replay_allowed']); self.assertFalse(result['user_appearance_accepted'])
        self.assertTrue(all(p['status'] == 'completed' for p in receipt['receipt']['phases']))

    def test_completed_attempt_never_returns_historical_live_state_as_current(self):
        self.run_transaction(); self.hooks.live['dirty'] = True
        count = len(self.hooks.calls)
        with self.assertRaises(FileExistsError): self.run_transaction()
        self.assertEqual(len(self.hooks.calls), count)

    def test_existing_claim_without_receipt_blocks_replay(self):
        retention.transaction_directory(self.root/'transactions', 'owner', 'attempt').mkdir(parents=True)
        with self.assertRaises(FileExistsError): self.run_transaction()
        self.assertEqual(self.hooks.calls, []); self.assertIsNone(self.receipt()['receipt'])

    def test_stale_state_dirty_source_wrong_owner_or_source_refuse_before_rollback(self):
        for change in ({'expected_state': 'later'}, {'dirty': True}, {'owner': 'someone'},
                       {'saved_file': {'path': self.args['source']['path'], 'sha256': 'changed'}}):
            with self.subTest(change=change):
                self.args['transaction_id'] = json.dumps(change, sort_keys=True)
                hooks = FakeHooks(self.args, self.hooks.live['guide_registry']); hooks.live.update(change)
                with self.assertRaises(ValueError): retention.run_retention(self.args, hooks, self.root/'transactions')
                self.assertNotIn('rollback', hooks.calls)

    def test_changed_pinned_input_refuses_before_native(self):
        Path(self.args['candidate']['path']).write_text('later save')
        with self.assertRaises(ValueError): self.run_transaction()
        self.assertEqual(self.hooks.calls, [])

    def test_already_loaded_candidate_still_rolls_back_but_skips_open(self):
        self.hooks.live.update(file=self.args['candidate']['path'], saved_file=self.args['candidate'])
        result = self.run_transaction()
        self.assertTrue(result['checkpoint_open_reused']); self.assertNotIn('open', self.hooks.calls)
        self.assertIn('rollback', self.hooks.calls)

    def test_restore_content_failure_stops_before_pose(self):
        self.hooks.restore_runtime = lambda *args: {'character_content_unchanged': False}
        with self.assertRaisesRegex(ValueError, 'preservation'): self.run_transaction()
        self.assertNotIn('pose', self.hooks.calls)

    def test_partial_failures_keep_last_started_phase_and_refuse_replay(self):
        for phase in ('open', 'restore', 'pose', 'display', 'save', 'observe'):
            with self.subTest(phase=phase):
                self.args['transaction_id'] = phase
                self.args['target'] = str(self.root/(phase+'.blend'))
                hooks = FakeHooks(self.args, self.hooks.live['guide_registry']); hooks.fail = phase
                with self.assertRaises(RuntimeError): retention.run_retention(self.args, hooks, self.root/'transactions')
                receipt = retention.inspect_receipt(self.root/'transactions', 'owner', phase)['receipt']
                self.assertEqual(receipt['status'], 'needs_reconciliation')
                self.assertEqual(receipt['phases'][-1], {'name': phase, 'status': 'started'})
                with self.assertRaises(FileExistsError): retention.run_retention(self.args, hooks, self.root/'transactions')
                self.assertEqual(hooks.calls.count(phase), 1)

    def test_changed_registry_or_target_before_save_preserves_external_work(self):
        for change in ('registry', 'target'):
            with self.subTest(change=change):
                self.args['transaction_id'] = change; self.args['target'] = str(self.root/(change+'.blend'))
                hooks = FakeHooks(self.args, self.hooks.live['guide_registry'])
                path = Path(hooks.live['guide_registry']['path'] if change == 'registry' else self.args['target'])
                original = path.read_bytes() if path.exists() else None
                hooks.after['display'] = lambda: path.write_bytes(b'concurrent external work')
                with self.assertRaises(ValueError): retention.run_retention(self.args, hooks, self.root/'transactions')
                self.assertNotIn('save', hooks.calls); self.assertEqual(path.read_bytes(), b'concurrent external work')
                if original is not None: path.write_bytes(original)

    def test_existing_target_never_overwritten(self):
        Path(self.args['target']).write_bytes(b'preserve')
        with self.assertRaises(ValueError): self.run_transaction()
        self.assertNotIn('guard', self.hooks.calls)

    def test_false_final_state_or_stale_geometry_refuses_completed_receipt(self):
        for change in ({'dirty': True}, {'owner': 'someone'}, {'geometry_state_id': 'stale'},
                       {'file': self.args['source']['path']}):
            with self.subTest(change=change):
                self.args['transaction_id'] = json.dumps(change, sort_keys=True)
                self.args['target'] = str(self.root/(str(len(list(self.root.iterdir())))+'.blend'))
                hooks = FakeHooks(self.args, self.hooks.live['guide_registry'])
                hooks.after['save'] = lambda: hooks.live.update(change)
                with self.assertRaises(ValueError): retention.run_retention(self.args, hooks, self.root/'transactions')
                self.assertTrue(Path(self.args['target']).is_file())

    def test_interrupted_receipt_write_after_save_does_not_repeat_save(self):
        original = retention._write
        def fail_after_save(path, data):
            if 'save' in data: raise OSError('disk unavailable after effect')
            original(path, data)
        with patch.object(retention, '_write', side_effect=fail_after_save):
            with self.assertRaises(OSError): self.run_transaction()
        self.assertTrue(Path(self.args['target']).exists())
        self.assertEqual(self.receipt()['receipt']['phases'][-1], {'name': 'save', 'status': 'started'})
        with self.assertRaises(FileExistsError): self.run_transaction()
        self.assertEqual(self.hooks.calls.count('save'), 1)

    def test_schema_excludes_diagnostics_and_arbitrary_native_actions(self):
        public = {k:v for k,v in self.args.items() if not k.startswith('_')}
        validate_arguments('retain_checkpoint', public)
        for change in ({'display': {'mode': 'DISTANCE'}}, {'pose': {'refresh': True}},
                       {'script': 'arbitrary.py'}, {'display': {'mode': 'GUIDE_WIRE', 'axis': 'VERTICAL'}}):
            with self.assertRaises(NativeBridgeError): validate_arguments('retain_checkpoint', {**public, **change})

    def test_retention_can_save_with_parts_set_aside_from_view(self):
        self.args['display'] = {'mode': 'GUIDE_WIRE', 'hide': ['upper lash', 'lower lash']}
        public = {k:v for k,v in self.args.items() if not k.startswith('_')}
        validate_arguments('retain_checkpoint', public)
        shown = []
        self.hooks.set_display = lambda display: shown.append(display) or display
        result = self.run_transaction()
        self.assertEqual(result['status'], 'saved')
        self.assertEqual(shown, [{'mode': 'GUIDE_WIRE', 'hide': ['upper lash', 'lower lash']}])
        with self.assertRaises(NativeBridgeError):
            validate_arguments('retain_checkpoint', {**public, 'display': {'mode': 'GUIDE_WIRE', 'hide': 'upper lash'}})


if __name__ == '__main__': unittest.main()
