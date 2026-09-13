"""Valid destination paths and durable completion repair after write failure."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .store import atomic_write
from . import test_decisions as fixtures


class AtomicPathTests(unittest.TestCase):
    def test_223_character_destination_does_not_need_260_character_temp(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);remaining=185-len(str(root))-1
            if not 1<=remaining<=255:self.skipTest('Temporary root exceeds this path fixture')
            target=root/('p'*remaining)/('a'*32+'.json')
            self.assertEqual(len(str(target)),223)
            atomic_write(target,b'completed')
            self.assertEqual(target.read_bytes(),b'completed')
            self.assertEqual(list(target.parent.iterdir()),[target])

    def test_valid_long_basename_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            # Windows also constrains full path length when long paths are off.
            width=min(240,250-len(d)-1) if os.name=='nt' else 240
            target=Path(d)/('b'*width)
            atomic_write(target,b'before');atomic_write(target,b'after')
            self.assertEqual(target.read_bytes(),b'after')

    def test_failed_replace_preserves_original_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'record.json';target.write_bytes(b'original')
            with patch('modeling_system.store.os.replace',side_effect=OSError('replace failed')):
                with self.assertRaises(OSError):atomic_write(target,b'new')
            self.assertEqual(target.read_bytes(),b'original')
            self.assertEqual(list(target.parent.iterdir()),[target])


class CompletionRepairTests(unittest.TestCase):
    setUp=fixtures.EpisodeTests.setUp
    tearDown=fixtures.EpisodeTests.tearDown

    def call_with_missing_marker(self,failed=False):
        class Native:
            calls=0
            def call(self,*a,**kw):
                self.calls+=1
                if failed:raise RuntimeError('Unknown synthetic effect')
                return {'status':'completed','checkpoint':'synthetic-only'}
        self.s.native=Native();episode=fixtures.EpisodeTests.episode(self)
        with patch('modeling_system.service.mark_return',side_effect=OSError('Completion write failed')):
            r=self.s.run_episode_operation(episode['episode'],'native_save_checkpoint',
                {'expected_state':'fixture','owner':'fixture','label':'fixture'})
        folder=self.s.store.root/'calls'/r['operation_handle']
        intent=json.loads((folder/'intent.json').read_bytes());lease=intent['lease']
        marker=self.s.store.root/'episode-completions'/episode['episode']/(lease['id']+'.json')
        self.assertFalse(marker.exists())
        return r,folder,lease,marker

    def test_public_reconciliation_reconstructs_marker_without_replay_or_rewriting_result(self):
        r,folder,lease,marker=self.call_with_missing_marker()
        before={name:(folder/name).read_bytes() for name in ('intent.json','result.json')}
        with patch.object(self.s.native,'call',side_effect=AssertionError('No replay')):
            fixed=self.s.reconcile_operation(r['operation_handle'])
            again=self.s.reconcile_operation(r['operation_handle'])
        self.assertEqual(fixed['completion_repair']['status'],'recovered from original durable result')
        self.assertEqual(again['completion_repair']['status'],'existing receipt; lease finalized')
        self.assertEqual(json.loads(marker.read_bytes())['completion_disposition'],'finished')
        retained=json.loads((self.s.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json')).read_bytes())
        self.assertEqual(retained['status'],'finished')
        self.assertEqual(before,{name:(folder/name).read_bytes() for name in before})

    def test_unknown_effect_does_not_gain_a_completion_marker(self):
        r,folder,lease,marker=self.call_with_missing_marker(failed=True)
        fixed=self.s.reconcile_operation(r['operation_handle'])
        self.assertIsNone(fixed['completion_repair'])
        self.assertFalse(marker.exists())
        self.assertEqual(fixed['effect_status'],'unknown')


if __name__=='__main__':unittest.main()
