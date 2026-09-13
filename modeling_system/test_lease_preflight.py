"""Synthetic pre-dispatch typo and narrowly provable legacy lease recovery."""
import json
import uuid
import unittest
from unittest.mock import patch
from . import test_decisions as fixtures
from .leases import acquire, list_leases
from .ledger import Conflict
from .store import atomic_write, canonical, digest


class LeasePreflightTests(unittest.TestCase):
    setUp=fixtures.EpisodeTests.setUp
    tearDown=fixtures.EpisodeTests.tearDown
    ep=fixtures.EpisodeTests.episode

    def test_invalid_arguments_never_acquire_lease_or_dispatch(self):
        e=self.ep()
        for operation,arguments in [
            ('reconcile_operation',{'handle':'a'*32,'effect_status':'none'}),
            ('unknown_operation',{}),('native_save_checkpoint',{'owner':'fixture'}),
        ]:
            with patch.object(self.s,'_native_call',side_effect=AssertionError('No native call')):
                result=self.s.execute('run_episode_operation',dict(episode=e['episode'],operation=operation,arguments=arguments))
            self.assertEqual(result['effect_status'],'refused before mutation dispatch')
            self.assertFalse(result['details']['lease_acquired'])
            self.assertEqual(list_leases(self.s,e['episode']),[])
            self.assertEqual(self.s.inspect_operations(e['episode'])['calls'],[])

    def legacy(self):
        e=self.ep();lease=acquire(self.s,e['episode'],'reconcile_operation')
        lease.update(status='needs effect reconciliation',result_status='failed',operation_handle=None)
        p=self.s.store.root/'episode-leases'/e['episode']/(lease['id']+'.json')
        atomic_write(p,canonical(lease))
        marker=self.s.store.root/'episode-completions'/e['episode']/(lease['id']+'.json')
        atomic_write(marker,canonical(dict(lease_id=lease['id'],episode=e['episode'],operation_handle=None,
            completion_disposition='needs effect reconciliation',result_status='failed')))
        receipt=dict(status='failed',operation='reconcile_operation',error_type='TypeError',
            summary="ModelingService.reconcile_operation() got an unexpected keyword argument 'effect_status'",
            arguments={'handle':uuid.uuid4().hex,'effect_status':'none'})
        key=self.s.store.put('operation',receipt)
        proof=self.root/'original-refusal.json';proof.write_bytes(canonical(dict(lease=lease,receipt=receipt)))
        args=dict(handle=lease['id'],episode=e['episode'],expected_lease=digest(p.read_bytes()),
            observed=dict(effect_status='confirmed_not_applied',lease_id=lease['id'],refusal_record=key,
                basis='Exact synthetic original return and lease association; no callable was entered'),evidence_paths=[str(proof)])
        return p,marker,args

    def test_exact_legacy_refusal_reconciles_without_replay_and_keeps_evidence(self):
        p,marker,args=self.legacy();before=marker.read_bytes()
        with patch.object(self.s,'_native_call',side_effect=AssertionError('No replay')):
            result=self.s.reconcile_operation(**args)
        self.assertEqual(result['effect_status'],'refused before mutation dispatch')
        self.assertEqual(json.loads(p.read_bytes())['status'],'finished')
        self.assertEqual(marker.read_bytes(),before)
        self.assertEqual(self.s.store.get(result['resolution'])['refusal_record'],args['observed']['refusal_record'])
        with self.assertRaises(Conflict):self.s.reconcile_operation(**args)

    def test_absence_or_later_good_state_is_not_refusal_proof(self):
        p,marker,args=self.legacy();before=p.read_bytes()
        args['observed']['refusal_record']=self.s.store.put('operation',dict(operation='reconcile_operation',status='failed',
            error_type='TypeError',summary='arbitrary error inside body',arguments={'handle':'a'*32}))
        with self.assertRaisesRegex(ValueError,'bind successfully'):self.s.reconcile_operation(**args)
        self.assertEqual(p.read_bytes(),before)

    def test_active_linked_and_ambiguous_leases_remain_blocked(self):
        p,marker,args=self.legacy();original=json.loads(p.read_bytes())
        for change in ({'status':'active'},{'operation_handle':'b'*32},{'operation':'native_save_checkpoint'}):
            atomic_write(p,canonical(dict(original,**change)));args['expected_lease']=digest(p.read_bytes())
            before=p.read_bytes()
            with self.assertRaisesRegex(ValueError,'active or ambiguous'):self.s.reconcile_operation(**args)
            self.assertEqual(p.read_bytes(),before)

    def test_finalization_keeps_allocated_handle_when_result_lacks_one(self):
        from .leases import finish, mark_return
        e=self.ep();lease=acquire(self.s,e['episode'],'record_outcome')
        mark_return(self.s,lease,{'status':'failed'});finish(self.s,lease,{'status':'failed'})
        self.assertEqual(list_leases(self.s,e['episode'])[0]['operation_handle'],lease['operation_handle'])


if __name__=='__main__':unittest.main()
