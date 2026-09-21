from copy import deepcopy
import unittest

from .controller import read_json, write_json
from .failed_task_recovery import reconcile_failed_task
from . import test_operating_session as fixtures


class FailedTaskRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.OperatingTests(); self.base.setUp(); self.addCleanup(self.base.doCleanups)
        self.base.items = [self.base.item('activate', 'display')]
        self.receipt = self.base.root / 'partial.json'
        self.calls = []
        def partial(item, context):
            self.calls.append(item['id'])
            write_json(self.receipt, {'opened': True, 'pose_applied': False, 'saved': False})
            raise AssertionError('Final target was not saved')
        self.session = self.base.session(execute=partial)
        self.session.run(max_steps=1)
        self.handle = self.session.record['results']['activate']['operation_handle']
        self.folder = self.session.service.store.root / 'calls' / self.handle
        self.raised = (self.folder / 'result.json').read_bytes()
        self.observed = {'effect_status':'resolved_failed', 'basis':'Actual partial receipts',
            'completed_effects':['open'], 'unapplied_effects':['pose','save'], 'uncertain_effects':[]}
        self.evidence = [{'kind':'file', 'path':str(self.receipt), 'role':'Actual partial effects'}]
        self.report = {'checks':{'display':{'status':'fail', 'evidence':self.evidence}},
                      'findings':[{'kind':'failure', 'scope':'activation', 'summary':'Open succeeded; remaining display and save did not run.'}]}

    def recover(self, **changes):
        args = dict(expected_handle=self.handle, observed=self.observed, evidence=self.evidence, report=self.report)
        args.update(changes)
        return reconcile_failed_task(self.session, 'activate', **args)

    def test_partial_failure_stays_failed_and_cannot_replay_original_task(self):
        recovered = self.recover()
        self.assertEqual(recovered['status'], 'failed')
        self.assertEqual((self.folder/'result.json').read_bytes(), self.raised)
        self.assertFalse((self.folder/'capability-result.json').exists())
        self.session.run(max_steps=1)
        self.assertEqual(self.calls, ['activate'])
        record = read_json(self.session.directory/'controller/controller.json')
        self.assertEqual(record['attempts'][0]['status'], 'failed')

    def test_unknown_effects_and_wrong_handle_remain_unresolved(self):
        observed = deepcopy(self.observed); observed['uncertain_effects'] = ['save']
        for args in ({'observed':observed}, {'expected_handle':'different'}, {'evidence':[]}):
            with self.assertRaises(ValueError): self.recover(**args)
        self.assertEqual(self.session.record['results']['activate']['status'], 'needs_reconciliation')

    def test_known_return_is_not_relabelled_as_handler_failure(self):
        write_json(self.folder/'capability-result.json', {'status':'completed'})
        with self.assertRaises(ValueError): self.recover()


if __name__ == '__main__': unittest.main()
