from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .native_recipes import RetainCheckpoint, NativeJob
from .store import Store


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.refs = {}
        for key in ('source','candidate','reopen'):
            path = self.root/(key+'.json'); path.write_text('{}' if key=='reopen' else key)
            self.refs[key] = {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        self.live = {'owner':'owner','dirty':False,'expected_state':'s0',
                     'file':self.refs['source']['path'],'saved_file':self.refs['source']}
        self.calls = []; self.reject_display = False
        self.service = SimpleNamespace(execute=self.execute,store=Store(self.root/'store'))
        self.item = {'id':'retain','reads':{k:v['sha256'] for k,v in self.refs.items()},
            'workbench':{'profile':'retention','native':True,'visual_review':'candidate'},
            'payload':{**self.refs,'label':'Reviewed material candidate','target':str(self.root/'visible.blend'),
                       'pose':{'controls':{'pose':1.}},'display':{'mode':'GUIDE_WIRE'}}}
        self.context = {'owner':'owner','cancelled':threading.Event(),'attempt_key':'attempt','progress':lambda **k:None}

    def execute(self, operation, args):
        self.calls.append(operation)
        if operation != 'native_inspect_live':
            self.assertEqual(args['expected_state'], self.live['expected_state'])
        if operation == 'native_open_checkpoint':
            self.live.update(file=args['source']['path'], saved_file=args['source'])
        elif operation == 'native_set_controls': self.live['dirty'] = True
        elif operation == 'native_set_display' and self.reject_display:
            return {'status':'conflicting','reason':'Later user edit'}
        elif operation == 'native_save_checkpoint':
            path=Path(args['path']); path.write_bytes(b'saved')
            self.live.update(file=str(path),dirty=False,saved_file={'path':str(path),'sha256':hashlib.sha256(b'saved').hexdigest()})
        self.live['expected_state'] = 'state'+str(len(self.calls))
        result = {'status':'saved' if operation == 'native_save_checkpoint' else 'completed','live':deepcopy(self.live)}
        # Exercise compact responses: expanding these is local store I/O, not
        # another expensive native observation.
        record = self.service.store.put('operation',result)
        return {'status':result['status'],'detail_available':True,'operation_record':record}

    def handler(self):
        return RetainCheckpoint(self.service,self.root/'run',verify_reopen=lambda row,candidate: row=={})

    def test_retention_uses_returned_state_and_keeps_each_native_mutation_guarded(self):
        result = self.handler()(self.item,self.context)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(self.calls,['native_inspect_live','native_open_checkpoint','native_set_controls','native_set_display','native_save_checkpoint'])
        self.assertFalse(result['checkpoint_open_reused'])
        self.assertEqual(result['native_calls']['native_inspect_live'],1)

    def test_exact_already_loaded_candidate_avoids_reopen_and_still_saves(self):
        self.live.update(file=self.refs['candidate']['path'],saved_file=self.refs['candidate'])
        result=self.handler()(self.item,self.context)
        self.assertTrue(result['checkpoint_open_reused']); self.assertNotIn('native_open_checkpoint',self.calls)

    def test_later_user_work_never_replaced(self):
        self.live['dirty']=True
        with self.assertRaises(ValueError): self.handler()(self.item,self.context)
        self.assertEqual(self.calls,['native_inspect_live'])

    def test_intervening_native_conflict_stops_without_save_or_retry(self):
        self.reject_display=True
        with self.assertRaises(RuntimeError): self.handler()(self.item,self.context)
        self.assertNotIn('native_save_checkpoint',self.calls)
        self.assertTrue((self.root/'run/retain/display.json').is_file())

    def test_changed_pinned_source_or_rejected_reopen_refuses_before_native(self):
        Path(self.refs['source']['path']).write_text('changed')
        with self.assertRaises(ValueError): self.handler()(self.item,self.context)
        self.assertEqual(self.calls,[])

    def test_reopen_verdict_is_required_and_existing_targets_preserved(self):
        handler=RetainCheckpoint(self.service,self.root/'run',verify_reopen=lambda row,c:False)
        with self.assertRaises(ValueError): handler(self.item,self.context)
        Path(self.item['payload']['target']).write_text('later user save')
        with self.assertRaises(ValueError): self.handler()(self.item,self.context)
        self.assertEqual(self.calls,[])

    def test_native_job_requires_bound_script_before_dispatch(self):
        item=deepcopy(self.item); item['payload']={'runner':self.refs['source'],'live_source':self.refs['source'],
            'job':{'input':self.refs['source']['path'],'source_sha256':self.refs['source']['sha256'],
                   'script':'unbound.py','dependency_hashes':{}}}
        with self.assertRaises(ValueError): NativeJob(self.service,self.root/'run')(item,self.context)
        self.assertEqual(self.calls,[])


if __name__ == '__main__': unittest.main()
