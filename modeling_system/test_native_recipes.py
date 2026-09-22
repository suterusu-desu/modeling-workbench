from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from .native_recipes import RetainCheckpoint, NativeJob, validate_native_job
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
        elif operation in ('native_save_checkpoint', 'native_retain_checkpoint'):
            path=Path(args.get('path', args.get('target'))); path.write_bytes(b'saved')
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

    def test_transaction_opt_in_uses_one_guarded_native_transaction_after_preflight(self):
        handler = RetainCheckpoint(self.service, self.root/'run', verify_reopen=lambda r,c: True, transaction=True)
        result = handler(self.item, self.context)
        self.assertEqual(self.calls, ['native_inspect_live', 'native_retain_checkpoint'])
        self.assertTrue(result['retention_transaction'])

    def test_transaction_refuses_refresh_and_cancellation_before_effects(self):
        handler = RetainCheckpoint(self.service, self.root/'run', verify_reopen=lambda r,c: True, transaction=True)
        self.item['payload']['pose']['refresh'] = True
        with self.assertRaises(RuntimeError): handler(self.item, self.context)
        self.assertEqual(self.calls, [])
        self.item['payload']['pose']['refresh'] = False
        self.context['cancelled'].set()
        with self.assertRaises(RuntimeError): handler(self.item, self.context)
        self.assertEqual(self.calls, [])

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

    def test_native_job_collects_phase_evidence_without_reclassifying_effect(self):
        for suffix, phases in [('valid', {'schema_version': 1, 'status': 'completed', 'stages': [
                {'name': 'driver_restore', 'elapsed_ms': 23., 'status': 'completed'}]}),
                ('invalid', {'schema_version': 1, 'status': 'completed', 'stages': [{'name': 'bad'}]})]:
            output = self.root/suffix; output.mkdir()
            (output/'native-stages.json').write_text(json.dumps(phases))
            (output/'receipt.json').write_text(json.dumps({'status': 'completed', 'source_unchanged': True,
                'source_sha256_before': self.refs['source']['sha256'], 'source_sha256_after': self.refs['source']['sha256']}))
            source = self.refs['source']
            checkpoint = self.root/(suffix+'.blend'); checkpoint.write_bytes(Path(source['path']).read_bytes())
            item = {'id': suffix, 'reads': {'source': source['sha256']}, 'description': 'Qualified isolated job',
                'workbench': {'native': True}, 'payload': {'runner': source, 'live_source': source,
                    'job': {'blender': 'qualified-blender', 'output_root': str(self.root/'output'),
                            'input': str(checkpoint), 'source_sha256': source['sha256'],
                            'script': source['path'], 'dependency_hashes': {source['path']: source['sha256']}}}}
            def launch(*args, **kwargs):
                kwargs['stdout'].write(json.dumps({'status': 'completed', 'output': str(output)})+'\n')
                return SimpleNamespace(pid=1, returncode=0, poll=lambda: 0)
            with patch('modeling_system.native_recipes.subprocess.Popen', side_effect=launch):
                result = NativeJob(self.service, self.root/'jobs')(item, self.context)
            self.assertEqual(result['status'], 'completed')
            self.assertIn(str(output/'native-stages.json'), result['evidence'])
            if suffix == 'valid':
                self.assertEqual(result['native_stages_ms']['worker_driver_restore'], 23.)
            else:
                self.assertTrue(result['worker_timing_status'].startswith('invalid'))
                self.assertEqual(set(result['native_stages_ms']), {'isolated_job'})

    def test_missing_runner_field_refuses_before_live_access_or_output(self):
        item = {'workbench': {'native': True}, 'payload': {'job': {'script': 'script.py'}}}
        with self.assertRaisesRegex(ValueError, 'blender'):
            NativeJob(self.service, self.root/'jobs')(item, self.context)
        self.assertEqual(self.calls, []); self.assertFalse((self.root/'jobs').exists())

    def test_runner_preflight_is_structural_and_has_no_file_or_native_effects(self):
        ref = {'path': 'bound-file', 'sha256': 'a'*64}
        item = {'workbench': {'native': True}, 'payload': {'runner': ref, 'live_source': ref,
            'job': {'blender': 'blender', 'input': 'source.blend', 'script': 'worker.py', 'output_root': 'new-run',
                    'source_sha256': 'a'*64, 'dependency_hashes': {'worker.py': 'b'*64}}}}
        with patch('modeling_system.native_recipes.checked_file', side_effect=AssertionError('No I/O')):
            validate_native_job(item)
            for key, value in [('input', 'not-a-checkpoint.json'), ('timeout_seconds', float('inf')),
                               ('threads', -1), ('dependency_hashes', {})]:
                bad = deepcopy(item); bad['payload']['job'][key] = value
                with self.assertRaises(ValueError): validate_native_job(bad)


if __name__ == '__main__': unittest.main()
