"""Synthetic qualification of composed private scopes; no provider/native IO."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np

from . import cooperative_scopes as runner
from .runtime import source_manifest
from modeling_system import test_operating_session as fixtures
from modeling_system.controller import write_json, read_json


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.OperatingTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.root=self.f.root
        for name in ('AGENTS.md','MODELING.md','modeling-workspace.json'):
            (self.root/name).write_text('{}',encoding='utf-8')
        self.arrays=self.root/'saved.npz'
        np.savez(self.arrays,baseline=np.zeros((3,3)),target=np.ones((3,3)))
        scopes=[]
        for i in range(2):
            scope={'task':'measure'+str(i),'description':'Measure the prepared effect in a qualified region',
                'completion':'Exact numerical residuals','capability':'saved effect','method':'corresponding displacement',
                'source_label':'Synthetic matched saved arrays','facts':'Known corresponding points',
                'limits':'No appearance acceptance','operation':'prepared_effect',
                'files':{'arrays':str(self.arrays)},'bindings':{'evidence':['arrays']},
                'inputs':{name:{'path':str(self.arrays),'sha256':runner.sha(self.arrays),'array':name}
                          for name in ('baseline','target')},'parameters':{'tolerance':.01*(i+1)}}
            p=self.root/(str(i)+'.json');write_json(p,scope)
            scopes.append({'path':str(p),'sha256':runner.sha(p)})
        self.manifest=self.root/'manifest.json'
        write_json(self.manifest,{'workspace':str(self.root),'directory':str(self.root/'combined'),
            'runtime_revision':source_manifest()['revision'], 'state_file':'MODELING.md','owner':'native owner',
            'episode':self.f.episode,'objective':'Compare qualified saved effects','scopes':scopes,
            'task_order':['measure0','measure1'],'max_steps':4})

    def create(self):
        with patch.object(runner,'ModelingService',return_value=self.f.service):
            session,manifest=runner.create(self.manifest)
        return session

    def test_composed_tasks_execute_once_and_share_fresh_reads(self):
        session=self.create()
        self.assertEqual(session.run(max_steps=4)['status'],'completed')
        self.assertEqual(set(session.record['results']),{'measure0','measure1'})
        original={k:v['operation_handle'] for k,v in session.record['results'].items()}
        with patch.object(runner,'sha',wraps=runner.sha) as hashes:
            session.observe_context()
        paths=[str(Path(call.args[0]).resolve()) for call in hashes.call_args_list]
        self.assertEqual(len(paths),len(set(paths)))
        self.assertEqual(self.create().run(max_steps=4)['status'],'completed')
        self.assertEqual({k:v['operation_handle'] for k,v in read_json(session.path)['results'].items()},original)

    def test_changed_inputs_do_not_execute_or_redefine_frozen_tasks(self):
        session=self.create()
        frozen=(session.directory/'qualified-tasks.json').read_bytes()
        (self.root/'MODELING.md').write_text('Later user scope',encoding='utf-8')
        restarted=self.create()
        self.assertEqual(restarted.run(max_steps=4)['status'],'needs_review')
        self.assertEqual(restarted.record['results'],{})
        self.assertEqual((session.directory/'qualified-tasks.json').read_bytes(),frozen)

    def test_native_scope_and_changed_source_scope_are_refused(self):
        manifest=read_json(self.manifest);p=Path(manifest['scopes'][0]['path'])
        scope=read_json(p);scope['native_job']={'operation':'not allowed'};write_json(p,scope)
        with self.assertRaisesRegex(ValueError,'scope changed'):self.create()
        manifest['scopes'][0]['sha256']=runner.sha(p);write_json(self.manifest,manifest)
        with self.assertRaisesRegex(ValueError,'no native capability'):self.create()


if __name__=='__main__':unittest.main()
