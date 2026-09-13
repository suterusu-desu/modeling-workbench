"""Numerical intervention, invalidation, reuse and finite coverage regressions."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .service import ModelingService
from . import test_decisions as fixtures
from .responses import pointwise

class ResponseTests(unittest.TestCase):
    setUp=fixtures.EpisodeTests.setUp
    tearDown=fixtures.EpisodeTests.tearDown

    def response(self,kind='exact_linear',missing=None):
        dep=self.s.record_dependencies(self.state)['dependencies']
        ev={'kind':'file','path':str(self.image),'role':'fixture ancestry and guide evidence'}
        g=self.s.register_semantic_graph([{'id':'face','role':'fixture owned face','interpretation':'construction_evidence'}],[],
               {'included':['fixture face'],'missing':[]},dep)['graph']
        valid=self.s.validate_semantics(g,dep,'valid','Explicit fixture semantic ownership',[ev])['validation']
        z=dict(data=np.ones(3),indices=np.arange(3),indptr=np.arange(4),shape=np.array([3,3]),
               baseline_input=np.array([0.,0.,2.]),baseline_output=np.array([0.,0.,2.]),output_ids=np.arange(3))
        if kind=='exact_piecewise':
            z.update(limit=np.array([.1,.1,2.1]),epsilon=np.array(.1),selected=np.ones(3))
            z['baseline_output']=pointwise(z,z['baseline_input'])
        path=self.root/'response.npz';np.savez_compressed(path,**z)
        spec=dict(kind=kind,adapter='sparse_pointwise_v1',input_domain={'object':'Face','axis':1,'frame':'world'},
                  output_domain={'object':'Face','axis':1,'frame':'world'},units='fixture world units',
                  validity={'dependency_domains':['objects','pose','guide','bindings','targets'],'max_abs_delta':.5,'baseline_tolerance':1e-8},
                  influence={'included':['all three fixture vertices'],'missing':missing or [],'boundary':['triangle edges'],
                             'complete_for_controls':[0,1,2]})
        r=self.s.register_response(dep,valid,spec,str(path),[ev])
        return r['response'],dep,ev

    def test_prediction_native_and_visual_dispositions_stay_separate(self):
        response,dep,ev=self.response()
        p=self.s.predict_response(response,dep,{'0':0.0},[ev])
        report=self.s.compare_prediction(p['prediction'],self.state,'Face',1,1e-9,
            {'whole':'inspected','close':'inspected','angles':['front'],'guide_depth_sections':'fixture known plane',
             'motion':'not inspected','appearance':{'status':'rejected','reason':'broader visible regression'},'evidence':[ev]})
        self.assertEqual(report['numerical_status'],'matches')
        self.assertEqual(report['inspection']['appearance']['status'],'rejected')
        self.assertIn('not appearance approval',report['user_appearance_acceptance'])

    def test_stale_pose_support_and_out_of_range_refused(self):
        response,dep,ev=self.response()
        value=self.s.store.get(dep,'dependency_state');value['pose']={'blink':.5}
        changed=self.s.store.put('dependency_state',value)
        with self.assertRaisesRegex(ValueError,'dependencies changed'):self.s.predict_response(response,changed,{'0':.1},[ev])
        with self.assertRaisesRegex(ValueError,'validity'):self.s.predict_response(response,dep,{'0':1},[ev])
        value=self.s.store.get(dep);value['objects']['Face']['source']='changed ocular modifier'
        changed=self.s.store.put('dependency_state',value)
        with self.assertRaises(ValueError):self.s.predict_response(response,changed,{'0':.1},[ev])

    def test_piecewise_prediction_crosses_branch_correctly(self):
        response,dep,ev=self.response('exact_piecewise')
        p=self.s.predict_response(response,dep,{'0':.4},[ev])
        self.assertAlmostEqual(p['predicted'][0],.1)
        self.assertNotAlmostEqual(p['predicted'][0],.4)
        self.assertEqual(p['kind'],'exact_piecewise')

    def test_changed_view_reuses_geometry_but_not_capture(self):
        response,dep,ev=self.response()
        other=self.s.store.get(dep);other['view']='another camera';new=self.s.store.put('dependency_state',other)
        self.assertTrue(self.s.check_reuse(response,dep,new,['objects'])['reusable'])
        self.assertFalse(self.s.check_reuse(response,dep,new,['view'])['reusable'])

    def test_internally_consistent_but_false_native_baseline_refused(self):
        response,dep,ev=self.response()
        record=self.s.store.get(response)
        path=self.s.store.resolve_blob(record['arrays'])
        with np.load(path) as loaded:z={k:loaded[k] for k in loaded.files}
        z['baseline_input']=z['baseline_input']+1;z['baseline_output']=z['baseline_output']+1
        fake=self.root/'fake-native.npz';np.savez_compressed(fake,**z)
        with self.assertRaisesRegex(ValueError,'actual recorded native'):
            self.s.register_response(dep,record['semantic_validation'],record['spec'],str(fake),[ev])

    def test_incomplete_influence_is_explicit_and_blocks_native_proposal(self):
        response,dep,ev=self.response(missing=['lash neighbor unmeasured'])
        p=self.s.predict_response(response,dep,{'0':.1},[ev])
        self.assertFalse(p['influence']['complete_for_this_change'])
        self.assertIn('partial influence',p['readiness'])

    def test_method_promotion_requires_supported_reuse(self):
        e=fixtures.EpisodeTests.episode(self)
        j=self.s.reconcile_episode(e['episode'],e['revision'],character={'status':'unresolved','reason':'Appearance pending'},
            method={'status':'supported','reason':'Recorded failure recovered'},
            evidence=[{'kind':'file','path':str(self.image),'role':'recovery evidence'}],applicability='fixture only')['judgment']
        p=self.s.promote_procedure('recover',[j],'Recover same result',['recovery'],{},['fixture only'])
        self.assertEqual(p['level'],'local')
        with self.assertRaisesRegex(ValueError,'distinct episodes'):
            self.s.promote_procedure('recover',[j,j],'Recover',['recovery'],{},['fixture only'],level='reusable')

if __name__=='__main__':unittest.main()
