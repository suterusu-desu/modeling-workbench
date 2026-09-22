from copy import deepcopy
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import test_operating_session as fixtures
from .candidate_pipeline import CandidatePipeline
from .method_catalog import MethodCatalog, ParameterizedCatalog
from .preparation import (ArrayPreparation, active_vertex_coverage, prepared_effect,
                          compose_correspondence, pose_correspondence, prepare_arrays)
from .retained_context import RetainedContext
from .controller import read_json


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.OperatingTests(); self.base.setUp()
        self.addCleanup(self.base.doCleanups)

    def method(self, key, **extra):
        return {'id': key, 'description': {'mechanism': key, 'limits': 'Qualified synthetic input only'},
                'build': lambda state, previous: self.base.item(key, lane='preparation'), **extra}

    def test_methods_are_real_choices_and_selected_result_routes_followup(self):
        methods = MethodCatalog([self.method('path', candidates=lambda s, p: [self.base.item('apply-path', 'appearance_edit', lane='repair')]),
                                 self.method('section', candidates=lambda s, p: [self.base.item('apply-section', 'appearance_edit', lane='repair')])])
        pipeline = CandidatePipeline(self.base.root/'pipeline', revision='methods-v1', prepare=methods,
            candidates=methods.followup('candidates'),
            verify=lambda s, p: [self.base.item('verify', 'verification', lane='verification')],
            retain=lambda s, p: [self.base.item('retain', 'retention', lane='repair')])
        session = self.base.session(); session.user_catalog = pipeline
        session.select = lambda state, actions, plan: 'section'
        self.assertEqual(session.run(max_steps=5)['status'], 'needs_review')
        self.assertEqual(self.base.ran, ['section', 'apply-section'])
        self.assertEqual(session.metrics()['owner_choices_by_lane'], {'preparation': 1})
        self.assertEqual(session.metrics()['action_routes']['single_executable'], 1)




    def test_selected_method_native_continuation_runs_directly(self):
        def candidate(state, previous):
            row = self.base.item('apply', 'appearance_edit', lane='repair', native=True)
            return row
        methods = MethodCatalog([self.method('prepare', candidate=candidate)])
        pipeline = CandidatePipeline(self.base.root/'pipeline', revision='one-method', prepare=methods,
            candidates=methods.followup('candidate'), verify=lambda s,p: [self.base.item('verify','verification')],
            retain=lambda s,p: [self.base.item('retain','retention')])
        session = self.base.session(); session.user_catalog = pipeline
        self.assertEqual(session.run(max_steps=4)['status'], 'needs_review')
        self.assertEqual(self.base.ran,['prepare','apply'])
        self.assertEqual(len(self.base.batches),0)

    def test_parameterized_binding_reuses_implementation_and_excludes_missing_support(self):
        def bind(state, previous, parameters):
            return self.base.item(parameters['target'])
        bindings = [{'id':'sections','implementation':'inspect','description':'Compare measured sections',
                     'parameters':{'target':'section_paths'}},
                    {'id':'motion','implementation':'inspect','description':'Compare declared material pairs through motion',
                     'parameters':{'target':'motion_pairs'}, 'conditions':{'paired_motion':True}}]
        catalog = ParameterizedCatalog({'inspect':bind}, bindings)
        self.assertEqual([r['id'] for r in catalog(self.base.state)],['section_paths'])
        self.base.state['public_state']['paired_motion']=True
        rows=catalog(self.base.state)
        self.assertEqual(len(rows),2)
        self.assertEqual({r['workbench']['method_choice'] for r in rows}, {'sections','motion'})
        self.assertEqual(rows[1]['workbench']['recipe']['implementation'],'inspect')
        duplicate=deepcopy(bindings[0]);duplicate['id']='another_name'
        with self.assertRaises(ValueError): ParameterizedCatalog({'inspect':bind},[bindings[0],duplicate])

    def test_parameterized_selected_method_routes_its_original_parameters_into_native_followup(self):
        def bind(state,previous,parameters): return self.base.item('prepare_'+parameters['region'])
        def apply(state,previous,parameters):
            self.assertEqual(previous['task']['id'],'prepare_inner')
            return self.base.item('apply_'+parameters['region'],'appearance_edit',native=True,lane='repair')
        catalog=ParameterizedCatalog({'material':{'build':bind,'candidate':apply}},[
            {'id':'inner','implementation':'material','description':'Qualified inner material law','parameters':{'region':'inner'}},
            {'id':'outer','implementation':'material','description':'Qualified outer material law','parameters':{'region':'outer'}}])
        pipeline=CandidatePipeline(self.base.root/'pipeline',revision='parameters',prepare=catalog,
            candidates=catalog.followup('candidate'),verify=lambda s,p:[self.base.item('verify','verification')],
            retain=lambda s,p:[self.base.item('retain','retention')])
        session=self.base.session();session.user_catalog=pipeline
        session.run(max_steps=4)
        self.assertEqual(self.base.ran,['prepare_inner','apply_inner'])
        self.assertEqual(len(self.base.batches),1)
        self.assertEqual(session.metrics()['parameterized_implementations'],{'material':2})

    def test_review_timing_is_attached_to_actual_evidence_and_unknowns_are_visible(self):
        self.base.items=[self.base.item('candidate')]
        session=self.base.session();session.run(max_steps=1)
        with patch('modeling_system.operating_session.time.time', return_value=100):
            opened=session.begin_review('candidate')
            self.assertEqual(opened['timer'],session.begin_review('candidate')['timer'])
        with patch('modeling_system.operating_session.time.time', return_value=109):
            session.record_review('candidate',expected_basis=opened['basis'],
                judgment={'disposition':'useful','scope':'fixture','reason':'Inspected actual returned image','next_question':'motion'},
                evidence=[{'kind':'file','path':str(self.base.base.image),'role':'image'}])
        self.assertEqual(session.metrics()['reported_astra_seconds'],9.)
        self.assertEqual(session.metrics()['visual_reviews_without_timing'],0)

    def test_exact_conditions_and_missing_support_do_not_invent_options(self):
        methods = MethodCatalog([self.method('one', conditions={'support_kind': 'sections'})])
        self.assertEqual(methods(self.base.state), [])
        self.base.state['public_state']['support_kind'] = 'sections'
        self.assertEqual(len(methods(self.base.state)), 1)

    def test_timers_survive_restart_and_finish_idempotently(self):
        session = self.base.session()
        with patch('modeling_system.operating_session.time.time', return_value=100):
            token = session.begin_intervention(kind='catalog_preparation', reason='Qualify real alternatives')
        self.assertEqual(session.metrics()['open_intervention_timers'], [token])
        with patch('modeling_system.operating_session.time.time', return_value=112.5):
            session = self.base.session()
            evidence = [{'kind': 'file', 'path': str(self.base.base.image), 'role': 'Actual source'}]
            first = session.end_intervention(token, evidence=evidence)
            self.assertEqual(session.end_intervention(token, evidence=evidence), first)
        self.assertEqual(session.metrics()['reported_astra_seconds'], 12.5)
        self.assertEqual(session.metrics()['astra_interventions']['catalog_preparation'], 1)

    def test_experience_is_fresh_private_and_batched_with_choices(self):
        path = self.base.root/'private-lesson.md'
        path.write_text('A failed section fit used the wrong coordinate frame.', encoding='utf-8')
        session = self.base.session()
        session.user_catalog = MethodCatalog([self.method('section'), self.method('path')])
        session.experience = RetainedContext(self.base.service, query='section coordinate',
            sources=[{'path': str(path), 'role': 'experience'}],
            project=lambda row: {'lesson': row['excerpt']})
        batches = []
        before = session.observe()
        session.select = lambda state, actions, plan: 'section'
        session.run(max_steps=1)
        self.assertTrue(before['observations']['experience']['passages'])
        self.assertNotIn(str(path), json.dumps(before['observations']['experience']))
        path.write_text('A section fit changed; use the recorded world coordinate frame.', encoding='utf-8')
        after = session.observe()
        self.assertNotEqual(before['values']['operating_experience'], after['values']['operating_experience'])

    def test_settled_failure_can_offer_real_recovery_choices_without_replay(self):
        failed = self.base.item('failed')
        self.base.items = [failed]
        self.base.results['failed'] = {'status': 'failed', 'workbench': {'checks': {}, 'findings': []}}
        session = self.base.session(); session.run(max_steps=1)
        methods = MethodCatalog([self.method('repair_input'), self.method('use_supported_path')])
        def catalog(state, outcomes):
            rows = methods(state, outcomes)
            for row in rows:
                row['lane'] = 'recovery'; row['requires'] = {'failed': ['failed']}
            return [failed] + rows
        session.user_catalog = catalog
        session.run(max_steps=1)
        self.assertEqual(self.base.ran, ['failed', 'repair_input'])
        self.assertEqual(session.metrics()['owner_choices_by_lane'], {'recovery': 1})


class PreparationTests(unittest.TestCase):
    def test_active_geometry_excludes_unused_vertices_and_effect_exposes_noop(self):
        result = active_vertex_coverage(8, [[0,1,2], [1,3,2]], [0,1,2,3])
        self.assertTrue(result['complete_active_coverage']); self.assertEqual(result['unused_vertex_count'],4)
        baseline = np.zeros((8,3)); target = baseline.copy(); target[7]=100
        effect = prepared_effect(baseline, target, active_indices=[0,1,2,3], tolerance=1e-6)
        self.assertFalse(effect['has_effect_above_tolerance'])
        self.assertEqual(effect['changed_count'], 0)

    def test_continuous_composition_keeps_original_stationary_contribution(self):
        baseline = np.array([[1.,2.,3.], [2.,3.,4.], [3.,4.,5.]])
        before = np.ones((3,3)); target = before + 2
        weights = np.array([0., .3, 1.])
        result = compose_correspondence(baseline,before,target,weights)
        np.testing.assert_array_equal(result['target'][0],baseline[0])
        np.testing.assert_allclose(result['target'] - target*weights[:,None], result['preserved_contribution'])

    def test_pose_comparison_uses_correspondence_and_saved_values(self):
        guide=np.array([[0.,1.,0.],[1.,1.,0.]])
        result=pose_correspondence(guide, np.stack([guide+1,guide,guide+.4]), [0.,.4,1.])
        self.assertEqual(result['best_sample_index'],1)
        self.assertEqual(result['rms'][1],0.)

    def test_declarative_preparation_checks_bytes_and_runs_in_existing_handler(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.npz'; np.savez(path, baseline=np.zeros((3,3)), target=np.ones((3,3)))
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            inputs={key:{'path':str(path),'sha256':sha,'array':key} for key in ('baseline','target')}
            handler=ArrayPreparation(Path(d)/'outputs')
            item={'id':'prepare','reads':{'source':sha},'workbench':{'profile':'analysis'},
                  'payload':{'operation':'prepared_effect','inputs':inputs,'parameters':{'tolerance':1e-6}}}
            result=handler(item,{'attempt_key':'one'})
            self.assertTrue(result['summary']['has_effect_above_tolerance'])
            self.assertTrue(Path(result['prepared']['arrays.npz']['path']).exists())
            item['payload']['parameters']['tolerance']=2.
            item['payload']['stop_on_no_effect']=True
            self.assertEqual(handler(item,{'attempt_key':'no-op'})['status'],'no_progress')
            path.write_bytes(b'changed')
            result=handler(item,{'attempt_key':'two'})
            self.assertEqual(result['status'],'failed')


if __name__ == '__main__': unittest.main()
