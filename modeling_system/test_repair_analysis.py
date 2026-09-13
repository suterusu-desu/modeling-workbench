"""Behavioral checks of bounded XYZ diagnostics, using isolated retained stores."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import anyio
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .service import ModelingService


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelingService(store=self.root/'store')
        # One control moves a fit point in X AND a neighboring layer in Y.
        # Another control can independently preserve that neighbor.
        matrix = sparse.csr_matrix([[1., 0.], [0., 0.], [0., 0.],
                                    [0., 0.], [2., 1.], [0., 0.]])
        np.savez(self.root/'response.npz', data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                 shape=matrix.shape, baseline_xyz=np.zeros((2, 3)), lower_delta=[-2., -3.], upper_delta=[2., 3.])
        np.savez(self.root/'native.npz', co=np.zeros((2, 3)))
        def ref(name):
            p=self.root/name
            return dict(path=name, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        domain=dict(source_state_id='fixture-state', pose={'blink':1.}, frame='world', units='scene units', dependency_fingerprint='fixture-dependencies')
        support=dict(status='qualified', correspondence='qualified', visibility='visible',
                     registration=dict(status='qualified', error_bound=0.), evidence=['inert qualified fixture'])
        self.case=dict(schema_version=1, question='Fit the point while preserving its coupled neighbor', state=domain,
            outputs=[dict(object='face', native_vertex=0, layer='skin'),dict(object='face',native_vertex=1,layer='skin')],
            controls=[dict(id='shape-X',meaning='fixture shape displacement'),dict(id='neighbor-Y',meaning='independent preservation control')],
            response=dict(domain,kind='exact_linear',validity_status='validated',validation_evidence=['inert exact fixture'],arrays=ref('response.npz')),
            native_baselines=[dict(object='face',frame='world',source_state_id='fixture-state',file=ref('native.npz'),xyz_key='co')],
            required_layers=['skin'], layers=[dict(name='skin',complete_for_controls=['shape-X','neighbor-Y'],missing=[],evidence=['fixture complete influence'])],
            targets=[dict(id='fit-X',region='target',terms=[dict(output=0,direction=[1,0,0])],lower=.999,upper=1.001,scale=.001,support=support)],
            preservation=[dict(id='preserve-Y',region='neighbor',terms=[dict(output=1,direction=[0,1,0])],lower=-.001,upper=.001,scale=.001,reason='preserve neighbor',evidence=['fixture preservation'])])

    def run_case(self, delta=None):
        path=self.root/'case.json'; path.write_text(json.dumps(self.case))
        result=self.service.analyze_repair(str(path),delta)
        full=self.service.store.get(result['analysis'],'repair_analysis')
        return result,full

    def test_coupled_candidate_preserves_neighbor_and_retains_arrays(self):
        result,full=self.run_case()
        self.assertEqual(result['disposition'],'candidate')
        self.assertFalse(result['native_ready'])
        with np.load(self.service.store.resolve_blob(full['prediction'])) as data:
            self.assertAlmostEqual(data['predicted_xyz'][0,0],1.,places=5)
            self.assertLess(abs(data['predicted_xyz'][1,1]),.00101)
            self.assertAlmostEqual(data['control_delta'][1],-2.,places=2)
        self.assertEqual(full['layer_effects'][0]['outputs'],2)

    def test_missing_target_and_acquisition_do_not_solve_or_dispatch(self):
        target=self.case['targets'][0]; target['support']['status']='missing'; target.pop('lower'); target.pop('upper')
        self.case['acquisition']=dict(views=[dict(subjects=['fit-X'],observation='recorded-view',visibility='visible',new_capture_required=False)],
            identity_evidence=['identity'],stationary_context=['whole face'],new_form_question='What form belongs at this missing surface?',video_phase='closed')
        result,full=self.run_case()
        self.assertEqual(result['disposition'],'needs_evidence_or_revision')
        nomination=next(n for n in full['acquisition_nominations'] if n['cause']=='missing_target')
        self.assertEqual(nomination['generation'],'conditional_on_reviewed_capture_and_existing_authorization')
        self.assertFalse(nomination['dispatch_authorized'])
        self.case['acquisition']['existing_requests']=['claimed job; do not repeat']
        _,full=self.run_case()
        self.assertTrue(all(n['generation']=='not_justified_by_this_diagnostic' for n in full['acquisition_nominations']))

    def test_preservation_conflict_is_not_insufficient_fit_controls(self):
        self.case['preservation'][0]['terms']=[dict(output=0,direction=[1,0,0])]
        result,full=self.run_case()
        codes={d['code'] for d in full['diagnostics']}
        self.assertIn('conflicting_constraints',codes)
        self.assertNotIn('insufficient_controls',codes)
        self.assertEqual(full['numerical']['status'],'diagnostic_trial_only')
        self.assertTrue(full['constraints'][0]['outputs'])
        self.assertTrue(any(n['surface'] and n['region'] for n in full['acquisition_nominations']))

    def test_optimizer_stop_is_not_geometric_infeasibility_or_generation_need(self):
        # An unsuccessful numerical termination must not be relabeled as a
        # missing surface or insufficient controls, even with a returned point.
        with patch('modeling_system.repair_analysis.linprog',return_value=SimpleNamespace(
                status=4,message='Synthetic numerical stop',x=np.zeros(3))):
            result,full=self.run_case()
        codes={d['code'] for d in full['diagnostics']}
        self.assertIn('solver_unresolved',codes)
        self.assertTrue(codes.isdisjoint({'missing_target','insufficient_controls','conflicting_constraints'}))
        self.assertFalse(result['native_ready'])
        self.assertTrue(all(n['generation']=='not_justified_by_this_diagnostic' for n in full['acquisition_nominations']))

    def test_grouped_nomination_keeps_exact_rows_and_thin_visible_view(self):
        target=self.case['targets'][0]; target['support']['status']='missing'
        other=copy.deepcopy(target); other.update(id='fit-Y',terms=[dict(output=0,direction=[0,1,0])])
        self.case['targets'].append(other)
        self.case['acquisition']=dict(views=[dict(subjects=['fit-Y'],observation='visible-view',
            role='oblique',pixel=[31,42],visibility='visible',new_capture_required=False,metadata='x'*20000)],
            native_motion_baselines='native samples',video_phase=dict(status='missing'))
        result,full=self.run_case()
        nomination=full['acquisition_nominations'][0]
        self.assertEqual(nomination['related_subjects'],['fit-X','fit-Y'])
        self.assertEqual(nomination['diagnostic_indices'],[0,1])
        self.assertEqual(nomination['native_motion_baselines'],'native samples')
        self.assertEqual(nomination['video_phase'],dict(status='missing'))
        self.assertEqual(result['nominations'][0]['preferred_view']['pixel'],[31,42])
        self.assertNotIn('metadata',result['nominations'][0]['preferred_view'])
        self.assertEqual(full['nomination_coverage']['represented_diagnostics'],2)
        self.assertEqual(full['nomination_coverage']['deferred'],0)

    def test_grouping_never_merges_different_native_points(self):
        from .repair_analysis import _nominations
        issues=[dict(code='missing_target',subject=str(i),region='eye',outputs=[dict(object='face',native_vertex=i)],
            reason='missing',next_action='inspect') for i in range(14)]
        issues.append(dict(code='response_outside_validity',subject='range',reason='unknown',next_action='qualify'))
        nominations,coverage=_nominations(self.case,issues)
        self.assertEqual(len(nominations),12)
        self.assertEqual(nominations[1]['cause'],'response_outside_validity')
        self.assertEqual(coverage['total_groups'],15)
        self.assertEqual(coverage['deferred'],3)

    def test_conflicting_targets_are_not_mislabeled_as_control_failure(self):
        other=copy.deepcopy(self.case['targets'][0]); other.update(id='other-source',lower=-1.001,upper=-.999)
        self.case['targets'].append(other)
        _,full=self.run_case()
        codes={d['code'] for d in full['diagnostics']}
        self.assertIn('contradictory_targets',codes)
        self.assertNotIn('insufficient_controls',codes)

    def test_one_sided_clearance_is_not_an_unknown_bound(self):
        self.case['preservation'][0].update(lower=.2,upper=None,upper_unbounded=True)
        result,full=self.run_case()
        self.assertEqual(result['disposition'],'candidate')
        row=next(r for r in full['constraints'] if r['id']=='preserve-Y')
        self.assertGreaterEqual(row['value'],.2-1e-10)
        self.assertIsNone(row['upper'])

    def test_unreachable_target_names_control_failure(self):
        self.case['targets'][0]['terms']=[dict(output=0,direction=[0,0,1])]
        _,full=self.run_case()
        self.assertIn('insufficient_controls',{d['code'] for d in full['diagnostics']})

    def test_complete_face_does_not_hide_missing_dependent_layer(self):
        self.case['required_layers'].append('lash')
        result,full=self.run_case()
        self.assertEqual(full['numerical']['status'],'candidate')
        self.assertEqual(result['disposition'],'needs_evidence_or_revision')
        self.assertTrue(any(d['subject']=='lash' for d in full['diagnostics']))

    def test_validity_ranges_and_baseline_are_separate(self):
        _,full=self.run_case([3.,0.])
        self.assertIsNone(full['prediction'])
        self.assertIn('response_outside_validity',{d['code'] for d in full['diagnostics']})
        self.case['response']['pose']={'blink':.8}
        _,full=self.run_case([0.,0.])
        self.assertIsNone(full['prediction'])

    def test_registration_uncertainty_is_not_a_control_failure(self):
        self.case['targets'][0]['support']['registration']['error_bound']=.01
        _,full=self.run_case()
        self.assertIn('registration_uncertain',{d['code'] for d in full['diagnostics']})
        self.assertNotIn('insufficient_controls',{d['code'] for d in full['diagnostics']})

    def test_nonconvergence_stays_unresolved(self):
        with patch('modeling_system.repair_analysis.linprog',return_value=SimpleNamespace(status=1,x=None,message='time limit')):
            result,full=self.run_case()
        self.assertIn('solver_unresolved',{d['code'] for d in full['diagnostics']})
        self.assertNotIn('insufficient_controls',{d['code'] for d in full['diagnostics']})
        self.assertIsNone(full['prediction'])

    def test_unknown_range_remains_unknown(self):
        path=self.root/'response.npz'
        with np.load(path) as data:
            values={k:data[k] for k in data.files if k not in ('lower_delta','upper_delta')}
        np.savez(path,**values)
        self.case['response']['arrays']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        result,full=self.run_case([0.,0.])
        self.assertIsNone(full['prediction'])
        self.assertTrue(any(d['subject']=='control_ranges' for d in full['diagnostics']))

    def test_changed_input_refused(self):
        (self.root/'native.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Repair input changed'):
            self.run_case()

    def test_public_mcp_and_exact_bounded_expansions(self):
        from .mcp_server import ModelingMCP
        path=self.root/'case.json'; path.write_text(json.dumps(self.case))
        async def run():
            server=ModelingMCP(self.service)
            schemas={t.name:t.input_schema for t in await server.list_tools()}
            self.assertIn('analyze_repair',schemas)
            result=await server.call_tool('analyze_repair',dict(case_path=str(path)))
            value=result.structured_content
            self.assertEqual(value['disposition'],'candidate')
            self.assertEqual(value,json.loads(next(c.text for c in result.content if c.type=='text')))
            for descriptor in value['reads'].values():
                expanded=await server.call_tool(descriptor['operation'],descriptor['arguments'])
                self.assertLessEqual(len(json.dumps(expanded.structured_content,ensure_ascii=False,separators=(',',':'))),8000)
            full=self.service.store.get(value['analysis'],'repair_analysis')
            self.assertEqual(full['case']['response']['arrays']['asset']['sha256'],self.case['response']['arrays']['sha256'])
        anyio.run(run)


if __name__=='__main__':
    unittest.main()
