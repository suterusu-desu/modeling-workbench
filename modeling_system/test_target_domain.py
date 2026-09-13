"""Mask-filtered targets and fit-only edges must not appear complete."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .service import ModelingService
from .store import canonical,digest


class TargetDomainTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=ModelingService(workspace=self.root,store=self.root/'store')
        self.state=dict(source_state_id='fixture',pose={'bend':1},frame='local',units='unit',dependency_fingerprint='mesh-a')
        self.arrays=dict(point_ids=np.array([10,11,12,13]),candidate=np.array([1,1,1,0]),selected=np.array([1,1,0,0]),
            response_covered=np.array([1,1,0,0]),mask_values=np.array([1.,1.,0.,0.]),
            semantic_values=np.array([[0.,1.],[0.,2.],[np.nan,np.inf],[0.,4.]]),
            edges=np.array([[0,1],[1,2],[2,3]]),fit_checks=np.array([[0,1]]),transition_checks=np.empty((0,2),dtype=int))
        metadata=lambda meaning:dict(meaning=meaning,evidence=['exact synthetic fixture'])
        self.case=dict(schema_version=1,question='Did the authored mask hide candidate skin and join checks?',state=self.state,
            inventory=dict(point_identity='Synthetic native IDs',basis='Independent geometric screen',
                independent_of_support_mask=True,evidence=['synthetic known geometry'],
                coverage=dict(status='complete_for_declared_screen',missing=[])),
            selection=metadata('Currently qualified fit points'),response=metadata('Measured finite response coverage'),
            topology=metadata('Complete undirected triangle edge universe in declared patch'),
            semantics=dict(metadata('Historical numeric material labels, not anatomical qualification'),fields=['row','column']),
            support_mask=dict(metadata('Authored deformation influence mask'),threshold=.5),exclusions=[],
            checks=[dict(metadata('Fit-point guide gradient'),id='fit',edge_array='fit_checks',required_categories=['fit_fit']),
                    dict(metadata('Relative transition deformation'),id='transition',edge_array='transition_checks',required_categories=['fit_context'])])

    def run_case(self,expected=None):
        np.savez(self.root/'inventory.npz',**self.arrays)
        self.case['arrays']=dict(path='inventory.npz',sha256=digest((self.root/'inventory.npz').read_bytes()))
        (self.root/'case.json').write_bytes(canonical(self.case))
        result=self.s.inspect_target_domain(str(self.root/'case.json'),expected or self.state)
        return result,self.s.store.get(result['analysis'],'target_domain')

    def test_mask_disagreement_unknown_semantics_and_missing_transition_are_separate(self):
        r,p=self.run_case();s=r['summary']
        self.assertEqual(s['candidates_outside_authored_mask'],1)
        self.assertEqual(s['candidates_with_unknown_numeric_labels'],1)
        self.assertEqual(s['candidates_without_response'],1)
        self.assertEqual(s['unselected_candidates_without_exclusion_provenance'],1)
        self.assertEqual(s['fit_context_edges_checked_by_any_family'],0)
        self.assertEqual(p['checks'][1]['categories']['fit_context']['unchecked'],1)
        row=next(p for p in p['points'] if p['id']=='12')
        self.assertEqual(row['semantics'],{'row':None,'column':None})
        self.assertEqual(row['numeric_label_status'],'unknown')
        self.assertTrue(any('Review mask-excluded' in a for a in r['next_actions']))
        canonical(p);self.assertFalse(r['native_ready']);self.assertFalse(r['target_admission'])

    def test_new_check_does_not_grant_target_admission_or_replace_other_families(self):
        self.arrays['transition_checks']=np.array([[1,2]])
        r,p=self.run_case()
        self.assertEqual(r['summary']['fit_context_edges_checked_by_any_family'],1)
        self.assertEqual(p['checks'][0]['categories']['fit_context']['checked'],0)
        self.assertEqual(p['checks'][1]['categories']['fit_context']['checked'],1)
        self.assertEqual(r['summary']['selected_points'],2)
        self.assertFalse(r['target_admission'])

    def test_exclusion_provenance_is_retained_without_relabeling_mask_as_anatomy(self):
        self.case['exclusions']=[dict(points=['12'],interpretation='trial_restriction',reason='Old support mask',evidence=['fixture mask'])]
        r,p=self.run_case()
        self.assertEqual(r['summary']['unselected_candidates_without_exclusion_provenance'],0)
        self.assertEqual(r['summary']['candidates_outside_authored_mask'],1)
        self.assertEqual(p['points'][-1]['exclusion_records'],[0])

    def test_supported_region_does_not_invent_unknown_numeric_labels(self):
        self.case['correspondence_reviews']=[dict(points=['10','11','12'],status='supported',reason='Independent connected region evidence',evidence=['fixture construction'])]
        r,p=self.run_case()
        self.assertEqual(r['summary']['candidates_without_supported_correspondence'],0)
        self.assertEqual(r['summary']['candidates_with_unknown_numeric_labels'],1)
        row=next(x for x in p['points'] if x['id']=='12')
        self.assertEqual(row['correspondence_status'],'supported')
        self.assertEqual(row['numeric_label_status'],'unknown')
        self.assertFalse(any('Review mask-excluded' in a or 'Review candidate region' in a for a in r['next_actions']))

    def test_rejected_correspondence_is_not_automatically_reopened(self):
        self.case['correspondence_reviews']=[dict(points=['10','11','12'],status='rejected',reason='Wrong region',evidence=['fixture review'])]
        r,_=self.run_case()
        self.assertFalse(any('Review mask-excluded' in a or 'Review candidate region' in a for a in r['next_actions']))
        self.assertFalse(r['target_admission'])

    def test_mask_filtered_inventory_cannot_claim_independent_completeness(self):
        self.case['inventory']['independent_of_support_mask']=False
        with self.assertRaisesRegex(ValueError,'inherit support-mask'):self.run_case()
        self.case['inventory']['coverage']['status']='declared_subset'
        r,_=self.run_case();self.assertFalse(r['summary']['independent_geometric_screen'])

    def test_bad_edges_masks_and_state_are_refused(self):
        with self.assertRaisesRegex(ValueError,'state/pose'):self.run_case(dict(self.state,frame='other'))
        self.arrays['mask_values'][0]=np.nan
        with self.assertRaisesRegex(ValueError,'Mask values'):self.run_case()
        self.arrays['mask_values'][0]=1
        self.arrays['fit_checks']=np.array([[0,2]])
        with self.assertRaisesRegex(ValueError,'edge universe'):self.run_case()
        self.arrays['fit_checks']=np.array([[0,1],[1,0]])
        with self.assertRaisesRegex(ValueError,'duplicate'):self.run_case()

    def test_exact_arrays_are_retained_after_original_disappears(self):
        r,p=self.run_case();(self.root/'inventory.npz').unlink()
        retained=self.s.store.resolve_blob(p['case']['arrays']['asset'])
        with np.load(retained,allow_pickle=False) as z:self.assertTrue(np.isnan(z['semantic_values'][2,0]))
        self.assertEqual(self.s.read_record(r['analysis'])['points'],p['points'])


if __name__=='__main__':unittest.main()
