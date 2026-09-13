"""Counterexamples to continuity inferred from selection, normals or proximity."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .service import ModelingService
from .store import canonical,digest


class SurfaceCorrespondenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=ModelingService(self.root,store=self.root/'store')
        self.state=dict(source_state_id='fixture',pose={'bend':1},frame='world',units='unit',dependency_fingerprint='exact')
        target=dict(self.state,source_state_id='target',target_id='surface')
        self.edges=np.array([[1,2],[2,3]])
        meta=lambda meaning:dict(meaning=meaning,evidence=['synthetic geometry'])
        coverage=dict(status='complete_for_declared_query',missing=[])
        self.case=dict(schema_version=1,question='Does selection hide a branch gap?',state=self.state,
            source_topology=dict(meta('Native polygon edges'),kind='native_polygon_edges'),roles=meta('Owner-authored region role'),
            selection=meta('Declared retained target subset'),orientation=meta('World winding normals, not anatomy'),
            section_graphs=[dict(meta('Exact fixed plane cut'),id='cut',context=dict(target,plane=dict(axis='X',value=0,tolerance=1e-10)),coverage=coverage,
                segments=[dict(triangle=10,nodes=['e:1:2','e:2:3'],selected=True),
                          dict(triangle=11,nodes=['e:2:3','e:3:4'],selected=True),
                          dict(triangle=20,nodes=['e:8:9','e:9:10'],selected=True)])],
            paths=[dict(meta('Authored source order'),id='path',kind='material_edges',coverage=coverage,
                target_context=dict(target,query='all forward hits in supplied geometry'),
                all_alternatives_retained=True,section_graph='cut',stations=[
                    dict(source_vertex=1,point=[0,0,0],alternatives=[self.hit(99,False,0),self.hit(10,True,1)]),
                    dict(source_vertex=2,point=[0,0,1],alternatives=[]),
                    dict(source_vertex=3,point=[0,0,2],alternatives=[self.hit(20,True,0)])])])

    @staticmethod
    def hit(triangle,selected,rank):
        return dict(triangle=triangle,selected=selected,full_order=rank,point=[0,rank,0],barycentric=[.2,.3,.5],normal=[0,1,0])

    def run_case(self,expected=None):
        np.savez(self.root/'edges.npz',source_edges=self.edges)
        self.case['arrays']=dict(path='edges.npz',sha256=digest((self.root/'edges.npz').read_bytes()))
        (self.root/'case.json').write_bytes(canonical(self.case))
        result=self.s.inspect_surface_correspondence(str(self.root/'case.json'),expected or self.state)
        return result,self.s.store.get(result['analysis'],'surface_correspondence')

    def test_gap_is_not_bridged_and_foreground_is_retained(self):
        r,p=self.run_case()
        self.assertEqual(r['summary']['transition_relations'],{'selection_gap':2})
        self.assertEqual(r['summary']['selected_first_behind_full_first'],1)
        self.assertEqual(len(p['stations'][0]['alternatives']),2)
        self.assertEqual(p['stations'][1]['alternatives'],[])
        self.assertFalse(r['target_admission']);self.assertFalse(r['native_ready'])
        self.assertTrue(all(not t['continuous_correspondence_proven'] for t in p['transitions']))

    def test_exact_nodes_define_components_not_supplied_labels_or_normals(self):
        s=self.case['paths'][0]['stations'];s[1]['alternatives']=[self.hit(11,True,0)]
        for row in s:
            for hit in row['alternatives']:hit['selected_section_component']=123;hit['normal']=[0,-1,0]
        r,p=self.run_case()
        self.assertEqual([t['section_relation'] for t in p['transitions']],
            ['same_selected_section_component','different_selected_section_components'])
        self.assertTrue(any('not global' in x for x in p['limits']))
        self.assertNotEqual(p['stations'][1]['selected_components'],[123])

    def test_source_edge_gap_is_separate_from_target_relation(self):
        self.edges=np.array([[1,3],[2,3]])
        self.case['paths'][0]['stations'][1]['alternatives']=[self.hit(11,True,0)]
        r,p=self.run_case();self.assertEqual(r['summary']['missing_source_edges'],1)
        self.assertFalse(p['transitions'][0]['source_edge_connected'])
        self.assertEqual(p['transitions'][0]['section_relation'],'same_selected_section_component')

    def test_missing_graph_and_incomplete_query_stay_unknown(self):
        p=self.case['paths'][0];p.pop('section_graph')
        p['coverage']=dict(status='declared_subset',missing=['ray range beyond recorded geometry'])
        r,_=self.run_case();self.assertEqual(r['summary']['transition_relations'],{'unknown_selection_coverage':2})
        p['stations'][1]['alternatives']=[self.hit(11,True,0)]
        p['coverage']=dict(status='complete_for_declared_query',missing=[])
        r,_=self.run_case();self.assertEqual(r['summary']['transition_relations'],{'unknown_branch_evidence':2})

    def test_multiple_branches_and_coincident_triangle_owners_are_retained(self):
        self.case['paths'][0]['stations'][1]['alternatives']=[self.hit(11,True,0),self.hit(20,True,1)]
        r,p=self.run_case();self.assertEqual(r['summary']['transition_relations'],{'ambiguous_section_branches':2})
        self.assertEqual(len(p['stations'][1]['alternatives']),2)
        self.assertEqual(len(p['stations'][1]['selected_components']),2)

    def test_spatial_probes_do_not_claim_native_material_connectivity(self):
        self.case['paths'][0]['kind']='spatial_probes'
        _,p=self.run_case();self.assertTrue(all(t['source_edge_connected'] is None for t in p['transitions']))

    def test_bad_state_order_selection_and_geometry_are_refused(self):
        with self.assertRaisesRegex(ValueError,'state/pose'):self.run_case(dict(self.state,frame='other'))
        hit=self.case['paths'][0]['stations'][0]['alternatives'][1]
        hit['full_order']=8
        with self.assertRaisesRegex(ValueError,'order'):self.run_case()
        hit['full_order']=1;hit['selected']=False
        with self.assertRaisesRegex(ValueError,'selection disagrees'):self.run_case()
        hit['selected']=True;hit['barycentric']=[1,1,1]
        with self.assertRaisesRegex(ValueError,'barycentric'):self.run_case()

    def test_graph_from_another_pose_is_not_reused(self):
        self.case['section_graphs'][0]['context']['pose']={'bend':0}
        with self.assertRaisesRegex(ValueError,'target state/pose'):self.run_case()

    def test_graph_from_another_plane_is_not_reused(self):
        self.case['paths'][0]['stations'][0]['point'][0]=.1
        with self.assertRaisesRegex(ValueError,'section plane'):self.run_case()

    def test_pinned_inputs_and_bounded_reads_survive_original_changes(self):
        r,p=self.run_case();(self.root/'edges.npz').write_bytes(b'changed')
        asset=p['case']['arrays']['asset'];self.assertNotEqual(self.s.store.resolve_blob(asset).read_bytes(),b'changed')
        descriptor=r['reads']['transitions'];out=self.s.execute(descriptor['operation'],descriptor['arguments'])
        self.assertIn('selection_gap',json.dumps(out))


if __name__=='__main__':unittest.main()
