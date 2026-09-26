"""Regression cases for geometry truth, historical state and observation association."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .store import canonical,digest
from .workbench import Workbench
from . import geometry


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.a={'co':np.array([[0.,0,0],[2,0,0],[0,2,0]]),'tri':np.array([[0,1,2]],dtype=np.int32),
                'triangle_component':np.array([0])}

    def test_nearest_face_edge_and_vertex(self):
        for point,expected in [([.5,.5,3],[.5,.5,0]),([2,2,0],[1,1,0]),([-2,-2,0],[0,0,0])]:
            result=geometry.nearest_surface(self.a,point)
            np.testing.assert_allclose(result['point'],expected)

    def test_endpoint_displacement_does_not_claim_out_and_back_travel(self):
        middle=dict(self.a,co=self.a['co']+np.array([3.,4.,0.]))
        endpoint=geometry.compare(self.a,self.a)
        outward=geometry.compare(self.a,middle);returning=geometry.compare(middle,self.a)
        self.assertEqual(endpoint['maximum_distance'],0.)
        self.assertEqual(outward['maximum_distance']+returning['maximum_distance'],10.)
        self.assertIn('cumulative travel is not measured',endpoint['measurement'])

    def test_degenerate_triangle_and_no_supported_surface(self):
        a={'co':np.array([[0.,0,0],[0,0,0],[2,0,0]]),'tri':self.a['tri']}
        np.testing.assert_allclose(geometry.nearest_surface(a,[1,3,0])['point'],[1,0,0])
        self.assertEqual(geometry.nearest_surface(self.a,[0,0,0],{'component':7})['status'],'unsupported')
        with self.assertRaises(ValueError):geometry.nearest_surface(self.a,[0,0,0],{'attribute':'FACE__missing'})

    def test_fixed_plane_and_coplanar_limit(self):
        section=geometry.plane_sections(self.a,0,1)
        self.assertEqual(len(section['segments']),1)
        np.testing.assert_allclose(sorted(section['segments'][0]),[[1,0,0],[1,1,0]])
        result=geometry.plane_sections(self.a,2,0)
        self.assertEqual(result['coplanar_triangles_excluded'],1)
        self.assertEqual(result['segments'],[])

    def test_same_counts_do_not_establish_topology_correspondence(self):
        after=dict(self.a,tri=np.array([[0,2,1]],dtype=np.int32))
        with self.assertRaises(ValueError):geometry.compare(self.a,after)

    def test_diagonal_changes_require_explicit_matching_polygon_correspondence(self):
        before={'co':np.array([[0.,0,0],[1,0,0],[1,1,0],[0,1,0]]),
                'tri':np.array([[0,1,2],[0,2,3]],np.int32),'loops':np.array([0,1,2,3]),
                'polygon_starts':np.array([0]),'polygon_lengths':np.array([4])}
        after=dict(before,tri=np.array([[0,1,3],[1,2,3]],np.int32),co=before['co']+[0,0,1])
        with self.assertRaises(ValueError):geometry.compare(before,after)
        result=geometry.compare(before,after,'recorded_polygon_loops')
        self.assertEqual(result['changed_vertices'],4)
        self.assertEqual(result['maximum_distance'],1)
        self.assertFalse(result['triangle_layout_equal'])
        after['loops']=np.array([1,2,3,0])
        with self.assertRaises(ValueError):geometry.compare(before,after,'recorded_polygon_loops')


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.wb=Workbench(self.root/'store')

    def tearDown(self):self.temp.cleanup()

    def native(self,name,z=0):
        arrays={'co':np.array([[0.,0,z],[2,0,z],[0,2,z]]),'tri':np.array([[0,1,2]],np.int32)}
        path=self.root/(name+'.npz');np.savez_compressed(path,**arrays)
        ah=digest(canonical({k:digest(v.tobytes()) for k,v in arrays.items()}))
        state={'objects':[{'name':'Face','type':'MESH','arrays':str(path),'geometry_hash':ah,'source':{}}],
               'controls':{'blink':z},'selected_guide':'Neutral','references':{}}
        record={'state':state,'state_id':digest(canonical(state)),'coverage':[{'name':'Face','included':True}]}
        record_path=self.root/(name+'.json');record_path.write_text(json.dumps(record),encoding='utf-8')
        return record_path,record,arrays

    def test_snapshot_survives_source_replacement_and_current_change(self):
        path,record,a=self.native('baseline')
        before=self.wb.import_scene(path)['state']
        q=self.wb.open_question('Close the eye',before,'eye','Preserve identity')['question']
        path2,_,_=self.native('later',2)
        later=self.wb.import_scene(path2)['state']
        self.wb.open_question('Later pose',later,'eye','Preserve identity')
        np.savez_compressed(self.root/'baseline.npz',co=a['co']+100,tri=a['tri'])
        hit=self.wb.query_geometry(before,'Face','nearest',point=[.5,.5,1])
        np.testing.assert_allclose(hit['summary']['point'],[.5,.5,0])
        self.assertEqual(hit['mode'],'immutable snapshot')
        with self.assertRaises(RuntimeError):self.wb.store.set_current(q,expected=q)

    def test_source_and_stored_corruption_are_detected(self):
        path,record,a=self.native('source')
        state=self.wb.import_scene(path)['state']
        np.savez_compressed(self.root/'source.npz',co=a['co']+1,tri=a['tri'])
        with self.assertRaises(ValueError):self.wb.import_scene(path)
        asset=self.wb.store.get(state,'state')['objects'][0]['asset']
        (self.wb.store.root/asset['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):self.wb.arrays(state,'Face')

    def test_observation_scope_and_projection_are_not_assumed(self):
        path,record,_=self.native('source')
        state=self.wb.import_scene(path)['state']
        q=self.wb.open_question('Closed eye',state,'eye','Smooth skin')['question']
        image=self.root/'evidence.png';image.write_bytes(b'fixture: bytes used only for association tests')
        old=self.wb.attach_observation(q,image,{'matrix_world':np.eye(4).tolist()})
        self.assertEqual(old['status'],'partial')
        with self.assertRaises(ValueError):self.wb.project(old['observation'],[[0,0,0]])
        with self.assertRaises(ValueError):self.wb.attach_observation(old['question'],image,{'scene_state_id':'wrong'})
        metadata={'scene_state_id':record['state_id'],'evaluated_camera_world':np.eye(4).tolist(),
                  'view_projection_matrix':np.eye(4).tolist(),'resolution':[1200,1000],'evaluated_owning_scene':'Fixture'}
        calibrated=self.wb.attach_observation(old['question'],image,metadata)
        projected=self.wb.project(calibrated['observation'],[[0,0,0],[2,0,0]])['points']
        self.assertEqual(projected[0]['pixel'],[600,500]);self.assertFalse(projected[1]['inside_clip'])
        self.assertIn('not tested',projected[0]['visibility'])

    def test_camera_revision_does_not_duplicate_geometry(self):
        path,record,_=self.native('source');state=self.wb.import_scene(path)['state']
        q=self.wb.open_question('Question',state,'eyes','Intent')['question']
        image=self.root/'evidence.png';image.write_bytes(b'fixture')
        one=self.wb.attach_observation(q,image,{'yaw':0})
        two=self.wb.attach_observation(one['question'],image,{'yaw':35})
        a=self.wb.store.get(one['observation'],'observation');b=self.wb.store.get(two['observation'],'observation')
        self.assertNotEqual(one['observation'],two['observation'])
        self.assertEqual(a['geometry_revision'],b['geometry_revision'])
        self.assertEqual(a['image']['path'],b['image']['path'])

    def test_outcomes_preserve_evidence_and_separate_method_from_character(self):
        path,_,_=self.native('source');state=self.wb.import_scene(path)['state']
        q=self.wb.open_question('Closed eye',state,'eyes','Smooth skin')['question']
        evidence=self.wb.query_geometry(state,'Face','nearest',point=[.5,.5,1])['record']
        with self.assertRaises(ValueError):
            self.wb.record_outcome(q,{'status':'supported'},{'status':'supported'},[], 'This pose')
        self.assertEqual(self.wb.store.current(),q)
        result=self.wb.record_outcome(q,
            {'status':'rejected','finding':'Candidate pinches the corner'},
            {'status':'supported','finding':'Matched evidence reveals the regression'},
            [evidence],'This topology and closed-eye pose only')
        outcome=self.wb.store.get(result['outcome'],'outcome')
        self.assertEqual(outcome['evidence'],[evidence])
        self.assertEqual(outcome['character']['status'],'rejected')
        self.assertEqual(outcome['method']['status'],'supported')
        self.assertEqual(outcome['user_appearance_acceptance'],'not implied')
        self.assertEqual(self.wb.inspect_situation()['outcomes'],[result['outcome']])
        self.assertEqual(self.wb.inspect_situation(q)['outcomes'],[])
        # The first outcome revised the question: its entry identity is now stale and is
        # refused before anything is written, leaving no orphan outcome record.
        from .ledger import PreconditionRefusal
        before=sorted(key for key,_ in self.wb.store.records(('outcome',)))
        with self.assertRaises(PreconditionRefusal) as refused:
            self.wb.record_outcome(q,{'status':'unresolved'},{'status':'unresolved'},[evidence],'Second look')
        self.assertEqual(refused.exception.modeling_effect_status,'refused before mutation dispatch')
        self.assertEqual(refused.exception.modeling_details['current'],result['question'])
        self.assertEqual(sorted(key for key,_ in self.wb.store.records(('outcome',))),before)
        self.assertEqual(self.wb.store.current(),result['question'])
        second=self.wb.record_outcome(self.wb.inspect_situation()['question'],{'status':'unresolved'},{'status':'unresolved'},[evidence],'Second look')
        self.assertEqual(self.wb.inspect_situation()['outcomes'],[result['outcome'],second['outcome']])


if __name__=='__main__':unittest.main()
