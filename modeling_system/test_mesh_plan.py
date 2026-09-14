import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .mesh_plan import compile_plan,mesh_revision,f32
from .native_mesh_batch import SaveBoundary,execute_batch,file_ref,rollback_batch

class MeshPlanTests(unittest.TestCase):
    def setUp(self):
        self.mesh=dict(ids=['0','1','2','3','4','5'],co=[[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],[2.,0.,0.],[2.,1.,0.]],
            faces=[['0','1','2','3'],['1','4','5','2']],face_materials=[0,1],face_smooth=[False,True])
    def test_split_merge_chain_preserves_actual_face_ownership_and_aliases(self):
        plan=compile_plan(self.mesh,[dict(id='split',op='split_vertex',selection=['1'],faces=[1],new_id='6'),
            dict(id='move',op='translate',selection='rim',delta=[0,.25,0]),
            dict(id='merge',op='merge_vertices',selection=['1','6'],survivor='1')],{'rim':['1']})
        self.assertEqual(plan['steps'][0]['aliases']['rim'],['1','6'])
        self.assertEqual(plan['after']['faces'],self.mesh['faces'])
        self.assertEqual(plan['after']['face_materials'],[0,1]);self.assertEqual(plan['after']['face_smooth'],[False,True])
        self.assertEqual(plan['aliases']['rim'],['1'])
        self.assertEqual(next(r for r in plan['steps'][0]['relations'] if r['sources']==['1'])['kind'],'split')
        self.assertEqual(next(r for r in plan['steps'][2]['relations'] if r['targets']==['1'])['kind'],'merged')
    def test_partial_alias_merge_refuses_entire_plan_before_effects(self):
        before=copy.deepcopy(self.mesh)
        with self.assertRaisesRegex(ValueError,'partially'):compile_plan(self.mesh,[dict(id='merge',op='merge_vertices',selection=['1','2'],survivor='1')],{'partial':['1']})
        self.assertEqual(self.mesh,before)
        for policy,expected in [('any',['1']),('all',[])]:
            p=compile_plan(self.mesh,[dict(id='merge',op='merge_vertices',selection=['1','2'],survivor='1',alias_policy=policy)],{'partial':['1']})
            self.assertEqual(p['aliases']['partial'],expected)

    def test_initial_checkpoint_handler_failure_has_durable_partial_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);native=['a'*64]
            bpy=SimpleNamespace(app=SimpleNamespace(background=True,driver_namespace={},handlers=SimpleNamespace(save_post=[])),data=SimpleNamespace(filepath='',is_dirty=False,objects={'mesh':object()}))
            b=SaveBoundary(bpy,'owner',lambda:native[0]);expected=b.stamp()
            def save(fake,path):
                Path(path).write_bytes(b'checkpoint');native[0]='b'*64;b.epoch+=1
                return file_ref(path)
            with patch('modeling_system.native_mesh_batch.read_mesh',return_value=self.mesh),patch('modeling_system.native_mesh_batch.save_copy',side_effect=save):
                with self.assertRaisesRegex(RuntimeError,'Checkpoint saving'):execute_batch(b,'mesh',[dict(id='move',op='translate',selection=['1'],delta=[0,.1,0])],p/'out',expected)
            receipt=json.loads((p/'out/receipt.json').read_bytes())
            self.assertEqual(receipt['status'],'partial_or_uncertain');self.assertEqual(receipt['pending_stage'],'rollback_checkpoint')
            self.assertEqual(receipt['applied_steps'],[]);self.assertIn('rollback',receipt);self.assertIn('partial',receipt)

    def test_failed_rollback_reopen_retains_separate_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);rollback=p/'rollback.blend';rollback.write_bytes(b'original')
            bpy=SimpleNamespace(app=SimpleNamespace(background=True,driver_namespace={},handlers=SimpleNamespace(save_post=[])),data=SimpleNamespace(filepath='',is_dirty=False),ops=SimpleNamespace(wm=SimpleNamespace(open_mainfile=lambda **kw: (_ for _ in ()).throw(RuntimeError('open failed')))))
            b=SaveBoundary(bpy,'owner',lambda:'a'*64);stamp=b.stamp()
            path=p/'receipt.json';path.write_text(json.dumps(dict(schema_version=1,status='batch_applied',owner='owner',before=stamp,after=stamp,rollback=file_ref(rollback))))
            original=path.read_bytes()
            def save(fake,path):Path(path).write_bytes(b'abandoned');return file_ref(path)
            with patch('modeling_system.native_mesh_batch.save_copy',side_effect=save):
                with self.assertRaisesRegex(RuntimeError,'open failed'):rollback_batch(b,path)
            self.assertEqual(path.read_bytes(),original)
            attempt=json.loads((p/'rollback-result.json').read_bytes());self.assertEqual(attempt['status'],'rollback_uncertain');self.assertFalse(attempt['replay_allowed'])
    def test_deletion_reports_removed_faces_and_no_fabricated_descendant(self):
        p=compile_plan(self.mesh,[dict(id='delete',op='delete_vertices',selection=['0'])])
        self.assertEqual(p['steps'][0]['removed_face_indices'],[0]);self.assertEqual(p['after']['face_materials'],[1])
        self.assertEqual(next(r for r in p['steps'][0]['relations'] if r['sources']==['0']),dict(kind='deleted',sources=['0'],targets=[]))
    def test_float32_precision_and_bad_future_step_are_explicit(self):
        p=compile_plan(self.mesh,[dict(id='move',op='translate',selection=['1'],delta=[2**-23,0,0])])
        self.assertEqual(p['after']['co'][1][0],f32(1+2**-23));self.assertNotEqual(mesh_revision(self.mesh),p['after_revision'])
        for step in [dict(id='x',op='arbitrary'),dict(id='x',op='translate',selection=['99'],delta=[0,0,0]),dict(id='x',op='split_vertex',selection=['1'],faces=[0,1],new_id='6')]:
            with self.assertRaises(ValueError):compile_plan(self.mesh,[dict(id='first',op='translate',selection=['0'],delta=[1,0,0]),step])
    def test_explicit_face_ids_survive_and_deleted_faces_are_retained_as_losses(self):
        self.mesh['face_ids']=['20','21']
        p=compile_plan(self.mesh,[dict(id='delete',op='delete_vertices',selection=['0'])])
        self.assertEqual(p['after']['face_ids'],['21']);self.assertEqual(p['steps'][0]['before']['face_ids'],['20','21'])

    def test_integer_identity_roundtrip_is_checked_before_native_work(self):
        invalid=copy.deepcopy(self.mesh);invalid['ids'][0]='00';invalid['faces'][0][0]='00'
        with self.assertRaises(ValueError):mesh_revision(invalid)
        with self.assertRaises(ValueError):compile_plan(self.mesh,[dict(id='split',op='split_vertex',selection=['1'],faces=[1],new_id='01')])

    def test_candidate_save_handler_change_cannot_be_success_or_automatic_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input.blend';source.write_bytes(b'source');current=[copy.deepcopy(self.mesh)]
            bpy=SimpleNamespace(app=SimpleNamespace(background=True,driver_namespace={},handlers=SimpleNamespace(save_post=[])),
                data=SimpleNamespace(filepath=str(source),is_dirty=False,objects={'fixture':object()}),context=SimpleNamespace(view_layer=SimpleNamespace(update=lambda:None)))
            b=SaveBoundary(bpy,'owner',lambda:mesh_revision(current[0]));self.addCleanup(b.close)
            def save(bpy,path):
                path.write_text(json.dumps(current[0]));b.handler()
                if path.name=='candidate.blend':current[0]['co'][0][0]=f32(current[0]['co'][0][0]+1.)
                return file_ref(path)
            def write(ob,value,attribute,*unused):current[0]=copy.deepcopy(value)
            with patch('modeling_system.native_mesh_batch.read_mesh',side_effect=lambda *a:copy.deepcopy(current[0])),patch('modeling_system.native_mesh_batch.write_mesh',side_effect=write),patch('modeling_system.native_mesh_batch.save_copy',side_effect=save):
                with self.assertRaisesRegex(RuntimeError,'Candidate save changed'):
                    execute_batch(b,'fixture',[dict(id='move',op='translate',selection=['1'],delta=[0,.25,0])],root/'run',b.stamp())
            r=json.loads((root/'run/receipt.json').read_bytes());self.assertEqual(r['status'],'partial_or_uncertain');self.assertFalse(r['automatic_rollback_allowed'])
            self.assertEqual(len(r['applied_steps']),1);self.assertTrue((root/'run/candidate.blend').exists())
            with self.assertRaises(ValueError):rollback_batch(b,root/'run/receipt.json')

    def test_save_boundary_detects_same_content_save_and_one_ulp_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.blend';path.write_bytes(b'synthetic saved bytes')
            native=[mesh_revision(self.mesh)]
            bpy=SimpleNamespace(app=SimpleNamespace(background=True,driver_namespace={},handlers=SimpleNamespace(save_post=[])),data=SimpleNamespace(filepath=str(path),is_dirty=False))
            b=SaveBoundary(bpy,'fixture-owner',lambda:native[0]);self.addCleanup(b.close);before=b.stamp();b.require(before)
            b.handler()
            with self.assertRaisesRegex(ValueError,'later'):b.require(before)
            before=b.stamp();native[0]=compile_plan(self.mesh,[dict(id='ulp',op='translate',selection=['1'],delta=[2**-23,0,0])])['after_revision']
            with self.assertRaises(ValueError):b.require(before)
            before=b.stamp();path.write_bytes(b'later user save')
            with self.assertRaises(ValueError):b.require(before)
            with self.assertRaisesRegex(ValueError,'already owns'):SaveBoundary(bpy,'other',lambda:native[0])
            bpy.app.driver_namespace.clear()
            with self.assertRaisesRegex(ValueError,'session changed'):b.stamp()

if __name__=='__main__':unittest.main()
