import json,tempfile,unittest
from pathlib import Path
from .service import ModelingService
from .store import canonical,digest
from .test_library_plan import fixture

class NativePlansTests(unittest.TestCase):
    def test_public_preparation_and_readonly_partial_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);s=ModelingService(p)
            case=dict(schema_version=1,object='Synthetic.Mesh',question='Move one explicit point',mesh=dict(ids=['0','1','2'],co=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],faces=[['0','1','2']],face_materials=[0],face_smooth=[False]),steps=[dict(id='move',op='translate',selection=['0'],delta=[0,0,1])])
            (p/'mesh.json').write_bytes(canonical(case))
            result=s.prepare_mesh_batch(str(p/'mesh.json'));self.assertEqual(result['status'],'preflighted_offline');self.assertFalse(result['native_ready'])
            (p/'library.json').write_bytes(canonical(fixture()))
            self.assertEqual(s.prepare_library_import(str(p/'library.json'),[],'new_')['counts']['objects'],2)
            checkpoint=p/'partial.blend';checkpoint.write_bytes(b'synthetic checkpoint, not Blender')
            r=dict(schema_version=1,status='partial_or_uncertain',owner='synthetic-owner',applied_steps=[dict(id='first')],pending_step='second',partial=dict(path=str(checkpoint),sha256=digest(checkpoint.read_bytes())))
            path=p/'receipt.json';path.write_bytes(canonical(r));pin=digest(path.read_bytes())
            before={str(f):digest(f.read_bytes()) for f in p.rglob('*') if f.is_file()}
            result=s.execute('inspect_native_transaction',dict(receipt_path=str(path),expected_sha256=pin))
            self.assertFalse(result['replay_allowed']);self.assertEqual(result['pending_step'],'second');self.assertTrue(result['artifacts']['partial']['bytes_verified'])
            self.assertEqual(before,{str(f):digest(f.read_bytes()) for f in p.rglob('*') if f.is_file()})
            checkpoint.write_bytes(b'later checkpoint')
            self.assertFalse(s.inspect_native_transaction(str(path),pin)['artifacts']['partial']['bytes_verified'])
            with self.assertRaises(ValueError):s.inspect_native_transaction(str(path),'0'*64)

if __name__=='__main__':unittest.main()
