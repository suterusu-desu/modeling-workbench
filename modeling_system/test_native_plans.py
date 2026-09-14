import json,tempfile,unittest
from pathlib import Path
from .service import ModelingService
from .store import canonical,digest
from .test_library_plan import fixture
from .bounded_reads import normalize_reads,json_chars

class NativePlansTests(unittest.TestCase):
    def test_named_plan_reads_and_legacy_descriptors_execute_bounded_readonly(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);s=ModelingService(p)
            mesh=dict(ids=[str(i) for i in range(4096)],co=[[float(i),0.,0.] for i in range(4096)],faces=[['0','1','2']],face_materials=[0],face_smooth=[False])
            case=dict(schema_version=1,object='Synthetic.Mesh',question='Preview one point',mesh=mesh,steps=[dict(id='move',op='translate',selection=['0'],delta=[0,0,1])])
            (p/'mesh.json').write_bytes(canonical(case));(p/'library.json').write_bytes(canonical(fixture()))
            results=[s.prepare_mesh_batch(str(p/'mesh.json')),s.prepare_library_import(str(p/'library.json'),[],'new_')]
            before={str(f):digest(f.read_bytes()) for f in p.rglob('*') if f.is_file()}
            for result in results:
                self.assertEqual(set(result['reads']),{'plan'})
                self.assertEqual(result['detail'],result['reads']['plan'])
                self.assertLess(json_chars(result),2500)
                descriptor=result['detail']
                original=canonical(descriptor)
                for shape in (result['reads'],descriptor):
                    named=normalize_reads(shape,default_name='plan')
                    self.assertEqual(named,{'plan':descriptor})
                    page=s.execute(named['plan']['operation'],named['plan']['arguments'])
                    self.assertEqual(page['source_revision'],result['plan'])
                    self.assertLessEqual(json_chars(page),8000)
                    named['plan']['arguments']['path'].append('caller-local-change')
                    self.assertEqual(canonical(descriptor),original)
            self.assertEqual(before,{str(f):digest(f.read_bytes()) for f in p.rglob('*') if f.is_file()})
            page=s.execute(results[0]['detail']['operation'],results[0]['detail']['arguments'])
            self.assertIn(b'deferred_value',canonical(page))

    def test_descriptor_normalizer_refuses_malformed_or_unbounded_shapes(self):
        for value in (None,[],{'operation':'read_record'},{'plan':'read_record'},
                      {'plan':{'operation':'read_record','arguments':[]}},
                      {'plan':{'operation':'read_record','arguments':{'value':'x'*17000}}}):
            with self.subTest(value_type=type(value).__name__),self.assertRaises(ValueError):normalize_reads(value)
        self.assertEqual(normalize_reads({}),{})

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
