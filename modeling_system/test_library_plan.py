import copy,unittest
from .library_plan import preflight

def fixture():
    kinds=[('Object','A'),('Object','B'),('Mesh','Shared'),('Material','Material')]
    return dict(schema_version=1,coverage='complete_declared_roots',roots=['Object:A','Object:B'],
        ids=[dict(id=k+':'+n,type=k,name=n,content_sha256='a'*64,counts=dict(vertices=4,edges=4,faces=1,loops=4) if k=='Mesh' else {}) for k,n in kinds],
        dependencies=[['Object:A','Mesh:Shared'],['Object:B','Mesh:Shared'],['Object:B','Object:A'],['Mesh:Shared','Material:Material']])

class LibraryPlanTests(unittest.TestCase):
    def test_shared_closure_counts_once_and_names_every_id(self):
        m=fixture();p=preflight(m,[('Object','A'),('Mesh','Shared')],'new_')
        self.assertEqual(p['counts']['vertices'],4);self.assertEqual(len(p['renames']),4);self.assertFalse(p['native_ready'])
        self.assertEqual(p['renames']['Object:B']['name'],'new_B')
    def test_no_silent_reuse_or_namespace_collision(self):
        with self.assertRaisesRegex(ValueError,'collision'):preflight(fixture(),[('Mesh','new_Shared')],'new_')
        m=fixture();m['ids'].append(dict(id='ShaderNodeTree:Tree',type='ShaderNodeTree',name='Tree',content_sha256='b'*64))
        m['dependencies'].append(['Material:Material','ShaderNodeTree:Tree'])
        with self.assertRaisesRegex(ValueError,'collision'):preflight(m,[('GeometryNodeTree','new_Tree')],'new_')
    def test_budget_rejects_closure_aggregate(self):
        with self.assertRaisesRegex(ValueError,'budget'):preflight(fixture(),[],'new_',{'vertices':3})
        with self.assertRaisesRegex(ValueError,'tighten'):preflight(fixture(),[],'new_',{'ids':9999})
    def test_missing_external_unreachable_and_duplicate_edges(self):
        for change in ('missing','unreachable','duplicate'):
            m=fixture()
            if change=='missing':m['dependencies'].append(['Object:A','Mesh:Absent'])
            if change=='unreachable':m['roots']=['Object:A']
            if change=='duplicate':m['dependencies'].append(m['dependencies'][0])
            with self.assertRaises(ValueError):preflight(m,[],'new_')
    def test_cycles_are_real_dependencies_not_an_import_ban(self):
        m=fixture();m['dependencies'].append(['Object:A','Object:B']);self.assertEqual(preflight(m,[],'n_')['counts']['objects'],2)
    def test_invalid_hash_names_and_types_refuse(self):
        for value in ('g'*64,'0'*63):
            m=fixture();m['ids'][0]['content_sha256']=value
            with self.assertRaises(ValueError):preflight(m,[],'n_')
        with self.assertRaises(ValueError):preflight(fixture(),[],'')
        m=fixture();m['ids'][0]['type']='Scene'
        with self.assertRaises(ValueError):preflight(m,[],'n_')

if __name__=='__main__':unittest.main()
