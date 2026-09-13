"""Exact splits, merges and introduced ancestry survive remapping and composition."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from .service import ModelingService
from .store import canonical,digest
from . import topology_lineage as lineage


class TopologyLineageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=ModelingService(self.root,store=self.root/'store');self.index=0

    def domain(self,state,ids):
        return dict(state=state,object='fixture',domain='vertex',topology_hash=digest(canonical(ids)),scope='Complete declared synthetic patch',ids=ids)

    def record(self,before,after,relations,edit_receipt=None):
        self.index+=1;folder=self.root/str(self.index);folder.mkdir()
        table=dict(schema_version=1,operation_id='op-'+str(self.index),before=before,after=after,relations=relations)
        data=canonical(table);(folder/'lineage.json').write_bytes(data)
        receipt=dict(operation_id=table['operation_id'],lineage_sha256=digest(data),status='topology_change_recorded',
            before_revision=digest(canonical(before)),after_revision=digest(canonical(after)))
        if edit_receipt:edit_receipt(receipt)
        (folder/'receipt.json').write_bytes(canonical(receipt))
        ref=lambda name:dict(path=name,sha256=digest((folder/name).read_bytes()))
        (folder/'case.json').write_bytes(canonical(dict(schema_version=1,lineage=ref('lineage.json'),operation_receipt=ref('receipt.json'))))
        return self.s.record_topology_lineage(str(folder/'case.json'))

    def fixture(self):
        before=self.domain('a',['a','b','c','d']);after=self.domain('b',['a1','a2','bc','n'])
        rows=[dict(kind='split',sources=['a'],targets=['a1','a2']),
            dict(kind='merged',sources=['b','c'],targets=['bc'],weights={'bc':{'b':.25,'c':.75}},weight_semantics='Operation-produced normalized blend'),
            dict(kind='deleted',sources=['d'],targets=[]),dict(kind='created',sources=[],targets=['n'])]
        return before,after,rows

    def test_partial_merge_is_ambiguous_and_split_is_exact(self):
        before,after,rows=self.fixture();r=self.record(before,after,rows)
        mapped=lineage.remap(self.s,r['lineage'],['a','b'],r['before_revision'])
        p=self.s.store.get(mapped['selection_record'],'topology_selection')
        self.assertEqual(mapped['status'],'ambiguous');self.assertIsNone(p['selection'])
        self.assertEqual(p['candidates'],['a1','a2','bc'])
        for policy,wanted in [('any',['a1','a2','bc']),('all',['a1','a2'])]:
            result=lineage.remap(self.s,r['lineage'],['a','b'],r['before_revision'],policy=policy)
            self.assertEqual(self.s.store.get(result['selection_record'])['selection'],wanted)
        self.assertFalse(r['native_ready'])

    def test_deletion_and_created_reverse_have_no_fabricated_counterpart(self):
        r=self.record(*self.fixture())
        d=lineage.remap(self.s,r['lineage'],['d'],r['before_revision'])
        self.assertEqual(self.s.store.get(d['selection_record'])['without_counterpart'],['d'])
        n=lineage.remap(self.s,r['lineage'],['n'],r['after_revision'],direction='reverse')
        p=self.s.store.get(n['selection_record']);self.assertEqual(p['without_counterpart'],['n'])
        self.assertEqual(p['introduced_ancestry'][0]['target'],'n')

    def test_reverse_partial_split_preserves_ambiguity(self):
        r=self.record(*self.fixture());q=lineage.remap(self.s,r['lineage'],['a1'],r['after_revision'],direction='reverse')
        self.assertEqual(q['status'],'ambiguous')

    def test_preserved_sources_can_also_contribute_to_new_derived_element(self):
        before=self.domain('a',['a','b']);after=self.domain('b',['a','b','mid'])
        rows=[dict(kind='preserved',sources=['a'],targets=['a']),dict(kind='preserved',sources=['b'],targets=['b']),
            dict(kind='derived',sources=['a','b'],targets=['mid'])]
        r=self.record(before,after,rows)
        q=lineage.remap(self.s,r['lineage'],['a'],r['before_revision'])
        p=self.s.store.get(q['selection_record'])
        self.assertEqual(p['candidates'],['a','mid']);self.assertEqual(p['ambiguous'][0]['target'],'mid')
        with self.assertRaisesRegex(ValueError,'multiply disposed'):
            self.record(before,after,rows+[dict(kind='deleted',sources=['a'],targets=[])])

    def test_composition_preserves_introduced_ancestry_and_weights_are_not_invented(self):
        before,mid,rows=self.fixture();a=self.record(before,mid,rows)
        after=self.domain('c',['joined','bc','a2'])
        b=self.record(mid,after,[dict(kind='merged',sources=['a1','n'],targets=['joined']),
            dict(kind='preserved',sources=['bc'],targets=['bc']),dict(kind='preserved',sources=['a2'],targets=['a2'])])
        c=self.s.compose_topology_lineage([a['lineage'],b['lineage']]);p=self.s.store.get(c['lineage'])
        joined=next(r for r in p['ancestry'] if r['target']=='joined')
        self.assertEqual(joined['sources'],['a']);self.assertTrue(joined['introduced_origins']);self.assertIsNone(joined['weights'])
        q=self.s.remap_topology_selection(c['lineage'],['a'],c['before_revision'])
        self.assertEqual(q['status'],'ambiguous')
        original=self.s.store.get(a['lineage']);bc=next(r for r in original['ancestry'] if r['target']=='bc')
        self.assertEqual(bc['weights'],{'b':.25,'c':.75})

    def test_stale_source_and_unrelated_chain_are_refused(self):
        r=self.record(*self.fixture())
        with self.assertRaisesRegex(ValueError,'stale'):lineage.remap(self.s,r['lineage'],['a'],'0'*64)
        with self.assertRaisesRegex(ValueError,'Unknown selected'):lineage.remap(self.s,r['lineage'],['unknown'],r['before_revision'])
        with self.assertRaisesRegex(ValueError,'endpoints differ'):lineage.compose(self.s,[r['lineage'],r['lineage']])

    def test_receipt_mismatch_and_incomplete_dispositions_are_refused(self):
        before,after,rows=self.fixture()
        with self.assertRaisesRegex(ValueError,'exact lineage'):self.record(before,after,rows,lambda r:r.update(lineage_sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'revision mismatch'):self.record(before,after,rows,lambda r:r.update(before_revision='0'*64))
        with self.assertRaisesRegex(ValueError,'explicit disposition'):self.record(before,after,rows[:-1])
        with self.assertRaisesRegex(ValueError,'multiply disposed'):self.record(before,after,rows+[rows[0]])

    def test_bad_weights_and_relation_shape_are_refused(self):
        before,after,rows=self.fixture();rows[1]['weights']['bc'].pop('c')
        with self.assertRaisesRegex(ValueError,'every source'):self.record(before,after,rows)
        rows[1].pop('weights');rows[0]['kind']='preserved'
        with self.assertRaisesRegex(ValueError,'cardinality'):self.record(before,after,rows)

    def test_original_pins_survive_source_changes_and_reads_are_bounded(self):
        r=self.record(*self.fixture());p=self.s.store.get(r['lineage']);(self.root/'1/lineage.json').write_bytes(b'changed')
        self.assertNotEqual(self.s.store.resolve_blob(p['evidence']['case']['lineage']['asset']).read_bytes(),b'changed')
        read=r['reads']['ancestry'];result=self.s.execute(read['operation'],read['arguments'])
        self.assertIn('introduced_origins',json.dumps(result))


if __name__=='__main__':unittest.main()
