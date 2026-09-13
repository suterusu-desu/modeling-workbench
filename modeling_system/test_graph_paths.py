"""Budget exhaustion and selection boundaries never become global absence."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from .service import ModelingService
from .store import canonical,digest
from .graph_paths import inspect


class GraphPathTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.s=ModelingService(self.root,store=self.root/'store')
        self.state=dict(source_state_id='fixture',pose={},frame='world',units='unit',dependency_fingerprint='graph')
        self.edges=np.array([[0,1],[0,2],[1,3],[2,3],[3,4]])
        self.nodes=np.ones(5,dtype=bool);self.selected=np.ones(5,dtype=bool)
        self.case=dict(schema_version=1,question='Which routes are established by the finite graph?',state=self.state,
            graph=dict(meaning='Exact synthetic edges',evidence=['fixture'],coverage=dict(status='complete_for_declared_graph',missing=[])),
            selection=dict(meaning='Explicit edge and node subset',evidence=['fixture']),start='a',goal='e')

    def run_case(self,**kwargs):
        np.savez(self.root/'graph.npz',node_ids=np.array(list('abcde')),edges=self.edges,allowed_nodes=self.nodes,allowed_edges=self.selected)
        self.case['arrays']=dict(path='graph.npz',sha256=digest((self.root/'graph.npz').read_bytes()))
        p=self.root/'case.json';p.write_bytes(canonical(self.case))
        r=self.s.inspect_graph_path(str(p),self.state,**kwargs);return r,self.s.store.get(r['analysis'],'graph_path')

    def test_equal_shortest_alternatives_are_retained(self):
        r,p=self.run_case();self.assertEqual(r['status'],'path_found')
        self.assertEqual(r['summary']['shortest_path_status'],'ambiguous_shortest_paths')
        self.assertEqual(p['path'],['a','b','d','e'])
        self.assertEqual(next(r for r in p['predecessors'] if r['node']=='d')['parents'],['b','c'])

    def test_node_or_edge_budget_exhaustion_stays_incomplete(self):
        for budget in [dict(max_nodes=2),dict(max_edges=1)]:
            r,_=self.run_case(**budget);self.assertEqual(r['status'],'incomplete_budget')
            self.assertEqual(r['summary']['shortest_path_status'],'unknown')

    def test_edge_selection_is_not_replaced_by_whole_graph_connectivity(self):
        self.selected[-1]=False;r,p=self.run_case()
        self.assertEqual(r['status'],'no_path_in_declared_selection');self.assertFalse(p['path'])
        self.assertTrue(any('global unreachability' in s for s in p['limits']))

    def test_excluded_endpoint_and_partial_coverage_remain_explicit(self):
        self.nodes[-1]=False;r,_=self.run_case();self.assertEqual(r['status'],'endpoint_outside_selection')
        self.nodes[-1]=True;self.selected[-1]=False
        self.case['graph']['coverage']=dict(status='declared_subset',missing=['Hidden geometry outside this graph'])
        r,_=self.run_case();self.assertEqual(r['summary']['coverage']['status'],'declared_subset')

    def test_unique_shortest_does_not_mean_unique_route(self):
        self.edges=np.array([[0,1],[1,4],[0,2],[2,3],[3,4]])
        r,p=self.run_case();self.assertEqual(r['summary']['shortest_path_status'],'unique_shortest_path')
        self.assertTrue(any('longer alternatives' in s for s in p['limits']))

    def test_invalid_graph_is_refused(self):
        self.edges[0]=[0,0]
        with self.assertRaisesRegex(ValueError,'Invalid graph'):self.run_case()


if __name__=='__main__':unittest.main()
