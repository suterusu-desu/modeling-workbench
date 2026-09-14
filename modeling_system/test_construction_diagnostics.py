import unittest
import numpy as np
from .construction_diagnostics import compare_bends, sample_residuals
from .geometry import plane_sections


class ConstructionDiagnosticsTests(unittest.TestCase):
    def test_improved_objective_does_not_hide_local_fold(self):
        flat = np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0]])
        bent = flat.copy(); bent[3,2] = 1
        tri = np.array([[0,1,2],[1,3,2]])
        result = compare_bends(flat,bent,tri,tagged_edges=[(1,2)],objective_before=8,objective_after=4)
        self.assertTrue(result['objective']['improved_with_more_large_bends'])
        self.assertEqual(result['after']['over_threshold'],1)
        self.assertEqual(result['tagged']['after']['over_threshold'],1)
        self.assertEqual(result['worst_added'][0]['edge'],[1,2])
        self.assertAlmostEqual(result['after']['maximum'],54.735610317)
        excluded = compare_bends(flat,bent,tri,excluded_edges=[(1,2)])
        self.assertEqual(excluded['measured_edges'],0)
        self.assertIsNone(excluded['after']['p95'])
        self.assertEqual(excluded['excluded_edge_counts']['boundary'],4)
        self.assertEqual(excluded['excluded_edge_counts']['explicitly_excluded'],1)
        np.testing.assert_array_equal(flat[:,2],0)

    def test_missing_additive_support_is_detected_before_target_interpretation(self):
        base = np.zeros((40,3)); observed = base.copy(); observed[:,1] = .002 + .003
        wrong = base.copy(); wrong[:,1] = .002
        result = sample_residuals(observed,wrong,1e-6,limit=3)
        self.assertFalse(result['sampled_match'])
        self.assertEqual(result['over_tolerance'],40)
        self.assertEqual(len(result['worst']),3)
        self.assertEqual(result['omitted_samples'],37)
        self.assertTrue(sample_residuals(observed,observed.copy(),1e-6)['sampled_match'])
        self.assertIn('no graph completeness',result['interpretation'])

    def test_bends_report_unmeasured_nonmanifold_and_winding(self):
        co = np.array([[0.,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]])
        r = compare_bends(co,co,np.array([[0,1,2],[0,1,3]]))
        self.assertEqual(r['excluded_edge_counts']['inconsistent_winding'],1)
        r = compare_bends(co,co,np.array([[0,1,2],[1,0,3],[0,1,4]]))
        self.assertEqual(r['excluded_edge_counts']['nonmanifold'],1)

    def test_invalid_or_empty_geometry_never_reports_success(self):
        with self.assertRaises(ValueError): sample_residuals([],[],1e-6)
        with self.assertRaises(ValueError): sample_residuals([[0,0,0]],[[0,np.nan,0]],1e-6)
        co = np.array([[0.,0,0],[1,0,0],[2,0,0]])
        with self.assertRaises(ValueError): compare_bends(co,co,np.array([[0,1,2]]))

    def test_exact_vertex_plane_crossing_retains_both_endpoints(self):
        arrays = dict(co=np.array([[0.,0,0],[-1,1,0],[1,1,0]]),tri=np.array([[0,1,2]]))
        result = plane_sections(arrays,0,0)
        self.assertEqual(result['segment_count'],1)
        self.assertEqual(result['triangle_indices'],[0])
        self.assertEqual({tuple(p) for p in result['segments'][0]},{(0.,0.,0.),(0.,1.,0.)})

    def test_plane_edge_contacts_tangency_and_coplanar_are_distinct(self):
        for third, count, coplanar in (([1,0,1],1,0),([0,0,1],0,1)):
            arrays = dict(co=np.array([[0.,0,0],[0,1,0],third]),tri=np.array([[0,1,2]]))
            result = plane_sections(arrays,0,0)
            self.assertEqual(result['segment_count'],count)
            self.assertEqual(result['coplanar_triangles_excluded'],coplanar)
        tangent = dict(co=np.array([[0.,0,0],[1,1,0],[1,0,1]]),tri=np.array([[0,1,2]]))
        self.assertEqual(plane_sections(tangent,0,0)['segment_count'],0)


if __name__ == '__main__': unittest.main()
