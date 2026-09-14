import unittest
import numpy as np
from .construction_diagnostics import compare_bends, sample_residuals
from .geometry import plane_sections, compare


class ConstructionDiagnosticsTests(unittest.TestCase):
    def test_pose_diagonal_change_requires_native_or_declared_fixed_chart_basis(self):
        # Independent nonplanar quad: the two diagonals yield analytically
        # different normal angles despite identical vertex identities.
        flat = np.array([[0.,0,0],[2,0,0],[2,1,0],[0,1,0]])
        bent = flat.copy(); bent[2,2] = 1
        first_diagonal = np.array([[0,1,2],[0,2,3]])
        second_diagonal = np.array([[0,1,3],[1,2,3]])
        pose_a = dict(co=flat,tri=first_diagonal)
        pose_b = dict(co=bent,tri=second_diagonal)
        with self.assertRaisesRegex(ValueError,'Topology differs'):
            compare(pose_a,pose_b)
        native_before = dict(co=flat,tri=second_diagonal)
        self.assertTrue(compare(native_before,pose_b)['triangle_layout_equal'])
        native = compare_bends(flat,bent,pose_b['tri'])
        fixed_chart = compare_bends(flat,bent,pose_a['tri'])
        self.assertEqual(native['worst_added'][0]['edge'],[1,3])
        self.assertEqual(fixed_chart['worst_added'][0]['edge'],[0,2])
        self.assertAlmostEqual(native['after']['maximum'],np.degrees(np.arccos(2/3)))
        self.assertAlmostEqual(fixed_chart['after']['maximum'],np.degrees(np.arccos(2/np.sqrt(10))))
        self.assertNotEqual(native['input_revision'],fixed_chart['input_revision'])
        # Native and fixed-chart sections also differ at the diagonal midpoint.
        native_cut = plane_sections(pose_b,0,1)
        fixed_cut = plane_sections(dict(co=bent,tri=first_diagonal),0,1)
        native_points = {tuple(p) for segment in native_cut['segments'] for p in segment}
        fixed_points = {tuple(p) for segment in fixed_cut['segments'] for p in segment}
        self.assertIn((1.,.5,0.),native_points)
        self.assertIn((1.,.5,.5),fixed_points)
        self.assertNotEqual(native_points,fixed_points)

    def test_polygon_loop_correspondence_does_not_certify_triangle_transfer(self):
        co = np.array([[0.,0,0],[2,0,0],[2,1,1],[0,1,0]])
        loops = dict(loops=np.array([0,1,2,3]),polygon_starts=np.array([0]),polygon_lengths=np.array([4]))
        a = dict(co=co,tri=np.array([[0,1,2],[0,2,3]]),**loops)
        b = dict(co=co,tri=np.array([[0,1,3],[1,2,3]]),**loops)
        result = compare(a,b,correspondence='recorded_polygon_loops')
        self.assertEqual(result['changed_vertices'],0)
        self.assertFalse(result['triangle_layout_equal'])
        with self.assertRaisesRegex(ValueError,'Topology differs'): compare(a,b)

    def test_fixed_anchor_rotating_frame_explains_motion_but_not_added_residual(self):
        anchors = np.tile([2.,3.,4.],(3,1))
        offsets = np.array([[1.,0,0],[0,2,0],[0,0,1]])
        baseline = np.array([[3.,3,4],[2,5,4],[2,3,5]])
        observed = np.array([[2.,4,4],[0,3,4],[2,3,5]])
        rotation = np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        # Expected positions derive from independent anchors/basis/offsets,
        # not a frame fitted to the dependent observed coordinates.
        frames = np.tile(rotation,(3,1,1))
        predicted = anchors + np.einsum('nij,nj->ni',frames,offsets)
        self.assertTrue(sample_residuals(anchors+offsets,baseline,1e-8)['sampled_match'])
        self.assertEqual(sample_residuals(baseline,observed,1e-8)['over_tolerance'],2)
        self.assertTrue(sample_residuals(predicted,observed,1e-8)['sampled_match'])
        unexplained = observed.copy(); unexplained[1,2] += .001
        report = sample_residuals(predicted,unexplained,1e-8,limit=1)
        self.assertFalse(report['sampled_match'])
        self.assertEqual(report['over_tolerance'],1)
        self.assertEqual(report['worst'][0]['sample'],1)
        self.assertAlmostEqual(report['maximum'],.001)
        self.assertFalse(sample_residuals(anchors+offsets[[1,0,2]],baseline,1e-8)['sampled_match'])
        np.testing.assert_array_equal(anchors,np.tile([2.,3.,4.],(3,1)))
        np.testing.assert_array_equal(observed,np.array([[2.,4,4],[0,3,4],[2,3,5]]))

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
