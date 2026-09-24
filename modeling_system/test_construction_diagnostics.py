import unittest
import numpy as np
from .construction_diagnostics import (compare_bends, compare_stretch, edge_values_at_vertices, local_reversals,
                                       sample_residuals, section_turns)
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

    def test_edge_values_sign_valleys_and_ridges_against_the_winding_normal(self):
        # Two triangles meeting along the y axis, wound so both normals are +z when flat.
        flat = np.array([[0.,0,0],[0,1,0],[-1,.5,0],[1,.5,0]])
        tri = np.array([[0,1,2],[1,0,3]])
        valley, ridge = flat.copy(), flat.copy()
        valley[[2,3],2] = 1.; ridge[[2,3],2] = -1.       # folded toward / away from the normal side
        for bent, sign in ((valley,-1.),(ridge,1.)):
            for order in (tri, tri[::-1]):               # the sign does not depend on which face comes first
                r = compare_bends(flat,bent,order,tagged_edges=[(0,1)],edge_values=True)
                ev = r['edge_values']
                np.testing.assert_array_equal(ev['edges'],[[0,1]])
                self.assertAlmostEqual(float(ev['after'][0]),90.)
                self.assertAlmostEqual(float(ev['after_signed'][0]),sign*90.)
                self.assertAlmostEqual(float(ev['before_signed'][0]),0.)
                self.assertTrue(bool(ev['tagged'][0]))
                self.assertAlmostEqual(r['after']['maximum'],90.)
        plain = compare_bends(flat,valley,tri)
        self.assertNotIn('edge_values',plain)
        import json; json.dumps(plain)                  # the default report stays JSON
        with self.assertRaises(ValueError): compare_bends(flat,valley,tri,edge_values=1)

    def test_edge_values_at_vertices_keeps_the_largest_magnitude(self):
        edges = np.array([[0,1],[1,2],[2,3]])
        out = edge_values_at_vertices(edges,[5.,-7.,2.],5)
        np.testing.assert_array_equal(out,[5.,-7.,-7.,2.,0.])
        with self.assertRaises(ValueError): edge_values_at_vertices(edges,[1.,2.],5)
        with self.assertRaises(ValueError): edge_values_at_vertices(edges,[1.,2.,3.],3)
        with self.assertRaises(ValueError): edge_values_at_vertices(edges,[1.,np.nan,3.],5)

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


class MaterialStretchTests(unittest.TestCase):
    def grid(self):
        x, y = np.meshgrid(np.arange(4.), np.arange(3.), indexing='ij')
        co = np.c_[x.ravel(), y.ravel(), np.zeros(12)]
        index = np.arange(12).reshape(4, 3); tri = []
        for i in range(3):
            for j in range(2):
                a, b, c, d = index[i, j], index[i+1, j], index[i+1, j+1], index[i, j+1]
                tri += [[a, b, c], [a, c, d]]
        return co, np.array(tri)

    def test_identity_rigid_motion_and_uniform_compression_are_measured_exactly(self):
        co, tri = self.grid()
        same = compare_stretch(co, co, tri)
        self.assertAlmostEqual(same['all']['smallest_stretch_minimum'], 1.)
        self.assertEqual(same['all']['compressed'], 0)
        angle = .7; rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        moved = compare_stretch(co, co @ rotation.T + [3, -2, 5], tri)
        self.assertAlmostEqual(moved['all']['smallest_stretch_minimum'], 1., places=12)
        self.assertAlmostEqual(moved['all']['largest_stretch_q99'], 1., places=12)
        self.assertEqual(same['all']['local_reversals'], 0); self.assertEqual(moved['all']['local_reversals'], 0)
        crowded = co * [.4, 1, 1]
        result = compare_stretch(co, crowded, tri)
        self.assertAlmostEqual(result['all']['smallest_stretch_minimum'], .4, places=12)
        self.assertEqual(result['all']['compressed'], len(tri))
        self.assertEqual(result['all']['normal_reversals'], 0)

    def test_folded_triangle_reports_a_normal_reversal_and_tags_split_the_domain(self):
        co, tri = self.grid()
        folded = co.copy(); folded[4] = [3.2, 1.5, 0]   # interior vertex pushed across its neighbors
        result = compare_stretch(co, folded, tri, tagged_triangles=np.array([0, 1]))
        self.assertGreater(result['all']['normal_reversals'], 0)
        self.assertGreater(result['all']['local_reversals'], 0)            # flipped against its own neighbours
        self.assertEqual(result['tagged']['count'], 2); self.assertEqual(result['untagged']['count'], len(tri) - 2)
        self.assertLessEqual(len(result['worst_compressed']), 12)
        self.assertIn(4, result['worst_compressed'][0]['vertices'])

    def test_material_turning_coherently_is_not_a_local_reversal(self):
        # A strip rolled onto a cylinder through 200 degrees (a closing lid margin rolls like this) and a patch
        # turned over rigidly: their normals oppose the reference direction, but no triangle is flipped
        # relative to its own neighbourhood.
        x, y = np.meshgrid(np.arange(3.), np.arange(15.) * .25, indexing='ij')
        flat = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        index = np.arange(x.size).reshape(x.shape); tri = []
        for i in range(x.shape[0] - 1):
            for j in range(x.shape[1] - 1):
                a, b, c, d = index[i, j], index[i+1, j], index[i+1, j+1], index[i, j+1]
                tri += [[a, b, c], [a, c, d]]
        tri = np.array(tri); theta = flat[:, 1] * (np.radians(200) / flat[:, 1].max())
        rolled = np.c_[flat[:, 0], np.sin(theta), 1 - np.cos(theta)]
        result = compare_stretch(flat, rolled, tri)
        self.assertGreater(result['all']['normal_reversals'], len(tri) // 3)
        self.assertEqual(result['all']['local_reversals'], 0)
        half_turn = flat * [1, -1, -1] + [0, 0, 2]                       # 180 degrees about the x axis
        turned = compare_stretch(flat, half_turn, tri)
        self.assertEqual(turned['all']['normal_reversals'], len(tri))
        self.assertEqual(turned['all']['local_reversals'], 0)
        np.testing.assert_array_equal(local_reversals(flat, flat, tri), np.zeros(len(tri), bool))

    def test_degenerate_reference_or_invalid_thresholds_refuse(self):
        co, tri = self.grid()
        flat = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'Degenerate'): compare_stretch(flat, flat, [[0, 1, 2]])
        for kwargs in ({'compressed_below': 1.2}, {'stretched_above': .9}, {'limit': 0}):
            with self.assertRaises(ValueError): compare_stretch(co, co, tri, **kwargs)
        collapsed = co.copy(); collapsed[:, 0] = 0
        self.assertEqual(compare_stretch(co, collapsed, tri)['all']['smallest_stretch_minimum'], 0.)


if __name__ == '__main__': unittest.main()


class SectionTurnTests(unittest.TestCase):
    def strip(self, nu=25, nx=4):
        u, x = np.meshgrid(np.linspace(0, 1, nu), np.linspace(-.1, .1, nx))
        rest = np.c_[x.ravel(), np.zeros(u.size), u.ravel()]
        tri = []
        for j in range(nx - 1):
            for i in range(nu - 1):
                a = j * nu + i; tri += [[a, a + 1, a + nu + 1], [a, a + nu + 1, a + nu]]
        return rest, np.array(tri), u.ravel()

    def test_rolled_band_keeps_its_turning_direction_and_an_s_adds_an_inflection(self):
        rest, tri, u = self.strip()
        arc = rest.copy(); angle = u * np.pi * .8; arc[:, 1] = -(1 - np.cos(angle)) / 2; arc[:, 2] = np.sin(angle) / 2
        s_shape = rest.copy(); s_shape[:, 1] = .08 * np.sin(2 * np.pi * u)
        result = section_turns(rest, np.stack([arc, s_shape]), tri, origin=[0., 0., 0.], normal=[1., 0., 0.])
        self.assertEqual(result['sections'], 1); self.assertEqual(result['reference_inflections'], 0)
        self.assertEqual(result['inflections_per_pose'][0], 0)
        self.assertGreaterEqual(result['inflections_per_pose'][1], 1)
        self.assertEqual(result['added_inflections_per_pose'], [0, result['inflections_per_pose'][1]])
        self.assertEqual(result['section_points'].shape[0], 3)            # reference plus both poses, same material
        np.testing.assert_allclose(result['section_points'][0][:, 0], 0, atol=1e-12)

    def test_refuses_bad_plane(self):
        rest, tri, _ = self.strip(5, 3)
        with self.assertRaisesRegex(ValueError, 'nonzero normal'):
            section_turns(rest, rest[None], tri, origin=[0, 0, 0], normal=[0, 0, 0])

