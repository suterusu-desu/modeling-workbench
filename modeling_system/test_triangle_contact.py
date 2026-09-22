import unittest
import numpy as np
from .triangle_contact import projected_triangle_contact, _clip


class TriangleContactTests(unittest.TestCase):
    def setUp(self):
        self.surface = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        self.triangles = np.array([[0, 1, 2]])

    def contact(self, **changes):
        values = dict(surface=self.surface, baseline=self.surface,
            triangles=self.triangles, obstacle=self.surface + [0, 0, -1],
            obstacle_triangles=self.triangles, frame=np.eye(3), margin=0., tolerance=1e-9)
        values.update(changes)
        return projected_triangle_contact(**values)

    def test_crossing_triangles_have_overlap_without_contained_vertices(self):
        surface = np.array([[0, 2, 0], [-2, -1, 0], [2, -1, 0]], float)
        obstacle = np.array([[0, -2, 1], [-2, 1, 1], [2, 1, 1]], float)
        result = self.contact(surface=surface, baseline=surface, obstacle=obstacle, bound_mode='clearance')
        self.assertEqual(result['overlap_pairs'], 1)
        self.assertEqual(result['critical_vertices'], 6)
        self.assertEqual(result['violating_vertices'], 6)
        self.assertAlmostEqual(result['minimum_slack'], -1.)
        self.assertTrue(result['coverage_complete']); self.assertFalse(result['passed'])

    def test_bound_branch_crossing_catches_violation_missed_at_polygon_vertices(self):
        baseline = self.surface.copy(); baseline[:, 2] = [-1, 1, 1]
        obstacle = self.surface.copy(); obstacle[:, 2] = [1, -1, -1]
        candidate = self.surface + [0, 0, -.5]
        result = self.contact(surface=candidate, baseline=baseline, obstacle=obstacle,
                              build_constraints=True, displacement_direction=[0., 0., 1.])
        self.assertEqual(result['branch_crossings'], 2)
        self.assertEqual(result['violating_vertices'], 2)
        self.assertAlmostEqual(result['minimum_slack'], -.5)
        matrix = np.zeros(tuple(result['constraint_shape']))
        np.add.at(matrix, (result['constraint_rows'], result['constraint_columns']), result['constraint_values'])
        self.assertAlmostEqual(float((matrix @ (candidate[:, 2] - baseline[:, 2]) - result['lower_bound']).min()), -.5)
        self.assertGreaterEqual(float((-result['lower_bound']).min()), 0.)

    def test_triangle_winding_does_not_change_contact(self):
        expected = self.contact(obstacle=self.surface, margin=.2, bound_mode='clearance')
        for source, obstacle in (([2, 1, 0], [0, 1, 2]), ([0, 1, 2], [2, 1, 0]), ([2, 1, 0], [2, 1, 0])):
            actual = self.contact(triangles=[source], obstacle_triangles=[obstacle],
                                  obstacle=self.surface, margin=.2, bound_mode='clearance')
            self.assertEqual(actual['critical_vertices'], expected['critical_vertices'])
            self.assertAlmostEqual(actual['minimum_slack'], expected['minimum_slack'])

    def test_near_boundary_clipping_never_extrapolates_beyond_subject(self):
        subject = np.array([[.2, -2e-6], [.4, -.5e-6], [.3, .2]])
        polygon = _clip(subject, np.array([[0., 0.], [1., 0.], [0., 1.]]), 1e-6)
        self.assertGreaterEqual(polygon[:, 0].min(), subject[:, 0].min())
        self.assertLessEqual(polygon[:, 0].max(), subject[:, 0].max())

    def test_fixed_vertices_and_zero_motion_keep_infeasible_constraints(self):
        result = self.contact(obstacle=self.surface, margin=.2, bound_mode='clearance', active_indices=[0],
                              build_constraints=True, displacement_direction=[0, 0, 1])
        self.assertEqual(result['fixed_infeasible_rows'], 2)
        self.assertEqual(tuple(result['constraint_shape']), (3, 1))
        result = self.contact(obstacle=self.surface, margin=.2, bound_mode='clearance',
                              build_constraints=True, displacement_direction=[0, 0, 1], displacement_scale=0.)
        self.assertEqual(result['fixed_infeasible_rows'], 3)
        self.assertEqual(len(result['constraint_values']), 0)

    def test_bad_frame_and_changed_projection_cannot_authorize_constraints(self):
        with self.assertRaisesRegex(ValueError, 'expected shape'):
            self.contact(frame=np.eye(3)[:, :2])
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            self.contact(frame=np.eye(3)*2)
        with self.assertRaisesRegex(ValueError, 'projected coordinates'):
            self.contact(surface=self.surface+[.1, 0, 0], build_constraints=True, displacement_direction=[0, 0, 1])
        with self.assertRaisesRegex(ValueError, 'projected coordinates'):
            self.contact(build_constraints=True, displacement_direction=[1, 0, 0])

    def test_degenerate_empty_overlap_and_out_of_range_topology_do_not_pass(self):
        for changes in ({'triangles': [[0, 0, 1]]}, {'obstacle_triangles': [[0, 0, 1]]},
                        {'obstacle': self.surface+[5, 0, 0]}):
            result = self.contact(**changes)
            self.assertFalse(result['coverage_complete']); self.assertFalse(result['passed'])
        with self.assertRaisesRegex(ValueError, 'topology index'):
            self.contact(triangles=[[0, 1, 3]])

    def test_obstacle_witness_uses_original_triangle_identity_after_degenerate_filter(self):
        result = self.contact(obstacle=self.surface, obstacle_triangles=[[0, 0, 1], [0, 1, 2]],
                              bound_mode='clearance', margin=.1)
        self.assertEqual(result['worst'][0]['obstacle_triangle'], 1)
        self.assertFalse(result['coverage_complete'])

    def test_explicit_depth_axis_and_scale_define_constraint_sign(self):
        result = self.contact(obstacle=self.surface, margin=.2, bound_mode='clearance',
                              build_constraints=True, displacement_direction=[0, 0, -1], displacement_scale=.5)
        np.testing.assert_allclose(result['constraint_values'], [-.5]*3)
        self.assertFalse(result['passed'])
        np.testing.assert_array_equal(self.surface[:, 2], [0, 0, 0])
        self.assertTrue(self.contact()['passed'])


if __name__ == '__main__': unittest.main()
