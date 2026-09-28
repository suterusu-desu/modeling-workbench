import unittest
import numpy as np
from .metric_fitting import conform_to_surface
from .test_metric_fitting import plane
from .preparation import prepare_arrays


def case():
    rest, tri = plane(-.8, .8, -.8, .8, z=.19, columns=5, rows=5)
    held = (abs(rest[:, 0]) == .8) | (abs(rest[:, 1]) == .8)
    lower, lt = plane(-1, 1, -1, 1, z=0., columns=3, rows=3)
    upper, ut = plane(-1, 1, -1, 1, z=.2, columns=3, rows=3)
    assignments = np.where(rest[:, 0] <= 0, 10, 20); assignments[held] = -1
    return dict(reference=rest, initial=rest.copy(), triangles=tri, held=held,
        support_positions=np.r_[lower, upper], support_triangles=np.r_[lt, ut+len(lower)],
        support_domains=np.r_[np.full(len(lt), 10), np.full(len(ut), 20)],
        support_assignment=assignments, qualified_domains=[10, 20], support_weights=(~held).astype(float),
        units='m', frame='synthetic XYZ', relative_area_tolerance=1e-10, distance_tolerance=1e-8, iterations=30)


class AssignedSupportTests(unittest.TestCase):
    def test_nearby_wrong_sheet_is_never_used_by_assigned_vertices(self):
        a = case(); result = conform_to_surface(**a)
        free = ~a['held']; left = free & (a['support_assignment'] == 10); right = free & ~left
        np.testing.assert_allclose(result['deformed'][left, 2], 0, atol=1e-8)
        np.testing.assert_allclose(result['deformed'][right, 2], .2, atol=1e-8)
        np.testing.assert_array_equal(result['deformed'][a['held']], a['initial'][a['held']])
        self.assertEqual(result['assignment_report']['status'], 'measured')
        self.assertEqual(result['public_metrics']['component_changes_during'], 0)
        feet = result['assigned_support_triangle'][free]
        np.testing.assert_array_equal(a['support_domains'][feet], a['support_assignment'][free])
        legacy = {k: v for k, v in a.items() if k not in ('support_domains', 'support_assignment', 'qualified_domains')}
        old = conform_to_surface(**legacy)
        self.assertGreater(float(old['deformed'][left, 2].min()), .19)
        self.assertNotIn('assignment_report', old)

    def test_one_assigned_domain_reproduces_single_support_result(self):
        a = case(); a['support_triangles'] = a['support_triangles'][:8]; a['support_positions'] = a['support_positions'][:9]
        a['support_domains'] = np.zeros(8, int); a['support_assignment'] = np.zeros(len(a['initial']), int)
        a['qualified_domains'] = [0]
        assigned = conform_to_surface(**a)
        for k in ('support_domains', 'support_assignment', 'qualified_domains'): a.pop(k)
        legacy = conform_to_surface(**a)
        for key in ('deformed', 'band_residual', 'beyond_boundary', 'support_offset', 'energies'):
            np.testing.assert_array_equal(assigned[key], legacy[key])
        self.assertEqual(assigned['public_metrics'], legacy['public_metrics'])

    def test_zero_band_residual_beyond_assigned_boundary_remains_unknown(self):
        a = case()
        # Narrow only the intended lower support; the broad upper sheet cannot substitute.
        a['support_positions'][:9, 0] *= .1
        r = conform_to_surface(**a)
        self.assertEqual(r['assignment_report']['status'], 'unknown')
        self.assertGreater(r['public_metrics']['beyond_boundary_after'], 0)
        self.assertLessEqual(r['public_metrics']['band_max_after'], 1e-8)
        self.assertTrue(r['assignment_report']['domains'][0]['beyond_boundary_vertices'])

    def test_unqualified_missing_unknown_and_partial_assignment_refuse(self):
        for change in ('missing', 'unqualified', 'unknown', 'partial'):
            a = case(); v = np.flatnonzero(~a['held'])[0]
            if change == 'missing': a['support_assignment'][v] = -1
            if change == 'unknown': a['support_assignment'][v] = 500
            if change == 'unqualified': a['qualified_domains'] = [20]
            if change == 'partial': a.pop('qualified_domains')
            with self.assertRaises(ValueError, msg=change): conform_to_surface(**a)

    def test_held_conflict_and_unweighted_material_are_reported_separately(self):
        a = case(); v = int(np.flatnonzero(a['held'])[0]); a['support_weights'][v] = 1.; a['support_assignment'][v] = 10
        free = int(np.flatnonzero(~a['held'])[0]); a['support_weights'][free] = 0.; a['support_assignment'][free] = -1
        r = conform_to_surface(**a)
        self.assertEqual(r['assignment_report']['status'], 'unknown')
        self.assertIn(v, r['assignment_report']['unknown_vertices'])
        self.assertIn(free, r['assignment_report']['unconstrained_free_vertices'])
        self.assertNotIn(free, r['assignment_report']['supported_vertices'])
        np.testing.assert_array_equal(r['deformed'][v], a['initial'][v])

    def test_disconnected_parts_need_distinct_domain_labels(self):
        a = case(); a['support_domains'][:] = 10; a['support_assignment'][~a['held']] = 10; a['qualified_domains'] = [10]
        with self.assertRaisesRegex(ValueError, 'connected component'): conform_to_surface(**a)

    def test_preparation_route_and_zero_weight_scope(self):
        a = case(); r = prepare_arrays('conform_to_surface', inputs={}, parameters=a)
        self.assertEqual(r['assignment_report']['status'], 'measured')
        a['support_weights'][:] = 0.; a['support_assignment'][:] = -1
        r = conform_to_surface(**a)
        self.assertEqual(r['assignment_report']['status'], 'unmeasured')
        self.assertEqual(r['assignment_report']['supported_vertices'], [])

    def test_assignment_changes_are_bound_to_input_revision(self):
        a = case(); first = conform_to_surface(**a)
        a['support_assignment'][~a['held']] = 20
        second = conform_to_surface(**a)
        self.assertNotEqual(first['assignment_report']['input_revision'], second['assignment_report']['input_revision'])

    def test_final_projection_keeps_assignments_after_a_weak_penalty(self):
        a = case(); a['band_weight'] = 1e-9
        result = conform_to_surface(**a)
        free = ~a['held']
        self.assertGreater(result['public_metrics']['projection_max_move'], .1)
        feet = result['assigned_support_triangle'][free]
        np.testing.assert_array_equal(a['support_domains'][feet], a['support_assignment'][free])
        self.assertEqual(result['assignment_report']['status'], 'measured')
        a['project_active'] = False
        self.assertEqual(conform_to_surface(**a)['assignment_report']['status'], 'unknown')

    def test_coincident_supports_keep_separate_normals_and_ownership(self):
        a = case(); a['support_positions'][9:, 2] = 0.
        a['support_triangles'][8:] = a['support_triangles'][8:, ::-1]
        a['lower'] = a['upper'] = .05
        r = conform_to_surface(**a); free = ~a['held']
        np.testing.assert_allclose(r['deformed'][free & (a['support_assignment'] == 10), 2], .05, atol=1e-8)
        np.testing.assert_allclose(r['deformed'][free & (a['support_assignment'] == 20), 2], -.05, atol=1e-8)
        np.testing.assert_array_equal(a['support_domains'][r['assigned_support_triangle'][free]], a['support_assignment'][free])


if __name__ == '__main__':
    unittest.main()
