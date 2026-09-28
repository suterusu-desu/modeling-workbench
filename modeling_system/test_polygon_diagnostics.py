"""Native polygon shape witnesses independent of changing triangle-row layouts."""
import copy
import json
import unittest
import numpy as np
from .construction_diagnostics import compare_polygon_shapes


def record(points=None, diagonal=0):
    return {'co': np.array(points if points is not None else [[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        'tri': np.array([[0, 1, 2], [0, 2, 3]] if diagonal == 0 else [[0, 1, 3], [1, 2, 3]]),
        'loops': np.arange(4), 'polygon_starts': np.array([0]), 'polygon_lengths': np.array([4]),
        'triangle_polygon': np.array([0, 0])}


def compare(a, b, ids=(0,), tolerance=1e-10):
    return compare_polygon_shapes(a, b, list(ids), units='m', frame='synthetic common frame', relative_tolerance=tolerance)


class PolygonDiagnosticTests(unittest.TestCase):
    def test_planar_quad_and_changed_native_diagonal(self):
        a, b = record(), record(diagonal=1)
        r = compare(a, b); p = r['polygons'][0]
        self.assertEqual(r['status'], 'measured')
        self.assertFalse(r['triangle_layout_equal']); self.assertFalse(p['native_triangle_rows_equal'])
        self.assertEqual(p['before']['native_internal_angles'][0]['edge'], [0, 2])
        self.assertEqual(p['after']['native_internal_angles'][0]['edge'], [1, 3])
        self.assertEqual(p['projected_crossing_transition'], 'absent_in_both')
        self.assertEqual(p['before_after_area_normal_dot'], 1.)
        for side in ('before', 'after'):
            self.assertEqual(p[side]['maximum_native_dihedral_degrees'], 0.)
            self.assertEqual(p[side]['self_intersection']['status'], 'not_measured')
            for t in p[side]['native_triangles']:
                self.assertEqual(t['normal'], [0., 0., 1.]); self.assertAlmostEqual(t['area'], .5)

    def test_rigid_half_turn_is_facing_change_not_new_shape_defect(self):
        a, b = record(), record()
        b['co'] = b['co'] @ np.diag([1., -1., -1.])+[8, 3, 2]
        p = compare(a, b)['polygons'][0]
        self.assertEqual(p['before_after_area_normal_dot'], -1.)
        self.assertEqual(p['after']['maximum_native_dihedral_degrees'], 0.)
        self.assertEqual(p['projected_crossing_transition'], 'absent_in_both')

    def test_projected_bow_tie_is_separate_from_spatial_intersection(self):
        a = record(diagonal=1)
        b = record([[0., 0, 0], [1, 1, .2], [0, 1, 0], [1, 0, .2]], diagonal=1)
        p = compare(a, b)['polygons'][0]; after = p['after']
        self.assertEqual(p['projected_crossing_transition'], 'introduced_in_projection')
        self.assertGreater(after['maximum_native_dihedral_degrees'], 140)
        crossings = [c for c in after['best_fit_plane']['projected_contacts'] if c['kind'] == 'proper_crossing']
        self.assertTrue(crossings)
        # A lifted corner changes the depth where the projected edges cross.
        b['co'][1, 2] = .4
        p = compare(a, b)['polygons'][0]
        crossings = [c for c in p['after']['best_fit_plane']['projected_contacts'] if c['kind'] == 'proper_crossing']
        self.assertGreater(crossings[0]['spatial_gap_at_projected_crossing'], .05)
        self.assertEqual(p['after']['self_intersection']['status'], 'not_measured')

    def test_baseline_crossing_is_not_reported_as_new(self):
        a = record([[0., 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
        b = copy.deepcopy(a); b['co'] *= 2
        p = compare(a, b)['polygons'][0]
        self.assertEqual(p['projected_crossing_transition'], 'present_in_both')
        self.assertEqual(p['before']['maximum_native_dihedral_degrees'], 180.)
        self.assertIsNone(p['before']['polygon_area_normal'])
        self.assertEqual(compare(a, record())['polygons'][0]['projected_crossing_transition'], 'absent_after_in_projection')

    def test_positive_fixed_reference_plane_areas_do_not_hide_native_twist(self):
        a = record([[0., 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]], diagonal=1)
        b = copy.deepcopy(a); b['co'][[0, 2], 2] = 3
        t = b['co'][b['tri']]
        # A positive signed area in the original XY plane passes both triangles.
        self.assertTrue((np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])[:, 2] > 0).all())
        p = compare(a, b)['polygons'][0]
        self.assertGreater(p['after']['maximum_native_dihedral_degrees'], 140.)
        self.assertEqual(p['projected_crossing_transition'], 'introduced_in_projection')
        self.assertEqual(p['after']['self_intersection']['status'], 'not_measured')

    def test_concavity_and_actual_native_diagonal_are_distinct(self):
        # Same concave perimeter, different actual native triangulations. Only
        # one diagonal covers it without opposite normals; loop projection is simple.
        a = record([[0., 0, 0], [1, 0, 0], [.2, .2, 0], [0, 1, 0]])
        b = copy.deepcopy(a); b['tri'] = record(diagonal=1)['tri']
        p = compare(a, b)['polygons'][0]
        self.assertEqual(p['projected_crossing_transition'], 'absent_in_both')
        self.assertEqual(p['before']['maximum_native_dihedral_degrees'], 0.)
        self.assertEqual(p['after']['maximum_native_dihedral_degrees'], 180.)
        self.assertLess(p['after']['native_triangles'][1]['dot_polygon_area_normal'], 0)

    def test_degenerate_geometry_remains_measured_with_unknown_normals(self):
        a = record(); b = record([[0., 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]])
        r = compare(a, b); p = r['polygons'][0]
        self.assertEqual(r['status'], 'measured')
        self.assertEqual(p['after']['native_degenerate_triangle_ids'], [0, 1])
        self.assertEqual(p['after']['degenerate_loop_edges'], [0])
        self.assertIsNone(p['after']['maximum_native_dihedral_degrees'])
        self.assertFalse(p['after']['native_dihedral_complete'])
        self.assertEqual(p['projected_crossing_transition'], 'unknown')
        json.dumps(r, allow_nan=False)
        b['co'][:] = 0
        self.assertEqual(compare(a, b)['polygons'][0]['after']['extent'], 0)

    def test_nonunique_best_fit_plane_is_unknown(self):
        b = record([[1., 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        p = compare(record(), b)['polygons'][0]
        self.assertEqual(p['after']['best_fit_plane']['status'], 'unknown')
        self.assertIsNone(p['after']['best_fit_plane']['proper_crossing'])
        self.assertIsNotNone(p['after']['maximum_native_dihedral_degrees'])

    def test_ngon_and_noncontiguous_native_row_ownership(self):
        a = {'co': np.array([[0., 0, 0], [1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0], [0, 2, 0]]),
             'tri': np.array([[0, 1, 2], [4, 3, 5], [0, 2, 3], [0, 3, 4]]),
             'loops': np.array([0, 1, 2, 3, 4, 4, 3, 5]), 'polygon_starts': np.array([0, 5]),
             'polygon_lengths': np.array([5, 3]), 'triangle_polygon': np.array([0, 1, 0, 0])}
        b = copy.deepcopy(a); b['tri'] = b['tri'][[1, 2, 0, 3]]; b['triangle_polygon'] = b['triangle_polygon'][[1, 2, 0, 3]]
        r = compare(a, b, ids=(1, 0))
        self.assertEqual([p['polygon_id'] for p in r['polygons']], [1, 0])
        self.assertEqual([t['triangle_id'] for t in r['polygons'][0]['after']['native_triangles']], [0])
        self.assertEqual([t['triangle_id'] for t in r['polygons'][1]['after']['native_triangles']], [1, 2, 3])
        self.assertEqual(len(r['polygons'][1]['after']['native_internal_angles']), 2)

    def test_touches_are_not_strict_crossings(self):
        b = record([[0., 0, 0], [1, 0, 0], [.5, 0, 0], [0, 1, 0]])
        p = compare(record(), b)['polygons'][0]['after']
        self.assertFalse(p['best_fit_plane']['proper_crossing'])
        self.assertTrue(any(c['kind'] == 'touch_or_overlap' for c in p['best_fit_plane']['projected_contacts']))
        self.assertEqual(p['native_degenerate_triangle_ids'], [0])

    def test_missing_or_invalid_loop_witness_never_falls_back(self):
        for field in ('loops', 'polygon_starts', 'polygon_lengths', 'triangle_polygon'):
            b = record(); del b[field]
            r = compare(record(), b)
            self.assertEqual(r['status'], 'unknown'); self.assertEqual(r['polygons'], [])
        b = record(); b['triangle_polygon'] = np.array([0, 1])
        self.assertEqual(compare(record(), b)['status'], 'unknown')
        b = record(); b['tri'][1] = b['tri'][0]
        self.assertEqual(compare(record(), b)['status'], 'unknown')

    def test_changed_recorded_identity_refuses_even_if_geometry_matches(self):
        b = record(); b['loops'] = b['loops'][::-1]; b['tri'] = b['tri'][:, ::-1]
        r = compare(record(), b)
        self.assertEqual(r['status'], 'unknown'); self.assertIn('identity differs', r['reasons'][0]['reason'])

    def test_scale_translation_and_no_mutation(self):
        a = record(); b = record([[0., 0, 0], [1, 0, 0], [1, 1, .3], [0, 1, 0]])
        original = copy.deepcopy(b); first = compare(a, b)['polygons'][0]['after']
        for scale in (1e-5, 1000.):
            c, d = copy.deepcopy(a), copy.deepcopy(b)
            c['co'] = c['co']*scale+[10, 20, 30]; d['co'] = d['co']*scale+[10, 20, 30]
            after = compare(c, d)['polygons'][0]['after']
            self.assertAlmostEqual(after['maximum_native_dihedral_degrees'], first['maximum_native_dihedral_degrees'], places=6)
            self.assertAlmostEqual(after['best_fit_plane']['rms_distance']/scale, first['best_fit_plane']['rms_distance'], places=8)
        for field in original:
            np.testing.assert_array_equal(original[field], b[field])

    def test_receipt_binds_actual_tessellations_and_tolerance(self):
        a, b = record(), record(diagonal=1)
        first = compare(a, a); second = compare(a, b); third = compare(a, b, tolerance=1e-9)
        self.assertEqual(len({r['input_revision'] for r in (first, second, third)}), 3)
        self.assertNotEqual(second['array_sha256']['before']['tri'], second['array_sha256']['after']['tri'])

    def test_malformed_parameters_and_nonfinite_data_refuse(self):
        for ids in ([], [0, 0], [-1], [1], [0.], [True]):
            with self.assertRaises(ValueError): compare(record(), record(), ids=ids)
        for tol in (0., -1., np.nan, np.inf, 1.):
            with self.assertRaises(ValueError): compare(record(), record(), tolerance=tol)
        b = record(); b['co'][0] = np.nan
        with self.assertRaises(ValueError): compare(record(), b)


if __name__ == '__main__':
    unittest.main()
