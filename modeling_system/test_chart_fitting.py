"""Synthetic cross-cell material redistribution on explicitly declared source charts."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult, minimize
from .metric_fitting import conform_to_surface
from .test_ordered_fitting import fan
from .test_metric_fitting import plane


def chart_fan(curved=False, scale=1., translation=0.):
    a = fan(scale=scale)
    source, st = plane(0, 1, 0, 1, z=0., columns=5, rows=5)
    uv = source[:, :2].copy()
    if curved:
        source[:, 2] = .3*source[:, 0]**2+.15*source[:, 1]**2
    c = a['ordered_charts'][0]
    refs, bary = [], []
    for point in a['initial'][:, :2]/scale:
        for index, face in enumerate(st):
            ab = np.linalg.solve((uv[face[1:]]-uv[face[0]]).T, point-uv[face[0]])
            b = np.r_[1-ab.sum(), ab]
            if b.min() >= -1e-14:
                refs.append(index); bary.append(np.maximum(b, 0)); break
        else:
            raise AssertionError('Synthetic fixture point outside source')
    refs, bary = np.asarray(refs), np.asarray(bary)
    source = source*scale+translation
    a.update(support_positions=source, support_triangles=st, support_domains=np.zeros(len(st), int),
             ordered_cell_mode='chart')
    c.update(source_uv=uv, support_triangle_ids=np.arange(len(st)),
             vertex_source_triangles=refs, initial_barycentric=bary)
    a['initial'] = np.einsum('nk,nkd->nd', bary, source[st[refs]])
    a['reference'] = a['initial'].copy()
    a['reference'][4] = np.array([.2, .7, .09375 if curved else 0])*scale+translation
    return a


class ChartFittingTests(unittest.TestCase):
    def assert_lifts(self, a, r):
        for c in r['ordered_report']['charts']:
            ids = c['vertices']; refs = np.asarray(c['source_triangle_ids']); bary = np.asarray(c['barycentric'])
            lift = np.einsum('nk,nkd->nd', bary, a['support_positions'][a['support_triangles'][refs]])
            np.testing.assert_allclose(r['deformed'][ids], lift, atol=1e-13, rtol=1e-14)
            np.testing.assert_array_equal(r['assigned_support_triangle'][ids], refs)
            self.assertGreaterEqual(bary.min(), 0.)
            self.assertGreaterEqual(c['minimum_area_ratio_after'], c['minimum_area_ratio']-1e-10)
            fixed = c['fixed_vertices']
            np.testing.assert_array_equal(r['deformed'][fixed], a['initial'][fixed])

    def test_reaches_rest_across_multiple_cells_and_preserves_boundary(self):
        a = chart_fan(); r = conform_to_surface(**a)
        self.assertEqual(r['ordered_report']['status'], 'converged')
        np.testing.assert_allclose(r['deformed'][4], a['reference'][4], atol=2e-6)
        c = r['ordered_report']['charts'][0]
        self.assertEqual(c['changed_cell_vertices'], [4])
        self.assertNotEqual(c['source_triangle_ids'][4], c['initial_source_triangle_ids'][4])
        self.assert_lifts(a, r)
        a['ordered_cell_mode'] = 'fixed'; fixed = conform_to_surface(**a)
        self.assertGreater(np.linalg.norm(fixed['deformed'][4]-a['reference'][4]), .3)

    def test_curved_source_crosses_cells_with_exact_actual_lifts(self):
        a = chart_fan(curved=True); r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['initial'][4]), .4)
        self.assertEqual(r['ordered_report']['charts'][0]['changed_cell_vertices'], [4])
        self.assertEqual(r['ordered_report']['lifted_facing_conflicts'], [])
        self.assert_lifts(a, r)

    def test_metric_and_uv_jacobians_across_cells(self):
        calls = []
        def checked(fun, z, **kwargs):
            # An interior point in a different cell, away from piecewise seams.
            probe = np.array([-.091, .311]); h = 1e-7
            value, gradient = fun(probe)
            numeric = [(fun(probe+np.eye(2)[k]*h)[0]-fun(probe-np.eye(2)[k]*h)[0])/(2*h) for k in range(2)]
            np.testing.assert_allclose(gradient, numeric, atol=1e-7)
            constraint = kwargs['constraints']
            numeric = np.column_stack([(constraint['fun'](probe+np.eye(2)[k]*h)-
                                       constraint['fun'](probe-np.eye(2)[k]*h))/(2*h) for k in range(2)])
            np.testing.assert_allclose(constraint['jac'](probe), numeric, atol=1e-7)
            calls.append(value)
            return minimize(fun, z, **kwargs)
        a = chart_fan(curved=True); a['iterations'] = 1
        with patch('scipy.optimize.minimize', checked): conform_to_surface(**a)
        self.assertEqual(len(calls), 1)

    def test_roundoff_only_initial_realization_even_with_wide_band(self):
        for vertex in (0, 4):
            a = chart_fan(); a['initial'][vertex, 2] += 1e-7
            a.update(distance_tolerance=1e-3, lower=-.01, upper=.01)
            r = conform_to_surface(**a)
            self.assertEqual(r['ordered_report']['status'], 'invalid_initial')
            self.assertIn(vertex, r['ordered_report']['reasons'][0]['vertices'])
            np.testing.assert_array_equal(r['deformed'], a['initial'])

    def test_fixed_interior_and_implicit_chart_boundary_remain_bitwise(self):
        p, tri = plane(0, 1, 0, 1, z=0., columns=7, rows=7)
        # The chart is the central 3x3 subgrid, whose boundary is free in the full patch.
        selected = np.all((p[:, :2] >= .3) & (p[:, :2] <= .7), axis=1)
        faces = np.flatnonzero(selected[tri].all(1)); vertices = np.unique(tri[faces])
        held = np.any((p[:, :2] == 0) | (p[:, :2] == 1), axis=1); held[24] = True
        refs = np.full(len(p), -1, int); bary = np.zeros((len(p), 3))
        for v in vertices:
            f = faces[np.any(tri[faces] == v, axis=1)][0]; refs[v] = f
            bary[v, np.flatnonzero(tri[f] == v)[0]] = 1
        a = dict(reference=p.copy(), initial=p.copy(), triangles=tri, held=held,
            support_positions=p.copy(), support_triangles=tri, support_domains=np.zeros(len(tri), int),
            support_assignment=np.where(selected, 0, -1), qualified_domains=[0], support_weights=selected.astype(float),
            units='m', frame='synthetic', relative_area_tolerance=1e-12, ordered_cell_mode='chart', iterations=10,
            ordered_charts=[dict(source_uv=p[:, :2], support_triangle_ids=faces, candidate_triangle_ids=faces,
                vertex_source_triangles=refs, initial_barycentric=bary, minimum_area_ratio=.1, boundary_mode='fixed')])
        free_outside = ~held & ~selected; a['reference'][free_outside, 2] += .2
        r = conform_to_surface(**a)
        self.assertGreater(np.max(abs(r['deformed'][free_outside]-p[free_outside])), .01)
        self.assert_lifts(a, r)
        np.testing.assert_array_equal(r['deformed'][vertices], p[vertices])
        self.assertTrue(r['ordered_report']['uncovered_triangle_ids'])

    def test_reversed_concave_native_diagonal_is_invalid_despite_valid_uv(self):
        source = np.array([[0., 0, 0], [1, 0, 0], [.2, .2, 0], [0, 1, 0]])
        start = np.vstack([source, [.5, -1, 0], [.5, -.3, 0]])
        rest = start.copy(); rest[2] = [1, 1, 0]
        tri = np.array([[0, 1, 3], [1, 2, 3], [1, 0, 5], [0, 4, 5], [4, 1, 5]])
        r = conform_to_surface(rest, start, tri, np.array([1, 1, 1, 1, 1, 0]), source,
            np.array([[0, 1, 2], [0, 2, 3]]), units='m', frame='synthetic', relative_area_tolerance=1e-12,
            support_domains=np.zeros(2, int), support_assignment=[0, 0, 0, 0, -1, -1], qualified_domains=[0],
            support_weights=[1, 1, 1, 1, 0, 0], ordered_cell_mode='chart', ordered_charts=[dict(
                support_triangle_ids=np.arange(2), candidate_triangle_ids=np.arange(2),
                source_uv=np.array([[0., 0], [1, 0], [1, 1], [0, 1]]),
                vertex_source_triangles=np.array([0, 0, 0, 1, -1, -1]),
                initial_barycentric=np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0], [0, 0, 0]]),
                minimum_area_ratio=.1, boundary_mode='fixed')])
        self.assertEqual(r['ordered_report']['status'], 'invalid_initial')
        self.assertEqual(r['ordered_report']['reasons'][0]['charts'][0]['triangle_ids'], [1])
        np.testing.assert_array_equal(r['deformed'], start)

    def test_invalid_uv_and_missing_mode_declarations_refuse(self):
        a = chart_fan(); a['ordered_charts'][0]['source_uv'][12] = [0, 0]
        with self.assertRaises(ValueError): conform_to_surface(**a)
        a = chart_fan(); del a['ordered_charts']
        with self.assertRaisesRegex(ValueError, 'requires ordered_charts'): conform_to_surface(**a)
        a = chart_fan(); a['ordered_cell_mode'] = 'nearest'
        with self.assertRaisesRegex(ValueError, 'fixed or chart'): conform_to_surface(**a)

    def test_valid_uv_proposal_with_reversed_lifted_face_is_not_accepted(self):
        a = fan(True); a['ordered_cell_mode'] = 'chart'; a['iterations'] = 1
        a['ordered_charts'][0]['source_uv'] = np.array([[0., 0], [1, 0], [1, 1], [0, 1]])
        a['reference'][4] = [.5, .05, 0]
        def reversed_lift(fun, z, **kwargs):
            proposed = np.array([.28, -.15])/np.sqrt(2)
            self.assertGreater(kwargs['constraints']['fun'](proposed).min(), 0)
            return OptimizeResult(x=proposed, success=True, status=0, message='reversed lifted face', nit=1)
        with patch('scipy.optimize.minimize', reversed_lift): r = conform_to_surface(**a)
        step = r['ordered_report']['steps'][0]
        self.assertTrue(step['rejected_lifted_facing'])
        self.assertLess(step['fraction'], 1)
        self.assertEqual(r['ordered_report']['lifted_facing_conflicts'], [])
        self.assert_lifts(a, r)

    def test_outside_trial_extension_and_concave_boundary(self):
        a = fan(True); a['ordered_cell_mode'] = 'chart'; a['iterations'] = 1
        def outside(fun, z, **kwargs):
            self.assertTrue(np.isfinite(fun(np.array([2., 2.]))[0]))
            return OptimizeResult(x=np.array([2., 2.]), success=True, status=0, message='outside chart', nit=1)
        with patch('scipy.optimize.minimize', outside): r = conform_to_surface(**a)
        self.assert_lifts(a, r)
        self.assertLess(r['ordered_report']['steps'][0]['fraction'], 1)

    def test_nearby_other_sheet_never_substitutes_for_chart_owner(self):
        a = chart_fan(curved=True)
        count = len(a['support_positions']); faces = len(a['support_triangles'])
        other = a['support_positions'].copy(); other[:, 2] += 1e-6
        a['support_positions'] = np.vstack([a['support_positions'], other])
        a['support_triangles'] = np.vstack([a['support_triangles'], a['support_triangles']+count])
        a['support_domains'] = np.r_[np.zeros(faces, int), np.ones(faces, int)]
        c = a['ordered_charts'][0]; c['source_uv'] = np.vstack([c['source_uv'], c['source_uv']])
        r = conform_to_surface(**a)
        self.assertTrue((r['assigned_support_triangle'] < faces).all())
        self.assert_lifts(a, r)

    def test_preparation_routes_mode_and_final_projection_preserves_lifts(self):
        from .preparation import prepare_arrays
        a = chart_fan(curved=True); a.update(project_active=True, band_weight=1e-9)
        r = prepare_arrays('conform_to_surface', inputs={}, parameters=a)
        self.assertEqual(r['ordered_report']['cell_mode'], 'chart')
        self.assertEqual(r['public_metrics']['projection_max_move'], 0.)
        self.assert_lifts(a, r)

    def test_nonfinite_and_unsuccessful_zero_steps_are_blocked(self):
        for nonfinite in (True, False):
            def failed(fun, z, **kwargs):
                return OptimizeResult(x=np.full_like(z, np.nan) if nonfinite else z, success=False,
                                      status=8, message='synthetic failed step', nit=1)
            a = chart_fan()
            with patch('scipy.optimize.minimize', failed): r = conform_to_surface(**a)
            self.assertEqual(r['ordered_report']['status'], 'blocked')
            np.testing.assert_allclose(r['deformed'], a['initial'], atol=1e-16)

    def test_iteration_limit_is_not_convergence(self):
        a = chart_fan(); a.update(iterations=1, ordered_max_iterations=1)
        r = conform_to_surface(**a)
        self.assertEqual(r['ordered_report']['status'], 'iteration_limit')
        self.assertFalse(r['public_metrics']['converged'])
        self.assert_lifts(a, r)

    def test_unsafe_uv_proposal_cannot_cross_connected_area_bound(self):
        def oversized(fun, z, **kwargs):
            return OptimizeResult(x=np.array([.7, -.2]), success=True, status=0, message='oversized', nit=1)
        a = chart_fan(); a['reference'][4] = [.95, .05, 0]; a['iterations'] = 1
        with patch('scipy.optimize.minimize', oversized): r = conform_to_surface(**a)
        self.assertLess(r['ordered_report']['steps'][0]['fraction'], 1)
        self.assertFalse(r['public_metrics']['converged'])
        self.assert_lifts(a, r)

    def test_scale_translation_and_reversed_winding(self):
        for scale, translation in ((1e-4, 0.), (1., 10000.)):
            a = chart_fan(scale=scale, translation=translation)
            a['support_triangles'] = a['support_triangles'][:, ::-1].copy()
            a['triangles'] = a['triangles'][:, ::-1].copy()
            c = a['ordered_charts'][0]; c['initial_barycentric'] = c['initial_barycentric'][:, ::-1].copy()
            r = conform_to_surface(**a)
            self.assertTrue(r['ordered_report']['feasible'], r['ordered_report']['reasons'])
            np.testing.assert_allclose((r['deformed'][4]-translation)/scale, [.2, .7, 0], atol=3e-6)
            self.assert_lifts(a, r)

    def test_revision_binds_mode_source_and_solver_limits(self):
        a = chart_fan(); first = conform_to_surface(**a)['ordered_report']['input_revision']
        b = copy.deepcopy(a); b['ordered_cell_mode'] = 'fixed'
        second = conform_to_surface(**b)['ordered_report']['input_revision']
        a['support_positions'][6, 2] += .001
        third = conform_to_surface(**a)['ordered_report']['input_revision']
        a['ordered_max_iterations'] = 10
        fourth = conform_to_surface(**a)['ordered_report']['input_revision']
        self.assertEqual(len({first, second, third, fourth}), 4)


if __name__ == '__main__':
    unittest.main()
