"""Synthetic connected correspondence constraints inside the surface fitter."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from scipy.optimize import OptimizeResult
from .metric_fitting import conform_to_surface
from .ordered_fitting import _safe_fraction
from .preparation import prepare_arrays


def fan(concave=False, scale=1.):
    source = np.array([[0., 0, 0], [1, 0, 0], [.2, .2, 0] if concave else [1, 1, 0], [0, 1, 0]])
    st = np.array([[0, 1, 2], [0, 2, 3]])
    start = np.vstack([source, [.1, .08, 0] if concave else [.35, .2, 0]])
    tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    rest = start.copy(); rest[2] = [1, 1, 0]; rest[4] = [.65, .05 if concave else .25, 0]
    refs = np.array([0, 0, 0, 1, 0])
    bary = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [.58, .02, .4] if concave else [.65, .15, .2]])
    chart = dict(support_triangle_ids=np.arange(2), source_uv=source[:, :2].copy(),
        candidate_triangle_ids=np.arange(4), vertex_source_triangles=refs, initial_barycentric=bary,
        minimum_area_ratio=.2, boundary_mode='fixed')
    return dict(reference=rest*scale, initial=start*scale, triangles=tri, held=np.array([1, 1, 1, 1, 0]),
        support_positions=source*scale, support_triangles=st, support_domains=np.zeros(2, int),
        support_assignment=np.zeros(5, int), qualified_domains=[0], units='m', frame='synthetic',
        relative_area_tolerance=1e-12, distance_tolerance=1e-9*scale, position_tolerance=1e-8*scale,
        iterations=40, ordered_charts=[chart])


def signed_area(p, tri):
    a, b = p[tri[:, 1], :2]-p[tri[:, 0], :2], p[tri[:, 2], :2]-p[tri[:, 0], :2]
    return (a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/2


class OrderedFittingTests(unittest.TestCase):
    def test_interior_redistributes_without_freezing_material_indices(self):
        a = fan(); r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        self.assertTrue(r['public_metrics']['converged'])
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['initial'][4]), .2)
        np.testing.assert_allclose(r['deformed'][4], a['reference'][4], atol=2e-5)
        np.testing.assert_array_equal(r['deformed'][:4], a['initial'][:4])
        c = r['ordered_report']['charts'][0]
        b = np.asarray(c['barycentric']); cells = a['support_triangles'][c['source_triangle_ids']]
        np.testing.assert_allclose(r['deformed'], np.einsum('nk,nkd->nd', b, a['support_positions'][cells]), atol=1e-14)
        self.assertGreaterEqual(c['minimum_area_ratio_after'], .2-1e-9)

    def test_connected_faces_stop_overrunning_a_fixed_concave_collar(self):
        a = fan(True); r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        before = signed_area(a['initial'], a['triangles']); after = signed_area(r['deformed'], a['triangles'])
        self.assertGreaterEqual(float((after/before).min()), .2-1e-9)
        # This tests connected face order, beyond point membership in the source cell.
        cell_only = np.array([.5, .05, 0.]); trial = a['initial'].copy(); trial[4] = cell_only
        self.assertGreaterEqual(min([1-cell_only[0]-4*cell_only[1], cell_only[0]-cell_only[1], 5*cell_only[1]]), 0)
        self.assertLess(float(signed_area(trial, a['triangles']).min()), 0)
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['initial'][4]), .001)
        self.assertTrue(r['ordered_report']['steps'])
        old = conform_to_surface(**{k: v for k, v in a.items() if k != 'ordered_charts'})
        self.assertLess(float(signed_area(old['deformed'], a['triangles']).min()), 0)
        self.assertEqual(old['public_metrics']['beyond_boundary_after'], 0)
        self.assertEqual(old['assignment_report']['status'], 'measured')

    def test_area_bound_is_hard_even_when_rest_energy_wants_compression(self):
        a = fan(); a['reference'][4] = [.9, .01, 0]
        a['ordered_charts'][0]['minimum_area_ratio'] = .5
        r = conform_to_surface(**a)
        ratio = signed_area(r['deformed'], a['triangles'])/signed_area(a['initial'], a['triangles'])
        self.assertGreaterEqual(float(ratio.min()), .5-1e-9)
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['reference'][4]), .05)

    def test_bad_initial_correspondence_is_reported_without_a_deformation(self):
        for kind in ('off_source', 'held_conflict', 'reversal', 'band'):
            a = fan()
            if kind == 'off_source': a['initial'][4, 2] = .01
            if kind == 'held_conflict': a['initial'][0, 2] = .01
            if kind == 'reversal':
                a = fan(True); a['initial'][4] = [.5, .05, 0]
                a['ordered_charts'][0]['initial_barycentric'][4] = [.3, .45, .25]
            if kind == 'band': a.update(lower=.01, upper=.02)
            r = conform_to_surface(**a)
            self.assertEqual(r['ordered_report']['status'], 'invalid_initial', kind)
            self.assertFalse(r['ordered_report']['feasible'])
            self.assertFalse(r['public_metrics']['converged'])
            self.assertEqual(r['public_metrics']['iterations'], 0)
            np.testing.assert_array_equal(r['deformed'], a['initial'])

    def test_wrong_owner_missing_cells_and_boundary_policy_refuse(self):
        for kind in ('owner', 'cells', 'boundary', 'unassigned'):
            a = fan()
            if kind == 'owner': a['support_assignment'][4] = -1
            if kind == 'cells': a['ordered_charts'][0]['vertex_source_triangles'][4] = -1
            if kind == 'boundary': a['ordered_charts'][0]['boundary_mode'] = 'infer'
            if kind == 'unassigned':
                for k in ('support_domains', 'support_assignment', 'qualified_domains'): a.pop(k)
            with self.assertRaises(ValueError, msg=kind): conform_to_surface(**a)

    def test_failure_to_find_a_step_is_blocked_not_converged(self):
        def blocked(fun, z, **kwargs):
            return OptimizeResult(x=z.copy(), success=False, status=8, message='synthetic blocked search', nit=1)
        a = fan()
        with patch('scipy.optimize.minimize', blocked): r = conform_to_surface(**a)
        self.assertEqual(r['ordered_report']['status'], 'blocked')
        self.assertFalse(r['public_metrics']['converged'])
        self.assertTrue(r['ordered_report']['feasible'])
        np.testing.assert_array_equal(r['deformed'], a['initial'])

    def test_iteration_limit_is_not_convergence_or_global_infeasibility(self):
        a = fan(); a['iterations'] = 1
        r = conform_to_surface(**a)
        self.assertEqual(r['ordered_report']['status'], 'iteration_limit')
        self.assertFalse(r['public_metrics']['converged'])
        self.assertTrue(r['ordered_report']['feasible'])

    def test_safe_advance_checks_between_feasible_endpoints(self):
        # Positive at both endpoints but negative in the middle: endpoint checks miss this.
        t = _safe_fraction(np.array([4.]), np.array([-4.]), np.array([.5]))
        self.assertGreater(t, .1); self.assertLess(t, .15)
        self.assertGreater(.5-4*t+4*t*t, 0)

    def test_units_and_reversed_source_winding_do_not_change_the_fit(self):
        a = fan(); baseline = conform_to_surface(**a)
        tiny = conform_to_surface(**fan(scale=.001))
        np.testing.assert_allclose(tiny['deformed']/.001, baseline['deformed'], atol=1e-6)
        a['support_triangles'] = a['support_triangles'][:, ::-1]
        a['triangles'] = a['triangles'][:, ::-1]
        a['ordered_charts'][0]['initial_barycentric'] = a['ordered_charts'][0]['initial_barycentric'][:, ::-1]
        r = conform_to_surface(**a)
        np.testing.assert_allclose(r['deformed'], baseline['deformed'], atol=1e-6)

    def test_preparation_and_projection_use_the_same_order_constraints(self):
        a = fan(); a.update(band_weight=1e-9, project_active=True)
        r = prepare_arrays('conform_to_surface', inputs={}, parameters=a)
        self.assertTrue(r['ordered_report']['feasible'])
        self.assertEqual(r['public_metrics']['projection_max_move'], 0.)
        self.assertEqual(r['ordered_report']['uncovered_triangle_ids'], [])
        self.assertEqual(len(r['ordered_report']['charts'][0]['array_sha256']['initial_barycentric']), 64)

    def test_curved_source_realization_keeps_explicit_barycentric_cells(self):
        a = fan()
        # Two facets meeting at an oblique ridge; supplied UV is not an ambient XY projection.
        a['support_positions'][2, 2] = .4
        c = a['ordered_charts'][0]
        cells = a['support_triangles'][c['vertex_source_triangles']]
        a['initial'] = np.einsum('nk,nkd->nd', c['initial_barycentric'], a['support_positions'][cells])
        a['reference'] = a['initial'].copy(); a['reference'][4] = [.65, .25, .1]
        r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        b = np.asarray(r['ordered_report']['charts'][0]['barycentric'])
        np.testing.assert_allclose(r['deformed'], np.einsum('nk,nkd->nd', b, a['support_positions'][cells]), atol=1e-13)
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['initial'][4]), .1)

    def test_cells_limit_the_requested_move_without_claiming_the_endpoint(self):
        a = fan(); a['reference'][4] = [.2, .7, 0]
        r = conform_to_surface(**a)
        # The requested rest center lies across the fixed source-cell edge. The declared
        # solve can settle on that edge, but must disclose the cells and actual result.
        self.assertGreaterEqual(r['deformed'][4, 0]-r['deformed'][4, 1], -1e-10)
        self.assertGreater(np.linalg.norm(r['deformed'][4]-a['reference'][4]), .2)
        self.assertEqual(r['ordered_report']['charts'][0]['source_triangle_ids'][4], 0)

    def test_numeric_failure_cannot_escape_as_a_successful_fit(self):
        def failed(fun, z, **kwargs):
            return OptimizeResult(x=np.full_like(z, np.nan), success=False, status=8, message='nonfinite', nit=1)
        a = fan()
        with patch('scipy.optimize.minimize', failed): r = conform_to_surface(**a)
        self.assertEqual(r['ordered_report']['status'], 'blocked')
        np.testing.assert_array_equal(r['deformed'], a['initial'])

    def test_overlapping_chart_ownership_and_ambiguous_source_uv_refuse(self):
        a = fan(); a['ordered_charts'].append(copy.deepcopy(a['ordered_charts'][0]))
        with self.assertRaisesRegex(ValueError, 'share candidate vertices'): conform_to_surface(**a)
        a = fan(); a['ordered_charts'][0]['source_uv'][3] = [1, 0]
        with self.assertRaises(ValueError): conform_to_surface(**a)

    def test_disjoint_charts_and_unlisted_bridge_share_the_coupled_fit(self):
        from .test_metric_fitting import plane
        p, tri = plane(0, 6, 0, 4, z=0., columns=7, rows=5)
        p[:, 2] = .1*np.maximum(p[:, 0]-3, 0)
        held = (p[:, 0] == 0) | (p[:, 0] == 6) | (p[:, 1] == 0) | (p[:, 1] == 4)
        a = dict(reference=p.copy(), initial=p.copy(), triangles=tri, held=held,
            support_positions=p.copy(), support_triangles=tri, support_domains=np.zeros(len(tri), int),
            support_assignment=np.zeros(len(p), int), qualified_domains=[0], support_weights=np.zeros(len(p)),
            units='m', frame='synthetic', relative_area_tolerance=1e-12, iterations=15, ordered_charts=[])
        for choose in (p[:, 0] <= 2, p[:, 0] >= 4):
            faces = np.flatnonzero(choose[tri].all(1)); vertices = np.unique(tri[faces])
            refs = np.full(len(p), -1, int); bary = np.zeros((len(p), 3))
            for v in vertices:
                f = int(faces[np.any(tri[faces] == v, axis=1)][0]); refs[v] = f
                bary[v, int(np.flatnonzero(tri[f] == v)[0])] = 1
            a['support_weights'][vertices] = 1
            a['ordered_charts'].append(dict(support_triangle_ids=faces, candidate_triangle_ids=faces,
                source_uv=p[:, :2].copy(), vertex_source_triangles=refs, initial_barycentric=bary,
                minimum_area_ratio=.1, boundary_mode='fixed'))
        bridge = (~held) & (p[:, 0] == 3); a['reference'][bridge, 2] += .3
        r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        self.assertGreater(float(np.max(abs(r['deformed'][bridge]-p[bridge]))), .01, r['ordered_report']['steps'])
        self.assertEqual(len(r['ordered_report']['charts']), 2)
        self.assertTrue(r['ordered_report']['uncovered_triangle_ids'])
        for c in r['ordered_report']['charts']:
            ids = c['fixed_boundary_vertices']; np.testing.assert_array_equal(r['deformed'][ids], p[ids])
            self.assertGreaterEqual(c['minimum_area_ratio_after'], .1-1e-9)

    def test_unsafe_optimizer_proposal_is_limited_before_face_collapse(self):
        def oversized(fun, z, **kwargs):
            return OptimizeResult(x=np.array([.7, -.2]), success=True, status=0, message='oversized test step', nit=1)
        a = fan(); a['reference'][4] = [.95, .05, 0]; a['iterations'] = 1
        with patch('scipy.optimize.minimize', oversized): r = conform_to_surface(**a)
        self.assertTrue(r['ordered_report']['feasible'])
        self.assertLess(r['ordered_report']['steps'][0]['fraction'], 1)
        self.assertGreaterEqual(r['ordered_report']['charts'][0]['minimum_area_ratio_after'], .2-1e-10)
        self.assertFalse(r['public_metrics']['converged'])

    def test_report_revision_binds_cells_area_bounds_and_solver_parameters(self):
        a = fan(); first = conform_to_surface(**a)['ordered_report']['input_revision']
        a['ordered_charts'][0]['minimum_area_ratio'] = .3
        second = conform_to_surface(**a)['ordered_report']['input_revision']
        a['ordered_max_iterations'] = 20
        third = conform_to_surface(**a)['ordered_report']['input_revision']
        self.assertEqual(len({first, second, third}), 3)


if __name__ == '__main__':
    unittest.main()
