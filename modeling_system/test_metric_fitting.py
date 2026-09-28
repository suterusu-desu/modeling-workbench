"""Planar and surface metric invariants and ordinary preparation integration, without IO effects."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from scipy.sparse import coo_matrix, diags
from .metric_fitting import (planar_fem_metric, surface_fem_metric, relax_displacement, rigid_deform, conform_to_surface,
                             planar_relayout, projected_boundary, smooth_region)
from .geometry import closest_points, nearest_surface
from .preparation import ArrayPreparation


class PlanarMetricTests(unittest.TestCase):
    def setUp(self):
        # Nonuniform interior station: an index average does not reproduce planes.
        self.points = np.array([[0., 0], [2, 0], [2, 1], [0, 1], [.37, .29]])
        self.triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
        self.parameters = {'units': 'm', 'frame': 'recorded orthonormal chart', 'relative_area_tolerance': 1e-12}

    def metric(self, points=None, triangles=None):
        return planar_fem_metric(self.points if points is None else points,
            self.triangles if triangles is None else triangles, **self.parameters)

    def matrix(self, result):
        return coo_matrix((result['stiffness_values'], (result['stiffness_rows'],
                          result['stiffness_columns'])), shape=result['stiffness_shape']).tocsr()

    def test_nonuniform_chart_reproduces_affine_interior_and_keeps_boundary_flux(self):
        result = self.metric(); k = self.matrix(result)
        self.assertGreater(np.linalg.norm(self.points[4]-self.points[:4].mean(0)), .5)
        np.testing.assert_allclose(k @ np.ones(5), 0, atol=2e-15)
        np.testing.assert_allclose((k @ self.points)[4], 0, atol=2e-15)
        self.assertGreater(np.linalg.norm((k @ self.points)[:4]), 1)
        np.testing.assert_array_equal(result['interior_vertices'], [4])
        np.testing.assert_array_equal(result['boundary_vertices'], [0, 1, 2, 3])

    def test_affine_energy_matches_integral_and_mass_matches_area(self):
        result = self.metric(); k = self.matrix(result)
        field = 2*self.points[:, 0]-3*self.points[:, 1]+7
        self.assertAlmostEqual(float(field @ k @ field), 2*(2**2+3**2), places=12)
        self.assertAlmostEqual(result['lumped_mass'].sum(), 2)
        np.testing.assert_allclose(k.toarray(), k.toarray().T, atol=1e-15)
        self.assertGreaterEqual(np.linalg.eigvalsh(k.toarray()).min(), -1e-14)

    def test_changing_length_units_preserves_stiffness_and_scales_mass_and_bending(self):
        first, second = self.metric(), self.metric(points=self.points*1000)
        a, b = self.matrix(first), self.matrix(second)
        np.testing.assert_allclose(a.toarray(), b.toarray(), rtol=2e-14, atol=1e-14)
        np.testing.assert_allclose(second['lumped_mass'], first['lumped_mass']*1e6)
        qa = a.T @ diags(1/first['lumped_mass']) @ a
        qb = b.T @ diags(1/second['lumped_mass']) @ b
        np.testing.assert_allclose(qb.toarray(), qa.toarray()/1e6, rtol=3e-14, atol=1e-18)

    def test_rigid_chart_transform_and_triangle_winding_do_not_change_metric(self):
        reference = self.matrix(self.metric()).toarray()
        rotation = np.array([[.6, -.8], [.8, .6]])
        triangles = self.triangles.copy(); triangles[1] = triangles[1, ::-1]
        result = self.metric(points=self.points @ rotation.T+[20, -50], triangles=triangles)
        np.testing.assert_allclose(self.matrix(result).toarray(), reference, rtol=2e-13, atol=2e-13)
        self.assertEqual(result['public_metrics']['orientation_counts'], [3, 1])

    def test_dirichlet_affine_field_is_recovered_without_invented_offset(self):
        result = self.metric(); k = self.matrix(result).toarray()
        field = .7*self.points[:, 0]-.4*self.points[:, 1]+3
        value = -k[4, :4] @ field[:4]/k[4, 4]
        self.assertAlmostEqual(value, field[4], places=13)

    def test_no_interior_reports_unknown_affine_check_not_a_pass(self):
        result = self.metric(points=self.points[:3], triangles=[[0, 1, 2]])
        self.assertIsNone(result['public_metrics']['affine_interior_defect'])
        self.assertEqual(result['interior_affine_load'].shape, (0, 2))

    def test_obtuse_elements_keep_signed_weights_and_valid_energy(self):
        result = self.metric(points=[[0., 0], [2., 0], [.1, .1]], triangles=[[0, 1, 2]])
        self.assertGreater(result['public_metrics']['positive_offdiagonal_entries'], 0)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.matrix(result).toarray()).min(), -1e-13)

    def test_invalid_local_domains_refuse_without_inventing_adjacency(self):
        cases = [
            (self.points, np.vstack([self.triangles, self.triangles[0, ::-1]]), 'Duplicate'),
            (np.vstack([self.points, [9, 9]]), self.triangles, 'positive mass'),
            ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]], 'Degenerate'),
            ([[0, 0], [1, 0], [0, 1], [1, 1]], [[0, 1, 2], [1, 0, 3]], 'overlap'),
            ([[0, 0], [1, 0], [0, 1], [0, -1], [1, 2]], [[0, 1, 2], [1, 0, 3], [0, 1, 4]], 'Nonmanifold')]
        for points, triangles, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.metric(points=points, triangles=triangles)

    def test_units_frame_and_scale_free_condition_limit_are_explicit(self):
        for changed in ({'units': ''}, {'frame': ''}, {'relative_area_tolerance': 0},
                        {'relative_area_tolerance': float('nan')}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                planar_fem_metric(self.points, self.triangles, **{**self.parameters, **changed})
        thin = [[0., 0], [1., 0], [.5, 1e-14]]
        for scale in (1e-3, 1., 1e3):
            with self.assertRaisesRegex(ValueError, 'ill-conditioned'):
                self.metric(points=np.asarray(thin)*scale, triangles=[[0, 1, 2]])

    def test_preparation_persists_metric_arrays_and_publishes_only_named_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.npz'
            np.savez(source, chart=self.points, triangles=self.triangles)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'planar_fem_metric', 'parameters': self.parameters,
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                               for name in ('chart', 'triangles')}}}
            result = ArrayPreparation(root/'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertLess(result['summary']['affine_interior_defect'], 1e-12)
            detail = json.loads(Path(result['prepared']['result.json']['path']).read_text())
            with np.load(result['prepared']['arrays.npz']['path'], allow_pickle=False) as arrays:
                self.assertIn('lumped_mass', arrays)
                matrix = coo_matrix((arrays['stiffness_values'],
                    (arrays['stiffness_rows'], arrays['stiffness_columns'])), shape=detail['stiffness_shape'])
                np.testing.assert_allclose(matrix.toarray(), self.matrix(self.metric()).toarray())
            self.assertEqual(detail['metric']['mass_units'], 'length^2')
            self.assertNotIn(str(source), result['workbench']['findings'][0]['summary'])


class SurfaceMetricTests(unittest.TestCase):
    parameters = {'units': 'm', 'frame': 'recorded world frame', 'relative_area_tolerance': 1e-12}

    def matrix(self, result):
        return coo_matrix((result['stiffness_values'], (result['stiffness_rows'],
                          result['stiffness_columns'])), shape=result['stiffness_shape']).toarray()

    @staticmethod
    def grid(nx=5, ny=4, spacing=(.3, .25), jitter=(.03, -.02)):
        # Nonuniform station offsets keep this from being an index-regular lattice.
        x, y = np.meshgrid(np.arange(nx) * spacing[0], np.arange(ny) * spacing[1], indexing='ij')
        points = np.c_[x.ravel(), y.ravel()]
        points[(np.arange(len(points)) % 3) == 1] += jitter
        index = np.arange(nx * ny).reshape(nx, ny); triangles = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                a, b, c, d = index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]
                triangles += [[a, b, c], [a, c, d]]
        return points, np.array(triangles), index

    def test_flat_patch_in_space_matches_the_planar_metric(self):
        chart, triangles, _ = self.grid()
        rotation = np.linalg.qr(np.array([[.3, -.8, .5], [.9, .2, -.1], [.1, .6, .8]]))[0]
        positions = np.c_[chart, np.zeros(len(chart))] @ rotation.T + [5., -2., 9.]
        planar, surface = planar_fem_metric(chart, triangles, **self.parameters), surface_fem_metric(positions, triangles, **self.parameters)
        np.testing.assert_allclose(self.matrix(surface), self.matrix(planar), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(surface['lumped_mass'], planar['lumped_mass'], rtol=1e-12)
        self.assertLess(surface['public_metrics']['interior_mean_curvature_max'], 1e-9)
        np.testing.assert_array_equal(surface['interior_vertices'], planar['interior_vertices'])

    def test_isometric_fold_keeps_the_intrinsic_metric_while_a_projection_distorts(self):
        chart, triangles, index = self.grid(jitter=(0., -.02))   # straight columns: the crease follows grid edges
        flat = np.c_[chart, np.zeros(len(chart))]
        folded = flat.copy(); crease = chart[index[2, 0], 0]
        side = chart[:, 0] > crease + 1e-12; angle = np.radians(70)
        dx = chart[side, 0] - crease
        folded[side, 0] = crease + dx * np.cos(angle); folded[side, 2] = dx * np.sin(angle)
        # The crease runs along grid edges, so no triangle straddles it: the fold is isometric.
        self.assertTrue(np.all((chart[triangles, 0] <= crease + 1e-12).all(1) | (chart[triangles, 0] >= crease - 1e-12).all(1)))
        before, after = surface_fem_metric(flat, triangles, **self.parameters), surface_fem_metric(folded, triangles, **self.parameters)
        np.testing.assert_allclose(self.matrix(after), self.matrix(before), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(after['lumped_mass'], before['lumped_mass'], rtol=1e-12)
        projected = planar_fem_metric(folded[:, :2], triangles, **self.parameters)
        self.assertGreater(np.abs(self.matrix(projected) - self.matrix(before)).max(), .1)
        self.assertLess(projected['lumped_mass'].sum(), before['lumped_mass'].sum() * .8)

    def test_sphere_patch_reports_curvature_instead_of_a_reproduction_error(self):
        chart, triangles, _ = self.grid(9, 9, (.05, .05))
        radius = 2.; chart = chart - chart.mean(0)
        direction = np.c_[chart, np.full(len(chart), radius)]
        positions = radius * direction / np.linalg.norm(direction, axis=1, keepdims=True)
        result = surface_fem_metric(positions, triangles, **self.parameters)
        curvature = result['public_metrics']['interior_mean_curvature_max']
        self.assertAlmostEqual(curvature, 1 / radius, delta=.15 / radius)
        np.testing.assert_allclose(self.matrix(result) @ np.ones(len(positions)), 0, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(self.matrix(result)).min(), -1e-12)

    def test_mixed_winding_is_reported_not_refused(self):
        chart, triangles, _ = self.grid()
        positions = np.c_[chart, .1 * chart[:, 0] ** 2]
        flipped = triangles.copy(); flipped[3] = flipped[3, ::-1]
        a, b = surface_fem_metric(positions, triangles, **self.parameters), surface_fem_metric(positions, flipped, **self.parameters)
        np.testing.assert_allclose(self.matrix(b), self.matrix(a), rtol=1e-12, atol=1e-13)
        self.assertEqual(a['public_metrics']['inconsistently_wound_edges'], 0)
        self.assertGreater(b['public_metrics']['inconsistently_wound_edges'], 0)

    def test_invalid_surfaces_refuse(self):
        chart, triangles, _ = self.grid()
        positions = np.c_[chart, np.zeros(len(chart))]
        cases = [(chart, triangles, 'Finite 3D'),
                 (positions, np.vstack([triangles, triangles[0, ::-1]]), 'Duplicate'),
                 ([[0., 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], 'Degenerate'),
                 ([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], [[0, 1, 2], [1, 0, 3], [0, 1, 4]], 'Nonmanifold')]
        for points, tri, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                surface_fem_metric(points, tri, **self.parameters)
        with self.assertRaises(ValueError):
            surface_fem_metric(positions, triangles, **{**self.parameters, 'units': ''})

    def test_preparation_route_runs_the_surface_metric(self):
        chart, triangles, _ = self.grid()
        positions = np.c_[chart, .2 * chart[:, 1] ** 2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, positions=positions, triangles=triangles)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'surface_fem_metric', 'parameters': self.parameters,
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                               for name in ('positions', 'triangles')}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            detail = json.loads(Path(result['prepared']['result.json']['path']).read_text())
            self.assertEqual(detail['metric']['type'], 'intrinsic_surface_p1')
            with np.load(result['prepared']['arrays.npz']['path'], allow_pickle=False) as arrays:
                np.testing.assert_allclose(arrays['lumped_mass'], surface_fem_metric(positions, triangles, **self.parameters)['lumped_mass'])


class RelaxDisplacementTests(unittest.TestCase):
    parameters = {'units': 'm', 'frame': 'recorded world frame', 'relative_area_tolerance': 1e-12}

    def patch(self, n=9, curved=True):
        chart, triangles, index = SurfaceMetricTests.grid(n, n, (.1, .1), jitter=(.012, -.008))
        rest = np.c_[chart, .15 * chart[:, 0] ** 2 if curved else np.zeros(len(chart))]   # gently curved sheet
        rings = np.zeros(len(rest), bool)
        rings[index[:2].ravel()] = rings[index[-2:].ravel()] = rings[index[:, :2].ravel()] = rings[index[:, -2:].ravel()] = True
        return rest, triangles, index, rings

    def test_affine_displacement_on_a_flat_patch_is_reproduced_and_held_vertices_stay_exact(self):
        # Ambient affine fields are harmonic on a flat patch in any orientation; on a curved patch
        # the surface Laplacian of ambient coordinates is the curvature normal, so only approximately.
        rest, triangles, _, held = self.patch(curved=False)
        rotation = np.linalg.qr(np.array([[.3, -.8, .5], [.9, .2, -.1], [.1, .6, .8]]))[0]
        rest = rest @ rotation.T
        matrix = np.array([[.9, .1, 0], [-.05, 1.1, .02], [.03, 0, 1.]])
        posed = rest @ matrix.T + [.2, -.1, .05]
        result = relax_displacement(rest, posed, triangles, held, **self.parameters)
        np.testing.assert_allclose(result['relaxed'], posed, atol=1e-10)
        np.testing.assert_array_equal(result['delta'][held], 0)

    def test_crowded_interior_is_relieved_without_moving_held_material(self):
        rest, triangles, index, held = self.patch()
        posed = rest.copy(); centre = index[4, 4]
        # Crowd the interior toward one station: a sharp tangential pinch, the boundary unchanged.
        distance = np.linalg.norm(rest[:, :2] - rest[centre, :2], axis=1)
        pull = .85 * np.exp(-(distance / .15) ** 2)[:, None] * (rest[centre] - rest)
        posed[~held] += pull[~held]
        result = relax_displacement(rest, posed, triangles, held, **self.parameters)
        metrics = result['public_metrics']
        self.assertLess(metrics['compressed_after'], metrics['compressed_before'])
        self.assertGreater(metrics['smallest_stretch_q01_after'], metrics['smallest_stretch_q01_before'])
        np.testing.assert_array_equal(result['relaxed'][held], posed[held])

    def test_region_of_a_larger_mesh_leaves_other_vertices_untouched(self):
        rest, triangles, index, held = self.patch()
        extra = np.array([[5., 5, 5], [6, 5, 5], [5, 6, 5]])
        full_rest, full_posed = np.vstack([rest, extra]), np.vstack([rest * 1.01, extra + .3])
        mask = np.r_[held, np.zeros(3, bool)]                      # unused vertices need no flag value
        result = relax_displacement(full_rest, full_posed, triangles, mask, **self.parameters)
        np.testing.assert_array_equal(result['delta'][-3:], 0)
        self.assertTrue(np.all(np.isin(result['free_vertices'], np.unique(triangles))))

    def test_free_boundary_vertex_refuses(self):
        rest, triangles, index, held = self.patch()
        loose = held.copy(); loose[index[0, 4]] = False
        with self.assertRaisesRegex(ValueError, 'interior'):
            relax_displacement(rest, rest, triangles, loose, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'held flag'):
            relax_displacement(rest, rest, triangles, held[:-1], **self.parameters)

    def test_preparation_route_relaxes_with_a_held_array(self):
        rest, triangles, index, held = self.patch()
        posed = rest.copy(); posed[index[4, 4]] += [.03, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, reference=rest, deformed=posed, triangles=triangles, held=held.astype(np.int8))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'relax_displacement', 'parameters': self.parameters,
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                               for name in ('reference', 'deformed', 'triangles', 'held')}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertIn('compressed_after', result['summary'])


class RigidDeformTests(unittest.TestCase):
    parameters = RelaxDisplacementTests.parameters

    def folded_sheet(self, degrees):
        """A flat sheet whose held boundary is folded along x = median by the given angle (a closing lid)."""
        rest, triangles, index, held = RelaxDisplacementTests().patch(n=13, curved=False)
        middle = np.median(rest[:, 0]); angle = np.radians(degrees)
        folded = rest.copy(); right = rest[:, 0] > middle; lever = rest[right, 0] - middle
        folded[right, 0] = middle + lever * np.cos(angle); folded[right, 2] = lever * np.sin(angle)
        initial = rest.copy(); initial[held] = folded[held]
        return rest, triangles, held, initial

    def test_rigid_motion_of_all_handles_is_followed_exactly(self):
        rest, triangles, _, held = RelaxDisplacementTests().patch()
        rotation = np.linalg.qr(np.array([[.3, -.8, .5], [.9, .2, -.1], [.1, .6, .8]]))[0]
        moved = rest @ rotation.T + [.2, -.1, .05]
        initial = rest.copy(); initial[held] = moved[held]
        result = rigid_deform(rest, initial, triangles, held, iterations=400, **self.parameters)
        np.testing.assert_allclose(result['deformed'], moved, atol=1e-6)
        np.testing.assert_array_equal(result['deformed'][held], initial[held])
        self.assertLess(result['public_metrics']['energy_last'], 1e-10)

    def test_folding_handles_bend_the_sheet_where_linear_interpolation_crowds(self):
        # The fold rotates the right half by 150 degrees: the linear displacement interpolation
        # shortens the chord across the bend, the rotation-aware solve keeps lengths.
        rest, triangles, held, initial = self.folded_sheet(150)
        linear = relax_displacement(rest, initial, triangles, held, **self.parameters)
        result = rigid_deform(rest, linear['relaxed'], triangles, held, iterations=300, **self.parameters)
        metrics, crowded = result['public_metrics'], linear['public_metrics']
        self.assertGreater(metrics['smallest_stretch_q01_after'], crowded['smallest_stretch_q01_after'] + .3)
        self.assertGreater(crowded['compressed_after'], 0); self.assertEqual(metrics['compressed_after'], 0)
        self.assertTrue(metrics['converged']); self.assertLess(result['energies'][-1], result['energies'][0])

    def test_soft_targets_pull_free_vertices_and_zero_weight_changes_nothing(self):
        rest, triangles, held, initial = self.folded_sheet(90)
        plain = rigid_deform(rest, initial, triangles, held, iterations=40, **self.parameters)['deformed']
        zero = rigid_deform(rest, initial, triangles, held, iterations=40, targets=rest, target_weights=np.zeros(len(rest)),
                            **self.parameters)['deformed']
        np.testing.assert_allclose(zero, plain, atol=1e-12)
        goal = rest + [0, 0, .05]
        pulled = rigid_deform(rest, initial, triangles, held, iterations=60, targets=goal, target_weights=np.full(len(rest), 1e6),
                              **self.parameters)
        free = ~held
        np.testing.assert_allclose(pulled['deformed'][free], goal[free], atol=1e-4)
        np.testing.assert_array_equal(pulled['deformed'][held], initial[held])
        self.assertEqual(pulled['public_metrics']['soft_target_vertices'], int(free.sum()))
        with self.assertRaisesRegex(ValueError, 'both targets'):
            rigid_deform(rest, initial, triangles, held, targets=goal, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'nonnegative weight'):
            rigid_deform(rest, initial, triangles, held, targets=goal, target_weights=-np.ones(len(rest)), **self.parameters)

    def test_intervals_keep_free_vertices_inside_and_loose_intervals_change_nothing(self):
        rest, triangles, held, initial = self.folded_sheet(90)
        free, n = ~held, len(rest)
        plain = rigid_deform(rest, initial, triangles, held, iterations=60, **self.parameters)
        loose = rigid_deform(rest, initial, triangles, held, iterations=60, interval_axis=2, lower=plain['deformed'][:, 2] - 1,
                             upper=np.full(n, np.nan), **self.parameters)
        np.testing.assert_allclose(loose['deformed'], plain['deformed'], atol=1e-12)
        self.assertEqual(loose['public_metrics']['interval_active'], 0)
        ceiling = .02
        self.assertGreater(plain['deformed'][free, 2].max(), ceiling + .01)
        capped = rigid_deform(rest, initial, triangles, held, iterations=200, interval_axis=2, lower=np.full(n, -np.inf),
                              upper=np.full(n, ceiling), **self.parameters)
        metrics = capped['public_metrics']
        self.assertLessEqual(capped['deformed'][free, 2].max(), ceiling)
        np.testing.assert_array_equal(capped['deformed'][held], initial[held])
        self.assertGreater(metrics['interval_active'], 0); self.assertEqual(metrics['interval_outside_after'], 0)
        self.assertEqual(metrics['interval_vertices'], int(free.sum()))
        self.assertLess(metrics['interval_penalty_residual'], 1e-3)
        # Only the bounded coordinate is pulled: the free material still follows the fold sideways.
        self.assertGreater(np.abs(capped['deformed'][free, 0] - rest[free, 0]).max(), .05)
        # A soft band without projection pulls toward it but the shape term keeps the material smooth.
        soft = rigid_deform(rest, initial, triangles, held, iterations=200, interval_axis=2, lower=np.full(n, -np.inf),
                            upper=np.full(n, ceiling), interval_weight=.5, project_active=False, **self.parameters)
        top = soft['deformed'][free, 2].max()
        self.assertLess(top, plain['deformed'][free, 2].max()); self.assertGreater(top, ceiling)
        self.assertFalse(soft['public_metrics']['interval_projected'])
        self.assertGreater(soft['public_metrics']['interval_outside_after'], 0)

    def test_interval_arguments_are_checked(self):
        rest, triangles, held, initial = self.folded_sheet(30)
        n = len(rest); bounds = dict(lower=np.zeros(n), upper=np.ones(n))
        with self.assertRaisesRegex(ValueError, 'interval_axis'):
            rigid_deform(rest, initial, triangles, held, interval_axis=3, **bounds, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'interval_axis'):
            rigid_deform(rest, initial, triangles, held, lower=np.zeros(n), **self.parameters)
        with self.assertRaisesRegex(ValueError, 'interval_axis'):
            rigid_deform(rest, initial, triangles, held, interval_axis=1, lower=np.zeros(n - 1), upper=np.ones(n - 1), **self.parameters)
        with self.assertRaisesRegex(ValueError, 'lower <= upper'):
            rigid_deform(rest, initial, triangles, held, interval_axis=1, lower=np.ones(n), upper=np.zeros(n), **self.parameters)
        with self.assertRaisesRegex(ValueError, 'interval_weight'):
            rigid_deform(rest, initial, triangles, held, interval_axis=1, interval_weight=0., **bounds, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'project_active'):
            rigid_deform(rest, initial, triangles, held, interval_axis=1, project_active=1, **bounds, **self.parameters)

    def test_region_of_a_larger_mesh_and_invalid_inputs(self):
        rest, triangles, index, held = RelaxDisplacementTests().patch()
        extra = np.array([[5., 5, 5], [6, 5, 5], [5, 6, 5]])
        full_rest, full_initial = np.vstack([rest, extra]), np.vstack([rest, extra + .3])
        result = rigid_deform(full_rest, full_initial, triangles, np.r_[held, np.zeros(3, bool)], **self.parameters)
        np.testing.assert_array_equal(result['deformed'][-3:], full_initial[-3:])
        loose = held.copy(); loose[index[0, 4]] = False
        with self.assertRaisesRegex(ValueError, 'interior'):
            rigid_deform(rest, rest, triangles, loose, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'held flag'):
            rigid_deform(rest, rest, triangles, held[:-1], **self.parameters)
        with self.assertRaisesRegex(ValueError, 'iterations'):
            rigid_deform(rest, rest, triangles, held, iterations=0, **self.parameters)

    def test_preparation_route_deforms_with_a_held_array(self):
        rest, triangles, held, initial = self.folded_sheet(60)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, reference=rest, initial=initial, triangles=triangles, held=held.astype(np.int8))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'rigid_deform', 'parameters': dict(self.parameters, iterations=40),
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                               for name in ('reference', 'initial', 'triangles', 'held')}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertIn('energy_last', result['summary'])

    def test_preparation_route_takes_interval_arrays(self):
        rest, triangles, held, initial = self.folded_sheet(60)
        n = len(rest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, reference=rest, initial=initial, triangles=triangles, held=held.astype(np.int8),
                     lower=np.full(n, -1.), upper=np.full(n, .05))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'rigid_deform', 'parameters': dict(self.parameters, iterations=60, interval_axis=2),
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                               for name in ('reference', 'initial', 'triangles', 'held', 'lower', 'upper')}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertIn('interval_active', result['summary'])


def grid_triangles(columns, rows):
    index = np.arange(columns * rows).reshape(rows, columns)
    a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
    return np.r_[np.c_[a, b, c], np.c_[a, c, d]]


def plane(x0, x1, y0, y1, z=0., columns=9, rows=9):
    """A flat support facing +z (counter-clockwise seen from +z)."""
    x, y = np.meshgrid(np.linspace(x0, x1, columns), np.linspace(y0, y1, rows))
    return np.c_[x.ravel(), y.ravel(), np.full(x.size, float(z))], grid_triangles(columns, rows)


class ClosestPointTests(unittest.TestCase):
    def test_matches_an_exhaustive_search_on_random_queries(self):
        rng = np.random.default_rng(3)
        co, tri = plane(-1, 1, -1, 1, columns=7, rows=5)
        co[:, 2] = .3 * np.sin(2 * co[:, 0]) * np.cos(3 * co[:, 1])
        queries = rng.uniform(-1.5, 1.5, (60, 3))
        result = closest_points(queries, co, tri)
        for q, hit, distance in zip(queries, result['points'], result['distances']):
            exact = nearest_surface({'co': co, 'tri': tri}, q)
            np.testing.assert_allclose(distance, exact['distance'], atol=1e-12)
            np.testing.assert_allclose(hit, exact['point'], atol=1e-9)
        corners = co[tri[result['triangles']]]
        np.testing.assert_allclose(np.einsum('nk,nkd->nd', result['barycentric'], corners), result['points'], atol=1e-12)

    def test_a_large_nearby_triangle_with_a_distant_centroid_is_found(self):
        # A long sliver passes 0.05 below the query; its centroid is 30 away. Small triangles near the query are 0.2 away.
        co = np.array([[-1, -.05, 0], [60, -.05, 0], [60, -.06, 0],        # sliver
                       [-.2, .2, 0], [.2, .2, 0], [0, .5, 0]], float)      # small triangle, centroid close to the query
        tri = np.array([[0, 1, 2], [3, 4, 5]])
        result = closest_points([[0., 0., 0.]], co, tri)
        self.assertEqual(int(result['triangles'][0]), 0)
        self.assertAlmostEqual(float(result['distances'][0]), .05, places=12)

    def test_no_queries_give_empty_results(self):
        co, tri = plane(-1, 1, -1, 1, columns=3, rows=3)
        result = closest_points(np.zeros((0, 3)), co, tri)
        self.assertEqual(result['points'].shape, (0, 3)); self.assertEqual(result['barycentric'].shape, (0, 3))
        self.assertEqual(result['distances'].shape, (0,)); self.assertEqual(result['triangles'].shape, (0,))

    def test_barycentric_weights_reconstruct_the_point_on_a_degenerate_triangle(self):
        co = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]]); tri = np.array([[0, 1, 2]])
        result = closest_points([[1., 1, 0], [3., 0, 0], [.25, -2, 0]], co, tri)
        np.testing.assert_allclose(result['points'], [[1, 0, 0], [2, 0, 0], [.25, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(result['barycentric'] @ co, result['points'], atol=1e-12)
        np.testing.assert_allclose(result['barycentric'].sum(axis=1), 1)


class ConformToSurfaceTests(unittest.TestCase):
    parameters = {'units': 'synthetic length', 'frame': 'synthetic orthonormal', 'relative_area_tolerance': 1e-9}

    def sheet(self, columns=11, rows=11, z=0., size=1.):
        rest, tri = plane(-size, size, -size, size, z=z, columns=columns, rows=rows)
        held = (np.abs(rest[:, 0]) == size) | (np.abs(rest[:, 1]) == size)
        return rest, tri, held

    def cylinder_case(self):
        """A sheet 1.1 times wider than an arc of a unit cylinder about z that turns 80 degrees away from the view axis
        (-y) on each side, laid over the arc with its boundary on it and seeded with pleats along the surface normal."""
        angles = np.radians(np.linspace(-88, 88, 89)); heights = np.linspace(-.6, .6, 25)
        a, h = np.meshgrid(angles, heights)
        support = np.c_[np.sin(a).ravel(), -np.cos(a).ravel(), h.ravel()]
        support_tri = grid_triangles(len(angles), len(heights))
        u, v = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(-.4, .4, 17)); u, v = u.ravel(), v.ravel()
        turn = np.radians(80) * u
        rest = np.c_[1.1 * np.radians(80) * u, np.zeros_like(u), v]
        pleat = .06 * np.sin(3 * np.pi * u) ** 2 * np.cos(np.pi * v / .8)
        initial = np.c_[(1 + pleat) * np.sin(turn), -(1 + pleat) * np.cos(turn), v]
        held = (np.abs(u) == 1) | (np.abs(v) == .4)
        return rest, initial, grid_triangles(41, 17), held, support, support_tri, turn

    def test_the_sheet_stays_on_a_support_that_turns_away_where_a_view_axis_band_does_not(self):
        rest, initial, tri, held, support, support_tri, turn = self.cylinder_case()
        free = ~held; steep = free & (np.abs(turn) > np.radians(55))
        result = conform_to_surface(rest, initial, tri, held, support, support_tri, iterations=100, **self.parameters)
        # The same shape term with a band on the view axis only (the cylinder's depth at each vertex's column).
        depth = -np.cos(turn)
        banded = rigid_deform(rest, initial, tri, held, iterations=100, interval_axis=1, lower=depth, upper=depth,
                              **self.parameters)

        def off_cylinder(x):
            return np.abs(np.hypot(x[:, 0], x[:, 1]) - 1)
        chord = 1 - np.cos(np.radians(1))                  # facets of the support lie inside the cylinder by at most this
        metrics = result['public_metrics']
        self.assertTrue(metrics['converged'])
        self.assertLess(off_cylinder(result['deformed'])[free].max(), chord + 1e-9)
        self.assertEqual(metrics['band_outside_after'], 0); self.assertLess(metrics['band_max_after'], 1e-9)
        self.assertEqual(metrics['unsupported_after'], 0); self.assertEqual(metrics['supported_after'], int(free.sum()))
        self.assertEqual(metrics['support_facing_after'][1], 0)
        self.assertEqual(metrics['local_reversals_after'], 0)
        # The view-axis band only bounds the depth component of the pleat, which vanishes where the support turns away.
        self.assertGreater(off_cylinder(banded['deformed'])[steep].max(), 10 * chord)
        # The surplus is compressed along the surface and reported, not removed.
        self.assertLess(metrics['smallest_stretch_q01_after'], .95)
        np.testing.assert_array_equal(result['deformed'][held], initial[held])

    def test_all_zero_weights_reproduce_rigid_deform_exactly(self):
        rest, initial, tri, held, support, support_tri, _ = self.cylinder_case()
        plain = rigid_deform(rest, initial, tri, held, iterations=30, **self.parameters)
        zero = conform_to_surface(rest, initial, tri, held, support, support_tri, support_weights=0., iterations=30,
                                  **self.parameters)
        np.testing.assert_array_equal(zero['deformed'], plain['deformed'])
        self.assertEqual(zero['energies'], plain['energies'])
        self.assertEqual(zero['public_metrics']['support_vertices'], 0)

    def test_zero_weight_vertices_are_not_constrained_but_follow_their_neighbours(self):
        rest, initial, tri, held, support, support_tri, turn = self.cylinder_case()
        weight = (turn < 0).astype(float)                  # one half on the support, the other half free of it
        result = conform_to_surface(rest, initial, tri, held, support, support_tri, support_weights=weight,
                                    iterations=100, **self.parameters)
        metrics, free = result['public_metrics'], ~held
        self.assertEqual(metrics['support_vertices'], int((free & (weight > 0)).sum()))
        self.assertEqual(metrics['unconstrained_free_vertices'], int((free & (weight == 0)).sum()))
        self.assertEqual(metrics['band_outside_after'], 0)
        self.assertTrue(np.isnan(result['support_offset'][free & (weight == 0)]).all())
        # Facing is only measured where all three corners are weighted, and the report says how much that covers.
        measured = metrics['support_facing_measured_triangles']
        self.assertLess(measured, metrics['patch_triangles']); self.assertEqual(sum(metrics['support_facing_after']), measured)
        self.assertTrue(any('facing measured on' in text for text in result['warnings']))
        # Coupled through the shape term, the unweighted half is not the plain solve's result.
        plain = rigid_deform(rest, initial, tri, held, iterations=100, **self.parameters)['deformed']
        self.assertGreater(np.abs(result['deformed'] - plain)[free & (weight == 0)].max(), 1e-6)

    def test_offset_band_is_measured_along_the_support_normal_on_its_winding_side(self):
        support, support_tri = plane(-2, 2, -2, 2)
        rest, tri, held = self.sheet(z=.15)
        inside = conform_to_surface(rest, rest, tri, held, support, support_tri, lower=.1, upper=.2, **self.parameters)
        np.testing.assert_allclose(inside['deformed'], rest, atol=1e-12)  # already inside the band: nothing moves
        self.assertEqual(inside['public_metrics']['band_active'], 0)
        np.testing.assert_allclose(inside['support_offset'], .15, atol=1e-12)
        lifted = conform_to_surface(rest, rest, tri, held, support, support_tri, lower=.25, upper=.4, **self.parameters)
        height = lifted['deformed'][~held, 2]
        # Inside the band material is free (the held ring bows the sheet up); the active ones finish on the bound.
        self.assertTrue(((height > .25 - 1e-12) & (height < .4 + 1e-12)).all())
        self.assertGreater(int(np.count_nonzero(np.abs(height - .25) < 1e-12)), 0)
        self.assertGreater(height.max(), .26)
        self.assertEqual(lifted['public_metrics']['band_outside_after'], 0)
        self.assertEqual(lifted['public_metrics']['held_conflicts'], int(held.sum()))   # held rows stay at .15
        np.testing.assert_array_equal(lifted['deformed'][held], rest[held])
        flipped = conform_to_surface(rest, rest, tri, held, support, support_tri[:, ::-1], lower=-.2, upper=-.1,
                                     **self.parameters)
        np.testing.assert_allclose(flipped['support_offset'], -.15, atol=1e-12)
        self.assertEqual(flipped['public_metrics']['band_active'], 0)

    def test_held_vertices_keep_their_bytes_and_their_conflict_is_reported(self):
        support, support_tri = plane(-2, 2, -2, 2)
        rest, tri, held = self.sheet()
        initial = rest.copy(); initial[held, 2] = .05                      # the held ring sits off the zero band
        initial[~held, 2] = .3
        result = conform_to_surface(rest, initial, tri, held, support, support_tri, **self.parameters)
        np.testing.assert_array_equal(result['deformed'][held], initial[held])
        metrics = result['public_metrics']
        self.assertEqual(metrics['held_conflicts'], int(held.sum())); self.assertAlmostEqual(metrics['held_conflict_max'], .05)
        self.assertEqual(metrics['band_outside_after'], 0)
        self.assertTrue(any('Held vertices' in text for text in result['warnings']))
        unweighted = np.where(held, 0., 1.)
        quiet = conform_to_surface(rest, initial, tri, held, support, support_tri, support_weights=unweighted, **self.parameters)
        self.assertEqual(quiet['public_metrics']['held_conflicts'], 0)

    def test_material_pushed_past_an_open_support_boundary_is_not_counted_as_supported(self):
        support, support_tri = plane(-2, .3, -2, 2)                      # ends at x = .3 under the sheet
        rest, tri, held = self.sheet()
        initial = rest.copy(); initial[~held, 2] = .2
        result = conform_to_surface(rest, initial, tri, held, support, support_tri, **self.parameters)
        metrics = result['public_metrics']
        beyond = ~held & (rest[:, 0] > .3 + 1e-9)
        # The normal pull lands them level with the support: zero normal residual, yet off its edge.
        self.assertLess(np.abs(result['band_residual'][beyond]).max(), 1e-9)
        self.assertEqual(metrics['beyond_boundary_after'], int(beyond.sum()))
        self.assertEqual(metrics['unsupported_after'], int(beyond.sum()))
        self.assertEqual(metrics['supported_after'], int((~held).sum() - beyond.sum()))
        np.testing.assert_allclose(result['beyond_boundary'][beyond], rest[beyond, 0] - .3, atol=1e-6)
        self.assertTrue(any('open boundary' in text for text in result['warnings']))

    def test_moving_between_disconnected_support_parts_is_reported(self):
        left, left_tri = plane(-1.5, 1.2, -2, 2)
        right, right_tri = plane(1.3, 4, -2, 2)
        support = np.r_[left, right]; support_tri = np.r_[left_tri, right_tri + len(left)]
        rest, tri, held = self.sheet(size=.5)
        initial = rest.copy(); initial[held, 0] += 1.5                       # the held ring moves over the right part
        result = conform_to_surface(rest, initial, tri, held, support, support_tri, iterations=100, **self.parameters)
        metrics = result['public_metrics']
        self.assertEqual(metrics['support_components'], 2)
        self.assertGreater(metrics['component_switches'], 0)
        self.assertTrue(any('disconnected' in text for text in result['warnings']))
        np.testing.assert_allclose(result['deformed'], rest + [1.5, 0, 0], atol=1e-6)

    def test_convergence_needs_the_positions_to_settle(self):
        rest, initial, tri, held, support, support_tri, _ = self.cylinder_case()
        settled = conform_to_surface(rest, initial, tri, held, support, support_tri, iterations=100, **self.parameters)
        metrics = settled['public_metrics']
        self.assertTrue(metrics['converged'])
        self.assertLessEqual(metrics['last_step_max_move'], metrics['position_tolerance'])
        strict = conform_to_surface(rest, initial, tri, held, support, support_tri, iterations=100, position_tolerance=0.,
                                    **self.parameters)
        self.assertFalse(strict['public_metrics']['converged'])
        self.assertEqual(strict['public_metrics']['iterations'], 100)
        self.assertTrue(any('Iteration limit' in text for text in strict['warnings']))
        # Solve convergence and support of the result are separate: the result is still on the support.
        self.assertEqual(strict['public_metrics']['band_outside_after'], 0)

    def test_the_final_projection_reports_moves_between_support_parts(self):
        # One free vertex beyond the edge of an upper part; a weak band leaves it there, the projection drops it level
        # with that part, where a lower part is nearer, and then onto the lower part.
        upper, upper_tri = plane(-2, 0, -2, 2)
        lower, lower_tri = plane(0, 2, -2, 2, z=-.35)
        support = np.r_[upper, lower]; support_tri = np.r_[upper_tri, lower_tri + len(upper)]
        rest, tri = plane(.3, .5, -.1, .1, z=.3, columns=3, rows=3)
        held = np.ones(9, bool); held[4] = False
        weight = np.where(held, 0., 1.)
        result = conform_to_surface(rest, rest, tri, held, support, support_tri, support_weights=weight, band_weight=1e-9,
                                    **self.parameters)
        metrics = result['public_metrics']
        self.assertEqual(metrics['component_changes_during'], 0)
        self.assertGreater(metrics['component_changes_projection'], 0); self.assertEqual(metrics['component_switches'], 1)
        np.testing.assert_allclose(result['deformed'][4], [.4, 0, -.35], atol=1e-6)
        self.assertTrue(any('disconnected' in text for text in result['warnings']))

    def test_refuses_supports_without_one_side(self):
        support, support_tri = plane(-2, 2, -2, 2)
        rest, tri, held = self.sheet()
        mixed = support_tri.copy(); mixed[0] = mixed[0, ::-1]
        with self.assertRaisesRegex(ValueError, 'Inconsistently wound'):
            conform_to_surface(rest, rest, tri, held, support, mixed, **self.parameters)
        fins = np.r_[support, [[0, 0, 1.], [0, 0, -1.]]]; a, b = support_tri[0, :2]      # a third face on one edge
        with self.assertRaisesRegex(ValueError, 'Nonmanifold'):
            conform_to_surface(rest, rest, tri, held, fins, np.r_[support_tri, [[b, a, len(support)], [a, b, len(support) + 1]]],
                               **self.parameters)
        with self.assertRaisesRegex(ValueError, 'lower <= upper'):
            conform_to_surface(rest, rest, tri, held, support, support_tri, lower=.2, upper=.1, **self.parameters)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            conform_to_surface(rest, rest, tri, held, support, support_tri, support_weights=-1., **self.parameters)

    def test_preparation_route_takes_the_support_arrays(self):
        support, support_tri = plane(-2, 2, -2, 2)
        rest, tri, held = self.sheet()
        initial = rest.copy(); initial[~held, 2] = .1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, reference=rest, initial=initial, triangles=tri, held=held.astype(np.int8),
                     support_positions=support, support_triangles=support_tri, support_weights=np.ones(len(rest)))
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            names = ('reference', 'initial', 'triangles', 'held', 'support_positions', 'support_triangles', 'support_weights')
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'conform_to_surface', 'parameters': dict(self.parameters, upper=.02),
                    'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name} for name in names}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['summary']['band_outside_after'], 0)


class PlanarRelayoutTests(unittest.TestCase):
    """Collapsed material re-laid in a projection plane, depth from its surroundings or a map."""

    @staticmethod
    def grid(n=9, spacing=.1):
        a, b = np.meshgrid(np.arange(n) * spacing, np.arange(n) * spacing)
        depth = .2 * (a - .4) ** 2 + .1 * (b - .4) ** 2                    # a gently curved sheet, seen along y
        positions = np.c_[a.ravel(), depth.ravel(), b.ravel()]
        tri = []
        for j in range(n - 1):
            for i in range(n - 1):
                v = j * n + i; tri += [[v, v + 1, v + n + 1], [v, v + n + 1, v + n]]
        return positions, np.array(tri), n

    def test_collapsed_block_is_relaid_and_given_the_surrounding_depth(self):
        rest, tri, n = self.grid()
        ij = np.array([(i, j) for j in range(n) for i in range(n)])
        block = (ij[:, 0] >= 3) & (ij[:, 0] <= 5) & (ij[:, 1] >= 3) & (ij[:, 1] <= 5)
        posed = rest.copy(); posed[block, 2] = .42 + .01 * (ij[block, 1] - 4)   # rows squeezed together (collapsed quads)
        posed[block, 1] += .02                                                  # and pushed off the surface
        samples = ~block & (np.abs(ij[:, 0] - 4) <= 3) & (np.abs(ij[:, 1] - 4) <= 3)
        result = planar_relayout(posed, tri, block, depth_samples=samples, reference=rest)
        out = result['relaid']
        np.testing.assert_allclose(out[block][:, [0, 2]], rest[block][:, [0, 2]], atol=1e-9)   # the uniform grid is harmonic
        np.testing.assert_allclose(out[block, 1], rest[block, 1], atol=4e-3)                  # depth from the surroundings
        np.testing.assert_array_equal(out[~block], posed[~block])                              # held vertices exact bytes
        m = result['public_metrics']
        self.assertGreater(m['compressed_before'], 0); self.assertEqual(m['compressed_after'], 0)
        self.assertEqual(m['plane_flips_after'], 0)

    def test_depth_can_come_from_a_depth_map(self):
        rest, tri, n = self.grid()
        free = np.zeros(len(rest), bool); free[4 * n + 4] = True
        posed = rest.copy(); posed[4 * n + 4] += [.03, .05, -.02]
        window, cell = (-.05, .85, -.05, .85), .01
        a = window[0] + (np.arange(90) + .5) * cell
        A, B = np.meshgrid(a, a); plane = .3 + .1 * A - .05 * B                 # a guide surface as a front depth map
        out = planar_relayout(posed, tri, free, depth_map=plane, window=window, cell=cell)['relaid']
        np.testing.assert_allclose(out[4 * n + 4, [0, 2]], [.4, .4], atol=1e-9)
        self.assertAlmostEqual(out[4 * n + 4, 1], .3 + .1 * .4 - .05 * .4, places=6)

    @staticmethod
    def fan(nr=7, na=9):
        """An uneven fan around a corner at the origin (radii grow geometrically), seen along y."""
        r = .05 * 1.35 ** np.arange(nr); a = np.radians(np.linspace(10, 80, na))
        R, A = np.meshgrid(r, a, indexing='ij')
        positions = np.c_[(R * np.cos(A)).ravel(), np.zeros(R.size), (R * np.sin(A)).ravel()]
        tri = []
        for i in range(nr - 1):
            for j in range(na - 1):
                v = i * na + j; tri += [[v, v + na, v + na + 1], [v, v + na + 1, v + 1]]
        return positions, np.array(tri), nr, na

    def test_rigid_layout_turns_an_uneven_fan_that_the_harmonic_layout_distorts(self):
        rest, tri, nr, na = self.fan()
        th = np.radians(-25.); rot = np.array([[np.cos(th), 0, -np.sin(th)], [0, 1, 0], [np.sin(th), 0, np.cos(th)]])
        turned = rest @ rot.T                                                    # the whole fan turned about the corner
        ij = np.array([(i, j) for i in range(nr) for j in range(na)])
        free = (ij[:, 0] > 0) & (ij[:, 0] < nr - 1) & (ij[:, 1] > 0) & (ij[:, 1] < na - 1)
        posed = turned.copy()
        posed[free, 2] += .01 * np.where(ij[free, 1] % 2, 1., -1.)              # buckled into a zigzag across the columns
        rigid = planar_relayout(posed, tri, free, depth_map=np.zeros((20, 20)), window=(-.5, .5, -.5, .5), cell=.05,
                                reference=rest, layout='rigid')
        np.testing.assert_allclose(rigid['relaid'][free][:, [0, 2]], turned[free][:, [0, 2]], atol=1e-6)
        harmonic = planar_relayout(posed, tri, free, depth_map=np.zeros((20, 20)), window=(-.5, .5, -.5, .5), cell=.05)
        self.assertGreater(np.abs(harmonic['relaid'][free] - turned[free]).max(), 2e-3)  # uniform weights even the spacing out
        self.assertEqual(rigid['public_metrics']['layout'], 'rigid')
        with self.assertRaisesRegex(ValueError, 'reference'):
            planar_relayout(posed, tri, free, depth_map=np.zeros((20, 20)), window=(-.5, .5, -.5, .5), cell=.05, layout='rigid')
        with self.assertRaisesRegex(ValueError, 'layout'):
            planar_relayout(posed, tri, free, depth_map=np.zeros((20, 20)), window=(-.5, .5, -.5, .5), cell=.05, layout='even')

    def test_depth_map_can_be_low_passed(self):
        rest, tri, n = self.grid()
        free = np.zeros(len(rest), bool); free[4 * n + 4] = True
        window, cell = (-.055, .845, -.055, .845), .01                          # the vertex sits on a cell centre
        a = window[0] + (np.arange(90) + .5) * cell
        A, B = np.meshgrid(a, a); plane = .3 + .1 * A - .05 * B
        ripple = plane + .01 * np.where((np.arange(90)[None, :] + np.arange(90)[:, None]) % 2, 1., -1.)   # fine corrugation
        ripple[:5, :5] = np.nan                                                  # an empty corner stays empty
        sharp = planar_relayout(rest, tri, free, depth_map=ripple, window=window, cell=cell)['relaid'][4 * n + 4, 1]
        smooth = planar_relayout(rest, tri, free, depth_map=ripple, window=window, cell=cell, depth_blur=3.)
        exact = .3 + .1 * .4 - .05 * .4
        self.assertLess(abs(smooth['relaid'][4 * n + 4, 1] - exact), 1e-3)
        self.assertGreater(abs(sharp - exact), 5e-3)
        self.assertEqual(smooth['public_metrics']['depth_blur_cells'], 3.)
        with self.assertRaisesRegex(ValueError, 'depth_blur'):
            planar_relayout(rest, tri, free, depth_map=ripple, window=window, cell=cell, depth_blur=-1.)

    def test_refuses_boundary_free_vertices_and_ambiguous_depth(self):
        rest, tri, n = self.grid(4)
        free = np.zeros(len(rest), bool); free[0] = True
        with self.assertRaisesRegex(ValueError, 'interior'):
            planar_relayout(rest, tri, free, depth_samples=~free)
        free[:] = False; free[n + 1] = True
        with self.assertRaisesRegex(ValueError, 'exactly one depth source'):
            planar_relayout(rest, tri, free)
        with self.assertRaisesRegex(ValueError, 'must not include'):
            planar_relayout(rest, tri, free, depth_samples=np.ones(len(rest), bool))
        with self.assertRaisesRegex(ValueError, 'does not cover'):
            planar_relayout(rest, tri, free, depth_map=np.full((2, 2), np.nan), window=(0, .3, 0, .3), cell=.15)

    def test_available_for_array_preparation(self):
        from .preparation import ARRAY_OPERATIONS
        self.assertIn('planar_relayout', ARRAY_OPERATIONS)


if __name__ == '__main__': unittest.main()


class ProjectedBoundaryTests(unittest.TestCase):
    """Boundary loops of a patch in a projection plane (x right, z up; y is the depth the projection drops)."""
    units = {'units': 'synthetic length', 'frame': 'synthetic orthonormal'}

    @staticmethod
    def fan(ring, centre, depth=None):
        """A disc: one centre vertex (index 0) joined to a counter-clockwise ring (indices 1..m)."""
        ring = np.asarray(ring, float); xz = np.r_[[centre], ring]
        y = np.zeros(len(xz)) if depth is None else np.r_[0., depth]
        m = len(ring)
        return np.c_[xz[:, 0], y, xz[:, 1]], np.array([[0, 1 + k, 1 + (k + 1) % m] for k in range(m)])

    @staticmethod
    def grid(n):
        x, z = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
        positions = np.c_[x.ravel(), np.zeros(n * n), z.ravel()]
        index = np.arange(n * n).reshape(n, n)
        return positions, grid_triangles(n, n), index

    def report(self, positions, triangles, **options):
        return projected_boundary(positions, triangles, **self.units, **options)

    def test_a_convex_loop_and_a_simple_reflex_loop(self):
        square = self.report(*self.fan([[0, 0], [1, 0], [1, 1], [0, 1]], [.5, .5]))
        loop = square['loops'][0]
        self.assertEqual(loop['vertices'], [1, 2, 3, 4]); self.assertEqual(loop['orientation'], 'counterclockwise')
        self.assertAlmostEqual(loop['signed_area'], 1.); self.assertEqual(loop['turning_number'], 1)
        self.assertEqual(loop['reflex_corners'], []); self.assertEqual(loop['convex_corners'], 4)
        ell = self.report(*self.fan([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], [.5, .5]))
        corners = ell['loops'][0]['reflex_corners']
        self.assertEqual([c['vertex'] for c in corners], [4])            # the inner corner of the L, ring point (1, 1)
        self.assertAlmostEqual(corners[0]['domain_angle_degrees'], 270.)
        metrics = ell['public_metrics']
        self.assertEqual((metrics['crossings'], metrics['touches'], metrics['reflex_corners']), (0, 0, 1))
        self.assertEqual(ell['loops'][0]['interior_angles'], 'defined on the domain side')

    def test_a_projected_bow_tie_crosses_and_leaves_interior_angles_undefined(self):
        # The ring's far end twists. The two crossing boundary segments lie at depths 0 and 0.5: they cross only in the
        # projection (a claim about these two segments, not about the triangles of the fan).
        positions, triangles = self.fan([[0, 0], [2, 2], [2, 0], [0, 2]], [1, .5], depth=[0, 0, .5, .5])
        result = self.report(positions, triangles)
        self.assertEqual(result['public_metrics']['crossings'], 1)
        crossing = result['crossings'][0]
        self.assertEqual(sorted(tuple(e['edge']) for e in crossing['edges']), [(1, 2), (3, 4)])
        np.testing.assert_allclose(crossing['point'], [1, 1])
        # At the crossing (the midpoint of both segments here) the segments are 0.5 apart in depth.
        depth_at = [(positions[a, 1] + positions[b, 1]) / 2 for a, b in ((1, 2), (3, 4))]
        self.assertAlmostEqual(abs(depth_at[1] - depth_at[0]), .5)
        loop = result['loops'][0]
        self.assertEqual(loop['turning_number'], 0); self.assertIsNone(loop['reflex_corners'])
        self.assertTrue(loop['interior_angles'].startswith('undefined'))
        self.assertIn('crossing', loop['issues'])

    def test_touches_overlaps_backtracking_and_collapsed_edges(self):
        touch = self.report(*self.fan([[0, 0], [4, 0], [4, 4], [2, 0], [0, 4]], [1, 2]))
        self.assertEqual(touch['public_metrics']['crossings'], 0)
        self.assertEqual(sorted(tuple(t['edges'][1]['edge']) for t in touch['touches']), [(3, 4), (4, 5)])
        self.assertTrue(all(t['edges'][0]['edge'] == [1, 2] for t in touch['touches']))
        overlap = self.report(*self.fan([[0, 0], [4, 0], [4, 2], [3, 0], [1, 0], [0, 2]], [2, 1.5]))
        self.assertEqual(len(overlap['collinear_overlaps']), 1)
        self.assertAlmostEqual(overlap['collinear_overlaps'][0]['overlap_length'], 2.)
        back = self.report(*self.fan([[0, 0], [2, 0], [1, 0], [1, 2]], [.5, 1]))
        self.assertEqual([b['vertex'] for b in back['adjacent_overlaps']], [2])   # (2, 0) folds back along (0,0)-(2,0)
        collapsed = self.report(*self.fan([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], [.5, .5], depth=[0, 0, .3, 0, 0]))
        self.assertEqual([c['edge'] for c in collapsed['collapsed_edges']], [[2, 3]])
        self.assertIsNone(collapsed['loops'][0]['reflex_corners'])
        # Ordinary adjacent edges of a clean loop are never reported.
        clean = self.report(*self.fan([[0, 0], [1, 0], [1, 1], [0, 1]], [.5, .5]))
        self.assertEqual(clean['touches'] + clean['adjacent_overlaps'] + clean['collapsed_edges'], [])

    def test_separate_components_holes_and_pinched_vertices(self):
        a, ta = self.fan([[0, 0], [1, 0], [1, 1], [0, 1]], [.5, .5])
        b, tb = self.fan([[3, 0], [4, 0], [4, 1], [3, 1]], [3.5, .5])
        two = self.report(np.r_[a, b], np.r_[ta, tb + len(a)])
        self.assertEqual((two['public_metrics']['components'], two['public_metrics']['loops']), (2, 2))
        self.assertEqual([loop['inside_loops'] for loop in two['loops']], [[], []])
        # An annulus: a 4x4 grid without its middle quad. The hole is a second loop of the same component; seen from
        # the material its four corners are reflex, which an outer convex loop says nothing about.
        positions, triangles, index = self.grid(4)
        hole = [r for r, t in enumerate(triangles) if set(t) <= set(index[1:3, 1:3].ravel())]
        ring = self.report(positions, np.delete(triangles, hole, axis=0))
        self.assertEqual((ring['public_metrics']['components'], ring['public_metrics']['loops']), (1, 2))
        outer, inner = sorted(ring['loops'], key=lambda loop: -abs(loop['signed_area']))
        self.assertEqual((outer['orientation'], inner['orientation']), ('counterclockwise', 'clockwise'))
        self.assertEqual(inner['inside_loops'], [outer['loop']]); self.assertEqual(outer['inside_loops'], [])
        self.assertEqual(outer['reflex_corners'], []); self.assertEqual(len(inner['reflex_corners']), 4)
        self.assertEqual(sorted(inner['vertices']), sorted(index[1:3, 1:3].ravel().tolist()))
        # Two discs sharing one ring vertex: the loops split there instead of tracing a figure of eight.
        c, tc = self.fan([[1, 1], [2, 1], [2, 2], [1, 2]], [1.5, 1.5])
        shared = np.r_[a, c[[0, 2, 3, 4]]]
        tc = np.where(tc == 1, 3, np.where(tc == 0, 5, tc + 4))           # c's first ring point is a's point (1, 1)
        pinch = self.report(shared, np.r_[ta, tc])
        self.assertEqual(pinch['pinched_vertices'], [3])
        self.assertEqual(pinch['public_metrics']['loops'], 2)
        self.assertEqual(sorted(len(loop['vertices']) for loop in pinch['loops']), [4, 4])
        self.assertEqual(pinch['public_metrics']['ambiguous_vertices'], 0)
        # Two simple loops that cross each other: neither is called a hole of the other.
        d, td = self.fan([[.5, .5], [1.5, .5], [1.5, 1.5], [.5, 1.5]], [1, 1])
        crossing = self.report(np.r_[a, d], np.r_[ta, td + len(a)])
        self.assertEqual(crossing['public_metrics']['crossings'], 2)
        self.assertEqual([loop['inside_loops'] for loop in crossing['loops']], [[], []])
        self.assertEqual([loop['nesting_unresolved_with'] for loop in crossing['loops']], [[1], [0]])

    def test_the_free_domain_is_what_planar_relayout_re_lays(self):
        positions, triangles, index = self.grid(7)
        free = np.zeros(len(positions), bool); free[index[2:5, 2:5].ravel()] = True
        patch = self.report(positions, triangles)
        self.assertEqual(patch['public_metrics']['domain'], 'the supplied patch')
        self.assertEqual(len(patch['loops'][0]['vertices']), 24)             # the outer perimeter of the patch
        domain = self.report(positions, triangles, free=free)
        loop = domain['loops'][0]
        # The held ring around the free block; two of its corners lie only on triangles without a free vertex.
        ring = set(index[1:6, 1:6].ravel().tolist()) - set(index[2:5, 2:5].ravel().tolist())
        self.assertLessEqual(set(loop['vertices']), ring); self.assertEqual(len(loop['vertices']), 14)
        self.assertEqual(ring - set(loop['vertices']), {int(index[1, 5]), int(index[5, 1])})
        self.assertFalse(free[loop['vertices']].any())
        self.assertEqual(domain['public_metrics']['interior_held_vertices'], 0)
        # A held handle inside the free region is not on any loop: it is reported, not qualified.
        handle = free.copy(); handle[index[3, 3]] = False
        held_inside = self.report(positions, triangles, free=handle)
        self.assertEqual(held_inside['interior_held_vertices'], [int(index[3, 3])])
        rim = free.copy(); rim[index[0, 3]] = True
        self.assertEqual(self.report(positions, triangles, free=rim)['free_on_patch_boundary'], [int(index[0, 3])])
        with self.assertRaisesRegex(ValueError, 'belong to the patch'):     # as planar_relayout, no silent subset
            self.report(np.r_[positions, [[9., 9, 9]]], triangles, free=np.r_[np.flatnonzero(free), len(positions)])

    def test_malformed_topology_is_reported_and_malformed_arrays_refuse(self):
        positions, triangles = self.fan([[0, 0], [1, 0], [1, 1], [0, 1]], [.5, .5])
        fin = self.report(np.r_[positions, [[.5, 1, .5]]], np.r_[triangles, [[0, 1, 5]]])   # a third face on edge (0, 1)
        self.assertEqual(fin['nonmanifold_edges'], [[0, 1]])
        flipped = triangles.copy(); flipped[0] = flipped[0, ::-1]
        wound = self.report(positions, flipped)
        self.assertEqual(wound['public_metrics']['inconsistent_edges'], 2)
        self.assertGreater(wound['public_metrics']['open_chains'], 0)
        for bad, message in ((np.r_[triangles, triangles[:1]], 'Duplicate'), (np.r_[triangles, [[1, 1, 2]]], 'repeats'),
                             (triangles + 10, 'indexing')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.report(positions, bad)
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            self.report(positions, triangles, plane_basis=[[1, 0, 0], [1, 0, 0]])
        with self.assertRaisesRegex(ValueError, 'orthonormal'):
            self.report(positions, triangles, plane_basis=[[1 + 1e-6, 0, 0], [0, 0, 1]])
        with self.assertRaisesRegex(ValueError, 'integer axes'):
            self.report(positions, triangles, plane_axes=(0, 2.5))
        # A closed part touching the ring at one vertex: its boundary degree stays 2, yet it has two triangle fans.
        tetra = np.array([[1., .3, 0], [1., .6, .3], [1.3, .5, 0]])
        closed = np.array([[1, 6, 5], [1, 7, 6], [1, 5, 7], [5, 6, 7]])
        touching = self.report(np.r_[positions, tetra], np.r_[triangles, closed])
        self.assertEqual(touching['multi_fan_vertices'], [1]); self.assertEqual(touching['pinched_vertices'], [])
        self.assertEqual(sorted(c['euler_characteristic'] for c in touching['components']), [1, 2])
        alone = self.report(np.r_[positions, tetra], closed)
        self.assertEqual(alone['public_metrics']['loops'], 0); self.assertIn('no boundary', alone['warnings'][0])
        # A sheet seen edge on: the boundary has no span in this plane.
        edge_on = self.report(np.array([[1., 0, .5], [0, 0, 0], [0, 0, 1], [0, 0, 2]]),
                              np.array([[0, 1, 2], [0, 2, 3]]), plane_axes=(0, 1), tolerance=1e-12)
        self.assertEqual(edge_on['public_metrics']['collapsed_edges'], 2)
        self.assertIsNone(edge_on['loops'][0]['reflex_corners'])
        with self.assertRaisesRegex(ValueError, 'units and frame'):
            projected_boundary(positions, triangles, units='', frame='f')

    def test_scale_translation_tolerance_and_supplied_indexing(self):
        # A vertex 1e-3 from a non-adjacent edge (a near touch); the patch sits among unused vertices at shifted IDs.
        ring = [[0, 0], [4, 0], [4, 4], [2, 1e-3], [0, 4]]
        positions, triangles = self.fan(ring, [1, 2])
        offset = 7; big = np.zeros((offset + len(positions) + 3, 3)); big[offset:offset + len(positions)] = positions
        base = self.report(big, triangles + offset)
        self.assertEqual(base['public_metrics']['touches'], 0)
        self.assertEqual(base['loops'][0]['vertices'], [8, 9, 10, 11, 12])
        self.assertAlmostEqual(base['public_metrics']['tolerance'], 1e-9 * np.hypot(4, 4))
        near = self.report(big, triangles + offset, tolerance=2e-3)
        self.assertEqual(near['public_metrics']['touches'], 2)
        self.assertEqual(sorted(e['triangle'] for t in near['touches'] for e in t['edges'] if e['edge'] == [8, 9]), [0, 0])
        for scale, shift in ((1e-3, 1e6), (1e3, -5e4)):
            moved = big * scale + [shift, 3., shift]
            same = self.report(moved, triangles + offset)
            self.assertEqual(same['public_metrics']['touches'], 0)
            self.assertEqual(same['loops'][0]['vertices'], base['loops'][0]['vertices'])
            self.assertAlmostEqual(same['loops'][0]['signed_area'] / scale ** 2, base['loops'][0]['signed_area'], places=6)
            self.assertEqual(self.report(moved, triangles + offset, tolerance=2e-3 * scale)['public_metrics']['touches'], 2)
        # The same patch turned into another plane, read through an explicit basis, gives the same loop.
        turn = np.linalg.qr(np.array([[.3, -.8, .5], [.9, .2, -.1], [.1, .6, .8]]))[0]
        turned = self.report(big @ turn.T, triangles + offset, plane_basis=(turn @ np.eye(3)[[0, 2]].T).T)
        self.assertAlmostEqual(turned['loops'][0]['signed_area'], base['loops'][0]['signed_area'])

    def test_classification_does_not_depend_on_the_coordinate_scale(self):
        # A random star-shaped ring: products of orientation determinants underflow at tiny scales, sign tests do not.
        rng = np.random.default_rng(205)
        angle = np.sort(rng.uniform(0, 2 * np.pi, 8)); radius = rng.uniform(.1, 2, 8)
        positions, triangles = self.fan(np.c_[radius * np.cos(angle), radius * np.sin(angle)], [0, 0])
        keys = ('crossings', 'touches', 'collinear_overlaps', 'collapsed_edges', 'reflex_corners')
        base = self.report(positions, triangles)
        counts = [base['public_metrics'][k] for k in keys]
        self.assertEqual(counts, [0, 0, 0, 0, 3])
        for scale in (1e-90, 1e90):
            scaled = self.report(positions * scale, triangles)
            self.assertEqual([scaled['public_metrics'][k] for k in keys], counts)
            self.assertAlmostEqual(scaled['loops'][0]['signed_area'] / scale ** 2, base['loops'][0]['signed_area'])
            json.dumps(scaled, allow_nan=False)
        with self.assertRaisesRegex(ValueError, 'supported range'):
            self.report(positions * 1e200, triangles)
        with self.assertRaisesRegex(ValueError, 'below 1e150'):
            self.report(positions, triangles, tolerance=1e308)

    def test_preparation_route(self):
        # A crossing loop: undefined values must travel as nulls through the JSON report.
        positions, triangles = self.fan([[0, 0], [2, 2], [2, 0], [0, 2]], [1, .5], depth=[0, .5, .5, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.npz'
            np.savez(source, positions=positions, triangles=triangles)
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            item = {'reads': {'geometry': digest}, 'workbench': {'profile': 'analysis'},
                    'payload': {'operation': 'projected_boundary', 'parameters': dict(self.units),
                                'inputs': {name: {'path': str(source), 'sha256': digest, 'array': name}
                                           for name in ('positions', 'triangles')}}}
            result = ArrayPreparation(root / 'output')(item, {'attempt_key': 'ordinary'})
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['summary']['crossings'], 1)
            detail = json.loads(Path(result['prepared']['result.json']['path']).read_text())
            self.assertIsNone(detail['loops'][0]['reflex_corners'])


class SmoothRegionTests(unittest.TestCase):
    def grid(self, n=21):
        x, z = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n)); P = np.c_[x.ravel(), np.zeros(n * n), z.ravel()]
        ids = np.arange(n * n).reshape(n, n)
        T = np.array([t for r in range(n - 1) for c in range(n - 1) for t in
                      ([ids[r, c], ids[r + 1, c], ids[r, c + 1]], [ids[r, c + 1], ids[r + 1, c], ids[r + 1, c + 1]])])
        border = np.r_[ids[0], ids[-1], ids[:, 0], ids[:, -1]]
        return P, T, ids, np.unique(border)

    def test_lumps_go_the_broad_shape_stays_and_held_points_keep_their_bytes(self):
        P, T, ids, border = self.grid()
        rng = np.random.default_rng(3); broad = .2 * np.exp(-(P[:, 0] ** 2 + P[:, 2] ** 2) / .3)
        lumpy = P.copy(); lumpy[:, 1] = broad + .01 * rng.standard_normal(len(P))
        weights = np.ones(len(P)); weights[ids[:, 10:]] = 0.                      # the right half is not smoothed
        out = smooth_region(lumpy, T, held=border, weights=weights, iterations=20)['positions']
        self.assertTrue(np.array_equal(out[border], lumpy[border]))
        self.assertTrue(np.array_equal(out[ids[:, 10:]], lumpy[ids[:, 10:]]))
        inner = np.setdiff1d(ids[1:-1, 1:9].ravel(), border)
        self.assertLess(np.std(out[inner, 1] - broad[inner]), np.std(lumpy[inner, 1] - broad[inner]) / 2)
        centre = ids[10, 5]
        self.assertGreater(out[centre, 1], .9 * broad[centre])                   # Taubin does not shrink the bump away

    def test_the_rest_and_the_end_shape_are_smoothed_by_one_operator(self):
        P, T, ids, border = self.grid()
        rng = np.random.default_rng(5); rest = P.copy(); rest[:, 1] = .01 * rng.standard_normal(len(P))
        change = np.c_[np.zeros(len(P)), .05 * (P[:, 2] > 0), np.zeros(len(P))]       # a crude closing change
        stack = smooth_region(np.stack([rest, rest + change]), T, held=border, iterations=15)['positions']
        alone = smooth_region(change, T, held=border, iterations=15)['positions']
        np.testing.assert_allclose(stack[1] - stack[0], alone, atol=1e-12)         # the motion gets the same smoothing
        only_rest = smooth_region(rest, T, held=border, iterations=15)['positions'] + change
        self.assertGreater(np.abs(only_rest - stack[1]).max(), 1e-3)             # smoothing the rest alone does not

    def test_refusals(self):
        P, T, ids, border = self.grid(5)
        with self.assertRaisesRegex(ValueError, 'mu < -lam'):
            smooth_region(P, T, lam=.5, mu=-.4)
        with self.assertRaisesRegex(ValueError, 'Weights'):
            smooth_region(P, T, weights=np.full(len(P), 2.))
        with self.assertRaisesRegex(ValueError, 'Triangles'):
            smooth_region(P, T + 100)


if __name__ == '__main__':
    unittest.main()
