"""Planar and surface metric invariants and ordinary preparation integration, without IO effects."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from scipy.sparse import coo_matrix, diags
from .metric_fitting import planar_fem_metric, surface_fem_metric, relax_displacement, rigid_deform, planar_relayout
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
