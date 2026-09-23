"""Planar and surface metric invariants and ordinary preparation integration, without IO effects."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import numpy as np
from scipy.sparse import coo_matrix, diags
from .metric_fitting import planar_fem_metric, surface_fem_metric
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


if __name__ == '__main__': unittest.main()
