import unittest
import json
import numpy as np
from .source_coverage import source_coverage, _edges
from .preparation import prepare_arrays


def fixture():
    uv = np.array([[0., 0.], [1, 0], [1, 1], [0, 1], [.5, .5]])
    xyz = np.column_stack([uv, [0, 0, 0, 0, .4]])
    tri = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    refs = np.array([0, 0, 1, 2, 0])
    bary = np.array([[1., 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]])
    return dict(source_positions=xyz, source_triangles=tri, source_uv=uv, source_components=np.zeros(4, int),
        source_qualified=np.ones(4, bool), positions=xyz.copy(), triangles=tri.copy(),
        correspondence_triangles=refs, correspondence_barycentric=bary,
        footprint=np.arange(4), domain=np.arange(4), source_boundary=np.array(sorted(_edges(tri)[0])),
        boundary=np.array(sorted(_edges(tri)[0])), component=0, units='m', frame='saved common XYZ', resolution=32,
        provenance={'source': {'mesh': 'synthetic patch', 'pose': 0, 'capture_sha256': 'a'*64},
                    'candidate': {'mesh': 'synthetic patch', 'pose': 0, 'capture_sha256': 'b'*64},
                    'qualification': 'synthetic continuous chart'})


class SourceCoverageTests(unittest.TestCase):
    def run_case(self, **changes):
        args = fixture(); args.update(changes)
        return source_coverage(**args)

    def test_unchanged_curved_source_has_zero_corresponding_deviation_and_uncertainty(self):
        r = self.run_case()
        self.assertEqual(r['status'], 'measured')
        self.assertEqual(r['orientation']['reversed_triangle_ids'], [])
        self.assertLess(r['deviation']['maximum'], 1e-14)
        self.assertEqual(r['boundary']['sampled_max_uv'], 0)
        for key in ('gap_area_uv_interval', 'overlap_area_uv_interval'):
            self.assertEqual(r['coverage'][key][0], 0)
            self.assertGreater(r['coverage'][key][1], 0)
        self.assertNotIn('passed', r)

    def test_reversed_actual_triangles_are_reported(self):
        tri = fixture()['triangles']; tri[1] = tri[1, ::-1]
        r = self.run_case(triangles=tri)
        self.assertEqual(r['orientation']['reversed_triangle_ids'], [1])
        self.assertEqual(r['status'], 'measured')

    def test_swapped_correspondence_on_surface_detects_fold(self):
        a = fixture()
        a['source_uv'][4] = [.35, .55]
        a['source_positions'][4, :2] = [.35, .55]
        a['positions'][4, :2] = [.35, .55]
        a['positions'][[0, 1]] = a['positions'][[1, 0]]
        a['correspondence_barycentric'][[0, 1]] = a['correspondence_barycentric'][[1, 0]]
        r = source_coverage(**a)
        self.assertTrue(r['orientation']['reversed_triangle_ids'])
        self.assertGreater(r['coverage']['overlap_area_uv_interval'][0], 0)
        self.assertGreater(r['coverage']['gap_area_uv_interval'][0], 0)

    def test_gap_has_area_interval_containing_true_area(self):
        a = fixture(); a['domain'] = np.arange(3)
        a['boundary'] = np.array(sorted(_edges(a['triangles'][a['domain']])[0]))
        r = source_coverage(**a); low, high = r['coverage']['gap_area_uv_interval']
        self.assertGreater(low, 0); self.assertLessEqual(low, .25); self.assertGreaterEqual(high, .25)

    def test_overlap_multiplicity_with_distinct_material_vertices(self):
        a = fixture()
        for k in ('positions', 'correspondence_triangles', 'correspondence_barycentric'):
            a[k] = np.concatenate([a[k], a[k][:3]])
        a['triangles'] = np.vstack([a['triangles'], [5, 6, 7]])
        a['domain'] = np.arange(5); a['boundary'] = np.array(sorted(_edges(a['triangles'])[0]))
        r = source_coverage(**a)
        low, high = r['coverage']['overlap_area_uv_interval']
        self.assertGreater(low, 0); self.assertLessEqual(low, .5); self.assertGreaterEqual(high, .5)
        self.assertIn('2', r['coverage']['certain_multiplicity_cells'])

    def test_missing_extrapolated_and_unqualified_correspondence_stay_unknown(self):
        for change in ('missing', 'beyond', 'unqualified', 'component'):
            a = fixture()
            if change == 'missing': a['correspondence_triangles'][0] = -1
            if change == 'beyond': a['correspondence_barycentric'][0] = [-.1, 1.1, 0]
            if change == 'unqualified': a['source_qualified'][0] = False
            if change == 'component': a['source_components'][0] = 1
            r = source_coverage(**a)
            self.assertEqual(r['status'], 'unknown'); self.assertIsNone(r['coverage'])

    def test_ambiguous_source_chart_is_unknown(self):
        a = fixture(); a['source_uv'][4] = [1.5, .5]
        r = source_coverage(**a)
        self.assertEqual(r['status'], 'unknown')

    def test_connected_consistently_wound_but_overlapping_source_is_unknown(self):
        a = fixture()
        angles = np.linspace(0, 4*np.pi, 9)
        uv = np.array([[radius*np.cos(t), radius*np.sin(t)] for t in angles for radius in (1., 2.)])
        tri = np.array([face for i in range(8) for face in
                        ([2*i, 2*i+1, 2*i+2], [2*i+1, 2*i+3, 2*i+2])])
        xyz = np.column_stack([uv, np.zeros(len(uv))])
        edges = np.array(sorted(_edges(tri)[0]))
        a.update(source_positions=xyz, source_uv=uv, source_triangles=tri,
                 source_components=np.zeros(len(tri), int), source_qualified=np.ones(len(tri), bool),
                 positions=xyz, triangles=tri, footprint=np.arange(len(tri)), domain=np.arange(len(tri)),
                 source_boundary=edges, boundary=edges, correspondence_triangles=np.zeros(len(uv), int),
                 correspondence_barycentric=np.tile([1., 0, 0], (len(uv), 1)))
        r = source_coverage(**a)
        self.assertEqual(r['status'], 'unknown')
        self.assertIn('Ambiguous source UV', r['reasons'][0])

    def test_boundary_bound_contains_known_gap_boundary_distance(self):
        a = fixture(); a['domain'] = np.arange(3)
        a['boundary'] = np.array(sorted(_edges(a['triangles'][:3])[0]))
        r = source_coverage(**a)
        # New radial edges reach the source square's centre, .5 from its boundary.
        self.assertAlmostEqual(r['boundary']['sampled_max_uv'], .5)
        self.assertGreaterEqual(r['boundary']['upper_bound_uv'], .5)

    def test_changed_triangulation_with_same_feet_can_change_curved_surface(self):
        a = fixture(); tri = np.array([[0, 1, 2], [0, 2, 3]])
        a.update(triangles=tri, domain=np.arange(2), boundary=np.array(sorted(_edges(tri)[0])))
        r = source_coverage(**a)
        self.assertEqual(r['status'], 'measured')
        self.assertGreater(r['deviation']['maximum'], .1)
        self.assertEqual(r['orientation']['reversed_triangle_ids'], [])

    def test_incomplete_boundary_is_unknown(self):
        a = fixture(); a['boundary'] = a['boundary'][:-1]
        self.assertEqual(source_coverage(**a)['status'], 'unknown')

    def test_interior_deviation_uses_declared_source_not_nearest(self):
        a = fixture(); a['positions'][4, 2] += .3
        r = source_coverage(**a)
        self.assertAlmostEqual(r['deviation']['maximum'], .15)
        self.assertEqual(r['deviation']['missing_samples'], 0)

    def test_global_mirrored_source_chart_is_valid(self):
        a = fixture(); a['source_uv'][:, 0] *= -1
        r = source_coverage(**a)
        self.assertEqual(r['status'], 'measured'); self.assertEqual(r['orientation']['reversed_triangle_ids'], [])

    def test_outside_footprint_interior_is_unknown_and_reported(self):
        a = fixture(); a['footprint'] = np.arange(3)
        a['source_boundary'] = np.array(sorted(_edges(a['source_triangles'][:3])[0]))
        r = source_coverage(**a)
        self.assertEqual(r['status'], 'unknown')
        self.assertAlmostEqual(r['coverage']['candidate_outside_area_uv_sum'], .25)
        self.assertGreater(r['deviation']['missing_samples'], 0)

    def test_higher_resolution_tightens_unchanged_bounds(self):
        low, high = self.run_case(resolution=16), self.run_case(resolution=64)
        self.assertLess(high['coverage']['unresolved_area_uv'], low['coverage']['unresolved_area_uv'])

    def test_provenance_changes_with_actual_triangles_and_settings(self):
        a = fixture(); first = source_coverage(**a)
        a['triangles'] = a['triangles'][::-1].copy(); a['resolution'] = 64
        second = source_coverage(**a)
        self.assertNotEqual(first['array_sha256']['triangles'], second['array_sha256']['triangles'])
        self.assertNotEqual(first['parameters'], second['parameters'])

    def test_budget_and_bad_arrays_do_not_appear_measured(self):
        self.assertEqual(self.run_case(max_pairs=1)['status'], 'unknown')
        with self.assertRaises(ValueError): self.run_case(domain=[.5])
        with self.assertRaises(ValueError): self.run_case(provenance={})
        with self.assertRaises(ValueError): self.run_case(resolution=1024)

    def test_finite_large_coordinates_have_finite_serializable_results(self):
        a = fixture(); a['positions'][4, 2] = 1e200
        r = source_coverage(**a)
        json.dumps(r, allow_nan=False)
        self.assertEqual(r['status'], 'measured')
        self.assertTrue(np.isfinite(r['deviation']['maximum']))

    def test_raw_dtype_and_parameters_are_bound(self):
        a = fixture(); first = source_coverage(**a)
        a['source_uv'] = a['source_uv'].astype(np.float32)
        second = source_coverage(**a)
        self.assertNotEqual(first['array_sha256']['source_uv'], second['array_sha256']['source_uv'])
        self.assertNotEqual(first['input_sha256'], second['input_sha256'])
        a['resolution'] = 64
        self.assertNotEqual(second['input_sha256'], source_coverage(**a)['input_sha256'])

    def test_subcell_slit_is_not_certified_covered(self):
        from .triangle_contact import _barycentric
        a = fixture()
        uv = np.array([[0., 0], [.4999, 0], [.4999, 1], [0, 1],
                       [.5001, 0], [1, 0], [1, 1], [.5001, 1]])
        refs, weights, positions = [], [], []
        for point in uv:
            for t, corners in enumerate(a['source_uv'][a['source_triangles']]):
                b = _barycentric(point[None], corners)[0]
                if (b >= 0).all():
                    refs.append(t); weights.append(b); positions.append(b @ a['source_positions'][a['source_triangles'][t]])
                    break
        a.update(positions=np.array(positions), correspondence_triangles=np.array(refs),
                 correspondence_barycentric=np.array(weights), triangles=np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]]),
                 resolution=16)
        a['boundary'] = np.array(sorted(_edges(a['triangles'])[0]))
        r = source_coverage(**a)
        low, high = r['coverage']['gap_area_uv_interval']
        self.assertLessEqual(low, .0002); self.assertGreaterEqual(high, .0002)
        self.assertGreater(r['coverage']['unresolved_area_uv'], 0)

    def test_recorded_preparation_registry(self):
        a = fixture()
        r = prepare_arrays('source_coverage', inputs={}, parameters=a)
        self.assertEqual(r['status'], 'measured')


if __name__ == '__main__':
    unittest.main()
