import unittest
import numpy as np
from .guide_fitting import (triangle_sections, closest_on_segments, point_at_coordinate,
    section_intersections, interpolate_supported, remap_material_path, prepare_section_fit)


class GuideFittingTests(unittest.TestCase):
    def test_exact_vertex_edge_and_coplanar_section_provenance(self):
        points = np.array([[0, 0, 0], [1, 1, 0], [-1, 1, 0], [0, 2, 0], [0, 0, 1]])
        result = triangle_sections(points, [[0, 1, 2], [0, 3, 1], [0, 3, 4]], 0)
        np.testing.assert_allclose(result['segments'], [[[0, 0, 0], [0, 1, 0]], [[0, 0, 0], [0, 2, 0]]])
        self.assertEqual(result['triangles'].tolist(), [0, 1])
        self.assertEqual(result['coplanar_triangles'], [2])

    def test_projection_and_coordinate_parallel_segment_are_finite(self):
        segments = np.array([[[0, 2, 0], [0, 2, 2]], [[0, 3, 1], [0, 3, 1]]])
        hit, parent = point_at_coordinate(2, segments, .5)
        np.testing.assert_allclose(hit, [0, 2, .5]); self.assertEqual(parent, 0)
        self.assertEqual(point_at_coordinate(4, segments, .5), (None, None))
        hit, parent = closest_on_segments([0, 3, 1], segments)
        np.testing.assert_allclose(hit, [0, 3, 1]); self.assertEqual(parent, 1)

    def test_intersection_ancestry_and_collinear_ambiguity(self):
        first = [[[0, 0, 0], [0, 2, 0]]]
        second = [[[0, 1, -1], [0, 1, 1]], [[0, 1, 0], [0, 3, 0]]]
        result = section_intersections(first, second)
        np.testing.assert_allclose(result['points'], [[0, 1, 0]])
        self.assertEqual(result['segment_pairs'], [[0, 0]])
        self.assertEqual(result['ambiguous_overlaps'], [[0, 1]])
        with self.assertRaises(ValueError):
            section_intersections(first, [[[1, 1, -1], [1, 1, 1]]])

    def test_bounded_gap_preserves_direct_data_and_parent_support(self):
        points, records = interpolate_supported([0, 1, 3], [[0, 0, 0], [np.nan]*3, [3, 6, 0]],
            [True, False, True], [[10], None, [20]], max_gap=3, max_missing=1)
        np.testing.assert_allclose(points, [[0, 0, 0], [1, 2, 0], [3, 6, 0]])
        self.assertEqual(records[1]['parents'], [[10], [20]])
        self.assertEqual(records[1]['kind'], 'interpolated')
        for gap, missing, mask in [(2, 1, [True, False, True]), (3, 0, [True, False, True]), (3, 1, [False, True, True])]:
            with self.assertRaises(ValueError):
                interpolate_supported([0, 1, 3], np.zeros((3, 3)), mask, [[10], [15], [20]], max_gap=gap, max_missing=missing)

    def test_material_remap_keeps_order_and_exact_endpoints(self):
        result = remap_material_path([[0, 0, 0], [1, 0, 0], [3, 0, 0]],
                                     [[0, 0, 0], [0, 2, 0], [0, 2, 4]])
        np.testing.assert_allclose(result['points'], [[0, 0, 0], [0, 2, 0], [0, 2, 4]])
        np.testing.assert_allclose(result['material_fraction'], [0, 1/3, 1])

    def test_coordinate_space_changes_motion_and_zero_weight_is_exact(self):
        baseline = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
        target = baseline + [0, 1, 0]
        args = ([0, 1, 2], baseline, target, [True]*3, [[0], [1], [2]], [0, 1, 1])
        matrix = np.eye(4); matrix[0, 0] = 2; matrix[:3, 3] = [8, 9, 10]
        world = prepare_section_fit(*args, max_gap=1, max_missing=0, space='world', object_matrix=matrix)
        frames = np.repeat(np.eye(3)[None], 3, axis=0)
        frames[1] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        local = prepare_section_fit(*args, max_gap=1, max_missing=0, space='attachment', frames=frames)
        np.testing.assert_array_equal(world['target'][0], baseline[0])
        np.testing.assert_allclose(world['encoded_displacement'] @ matrix[:3, :3].T, world['world_displacement'])
        np.testing.assert_allclose(np.einsum('nij,nj->ni', frames, local['encoded_displacement']), world['world_displacement'])
        self.assertFalse(np.allclose(local['encoded_displacement'], world['encoded_displacement']))
        self.assertFalse(world['appearance_accepted'])
        with self.assertRaises(ValueError):
            prepare_section_fit(*args, max_gap=1, max_missing=0, space='world')


if __name__ == '__main__': unittest.main()
