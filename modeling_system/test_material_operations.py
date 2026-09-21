from copy import deepcopy
from types import SimpleNamespace
import unittest
import numpy as np
from .material_operations import (surface_realization, fit_landmark_field,
    material_trajectory, attachment_motion, install_material_trajectory)


class MaterialTests(unittest.TestCase):
    def test_unused_points_are_reported_but_never_replace_real_surface_errors(self):
        expected = np.zeros((7, 3)); actual = expected.copy()
        actual[6] = 10
        row = surface_realization(expected, actual, [[0,1,2], [2,1,4]], tolerance=1e-7)
        self.assertTrue(row['passed']); self.assertEqual(row['surface']['sample_count'], 4)
        self.assertGreater(row['unused']['maximum'], 10)
        actual[4,0] = .001
        self.assertFalse(surface_realization(expected, actual, [[0,1,2], [2,1,4]], tolerance=1e-7)['passed'])
        for faces in ([], [[0,1,8]], [[0.,1.,2.]]):
            with self.assertRaises(ValueError): surface_realization(expected, actual, faces, tolerance=1e-7)

    def test_affine_preserves_thickness_and_residual_honors_curved_landmarks(self):
        anchors = np.array([[0,0,0], [1,0,.1], [0,1,.1], [1,1,0], [.5,.2,.15], [.2,.5,.04]])
        guide = anchors @ np.diag([2.,3.,1.5]) + [2,1,-1]
        query = np.array([[.25,.3,.1], [.25,.3,.2]])
        affine = fit_landmark_field(query, anchors, guide, method='affine')
        np.testing.assert_allclose(affine['target'], query @ np.diag([2.,3.,1.5]) + [2,1,-1])
        guide[:,2] += .07 * anchors[:,0]**2
        curved = fit_landmark_field(anchors, anchors, guide, method='affine_thin_plate', chart_axes=[0,1])
        np.testing.assert_allclose(curved['target'], guide, atol=1e-12)
        with self.assertRaises(ValueError): fit_landmark_field(query, anchors, guide, method='affine_thin_plate')
        with self.assertRaises(ValueError): fit_landmark_field(query, anchors*0, guide, method='affine')
        with self.assertRaises(ValueError): fit_landmark_field(query, anchors, anchors*[-1,1,1], method='affine')

    def test_shared_material_law_reaches_guides_and_preserves_unselected_motion(self):
        frames = np.array([[[0,0,0],[0,1,0]], [[1,0,1],[1,1,1]], [[2,0,0],[2,1,0]]],float)
        poses = [0,.2,.4,.7,1]
        row = material_trajectory(frames, [0,.4,1], poses)
        np.testing.assert_allclose(row['target'][[0,2,4]], frames)
        np.testing.assert_allclose(row['target'][1], (frames[0]+frames[1])/2)
        baseline = np.full((5,2,3),10.)
        blended = material_trajectory(frames,[0,.4,1],poses,baseline=baseline,support=[0.,.25])['target']
        np.testing.assert_array_equal(blended[:,0],baseline[:,0])
        np.testing.assert_allclose(blended[:,1], .75*baseline[:,1]+.25*row['target'][:,1])
        with self.assertRaises(ValueError): material_trajectory(frames,[0,.4,1],[1.1])

    def test_attachment_pairs_measure_relative_motion_without_nearest_point_guessing(self):
        first = np.zeros((3,3,3)); second = first.copy()
        second[:,2,1] = [0,.1,0]
        result = attachment_motion(first,second,[0,.5,1],first_indices=[0],second_indices=[2])
        np.testing.assert_allclose(result['maximum_by_pose'], [0,.1,0])
        self.assertAlmostEqual(result['relative_motion_maximum'], .2)
        with self.assertRaises(ValueError): attachment_motion(first,second,[0,0,1],first_indices=[0],second_indices=[2])

    def test_occupied_offset_refuses_before_any_graph_or_attribute_mutation(self):
        class Control:
            def path_resolve(self, path): return 0.
        class Graph:
            nodes = [SimpleNamespace(bl_idname='GeometryNodeSetPosition',label='owner',inputs={
                'Position': SimpleNamespace(is_linked=True),
                'Offset': SimpleNamespace(is_linked=True, default_value=(0,0,0))})]
            def copy(self): raise AssertionError('Must refuse before mutation')
        ob = SimpleNamespace(modifiers={'field':SimpleNamespace(node_group=Graph())},
            data=SimpleNamespace(vertices=[0,1,2],attributes={}))
        with self.assertRaises(ValueError):
            install_material_trajectory(None,ob=ob,control=Control(),modifier_name='field',position_label='owner',
                control_path='pose',attribute_prefix='material_',keyframes=np.zeros((2,3,3)),knots=[0,1],support=[1,1,1])


if __name__ == '__main__': unittest.main()
