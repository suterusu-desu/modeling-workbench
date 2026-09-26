"""Registration of a generated mesh onto an accepted neutral, on a synthetic head with a changed eye region."""
import unittest

import numpy as np

from .registration import apply_registration, inside_boxes, register_rigid


def head(n=40):
    """An egg-shaped head (z up, face toward -y) with a nose bump, sampled on a latitude-longitude grid."""
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), np.linspace(.15, np.pi - .15, n))
    x, y, z = .09 * np.sin(v) * np.cos(u), .11 * np.sin(v) * np.sin(u), .13 * np.cos(v)
    y = y - .02 * np.exp(-((x / .015) ** 2 + ((z + .01) / .02) ** 2)) * (y < 0)
    return np.c_[x.ravel(), y.ravel(), z.ravel() + .6]


EYES = [{'min': [.015, -.2, .6], 'max': [.06, -.05, .65], 'mirror': 0}]


def rotation(axis, degrees):
    a = np.radians(degrees); c, s = np.cos(a), np.sin(a)
    return {'x': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.reference = head()
        self.eyes = inside_boxes(self.reference, EYES)
        changed = self.reference.copy(); changed[self.eyes, 1] -= .006           # the pose guide's lids come forward
        self.changed = changed                                                     # generator output is upright:
        self.generated = ((changed - [0, 0, .6]) @ rotation('z', 7.).T) / 2.5 + [.3, -.1, 1.2]   # own yaw, scale, place

    def test_recovers_the_transform_on_stationary_anatomy_and_reports_the_changed_region(self):
        result = register_rigid(self.generated, self.reference, exclude=EYES)
        m = result['public_metrics']
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['scale'], 2.5, places=6)
        self.assertLess(m['stationary_distance_p95'], 1e-6)
        self.assertGreater(m['excluded_distance_median'], .004)                 # the guide's difference survives
        still = ~self.eyes
        np.testing.assert_allclose(result['positions'][still], self.reference[still], atol=1e-6)
        np.testing.assert_allclose(apply_registration(self.generated, result), result['positions'], atol=1e-12)

    def test_a_stationary_mask_on_the_reference_does_the_same(self):
        result = register_rigid(self.generated, self.reference, stationary=~self.eyes)
        self.assertLess(result['public_metrics']['stationary_distance_p95'], 1e-6)

    def test_a_tilted_mesh_needs_its_scale_given(self):
        tilted = ((self.changed - [0, 0, .6]) @ (rotation('z', 7.) @ rotation('x', -6.)).T) / 2.5 + [.3, -.1, 1.2]
        by_height = register_rigid(tilted, self.reference, exclude=EYES)
        given = register_rigid(tilted, self.reference, exclude=EYES, scale=2.5)
        self.assertGreater(by_height['public_metrics']['stationary_distance_p95'], 1e-4)
        self.assertLess(given['public_metrics']['stationary_distance_p95'], 1e-6)

    def test_registering_on_the_changed_region_too_drags_the_fit(self):
        loose = register_rigid(self.generated, self.reference, trim=1.)['public_metrics']
        self.assertGreater(loose['whole_distance_p95'], 1e-4)

    def test_boxes_and_refusals(self):
        self.assertEqual(inside_boxes([[.03, -.1, .62], [-.03, -.1, .62], [0, 0, 0]], EYES).tolist(), [True, True, False])
        with self.assertRaisesRegex(ValueError, 'min <= max'):
            inside_boxes([[0, 0, 0]], [{'min': [1, 0, 0], 'max': [0, 1, 1]}])
        with self.assertRaisesRegex(ValueError, 'three finite'):
            register_rigid([[0, 0, 0]], self.reference)
        with self.assertRaisesRegex(ValueError, 'Scale must'):
            register_rigid(self.generated, self.reference, scale=-1.)
        with self.assertRaisesRegex(ValueError, 'stationary vertex'):
            register_rigid(self.generated, self.reference, stationary=np.zeros(len(self.reference), bool))
        with self.assertRaisesRegex(ValueError, "initial must"):
            register_rigid(self.generated, self.reference, initial='centroid')


if __name__ == '__main__':
    unittest.main()
