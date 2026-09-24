"""In-between construction from two established poses: pace, rolled paths and rigid pieces (synthetic, no IO)."""
import unittest
import numpy as np
from .motion_paths import motion_pace, path_positions, hinge_motion, keep_clearance
from .preparation import ARRAY_OPERATIONS


def sheet(nx=9, ny=7, spacing=.1):
    x, y = np.meshgrid(np.arange(nx) * spacing, np.arange(ny) * spacing)
    points = np.c_[x.ravel(), y.ravel(), np.zeros(nx * ny)]
    tri = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i; tri += [[a, a + 1, a + nx + 1], [a, a + nx + 1, a + nx]]
    return points, np.array(tri)


def rotate_x(points, angle, centre=(0., 0., 0.)):
    c, s = np.cos(angle), np.sin(angle); R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (np.asarray(points) - centre) @ R.T + centre


class MotionPaceTests(unittest.TestCase):
    def test_recovers_each_regions_pace_and_fills_points_that_barely_move(self):
        rest, _ = sheet(); end = rest + [0., 0., .4]
        phases = np.linspace(0, 1, 6)
        early = rest[:, 0] < .35                                            # one region runs ahead of the other
        progress = np.where(early[None], np.sqrt(phases)[:, None], (phases ** 2)[:, None])
        samples = rest[None] + progress[..., None] * (end - rest)[None]
        still = 30; end_still = end.copy(); end_still[still] = rest[still]  # a point that does not move at all
        samples[:, still] = rest[still]
        result = motion_pace(rest, end_still, samples, sigma=.02, cutoff=.12)
        pace = result['pace']
        np.testing.assert_allclose(pace[:, early & (np.arange(len(rest)) != still)], np.sqrt(phases)[:, None]
                                   .repeat((early & (np.arange(len(rest)) != still)).sum(), 1), atol=1e-4)
        self.assertTrue(np.all(np.diff(pace, axis=0) >= 0)); np.testing.assert_array_equal(pace[0], 0); np.testing.assert_array_equal(pace[-1], 1)
        self.assertTrue(0 < pace[2, still] < 1)                              # took its neighbours' pace
        self.assertEqual(result['public_metrics']['moving_points'], len(rest) - 1)

    def test_overshoot_is_clipped_and_made_monotone(self):
        rest, _ = sheet(4, 3); end = rest + [0., .2, 0.]
        progress = np.array([0, .6, 1.3, .9, 1.])                           # overshoots, then comes back
        samples = rest[None] + progress[:, None, None] * (end - rest)[None]
        result = motion_pace(rest, end, samples, sigma=.05)
        np.testing.assert_allclose(result['pace'][:, 0], [0, .6, 1, 1, 1])
        self.assertGreater(result['public_metrics']['overshoot_samples_clipped'], 0)

    def test_refuses_bad_inputs(self):
        rest, _ = sheet(3, 3)
        with self.assertRaisesRegex(ValueError, 'two or more phases'):
            motion_pace(rest, rest + 1, (rest + 1)[None], sigma=.1)
        with self.assertRaisesRegex(ValueError, 'positive smoothing'):
            motion_pace(rest, rest + 1, np.stack([rest, rest + 1]), sigma=0)


class PathPositionTests(unittest.TestCase):
    def test_rolled_path_stays_outside_the_obstacle_the_chord_cuts(self):
        # a band on a unit sphere turned 100 degrees about the sphere's centre
        rest = np.array([[x, 0., 1.] for x in (-.2, 0., .2)]); rest /= np.linalg.norm(rest, axis=1, keepdims=True)
        end = rotate_x(rest, np.radians(100))
        pace = np.full(3, .5)
        chord = path_positions(rest, end, pace)['positions']
        rolled = path_positions(rest, end, pace, pivot=[0., 0., 0.], axis=[1., 0., 0.])
        self.assertTrue(np.all(np.linalg.norm(chord, axis=1) < .7))       # the chord cuts deep into the sphere
        np.testing.assert_allclose(np.linalg.norm(rolled['positions'], axis=1), 1, atol=1e-12)
        np.testing.assert_allclose(rolled['positions'], rotate_x(rest, np.radians(50)), atol=1e-12)   # no sideways drift
        self.assertEqual(rolled['public_metrics']['chord_points_closer_to_pivot_than_both_ends'], 3)
        sphere = path_positions(rest, end, pace, pivot=[0., 0., 0.])['positions']   # slerp keeps the distance ...
        np.testing.assert_allclose(np.linalg.norm(sphere, axis=1), 1, atol=1e-12)
        self.assertGreater(np.abs(sphere[0, 0]), np.abs(rest[0, 0]) + .05)  # ... but a great circle swings sideways

    def test_ends_exact_blend_and_phase_rows(self):
        rest, _ = sheet(4, 3); rest = rest + [0, 0, 1.]; end = rotate_x(rest, .8)
        rows = np.stack([np.zeros(len(rest)), np.full(len(rest), .3), np.ones(len(rest))])
        out = path_positions(rest, end, rows, pivot=[0., 0., 0.])['positions']
        np.testing.assert_allclose(out[0], rest, atol=1e-12); np.testing.assert_allclose(out[-1], end, atol=1e-12)
        straight = path_positions(rest, end, rows[1], pivot=[0., 0., 0.], roll_weight=np.zeros(len(rest)))['positions']
        np.testing.assert_allclose(straight, .7 * rest + .3 * end, atol=1e-12)

    def test_refuses_bad_inputs(self):
        rest, _ = sheet(3, 3)
        with self.assertRaisesRegex(ValueError, 'Pace must be'):
            path_positions(rest, rest + 1, np.full(len(rest), 1.5))
        with self.assertRaisesRegex(ValueError, 'half turn'):
            path_positions(np.array([[0, 0, 1.]]), np.array([[0, 0, -1.]]), np.array([.5]), pivot=[0, 0, 0], axis=[1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'needs a pivot'):
            path_positions(rest, rest + 1, np.zeros(len(rest)), roll_weight=np.ones(len(rest)))
        with self.assertRaisesRegex(ValueError, 'Opposite directions'):
            path_positions(np.array([[0, 0, 1.]]), np.array([[0, 0, -1.]]), np.array([.5]), pivot=[0, 0, 0])


class HingeMotionTests(unittest.TestCase):
    def test_rigid_piece_turns_as_one_and_keeps_its_shape(self):
        rest, tri = sheet(5, 4); pivot = np.array([0., 0., 0.])
        end = rotate_x(rest, np.radians(60), pivot)
        result = hinge_motion(rest, end, np.arange(len(rest)), pivot, np.full(len(rest), .5))
        self.assertAlmostEqual(result['public_metrics']['angle_degrees'], 60, places=6)
        self.assertLess(result['public_metrics']['residual_max'], 1e-12)
        mid = result['positions']; np.testing.assert_allclose(mid, rotate_x(rest, np.radians(30), pivot), atol=1e-12)
        edges = np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]
        length = lambda p: np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
        np.testing.assert_allclose(length(mid), length(rest), atol=1e-12)
        chord_mid = .5 * (rest + end)                                       # the chord shrinks the piece instead
        self.assertLess((length(chord_mid) / length(rest)).min(), np.cos(np.radians(30)) + 1e-9)

    def test_residual_carried_linearly_ends_exact_and_blend(self):
        rest, _ = sheet(5, 4); pivot = np.array([0., -.1, 0.])
        end = rotate_x(rest, np.radians(40), pivot); end[:, 0] *= 1.1          # not rigid: stretched along x
        pace = np.stack([np.zeros(len(rest)), np.full(len(rest), .4), np.ones(len(rest))])
        result = hinge_motion(rest, end, np.arange(len(rest)), pivot, pace)
        np.testing.assert_allclose(result['positions'][0], rest, atol=1e-12); np.testing.assert_allclose(result['positions'][-1], end, atol=1e-12)
        self.assertGreater(result['public_metrics']['residual_rms'], 0); self.assertLess(result['public_metrics']['rigid_share'], 1)
        base = np.repeat(rest[None], 3, 0); off = hinge_motion(rest, end, np.arange(len(rest)), pivot, pace, base=base,
                                                              weights=np.zeros(len(rest)))
        np.testing.assert_allclose(off['positions'], base)

    def test_refuses_bad_inputs(self):
        rest, _ = sheet(3, 3)
        with self.assertRaisesRegex(ValueError, 'three member'):
            hinge_motion(rest, rest + 1, np.array([0, 1]), [0, 0, 0], np.zeros(len(rest)))
        with self.assertRaisesRegex(ValueError, 'Weights must'):
            hinge_motion(rest, rest + 1, np.arange(4), [0, 0, 0], np.zeros(len(rest)), weights=np.full(len(rest), 2.))

    def test_registered_as_array_preparation_operations(self):
        for name in ('motion_pace', 'path_positions', 'hinge_motion', 'keep_clearance'):
            self.assertIn(name, ARRAY_OPERATIONS)



class ClearanceTests(unittest.TestCase):
    def setUp(self):
        u = np.random.default_rng(3).normal(size=(6000, 3)); self.sphere = u / np.linalg.norm(u, axis=1, keepdims=True)

    def test_points_inside_are_pushed_out_along_their_direction_and_others_keep_their_bytes(self):
        pts = np.array([[0, 0, .5], [0, 0, 1.2], [.6, .6, 0], [0, 0, .99]])
        result = keep_clearance(pts, self.sphere, [0, 0, 0], .05, angular_radius_degrees=4.)
        np.testing.assert_allclose(np.linalg.norm(result['positions'][[0, 2, 3]], axis=1), 1.05, atol=1e-9)
        np.testing.assert_array_equal(result['positions'][1], pts[1])
        np.testing.assert_allclose(np.cross(result['positions'][2], pts[2]), 0, atol=1e-12)     # radial push
        self.assertEqual(result['public_metrics']['pushed_per_phase'], [3])
        np.testing.assert_allclose(result['envelope'], 1, atol=1e-12)          # the unit sphere's measured envelope

    def test_soft_ramp_is_continuous_and_negative_clearance_opts_out(self):
        r = np.linspace(.95, 1.15, 81); pts = np.c_[np.zeros_like(r), np.zeros_like(r), r]
        out = np.linalg.norm(keep_clearance(pts, self.sphere, [0, 0, 0], .05, angular_radius_degrees=4., soft=.02)['positions'], axis=1)
        self.assertTrue(np.all(np.diff(out) >= -1e-12)); self.assertTrue(np.all(out >= 1.05 - 1e-12))
        np.testing.assert_allclose(out[r >= 1.07 + 1e-9], r[r >= 1.07 + 1e-9])
        skip = keep_clearance(pts, self.sphere, [0, 0, 0], -np.ones(len(r)), angular_radius_degrees=4.)['positions']
        np.testing.assert_array_equal(skip, pts)

    def test_directions_the_obstacle_does_not_cover_are_left_alone(self):
        cap = self.sphere[self.sphere[:, 2] > .5]
        pts = np.array([[0, 0, .5], [0, 0, -.5]])
        result = keep_clearance(pts, cap, [0, 0, 0], .0, angular_radius_degrees=4.)
        self.assertGreater(np.linalg.norm(result['positions'][0]), .99); np.testing.assert_array_equal(result['positions'][1], pts[1])
        self.assertEqual(result['public_metrics']['uncovered_points'], 1); self.assertTrue(np.isnan(result['envelope'][1]))

    def test_refuses_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, 'four finite obstacle'):
            keep_clearance(np.zeros((1, 3)), np.zeros((2, 3)), [0, 0, 0], .1)
        with self.assertRaisesRegex(ValueError, 'angular_radius'):
            keep_clearance(np.zeros((1, 3)), self.sphere, [0, 0, 0], .1, angular_radius_degrees=0)


if __name__ == '__main__':
    unittest.main()
