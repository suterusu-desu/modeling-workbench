"""In-between construction from two established poses: pace, rolled paths and rigid pieces (synthetic, no IO)."""
import unittest
import numpy as np
from .motion_paths import (motion_pace, path_positions, hinge_motion, keep_clearance, end_clearance, schedule_pace,
                           shared_schedule, smooth_step, travel_weight)
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
        for name in ('motion_pace', 'path_positions', 'hinge_motion', 'keep_clearance', 'end_clearance'):
            self.assertIn(name, ARRAY_OPERATIONS)



def fibonacci_sphere(n):
    k = np.arange(n) + .5; z = 1 - 2 * k / n; t = np.pi * (1 + 5 ** .5) * k; s = np.sqrt(1 - z * z)
    return np.c_[s * np.cos(t), s * np.sin(t), z]


def arc(start_deg, stop_deg, n, radius=1.):
    """Points in the x-z plane at the given angles from +z towards +x."""
    a = np.radians(np.linspace(start_deg, stop_deg, n))
    return radius * np.c_[np.sin(a), np.zeros(n), np.cos(a)]


class ClearanceTests(unittest.TestCase):
    def setUp(self):
        self.sphere = fibonacci_sphere(6000)

    def test_points_inside_are_pushed_out_along_their_direction_and_others_keep_their_bytes(self):
        pts = np.array([[0, 0, .5], [0, 0, 1.2], [.6, .6, 0], [0, 0, .99]])
        for envelope in ('smooth', 'max'):
            with self.subTest(envelope=envelope):
                result = keep_clearance(pts, self.sphere, [0, 0, 0], .05, angular_radius_degrees=4., envelope=envelope)
                np.testing.assert_allclose(np.linalg.norm(result['positions'][[0, 2, 3]], axis=1), 1.05, atol=1e-9)
                np.testing.assert_array_equal(result['positions'][1], pts[1])
                np.testing.assert_allclose(np.cross(result['positions'][2], pts[2]), 0, atol=1e-12)     # radial push
                self.assertEqual(result['public_metrics']['pushed_per_phase'], [3])
                np.testing.assert_allclose(result['envelope'], 1, atol=1e-12)  # the unit sphere's measured envelope

    def test_soft_ramp_is_continuous_and_negative_clearance_opts_out(self):
        r = np.linspace(.95, 1.15, 81); pts = np.c_[np.zeros_like(r), np.zeros_like(r), r]
        for envelope in ('smooth', 'max'):
            with self.subTest(envelope=envelope):
                out = np.linalg.norm(keep_clearance(pts, self.sphere, [0, 0, 0], .05, angular_radius_degrees=4., soft=.02,
                                                    envelope=envelope)['positions'], axis=1)
                self.assertTrue(np.all(np.diff(out) >= -1e-12)); self.assertTrue(np.all(out >= 1.05 - 1e-12))
                np.testing.assert_allclose(out[r >= 1.07 + 1e-9], r[r >= 1.07 + 1e-9])
                skip = keep_clearance(pts, self.sphere, [0, 0, 0], -np.ones(len(r)), angular_radius_degrees=4.,
                                      envelope=envelope)['positions']
                np.testing.assert_array_equal(skip, pts)

    def test_directions_the_obstacle_does_not_cover_are_left_alone(self):
        cap = self.sphere[self.sphere[:, 2] > .5]
        pts = np.array([[0, 0, .5], [0, 0, -.5]])
        for envelope in ('smooth', 'max'):
            with self.subTest(envelope=envelope):
                result = keep_clearance(pts, cap, [0, 0, 0], .0, angular_radius_degrees=4., envelope=envelope)
                self.assertGreater(np.linalg.norm(result['positions'][0]), .99)
                np.testing.assert_array_equal(result['positions'][1], pts[1])
                self.assertEqual(result['public_metrics']['uncovered_points'], 1); self.assertTrue(np.isnan(result['envelope'][1]))

    def test_smooth_envelope_has_no_steps_where_the_maximum_jumps(self):
        # A lumpy obstacle: the largest distance in a cone jumps as its points enter and leave; pushes follow the jumps.
        lumpy = self.sphere * (1 + .02 * np.random.default_rng(5).uniform(-1, 1, len(self.sphere)))[:, None]
        pts = arc(-10, 10, 4001, .9)                                        # inside it: every point is pushed
        steps = {}
        for envelope in ('smooth', 'max'):
            out = keep_clearance(pts, lumpy, [0, 0, 0], .01, angular_radius_degrees=4., envelope=envelope)
            steps[envelope] = np.abs(np.diff(out['envelope'])).max()
            self.assertEqual(out['public_metrics']['pushed_per_phase'], [len(pts)])
        self.assertGreater(steps['max'], 2e-3)                              # a visible step at this sampling
        self.assertLess(steps['smooth'], 2e-4)                              # continuous: tiny change per .005 degree

    def test_smooth_pushes_fade_out_continuously_at_the_obstacles_edge(self):
        cap = self.sphere[self.sphere[:, 2] > np.cos(np.radians(30))]       # a 30-degree cap of the unit sphere
        steps = {}
        for n in (2001, 4001):                                              # the same crossing of its edge, twice as fine
            pts = arc(20, 40, n, .9)
            push = {e: keep_clearance(pts, cap, [0, 0, 0], .05, angular_radius_degrees=4., envelope=e)['push'] for e in ('smooth', 'max')}
            self.assertAlmostEqual(push['smooth'][0], .15, places=9); self.assertEqual(push['smooth'][-1], 0.)
            steps[n] = {e: np.abs(np.diff(v)).max() for e, v in push.items()}
        self.assertLess(steps[2001]['smooth'], .01)                          # fades over the kernel's reach
        self.assertLess(steps[4001]['smooth'], .6 * steps[2001]['smooth'])  # continuous: finer sampling, smaller steps
        self.assertGreater(min(steps[2001]['max'], steps[4001]['max']), .1)  # the maximum drops the whole push at once

    def test_end_clearance_is_the_smaller_capped_end_value_and_skips_inside_or_uncovered(self):
        moved = self.sphere + [0, 0, .02]                                    # the obstacle moves between the poses
        rest = np.array([[0, 0, 1.03], [0, 0, 1.2], [0, 0, .9], [0, 0, -1.3]])
        end = np.array([[0, 0, 1.06], [0, 0, 1.1], [0, 0, 1.1], [0, 0, -1.3]])
        cap_half = self.sphere[self.sphere[:, 2] > 0]
        out = end_clearance(rest, end, cap_half, cap_half + [0, 0, .02], [0, 0, 0], angular_radius_degrees=4.)
        np.testing.assert_allclose(out['reference_clearance'][:3], [.03, .2, -.1], atol=1e-9)
        np.testing.assert_allclose(out['end_clearance'][:3], [.04, .08, .08], atol=1e-3)
        np.testing.assert_allclose(out['clearance'][:2], [.03, .08], atol=1e-3)
        self.assertEqual(list(out['clearance'][2:]), [-1., -1.])            # inside at rest; uncovered
        capped = end_clearance(rest, end, self.sphere, moved, [0, 0, 0], cap=.05, angular_radius_degrees=4.)
        np.testing.assert_allclose(capped['clearance'][:2], [.03, .05], atol=1e-3)
        self.assertEqual(capped['public_metrics']['capped'], 2)             # the far points keep only the margin
        with self.assertRaisesRegex(ValueError, 'cap must'):
            end_clearance(rest, end, self.sphere, moved, [0, 0, 0], cap=-1.)

    def test_refuses_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, 'four finite obstacle'):
            keep_clearance(np.zeros((1, 3)), np.zeros((2, 3)), [0, 0, 0], .1)
        with self.assertRaisesRegex(ValueError, 'angular_radius'):
            keep_clearance(np.zeros((1, 3)), self.sphere, [0, 0, 0], .1, angular_radius_degrees=0)
        with self.assertRaisesRegex(ValueError, "Envelope must"):
            keep_clearance(np.ones((1, 3)), self.sphere, [0, 0, 0], .1, envelope='mean')


class ScheduleTests(unittest.TestCase):
    phases = np.linspace(0, 1, 11)

    def band(self):
        """Six points along a band: the last two lag (stay put until late), as a corner that closes in the last tenth."""
        g = self.phases[:, None]
        early = np.repeat(g, 4, axis=1)
        late = np.repeat(np.clip((g - .9) / .1, 0, 1), 2, axis=1)
        return np.hstack([early, late])

    def test_smooth_step_values_endpoints_and_broadcast(self):
        np.testing.assert_allclose(smooth_step([-1, 0, .25, .5, 1, 2], 0, 1), [0, 0, .15625, .5, 1, 1])
        onset = np.array([.6, .8]); table = smooth_step(self.phases[:, None], onset - .2, onset)
        self.assertEqual(table.shape, (11, 2))
        np.testing.assert_allclose(table[[4, 6, 8], 0], [0, 1, 1], atol=1e-12)
        np.testing.assert_allclose(table[[6, 8], 1], [0, 1], atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'start < end'): smooth_step([0.], 1, 1)
        with self.assertRaisesRegex(ValueError, 'start < end'): smooth_step([np.nan], 0, 1)

    def test_blend_moves_the_weighted_region_onto_the_shared_schedule_and_keeps_the_rest(self):
        pace = self.band(); common = shared_schedule(pace, [0, 1, 2, 3])
        np.testing.assert_allclose(common, self.phases)
        w = np.array([0, 0, 0, 0, .5, 1.])
        r = schedule_pace(pace, common, w)
        out = r['pace']
        np.testing.assert_array_equal(out[:, :4], pace[:, :4])            # weight 0: exact bytes
        np.testing.assert_allclose(out[:, 5], self.phases)                # weight 1: the schedule
        np.testing.assert_allclose(out[:, 4], .5 * pace[:, 4] + .5 * self.phases)
        self.assertTrue((np.diff(out, axis=0) >= 0).all())
        self.assertEqual(r['public_metrics']['changed_points'], 2)
        self.assertAlmostEqual(r['public_metrics']['max_change'], .9)

    def test_floor_raises_only_where_the_region_lags_and_never_delays(self):
        pace = self.band(); late = smooth_step(self.phases, .5, .8)
        out = schedule_pace(pace, late, np.ones(6), mode='floor')['pace']
        np.testing.assert_array_equal(out, np.maximum(pace, late[:, None]))
        np.testing.assert_array_equal(out[:6, :4], pace[:6, :4])          # ahead of the floor until .5
        np.testing.assert_allclose(out[8:, 4:], 1.)                       # the lagging points closed by .8
        per_point = smooth_step(self.phases[:, None], np.full(6, .3), np.full(6, .6))
        np.testing.assert_allclose(schedule_pace(pace, per_point, np.full(6, .5), mode='floor')['pace'][-1], 1.)

    def test_median_and_weighted_mean_schedules_are_monotone(self):
        pace = self.band()
        np.testing.assert_allclose(shared_schedule(pace, np.array([True, True, False, False, True, True])),
                                   np.median(pace[:, [0, 1, 4, 5]], axis=1))
        mean = shared_schedule(pace, [0, 5], weights=[3., 1.], statistic='mean')
        np.testing.assert_allclose(mean, .75 * pace[:, 0] + .25 * pace[:, 5])
        self.assertTrue((np.diff(mean) >= 0).all())

    def test_refuses_bad_inputs(self):
        pace = self.band(); w = np.ones(6)
        with self.assertRaisesRegex(ValueError, 'monotone over phases for every point'): schedule_pace(pace[::-1], self.phases, w)
        with self.assertRaisesRegex(ValueError, 'schedule must be monotone'): schedule_pace(pace, self.phases[::-1], w)
        with self.assertRaisesRegex(ValueError, 'schedule must be values'): schedule_pace(pace, self.phases * 2, w)
        with self.assertRaisesRegex(ValueError, 'Weights must'): schedule_pace(pace, self.phases, np.full(6, 1.5))
        with self.assertRaisesRegex(ValueError, 'Weights must'): schedule_pace(pace, self.phases, np.ones(5))
        with self.assertRaisesRegex(ValueError, 'mode must'): schedule_pace(pace, self.phases, w, mode='max')
        with self.assertRaisesRegex(ValueError, 'Members'): shared_schedule(pace, [])
        with self.assertRaisesRegex(ValueError, 'Members'): shared_schedule(pace, [9])
        with self.assertRaisesRegex(ValueError, 'statistic'): shared_schedule(pace, [0], statistic='mode')
        with self.assertRaisesRegex(ValueError, 'Weights must be nonnegative'): shared_schedule(pace, [0, 1], weights=[0, 0], statistic='mean')

    def test_registered_as_array_preparation_operations(self):
        for name in ('schedule_pace', 'shared_schedule', 'travel_weight'):
            self.assertIn(name, ARRAY_OPERATIONS)


class TravelWeightTests(unittest.TestCase):
    def test_smoothed_travel_stops_neighbours_alternating_near_the_threshold(self):
        points, _ = sheet(nx=31, ny=11, spacing=.01)
        x = points[:, 0]; noise = np.where((np.arange(len(points)) % 2) == 0, .0004, -.0004)
        travel = np.clip(.004 * x / x.max(), 0, None) + noise                  # small travel, noisy around the threshold
        raw = travel_weight(points, travel, low=.0012, high=.0022)
        smooth = travel_weight(points, travel, low=.0012, high=.0022, sigma=.012)
        np.testing.assert_allclose(raw['weight'], smooth_step(travel, .0012, .0022))
        self.assertIsNone(raw['public_metrics']['neighbour_jump_p99_before'])
        m = smooth['public_metrics']
        self.assertGreater(m['neighbour_jump_p99_before'], 3 * m['neighbour_jump_p99_after'])
        self.assertTrue(np.all(smooth['weight'][x < .005] == 0) and np.all(smooth['weight'][x > .27] == 1))

    def test_refusals(self):
        points, _ = sheet(nx=3, ny=3)
        with self.assertRaisesRegex(ValueError, 'one travel value'): travel_weight(points, np.zeros(4), low=0., high=1.)
        with self.assertRaisesRegex(ValueError, 'low < high'): travel_weight(points, np.zeros(9), low=1., high=1.)
        with self.assertRaisesRegex(ValueError, 'cutoff'): travel_weight(points, np.zeros(9), low=0., high=1., sigma=.1, cutoff=-1.)


if __name__ == '__main__':
    unittest.main()
