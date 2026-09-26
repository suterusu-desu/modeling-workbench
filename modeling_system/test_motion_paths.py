"""In-between construction from two established poses: pace, rolled paths and rigid pieces (synthetic, no IO)."""
import unittest
import numpy as np
from .motion_paths import (motion_pace, path_positions, hinge_motion, keep_clearance, end_clearance, schedule_pace,
                           shared_schedule, smooth_step, travel_weight, hinge_change, hinge_landing, hinge_carry, hinge_lift)
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


def cylinder_points(x, radius, angle_degrees):
    """Points at axial positions x, distance `radius` from the x axis, at an angle about +x measured right-handed from -y
    (the front): 0 is in front, positive angles turn the front point downward (toward -z)."""
    a = np.radians(np.broadcast_to(angle_degrees, np.shape(x))); r = np.broadcast_to(radius, np.shape(x))
    return np.c_[x, -r * np.cos(a), -r * np.sin(a)]


def rodrigues(v, k, angle):
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return v * c + np.cross(k, v) * s + k * (v @ k)[:, None] * (1 - c)


class HingeConstructionTests(unittest.TestCase):
    def test_hinge_change_recovers_a_right_handed_turn_radial_and_axial_change(self):
        rest = cylinder_points(np.linspace(-.5, .5, 11), 1.1, np.linspace(-50, -20, 11))
        end = rotate_x(rest, np.radians(30))                                  # right-handed about +x
        end = end + np.c_[np.full(11, .02), np.zeros((11, 2))]                 # slide along the axis
        grow = end.copy(); grow[:, 1:] *= 1.05                                  # and move out from it
        result = hinge_change(rest, grow, [0., 0., 0.], [2., 0., 0.])
        np.testing.assert_allclose(result['turn'], np.radians(30), atol=1e-12)
        np.testing.assert_allclose(result['axial'], .02, atol=1e-12)
        np.testing.assert_allclose(result['radial'], 1.1 * .05, atol=1e-12)
        self.assertAlmostEqual(result['public_metrics']['turn_degrees_max'], 30., places=9)

    def test_hinge_change_refusals(self):
        rest = cylinder_points(np.linspace(-.5, .5, 5), 1., 0.)
        with self.assertRaisesRegex(ValueError, 'on the hinge axis'):
            hinge_change(np.r_[rest, [[.2, 0., 0.]]], np.r_[rest, [[.3, 0., 0.]]], [0, 0, 0], [1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'half turn'):
            hinge_change(rest, rotate_x(rest, np.pi), [0, 0, 0], [1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'nonzero hinge axis'):
            hinge_change(rest, rest, [0, 0, 0], [0, 0, 0])

    def test_edge_lands_on_the_landing_line_and_the_band_follows_by_its_weight(self):
        x = np.linspace(-.6, .6, 13)
        edge = cylinder_points(x, 1.1, -40.)                                    # upper margin, above the front
        band = cylinder_points(x, 1.15, -55.)                                   # skin above it
        still = cylinder_points(x, 1.2, 60.)                                    # below the landing line: stays
        positions = np.r_[edge, band, still]; n = len(x)
        landing = cylinder_points(np.linspace(-.7, .7, 29), 1.1, 35.)           # lower margin at rest
        weights = np.r_[np.ones(n), np.full(n, .5), np.zeros(n)]
        result = hinge_landing(positions, np.arange(n), landing, [0., 0., 0.], [1., 0., 0.], weights=weights, outside=.001)
        out = result['positions']
        np.testing.assert_allclose(out[:n], cylinder_points(x, 1.101, 35.), atol=1e-12)
        np.testing.assert_allclose(result['edge_turn'], np.radians(75.), atol=1e-12)
        np.testing.assert_allclose(result['turn'][n:2 * n], np.radians(37.5), atol=1e-12)
        np.testing.assert_allclose(out[n:2 * n], cylinder_points(x, 1.15 + .0005, -55. + 37.5), atol=1e-12)
        self.assertTrue(np.array_equal(out[2 * n:], positions[2 * n:]))         # weight 0 keeps its exact bytes
        np.testing.assert_allclose(out[:, 0], positions[:, 0], atol=1e-12)     # nothing slides along the axis
        m = result['public_metrics']
        self.assertAlmostEqual(m['landed_gap_to_landing_line_max'], .001, places=9)
        self.assertEqual(m['edge_points_beyond_landing_range'], 0)

    def test_corners_land_from_the_rest_blended_into_the_retained_pose(self):
        x = np.linspace(-.6, .6, 25); n = len(x)
        apart = 25. * np.clip((-.3 - x) / .3, 0, 1)             # the retained rims stand apart toward the low corner
        rest = np.r_[cylinder_points(x, 1.1, -40.), cylinder_points(x, 1.15, -55.), cylinder_points(x, 1.2, 60.)]
        kept = np.r_[cylinder_points(x, 1.1, -40. - apart), cylinder_points(x, 1.15, -55. - apart), rest[2 * n:]]
        landing = cylinder_points(np.linspace(-.7, .7, 29), 1.1, 35.)
        weights = np.r_[np.ones(n), np.full(n, .5), np.zeros(n)]
        land = lambda P, **kw: hinge_landing(P, np.arange(n), landing, [0., 0., 0.], [1., 0., 0.], weights=weights, **kw)
        plain, from_rest = land(kept)['positions'], land(rest)['positions']
        result = land(kept, rest=rest, corner_blend=(.2, 0.)); out = result['positions']
        band = slice(n, 2 * n)
        np.testing.assert_allclose(out[:n], plain[:n], atol=1e-12)                 # the edge lands on the line either way
        np.testing.assert_allclose(out[n], from_rest[n], atol=1e-12)               # at the corner: the rest's landing
        far = n + np.flatnonzero(x >= -.4 + 1e-9)
        np.testing.assert_allclose(out[far], plain[far], atol=1e-12)               # past the blend: the retained landing
        between = n + np.flatnonzero((x > -.6) & (x < -.4))
        lo, hi = np.minimum(plain[between], from_rest[between]), np.maximum(plain[between], from_rest[between])
        self.assertTrue(np.all((out[between] >= lo - 1e-12) & (out[between] <= hi + 1e-12)))
        self.assertGreater(np.linalg.norm(plain[n] - from_rest[n]), .1)            # the extra turn the blend removes
        self.assertTrue(np.array_equal(out[2 * n:], kept[2 * n:]))                  # weight 0 stays
        np.testing.assert_allclose(np.degrees(result['turn'][band][0]), (-55. + 37.5) - (-55. - 25.), atol=1e-9)
        self.assertEqual(result['public_metrics']['corner_blend']['widths'], [.2, 0.])
        np.testing.assert_allclose(land(kept, rest=rest, corner_blend=(0., .2))['positions'][band][:4], plain[band][:4],
                                   atol=1e-12)                                     # only the named end blends
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            land(kept, rest=rest, corner_blend=-.1)
        with self.assertRaisesRegex(ValueError, 'one finite position'):
            land(kept, rest=rest[:-1], corner_blend=.2)

    def test_the_band_turn_is_smoothed_beside_a_blunt_corner_and_the_edge_lands_exactly(self):
        # A blunt corner: from the pinned canthus (on the facing rim) the margin rises almost across the axis, so its
        # first points need turns from 0 to nearly the full closing turn within .006 of axis.
        xe = np.r_[-.6, -.599, -.597, -.594, np.linspace(-.55, .6, 24)]; ne = len(xe)
        ae = np.r_[30., 0., -20., -35., np.full(24, -40.)]
        xb = np.linspace(-.6, .6, 241)
        positions = np.r_[cylinder_points(xe, 1.1, ae), cylinder_points(xb, 1.15, -55.)]
        landing = cylinder_points(np.linspace(-.7, .7, 561), 1.1, 30.)
        weights = np.r_[0., np.ones(ne - 1), np.full(len(xb), .6)]                           # the corner is pinned
        land = lambda **kw: hinge_landing(positions, np.arange(ne), landing, [0., 0., 0.], [1., 0., 0.],
                                          weights=weights, **kw)
        plain = land(); smooth = land(band_smooth={'sigma': .01, 'within': .04, 'fade': .04, 'ends': 'low'})
        np.testing.assert_allclose(smooth['positions'][:ne], plain['positions'][:ne], atol=1e-12)   # the edge lands exactly
        band = slice(ne, None)
        step = lambda r: np.abs(np.diff(np.degrees(r['turn'][band])) / np.diff(xb))
        corner = xb[:-1] < -.55
        self.assertGreater(step(plain)[corner].max(), 2 * step(smooth)[corner].max())      # the step beside the corner
        far = xb > -.6 + .04 + .04 + 1e-9
        np.testing.assert_allclose(smooth['turn'][band][far], plain['turn'][band][far], atol=1e-12)   # untouched past fade
        info = smooth['public_metrics']['band_smooth']
        self.assertEqual((info['ends'], info['sigma']), ('low', .01))
        self.assertGreater(info['largest_turn_change_degrees'], 1.)
        high_only = land(band_smooth={'sigma': .01, 'within': .04, 'fade': .04, 'ends': 'high'})
        np.testing.assert_allclose(high_only['turn'][band][xb < 0], plain['turn'][band][xb < 0], atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'band_smooth needs'):
            land(band_smooth={'sigma': 0., 'within': .04, 'fade': .04})
        with self.assertRaisesRegex(ValueError, 'band_smooth needs'):
            land(band_smooth={'sigma': .01})

    def test_edge_beyond_the_landing_range_takes_its_end_values_and_is_counted(self):
        x = np.linspace(-1., 1., 11)
        landing = cylinder_points(np.linspace(-.5, .5, 11), 1., np.linspace(20., 40., 11))
        result = hinge_landing(cylinder_points(x, 1., -30.), np.arange(11), landing, [0, 0, 0], [1, 0, 0])
        self.assertEqual(result['public_metrics']['edge_points_beyond_landing_range'], 6)
        np.testing.assert_allclose(np.degrees(result['edge_turn'][[0, -1]]), [50., 70.], atol=1e-9)

    def test_landing_refusals(self):
        edge = cylinder_points(np.linspace(-.5, .5, 5), 1., -30.); landing = cylinder_points(np.linspace(-.5, .5, 5), 1., 30.)
        with self.assertRaisesRegex(ValueError, 'two distinct edge'):
            hinge_landing(edge, np.array([1, 1]), landing, [0, 0, 0], [1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'two finite landing'):
            hinge_landing(edge, np.arange(5), landing[:1], [0, 0, 0], [1, 0, 0])
        with self.assertRaisesRegex(ValueError, 'Weights must'):
            hinge_landing(edge, np.arange(5), landing, [0, 0, 0], [1, 0, 0], weights=np.full(5, 1.5))
        with self.assertRaisesRegex(ValueError, 'on the hinge axis'):
            hinge_landing(edge, np.arange(5), np.r_[landing, [[0., 0., 0.]]], [0, 0, 0], [1, 0, 0])

    def test_attachments_ride_their_host_and_the_native_formula_matches(self):
        x = np.linspace(-.6, .6, 13)
        host = cylinder_points(x, 1.1, -40.); host_end = rotate_x(host, np.radians(70.)) * [1., 1.02, 1.02]
        fins = host + np.c_[np.zeros(13), -.03 * np.ones(13), -.02 * np.ones(13)]   # lash fins in front of and above it
        result = hinge_carry(fins, host, host_end, [0., 0., 0.], [1., 0., 0.], fraction=[0., .5, 1.])
        pos = result['positions']
        self.assertEqual(pos.shape, (3, 13, 3))
        np.testing.assert_allclose(pos[0], fins, atol=1e-12)
        np.testing.assert_array_equal(result['host_index'], np.arange(13))
        np.testing.assert_allclose(result['turn'], np.radians(70.), atol=1e-12)
        k = np.array([1., 0., 0.])
        for row, f in zip(pos, (0., .5, 1.)):                                    # what a node group computes per point
            s = fins @ k; r = fins - s[:, None] * k; length = np.linalg.norm(r, axis=1)
            native = ((s + f * result['axial'])[:, None] * k
                      + rodrigues(r, k, f * result['turn']) * ((length + f * result['radial']) / length)[:, None])
            np.testing.assert_allclose(row, native, atol=1e-12)
        self.assertLess(result['public_metrics']['seat_distance_change_max'], .002)
        single = hinge_carry(fins, host, host_end, [0, 0, 0], [1, 0, 0], fraction=.25, host_index=np.arange(13))
        self.assertEqual(single['positions'].shape, (13, 3))

    def test_carry_refusals(self):
        host = cylinder_points(np.linspace(-.5, .5, 5), 1., -30.)
        with self.assertRaisesRegex(ValueError, 'One host index'):
            hinge_carry(host, host, host, [0, 0, 0], [1, 0, 0], host_index=np.arange(4))
        with self.assertRaisesRegex(ValueError, 'fraction'):
            hinge_carry(host, host, host, [0, 0, 0], [1, 0, 0], fraction=[[0., 1.]])
        with self.assertRaisesRegex(ValueError, 'attached point on the hinge axis'):
            hinge_carry(np.array([[.1, 0., 0.]]), host, host, [0, 0, 0], [1, 0, 0])

    def test_registered_as_array_preparation_operations(self):
        for name in ('hinge_change', 'hinge_landing', 'hinge_carry'):
            self.assertIn(name, ARRAY_OPERATIONS)

    def test_the_documented_hinged_lid_recipe(self):
        """build-the-mechanism-first.md: land, roll with one pace, keep clearance, carry the lashes, check the closing."""
        from .construction_diagnostics import closing_edges
        x = np.linspace(-.6, .6, 13); n = len(x); centre, axis = [0., 0., 0.], [1., 0., 0.]
        rest = np.r_[cylinder_points(x, 1.1, -40.), cylinder_points(x, 1.15, -55.),     # upper margin, band above it
                     cylinder_points(x, 1.1, 30.), cylinder_points(x, 1.15, 45.)]       # lower margin, skin below it
        margin, lower = np.arange(n), np.arange(2 * n, 3 * n)
        band = np.r_[np.ones(n), np.full(n, .6), np.zeros(2 * n)]
        eye = cylinder_points(np.repeat(np.linspace(-.7, .7, 15), 37), 1., np.tile(np.linspace(-90, 90, 37), 15))
        closed = hinge_landing(rest, margin, rest[lower], centre, axis, weights=band, outside=.001)['positions']
        phases = np.linspace(0., 1., 11)
        pace = np.repeat(phases[:, None], len(rest), axis=1)                               # one pace: one timing
        rolled = path_positions(rest, closed, pace, pivot=centre, axis=axis)['positions']
        need = end_clearance(rest, closed, eye, eye, centre, cap=.05)['clearance']
        kept = keep_clearance(rolled, eye, centre, need)
        self.assertEqual(float(np.abs(kept['push']).max()), 0.)                             # the roll stays clear
        chord = path_positions(rest, closed, pace)['positions']
        self.assertLess(np.linalg.norm(chord[5, margin], axis=1).min(), 1.)                 # a chord cuts the eye
        lashes = rest[margin] + [0., -.03, -.02]
        carried = hinge_carry(lashes, rest[margin], closed[margin], centre, axis, fraction=phases)
        self.assertLess(carried['public_metrics']['seat_distance_change_max'], .002)
        report = closing_edges(rest, kept['positions'], margin, lower, pivot=centre, axis=axis)['summary']
        self.assertLess(report['facing_travel_share_max'], 1e-12)                          # the lower lid stays
        self.assertLess(report['closure_spread_max'], 1e-9)
        self.assertLess(report['roll_deviation_share_max'], 1e-9)
        self.assertLess(report['seam_to_facing_rest_share_max'], .002)
        np.testing.assert_allclose(kept['positions'][:, 3 * n:], np.broadcast_to(rest[3 * n:], (11, n, 3)), atol=1e-12)


class HingeLiftTests(unittest.TestCase):
    """A lower margin that sags below its corners, raised on the same hinge as the upper lid."""

    def setUp(self):
        self.x = np.linspace(-.6, .6, 13); n = len(self.x)
        self.sag = 15. * (1 - (self.x / .6) ** 2)                             # degrees lower in the middle
        self.upper = cylinder_points(self.x, 1.1, -40.)
        self.lower = cylinder_points(self.x, 1.1, 30. + self.sag)
        self.below = cylinder_points(self.x, 1.15, 45. + self.sag)
        self.rest = np.r_[self.upper, self.lower, self.below]
        self.up_idx, self.lo_idx, self.below_idx = np.arange(n), np.arange(n, 2 * n), np.arange(2 * n, 3 * n)
        self.corners = np.array([n, 2 * n - 1])                                # the lower margin's two ends

    def test_the_middle_rises_to_the_kept_share_and_the_band_follows(self):
        weights = np.r_[np.zeros(len(self.x)), np.ones(len(self.x)), np.full(len(self.x), .5)]
        result = hinge_lift(self.rest, self.lo_idx, self.corners, [0., 0., 0.], [1., 0., 0.], keep_share=.4,
                            weights=weights)
        m = result['public_metrics']; out = result['positions']
        self.assertAlmostEqual(m['lifted_depth_share'], .4 * m['rest_depth_share'], places=9)
        np.testing.assert_array_equal(out[self.corners], self.rest[self.corners])  # the corners stay
        np.testing.assert_array_equal(out[self.up_idx], self.rest[self.up_idx])   # weight 0 keeps its bytes
        np.testing.assert_allclose(out[:, 0], self.rest[:, 0], atol=1e-12)        # nothing slides along the axis
        np.testing.assert_allclose(np.linalg.norm(out[:, 1:], axis=1), np.linalg.norm(self.rest[:, 1:], axis=1),
                                   atol=1e-12)                                    # it stays on its circle
        np.testing.assert_allclose(result['turn'][self.below_idx], .5 * result['edge_turn'], atol=1e-12)
        self.assertEqual(m['unreachable_edge_points'], 0)

    def test_the_upper_lid_lands_on_the_lifted_edge_and_both_roll_at_one_rate(self):
        from .construction_diagnostics import closing_edges
        lifted = hinge_lift(self.rest, self.lo_idx, self.corners, [0, 0, 0], [1, 0, 0], keep_share=.3,
                            weights=np.r_[np.zeros(len(self.x)), np.ones(len(self.x)), np.full(len(self.x), .5)])
        weights = np.r_[np.ones(len(self.x)), np.zeros(2 * len(self.x))]
        closed = hinge_landing(lifted['positions'], self.up_idx, lifted['positions'][self.lo_idx], [0, 0, 0], [1, 0, 0],
                               weights=weights, outside=.001)['positions']
        phases = np.linspace(0, 1, 9)
        poses = path_positions(self.rest, closed, np.repeat(phases[:, None], len(self.rest), axis=1),
                               pivot=[0, 0, 0], axis=[1, 0, 0])['positions']
        report = closing_edges(self.rest, poses[1:], self.up_idx, self.lo_idx, pivot=[0, 0, 0], axis=[1, 0, 0],
                               against='pose')
        self.assertLess(report['summary']['seam_to_facing_share_max'], .002)
        for row in report['poses']:
            parts = row['turn_share']['parts_median']; self.assertLess(max(parts) - min(parts), 1e-9)

    def test_refusals_and_registration(self):
        with self.assertRaisesRegex(ValueError, 'keep_share'):
            hinge_lift(self.rest, self.lo_idx, self.corners, [0, 0, 0], [1, 0, 0], keep_share=1.5)
        with self.assertRaisesRegex(ValueError, 'corner'):
            hinge_lift(self.rest, self.lo_idx, np.array([3, 3]), [0, 0, 0], [1, 0, 0], keep_share=.5)
        with self.assertRaisesRegex(ValueError, 'runs along'):
            hinge_lift(self.rest, self.lo_idx, self.corners, [0, 0, 0], [0, 0, 1], keep_share=.5)
        self.assertIn('hinge_lift', ARRAY_OPERATIONS)


if __name__ == '__main__':
    unittest.main()
