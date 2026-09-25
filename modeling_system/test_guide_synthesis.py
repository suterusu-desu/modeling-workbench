"""Pose guides from generated variants joined onto an accepted neutral (synthetic, no IO)."""
import unittest
import numpy as np
from .guide_synthesis import (front_depth, remove_thin_relief, stationary_offset, pose_change, fit_depth_field,
                              evaluate_depth_field, change_band, band_excess, height_field_mesh, keep_in_front,
                              front_retreat, fit_band_field)
from .preparation import ARRAY_OPERATIONS

WINDOW, CELL = (0., 1., 0., 1.), .02          # a 50 x 50 chart


def chart():
    a = (np.arange(50) + .5) * CELL
    return np.meshgrid(a, a)


def dome(A, B, centre=(.5, .5), radius=.3, height=.1):
    r2 = ((A - centre[0]) ** 2 + (B - centre[1]) ** 2) / radius ** 2
    return -height * np.clip(1 - r2, 0, None)       # nearer (smaller) in the middle


def square(depth, size=2.):
    positions = np.array([[-size, depth, -size], [size, depth, -size], [size, depth, size], [-size, depth, size]], float)
    return positions, np.array([[0, 1, 2], [0, 2, 3]])


class FrontDepthTests(unittest.TestCase):
    def test_keeps_the_nearest_surface_and_its_triangle(self):
        back, tb = square(.5)
        front = np.array([[.2, .2, .2], [.8, .2, .2], [.2, .2, .8]])          # a smaller triangle nearer the viewer
        positions = np.r_[back, front]; triangles = np.r_[tb, [[4, 5, 6]]]
        result = front_depth(positions, triangles, window=WINDOW, cell=CELL)
        A, B = chart(); inside = (A > .2) & (B > .2) & (A + B < 1.)
        np.testing.assert_allclose(result['depth'][inside], .2)
        np.testing.assert_allclose(result['depth'][~inside & (A + B > 1.02)], .5)
        self.assertTrue(np.all(result['triangle_index'][inside] == 2))
        self.assertEqual(result['public_metrics']['covered_share'], 1.)
        farthest = front_depth(positions, triangles, window=WINDOW, cell=CELL, front='max')
        np.testing.assert_allclose(farthest['depth'], .5)

    def test_interpolates_depth_across_a_tilted_triangle_and_leaves_empty_cells(self):
        positions = np.array([[0., 0., 0.], [1., .5, 0.], [0., 0., 1.]]); triangles = np.array([[0, 1, 2]])
        result = front_depth(positions, triangles, window=WINDOW, cell=CELL)
        A, B = chart(); inside = A + B < .98
        np.testing.assert_allclose(result['depth'][inside], .5 * A[inside], atol=1e-12)
        self.assertTrue(np.isnan(result['depth'][A + B > 1.02]).all())

    def test_refuses_bad_triangles_and_axes(self):
        positions, triangles = square(.5)
        with self.assertRaisesRegex(ValueError, 'index the supplied'):
            front_depth(positions, triangles + 4, window=WINDOW, cell=CELL)
        with self.assertRaisesRegex(ValueError, 'distinct axes'):
            front_depth(positions, triangles, window=WINDOW, cell=CELL, depth_axis=0)


class ThinReliefTests(unittest.TestCase):
    def fine(self):
        a = (np.arange(100) + .5) * .01
        A, B = np.meshgrid(a, a)
        return A, B, dome(A, B, radius=.8, height=.02)                   # curvature about .06, no crease in the chart

    def test_removes_thin_strands_in_front_and_keeps_a_broad_form(self):
        A, B, clean = self.fine()
        spiky = clean.copy(); spiky[:, 30:32] -= .01; spiky[60:62, :] -= .01      # strands 2 cells wide, nearer
        result = remove_thin_relief(spiky, size=5)
        self.assertLess(np.abs(result['depth'] - clean).max(), 5e-4)
        self.assertGreater(result['public_metrics']['max_removed'], .009)

    def test_flattens_a_strongly_curved_form_by_its_curvature_and_keeps_slopes(self):
        A, B = chart(); curved = dome(A, B)                                    # curvature 2h/R^2 = 2.2
        raised = remove_thin_relief(curved, size=5)['depth'] - curved
        half_diagonal = np.sqrt(2) * 2 * CELL
        self.assertTrue(np.all(raised >= -1e-12))
        self.assertLess(raised.max(), 2.3 * half_diagonal ** 2)                    # curvature x half-diagonal^2
        tilted = .3 * A - .2 * B
        np.testing.assert_allclose(remove_thin_relief(tilted, size=5)['depth'][2:-2, 2:-2], tilted[2:-2, 2:-2], atol=1e-12)
        # a strand across a crease (slope .3 on one side, 0 on the other) is filled within slope change x half-width
        crease = np.minimum(.3 * (A - .5), 0.); strand = crease.copy(); strand[20:22, :] -= .05
        error = np.abs(remove_thin_relief(strand, size=5)['depth'] - crease)[2:-2, 2:-2]
        self.assertLess(error.max(), .3 * 2 * CELL + 1e-12)

    def test_mirrors_for_the_other_depth_convention_and_keeps_empty_cells(self):
        A, B, clean = self.fine(); clean = -clean
        spiky = clean.copy(); spiky[:, 40:42] += .01; spiky[0, 0] = np.nan
        result = remove_thin_relief(spiky, size=5, front='max')
        self.assertTrue(np.isnan(result['depth'][0, 0]))
        self.assertLess(np.nanmax(np.abs(result['depth'] - clean)), 5e-4)
        with self.assertRaisesRegex(ValueError, 'odd integer'):
            remove_thin_relief(spiky, size=4)


class StationaryOffsetTests(unittest.TestCase):
    def test_recovers_a_smooth_offset_on_the_still_support_only(self):
        A, B = chart(); target = dome(A, B)
        smooth = .01 + .02 * A - .01 * B ** 2
        moving = -.03 * np.exp(-((A - .5) ** 2 + (B - .5) ** 2) / .01)                  # a part that moves: not support
        source = target - smooth + moving
        support = ((A - .5) ** 2 + (B - .5) ** 2) > .12
        result = stationary_offset(target, source, support, window=WINDOW, cell=CELL, stride=2, smoothing=0.)
        np.testing.assert_allclose(result['offset'][support], smooth[support], atol=1e-3)
        self.assertLess(result['public_metrics']['support_mad_after'], .1 * result['public_metrics']['support_mad_before'])
        with self.assertRaisesRegex(ValueError, 'Fewer than 10'):
            stationary_offset(target, source, np.zeros_like(support), window=WINDOW, cell=CELL)

    def test_trimming_drops_support_that_moved(self):
        A, B = chart(); target = dome(A, B)
        smooth = .01 + .02 * A - .01 * B ** 2
        moved = .015 * np.exp(-((A - .2) ** 2 + (B - .8) ** 2) / .004)                   # declared still, but it moved
        source = target - smooth + moved
        support = ((A - .5) ** 2 + (B - .5) ** 2) > .12
        # a smoothed fit (an interpolating one absorbs the moved part and leaves nothing to trim)
        plain = stationary_offset(target, source, support, window=WINDOW, cell=CELL, stride=2, smoothing=1e-3)
        trimmed = stationary_offset(target, source, support, window=WINDOW, cell=CELL, stride=2, smoothing=1e-3, trim=3.,
                                    trim_floor=2e-4)
        clean = support & (moved < 1e-4)
        err = lambda r: np.abs(r['offset'] - smooth)[clean].max()
        self.assertLess(err(trimmed), 1e-3)
        self.assertLess(err(trimmed), .5 * err(plain))
        self.assertFalse(trimmed['support_used'][support & (moved > .005)].any())
        self.assertGreater(trimmed['public_metrics']['support_kept_share'], .8)
        self.assertNotIn('support_kept_share', plain['public_metrics'])
        self.assertTrue(np.array_equal(plain['support_used'], support & np.isfinite(target) & np.isfinite(source)))
        with self.assertRaisesRegex(ValueError, 'trim'):
            stationary_offset(target, source, support, window=WINDOW, cell=CELL, trim=-1.)
        with self.assertRaisesRegex(ValueError, 'trim'):
            stationary_offset(target, source, support, window=WINDOW, cell=CELL, trim=3., trim_rounds=0)


class PoseChangeTests(unittest.TestCase):
    def test_recovers_the_common_change_and_reports_the_disagreement(self):
        A, B = chart(); neutral = dome(A, B); rng = np.random.default_rng(3)
        change = -.02 * np.exp(-((A - .5) ** 2 + (B - .6) ** 2) / .02)                  # a lid coming forward
        support = ((A - .5) ** 2 + (B - .55) ** 2) > .16
        style = .004 * np.sin(6 * A) * np.cos(5 * B)                                     # the generator's own detail
        noise = lambda: rng.normal(0, .0005, A.shape)
        rest = np.stack([neutral + style + k * .003 + .002 * A * k + noise() for k in range(4)])
        pose = np.stack([neutral + style + change - k * .002 + .001 * B * k + noise() for k in range(4)])
        result = pose_change(rest, pose, support, window=WINDOW, cell=CELL, reference=neutral, stride=2, smoothing=1e-8)
        np.testing.assert_allclose(result['change'], change, atol=2.5e-3)
        self.assertLess(np.median(np.abs(result['change'] - change)), 1e-3)
        self.assertLess(result['public_metrics']['spread_median'], 2e-3)
        # the rest consensus is aligned to the reference up to the generator's own style
        self.assertLess(np.median(np.abs(result['rest_consensus'] - neutral - style)), 1.5e-3)

    def test_holes_are_screened_after_alignment(self):
        A, B = chart(); neutral = dome(A, B); support = ((A - .5) ** 2 + (B - .5) ** 2) > .16
        hole = (np.abs(A - .5) < .06) & (np.abs(B - .5) < .04)
        rest = np.stack([neutral + .03 + .01 * k for k in range(3)])               # far behind before alignment
        cut = rest.copy(); cut[1][hole] += .05                                     # one variant is cut through
        result = pose_change(cut, rest, support, window=WINDOW, cell=CELL, reference=neutral, stride=2, smoothing=1e-8,
                             behind=.01)
        np.testing.assert_allclose(result['rest_consensus'][hole], neutral[hole], atol=1e-4)
        self.assertTrue(np.all(result['rest_count'][hole] == 2))
        self.assertTrue(np.all(result['rest_count'][~hole] == 3))
        with self.assertRaisesRegex(ValueError, 'hole screen'):
            pose_change(rest, rest, support, window=WINDOW, cell=CELL, behind=0.)

    def test_a_misplaced_feature_shows_as_spread(self):
        A, B = chart(); neutral = dome(A, B); support = ((A - .5) ** 2 + (B - .5) ** 2) > .16
        rest = np.stack([neutral] * 3)
        pose = np.stack([neutral - .02 * np.exp(-((A - .5 - s) ** 2 + (B - .5) ** 2) / .005) for s in (-.06, 0., .06)])
        result = pose_change(rest, pose, support, window=WINDOW, cell=CELL, stride=2, smoothing=1e-8)
        near = (np.abs(A - .5) < .12) & (np.abs(B - .5) < .06)
        self.assertGreater(np.nanmedian(result['spread'][near]), 3e-3)


class DepthFieldTests(unittest.TestCase):
    def test_fits_a_smooth_field_ignores_outliers_and_honours_pins(self):
        rng = np.random.default_rng(1); points = rng.uniform(0, 1, (1500, 2))
        truth = lambda p: .01 * np.sin(3 * p[:, 0]) * np.cos(2 * p[:, 1])
        values = truth(points); spikes = rng.choice(len(points), 60, replace=False); values[spikes] -= .02
        pins = np.array([[.3, .7]])
        result = fit_depth_field(points, values, knots=.1, bending=1e-4, robust=.002, pins=pins, pin_values=[.004],
                                 box=(0, 1, 0, 1), evaluate_at=np.array([[.5, .5], [.3, .7]]))
        clean = np.setdiff1d(np.arange(len(points)), spikes)
        self.assertLess(np.median(np.abs(result['fitted'][clean] - truth(points[clean]))), 6e-4)
        self.assertLess(result['public_metrics']['pin_max_abs_error'], 5e-4)
        self.assertGreaterEqual(result['public_metrics']['downweighted_samples'], 50)
        again = evaluate_depth_field(result['coefficients'], result['box'], result['knots'], np.array([[.5, .5], [.3, .7]]))
        np.testing.assert_allclose(again['values'], result['evaluated'])

    def test_zero_anchors_hold_the_field_where_nothing_should_move(self):
        rng = np.random.default_rng(2); points = rng.uniform(.3, .7, (400, 2)); values = np.full(len(points), .01)
        ring = np.array([[np.cos(t) * .45 + .5, np.sin(t) * .45 + .5] for t in np.linspace(0, 2 * np.pi, 80, endpoint=False)])
        result = fit_depth_field(points, values, knots=.08, bending=1e-3, anchors=ring, anchor_weight=5., box=(0, 1, 0, 1),
                                 evaluate_at=ring)
        self.assertLess(np.abs(result['evaluated']).max(), 2e-3)
        self.assertLess(np.median(np.abs(result['fitted'] - .01)), 1.5e-3)

    def test_refuses_bad_input(self):
        with self.assertRaisesRegex(ValueError, 'inside the box'):
            fit_depth_field(np.array([[2., 2.]]), np.array([0.]), knots=.1, bending=0., box=(0, 1, 0, 1))
        with self.assertRaisesRegex(ValueError, 'Positive knot'):
            fit_depth_field(np.array([[.5, .5]]), np.array([0.]), knots=0., bending=0.)


class BandTests(unittest.TestCase):
    def test_band_adds_slope_times_placement_uncertainty(self):
        A, B = chart(); spread = np.full(A.shape, .001); change = .05 * A                 # slope .05 per chart unit
        band = change_band(spread, change, cell=CELL, position_uncertainty=.02, accuracy=.0005)['band']
        np.testing.assert_allclose(band, np.sqrt(.001 ** 2 + (.05 * .02) ** 2 + .0005 ** 2))
        flat = change_band(spread, accuracy=.0005)['band']
        np.testing.assert_allclose(flat, np.hypot(.001, .0005))
        with self.assertRaisesRegex(ValueError, 'cell size'):
            change_band(spread, position_uncertainty=.01)

    def test_excess_is_zero_inside_and_signed_outside(self):
        result = band_excess(np.array([0., 1., -1., .5]), np.array([-.5, -.5, -.5, -.5]), np.array([.5, .5, .5, .5]))
        np.testing.assert_allclose(result['excess'], [0., .5, -.5, 0.])
        self.assertEqual(result['public_metrics']['inside_share'], .5)


class BandFieldTests(unittest.TestCase):
    def setUp(self):
        g = np.linspace(.02, .98, 25); X, Y = np.meshgrid(g, g); self.points = np.c_[X.ravel(), Y.ravel()]
        x = self.points[:, 0]
        # left of x = .5 every point must come forward by at least .01; right of it anything in [-.02, .01] will do
        self.lower = np.where(x < .5, -.03, -.02); self.upper = np.where(x < .5, -.01, .01)
        self.anchors = np.c_[np.ones(25) * .999, g]

    def slope_at_step(self, fitted):
        order = np.argsort(self.points[:, 0]); x = self.points[:, 0]
        row = np.abs(self.points[:, 1] - .5) < .03
        xs, fs = x[row], fitted[row]; k = np.argsort(xs); xs, fs = xs[k], fs[k]
        near = (xs[1:] > .4) & (xs[:-1] < .6)
        return float(np.abs(np.diff(fs) / np.diff(xs))[near].max())

    def test_points_inside_move_within_their_band_so_the_field_stays_smooth(self):
        kw = dict(knots=.05, bending=1e-3, box=(0, 1, 0, 1), anchors=self.anchors, anchor_weight=5.)
        edge = fit_depth_field(self.points, np.clip(0., self.lower, self.upper), **kw)
        band = fit_band_field(self.points, self.lower, self.upper, **kw)
        self.assertGreater(self.slope_at_step(edge['fitted']), 3 * self.slope_at_step(band['fitted']))
        m = band['public_metrics']
        self.assertTrue(m['converged']); self.assertLess(m['max_violation'], 1e-3); self.assertGreater(m['inside_after'], .9)
        held = fit_band_field(self.points, self.lower, self.upper, hold=1., **kw)      # a strong hold keeps inside points near 0
        self.assertGreater(self.slope_at_step(held['fitted']), self.slope_at_step(band['fitted']))
        again = evaluate_depth_field(band['coefficients'], band['box'], band['knots'], self.points)['values']
        np.testing.assert_allclose(again, band['fitted'], atol=1e-12)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'intervals'):
            fit_band_field(self.points, self.upper, self.lower, knots=.1, bending=0.)
        with self.assertRaisesRegex(ValueError, 'iteration'):
            fit_band_field(self.points, self.lower, self.upper, knots=.1, bending=0., iterations=0)
        with self.assertRaisesRegex(ValueError, 'hold'):
            fit_band_field(self.points, self.lower, self.upper, knots=.1, bending=0., hold=-1.)


class HeightFieldTests(unittest.TestCase):
    def test_mesh_skips_empty_cells_and_cliffs(self):
        depth = np.zeros((3, 3)); window = (0., .3, 0., .3)
        full = height_field_mesh(depth, window=window, cell=.1)
        self.assertEqual((len(full['positions']), len(full['triangles'])), (9, 8))
        np.testing.assert_allclose(sorted(set(full['positions'][:, 0].round(6))), [.05, .15, .25])
        holed = depth.copy(); holed[1, 1] = np.nan
        self.assertEqual(len(height_field_mesh(holed, window=window, cell=.1)['triangles']), 0)
        cliff = depth.copy(); cliff[:, 2] = .5
        self.assertEqual(len(height_field_mesh(cliff, window=window, cell=.1, max_step=.1)['triangles']), 4)


class KeepInFrontTests(unittest.TestCase):
    """A guide surface kept in front of an obstacle (an eyeball) by a margin."""

    def test_exact_minimum_moves_only_what_lies_behind(self):
        A, B = chart()
        lid = .05 + .2 * (B - .5)                        # a sloping sheet: behind the ball below, in front above
        ball = dome(A, B, radius=.4, height=.1)
        out = keep_in_front(lid, ball, .01)
        np.testing.assert_allclose(out['depth'], np.minimum(lid, ball - .01))
        self.assertTrue((out['push'] >= 0).all())
        self.assertTrue((out['depth'] <= ball - .01 + 1e-12).all())
        self.assertEqual(out['public_metrics']['pushed'], int((lid > ball - .01).sum()))

    def test_smooth_join_stays_near_the_minimum_and_nearer_than_both(self):
        A, B = chart()
        lid = .05 + .2 * (B - .5); ball = dome(A, B, radius=.4, height=.1)
        k = .004; out = keep_in_front(lid, ball, .01, softness=k)['depth']
        exact = np.minimum(lid, ball - .01)
        self.assertTrue((out <= exact + 1e-12).all())
        self.assertTrue((out >= exact - k * np.log(2) - 1e-12).all())
        far = np.abs(lid - (ball - .01)) > 10 * k       # away from the crossing it is the minimum
        np.testing.assert_allclose(out[far], exact[far], atol=1e-6)
        d2 = np.abs(np.diff(out, 2, axis=0)).max()       # no kink where the surfaces cross
        self.assertLess(d2, np.abs(np.diff(exact, 2, axis=0)).max())

    def test_uncovered_cells_and_the_other_front(self):
        A, B = chart()
        lid = np.full(A.shape, .05); ball = dome(A, B, radius=.3, height=.1); ball[:, :10] = np.nan
        out = keep_in_front(lid, ball, .01)['depth']
        np.testing.assert_array_equal(out[:, :10], lid[:, :10])
        flip = keep_in_front(-lid, -ball, .01, front='max')['depth']      # larger depth nearer: the mirror image
        np.testing.assert_allclose(flip, -out)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'margin and softness'):
            keep_in_front(np.zeros((3, 3)), np.zeros((3, 3)), -.01)
        with self.assertRaisesRegex(ValueError, 'obstacle'):
            keep_in_front(np.zeros((3, 3)), np.zeros((4, 3)), .01)
        with self.assertRaisesRegex(ValueError, 'front'):
            keep_in_front(np.zeros((3, 3)), np.zeros((3, 3)), .01, front='near')


class FrontRetreatTests(unittest.TestCase):
    """Skin that moves away from the viewer between two poses (a closing eye 'sucking in')."""

    def test_counts_a_sunken_patch(self):
        A, B = chart()
        rest = np.zeros(A.shape); pose = rest.copy()
        sunk = (A - .3) ** 2 + (B - .3) ** 2 < .1 ** 2
        pose[sunk] = .004; pose[~sunk & (A > .7)] = -.002                   # a dent, and skin that came forward
        m = front_retreat(rest, pose, threshold=.001)['public_metrics']
        self.assertAlmostEqual(m['retreating_share'], sunk.mean())
        self.assertAlmostEqual(m['max'], .004)
        sup = front_retreat(rest, pose, support=A > .7)['public_metrics']
        self.assertEqual(sup['retreating_share'], 0.); self.assertAlmostEqual(sup['max'], -.002)
        other = front_retreat(-rest, -pose, front='max', threshold=.001)['public_metrics']
        self.assertAlmostEqual(other['retreating_share'], m['retreating_share'])

    def test_empty_cells_and_refusals(self):
        rest = np.full((4, 4), np.nan); m = front_retreat(rest, rest)['public_metrics']
        self.assertEqual(m['cells'], 0); self.assertIsNone(m['median'])
        with self.assertRaisesRegex(ValueError, 'support'):
            front_retreat(np.zeros((3, 3)), np.zeros((3, 3)), support=np.ones((3, 3)))
        with self.assertRaisesRegex(ValueError, 'threshold'):
            front_retreat(np.zeros((3, 3)), np.zeros((3, 3)), threshold=-1.)


class RegistrationTests(unittest.TestCase):
    def test_operations_are_available_for_array_preparation(self):
        for name in ('front_depth', 'remove_thin_relief', 'stationary_offset', 'pose_change', 'fit_depth_field',
                     'evaluate_depth_field', 'change_band', 'band_excess', 'height_field_mesh', 'keep_in_front',
                     'front_retreat', 'fit_band_field'):
            self.assertIn(name, ARRAY_OPERATIONS)


if __name__ == '__main__':
    unittest.main()
