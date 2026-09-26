"""Standard construction checks on a synthetic eye: a grid lid over a round eye, hinged about the x axis."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .checks import LIMITS, applicable_checks, load_states, run_checks, standard_cells, standard_policy
from .motion_paths import hinge_carry, hinge_landing, path_positions
from .preservation import file_ref

X = np.linspace(-.6, .6, 13)
ROWS = [(-85., 1.25), (-70., 1.2), (-55., 1.15), (-40., 1.1), (30., 1.1), (45., 1.15), (60., 1.2)]   # brow .. cheek
BROW, UPPER, LOWER, CHEEK = 0, 3, 4, 6
PHASES = [0., .25, .5, .75, 1.]
CENTRE, AXIS = [0., 0., 0.], [1., 0., 0.]


def ring(angle, radius, x=X):
    a = np.radians(angle)
    return np.c_[x, np.full(len(x), -radius * np.cos(a)), np.full(len(x), -radius * np.sin(a))]


def row(r):
    return np.arange(r * len(X), (r + 1) * len(X))


def grid_triangles():
    tri = []
    for r in range(len(ROWS) - 1):
        if r == UPPER:                                   # the opening between the two margins has no faces
            continue
        a, b = row(r), row(r + 1)
        for c in range(len(X) - 1):
            tri += [[a[c], b[c], a[c + 1]], [a[c + 1], b[c], b[c + 1]]]
    return np.array(tri)


REST = np.concatenate([ring(a, r) for a, r in ROWS])
TRI = grid_triangles()
REGION = np.arange(len(X), (len(ROWS) - 1) * len(X))                      # every row except brow and cheek
WEIGHTS = np.zeros(len(REST)); WEIGHTS[row(1)] = .2; WEIGHTS[row(2)] = .6; WEIGHTS[row(UPPER)] = 1.
EYE = np.concatenate([ring(a, 1., np.linspace(-.7, .7, 15)) for a in np.linspace(-90, 90, 37)])
BODY = np.array([[0., .5, -1.5], [.3, .5, -1.5], [-.3, .5, -1.5]])
CLOSED = hinge_landing(REST, row(UPPER), REST[row(LOWER)], CENTRE, AXIS, weights=WEIGHTS, outside=.001)['positions']


def hinged(pace_of=None, chord=False):
    """Skin poses: one roll about the hinge with one pace (or a per-point pace), or straight chords."""
    pace = np.array([np.full(len(REST), g) if pace_of is None else np.clip(pace_of(g), 0, 1) for g in PHASES])
    kw = {} if chord else {'pivot': CENTRE, 'axis': AXIS}
    return path_positions(REST, CLOSED, pace, **kw)['positions']


def states(skin, body=None, extra=None):
    out = {}
    for g, P in zip(PHASES, skin):
        out[f'{g}::Skin::co'] = P; out[f'{g}::Eye::co'] = EYE; out[f'{g}::Body::co'] = BODY if body is None else body[g]
        for name, positions in (extra or {}).items():
            out[f'{g}::{name}::co'] = positions
    out['0.0::Skin::tri'] = TRI
    return out


def pleated(skin):
    skin = skin.copy(); v, below = row(2)[6], row(UPPER)[6]
    skin[2][v] = 2 * skin[2][below] - skin[2][v]                   # pushed through the row below at one phase
    return skin


class StandardCheckTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(__import__('shutil').rmtree, self.root, True)
        np.savez(self.root / 'rest.npz', co=REST)
        self.declaration = {
            'object': 'Skin', 'region': REGION.tolist(), 'region_label': 'eye area', 'rest': {'path': 'rest.npz'},
            'protected': ['Body'], 'clearance': [{'obstacle': 'Eye', 'centre': CENTRE, 'minimum': .01}],
            'symmetry': {'axis': 0},
            'closing': {'moving': row(UPPER).tolist(), 'facing': row(LOWER).tolist(), 'pivot': CENTRE, 'axis': AXIS},
            'carrier': {'tolerance': .02}}

    def run_on(self, skin, declaration=None, baseline=None, body=None):
        return run_checks(declaration or self.declaration, states(skin, body), baseline, base=self.root)

    def status(self, report):
        return {c['id']: c['status'] for c in report['checks']}

    def failing(self, report):
        return sorted(k for k, v in self.status(report).items() if v == 'fail')

    def test_the_hinged_lid_passes_every_check(self):
        report = self.run_on(hinged())
        self.assertEqual(report['status'], 'pass', self.status(report))
        self.assertEqual([c['id'] for c in report['checks']], applicable_checks(self.declaration))
        carrier = next(c for c in report['checks'] if c['id'] == 'carrier_shapes')
        self.assertEqual(carrier['observed'], 2.)                      # a roll over a round eye needs a mid shape
        self.assertGreater(carrier['detail']['one_shape_error'], .02)

    def test_a_zipper_fails_the_closing_spread(self):
        zipper = lambda g: g * (1 + (REST[:, 0] - X.min()) / np.ptp(X))
        self.assertIn('closing_spread', self.failing(self.run_on(hinged(zipper))))

    def test_a_rising_lower_lid_fails_facing_travel(self):
        skin = hinged().copy()
        for k, g in enumerate(PHASES):
            for r in (LOWER, LOWER + 1):
                a, radius = ROWS[r]; skin[k][row(r)] = ring(a - 12. * g, radius)
        self.assertIn('facing_travel_share', self.failing(self.run_on(skin)))

    def test_a_designed_lower_lid_rise_passes_the_closing_checks_once_its_travel_is_allowed(self):
        lift = 20. * (1 - (X / .6) ** 2); skin = hinged().copy()
        for k, g in enumerate(PHASES):
            a = -40. + g * (70. - lift); skin[k][row(UPPER)] = np.c_[X, -1.1 * np.cos(np.radians(a)), -1.1 * np.sin(np.radians(a))]
            b = 30. - g * lift; skin[k][row(LOWER)] = np.c_[X, -1.1 * np.cos(np.radians(b)), -1.1 * np.sin(np.radians(b))]
        declaration = dict(self.declaration, limits={'facing_travel_share': .5}); del declaration['carrier']
        status = self.status(run_checks(declaration, states(skin), base=self.root))
        self.assertEqual([status[k] for k in ('facing_travel_share', 'closing_spread', 'seam_share',
                                              'roll_deviation_share')], ['pass'] * 4)
        self.assertIn('facing_travel_share', self.failing(run_checks(self.declaration, states(skin), base=self.root)))

    def test_straight_chords_cut_the_eye_and_leave_the_roll(self):
        failing = self.failing(self.run_on(hinged(chord=True)))
        self.assertIn('clearance_shortfall', failing)
        self.assertIn('roll_deviation_share', failing)

    def test_an_object_protected_against_rest_must_stay_still_whatever_the_baseline_did(self):
        rising = {g: BODY + [0., 0., .013 * g] for g in PHASES}           # the old design moved it
        baseline = states(hinged(), body=rising); still = {g: BODY for g in PHASES}
        against_baseline = self.run_on(hinged(), baseline=baseline, body=still)
        self.assertIn('protected_unchanged', self.failing(against_baseline))  # keeping it still reads as a change
        at_rest = dict(self.declaration, protected=[{'object': 'Body', 'against': 'rest'}])
        report = self.run_on(hinged(), at_rest, baseline=baseline, body=still)
        row_ = next(c for c in report['checks'] if c['id'] == 'protected_unchanged')
        self.assertEqual((row_['status'], row_['detail']['against']), ('pass', {'Body': 'own rest'}))
        self.assertEqual(self.status(self.run_on(hinged(), at_rest, baseline=states(hinged(), body=rising),
                                                 body=rising))['protected_unchanged'], 'fail')   # it moves: fail
        with self.assertRaisesRegex(ValueError, 'Protected objects'):
            run_checks(dict(self.declaration, protected=[{'object': 'Body', 'against': 'sometimes'}]), states(hinged()))

    def test_motion_outside_the_region_a_changed_rest_and_a_moved_protected_object_fail(self):
        skin = hinged().copy(); skin[2][row(BROW)] += [0., 0., .01]
        self.assertEqual(self.failing(self.run_on(skin)), ['still_outside'])
        moved = np.load(self.root / 'rest.npz')['co'].copy(); moved[row(UPPER)[6]] += .002
        np.savez(self.root / 'rest.npz', co=moved)
        self.assertEqual(self.failing(self.run_on(hinged())), ['rest_identity'])
        np.savez(self.root / 'rest.npz', co=REST)
        body = {g: BODY + [0., 0., .1 * g] for g in PHASES}
        self.assertEqual(self.failing(self.run_on(hinged(), body=body)), ['protected_unchanged'])

    def test_a_pleat_fails_folds_and_a_vertex_moving_back_fails_reversals(self):
        self.assertIn('folds', self.failing(self.run_on(pleated(hinged()))))
        skin = hinged().copy(); skin[3][row(UPPER)[6]] = skin[1][row(UPPER)[6]]
        self.assertIn('reversing_vertices', self.failing(self.run_on(skin)))

    def test_with_a_baseline_reversals_follow_the_baselines_movers_and_list_what_is_new(self):
        def lid(scale):
            skin = hinged().copy(); d = skin - REST
            skin[:, row(1)] = REST[row(1)] + .8 * d[:, row(1)]                     # a band just under the mover cut
            skin[2][row(1)] = REST[row(1)]                                          # that goes back at mid-blink
            for r in (2, UPPER):
                skin[:, row(r)] = REST[row(r)] + scale * d[:, row(r)]
            return skin
        baseline, candidate = lid(1.), lid(.85)            # the lid now closes a little less far; the band is unchanged
        alone = next(c for c in self.run_on(candidate)['checks'] if c['id'] == 'reversing_vertices')
        self.assertEqual(alone['observed'], 13.)            # on its own cut the unchanged band is followed
        report = self.run_on(candidate, baseline=states(baseline))
        rev = next(c for c in report['checks'] if c['id'] == 'reversing_vertices')
        self.assertEqual((rev['status'], rev['observed'], rev['baseline_observed']), ('pass', 0., 0.))
        self.assertEqual(rev['detail']['mover_cut_from'], 'baseline')
        new = candidate.copy(); v = row(2)[6]; new[3][v] = new[1][v]              # one vertex newly goes back
        rev = next(c for c in self.run_on(new, baseline=states(baseline))['checks'] if c['id'] == 'reversing_vertices')
        self.assertEqual((rev['status'], rev['detail']['new_vs_baseline'], rev['detail']['new_examples']), ('fail', 1, [int(v)]))

    def test_with_a_baseline_folds_list_the_triangles_newly_folded_and_unfolded(self):
        def pleat(skin, column):
            skin = skin.copy(); v, below = row(2)[column], row(UPPER)[column]
            skin[2][v] = 2 * skin[2][below] - skin[2][v]
            return skin
        report = self.run_on(pleat(hinged(), 3), baseline=states(pleat(hinged(), 8)))
        folds = next(c for c in report['checks'] if c['id'] == 'folds')
        self.assertEqual(folds['status'], 'pass')                                 # the same count ...
        self.assertEqual(folds['observed'], folds['baseline_observed'])
        detail = folds['detail']                                                  # ... in a different place
        self.assertGreater(detail['new_per_phase']['0.5'], 0)
        self.assertEqual(detail['new_per_phase']['0.5'], detail['unfolded_per_phase']['0.5'])
        self.assertTrue(all(row(2)[3] in t for t in detail['new_folded_triangles']['0.5']))
        self.assertTrue(all(row(2)[8] in t for t in detail['unfolded_triangles']['0.5']))

    def test_defects_inherited_from_the_baseline_pass_and_new_ones_fail(self):
        inherited = self.run_on(pleated(hinged()), baseline=states(pleated(hinged())))
        folds = next(c for c in inherited['checks'] if c['id'] == 'folds')
        self.assertEqual((folds['status'], folds['rule']), ('pass', 'increase over the baseline'))
        self.assertGreater(folds['observed'], 0)
        self.assertIn('folds', self.failing(self.run_on(pleated(hinged()), baseline=states(hinged()))))

    def test_clearance_follows_only_points_that_start_outside_the_obstacle(self):
        big = np.concatenate([ring(a, 1.12, np.linspace(-.7, .7, 15)) for a in np.linspace(-90, 90, 37)])
        declaration = dict(self.declaration, clearance=[{'obstacle': 'Big', 'centre': CENTRE, 'minimum': .01}])
        report = run_checks(declaration, states(hinged(), extra={'Big': big}), base=self.root)
        clearance = next(c for c in report['checks'] if c['id'] == 'clearance_shortfall')
        row_ = clearance['detail']['obstacles'][0]
        self.assertEqual(clearance['status'], 'pass')
        self.assertGreater(row_['inside_at_rest'], 0)               # both margins sit inside the larger envelope
        self.assertGreater(row_['followed_points'], 0)

    def test_broken_symmetry_fails(self):
        skin = hinged().copy(); skin[2][row(2)[1]] += [0., .01, 0.]
        self.assertIn('symmetry', self.failing(self.run_on(skin)))

    def test_baseline_values_are_reported_and_do_not_gate(self):
        zipper = lambda g: g * (1 + (REST[:, 0] - X.min()) / np.ptp(X))
        report = self.run_on(hinged(), baseline=states(hinged(zipper)))
        spread = next(c for c in report['checks'] if c['id'] == 'closing_spread')
        self.assertEqual(report['status'], 'pass')
        self.assertGreater(spread['baseline_observed'], LIMITS['closing_spread'])

    def test_limits_override_and_refusals(self):
        loose = dict(self.declaration, limits={'facing_travel_share': .5})
        self.assertEqual(standard_cells(loose)[applicable_checks(loose).index('facing_travel_share')]['rule'],
                         {'maximum': .5})
        with self.assertRaisesRegex(ValueError, 'known check'):
            run_checks(dict(self.declaration, limits={'nonsense': 1.}), states(hinged()), base=self.root)
        with self.assertRaisesRegex(ValueError, 'moving object and the region'):
            run_checks({'object': 'Skin'}, states(hinged()))
        with self.assertRaisesRegex(ValueError, 'rest pose, 0'):
            run_checks(self.declaration, {k.replace('0.0::', '0.1::'): v for k, v in states(hinged()).items()
                                          if k.startswith('0.0::')}, base=self.root)
        minimal = {'object': 'Skin', 'region': REGION.tolist()}
        self.assertEqual(applicable_checks(minimal), ['still_outside', 'folds', 'reversing_vertices'])
        no_tri = {k: v for k, v in states(hinged()).items() if not k.endswith('::tri')}
        report = run_checks(minimal, no_tri)
        self.assertEqual((report['status'], self.status(report)['folds']), ('unknown', 'unknown'))

    def test_the_closed_line_depth_is_measured_against_the_declared_target(self):
        corners = [int(row(LOWER)[0]), int(row(LOWER)[-1])]
        flat = dict(self.declaration, closing=dict(self.declaration['closing'], closed_depth={'target': 0., 'corners': corners}))
        report = self.run_on(hinged(), flat)
        depth = next(c for c in report['checks'] if c['id'] == 'closed_depth_error')
        self.assertEqual(depth['status'], 'pass')
        self.assertLess(depth['detail']['closed_depth_share'], .001)          # the seam rests just outside the lower line
        deep = dict(self.declaration, closing=dict(self.declaration['closing'], closed_depth={'target': .2, 'corners': corners}))
        self.assertIn('closed_depth_error', self.failing(self.run_on(hinged(), deep)))
        with self.assertRaisesRegex(ValueError, 'closed_depth needs'):
            run_checks(dict(self.declaration, closing=dict(self.declaration['closing'], closed_depth={'target': .1})),
                       states(hinged()), base=self.root)

    def test_states_load_from_a_saved_file_in_phase_order(self):
        np.savez(self.root / 'poses.npz', **states(hinged()))
        loaded = load_states(self.root / 'poses.npz')
        self.assertEqual(list(loaded), ['0.0', '0.25', '0.5', '0.75', '1.0'])
        self.assertEqual(set(loaded['0.0']), {'Skin', 'Eye', 'Body'})


def tapered(profile):
    """The hinged lid with each point's closure scaled by a profile of its position across (1 in the middle)."""
    f = profile(REST[:, 0] / .6)
    return hinged(lambda g: g * f)


def turned(points, degrees):
    """Points turned about the x axis through the eye's centre (positive toward the chin for this eye)."""
    a = np.radians(degrees); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return points @ R.T


class EyeCheckTests(unittest.TestCase):
    """Corners, moving band, gaze clearance, lash carriage and combined closed poses (eye study checks 4-9)."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(__import__('shutil').rmtree, self.root, True)
        self.corners = [int(row(UPPER)[0]), int(row(UPPER)[-1])]
        self.closing = {'moving': row(UPPER).tolist(), 'facing': row(LOWER).tolist(), 'pivot': CENTRE, 'axis': AXIS}

    def status(self, report, *keys):
        found = {c['id']: c for c in report['checks']}
        return [found[k]['status'] for k in keys]

    def check(self, report, key):
        return next(c for c in report['checks'] if c['id'] == key)

    def declare(self, **extra):
        return {'object': 'Skin', 'region': REGION.tolist(), **extra}

    def test_corners_that_stay_and_a_margin_rising_smoothly_pass(self):
        declaration = self.declare(closing=dict(self.closing, corners=self.corners))
        smooth = run_checks(declaration, states(tapered(lambda u: np.cos(np.pi / 2 * u))))
        self.assertEqual(self.status(smooth, 'corner_travel_share', 'corner_ramp'), ['pass', 'pass'])
        self.assertGreater(self.check(smooth, 'corner_ramp')['detail']['ramp_from_start'], .3)
        pinched = run_checks(declaration, states(tapered(lambda u: np.where(np.abs(u) > .99, 0., 1.))))
        self.assertEqual(self.status(pinched, 'corner_travel_share', 'corner_ramp'), ['pass', 'fail'])   # full travel one step in
        sliding = run_checks(declaration, states(hinged()))                          # the corners close with the lid
        self.assertEqual(self.status(sliding, 'corner_travel_share')[0], 'fail')
        self.assertGreater(self.check(sliding, 'corner_travel_share')['observed'], 1.)

    def test_triangles_squeezed_beside_a_corner_are_counted_and_new_ones_named(self):
        declaration = self.declare(closing=dict(self.closing, corners=self.corners, corner_compression=True))
        good = hinged()
        squeezed = good.copy()                          # the column beside the low corner pressed to a fifth of its width
        for k, g in enumerate(PHASES):
            for r in (1, 2, UPPER):
                a, b = row(r)[0], row(r)[1]
                squeezed[k][b, 0] = good[k][a, 0] + .2 * g * (good[k][b, 0] - good[k][a, 0]) + (1 - g) * (good[k][b, 0] - good[k][a, 0])
        base = self.check(run_checks(declaration, states(good)), 'corner_compression')
        worse = run_checks(declaration, states(squeezed), states(good))
        row_ = self.check(worse, 'corner_compression')
        self.assertEqual((row_['status'], row_['rule']), ('fail', 'increase over the baseline'))
        self.assertGreater(row_['observed'], base['observed'])
        new = row_['detail']['new_squeezed_triangles']
        self.assertTrue(new and all(any(v in (row(r)[1] for r in (1, 2, UPPER)) for v in t) for t in new))
        better = self.check(run_checks(declaration, states(good), states(squeezed)), 'corner_compression')
        self.assertEqual(better['status'], 'pass')
        self.assertEqual(len(better['detail']['relieved_triangles']), len(new))
        with self.assertRaisesRegex(ValueError, 'corner_compression needs'):
            applicable_checks(self.declare(closing=dict(self.closing, corner_compression=True)))

    def test_the_closed_line_depth_can_use_the_closing_corners(self):
        closing = dict(self.closing, corners=[int(row(LOWER)[0]), int(row(LOWER)[-1])], closed_depth={'target': 0.})
        report = run_checks(self.declare(closing=closing), states(hinged()))
        self.assertEqual(self.status(report, 'closed_depth_error'), ['pass'])

    def test_a_band_that_stops_passes_and_one_leaking_into_the_brow_does_not(self):
        declaration = self.declare(closing=self.closing, band={}, limits={'band_reach': .4})
        stopping = hinged().copy(); stopping[:, row(1)] = REST[row(1)]               # row two above the margin stays
        report = run_checks(declaration, states(stopping))
        self.assertEqual(self.status(report, 'band_reach'), ['pass'])
        self.assertAlmostEqual(self.check(report, 'band_reach')['observed'], (1.2 * np.sin(np.radians(70)) -
                                                                              1.1 * np.sin(np.radians(40))) / 1.2, places=6)
        leaking = hinged().copy()
        for k, g in enumerate(PHASES):
            leaking[k][row(BROW)] = turned(REST[row(BROW)], 35. * g)                   # the brow moves with the lid
        report = run_checks(declaration, states(leaking))
        self.assertEqual(self.status(report, 'band_reach'), ['fail'])
        self.assertTrue(self.check(report, 'band_reach')['detail']['at_least'])
        within = run_checks(dict(declaration, limits={}), states(leaking))              # .45 w, but still moving there
        self.assertEqual(self.status(within, 'band_reach'), ['unknown'])

    def test_clearance_is_repeated_with_the_eye_turned_to_its_gaze_limits(self):
        bulge = ring(0., 1.12, np.linspace(-.3, .3, 13))
        bulge = np.concatenate([turned(bulge, a) for a in np.arange(40., 50.5, 1.)])      # sampled finer than 2 degrees
        cornea = np.concatenate([EYE, bulge])                        # a bulge under the still lower lid at rest gaze
        entry = {'obstacle': 'Cornea', 'centre': CENTRE, 'minimum': .01}
        ahead = run_checks(self.declare(clearance=[entry]), states(hinged(), extra={'Cornea': cornea}))
        self.assertEqual(self.status(ahead, 'clearance_shortfall'), ['pass'])
        gaze = dict(entry, gaze={'pivot': CENTRE, 'rotations': [[[1., 0., 0.], 30.], [[1., 0., 0.], -30.]]})
        limits = run_checks(self.declare(clearance=[gaze]), states(hinged(), extra={'Cornea': cornea}))
        clearance = self.check(limits, 'clearance_shortfall')
        self.assertEqual(clearance['status'], 'fail')                 # looking down turns the bulge into the lid's path
        self.assertEqual(clearance['detail']['obstacles'][0]['gaze'], '-30 deg about [1.0, 0.0, 0.0]')
        self.assertGreater(clearance['observed'], .02)                 # the margin passes .02 inside the bulge
        self.assertEqual(len(clearance['detail']['obstacles'][0]['gazes_measured']), 3)

    def lashes(self, fraction=None, turn=0., stretch=1.):
        margin = row(UPPER); roots = REST[margin] * 1.01; tips = roots + [0., -.3, .1]
        lash = np.concatenate([roots, tips])
        carried = hinge_carry(lash, REST[margin], CLOSED[margin], CENTRE, AXIS,
                              fraction=PHASES if fraction is None else fraction)['positions']
        n = len(margin); out = {}
        for k, g in enumerate(PHASES):
            r, t = carried[k][:n], carried[k][n:]
            v = turned(t - r, -turn * g) * (1 + (stretch - 1) * g)              # the lash's own turn and stretch
            out[g] = np.concatenate([r, r + v])
        declaration = self.declare(closing=self.closing, lash={'object': 'Lash', 'roots': list(range(n)),
                                                               'tips': list(range(n, 2 * n))})
        return declaration, {f'{g}::Lash::co': P for g, P in out.items()}

    def lash_report(self, **kw):
        declaration, arrays = self.lashes(**kw)
        return run_checks(declaration, {**states(hinged()), **arrays})

    def test_a_lash_carried_with_the_lid_passes(self):
        report = self.lash_report()
        self.assertEqual(self.status(report, 'lash_travel', 'lash_turn', 'lash_length', 'lash_timing'), ['pass'] * 4)
        turn = self.check(report, 'lash_turn')
        self.assertLess(turn['observed'], .1)                         # no turn of its own on the lid
        self.assertGreater(turn['detail']['absolute_max'], 60.)        # while the lid rolls it through the blink
        self.assertAlmostEqual(self.check(report, 'lash_travel')['detail']['ratio_median'], 1.01, places=4)

    def test_lash_pairs_from_rest_positions_group_by_carrier_and_root_on_the_margin(self):
        from .checks import lash_pairs
        margin = row(UPPER); n = len(margin)
        roots = REST[margin] + [0., -.0002, 0.]                                # on the margin
        strands = [roots + k * np.array([0., -.02, .01]) for k in (1, 2, 3)]     # strands, no faces
        stray = REST[margin[6]] + [0., -.004, .002] + np.outer([0, 1, 2], [0., -.02, .01])   # a clump 4 mm off
        lash = np.concatenate([roots, *strands, stray])
        carrier = np.r_[np.tile(margin, 4), np.full(3, row(2)[6])]
        pairs = lash_pairs(lash, REST, margin, carrier=carrier)
        self.assertEqual((pairs['groups'], pairs['groups_with_root'], pairs['groups_without_root']), (n + 1, n, 1))
        self.assertEqual(sorted(set(pairs['roots'])), list(range(n)))
        self.assertEqual(len(pairs['tips']), 3 * n)
        self.assertTrue(all(pairs['host'][k] == margin[pairs['roots'][k]] for k in range(len(pairs['roots']))))
        self.assertLess(pairs['root_distance_max'], .0005)
        default = lash_pairs(lash, REST, margin)                              # carried by the nearest margin vertex
        self.assertEqual(default['groups_without_root'], 0)
        carried = hinge_carry(lash, REST[margin], CLOSED[margin], CENTRE, AXIS, fraction=PHASES)['positions']
        declaration = self.declare(closing=self.closing, lash={'object': 'Lash', **{k: pairs[k] for k in ('roots', 'tips', 'host')}})
        report = run_checks(declaration, {**states(hinged()), **{f'{g}::Lash::co': carried[k] for k, g in enumerate(PHASES)}})
        self.assertEqual(self.status(report, 'lash_travel', 'lash_turn', 'lash_length', 'lash_timing'), ['pass'] * 4)
        with self.assertRaisesRegex(ValueError, 'one skin vertex per lash point'):
            lash_pairs(lash, REST, margin, carrier=carrier[:-1])

    def test_a_lash_left_behind_lagging_flipping_or_stretching_fails(self):
        behind = self.lash_report(fraction=.7 * np.asarray(PHASES))
        self.assertEqual(self.status(behind, 'lash_travel'), ['fail'])
        lagging = self.lash_report(fraction=np.asarray(PHASES) ** 2)
        self.assertEqual(self.status(lagging, 'lash_travel', 'lash_timing'), ['pass', 'fail'])
        flipping = self.lash_report(turn=50.)
        self.assertEqual(self.status(flipping, 'lash_turn', 'lash_travel'), ['fail', 'pass'])
        self.assertAlmostEqual(self.check(flipping, 'lash_turn')['observed'], 50., delta=.1)
        stretching = self.lash_report(stretch=1.4)
        self.assertEqual(self.status(stretching, 'lash_length', 'lash_turn'), ['fail', 'pass'])

    def test_combined_closed_poses_must_meet_at_the_seam(self):
        def combination(name, closed):
            skin = hinged().copy(); skin[-1] = closed
            np.savez(self.root / f'{name}.npz', **states(skin))
            return {'name': name, 'poses': f'{name}.npz'}
        meets = combination('closed smile', CLOSED)
        raised = CLOSED.copy()
        for r in (LOWER, LOWER + 1):
            a, radius = ROWS[r]; raised[row(r)] = ring(a - 10., radius)             # a lower-lid raise added to the blink
        crosses = combination('blink plus raise', raised)
        short = combination('blink plus surprise', hinged(lambda g: np.full(len(REST), .7 * g))[-1])
        good = run_checks(self.declare(closing=self.closing, combinations=[meets]), states(hinged()), base=self.root)
        self.assertEqual(self.status(good, 'combination_seam'), ['pass'])
        bad = run_checks(self.declare(closing=self.closing, combinations=[meets, crosses, short]), states(hinged()),
                         base=self.root)
        seam = self.check(bad, 'combination_seam')
        self.assertEqual(seam['status'], 'fail')
        shares = {entry['name']: entry['median_share'] for entry in seam['detail']['combinations']}
        self.assertLess(shares['blink plus raise'], -.1)             # past the lower lid
        self.assertGreater(shares['blink plus surprise'], .1)        # stops short of closing
        self.assertLess(abs(shares['closed smile']), .01)

    def test_declaration_refusals(self):
        with self.assertRaisesRegex(ValueError, 'need the closing edges'):
            applicable_checks(self.declare(band={}))
        with self.assertRaisesRegex(ValueError, 'matching root and tip'):
            applicable_checks(self.declare(closing=self.closing, lash={'object': 'Lash', 'roots': [0, 1], 'tips': [2]}))
        with self.assertRaisesRegex(ValueError, 'Gaze needs'):
            applicable_checks(self.declare(clearance=[{'obstacle': 'Eye', 'gaze': {'pivot': [0, 0]}}]))
        with self.assertRaisesRegex(ValueError, 'two vertices'):
            applicable_checks(self.declare(closing=dict(self.closing, corners=[1])))

class StandardPolicyTests(unittest.TestCase):
    """The declaration becomes a PreservationPolicy that measures a native trial's saved poses."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(__import__('shutil').rmtree, self.root, True)
        np.savez(self.root / 'rest.npz', co=REST)
        declaration = {'object': 'Skin', 'region': REGION.tolist(), 'rest': {'path': 'rest.npz'},
                       'clearance': [{'obstacle': 'Eye', 'centre': CENTRE, 'minimum': .01}],
                       'closing': {'moving': row(UPPER).tolist(), 'facing': row(LOWER).tolist(),
                                   'pivot': CENTRE, 'axis': AXIS}}
        (self.root / 'declaration.json').write_text(json.dumps(declaration))
        for name in ('construction.py', 'AUTHORITY.md', 'guide.npz'):
            (self.root / name).write_text(name)
        self.policy = standard_policy(self.root / 'declaration.json', self.root / 'requirements.json',
                                      construction=self.root / 'construction.py', authority=[self.root / 'AUTHORITY.md'],
                                      guide=self.root / 'guide.npz')

    def trial(self, skin, name):
        folder = self.root / name; folder.mkdir()
        np.savez(folder / 'evaluated.npz', **states(skin)); np.savez(folder / 'source-evaluated.npz', **states(hinged()))
        (folder / 'candidate.blend').write_bytes(name.encode())
        return {'status': 'completed', 'subject': file_ref(folder / 'candidate.blend')}

    def item(self, key, profile):
        from .operating_session import PROFILES
        return {'id': key, 'revision': '1', 'lane': 'repair', 'handler': 'run', 'description': 'Synthetic trial',
                'completion_condition': 'Saved poses', 'reads': {'source': 's1'}, 'writes': [], 'requires': {},
                'payload': {}, 'workbench': {'profile': profile, 'capability': 'hinged lid', 'method': 'one roll',
                                             'bindings': {k: ['source'] for k in PROFILES[profile]['inputs']}}}

    def assess(self, skin, name):
        task = self.policy.bind_task(self.item('trial', 'appearance_edit'), stage='apply', adapter='standard')
        prepared, consumption = self.policy.prepare(task, {})
        self.assertIn('declaration', prepared['payload']['required_inputs']['standard_checks'])
        result = self.trial(skin, name)
        result['preservation'] = self.policy.assess(task, result, consumption)
        return task, result

    def test_a_trial_is_measured_by_the_declared_checks_and_retention_carries_them(self):
        self.assertEqual([c['id'] for c in self.policy.document['outcomes'][0]['cells']],
                         ['rest_identity', 'still_outside', 'clearance_shortfall', 'folds', 'reversing_vertices',
                          'facing_travel_share', 'closing_spread', 'seam_share', 'roll_deviation_share'])
        task, result = self.assess(hinged(), 'good')
        self.assertEqual(result['preservation']['status'], 'pass')
        retain = self.item('retain', 'retention'); retain['payload']['candidate'] = result['subject']
        retain['requires'] = {'trial': ['completed']}
        retain = self.policy.bind_task(retain, stage='retain', adapter='standard', previous={'task': task, 'result': result})
        self.policy.preflight(retain, {'trial': {'task': task, 'result': result}})

    def test_a_zipper_trial_fails_its_cell(self):
        zipper = lambda g: g * (1 + (REST[:, 0] - X.min()) / np.ptp(X))
        _, result = self.assess(hinged(zipper), 'zipper')
        checks = {c['cell']: c['status'] for c in result['preservation']['checks']}
        self.assertEqual(result['preservation']['status'], 'fail')
        self.assertEqual(checks['closing_spread'], 'fail')

    def test_the_requirements_file_is_not_silently_replaced(self):
        changed = json.loads((self.root / 'declaration.json').read_text()); changed['limits'] = {'folds': 3}
        (self.root / 'declaration.json').write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'different requirements'):
            standard_policy(self.root / 'declaration.json', self.root / 'requirements.json',
                            construction=self.root / 'construction.py', authority=[self.root / 'AUTHORITY.md'],
                            guide=self.root / 'guide.npz')


if __name__ == '__main__':
    unittest.main()
