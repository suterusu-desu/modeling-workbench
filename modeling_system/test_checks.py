"""Standard construction checks on a synthetic eye: a grid lid over a round eye, hinged about the x axis."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .checks import LIMITS, applicable_checks, load_states, run_checks, standard_cells, standard_policy
from .motion_paths import hinge_landing, path_positions
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
