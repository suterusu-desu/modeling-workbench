"""Face construction audit on a synthetic face: a grid with a mouth slit (mouth width 1), rigid teeth and tongue behind
it, a jaw hinged 2.4 mouth widths behind and .5 above the lips, eyes above. Each check passes the good construction
and fails the mistake it exists for."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from .face_checks import LIMITS, run_face_audit
from .study import jaw_weights, rigid_motion, seam_rings, viseme_mix

XS, ZS = np.round(np.linspace(-2, 2, 41), 9), np.round(np.linspace(-2, 3, 51), 9)
SEAM_ROW = int(np.flatnonzero(ZS == 0)[0])


def face_mesh():
    ids, P = {}, []
    for j, z in enumerate(ZS):
        for i, x in enumerate(XS):
            ids[i, j, 0] = len(P); P.append((x, 0., z))
            if j == SEAM_ROW and abs(x) <= .5 + 1e-9:
                ids[i, j, 1] = len(P); P.append((x, 0., z))           # the lower lip: a copy, not welded
    faces = []
    for j in range(len(ZS) - 1):
        for i in range(len(XS) - 1):
            corner = lambda a, b: ids.get((a, b, 1), ids[a, b, 0]) if b == SEAM_ROW and j == SEAM_ROW - 1 and all(
                abs(XS[c]) <= .5 + 1e-9 for c in (i, i + 1)) else ids[a, b, 0]
            faces.append([corner(i, j), corner(i + 1, j), corner(i + 1, j + 1), corner(i, j + 1)])
    upper = np.array([ids[i, SEAM_ROW, 0] for i in range(len(XS)) if abs(XS[i]) <= .5 + 1e-9])
    lower = np.array([ids[i, SEAM_ROW, 1] for i in range(len(XS)) if abs(XS[i]) <= .5 + 1e-9])
    return np.array(P), faces, upper, lower


P, FACES, UPPER, LOWER = face_mesh()
BELOW = (P[:, 2] < 0); BELOW[LOWER] = True                     # material that belongs to the jaw
R_XZ = np.hypot(P[:, 0], P[:, 2])
LID = np.flatnonzero((np.abs(P[:, 2] - 2.) < 1e-9) & (np.abs(P[:, 0]) >= .5) & (np.abs(P[:, 0]) <= 1.2))
RING = np.flatnonzero((R_XZ >= 1.35) & (R_XZ <= 1.45))           # a stitched attachment ring round the mouth
MOUTH_AREA = np.flatnonzero(R_XZ < 1.3); LOWER_FACE = np.flatnonzero(P[:, 2] < .7)
CHIN = int(np.argmin(np.linalg.norm(P - [0., 0., -.5], axis=1)))
TEETH_X = np.linspace(-.4, .4, 9)
UPPER_TEETH = np.array([(x, y, z) for x in TEETH_X for y in (.12, .18) for z in (.03, .15)])
LOWER_TEETH = np.array([(x, y, z) for x in TEETH_X for y in (.12, .18) for z in (-.03, -.15)])
TONGUE = np.array([(x, y, -.1) for x in np.linspace(-.3, .3, 7) for y in (.25, .35, .45)])
HINGE, ANGLE = np.array([0., 2.4, .5]), 8.


def turn(X, degrees=ANGLE, hinge=HINGE):
    a = np.radians(degrees); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    return hinge + (np.asarray(X) - hinge) @ R.T


def falloff(radius, reach, centre=(0., 0.)):
    r = np.hypot(P[:, 0] - centre[0], P[:, 2] - centre[1])
    return np.clip(1 - r / reach, 0, 1) ** 2


def jaw_open(weight=None, hinge=HINGE, lift=.135):
    w = np.clip(1 - np.linalg.norm(P - [0., 0., -.5], axis=1) / 2., 0, 1) * BELOW if weight is None else weight
    d = w[:, None] * (turn(P, hinge=hinge) - P)
    up = ~BELOW & (P[:, 2] >= 0)
    d[up, 2] += lift * np.clip(1 - R_XZ[up] / .6, 0, 1)                 # the upper lip lifts on its own
    return d


def lips_move(dz_upper=0., dz_lower=0., reach=.5):
    d = np.zeros_like(P); f = np.clip(1 - np.abs(P[:, 0]) / reach, 0, 1) * (np.abs(P[:, 2]) < .35)
    d[~BELOW, 2] += dz_upper * f[~BELOW]; d[BELOW, 2] += dz_lower * f[BELOW]
    return d


SKIN = {
    'A': jaw_open(),
    'I': np.c_[.1 * P[:, 0] * falloff(1, 1.2), np.zeros(len(P)), np.zeros(len(P))],
    'O': np.c_[-.08 * P[:, 0] * falloff(1, 1.2), -.05 * falloff(1, 1.2), np.zeros(len(P))],
    'smile': np.c_[np.zeros(len(P)), np.zeros(len(P)), .06 * falloff(1, 1.2) + .06 * (falloff(1, .5, (.5, 0)) +
                                                                                    falloff(1, .5, (-.5, 0)))],
    'blink': np.c_[np.zeros(len(P)), np.zeros(len(P)), -.1 * (falloff(1, .35, (.85, 2)) + falloff(1, .35, (-.85, 2)))],
    'lip_raise': lips_move(dz_lower=.05),
}
smooth = lambda u: np.clip(u, 0, 1) ** 2 * (3 - 2 * np.clip(u, 0, 1))
SKIN.update({
    'vrc.v_aa': .85 * SKIN['A'], 'vrc.v_ih': .5 * SKIN['A'] + .5 * SKIN['I'], 'vrc.v_oh': .2 * SKIN['A'] + .7 * SKIN['O'],
    'vrc.v_pp': lips_move(-.01, .01),
    'smile_L': SKIN['smile'] * smooth((P[:, 0] + .1) / .2)[:, None],
    'smile_R': SKIN['smile'] * (1 - smooth((P[:, 0] + .1) / .2))[:, None],
    'blink_L': SKIN['blink'] * (P[:, 0] >= 0)[:, None], 'blink_R': SKIN['blink'] * (P[:, 0] < 0)[:, None],
})
ff = np.zeros_like(P); ff[LOWER[np.abs(P[LOWER, 0]) <= .4]] = [0., .12, .04]; SKIN['vrc.v_ff'] = ff
LOWER_KEYS = {'A': turn(LOWER_TEETH) - LOWER_TEETH}
LOWER_KEYS.update({'vrc.v_aa': .85 * LOWER_KEYS['A'], 'vrc.v_ih': .5 * LOWER_KEYS['A'], 'vrc.v_oh': .2 * LOWER_KEYS['A']})
TONGUE_KEYS = {k: v * (turn(TONGUE) - TONGUE) for k, v in (('A', 1.), ('vrc.v_aa', .85), ('vrc.v_ih', .5), ('vrc.v_oh', .2))}


def objects(skin=None, lower=None, tongue=None, info=None):
    keys = {**SKIN, **(skin or {})}
    names = ['Basis'] + list(keys)
    key_info = {n: {'index': i, 'name': n, 'relative_key': 'Basis'} for i, n in enumerate(names)}
    for name, relative in (info or {}).items():
        key_info[name]['relative_key'] = relative
    return {'Face': {'rest': P, 'keys': keys, 'edges': [], 'polygons': FACES, 'key_info': key_info},
            'Upper teeth': {'rest': UPPER_TEETH, 'keys': {}, 'edges': [], 'polygons': []},
            'Lower teeth': {'rest': LOWER_TEETH, 'keys': {**LOWER_KEYS, **(lower or {})}, 'edges': [], 'polygons': []},
            'Tongue': {'rest': TONGUE, 'keys': {**TONGUE_KEYS, **(tongue or {})}, 'edges': [], 'polygons': []}}


def declaration(**change):
    d = {'skin': 'Face', 'objects': {k: {'path': f'{k}.npz'} for k in ('Face', 'Upper teeth', 'Lower teeth', 'Tongue')},
         'sets': {'upper lip': UPPER.tolist(), 'lower lip': LOWER.tolist(), 'lid margin': LID.tolist(),
                  'attachment ring': RING.tolist(), 'mouth area': MOUTH_AREA.tolist(), 'lower face': LOWER_FACE.tolist()},
         'lips': {'upper': 'upper lip', 'lower': 'lower lip'},
         'shapes': {'A': {'region': 'lower face', 'still': ['lid margin']},
                    'I': {'region': 'mouth area', 'still': ['lid margin'], 'boundary': 1.6, 'seams': ['attachment ring']},
                    'O': {'region': 'mouth area', 'boundary': 1.6, 'seams': ['attachment ring']},
                    'smile': {'region': 'mouth area', 'still': ['lid margin']}},
         'controls': {'mouth_open': {'samples': np.linspace(0, 1, 11).tolist(), 'weights': {'A': np.linspace(0, 1, 11).tolist()}}},
         'jaw': {'shape': 'A', 'rigid': ['Lower teeth', 'Tongue'], 'chin': CHIN, 'upper_lip_share': .35},
         'visemes': {'slots': {'aa': 'vrc.v_aa', 'ih': 'vrc.v_ih', 'oh': 'vrc.v_oh', 'pp': 'vrc.v_pp', 'ff': 'vrc.v_ff'},
                     'basis': ['A', 'I', 'O'], 'pp': 'vrc.v_pp', 'ff': 'vrc.v_ff', 'upper_teeth': 'Upper teeth'},
         'pairs': [{'both': 'smile', 'left': 'smile_L', 'right': 'smile_R', 'split': 'feathered'},
                   {'both': 'blink', 'left': 'blink_L', 'right': 'blink_R', 'split': 'hard'}],
         'teeth': ['Upper teeth', 'Lower teeth'],
         'combinations': [{'name': 'smile + aa', 'weights': {'smile': 1., 'vrc.v_aa': 1.}},
                          {'name': 'pp', 'weights': {'vrc.v_pp': 1.}, 'press': .02}],
         'clips': [{'name': 'happy', 'weights': {'smile': .8, 'lip_raise': 0.}}],
         'expression_shapes': ['smile', 'lip_raise']}
    d.update(change)
    return d


def audit(d=None, **kw):
    return run_face_audit(d or declaration(), objects=objects(**kw))


def save_extractions(root):
    """The synthetic objects as study_extract.py files and the declaration beside them; returns its path."""
    for name, obj in objects().items():
        np.savez(root / f'{name}.npz', basis_world=obj['rest'], loops=np.array([v for f in obj['polygons'] for v in f], int),
                 sizes=np.array([len(f) for f in obj['polygons']], int), edges=np.zeros((0, 2), int),
                 **{f'key{i:04d}': delta for i, delta in enumerate(obj['keys'].values(), 1)})
        keys = [{'index': 0, 'name': 'Basis', 'relative_key': 'Basis'}] + [
            {'index': i, 'name': k, 'relative_key': 'Basis'} for i, k in enumerate(obj['keys'], 1)]
        (root / f'{name}.json').write_text(json.dumps({'object': name, 'keys': keys}), encoding='utf-8')
    path = root / 'audit.json'                         # not face.json: on Windows it is Face.json, an extraction
    path.write_text(json.dumps(declaration()), encoding='utf-8')
    return path

def status(report, key, subject=None):
    rows = [r for r in report['checks'] if r['id'] == key and (subject is None or r['subject'] == subject)]
    assert rows, (key, subject)
    return rows[0]['status'] if len(rows) == 1 else [r['status'] for r in rows]


def row(report, key, subject=None):
    return next(r for r in report['checks'] if r['id'] == key and (subject is None or r['subject'] == subject))


class FaceStudyTests(unittest.TestCase):
    def test_rigid_motion_finds_the_hinge_and_the_skin_weight(self):
        fit = rigid_motion(LOWER_TEETH, turn(LOWER_TEETH))
        self.assertAlmostEqual(fit['angle_degrees'], ANGLE, places=6)
        self.assertLess(fit['residual_rms'], 1e-9)
        h = fit['hinge_point']; self.assertAlmostEqual(h[1], 2.4, places=6); self.assertAlmostEqual(h[2], .5, places=6)
        w, off = jaw_weights(P, jaw_open(lift=0.), fit['rotation'], fit['translation'])
        self.assertAlmostEqual(w[CHIN], 1., places=6)
        self.assertLess(off[BELOW & (w > .1)].max(), 1e-6)

    def test_viseme_mix_and_seam_rings(self):
        mix = viseme_mix({'A': SKIN['A'], 'I': SKIN['I'], 'O': SKIN['O']}, SKIN['vrc.v_oh'])
        self.assertAlmostEqual(mix['weights']['A'], .2, places=6); self.assertAlmostEqual(mix['weights']['O'], .7, places=6)
        self.assertLess(mix['relative_residual'], 1e-9)
        edges = np.unique(np.sort(np.array([(f[i], f[(i + 1) % 4]) for f in FACES for i in range(4)]), axis=1), axis=0)
        ring = seam_rings(edges, np.r_[UPPER, LOWER], len(P))
        self.assertEqual(ring[UPPER[0]], 0.); self.assertEqual(ring[int(np.argmin(np.linalg.norm(P - [0, 0, .3], axis=1)))], 3.)


class FaceAuditTests(unittest.TestCase):
    def test_the_good_construction_passes_every_check(self):
        report = audit()
        failing = [(r['id'], r['subject'], r['observed']) for r in report['checks'] if r['status'] != 'pass']
        self.assertEqual(report['status'], 'pass', failing)
        self.assertEqual({r['id'] for r in report['checks']}, set(LIMITS) - set())
        jaw = report['report']['jaw']
        self.assertAlmostEqual(jaw['angle_degrees'], ANGLE, places=6)
        self.assertAlmostEqual(jaw['behind'], 2.4, places=6); self.assertAlmostEqual(jaw['above'], .5, places=6)
        self.assertLess(jaw['chord_sag_at_half'], .01)                          # a jaw needs no in-betweens
        self.assertLess(report['report']['visemes']['mixes']['oh']['relative_residual'], 1e-9)

    def test_C1_a_shape_leaking_outside_its_region_or_onto_the_lid_fails(self):
        leak = SKIN['smile'].copy(); leak[LID, 2] += .03
        cheek = SKIN['smile'] + np.c_[np.zeros(len(P)), np.zeros(len(P)), .02 * falloff(1, 1., (1.6, .8))]
        d = declaration(shapes={'smile': {'region': 'mouth area', 'still': ['lid margin']}})
        self.assertEqual(status(audit(d, skin={'smile': leak}), 'still_travel_share'), 'fail')
        self.assertEqual(status(audit(d, skin={'smile': cheek}), 'region_outside_share'), 'fail')

    def test_C2_C10_a_control_driving_a_phase_gated_guide_key_fails(self):
        bump = lambda s: smooth(np.minimum(s / .33, (1 - s) / .67))
        s = np.linspace(0, 1, 11)
        guide = np.c_[np.zeros(len(P)), -.1 * BELOW, np.zeros(len(P))]
        d = declaration(controls={'mouth_open': {'samples': s.tolist(), 'weights': {'A': s.tolist(),
                                                                                    'guide_key_mid': bump(s).tolist()}}})
        report = audit(d, skin={'guide_key_mid': guide})
        self.assertEqual([status(report, k) for k in ('phase_gated_keys', 'path_deviation')], ['fail', 'fail'])
        self.assertEqual(list(row(report, 'phase_gated_keys')['detail']['keys']), ['guide_key_mid'])
        late = declaration(controls={'mouth_open': {'samples': s.tolist(), 'weights': {'A': smooth((s - .33) / .67).tolist()}}})
        self.assertEqual(status(audit(late), 'phase_gated_keys'), 'fail')     # one key, but gated to the late phase
        back = declaration(controls={'mouth_open': {'samples': s.tolist(), 'weights': {'A': np.minimum(s, 1.2 - s).tolist()}}})
        self.assertEqual(status(audit(back), 'path_reversals'), 'fail')

    def test_C3_morphing_teeth_or_a_hinge_in_front_of_the_lips_fails(self):
        morph = turn(LOWER_TEETH) * [1.08, 1., 1.] - LOWER_TEETH
        self.assertEqual(status(audit(lower={'A': morph}), 'jaw_rigid_residual', 'Lower teeth'), 'fail')
        front = np.array([0., -.5, .5])
        report = audit(skin={'A': jaw_open(hinge=front)}, lower={'A': turn(LOWER_TEETH, hinge=front) - LOWER_TEETH},
                       tongue={'A': turn(TONGUE, hinge=front) - TONGUE})
        self.assertEqual(status(report, 'jaw_hinge_position'), 'fail')
        self.assertAlmostEqual(row(report, 'jaw_hinge_position')['detail']['behind'], -.5, places=6)

    def test_C4_skin_not_following_the_jaw_or_a_band_weight_fails(self):
        slide = jaw_open(); slide[BELOW] = np.c_[.1 * P[BELOW, 0], np.zeros(BELOW.sum()), np.zeros(BELOW.sum())]
        self.assertEqual(status(audit(skin={'A': slide}), 'jaw_skin_residual'), 'fail')
        dist = np.linalg.norm(P - P[CHIN], axis=1)
        band = np.where(dist < .4, .9, np.where(dist < 1., .1, np.where(dist < 1.6, .9, 0.))) * BELOW
        self.assertEqual(status(audit(skin={'A': jaw_open(weight=band)}), 'jaw_weight_monotone'), 'fail')
        dragged = jaw_open(); top = ~BELOW & (R_XZ < .8)
        dragged[top] = .6 * (turn(P[top]) - P[top])
        self.assertEqual(status(audit(skin={'A': dragged}), 'jaw_drags_upper_face'), 'fail')
        self.assertEqual(status(audit(skin={'A': jaw_open(lift=.03)}), 'upper_lip_share_error'), 'fail')

    def test_C5_a_lip_shape_past_its_boundary_or_falling_off_on_a_seam_fails(self):
        wide = np.c_[.1 * P[:, 0] * falloff(1, 2.2), np.zeros(len(P)), np.zeros(len(P))]
        report = audit(skin={'I': wide})
        self.assertEqual([status(report, 'falloff_beyond_boundary', 'I'), status(report, 'falloff_on_seam',
                                                                                 'I / attachment ring')], ['fail', 'fail'])
        rings = row(report, 'falloff_beyond_boundary', 'O')['detail']['largest_per_ring_from_seam']
        self.assertEqual(rings[-1], 0.)

    def test_C6_swapped_slots_an_open_pp_and_an_ff_short_of_the_teeth_fail(self):
        slots = {'aa': 'vrc.v_aa', 'ih': 'vrc.v_oh', 'oh': 'vrc.v_ih'}
        d = declaration(visemes=dict(declaration()['visemes'], slots=slots))
        self.assertEqual(row(audit(d), 'viseme_slot_mapping')['observed'], 2.)
        self.assertEqual(status(audit(skin={'vrc.v_pp': lips_move(.03, -.03)}), 'pp_gap'), 'fail')
        short = np.zeros_like(P); short[LOWER] = [0., 0., -.05]
        self.assertEqual(status(audit(skin={'vrc.v_ff': short}), 'ff_gap'), 'fail')

    def test_C7_halves_that_do_not_add_up_or_split_the_wrong_way_fail(self):
        self.assertEqual(status(audit(skin={'smile_R': .8 * SKIN['smile_R']}), 'lr_sum_error', 'smile'), 'fail')
        hard = {'smile_L': SKIN['smile'] * (P[:, 0] >= 0)[:, None], 'smile_R': SKIN['smile'] * (P[:, 0] < 0)[:, None]}
        report = audit(skin=hard)
        self.assertEqual((status(report, 'lr_midline', 'smile'), status(report, 'lr_sum_error', 'smile')), ('fail', 'pass'))
        soft = {'blink_L': SKIN['blink'] * smooth((P[:, 0] + 1.) / 2.)[:, None],
                'blink_R': SKIN['blink'] * (1 - smooth((P[:, 0] + 1.) / 2.))[:, None]}
        self.assertEqual(status(audit(skin=soft), 'lr_midline', 'blink'), 'fail')
        share = np.interp(P[:, 0], [-.15, -.05, 0., .1, .2, .3], [0., .2, .8, .3, .9, 1.])     # up, down, up again
        dip = {'smile_L': SKIN['smile'] * share[:, None], 'smile_R': SKIN['smile'] * (1 - share)[:, None]}
        self.assertEqual(status(audit(skin=dip), 'lr_crease', 'smile'), 'fail')

    def test_C8_crossing_lips_and_teeth_in_front_of_the_lips_fail(self):
        d = declaration(combinations=[{'name': 'raise + pp', 'weights': {'lip_raise': 1., 'vrc.v_pp': 1.}, 'press': .02},
                                      {'name': 'jaw forward', 'weights': {'jaw_forward': 1.}}])
        forward = {'jaw_forward': np.tile([0., -.2, 0.], (len(LOWER_TEETH), 1))}
        report = audit(d, lower=forward, skin={'jaw_forward': np.zeros_like(P)})
        self.assertEqual(status(report, 'lips_cross', 'raise + pp'), 'fail')
        self.assertAlmostEqual(row(report, 'lips_cross', 'raise + pp')['observed'], .07, places=6)
        self.assertEqual(status(report, 'teeth_behind_lips', 'jaw forward'), 'fail')
        self.assertEqual(status(report, 'lips_cross', 'jaw forward'), 'pass')

    def test_C9_a_key_stored_on_another_key_and_a_partial_clip_fail(self):
        report = audit(declaration(clips=[{'name': 'partial', 'weights': {'smile': .8}}]), info={'vrc.v_ih': 'vrc.v_aa'})
        self.assertEqual(status(report, 'single_frame_keys', 'Face'), 'fail')
        self.assertEqual(row(report, 'clip_full_state', 'partial')['detail']['unset'], ['lip_raise'])

    def test_declaration_files_limits_and_refusals(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        self.assertEqual(run_face_audit(save_extractions(root))['status'], 'pass')
        loose = declaration(limits={'jaw_hinge_position': 5.})
        self.assertEqual(row(audit(loose), 'jaw_hinge_position')['limit'], 5.)
        with self.assertRaisesRegex(ValueError, 'known check'):
            audit(declaration(limits={'nonsense': 1.}))
        with self.assertRaisesRegex(ValueError, 'needs the lips'):
            audit(declaration(lips=None))
        with self.assertRaisesRegex(ValueError, 'no object has a shape'):
            audit(declaration(shapes={'missing': {'region': 'mouth area'}}))
        with self.assertRaisesRegex(ValueError, 'rising samples'):
            audit(declaration(controls={'c': {'samples': [0, 1], 'weights': {'A': [0, 1]}}}))


if __name__ == '__main__':
    unittest.main()
