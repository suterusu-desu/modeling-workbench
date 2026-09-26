"""Baking a built motion to the fewest blend shapes; the FBX round trip runs when MODELING_BLENDER names a Blender."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from .bake import BASES, bake_shapes, clip_curve, evaluate, export_fbx, write_bake
from .motion_paths import hinge_carry, path_positions
from . import test_checks as eye

BLENDER = os.environ.get('MODELING_BLENDER')
PHASES = np.linspace(0., 1., 11)


def rolled():
    pace = np.repeat(PHASES[:, None], len(eye.REST), axis=1)
    return path_positions(eye.REST, eye.CLOSED, pace, pivot=eye.CENTRE, axis=eye.AXIS)['positions']


def chord():
    return eye.REST[None] + PHASES[:, None, None] * (eye.CLOSED - eye.REST)[None]


class BakeTests(unittest.TestCase):
    def test_a_straight_slide_is_one_exact_shape(self):
        bake = bake_shapes({'Skin': (eye.REST, chord())}, PHASES, tolerance=1e-6)
        self.assertEqual((bake['shape_count'], list(bake['drivers'])), (1, ['blink']))
        self.assertLess(bake['max_error'], 1e-12)

    def test_a_roll_over_a_round_eye_needs_the_mid_corrective_and_one_shape_cuts_the_eye(self):
        poses = rolled()
        probe = bake_shapes({'Skin': (eye.REST, poses)}, PHASES, tolerance=1e-12,
                            obstacles={'Skin': {'obstacle': eye.EYE, 'centre': eye.CENTRE}})
        one, two = probe['candidates'][0], probe['candidates'][1]
        self.assertGreater(one['max_error'], 8 * two['max_error'])                 # a 75-degree roll
        self.assertLess(one['objects']['Skin']['clearance']['closest'], 0.)     # the chord goes inside the eye
        self.assertGreater(two['objects']['Skin']['clearance']['closest'], 0.)
        bake = bake_shapes({'Skin': (eye.REST, poses)}, PHASES, tolerance=2 * two['max_error'])
        self.assertEqual(bake['drivers'], {'blink': 'main', 'blink_mid': 'mid'})
        self.assertTrue(bake['within_tolerance'])
        shapes = bake['shapes']['Skin']
        np.testing.assert_allclose(evaluate(eye.REST, shapes, 0., bake['drivers']), eye.REST, atol=1e-12)       # rest exact
        np.testing.assert_allclose(evaluate(eye.REST, shapes, 1., bake['drivers']), eye.CLOSED, atol=1e-12)     # closed exact
        for k, s in enumerate(PHASES):
            self.assertLessEqual(np.linalg.norm(evaluate(eye.REST, shapes, s, bake['drivers']) - poses[k], axis=1).max(),
                                 bake['objects']['Skin']['per_phase_error'][float(s)] + 1e-12)

    def test_lashes_and_skin_share_the_drivers(self):
        margin = eye.row(eye.UPPER); lashes = eye.REST[margin] + [0., -.03, -.02]
        carried = hinge_carry(lashes, eye.REST[margin], eye.CLOSED[margin], eye.CENTRE, eye.AXIS, fraction=PHASES)['positions']
        bake = bake_shapes({'Skin': (eye.REST, rolled()), 'Lash': (lashes, carried)}, PHASES, tolerance=.002)
        self.assertEqual(set(bake['shapes']), {'Skin', 'Lash'})
        self.assertEqual(set(bake['shapes']['Skin']), set(bake['shapes']['Lash']))
        self.assertTrue(bake['within_tolerance'])

    def test_the_clip_closes_holds_and_opens_on_the_drivers(self):
        clip = clip_curve({'blink': 'main', 'blink_mid': 'mid'}, fps=60)
        frames = clip['frames']
        self.assertEqual(len(frames), round((.083 + .033 + .217) * 60) + 1)
        self.assertEqual((frames[0]['weight'], frames[-1]['weight']), (0., 0.))
        hold = [f for f in frames if .083 <= f['time'] <= .116]
        self.assertTrue(hold and all(f['weights']['blink'] == 1. and f['weights']['blink_mid'] == 0. for f in hold))
        self.assertTrue(all(0. <= w <= 1. for f in frames for w in f['weights'].values()))
        self.assertAlmostEqual(BASES['mid'](.5), 1.)

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'Phases must rise'):
            bake_shapes({'Skin': (eye.REST, rolled())}, PHASES[::-1], tolerance=.01)
        bad = rolled().copy(); bad[0] += .01
        with self.assertRaisesRegex(ValueError, 'rest pose'):
            bake_shapes({'Skin': (eye.REST, bad)}, PHASES, tolerance=.01)
        with self.assertRaisesRegex(ValueError, 'Close and open'):
            clip_curve({'blink': 'main'}, timing={'close': 0.})

    @unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
    def test_fbx_round_trip_keeps_the_shapes_and_the_clip(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        bake = bake_shapes({'Skin': (eye.REST, rolled())}, PHASES, tolerance=.02)
        write_bake(root / 'bake.npz', bake, {'Skin': eye.TRI}, {'Skin': eye.REST})
        record = export_fbx(root / 'bake.npz', clip_curve(bake['drivers'], fps=30), blender=BLENDER,
                            output_root=root / 'export', timeout=300)
        self.assertEqual(record['status'], 'passed', record)
        skin = record['objects']['Skin']
        self.assertEqual(set(skin['position_error']), {'Basis', 'blink', 'blink_mid'})
        self.assertTrue(Path(record['fbx']['path']).is_file())


if __name__ == '__main__':
    unittest.main()
