"""Studying a reference construction on synthetic avatars; the extraction worker runs when a Blender is available."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .study import band_profile, blink_report, load_shapes, motion_models, order_loop, rings

BLENDER = os.environ.get('MODELING_BLENDER')


def grid(nx, nz, width=1., height=1.):
    x, z = np.meshgrid(np.linspace(-width / 2, width / 2, nx), np.linspace(0, height, nz))
    P = np.c_[x.ravel(), np.zeros(x.size), z.ravel()]
    idx = np.arange(nx * nz).reshape(nz, nx)
    edges = np.r_[np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()], np.c_[idx[:-1].ravel(), idx[1:].ravel()]]
    return P, idx, edges


class StudyTests(unittest.TestCase):
    def test_a_flat_curtain_reads_as_a_slide_and_a_hinged_lid_as_a_turn(self):
        P, idx, _ = grid(9, 5)
        band = idx[2:].ravel()                                                       # the upper rows move
        down = P.copy(); down[band] += [0., .03, -.4]                                # one direction: a curtain
        slide = motion_models(P, down, band, axes={'hinge': ([0, 1, .2], [1, 0, 0])})
        self.assertLess(slide['slide']['residual_share'], 1e-9)
        self.assertGreater(slide['hinge']['residual_share'], .05)
        a = np.radians(35.); R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        centre = np.array([0., 1., .2]); turned = P.copy(); turned[band] = (P[band] - centre) @ R.T + centre
        turn = motion_models(P, turned, band, axes={'hinge': (centre, [1, 0, 0])}, guess=(centre + .05, [1, .1, 0]))
        self.assertLess(turn['hinge']['residual_share'], 1e-9)
        self.assertGreater(turn['slide']['residual_share'], .05)
        self.assertLess(turn['free_axis']['residual_share'], 1e-6)

    def test_loop_rings_and_band_profile(self):
        P, idx, edges = grid(7, 7)
        square = [int(v) for v in np.r_[idx[2, 2:5], idx[3, 4], idx[4, 4:1:-1], idx[3, 2]]]
        loop = order_loop(square[::-1], edges)
        self.assertEqual(sorted(loop.tolist()), sorted(square))
        ring = rings(P, edges, loop, outward=2, inward=1)
        self.assertEqual((ring[idx[3, 3]], ring[idx[0, 3]], ring[idx[1, 3]]), (-1., 2., 1.))
        with self.assertRaisesRegex(ValueError, 'closed loop'):
            order_loop(square[:4], edges)
        margin = idx[2]; end = P.copy()
        for k, r in enumerate(idx[2:]):
            end[r, 2] -= .2 * max(0., 1 - k / 3)                                     # full at the margin, gone by 3 rows up
        profile = band_profile(P, end, margin, idx[2:].ravel(), width=1.)
        self.assertAlmostEqual(profile['rows'][0]['travel_share'], 1.)
        self.assertAlmostEqual(profile['reach_share'], .5, places=6)

    def test_blink_report_in_the_eyes_own_width(self):
        P, idx, _ = grid(9, 5, width=2.)
        upper, lower = idx[3], idx[1]; corners = (int(idx[2, 0]), int(idx[2, -1]))
        end = P.copy(); end[upper, 2] = P[idx[2], 2] - .1 * (1 - (P[upper, 0] / 1.) ** 2)   # a closed line .1 deep
        report = blink_report(P, end, upper, lower, corners)
        self.assertAlmostEqual(report['width'], 2.)
        self.assertAlmostEqual(report['closed_depth_share'], .05)
        self.assertEqual(report['lower_to_upper_travel'], 0.)
        self.assertEqual(report['corner_travel_share'], [0., 0.])

    @unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
    def test_extraction_worker_reads_every_moving_shape_key(self):
        from .test_live import SCENE_BUILDER
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        (root / 'build.py').write_text(SCENE_BUILDER)
        subprocess.run([BLENDER, '--background', '--factory-startup', '--python', str(root / 'build.py'), '--',
                        str(root / 'avatar.blend')], check=True, capture_output=True, timeout=180)
        job = {'script': str(Path(__file__).with_name('study_extract.py')), 'blender': BLENDER,
               'input': str(root / 'avatar.blend'), 'output_root': str(root / 'out'), 'object': 'Face',
               'resolution': 32, 'timeout_seconds': 300}
        (root / 'job.json').write_text(json.dumps(job))
        done = subprocess.run([sys.executable, '-m', 'modeling_system.isolated_blender', str(root / 'job.json')],
                              capture_output=True, text=True, timeout=400, cwd=str(Path(__file__).resolve().parents[1]))
        receipt = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt['status'], 'completed', done.stdout[-2000:])
        shapes = load_shapes(Path(receipt['output']) / 'Face.npz')
        self.assertEqual(list(shapes['keys']), ['Blink'])
        moved = np.linalg.norm(shapes['keys']['Blink'], axis=1) > 1e-6
        self.assertEqual(int(moved.sum()), 4)                                        # the cube's top four vertices
        np.testing.assert_allclose(shapes['keys']['Blink'][moved, 2], -.3, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
