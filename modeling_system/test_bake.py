"""Baking a built motion to the fewest blend shapes; the FBX round trip runs when MODELING_BLENDER names a Blender."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from .bake import (BASES, bake_poses, bake_shapes, check_unity_anim, clip_curve, evaluate, export_fbx, split_sides,
                   write_bake, write_unity_anim)
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

    def test_the_unity_clip_keys_every_shape_on_every_renderer(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        margin = eye.row(eye.UPPER); lashes = eye.REST[margin] + [0., -.03, -.02]
        carried = hinge_carry(lashes, eye.REST[margin], eye.CLOSED[margin], eye.CENTRE, eye.AXIS, fraction=PHASES)['positions']
        bake = bake_shapes({'Skin': (eye.REST, rolled()), 'Lash': (lashes, carried)}, PHASES, tolerance=.002)
        clip = clip_curve(bake['drivers'], fps=60); renderers = {'Skin': 'Body', 'Lash': 'Body/Lash'}
        written = write_unity_anim(root / 'blink.anim', clip, renderers)
        names = sorted(bake['drivers'])                      # every shape the bake chose, on every renderer
        self.assertEqual(sorted(written), [(r, s) for r in ('Body', 'Body/Lash') for s in names])
        text = (root / 'blink.anim').read_text(encoding='utf-8')
        self.assertTrue(text.startswith('%YAML 1.1\n%TAG !u! tag:unity3d.com,2011:\n--- !u!74 &7400000\nAnimationClip:'))
        float_curves = text.split('  m_FloatCurves:\n')[1].split('\n  m_PPtrCurves:')[0]
        editor_curves = text.split('  m_EditorCurves:\n')[1].split('\n  m_EulerEditorCurves:')[0]
        self.assertEqual(float_curves, editor_curves)
        parsed = {}                                          # parsed here, independently of check_unity_anim
        blocks = ('\n' + float_curves).split('\n  - curve:')[1:]
        for block in blocks:
            key = (re.search(r'\n    path: (.+)', block).group(1), re.search(r'attribute: (.+)', block).group(1))
            parsed[key] = [(float(t), float(v)) for t, v in re.findall(r'time: (\S+)\n\s+value: (\S+)', block)]
        self.assertEqual(set(parsed), {(r, f'blendShape.{s}') for r in ('Body', 'Body/Lash') for s in names})
        for (renderer, attribute), keys in parsed.items():
            shape = attribute.split('.', 1)[1]
            self.assertEqual(len(keys), len(clip['frames']))
            for (t, v), frame in zip(keys, clip['frames']):
                self.assertAlmostEqual(t, frame['time'], places=6)
                self.assertAlmostEqual(v, 100. * frame['weights'][shape], places=4)          # weights 0-100
        self.assertEqual(max(v for keys in parsed.values() for _, v in keys), 100.)
        self.assertEqual(check_unity_anim(root / 'blink.anim', clip, renderers)['status'], 'passed')
        # a missing object or shape fails the check
        write_unity_anim(root / 'skin-only.anim', clip, {'Skin': 'Body'})
        missing = check_unity_anim(root / 'skin-only.anim', clip, renderers)
        self.assertEqual((missing['status'], missing['missing']), ('failed', [('Body/Lash', s) for s in names]))
        one = next(b for b in blocks if f'blendShape.{names[-1]}\n' in b and 'path: Body\n' in b)
        (root / 'cut.anim').write_text(text.replace('  - curve:' + one, '', 1), encoding='utf-8')
        self.assertEqual(check_unity_anim(root / 'cut.anim', clip, renderers)['missing'], [('Body', names[-1])])
        with self.assertRaisesRegex(ValueError, 'not in the clip'):
            write_unity_anim(root / 'x.anim', clip, {'Skin': 'Body'}, shapes={'Skin': ['blink', 'not_baked']})
        with self.assertRaisesRegex(ValueError, 'its own renderer path'):
            write_unity_anim(root / 'x.anim', clip, {'Skin': 'Body', 'Lash': 'Body'})

    def test_a_bake_splits_into_left_and_right_shapes_for_winks(self):
        pair = np.concatenate([eye.REST, eye.REST * [-1, 1, 1] + [-2., 0., 0.]])          # two lids, mirrored at x = -1
        poses = np.concatenate([rolled(), rolled() * [-1, 1, 1] + [-2., 0., 0.]], axis=1)
        bake = bake_shapes({'Face': (pair, poses)}, PHASES, tolerance=1e-12, max_shapes=2)
        sided = split_sides(bake, {'Face': pair}, axis=0, plane=-1.)
        self.assertEqual(sided['drivers'], {'blink_L': 'main', 'blink_R': 'main', 'blink_L_mid': 'mid', 'blink_R_mid': 'mid'})
        shapes = sided['shapes']['Face']; n = len(eye.REST)
        for name, left, right in (('blink', 'blink_L', 'blink_R'), ('blink_mid', 'blink_L_mid', 'blink_R_mid')):
            L, R = shapes[left], shapes[right]
            np.testing.assert_allclose(L + R, bake['shapes']['Face'][name], atol=1e-15)      # the halves add up
            self.assertEqual((np.abs(L[n:]).max(), np.abs(R[:n]).max()), (0., 0.))           # + x holds the left
        wink = clip_curve({k: v for k, v in sided['drivers'].items() if '_L' in k})
        self.assertEqual(set(wink['frames'][0]['weights']), {'blink_L', 'blink_L_mid'})
        torn = pair.copy(); torn[eye.row(eye.UPPER)[0], 0] = -1.                          # a moving point on the midline
        with self.assertRaisesRegex(ValueError, 'move points on the midline'):
            split_sides(bake, {'Face': torn}, axis=0, plane=-1.)

    def test_export_names_replace_object_names_that_would_break_unity_paths(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        arrays = {f'{g}::Rig / Face::co': P for g, P in zip(PHASES, rolled())}; arrays['0.0::Rig / Face::tri'] = eye.TRI
        np.savez(root / 'poses.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'separates Unity paths'):
            bake_poses(root / 'poses.npz', ['Rig / Face'], tolerance=.05, output=root / 'bad.npz')
        report = bake_poses(root / 'poses.npz', ['Rig / Face'], tolerance=.05, output=root / 'bake.npz',
                            names={'Rig / Face': 'Face'}, sides={'axis': 0, 'plane': -.65})
        self.assertEqual(report['export_names'], {'Rig / Face': 'Face'})
        with np.load(report['bake_file']) as data:
            self.assertIn('Face::shape::blink_L', data.files)

    @unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
    def test_fbx_from_the_source_keeps_its_materials_and_uv_maps(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        build = f"""
import bpy, numpy as np
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_grid_add(x_subdivisions=8, y_subdivisions=8, size=1.)
ob = bpy.context.active_object; ob.name = 'Face'; ob.location = (.1, -.2, .3); ob.scale = (2., 2., 2.)
ob.data.uv_layers.new(name='Detail')
material = bpy.data.materials.new('Skin'); ob.data.materials.append(material)
bpy.context.view_layer.update()
M = np.array(ob.matrix_world); co = np.empty(len(ob.data.vertices) * 3); ob.data.vertices.foreach_get('co', co)
co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
tri = np.array([[p.vertices[0], p.vertices[i], p.vertices[i + 1]] for p in ob.data.polygons for i in range(1, len(p.vertices) - 1)])
np.savez(r'{(root / 'rest.npz').as_posix()}', co=co, tri=tri)
bpy.ops.wm.save_as_mainfile(filepath=r'{(root / 'source.blend').as_posix()}')
"""
        subprocess.run([BLENDER, '--background', '--factory-startup', '--python-expr', build], check=True,
                       capture_output=True, timeout=300)
        with np.load(root / 'rest.npz') as data:
            rest, tri = data['co'], data['tri']
        lift = np.c_[np.zeros((len(rest), 2)), .1 * np.exp(-np.sum((rest[:, :2] - [.1, -.2]) ** 2, 1) / .1)]
        poses = rest[None] + PHASES[:, None, None] * lift[None]
        bake = bake_shapes({'Face': (rest, poses)}, PHASES, tolerance=1e-9)
        write_bake(root / 'bake.npz', bake, {'Face': tri}, {'Face': rest})
        record = export_fbx(root / 'bake.npz', clip_curve(bake['drivers'], fps=30), blender=BLENDER,
                            output_root=root / 'export', timeout=300, source=root / 'source.blend')
        self.assertEqual(record['status'], 'passed', record)
        face = record['objects']['Face']
        self.assertEqual((face['materials']['source'], face['materials']['imported']), (['Skin'], ['Skin']))
        self.assertEqual(face['uv_maps']['imported'], ['Detail', 'UVMap'])
        self.assertLess(max(face['uv_maps']['largest_error'].values()), 1e-6)
        write_bake(root / 'moved.npz', bake, {'Face': tri}, {'Face': rest + [0., 0., .01]})   # not the source's rest
        refused = export_fbx(root / 'moved.npz', clip_curve(bake['drivers'], fps=30), blender=BLENDER,
                             output_root=root / 'export2', timeout=300, source=root / 'source.blend')
        self.assertEqual(refused['status'], 'failed')

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
