"""The construction-first verbs through the service (the route the CLI and MCP use)."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from .init_workspace import initialize
from .service import ModelingService
from . import test_checks as eye
from . import test_face_checks as face
from . import test_registration as heads


class ServiceVerbTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.root, True)
        initialize(self.root / 'ws', 'Synthetic'); self.service = ModelingService(self.root / 'ws')

    def run_op(self, operation, arguments):
        result = self.service.execute(operation, arguments)
        self.assertNotIn(result.get('status'), ('failed', 'needs attention', 'conflicting'), result)
        json.dumps(result, allow_nan=False)                                   # every transport returns plain JSON
        return result

    def test_the_verbs_are_service_operations(self):
        ops = set(self.service.operations())
        self.assertLessEqual({'check_candidate', 'register_guide', 'construct', 'study_blink', 'overlay_on_drawing',
                              'review_sheet', 'bake_poses', 'audit_face'}, ops)

    def test_construct_then_check(self):
        np.savez(self.root / 'lid.npz', positions=eye.REST, edge=eye.row(eye.UPPER), landing=eye.REST[eye.row(eye.LOWER)],
                 pivot=np.array(eye.CENTRE), axis=np.array(eye.AXIS), weights=eye.WEIGHTS)
        landed = self.run_op('construct', {'operation': 'hinge_landing', 'inputs': str(self.root / 'lid.npz'),
                                           'output': str(self.root / 'closed.npz'), 'arguments': {'outside': .001}})
        self.assertIn('positions', landed['output']['arrays'])
        self.assertEqual(landed['public_metrics']['edge_points_beyond_landing_range'], 0)
        with np.load(self.root / 'closed.npz') as closed:
            np.testing.assert_allclose(closed['positions'], eye.CLOSED, atol=1e-12)
        np.savez(self.root / 'poses.npz', **eye.states(eye.hinged()))
        (self.root / 'declaration.json').write_text(json.dumps({'object': 'Skin', 'region': eye.REGION.tolist(),
            'closing': {'moving': eye.row(eye.UPPER).tolist(), 'facing': eye.row(eye.LOWER).tolist(),
                        'pivot': eye.CENTRE, 'axis': eye.AXIS}}))
        report = self.run_op('check_candidate', {'declaration': str(self.root / 'declaration.json'),
                                                 'candidate': str(self.root / 'poses.npz')})
        self.assertEqual(report['status'], 'pass')
        refused = self.service.execute('construct', {'operation': 'nonsense', 'inputs': str(self.root / 'lid.npz'),
                                                     'output': str(self.root / 'x.npz')})
        self.assertEqual(refused['status'], 'failed')

    def test_bake_saved_poses(self):
        np.savez(self.root / 'evaluated.npz', **eye.states(eye.hinged()))
        report = self.run_op('bake_poses', {'poses': str(self.root / 'evaluated.npz'), 'objects': ['Skin'],
                                            'tolerance': .05, 'output': str(self.root / 'bake' / 'blink.npz')})
        self.assertEqual(report['drivers'], {'blink': 'main', 'blink_mid': 'mid'})
        clip = json.loads(Path(report['clip_file']).read_text())
        self.assertEqual(set(clip['frames'][0]['weights']), {'blink', 'blink_mid'})
        with np.load(report['bake_file']) as bake:
            self.assertEqual(sorted(bake.files), ['Skin::faces', 'Skin::rest', 'Skin::shape::blink', 'Skin::shape::blink_mid'])
        unity = self.run_op('bake_poses', {'poses': str(self.root / 'evaluated.npz'), 'objects': ['Skin'], 'tolerance': .05,
                                           'output': str(self.root / 'bake' / 'unity.npz'), 'renderers': {'Skin': 'Body'}})
        self.assertEqual((unity['anim_check']['status'], unity['anim_check']['curves']), ('passed', 2))
        self.assertTrue(Path(unity['anim_file']).is_file())
        refused = self.service.execute('bake_poses', {'poses': str(self.root / 'evaluated.npz'), 'objects': ['Skin'],
                                                      'tolerance': .05, 'output': str(self.root / 'bake' / 'x.npz'),
                                                      'renderers': {'Lash': 'Body/Lash'}})
        self.assertEqual(refused['status'], 'failed')

    def test_audit_a_face(self):
        report = self.run_op('audit_face', {'declaration': str(face.save_extractions(self.root))})
        self.assertEqual(report['status'], 'pass')
        self.assertAlmostEqual(report['report']['jaw']['behind'], 2.4, places=6)

    def test_register_a_guide(self):
        case = heads.RegistrationTests(); case.setUp()
        np.savez(self.root / 'generated.npz', co=case.generated); np.savez(self.root / 'neutral.npz', co=case.reference)
        result = self.run_op('register_guide', {'generated': str(self.root / 'generated.npz'),
                                                'reference': str(self.root / 'neutral.npz'),
                                                'output': str(self.root / 'registered.npz'), 'exclude': heads.EYES})
        self.assertAlmostEqual(result['scale'], 2.5, places=6)
        self.assertLess(result['public_metrics']['stationary_distance_p95'], 1e-6)

    def test_study_a_blink(self):
        P, idx, _ = __import__('modeling_system.test_study', fromlist=['grid']).grid(9, 5, width=2.)
        delta = np.zeros_like(P); delta[idx[3], 2] = P[idx[2], 2] - P[idx[3], 2] - .1 * (1 - P[idx[3], 0] ** 2)
        np.savez(self.root / 'Body.npz', basis_world=P, loops=np.zeros(0, int), sizes=np.zeros(0, int),
                 edges=np.zeros((0, 2), int), key0001=delta.astype(np.float32))
        (self.root / 'Body.json').write_text(json.dumps({'object': 'Body', 'keys': [{'index': 0, 'name': 'Basis'},
                                                                                     {'index': 1, 'name': 'blink'}]}))
        report = self.run_op('study_blink', {'shapes': str(self.root / 'Body.npz'), 'key': 'blink',
                                             'upper_margin': idx[3].tolist(), 'lower_margin': idx[1].tolist(),
                                             'corners': [int(idx[2, 0]), int(idx[2, -1])]})
        self.assertAlmostEqual(report['closed_depth_share'], .05, places=6)

    def test_built_options_shown_together_in_one_call(self):
        variants = {}
        for n, name in enumerate(('A', 'B', 'C')):
            variants[name] = {}
            for control in (0., .5, 1.):
                variants[name][str(control)] = {}
                for view in ('front', 'side'):
                    path = self.root / f'{name}-{control}-{view}.png'
                    Image.new('RGB', (64, 48), (80 * n, int(200 * control), 90 if view == 'front' else 180)).save(path)
                    variants[name][str(control)][view] = str(path)
        report = self.run_op('review_variants', {'variants': variants, 'output_dir': str(self.root / 'options'),
                                                 'labels': {'A': 'lower lid still', 'B': 'middle rise', 'C': 'avatar-like'},
                                                 'stills': [0., 1.]})
        self.assertEqual((report['variants'], report['views'], report['controls']), (['A', 'B', 'C'], ['front', 'side'], [0., .5, 1.]))
        self.assertEqual(len(report['videos']), 2)
        self.assertTrue(all(Path(p).is_file() for p in report['videos'] + report['stills']))
        with Image.open(report['stills'][0]) as sheet:
            self.assertGreater(sheet.size[0], sheet.size[1])                 # three option columns, two rows
        del variants['B']['0.5']
        refused = self.service.execute('review_variants', {'variants': variants, 'output_dir': str(self.root / 'bad')})
        self.assertEqual(refused['status'], 'failed')

    def test_show_on_a_drawing_and_in_a_sheet(self):
        Image.new('RGB', (100, 100), (255, 140, 40)).save(self.root / 'render.png')
        Image.new('RGB', (200, 150), (255, 255, 255)).save(self.root / 'drawing.png')
        overlay = self.run_op('overlay_on_drawing', {'render': str(self.root / 'render.png'), 'drawing': str(self.root / 'drawing.png'),
                                                     'render_points': [[20, 50], [80, 50]], 'drawing_points': [[40, 75], [160, 75]],
                                                     'output': str(self.root / 'overlay.png')})
        self.assertAlmostEqual(overlay['similarity_render_to_drawing']['scale'], 2.)
        sheet = self.run_op('review_sheet', {'rows': [{'label': 'closed', 'images': [str(self.root / 'overlay.png'), None]}],
                                             'columns': ['overlay', 'absent'], 'output': str(self.root / 'sheet.png')})
        self.assertTrue(Path(sheet['path']).is_file())


if __name__ == '__main__':
    unittest.main()
