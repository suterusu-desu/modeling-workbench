"""Trial, reopen and retain as plain calls; the native route runs when MODELING_BLENDER names a Blender."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

BLENDER = os.environ.get('MODELING_BLENDER')
CHANGE = '''
face = bpy.data.objects['Face']
blink = face.data.shape_keys.key_blocks['Blink']
for i, v in enumerate(face.data.vertices):
    if v.co.z > 0: blink.data[i].co.z -= .1          # the blink closes further
'''


class TrialLaneTests(unittest.TestCase):
    def test_refusals_without_a_native_route(self):
        from .trials import Trials

        class Service:
            workspace = Path(tempfile.gettempdir())
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        with self.assertRaisesRegex(ValueError, 'rest pose'):
            Trials(Service(), 'o', root, blender='b', reference={'working': 'F'}, objects=['F'], poses={'0.5': {}})
        lane = Trials(Service(), 'o', root, blender='b', reference={'working': 'F'}, objects=['F'], poses={'0': {}})
        with self.assertRaisesRegex(ValueError, 'construction script'):
            lane.trial('t', source=__file__)
        with self.assertRaisesRegex(ValueError, 'no completed trial'):
            lane.reopen('t')
        with self.assertRaisesRegex(ValueError, 'passed independent reopen'):
            lane.retain('t', target=root / 'x.blend', label='x', review={'judgment': 'j', 'watched': ['v']})


@unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
class NativeTrialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .init_workspace import initialize
        from .live import bind_reference_adapter, launch, wait_ready
        from .service import ModelingService
        from .test_live import REFERENCE, RESTORE, SCENE_BUILDER, free_port
        from .trials import Trials
        cls.root = Path(tempfile.mkdtemp()); ws = cls.ws = cls.root / 'character'
        initialize(ws, 'Synthetic')
        (ws / 'restore_rig.py').write_text(RESTORE); (ws / 'build.py').write_text(SCENE_BUILDER)
        (ws / 'change.py').write_text(CHANGE)
        cls.scene = ws / 'checkpoints' / 'start.blend'
        subprocess.run([BLENDER, '--background', '--factory-startup', '--python', str(ws / 'build.py'), '--',
                        str(cls.scene)], check=True, capture_output=True, timeout=180)
        cls.port = free_port()
        bind_reference_adapter(ws, REFERENCE, owner='tester', port=cls.port)
        cls.process = launch(ws, BLENDER, file=cls.scene, background=True)
        wait_ready(cls.port, timeout=120, process=cls.process)
        (ws / 'eye.json').write_text(json.dumps({'object': 'Face', 'region': list(range(8))}))
        (ws / 'narrow.json').write_text(json.dumps({'object': 'Face', 'region': [0, 1, 2, 3]}))
        poses = {'0.0': {'Blink': 0.}, '0.5': {'Blink': .5}, '1.0': {'Blink': 1.}}
        service = ModelingService(ws)
        make = lambda declaration: Trials(service, 'tester', ws / 'runtime' / 'trials', blender=BLENDER,
                                          reference=REFERENCE, objects=['Face'], poses=poses,
                                          declaration=ws / declaration, resolution=32, timeout=300)
        cls.lane, cls.narrow = make('eye.json'), make('narrow.json')

    @classmethod
    def tearDownClass(cls):
        from .live import shutdown
        try:
            shutdown(cls.port); cls.process.wait(timeout=60)
        finally:
            if cls.process.poll() is None:
                cls.process.kill()
            shutil.rmtree(cls.root, True)

    def test_trial_reopen_retain_and_a_failed_check_blocks_retention(self):
        trial = self.lane.trial('change01', source=self.scene, construction=self.ws / 'change.py')
        self.assertEqual((trial['status'], trial['checks']['status']), ('completed', 'pass'))
        out = Path(trial['output'])
        with np.load(out / 'evaluated.npz') as after, np.load(out / 'source-evaluated.npz') as before:
            self.assertAlmostEqual(float(np.abs(after['1.0::Face::co'] - before['1.0::Face::co']).max()), .1, places=6)
            self.assertEqual(float(np.abs(after['0.0::Face::co'] - before['0.0::Face::co']).max()), 0.)
        reopened = self.lane.reopen('change01')
        self.assertEqual(reopened['status'], 'passed')
        target = self.ws / 'checkpoints' / 'C01.blend'
        with self.assertRaisesRegex(ValueError, 'review'):
            self.lane.retain('change01', target=target, label='C01', review={'judgment': 'looks right'})
        kept = self.lane.retain('change01', target=target, label='C01 closes further',
                                review={'by': 'operator', 'judgment': 'one lid, closes further', 'watched': ['frames']})
        self.assertEqual(kept['status'], 'completed'); self.assertTrue(target.is_file())
        self.assertFalse(kept['user_appearance_accepted'])
        with self.assertRaisesRegex(ValueError, 'already used'):
            self.lane.trial('change01', source=target, construction=self.ws / 'change.py')
        second = self.narrow.trial('change02', source=target, construction=self.ws / 'change.py')
        self.assertEqual(second['checks']['failing'], ['still_outside'])
        self.assertEqual(self.narrow.reopen('change02')['status'], 'passed')
        with self.assertRaisesRegex(ValueError, 'Standard checks did not pass'):
            self.narrow.retain('change02', target=self.ws / 'checkpoints' / 'C02.blend', label='C02',
                               review={'judgment': 'j', 'watched': ['v']})
        verbs = [(row['verb'], row['tag'], row['status']) for row in self.lane.journal()]
        self.assertEqual(verbs[:3], [('trial', 'change01', 'completed'), ('reopen', 'change01', 'passed'),
                                     ('retain', 'change01', 'completed')])


if __name__ == '__main__':
    unittest.main()
