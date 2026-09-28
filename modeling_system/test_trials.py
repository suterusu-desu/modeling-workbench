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


    def test_a_trial_of_a_source_that_is_not_live_is_refused_before_it_uses_the_tag(self):
        from .trials import Trials
        import hashlib
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        live_file = root / 'current.blend'; live_file.write_bytes(b'live'); stale = root / 'older.blend'; stale.write_bytes(b'old')

        class Service:
            workspace = root

            def execute(self, operation, arguments):
                assert operation == 'native_inspect_live'
                return {'expected_state': 's', 'file': str(live_file), 'owner': 'o', 'dirty': False,
                        'saved_file': {'sha256': hashlib.sha256(b'live').hexdigest()}}
        lane = Trials(Service(), 'o', root / 'trials', blender='b', reference={'working': 'F'}, objects=['F'],
                      poses={'0': {}})
        for _ in range(2):                                   # refused twice: the tag was never used
            with self.assertRaisesRegex(ValueError, 'refused before it started .*the live file is .*current.blend'):
                lane.trial('attempt-1', source=stale, construction=__file__)
        self.assertFalse((root / 'trials' / 'attempt-1').exists())
        self.assertEqual([(line['tag'], line['status'], line['record']) for line in lane.journal()],
                         [('attempt-1', 'refused', None)] * 2)
        self.assertIn('current.blend', lane.journal()[0]['reason'])


    def test_retention_rechecks_the_inputs_of_a_preserved_region_verdict(self):
        """Offline: the native run is replaced by saved poses; the record binds what the checks measured."""
        from .trials import Trials
        from .test_checks import KEEP, PHASES, hinged, states
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        mask = np.zeros(len(hinged()[0]), bool); mask[KEEP] = True; np.savez(root / 'keep.npz', mask=mask)
        region = [int(v) for v in range(13, 78)]

        launched = {}

        def lane(name, preserved, during=None):
            declaration = {'object': 'Skin', 'region': region}
            if preserved:
                declaration['preserved'] = {'units': 'synthetic length', 'tolerance': 1e-6, 'phases': PHASES,
                                            'regions': [{'object': 'Skin', 'vertices': {'path': 'keep.npz', 'key': 'mask'}}]}
            (root / f'{name}.json').write_text(json.dumps(declaration))

            class Offline(Trials):
                def _check_live(self, tag, source_ref):
                    pass

                def _run(self, tag, verb, **kwargs):
                    launched[name] = [str(Path(p).resolve()) for p in kwargs['pinned']]
                    if during:
                        during()                                  # an edit while the native run is going
                    out = self.directory / tag / 'run'; out.mkdir(parents=True)
                    np.savez(out / 'evaluated.npz', **states(hinged())); np.savez(out / 'source-evaluated.npz', **states(hinged()))
                    (out / 'candidate.blend').write_bytes(b'candidate')
                    return {'status': 'completed', 'output': str(out)}

            class Service:
                workspace = root
            return Offline(Service(), 'o', root / name, blender='b', reference={'working': 'F'}, objects=['Skin'],
                           poses={'0': {}}, declaration=root / f'{name}.json')

        source = root / 'source.blend'; source.write_bytes(b'source')
        review = {'judgment': 'j', 'watched': ['v']}
        for name, preserved in (('preserved', True), ('legacy', False)):
            trials = lane(name, preserved)
            record = trials.trial('t', source=source, construction=__file__)
            self.assertEqual(record['checks']['status'], 'pass')
            if preserved:                                          # pinned before launch, rechecked at retention
                self.assertIn(str((root / 'keep.npz').resolve()), launched[name])
                self.assertEqual(set(record['checks']['inputs']),
                                 {'declaration', 'preserved:Skin[0]', 'candidate_poses', 'baseline_poses'})
            else:
                self.assertNotIn('inputs', record['checks'])
                self.assertNotIn(str((root / 'keep.npz').resolve()), launched[name])
            record['checks']['status'] = 'fail'; record['checks']['failing'] = ['synthetic']   # stop before native retention
            (root / name / 't' / 'trial.json').write_text(json.dumps(record))
            (root / name / 't' / 'reopen.json').write_text(json.dumps({'status': 'passed', 'verification': {}}))
            with self.assertRaisesRegex(ValueError, 'did not pass'):             # inputs unchanged: the usual gate
                trials.retain('t', target=root / 'x.blend', label='x', review=review)
        changed = mask.copy(); changed[KEEP[0]] = False; np.savez(root / 'keep.npz', mask=changed)
        with self.assertRaisesRegex(ValueError, 'changed or are missing since the trial'):
            lane('preserved', True).retain('t', target=root / 'x.blend', label='x', review=review,
                                           waive_checks='a waiver does not cover changed inputs')
        with self.assertRaisesRegex(ValueError, 'did not pass'):                 # legacy lanes are not tightened
            lane('legacy', False).retain('t', target=root / 'x.blend', label='x', review=review)
        # The selection edited while the trial runs: nothing is measured against it, and retention refuses.
        np.savez(root / 'keep.npz', mask=mask)
        edit = lambda: np.savez(root / 'keep.npz', mask=changed)
        during = lane('during', True, during=edit).trial('t', source=source, construction=__file__)
        self.assertEqual(during['checks']['status'], 'unknown')
        self.assertEqual(during['checks']['inputs_changed_during_trial'], [str((root / 'keep.npz').resolve())])
        (root / 'during' / 't' / 'reopen.json').write_text(json.dumps({'status': 'passed', 'verification': {}}))
        with self.assertRaisesRegex(ValueError, 'changed during the trial'):
            lane('during', True).retain('t', target=root / 'x.blend', label='x', review=review, waive_checks='no')
        # A preserved scope added to a plain declaration while the trial runs is not measured unbound.
        def add_scope():
            declaration = json.loads((root / 'added.json').read_text())
            declaration['preserved'] = {'units': 'u', 'tolerance': 0., 'phases': PHASES,
                                        'regions': [{'object': 'Skin', 'vertices': {'path': 'keep.npz', 'key': 'mask'}}]}
            (root / 'added.json').write_text(json.dumps(declaration))
        added = lane('added', False, during=add_scope).trial('t', source=source, construction=__file__)
        self.assertEqual((added['checks']['status'], added['checks']['failing']), ('unknown', ['preserved_regions']))
        (root / 'added' / 't' / 'reopen.json').write_text(json.dumps({'status': 'passed', 'verification': {}}))
        with self.assertRaisesRegex(ValueError, 'changed during the trial'):
            lane('added', False).retain('t', target=root / 'x.blend', label='x', review=review, waive_checks='no')


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
