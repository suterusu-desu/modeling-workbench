"""The reference adapter and live bridge: binding everywhere, the full native route when a Blender is available.

Set MODELING_BLENDER to a Blender executable to run the native tests; without it they are skipped.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest

from .live import ADAPTER, bind_reference_adapter, launch, shutdown, wait_ready
from .native_bridge import NativeBridge, NativeBridgeError

BLENDER = os.environ.get('MODELING_BLENDER')
REFERENCE = {'working': 'Face', 'character': ['Face'],
             'guides': {'Neutral': {'object': 'Guide neutral', 'role': 'identity'},
                        'Closed': {'object': 'Guide closed', 'role': 'pose_check', 'controls': {'Blink': 1.0}}},
             'controls': {'object': 'Controls', 'properties': ['blink']},
             'shape_keys': {'object': 'Face', 'keys': ['Blink']},
             'restore': {'path': 'restore_rig.py', 'function': 'restore'}}
SCENE_BUILDER = '''
import bpy, sys
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
bpy.ops.mesh.primitive_cube_add(size=1.); face = bpy.context.object; face.name = 'Face'
face.shape_key_add(name='Basis'); blink = face.shape_key_add(name='Blink'); blink.value = 0.
for i, v in enumerate(face.data.vertices):
    if v.co.z > 0: blink.data[i].co.z -= .3
for name in ('Guide neutral', 'Guide closed'):
    bpy.ops.mesh.primitive_cube_add(size=1.02); bpy.context.object.name = name
controls = bpy.data.objects.new('Controls', None); bpy.context.scene.collection.objects.link(controls); controls['blink'] = 0.
bpy.ops.wm.save_as_mainfile(filepath=sys.argv[sys.argv.index('--') + 1])
'''
RESTORE = '''
def restore(bpy, workspace, config):
    bpy.context.scene['restored_by_hook'] = True
    return 'restored'
'''


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); return s.getsockname()[1]


class BindTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.root, True)
        (self.root / 'modeling-workspace.json').write_text(json.dumps({'schema_version': 1, 'character': 'Synthetic',
                                                                       'native_owner': 'tester'}))
        (self.root / 'restore_rig.py').write_text(RESTORE)

    def test_binding_pins_the_adapter_and_the_restore_hook(self):
        config = bind_reference_adapter(self.root, REFERENCE, port=9911)
        binding = json.loads((self.root / 'modeling-workspace.json').read_text())
        self.assertEqual((binding['native_adapter'], binding['native_owner'], binding['native_bridge_port']),
                         ('blender_json_v1', 'tester', 9911))
        self.assertEqual(config['entrypoint'], {'path': str(ADAPTER),
                                                'sha256': hashlib.sha256(ADAPTER.read_bytes()).hexdigest()})
        self.assertEqual(config['dependencies'][0]['path'], str((self.root / 'restore_rig.py').resolve()))
        self.assertIn('set_display', config['operations'])
        self.assertEqual(binding['native_configuration']['reference']['guides']['Closed']['role'], 'pose_check')

    def test_refusals(self):
        with self.assertRaisesRegex(ValueError, 'working object'):
            bind_reference_adapter(self.root, {'guides': {}})
        with self.assertRaisesRegex(ValueError, 'role'):
            bind_reference_adapter(self.root, dict(REFERENCE, guides={'X': {'object': 'X', 'role': 'target'}}))
        (self.root / 'modeling-workspace.json').write_text(json.dumps({'schema_version': 1}))
        with self.assertRaisesRegex(ValueError, 'native owner'):
            bind_reference_adapter(self.root, dict(REFERENCE, restore=None))


@unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
class LiveRouteTests(unittest.TestCase):
    """NativeBridge -> live_bridge in a background Blender -> blender_operations -> reference adapter."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        (cls.root / 'restore_rig.py').write_text(RESTORE)
        builder = cls.root / 'build.py'; builder.write_text(SCENE_BUILDER)
        cls.scene = cls.root / 'scene.blend'
        subprocess.run([BLENDER, '--background', '--factory-startup', '--python', str(builder), '--', str(cls.scene)],
                       check=True, capture_output=True, timeout=180)
        cls.port = free_port()
        (cls.root / 'modeling-workspace.json').write_text(json.dumps({'schema_version': 1, 'native_owner': 'tester'}))
        bind_reference_adapter(cls.root, REFERENCE, port=cls.port)
        cls.process = launch(cls.root, BLENDER, file=cls.scene, background=True)
        wait_ready(cls.port, timeout=120, process=cls.process)
        cls.bridge = NativeBridge(cls.root, port=cls.port, timeout=120)

    @classmethod
    def tearDownClass(cls):
        try:
            shutdown(cls.port)
            cls.process.wait(timeout=60)
        finally:
            if cls.process.poll() is None:
                cls.process.kill()
            shutil.rmtree(cls.root, True)

    def call(self, operation, arguments=None, owner='tester'):
        return self.bridge.call(operation, arguments or {}, owner=owner)

    def live(self):
        return self.call('inspect_live')['live']

    def test_the_owner_drives_the_live_file_through_the_reference_adapter(self):
        live = self.live()
        self.assertEqual((live['owner'], live['guides']), ('tester', {'Neutral': 'identity', 'Closed': 'pose_check'}))
        self.assertEqual(live['controls'], {'blink': 0.0, 'Blink': 0.0})
        with self.assertRaises(NativeBridgeError):
            self.call('set_controls', {'expected_state': live['expected_state'], 'controls': {'Blink': .5}}, owner='other')
        with self.assertRaises(NativeBridgeError):
            self.call('set_controls', {'expected_state': 'stale', 'controls': {'Blink': .5}})
        shown = self.call('set_controls', {'expected_state': live['expected_state'], 'guide': 'Closed'})['live']
        self.assertEqual((shown['guide'], shown['guide_role'], shown['controls']['Blink']), ('Closed', 'pose_check', 1.0))
        wire = self.call('set_display', {'expected_state': shown['expected_state'], 'mode': 'GUIDE_WIRE',
                                         'hide': ['Controls']})['live']
        self.assertEqual((wire['display'], wire['hidden']), ('GUIDE_WIRE', ['Controls']))
        self.assertEqual(wire['content_fingerprint'], shown['content_fingerprint'])   # presentation changed no geometry
        target = self.root / 'saved.blend'
        saved = self.call('save_checkpoint', {'expected_state': wire['expected_state'], 'path': str(target),
                                              'label': 'synthetic save'})
        self.assertEqual(saved['saved']['sha256'], hashlib.sha256(target.read_bytes()).hexdigest())
        opened = self.call('open_checkpoint', {'expected_state': saved['live']['expected_state'],
                                               'source': {'path': str(target), 'sha256': saved['saved']['sha256']}})
        self.assertEqual(opened['restore'], 'restored')
        self.assertTrue(opened['live']['restored'])
        self.assertEqual(opened['live']['controls']['Blink'], 1.0)

    def test_overlap_renders_through_the_isolated_runner(self):
        import numpy as np, sys
        np.savez(self.root / 'closed-guide.npz', co=np.array([[0, -.6, -.2], [.4, -.6, .2], [-.4, -.6, .2], [0, -.9, 0]]),
                 tri=np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]))
        job = {'script': str(Path(__file__).with_name('overlap_views.py')), 'blender': BLENDER, 'input': str(self.scene),
               'output_root': str(self.root / 'overlaps'), 'workspace': str(self.root), 'reference': REFERENCE,
               'steps': [{'label': 'open', 'controls': {'Blink': 0.}, 'guide': 'Neutral'},
                         {'label': 'closed', 'controls': {'Blink': 1.}, 'guide': 'closed-guide.npz'}],
               'views': [{'name': 'front', 'yaw': 0.}, {'name': 'three-quarter-left', 'yaw': 40.}],
               'ortho_scale': 2.5, 'size': 200, 'resolution': 64, 'timeout_seconds': 300}
        (self.root / 'overlap-job.json').write_text(json.dumps(job))
        done = subprocess.run([sys.executable, '-m', 'modeling_system.isolated_blender', str(self.root / 'overlap-job.json')],
                              capture_output=True, text=True, timeout=400, cwd=str(Path(__file__).resolve().parents[1]))
        receipt = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt['status'], 'completed', done.stdout[-2000:])
        out = Path(receipt['output']); rows = json.loads((out / 'overlaps.json').read_text())
        self.assertEqual([(r['step'], r['view']) for r in rows], [('open', 'front'), ('open', 'three-quarter-left'),
                                                                 ('closed', 'front'), ('closed', 'three-quarter-left')])
        self.assertTrue(all((out / r['image']).stat().st_size > 0 for r in rows))


if __name__ == '__main__':
    unittest.main()
