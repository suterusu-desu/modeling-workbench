"""Native polygon capture and independent reopen on a synthetic deforming quad, through the isolated runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .checks import run_checks

HERE = Path(__file__).resolve().parent
BLENDER = os.environ.get('MODELING_BLENDER')
CAPTURE = '''
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('synthetic_pose_capture', JOB['poses_helper'])
capture = importlib.util.module_from_spec(spec); spec.loader.exec_module(capture)
mesh = bpy.data.meshes.new('Panel')
mesh.from_pydata([(0,0,0), (2,0,0), (.5,0,.5), (0,0,2)], [], [(0,1,2,3)])
mesh.update()
ob = bpy.data.objects.new('Panel', mesh); bpy.context.scene.collection.objects.link(ob)
ob.shape_key_add(name='Basis'); bend = ob.shape_key_add(name='Bend'); leak = ob.shape_key_add(name='Leak')
bend.data[1].co = (.5,0,.75); bend.data[2].co = (2,0,2)
leak.data[1].co.y += .003; leak.data[1].co.z += .004
reference = JOB['reference']
baseline_poses = {g: dict(v, Leak=0) for g, v in JOB['poses'].items()}
capture.save(OUT_DIR / 'source-evaluated.npz', capture.evaluate(['Panel'], baseline_poses, reference))
capture.save(OUT_DIR / 'evaluated.npz', capture.evaluate(['Panel'], JOB['poses'], reference))
# Preserve the existing direct evaluated(name) two-tuple API.
assert len(capture.evaluated('Panel')) == 2
capture.pose(reference, JOB['poses']['0.0'])
capture.ensure_camera(reference)
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
'''


def run_fixture(root, blender):
    """Run only synthetic factory scenes. A caller may retain root as the native verification receipt."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    script = root / 'capture.py'; script.write_text(CAPTURE, encoding='utf-8')
    common = {'blender': str(blender), 'threads': 2, 'resolution': 24, 'clay': False, 'timeout_seconds': 120,
              'reference': {'working': 'Panel', 'shape_keys': {'object': 'Panel', 'keys': ['Bend', 'Leak']}},
              'objects': ['Panel'],
              'poses': {'0.0': {'Bend': 0, 'Leak': 0}, '0.5': {'Bend': .5, 'Leak': 1}, '1.0': {'Bend': 1, 'Leak': 0}}}

    def run(name, **extra):
        job = dict(common, output_root=str(root / name), **extra)
        path = root / f'{name}.json'; path.write_text(json.dumps(job), encoding='utf-8')
        result = subprocess.run([sys.executable, '-B', '-m', 'modeling_system.isolated_blender', str(path)],
                                capture_output=True, text=True, timeout=150)
        if result.returncode:
            raise AssertionError(f'Isolated fixture failed: {result.stdout}\n{result.stderr}')
        receipt = json.loads((root / name / 'latest.json').read_text())
        if receipt['status'] != 'completed':
            raise AssertionError(receipt)
        return receipt

    first = run('capture', script=str(script), poses_helper=str(HERE / 'native_poses.py'))
    output = Path(first['output'])
    from .isolated_blender import sha256
    reopen = {'script': str(HERE / 'reopen_poses.py'), 'input': str(output / 'candidate.blend'),
              'candidate_sha256': sha256(output / 'candidate.blend'), 'tolerance': 1e-6}
    second = run('reopen', **reopen, candidate_evaluated=str(output / 'evaluated.npz'))
    with np.load(output / 'evaluated.npz') as z:
        stale = {k: z[k] for k in z.files}
    stale['0.5::Panel::loops'] = stale['0.5::Panel::loops'][::-1]
    stale_path = root / 'stale-witness.npz'; np.savez(stale_path, **stale)
    third = run('reopen-stale', **reopen, candidate_evaluated=str(stale_path))
    return first, second, third


class ReopenWitnessTests(unittest.TestCase):
    """Run the comparison worker with a stub evaluator; legacy and incomplete-capture handling need no Blender."""

    def test_legacy_missing_and_malformed_witness(self):
        again = {'0.0::Panel::co': np.zeros((3, 3)), '0.0::Panel::tri': np.array([[0, 1, 2]]),
                 '0.0::Panel::loops': np.array([0, 1, 2]), '0.0::Panel::polygon_starts': np.array([0]),
                 '0.0::Panel::polygon_lengths': np.array([3]), '0.0::Panel::triangle_polygon': np.array([0])}
        legacy = {k: v for k, v in again.items() if k.endswith(('::co', '::tri'))}
        incomplete = {k: v for k, v in again.items() if not k.endswith('::polygon_lengths')}
        malformed = dict(again); malformed['0.0::Panel::loops'] = again['0.0::Panel::loops'].astype(float)
        with tempfile.TemporaryDirectory(prefix='reopen-witness-') as folder:
            root = Path(folder); source = root / 'saved.npz'
            fake = SimpleNamespace(restore=lambda *a: None, evaluate=lambda *a: again,
                                   save=lambda path, arrays: np.savez(path, **arrays),
                                   pose=lambda *a: None, ensure_camera=lambda *a: None)
            spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda *a: None))
            job = {'script': str(HERE / 'reopen_poses.py'), 'reference': {}, 'objects': ['Panel'],
                   'poses': {'0.0': {}}, 'candidate_evaluated': str(source),
                   'candidate_sha256': 'synthetic-hash', 'input': 'synthetic.blend'}
            for saved, expected in ((again, 'passed'), (legacy, 'passed'), (incomplete, 'failed'), (malformed, 'failed')):
                with self.subTest(expected=expected, fields=list(saved)):
                    np.savez(source, **saved)
                    with patch('importlib.util.spec_from_file_location', return_value=spec), \
                            patch('importlib.util.module_from_spec', return_value=fake):
                        exec(compile((HERE / 'reopen_poses.py').read_text(), 'reopen_poses.py', 'exec'),
                             {'JOB': job, 'OUT_DIR': root})
                    report = json.loads((root / 'verification.json').read_text())
                    self.assertEqual(report['status'], expected)
                    if saved is legacy:
                        self.assertEqual(report['polygon_witness'], 'not recorded in source; positions only')
                    if saved is incomplete:
                        self.assertIn('0.0::Panel::polygon_lengths', report['topology_changed'])


@unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
class NativePolygonCaptureTests(unittest.TestCase):
    def test_changing_diagonal_preserves_positions_and_witness_reopens(self):
        with tempfile.TemporaryDirectory(prefix='polygon-capture-') as root:
            first, second, third = run_fixture(root, BLENDER)
            assert_fixture(first, second, third)


def assert_fixture(first, second, third):
    output = Path(first['output'])
    declaration = {'object': 'Panel', 'region': [0, 1, 2, 3], 'preserved': {
        'units': 'synthetic length', 'tolerance': 1e-6, 'phases': [0., .5, 1.],
        'regions': [{'object': 'Panel', 'vertices': [1]}]}}
    baseline = output / 'source-evaluated.npz'; candidate = output / 'evaluated.npz'
    with np.load(baseline) as z:
        assert not np.array_equal(z['0.0::Panel::tri'], z['1.0::Panel::tri']), 'fixture did not retessellate'
        for g in ('0.5', '1.0'):
            for key in ('loops', 'polygon_starts', 'polygon_lengths'):
                np.testing.assert_array_equal(z[f'0.0::Panel::{key}'], z[f'{g}::Panel::{key}'])
    rows = [next(c for c in run_checks(declaration, path, baseline)['checks'] if c['id'] == 'preserved_regions')
            for path in (baseline, candidate)]
    assert (rows[0]['status'], rows[0]['observed']) == ('pass', 0.), rows[0]
    assert rows[1]['status'] == 'fail' and abs(rows[1]['observed'] - .005) < 1e-6, rows[1]
    assert rows[1]['detail']['worst']['worst_phase'] == .5
    good = json.loads((Path(second['output']) / 'verification.json').read_text())
    bad = json.loads((Path(third['output']) / 'verification.json').read_text())
    assert good['status'] == 'passed' and good['polygon_witness'] == 'compared per pose', good
    assert bad['status'] == 'failed' and '0.5::Panel::loops' in bad['topology_changed'], bad
    return {'preservation': rows, 'reopen': good, 'stale_witness_reopen': bad}


if __name__ == '__main__':
    unittest.main()
