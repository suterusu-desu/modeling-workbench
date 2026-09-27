"""Helpers building options side by side: parallel isolated Blender jobs from one source, and parallel writers of the
store and a journal. The Blender part runs when MODELING_BLENDER names a Blender executable."""
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from .store import Store, append_line

BLENDER = os.environ.get('MODELING_BLENDER')
OPTION = '''
import json, bpy
name = JOB['option']
bpy.ops.mesh.primitive_cube_add(size=.1, location=(JOB['offset'], 0, 0)); bpy.context.active_object.name = name
(OUT_DIR / 'option.json').write_text(json.dumps({'option': name, 'objects': sorted(o.name for o in bpy.data.objects)}))
'''


def _writer(args):
    """One helper process: store the same evidence and record, and append its journal lines."""
    root, evidence, journal, worker = args
    store = Store(root); keys = set()
    for k in range(30):
        store.blob(evidence)
        keys.add(store.put('option_review', {'shared': True}))
        append_line(journal, json.dumps({'worker': worker, 'line': k, 'pad': 'x' * 200}))
    return sorted(keys)


class ParallelWriterTests(unittest.TestCase):
    def test_parallel_helpers_share_the_store_and_a_journal_without_loss(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        evidence = root / 'capture.png'; evidence.write_bytes(os.urandom(200000))
        journal = root / 'lane' / 'journal.jsonl'
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            keys = pool.map(_writer, [(str(root / 'store'), str(evidence), str(journal), w) for w in range(4)])
        self.assertEqual(len({tuple(k) for k in keys}), 1)                          # one record, the same key
        lines = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(lines), 4 * 30)                                          # every line whole, none lost
        self.assertEqual(sorted((l['worker'], l['line']) for l in lines), [(w, k) for w in range(4) for k in range(30)])
        stored = list((root / 'store' / 'assets').rglob('*.png'))
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].read_bytes(), evidence.read_bytes())
        self.assertFalse(list(journal.parent.glob('*.lock')))


@unittest.skipUnless(BLENDER and Path(BLENDER).is_file(), 'set MODELING_BLENDER to a Blender executable')
class ParallelBlenderTests(unittest.TestCase):
    def test_option_builds_run_side_by_side_from_one_source(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, root, True)
        source = root / 'source.blend'
        subprocess.run([BLENDER, '--background', '--factory-startup', '--python-expr',
                        f"import bpy; bpy.ops.wm.save_as_mainfile(filepath=r'{source.as_posix()}')"],
                       check=True, capture_output=True, timeout=300)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        script = root / 'option.py'; script.write_text(OPTION, encoding='utf-8')
        processes = []
        for k, name in enumerate(('A', 'B', 'C')):                            # one output root for all three
            job = {'script': str(script), 'blender': BLENDER, 'input': str(source), 'output_root': str(root / 'runs'),
                   'option': name, 'offset': .2 * k, 'resolution': 64, 'clay': False, 'timeout_seconds': 300}
            path = root / f'job-{name}.json'; path.write_text(json.dumps(job), encoding='utf-8')
            processes.append(subprocess.Popen([sys.executable, '-m', 'modeling_system.isolated_blender', str(path)],
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                              cwd=str(Path(__file__).resolve().parents[1])))
        receipts = [json.loads(p.communicate(timeout=400)[0].strip().splitlines()[-1]) for p in processes]
        self.assertEqual([r['status'] for r in receipts], ['completed'] * 3, receipts)
        outputs = [Path(r['output']) for r in receipts]
        self.assertEqual(len(set(outputs)), 3)                                    # three run folders
        for name, out in zip(('A', 'B', 'C'), outputs):
            option = json.loads((out / 'option.json').read_text())
            self.assertEqual(option['option'], name)
            self.assertIn(name, option['objects'])
            self.assertFalse({'A', 'B', 'C'} - {name} & set(option['objects']))    # no other helper's work
            self.assertTrue((out / 'candidate.blend').is_file() and (out / 'profile').is_dir())
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)  # the source untouched
        self.assertTrue(all(r.get('source_unchanged', True) for r in receipts))


if __name__ == '__main__':
    unittest.main()
