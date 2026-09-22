"""Runner contract checks with a fake subprocess; never starts Blender."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import isolated_blender as runner


class IsolatedRunnerTests(unittest.TestCase):
    def run_job(self, *, candidate=True, change_source=False):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        for name in ('blender', 'worker.py', 'source.blend'):
            (root/name).write_bytes(b'synthetic input')
        job = root/'job.json'
        job.write_text(json.dumps({'script': 'worker.py', 'blender': 'blender',
            'input': 'source.blend', 'output_root': 'runs'}))
        seen = {}
        class Process:
            pid = 1
            def __init__(self, command, **kwargs):
                seen.update(command=command, **kwargs)
                output = kwargs['cwd']
                (output/'inventory.json').write_text('{}')
                if candidate: (output/'candidate.blend').write_bytes(b'synthetic candidate')
                if change_source: (root/'source.blend').write_bytes(b'changed')
            def wait(self, timeout): return 0
        with patch.object(runner.subprocess, 'Popen', Process), \
                patch.object(runner.sys, 'argv', ['runner', str(job)]), \
                patch.dict(os.environ, {'PYTHONPATH':'incompatible-host-packages','PYTHONHOME':'host-python'}), \
                redirect_stdout(io.StringIO()):
            result = runner.main()
        return root, seen, result

    def test_profile_and_embedded_python_are_isolated(self):
        root, seen, result = self.run_job()
        self.assertEqual(result, 0)
        self.assertNotIn('PYTHONPATH', seen['env'])
        self.assertNotIn('PYTHONHOME', seen['env'])
        for suffix in ('RESOURCES','CONFIG','SCRIPTS','EXTENSIONS','DATAFILES'):
            self.assertTrue(Path(seen['env']['BLENDER_USER_'+suffix]).is_relative_to(seen['cwd']))
        self.assertIn('--disable-autoexec', seen['command'])
        self.assertTrue(Path(seen['command'][seen['command'].index('--python')+1]).is_file())
        receipt = json.loads((root/'runs/latest.json').read_text())
        self.assertTrue(receipt['source_unchanged'])
        self.assertEqual((seen['cwd']/'input.blend').read_bytes(), b'synthetic input')

    def test_successful_process_without_saved_candidate_is_failure(self):
        root, seen, result = self.run_job(candidate=False)
        self.assertEqual(result, 1)
        self.assertEqual(json.loads((root/'runs/latest.json').read_text())['status'], 'failed')

    def test_modified_source_is_failure_even_when_process_returns_zero(self):
        root, seen, result = self.run_job(change_source=True)
        self.assertEqual(result, 1)
        self.assertFalse(json.loads((root/'runs/latest.json').read_text())['source_unchanged'])


if __name__ == '__main__':
    unittest.main()
