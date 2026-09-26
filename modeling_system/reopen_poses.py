"""NativeJob worker for `trials.Trials.reopen`: re-evaluate a saved candidate in a fresh Blender and compare.

Runs inside the isolated runner's Blender on a copy of the candidate file (JOB, OUT_DIR). It runs the restore hook,
evaluates the same objects at the same poses as the trial and compares them with the trial's `evaluated.npz`
(`JOB['candidate_evaluated']`). Writes `reopen-evaluated.npz` and `verification.json`: status passed when every
position agrees within `JOB['tolerance']` (default 1e-6), the number of comparisons, the largest difference and the
candidate's hash (`JOB['candidate_sha256']`).
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location('_modeling_workbench_native_poses',
                                               Path(JOB['script']).with_name('native_poses.py'))
poses = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(poses)
reference = JOB['reference']
poses.restore(reference, Path(JOB.get('workspace') or OUT_DIR))
again = poses.evaluate(JOB['objects'], JOB['poses'], reference)
poses.save(OUT_DIR / 'reopen-evaluated.npz', again)
with np.load(JOB['candidate_evaluated']) as saved:
    keys = [k for k in again if k.endswith('::co')]
    missing = [k for k in keys if k not in saved.files]
    shape = [k for k in keys if k in saved.files and saved[k].shape != again[k].shape]
    differences = {k: float(np.abs(saved[k] - again[k]).max()) for k in keys if k in saved.files and k not in shape}
largest = max(differences.values(), default=float('inf'))
tolerance = float(JOB.get('tolerance', 1e-6))
passed = not missing and not shape and bool(differences) and largest <= tolerance
(OUT_DIR / 'verification.json').write_text(json.dumps({
    'status': 'passed' if passed else 'failed', 'comparisons': len(differences), 'maximum': largest,
    'tolerance': tolerance, 'missing': missing, 'shape_changed': shape,
    'worst': max(differences, key=differences.get) if differences else None,
    'candidate': {'path': JOB['input'], 'sha256': JOB['candidate_sha256']}}, indent=1), encoding='utf-8')
first = next(iter(JOB['poses']))
poses.pose(reference, JOB['poses'][first])
poses.ensure_camera(reference)
