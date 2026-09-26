"""NativeJob worker for `trials.Trials.trial`: poses before, the construction, poses after.

Runs inside the isolated runner's Blender (JOB, OUT_DIR). It runs the workspace's restore hook, saves the source's poses
(`source-evaluated.npz`), executes the pinned construction script (`JOB['construction']`, with the same JOB, OUT_DIR
and bpy) that installs the change, then saves the candidate's poses (`evaluated.npz`). The runner saves the candidate
file and its inventory afterwards. JOB keys: `construction`, `reference` (the workspace's reference configuration),
`workspace`, `poses` ({phase: {control: value}}, the rest phase 0 first), `objects` (the meshes to save).
"""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location('_modeling_workbench_native_poses',
                                               Path(JOB['script']).with_name('native_poses.py'))
poses = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(poses)
workspace = Path(JOB.get('workspace') or OUT_DIR)
reference = JOB['reference']
poses.restore(reference, workspace)
poses.save(OUT_DIR / 'source-evaluated.npz', poses.evaluate(JOB['objects'], JOB['poses'], reference))
construction = Path(JOB['construction']).resolve(strict=True)
exec(compile(construction.read_text(encoding='utf-8-sig'), str(construction), 'exec'), globals())
poses.save(OUT_DIR / 'evaluated.npz', poses.evaluate(JOB['objects'], JOB['poses'], reference))
first = next(iter(JOB['poses']))
poses.pose(reference, JOB['poses'][first])                      # leave the candidate at its rest pose
poses.ensure_camera(reference)
