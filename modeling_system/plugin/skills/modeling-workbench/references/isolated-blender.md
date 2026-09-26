# Isolated Blender jobs

The package includes `isolated_blender.py` and its standalone `blender_worker.py`.
Qualified owner-selected native capabilities may use this runner for isolated
construction/reopen jobs. Installing it grants no modeling authority. Jobs do
not drive the live Blender; source/guide/preservation qualification and the
single native owner remain the caller's responsibility.

The selected handler can invoke the workbench interpreter with
`-m modeling_system.isolated_blender JOB.json`, or supply the installed
`isolated_blender.__file__` as a runner path to an existing qualified native job.
No separate local skill installation is required for this runner.

Required job fields are `blender` (actual executable), `script` (qualified Blender
Python script) and `output_root`. Optional `input` selects a `.blend` file;
`scene` selects its working scene. File paths resolve relative to the job file.
The script receives `JOB`, `OUT_DIR` and `bpy`. Outputs belong under `OUT_DIR`.
Do not access the live Blender bridge or overwrite source assets from a worker.

The runner copies input to a new run directory and isolates all `BLENDER_USER_*`
profile directories. It starts factory background Blender with autoexec disabled
and removes host `PYTHONPATH`/`PYTHONHOME`. No offline site-packages are added to
Blender. Optional `threads` defaults to 2; `timeout_seconds` defaults to 240.

`cameras` maps view labels to existing camera names; otherwise the scene's camera
is used. `resolution` defaults to 640 and `clay` to true. The bootstrap executes
the supplied script, checks declared `protected_objects`, renders matched views,
writes `inventory.json` and saves `candidate.blend`. Protection here covers base
mesh coordinates, connectivity and object transforms only. It does not establish
guide fit, evaluated animation preservation, materials or artistic acceptance.

The run retains `job.json`, an isolated profile, copied input, log, candidate,
images, inventory and `receipt.json`. Nonzero exit, timeout, changed source,
missing inventory or missing candidate means failure. A candidate file by itself
does not establish completion. The caller reconciles the original receipt before
retrying. No result is automatically promoted into the live scene.

This is trusted local code, not a sandbox. Package tests verify launch/profile
and output contracts using a fake subprocess; actual Blender/version/scene
qualification belongs to a selected native operation before production use.
