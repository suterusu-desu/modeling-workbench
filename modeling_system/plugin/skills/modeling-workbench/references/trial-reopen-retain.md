# Trial, reopen, retain

Three plain calls change the native file. They keep what protects the work (the live file must be the clean source
under its owner, pinned inputs, isolated runs with receipts, an independent reopen, retention through the live owner)
and need no task records, catalogs or per-trial adapters.

```python
from modeling_system.service import ModelingService
from modeling_system.trials import Trials

lane = Trials(ModelingService(workspace), 'my-owner', workspace / 'runtime' / 'trials', blender=BLENDER,
              reference=reference,                       # the reference adapter's configuration (controls, restore)
              objects=['Face', 'Upper lash'],            # the meshes whose poses are saved and compared
              poses={'0.0': {'blink': 0.}, '0.5': {'blink': .5}, '1.0': {'blink': 1.}},
              declaration=workspace / 'eye-declaration.json')

lane.trial('lift03', source=workspace / 'checkpoints' / 'working.blend', construction=workspace / 'build_lift.py')
lane.reopen('lift03')
lane.retain('lift03', target=workspace / 'checkpoints' / 'lifted-lower-lid.blend', label='lifted lower lid',
            review={'by': 'owner', 'judgment': 'reads as one lid', 'watched': ['videos/lift03.mp4']})
```

**trial** runs an isolated copy of `source`, which must be the file open and clean in the owner's live Blender (so
later user work cannot be overwritten). A source that is not live is refused before the tag is used: the journal
records the refusal and what is live, and the same tag can be run once the right file is open. With `construction`, the packaged worker saves the source's poses, runs the
construction script (it installs the change, with the runner's `JOB`, `OUT_DIR` and `bpy`), saves the candidate's poses
and leaves the candidate at rest; with `script`, a workspace's own trial script runs instead and must write
`evaluated.npz` (and `source-evaluated.npz`) itself. Extra job keys go in `job`, extra pinned files in
`dependencies`. The declaration's [standard checks](construction-checks.md) measure the result; the record carries the
report and the failing check ids.

**reopen** opens the saved candidate in a fresh isolated Blender, runs the restore hook, evaluates the same objects at
the same poses and compares them with the trial's arrays (`tolerance`, default 1e-6). A workspace can pass its own
reopen `script`.

**retain** saves the candidate as a new checkpoint through the live owner (opened, posed and displayed as given, saved
to a new file that must not exist). It refuses unless the trial completed, the reopen passed, the standard checks
passed (or `waive_checks` records why a failing check is a design decision) and `review` names the judgment and the
motion watched. `review['by'] == 'owner'` records the owner's acceptance; anything else is an operator retention.

Each tag is used once. A failed or uncertain step is inspected from its receipts (`runtime/trials/<tag>/`), never
replayed; `journal.jsonl` lists every call with its status and record. `Trials.record(verb, tag)` reads a record back.

The earlier [operating sessions](operating-session.md) with `PreservationPolicy` remain available for workspaces
that use them; `checks.standard_policy` gives them the same standard checks without a hand-written adapter.
