# Reusable modeling operations

For array contracts, complete projected triangle contact, bounded reports and
measured scoped native setup, see [preparation and bootstrap](preparation-and-bootstrap.md).

Register a mechanism once, then bind character data and qualified alternatives.
Use `ParameterizedCatalog` for scope-specific parameters without copying Python
factories. Its implementation functions receive `(state, previous, parameters)`.
An implementation can also be `{build, candidates, ...}` with stage functions of
that signature; `catalog.followup("candidates")` keeps the selected parameters and
actual predecessor result through CandidatePipeline without another choice.
Bindings supply `id`, `implementation`, a meaningful `description`, `parameters`
and optional exact `conditions`. Missing support excludes a binding; identical
mechanism/parameter aliases are refused. No Cartesian strength sweep is generated.

`bind_array_method` is a ready-made implementation for `ArrayPreparation`:

```python
from modeling_system.method_catalog import ParameterizedCatalog, bind_array_method
from modeling_system.preparation import ArrayPreparation

methods = ParameterizedCatalog({"arrays": bind_array_method}, qualified_bindings)
# Each binding's parameters:
# {task, revision, question, method, operation, inputs, reads, writes,
#  optional parameters, lane, handler}
# inputs[name] = {path, sha256, array}; every hash is a current task read.
# Use catalog=methods (or a CandidatePipeline stage),
# handlers={"arrays": ArrayPreparation(run / "prepared")}, in OperatingSession.
```

Offer alternatives that answer different live questions, such as explicit
attachment-pair motion versus a guide-material fit where both are qualified.
Use applicability conditions to exclude missing correspondence. A single qualified native operation runs directly under its owner. `MethodCatalog.followup(stage)` marks the declared continuation of the actual chosen method; multiple alternatives require explicit owner choice.

## Material fields and verification

The following functions in `material_operations` also run as array recipes:

| Operation | Required meaning | Result |
| --- | --- | --- |
| `fit_landmark_field` | Actual corresponding source/guide anchors and source material; a qualified two-axis chart for the residual option | Full 3D affine transport or affine plus exact thin-plate landmark residual; explicit interpolation limits |
| `material_trajectory` | Complete evaluated guide-bound keyframe positions, increasing knots, supported poses, optional actual baseline and continuous support | One shared piecewise-linear material law preserving the unselected contribution |
| `surface_realization` | Corresponding expected/actual source vertices, actual source faces and declared tolerance | Face-owning surface verdict; unused points reported separately |
| `attachment_motion` | Explicit paired material indices and matching increasing pose samples | Gap vectors and relative motion; no guessed nearest-point or causal claim |

Do not derive full positions by adding a rest array to an additive correction:
its upstream carrier may move. Use complete actual evaluated keyframe positions.
Use actual source faces for realization; a historical vertex count or an unused
legacy point is not a visible-surface requirement. Preservation of unused data,
when required, remains a separate explicit check.

`install_material_trajectory` is a native helper called only inside a selected
controller capability. Supply exact object, control data path, modifier, position
node label, fresh attribute prefix, complete world-space keyframes, knots and
continuous support. It keeps the existing SetPosition owner and creates no
modifier. Default `composition="composed_zero_offset"` requires a linked Position
and zero/unlinked Offset. Explicit `composition="compose_position_offset"`
composes the existing Position (including implicit input position) and Offset
once, then blends the supported material into that result and clears Offset.
Existing attribute names refuse replay. Guide provenance, live bindings,
preservation, native realization, independent reopen and appearance still belong
to the ordinary candidate contract.

## Native jobs and retention

`NativeJob(service, run)` replaces candidate-specific process wrappers. A native
task payload supplies `runner` and `live_source` file references plus the existing
qualified isolated-runner `job` dictionary. Job input, runner, script and every
`dependency_hashes` entry must be bound task reads. The handler checks clean live
owner/source, saves the job/dispatch, executes the isolated runner and preserves
its result. Failure or missing terminal evidence needs reconciliation, never a
sibling retry. Cancellation lets the bounded runner finish its child safely;
the controller does not start another task.

`RetainCheckpoint(service, run, verify_reopen=...)` replaces copied retention
functions. Its retention task must have a current scoped visual review. Payload:

```python
{
    "source": source_ref, "candidate": candidate_ref, "reopen": reopen_ref,
    "target": new_visible_path, "label": retained_description,
    "pose": {"controls": qualified_controls, "guide": qualified_guide, "refresh": False},
    "display": qualified_display_arguments,
}
```

The workspace's `verify_reopen(record, candidate_ref)` must validate the actual
independent reopen result against this exact candidate and source hash. Never
replace it with a filename or an unconditional true result. Keep the verifier's
implementation bound as a task dependency. Native source/candidate/reopen hashes
are task reads; target must be new.

Retention performs one initial live inspection, then consumes each operation's
returned fresh state. Compact replies expand from the immutable local operation
record, without another Blender call. Every mutation still passes the native
expected-state guard, which detects intervening user edits. An already loaded,
clean, byte-matching candidate skips the redundant open; independent reopen
evidence is still mandatory. New candidates still open normally. The final save
must return the correct clean path and file hash. Every step and its duration is
retained, including partial failure, without automatic replay.

### Fixed retention transaction

Use `RetainCheckpoint(..., transaction=True)` only when the pinned private adapter
advertises `retain_checkpoint` and implements its synchronous main-thread
contract. It performs one preflight inspection and one native transaction, with
the same bound inputs, independent-reopen verifier and actual visual review.
Supply `display.mode="GUIDE_WIRE"`; pose refresh must be false. Diagnostic modes,
arbitrary scripts and geometry changes are outside this operation. There is no
silent fallback to another route or second native writer.

The reusable `native_retention.run_retention` orchestrator invokes adapter hooks
for fresh owner/expected-state verification, one rollback plus registry backup,
candidate open, runtime-only restoration with full content comparison, declared
controls/display, and a new clean save. It verifies source, candidate, reopen,
registry and implementation dependencies again before save and final completion.
The final observation must contain the real saved path/hash and clean state;
stale numerical arrays never become current merely because display succeeded.
Adapters must keep the entire operation synchronous without modal operators,
asynchronous jobs or yielding to another writer. If a route can yield, it needs
fresh guards at those boundaries and does not qualify for this shortcut.

Phase intent is persisted before effects. The transaction identity comes from
the controller attempt and handler location; an existing identity refuses even
after success. `native_retention.inspect_receipt(root, owner, transaction_id)`
reads historical evidence without observing Blender or replaying work. Reconcile
an interrupted or lost response using that receipt, the exact files and a fresh
observation through the existing controller. Never automatically restore an old
rollback over later user work. Target absence checks protect the cooperative
single-writer workflow; they are not an OS filesystem lock against other programs.

Adapter hooks are trusted, qualified code. `restore_runtime` returns
`character_content_unchanged=True` only after comparing real native content;
`verify_presentation` returns `declared_pose_guide_display_verified=True` only
after inspecting actual controls, guide and display. Their phase timings and
nested adapter timings support diagnosis, not a prediction of seconds saved.

Retention review compares the overall affected region and motion with both the
applicable guide and an earlier useful baseline. Technical exactness, endpoint
preservation and immediate-predecessor gains do not establish overall progress.
Keep rejected experiments for evidence without ratcheting the working baseline.
Preservation constraints must distinguish active output dependencies from legacy
stored attributes; confirm the actual native outputs before relaxing a hold.

## Effort and overhead

Call `review = session.begin_review(task)` before opening its actual evidence.
It returns the retained result and exact review basis, with a restart-safe timer.
Submit the usual `record_review(..., expected_basis=review["basis"], ...)` after
looking; the timer closes automatically. Missing review timers are counted as
unmeasured work. Time spent preparing a new mechanism or interpreting a failure
uses the existing explicit intervention timer.

Metrics include parameterized implementation use, tasks without recipe identity,
native stage durations/call counts, skipped redundant opens and unmeasured review
submissions. Elapsed intervals include interruptions; they are not token usage.
Do not infer Astra savings from shorter selection latency or incomplete work accounting.
