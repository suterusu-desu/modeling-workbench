# Evaluated interventions, delivery and recovery

These checks extend existing operations. They neither contact Blender nor change
the qualified native adapter. The sole owner still records, applies and reopens
native changes. Old records remain readable; absent evidence remains unknown.

## Before and after a correction

`propose_change(..., intervention=case)` retains numerical support alongside the
existing native proposal. The case names `state` (the experiment baseline),
`predicted_state` (an imported evaluated prediction), exact `pose` controls,
`guide_pose`, `mechanism`, `construction_evidence` record IDs, and nonnegative
`realization_tolerance`. Optional `correspondence` is `triangles` by default;
use `recorded_polygon_loops` only when native polygon identity was captured.
The pose's own triangles still govern guide hits.

`layers` covers every surface in the scoped baseline/prediction captures,
including unchanged dependents. Each row has `object`, `semantic_component`,
and `support` groups. Every group lists distinct vertex `indices`, a measured
`tolerance` in the recorded coordinate units, and its evidence-based `basis`:

- `kind: direct`: `qualification` names a current `guide_qualification` for
  this baseline and `guide_pose`; optional `selection` uses the existing geometry
  selector. Selected triangles must belong to qualified support. Predicted
  vertices are measured against that surface, not a global guide statistic.
- `kind: interpolated`: `neighbors` names directly supported vertices in this
  layer and `weights` gives normalized nonnegative weights. The check measures
  displacement against the weighted neighboring displacements. This contract
  describes this particular interpolation; do not claim another solver uses it.
- `kind: attachment`: `driver_object` names another captured layer,
  `driver_indices` its three noncollinear frame vertices, `weights` the three
  barycentric anchor weights, and `rest_offsets` the Nx3 measured local offsets
  for the dependent vertices. The frame axes are normalized edge 0->1,
  normal-cross-edge, and triangle normal. The predictor must reproduce the
  independent baseline before its candidate is checked. Changed drivers trace
  to qualified support; cycles and unsupported driver motion remain failures.

The output keeps two populations separate: actually changed vertices and all
surveyed support points. Direct, interpolated and attachment support never
become the same category. No fixed direct-support percentage is a pass criterion.
Construction evidence must show how the actual script/control used the
constraints; checking a prediction does not prove solver causality.

`mode: guide_fit` with unsupported/violating movement is blocked before native
dispatch. `mode: construction_repair` explicitly permits investigation of flawed
guides/controls, with `repair.reason`, `recovery_checkpoint: {path, sha256}`,
`revised_constraints`, and `validation_plan`. It stays labeled construction repair
and does not turn missing support into qualified support. Existing geometry and
temporary trial restrictions are not permanent design authority.

After native application, `resolve_trial(..., observed_status="applied",
realized_state=record)` adds a comparison without replaying the edit. It measures
prediction residual **and independently remeasures realized constraints**,
including unsupported movement and dependent layers. `retain_candidate` requires
this comparison when the proposal has an intervention. Inspect any disagreement
before local retention; retain honest repair/appearance limitations.

## Source review and export

Prepared reference jobs retain separate view/pose/crop, identity, motion-phase,
qualified-depth and provisional-anatomy roles. Supply disputed assumptions through
`settings.disputed_assumptions`. Use `reviewed_source` also for an image derived
from an earlier reviewed image. A review can record zero landmark drift while
rejecting identity. `inspect_workflow` reports transitive source validity;
completed jobs and their outputs remain recoverable when ancestors become stale.
No invalidation submits another job. Legacy jobs without declared ancestry cannot
retroactively gain independent provenance.

Add `delivery` to an existing `inspect_execution_receipt` case. It has
`source`, `current_source`, and `exports` references `{file, sha256}`;
`native_mechanism` (a pinned `{path, sha256}` native mechanism/control capture),
`scope`, `deferred`, `objects`, `materials`, `units`,
nonnegative `tolerance`, and `samples`. Each sample has `object`, `controls`,
`kind` (`anchor` or `between_anchor`), exact `source_sha256`/`export_sha256`, and
`comparison: {path, sha256}` pointing to an NPZ with corresponding finite Nx3
`source` and `output` arrays from the actual representations. Preserve independent
capture/correspondence receipts in the case evidence. A newer file name or date
does not replace source identity. Changed bytes make delivery stale. Between-anchor
coverage and residuals are separate from execution success; expand `reads.delivery`.
Use the largest measured errors to choose further inspection, not indiscriminate
oversampling of a discontinuous source. Finite samples never prove all motion.

## Recover without guessing the base

`snapshot_workspace(episode, recovery=manifest, expected_recovery=previous_id)`
selects a small immutable manifest. Omit `expected_recovery` only for the first
selection. It links `retained` and `baseline` checkpoints `{file, sha256}`,
`active_experiment` (or null), `native_owner`, `preview`, and `selected_runtime`
with actual `python` and `loaded_revision` from the selected process.
`preview.status` is `retained`, `trial`, `rejected` or `unknown`. A trial/rejected
preview needs a hashed `visibility_restore` receipt `{file, sha256}` recording
the previous visibility for the sole owner to restore. Saving that instruction
does not execute it or verify live Blender.

`inspect_situation()['recovery']` checks checkpoint hashes and distinguishes this
process, the selected runtime and the configured adapter. It never selects by
filename, timestamp, preview appearance or message dispatch. Peer delivery and
operator uptake require their actual replies/operation receipts. Use existing
`operation_context` for current relevant rules and exact lazy expansion.

## Local media recovery

Workspace `media_root` or `MODELING_MEDIA_ROOT` can select a short cache path;
otherwise the system temporary directory is used. Video inputs and extracted
frames are content addressed, atomically written and retained in the evidence
store. A partial decode keeps a progress manifest and completed frames. Correct
a path/landmark error against the same completed provider job; do not generate
again. `landmarks` needs paired nonempty finite Nx2 `source`/`output` pixels.
