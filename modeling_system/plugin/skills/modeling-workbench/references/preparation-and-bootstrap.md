# Preparation and scoped native setup

These are capabilities inside the existing OperatingSession. The owner selects
useful operations; the same episode retains results, failures and actual reviews.
No helper grants target authority, native ownership or appearance acceptance.

## Pure-array preparation

`ArrayPreparation` accepts the built-in recipe names, including
`projected_triangle_contact` and the [planar FEM metric](metric-fitting.md).
A private workspace can register additional pure
math under unique names without writing another handler:

```python
from modeling_system.preparation import ArrayPreparation
from modeling_system.preparation_contracts import PreparationOperation

handler = ArrayPreparation(run / "prepared", operations={
    "qualified_fit": PreparationOperation(
        fit_supported_material,
        arrays={"frame": {"shape": [3, 3]}, "source": {"shape": [None, 3]}},
        requires=("scipy",),
    )
})
```

The registered function receives declared arrays/parameters and returns arrays
and JSON-compatible data. It must have no native, provider or other external
effects. Bind its implementation and every actual source hash as task reads.
There is no provider-selected code import. Names and declared dependencies are
checked before archives load; contracts check numerical types, shape, nonempty
support and finite values. A missing module or malformed frame yields a retained,
settled preparation failure, with no unknown Blender effect.

Full arrays and metadata stay in private immutable output files. NumPy scalars
and nested arrays serialize as numerical JSON; nonfinite JSON needs an explicit
missing-data representation. Object arrays and pickle-dependent data are refused.
Compact reports omit whole optional metrics with an explicit omitted count,
never sentence fragments or limits. The full evidence remains linked.

Use `public_metrics` for an explicit, deliberately public mapping of useful
operation-specific measurements. Unlisted raw result fields stay private;
adding a `metrics` object alone does not publish it. Keys cannot override existing
built-in metrics. Nonfinite or malformed public values settle as a preparation
failure before successful output is saved. Keep source paths, character details
and unreviewed text outside this projection.

For fitting results, record what the objective regularizes: displacement,
final positions, or another declared quantity. If `final = baseline + delta`,
small variation in `delta` does not remove variation already in `baseline`.
A smooth displacement can preserve a source crease, including between direct
guide samples. When that is the observed failure, a guide-constrained objective
on the resulting surface is a different mechanism, not another smoothing
strength. Preserve the guide's intended shape, qualified motion directions and
boundary conditions; an objective change alone does not establish improvement.

Keep direct guide samples and interpolated points distinct, with their parent
identities and support limits. Record how the fitted quantity becomes the actual
native surface through inherited motion, support blending and attachments.
Coverage counts and zero opposed normals do not certify smooth shape: localize
the visible residual using saved geometry and matched views, then assess the
complete resulting patch. Return feasibility measurements and an appearance
judgment as separate evidence.

OperatingSession also converts numerical handler returns before journaling. If
a valid report is too long, it retains the known execution disposition and the
complete report, marks its checks unavailable for dependent use, and requests
compact report repair. Inspect the retained limitations, then use
`session.repair_report(task, report)`. This records the correction without invoking
the capability again. Malformed evidence and genuinely uncertain native effects
still require their ordinary reconciliation.

## Complete projected contact

Overlap vertices carry nonnegative barycentric weights for both surfaces through
clipping and bound crossings. Projection tolerance never expands triangle
half-spaces; cross products have area units. Ill-conditioned pairs use exact
rational clipping of the supplied floating-point coordinates. Projected witness
positions must still agree to arithmetic precision. This prevents thin fragments
from extrapolating depth outside their triangles. It does not qualify fragments
created by an upstream visibility or correspondence operation.

`triangle_contact.projected_triangle_contact` takes finite corresponding
`surface`/`baseline` positions, actual `triangles`, `obstacle` positions and
`obstacle_triangles`, an orthonormal three-column `frame`, and explicit `margin`
and `tolerance`. Position arrays have three columns; triangle arrays contain
actual integer indices. Frame columns define two projection axes and increasing
clearance depth. Use the actual pose's evaluated triangles.

Every positive-area triangle-overlap polygon is tested at all of its vertices.
For `bound_mode="preserve_baseline"`, also test where baseline depth crosses the
obstacle-plus-margin branch: a worst violation can occur there while all original
polygon vertices pass. The required height is the lesser of the two branches.
`bound_mode="clearance"` instead requires obstacle depth plus margin everywhere
in the overlap. Existing penetration and absolute clearance are different goals;
choose the policy from the actual modeling question.

Declare required surface separation, an optional numerical buffer and solver
residual tolerance separately. Record actual minimum separation alongside slack
against the buffered inequality: a missed buffer can coexist with positive
separation. Qualification follows the declared geometric requirement; a required
guide or anatomical clearance remains required. Preserve proposed arrays,
optimizer status and measured residuals even when a numerical target is missed.
Computation completion and target qualification are separate outcomes. Review the
retained result before deciding whether another solve would change the next edit.

Optional `active_indices` scope the incident source triangles. With
`build_constraints=True`, supply a unit `displacement_direction` parallel to the
depth axis and finite `displacement_scale`. The projected baseline/candidate
coordinates must agree. Returned COO arrays (`constraint_rows`,
`constraint_columns`, `constraint_values`, `constraint_shape`) and `lower_bound`
define `A @ displacement >= lower_bound`. Columns follow returned `active_indices`;
the displacement is relative to the supplied baseline, not the current candidate.
Inactive vertices remain fixed at baseline. Infeasible fixed rows are retained,
including a zero displacement envelope; they are not silently discarded.

Diagnostics retain overlap counts, branch crossings, minimum slack, bounded worst
witnesses with original triangle identities and explicit coverage. Degenerate
projections and no positive-area overlap leave coverage incomplete. This is
piecewise-linear depth contact in the declared frame and pose, not general 3D
collision, swept motion, visibility, anatomy, guide correspondence or appearance.
Triangle membership alone does not qualify an anatomical layer. No default
margin or character dimension is inferred.

## Scoped native bootstrap

Use the workspace's isolated native runner through `NativeJob`. Replace historical
script-prefix execution with a reusable private setup module over
`native_bootstrap.load_pinned_modules`. Bind each module and its transitive
dependencies in the task. The helper verifies all supplied module bytes before
executing any, avoiding stale bytecode or accidental historical construction.
Restore required drivers and inventory callbacks, relocate mutable diagnostics
into the isolated output, and preserve required support files.

Register `NativeJob` directly as the handler to use its read-only `preflight`
automatically during task observation, before inference or operation reservation.
It requires the runner executable, source checkpoint, script, output root, exact
source/dependency hashes and runner/live-source references; malformed fixed inputs
stay blocked with an actionable reason. Real source hashes and live ownership
are still checked at execution. For an existing wrapper, expose
`wrapper.preflight = validate_native_job` from `native_recipes`, or call that
validator while building the catalog. Other qualified handlers may expose the
same read-only `preflight(item)` hook and raise a public `ValueError` for an
invalid fixed configuration. A preflight must never call Blender, a provider or
an output writer. Meaningful operation/argument choices remain explicit owner decisions.

`begin_scoped_feature(feature_api, feature, mismatch, mechanism, evidence_dir,
scope_check=..., protected=..., repair_invalid_bindings=False)` uses the existing
feature API. It refreshes the actual feature fingerprint without a broad depth
refresh and requires freshness again after the operation's checks. Its private
`scope_check(record)` must actually verify all seven dependency roles: source,
guide, depth, sections, correspondence, preservation and adapter. Each returns
`{status: "pass", evidence: [{path, sha256}, ...]}`; the exact files are verified.
The callback, its data and referenced implementation must be bound job reads.
A stored declaration alone is not a native check.

Qualify the full affected feature and dependent layers, including source geometry,
transforms, modifier/position owners, correspondence and downstream attachments.
Do not infer coverage from a vertex count. If that closure is unavailable, keep
the existing broad setup for that operation. Scoped setup does not consume old
depth-cache results as fresh evidence. Invalid-binding repair remains explicit;
check bindings at the outcome. Atomic native state guards, exact input copies,
source-before/after hashes, saves and consequential independent reopen remain.

```python
from modeling_system.native_bootstrap import StageTimings

stages = StageTimings(output / "native-stages.json")
with stages.measure("driver_restore"):
    restore_qualified_drivers()
with stages.measure("scoped_checks"):
    verify_current_dependency_closure()
with stages.measure("construction"):
    execute_selected_operation()
with stages.measure("evidence"):
    capture_required_evidence()
stages.finish()
```

Phase blocks do not nest. Completed and failed phases persist; unfinished phases
remain unknown. NativeJob reads the bounded receipt into `native_stages_ms` under
`worker_` names, while `isolated_job` remains the encompassing wall time. Do not
sum those inclusive and component timings together. Invalid timing metadata
cannot change a known native outcome or trigger a replay. Compare ordinary runs
before claiming savings; shorter unrelated captures are not bootstrap benchmarks.

## Preserve the right things

Every exact-coordinate lock needs its reason and supporting evidence. Accepted
guide identity, protected user edits and the current implementation are separate.
A known source defect at rest may need repair alongside moving geometry. Keep
the accepted design and justified invariants, not an accidental current gap.

When a small fixed set of views is all required to resolve one question, keep that
capture bundle in code inside the selected operation. The owner chooses discretionary
work and meaningful alternatives; an underdetermined choice between equivalent
camera bundles should not block that operation. Actual image interpretation still
belongs to Astra, and a successful diagnostic is not a retained visual gain.
