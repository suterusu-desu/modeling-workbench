# Executable Jev methods

Use these APIs with the existing OperatingSession and CandidatePipeline. Register
qualified methods once and bind their saved inputs for each scope. Offer actual
alternative mechanisms, not arbitrary strength sweeps or cosmetic renamed trials.

```python
from modeling_system.method_catalog import MethodCatalog
from modeling_system.candidate_pipeline import CandidatePipeline
from modeling_system.retained_context import RetainedContext
from modeling_system.preparation import ArrayPreparation

methods = MethodCatalog([
    {"id": "section", "description": {
        "mechanism": "Fit qualified corresponding guide sections",
        "applies_when": "Section ancestry and material correspondence are available",
        "failures": "Raw facets may imprint ridges; preserve existing guide ownership"},
     "build": adapter.section_preparation,
     "candidates": adapter.apply_section_result},
    {"id": "path", "description": {
        "mechanism": "Remap corresponding material stations along an ordered guide path",
        "applies_when": "Root and endpoint identities establish the intended path",
        "failures": "Distance alone can assign the wrong semantic tip"},
     "build": adapter.path_preparation,
     "candidates": adapter.apply_path_result},
])
pipeline = CandidatePipeline(run / "pipeline", revision="qualified-scope-v1",
    prepare=methods, candidates=methods.followup("candidates"),
    verify=adapter.verify, retain=adapter.retain)
# Supply catalog=pipeline and the existing native handlers to create_session.
session.experience = RetainedContext(service,
    query=adapter.current_mechanism_query,
    project=adapter.public_experience_summary,
    sources=adapter.experience_sources)
```

Factories `(state, previous)` return qualified tasks, a task list, or None when
support is absent. Optional `conditions` use exact current context values before
calling a factory; missing conditions remain excluded. `methods.excluded` exposes
why. A factory still owns real qualification, dependency binding and meaningful
parameters. CandidatePipeline freezes the resulting tasks. Outside a pipeline,
the caller retains exact task definitions across restarts as for any WorkQueue.

The selected task retains `workbench.method_choice`. `followup(stage)` routes its
actual result to that method's next factory. Fixed native application after an
already selected preparation needs no duplicate inference. An unselected new
strategy retains action/defer even when it is the only applicable method. Fixed
saves, checks and recovery bookkeeping continue without model questions.

The same MethodCatalog supports diagnosis and recovery tasks. Recovery factories
bind original receipts and declare failed/no_progress prerequisites. Uncertain
effects must first be reconciled by the existing controller; a registry cannot
make them settled. Never hide old failures or run a sibling after rejection just
because its numerical target can execute.

Link methods with `remedy_methods=["diagnose_dependency", "compare_recovery"]`
using IDs registered in the same catalog. `method_checks` can supply specific
`method` and `coverage` questions. The final eligible action menu resolves those
links to actual task IDs; missing, completed or blocked remedies are not offered.
Jev answers applicability, coverage and a conditional remedy question in the
same batch as action selection. A relevant unmet condition can route directly
to the selected remedy, including across lanes, if that remedy's own checks
pass. A missing remedy returns the unresolved question rather than inventing
work or running the fit. The existing session continues from actual results.
Explicit `method_checks.remedies` still names exact currently eligible task IDs.

Supply `method_checks.prerequisites={"condition_id": "Specific current-stage missing condition"}`
when a general missing/unknown judgment would be ambiguous. One conditional
question in the same batch identifies the concrete unresolved condition, none,
or an unlisted gap. Its answer remains in `unmet_prerequisite`. Conflicting
applicability and reason answers stay visible and cannot waive the blocker.
Use completed source-bound observations in the current projection; remove stale
future-tense plans. Distinguish prerequisites for a recoverable trial from its
later native verification, visual review and retention. If a deferral is still
unexplained, resolve that specific uncertainty rather than repeat an unchanged
action question or fabricate a missing requirement.

Preservation bindings and remedy links alone do not generate semantic approval
questions. Code enforces observed prerequisites and actual preservation; Jev
selects among the resulting operations. Add `method_checks` questions only when
a specific semantic uncertainty still matters. A recoverable trial does not
need its future output to already be verified or visually accepted.

Register diagnostic and recovery mechanisms before fitting when they can change
the next edit. Offer a fit, an evidence-resolving diagnosis and a scoped recovery
only when each is executable and useful for the current uncertainty. Do not
manufacture alternatives or force redundant diagnostics after their evidence
already exists. Bind preserved evaluated positions into the actual fitter's
equations or fixed variables before solving, with explicit conflicts; a payload
label or post-hoc distance report alone does not preserve a fit.

## Retained experience in each choice

`RetainedContext` queries authority, outcome, judgment, procedure and automatically
retained visual review/lesson records. Exact passages, hashes and coverage stay in the private run's
`experience/` records. `project(row)` explicitly creates a short safe public
summary, or returns None. Identities, filenames and private locators are not
copied to the provider automatically. Dedicated review/lesson records project
only their already-public semantic fields. Projection precedes candidate limiting.
The shared `DecisionBudget` defaults to eight complete context passages within
8000 serialized bytes and a broader pool of 32 candidates within 16000 bytes.
Excluded and omitted coverage stays explicit; caveats are never clipped.
Queries may be strings or `(state, items, outcomes)` functions. Optional
`context(state)` supplies applicability fields.

OperatingSession supplies review retrieval by default, using the current public
situation and offered work as the query. Add source projection only when needed.
See [retained learning](retained-learning.md) for automatic review capture,
conditional lessons and migration without replaying historical candidates.

Jev receives the summaries alongside current findings before choosing. When the
candidate pool fits the initial context, independent relevance judgments share
the action-selection request. A broader pool uses one cached evidence-selection
request first; the resulting passages then enter the action request. This lets
previously omitted contradictions influence the choice. See
[typed control](typed-control.md) for staged retrieval, budget configuration and
recovery. Decision traces retain the rankings and passage judgments; a relevance
score never changes authority or visual judgment.
Unchanged choices retain the normal exact cache. Changed passages invalidate the
choice before execution. Missing history remains visible as missing coverage.

The store scans current record names and metadata on each lookup, reusing only
unchanged header classifications to skip unrelated receipts. Every selected
record still goes through byte reading and content-hash verification. New and
deleted records are discovered without restarting; no returned payload, guide,
native state or modeling dependency is cached by this optimization.

## Reusable array preparation

`ArrayPreparation(run / "prepared")` is a queued handler for declarative recipes.
Its payload contains `operation`, `inputs` and optional `parameters`. Each input
is `{path, sha256, array}`; its exact file hash must also be a task read. NumPy
archives are loaded without pickle. Outputs contain array and JSON artifacts
with hashes, checks and compact numerical findings. Inputs and failures stay
private; no arbitrary Python or Blender operation is executed.

Supported operations: `section_fit`, `material_path`, `compose_correspondence`,
`prepared_effect`, `active_vertex_coverage`, `pose_correspondence`,
`fit_landmark_field`, `material_trajectory`, `surface_realization`, and
`attachment_motion`. See [parameterized operations](operation-recipes.md) for
reusable data-bound catalogs, native handlers and review timing. The same
functions accept arrays directly inside existing qualified handlers.

- `prepared_effect` measures displacement on the declared active surface using
  the caller's actual qualification tolerance. With `stop_on_no_effect=True`,
  a no-op settles as no_progress and does not advance a CandidatePipeline.
- `active_vertex_coverage` distinguishes face-owning geometry from unused legacy
  vertices. It reports omitted active indices instead of certifying every base
  vertex as relevant.
- `compose_correspondence` preserves the existing stationary contribution while
  replacing the applicable moving contribution under original continuous weights.
  It never converts a diagnostic threshold into a hard anatomical mask.
- `pose_correspondence` compares explicitly corresponding material samples across
  actual saved poses. Lowest residual nominates a sampled correspondence; it does
  not admit a guide, invent samples or certify pose identity.

Supply these observations when they resolve the current uncertainty. They are
inputs to useful decisions, not mandatory extra passes for every edit. Qualified
character bindings and truly new native mechanisms still belong to the adapter.

## Observed workload

Begin actual Astra work with `token = session.begin_intervention(kind=..., reason=...)`
and finish with `session.end_intervention(token, evidence=...)`. Timers survive
restart and ending twice does not duplicate an interval. They measure elapsed
work intervals, including interruptions, not tokens or model compute. Open
timers and earlier unmeasured intervals remain explicit unknowns.

`session.metrics()` separates Jev selections by lane and offered option count,
fixed operations, single-executable continuation, native execution, actual review
submissions and measured Astra intervals. Cached valid choices count as Jev
selection, not fresh API requests; the provider ledger owns usage accounting.
`session_metrics.compare_sessions(baseline, current, comparison_scope=...)` compares
the retained measurements when scopes are explicitly comparable. Missing duration
prevents a time-difference claim. No avoided-turn or causal speedup is invented.
