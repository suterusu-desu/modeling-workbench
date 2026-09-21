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

## Retained experience in each choice

`RetainedContext` queries existing authority, outcome, judgment and procedure
records. Exact passages, hashes and coverage remain in the private run's
`experience/` records. `project(row)` explicitly creates a short safe public
summary, or returns None. Identities, filenames and private locators are not
copied to the provider automatically. Up to four matches per category are
retrieved; the public content has a bounded size and exposes omitted coverage.
Queries may be strings or `(state, items, outcomes)` functions. Optional
`context(state)` supplies applicability fields.

Jev receives the summaries alongside current findings before choosing. Up to four
independent passage-relevance judgments share the existing action-selection
request. The action questions already see the passages and do not assume another
question's answer. `last-batch.json` retains method probability rankings and
passage relevance; a relevance score never changes authority or visual judgment.
Unchanged choices retain the normal exact cache. Changed passages invalidate the
choice before execution. Missing history remains visible as missing coverage.

## Reusable array preparation

`ArrayPreparation(run / "prepared")` is a queued handler for declarative recipes.
Its payload contains `operation`, `inputs` and optional `parameters`. Each input
is `{path, sha256, array}`; its exact file hash must also be a task read. NumPy
archives are loaded without pickle. Outputs contain array and JSON artifacts
with hashes, checks and compact numerical findings. Inputs and failures stay
private; no arbitrary Python or Blender operation is executed.

Supported operations: `section_fit`, `material_path`, `compose_correspondence`,
`prepared_effect`, `active_vertex_coverage`, and `pose_correspondence`. The same
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
