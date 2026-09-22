# Required outcomes and retained methods

Use this path when edits must preserve an established guide-defined regional
shape or relationship through motion. Requirements are active data outside
ranked history. Lessons are conditional evidence about techniques and failures.
Neither replaces actual whole-region appearance review or user authority.

## Bind a scope once

`PreservationPolicy(requirements_ref, adapters)` loads a private hash-pinned
requirements document and contributes actual file revisions to the session.
Keep the construction reference current; an old rig description is historical.
If the useful baseline is unresolved, offer qualification or recovery rather
than inventing numeric targets or freezing a rejected shape.

```python
requirements = {
    "schema_version": 1,
    "construction": current_construction_ref, "authority": current_authority_ref,
    "public_construction": {"mechanism": "Deliberately public source-backed facts"},
    "outcomes": [{
        "id": "surrounding-surface", "description": "Preserve achieved surrounding shape",
        "baseline": evaluated_baseline_ref, "guide": applicable_guide_ref,
        "evidence": [reviewed_outcome_ref],
        "constraints": {"correspondence": correspondence_ref, "positions": position_ref},
        "cells": [{"id": "middle", "region": "surrounding surface", "pose": "middle pose",
                   "metric": "maximum evaluated deviation", "rule": {"maximum": qualified_tolerance}}],
    }],
    "rejections": [{"subject": rejected_candidate_ref, "evidence": user_review_ref,
                    "reason": "Deliberately public scoped rejection"}],
}
```

Every ref contains `path` and `sha256`. Each required cell names region, pose,
metric and upper bound (`maximum`, `max_increase`, or both). Increases compare
actual candidate and protected baseline measurements. Qualify metric direction:
signed distance alone is not penetration. Missing cells remain unknown; finite
pose samples do not certify unsampled motion or appearance.

An adapter supplies pinned `sources` and three callbacks:

- `coverage(item, document)` returns affected outcome IDs and bound current
  construction evidence. Follow actual active output dependencies, including
  transitions; stored legacy attributes alone do not establish influence.
- `bind(item, constraints)` returns executable `payload` and `consumed`, mapping
  each outcome ID to its exact constraint-input references. Those refs must
  enter the actual executable arguments and be used by the fitting/native code.
- `measure(item, result, constraints)` reads saved evaluated measurements and
  returns `subject`, `construction`, `kind` (`prepared_output` or
  `evaluated_output`), source `evidence`, and `outcomes[id]` with baseline/guide
  refs and `cells[cell_id]` containing region, pose, metric, observed and baseline
  numeric values. A self-reported pass flag has no effect.

The adapter's `subject` is a key path into the actual result, default
`["subject"]`. The measurement must match that exact artifact. The core computes
every verdict and re-reads saved measurements before promotion. Callback source
hashes and function bindings are pinned. These are trusted qualified callbacks,
not a sandbox or a generic anatomical solver: test that injected constraints
alter a relevant solve and reject a counterexample. Payload inclusion alone
cannot prove a private solver correctly implements its constraints.

Pass the policy to `OperatingSession(preservation=policy,
require_preservation=True)` and `CandidatePipeline(preservation=policy,
preservation_adapter="adapter-name")`. Standalone tasks use `policy.bind_task`.
The pipeline freezes and carries the requirement identity and actual predecessor
result through preparation, application, verification, review and retention.
Verification can acquire missing evidence; promotion requires passing relevant
checks. A failed check preserves the completed effect without replay. Existing
checkpoint retention still owns native saves and recovery.

`require_preservation=True` blocks affected appearance/retention operations if
the policy is omitted, while unrelated analysis remains available. Set this in
the workspace's normal operator bridge; the public default stays compatible
with old scopes. A scoped structural reconstruction must carry and restore the
original guide-defined outcomes, not silently waive them.

## Local correction and connected repair

Preserve the guide-defined outcome within a qualified tolerance, not every raw
control value. Distinguish exact identity/topology constraints from allowable
guide error and allowable increase over an earlier useful evaluated fit. State
units, region, pose, correspondence, and the evidence supporting each bound.
Numerical solver tolerances are separate from visual/guide tolerances.

Start with a local solve using those constraints. If it cannot satisfy the
required correction, distinguish a measured constraint conflict from missing
support or solver failure. Offer a coupled correction over the smallest
qualified connected transition, retaining the same guide targets and bounds.
The local correction and surrounding repair form one recoverable candidate;
do not promote an intermediate broken surround and promise to fix it later.
Jev can select this registered remedy from actual feasibility evidence. Changing
the target, relaxing an authority-defined bound, or widening into unsupported
anatomy requires new qualification; it is not an automatic fallback.

`correction_scope` is available through the existing pinned-array preparation
handler. Inputs are a qualified linear `response` matrix, current guide
`residual`, per-output `tolerance`, `local_controls`, `coupled_controls` and
per-control `control_radius`. Supply `units`, a `qualification` description and
an explicit `numerical_tolerance` (1e-10 through 1e-7). Rows must cover affected
guide/relationship measurements through the relevant motion; columns represent
actual controls. It finds a minimum-absolute-movement feasibility witness under
`abs(residual + response @ step) <= tolerance`. Controls outside the proposed
scope remain fixed. It tests connected controls only when the local linear
problem is infeasible, using unchanged guide bounds; solver failures remain
unknown. Residual and bound violations are checked independently of the solver
success flag. Sparse matrices are supported when called directly.

This screens only the declared linear response and trust region; nonlinear
Blender behavior requires actual evaluated verification. No generic helper can
invent that response, anatomical scope, visual tolerance or missing motion.
PreservationPolicy still checks the actual candidate against its source-bound
requirements, followed by whole-region comparison to the guide and earlier
useful appearance. A scoped improvement is retained only when the overall
candidate improves without unacceptable collateral loss.

## Jev and durable learning

Required outcomes/current construction enter decision state outside
`RetainedContext` ranking. Actual source, coverage, consumed constraints and
evaluated output checks stay in code. Binding preservation does not add a second
abstract approval question over an eligible Jev-selected operation. Author
semantic method/coverage questions only for specific unresolved conditions.
Optional `decision.method_checks.remedies` names currently qualified operations;
Jev can choose the relevant prerequisite in the same batch. Only the selected
branch is consumed, preserving exact cached-answer reuse and existing accounting.
MethodCatalog/ParameterizedCatalog also accept deliberate `method_checks` with
`method` and/or `coverage` question text. Do not ask Jev to rediscover known facts,
judge unseen geometry, waive a failed condition or score likeness.

Ordinary visual reviews continue entering the existing learning store.
`learning.record_method_experience` and `import_method_experiences` retain reviewed
techniques and failures with the existing five-field lesson (mechanism,
observation, conditions, next_use, limits), exact evidence, optional
protected_outcomes links, public interpretation and explicit authority.

Keep `knowledge_status` (demonstrated, conditional, unverified, superseded)
separate from `candidate_disposition` (scoped_success, failed, rejected,
unresolved). Demonstrated knowledge can concern a rejected candidate. Optional
classification is recurring_failure, supported_lesson or unresolved_hypothesis.
Use a known classification directly; any Jev interpretation needs reviewer
confirmation and the same sources/limits, not another automatic promotion.

Reviewed historical imports deduplicate by mechanism and evidence. Preserve
unreviewed inventories without promoting their claims. Changed interpretations
require explicit `supersedes` IDs; original records remain readable, and a
reviewer/model summary cannot override explicit user rejection. Private locators
and unreviewed historical text stay out of provider/public projections.

Attach `experience=...` to `session.record_review` or resume with
`session.record_review_experience(review_fact, ...)`. The actual completed review
is saved first, so learning can resume without repeating modeling or review.
Verify import and mechanism retrieval in a fresh process using the installation.
