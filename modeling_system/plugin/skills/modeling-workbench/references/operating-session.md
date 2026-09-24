# Direct owner operation

The owner chooses qualified operations, runs them through the existing native
adapter and interprets actual images. `direct_session.create_session` composes
the decision episode, queue, controller, service journal, preservation and
experience. No inference runtime or provider ledger participates.

## Bind a session

```python
from modeling_system.direct_session import create_session

session = create_session(
    queue_directory, service=service, episode=active_episode, owner=sole_owner,
    goal={"objective": bounded_modeling_goal},
    observe_context=read_actual_revisions, catalog=qualified_catalog,
    handlers=qualified_handlers, task_order=explicit_task_order,
    preservation=required_outcomes, report=adapt_existing_receipts,
)
session.run(max_steps=authorized_step_bound)
```

`report`, `preservation` and `task_order` are optional constructor arguments;
affected appearance/retention tasks require a bound preservation policy to run.
An eligible singleton or fixed housekeeping item runs directly. With multiple
eligible operations, `task_order` chooses the first eligible named task. A local
`select(state, actions, plan)` callback can replace the order and must return an
eligible ID or `{"status": "needs_review", "reason": ...}`. Supply only one
selection mechanism. Without one, multiple choices yield a scoped handoff.
Substantive edits cannot be labeled required housekeeping.

The context callback returns actual owner, authority revision, dependency values,
active native operations and compact observed state. The catalog describes only
authorized qualified work. Observe and catalog callbacks must not mutate Blender.
The controller revalidates exact action definitions and relevant dependencies
before dispatch. Each native handler also checks atomic expected state at its
effect boundary. An active native operation blocks another writer.

`ModelingService` can invoke existing native inspection/display operations
directly with the owner and expected-state contract. `NativeJob` and
`RetainCheckpoint` can be called directly with qualified task/context records or
used as session handlers. Direct use retains their native guards and receipts;
it does not automatically journal session contracts. Use a session when those
contract checks and candidate dependencies are needed.

## Modeling relationships

Each task declares `workbench.profile`, capability, method, native flag and
`bindings` from roles to exact task read keys. `operating_protocol()` reports
current input/output profiles. Appearance edits bind identity, baseline, guide,
depth, sections, correspondence, preservation and adapter. These must constrain
the actual fitter; metadata or a later distance check is insufficient.

`requires` specifies allowed settled outcomes of prerequisite tasks. `consumes`
names their required checks. Unknown or failed checks block dependent use.
Retention declares a candidate prerequisite and `visual_review` plus independent
reopen evidence. User acceptance is separate from operator retention.

Use [candidate pipelines](candidate-pipeline.md), [parameterized operations](operation-recipes.md)
and [explicit typed arguments](typed-control.md) to reuse mechanisms. Catalogs
must omit inapplicable methods or expose missing support as blocked work. Never
replace an unknown condition with automatic success to keep a queue moving.

## Evidence, review and learning

Each effect has a durable operation handle before dispatch. Returned reports
retain exact evidence, measured/hypothetical findings, profile checks and actual
procedure context. Current findings, visual feedback and relevant retained
experience enter the next owner observation. Historical evidence never overwrites
current authority. `WorkLimits` bounds local catalogs and context, not spending.

Compare baseline and candidate at matched views, poses and states. `review_sheets.compose_review_sheet(rows,
columns, output)` arranges the actual renders in a labeled grid, pins each source image's hash for the review
evidence and reports missing images or differing image sizes (an unmatched comparison) instead of hiding them;
`review_sheets.rows_from_frames(frames, views=..., poses=..., states=...)` builds the rows from recorded frames.
It arranges images only: look at every one, and keep the overlap view (character solid, guide wire) as the
guide comparison. Blender's Workbench final render draws an object whose display type is Wire as a solid surface,
so a guide set to Wire hides the character instead of overlapping it; render the guide's edges with a temporary
Wireframe modifier (replace mode, a thickness well below the features being compared) and remove it afterwards,
then check that the character actually shows between the wires before using the image.

Read the exact review basis and inspect linked images before submitting:

```python
session.begin_review(candidate_task)
session.record_review(candidate_task,
    expected_basis=session.review_basis(candidate_task),
    judgment={"disposition": "useful", "scope": reviewed_scope,
              "reason": actual_visual_finding, "next_question": remaining_question},
    evidence=actual_image_links,
)
```

Review dispositions are useful, rejected or unresolved. Reviews are immutable
episode facts in a separate index; they do not overwrite the running queue.
`record_review_experience` and `learning.record_method_experience` retain supported
techniques, counterexamples, applicability and limits. Read [retained learning](retained-learning.md).

`begin_intervention`/`end_intervention` and review timers retain observed workload.
`metrics()` reports explicit owner selections, singleton routes, actual operation
times, native stages and missing coverage. No counterfactual savings are inferred.

## Recovery and history

Use `recover(task)` to restore the exact completed episode result after an index
interruption. `repair_report(task, report)` repairs evidence presentation without
repeating a known effect. Neither route invents a successful native outcome.
Uncertain effects remain stopped until reconciled from their original receipts.

A capability can also raise after its inner effect, for example a wrapper error
after a completed native job, leaving no qualified return. Inspect the actual
effect, reconcile the original operation with
`service.reconcile_operation(handle, observed={"effect_status": "resolved_failed"
or "confirmed_not_applied", "basis": ...}, evidence_paths=[...])`, then call
`session.settle_reconciled(task, expected_handle=handle)`. The task is recorded
as failed with a link to that reconciliation; nothing is dispatched and no
success is reported. Dependents see a failed prerequisite, and a corrected task
revision runs as new work in the same session. A `confirmed_returned`
reconciliation still uses `recover` or `repair_report`.

`recover_selection(expected_selection=..., observed=..., evidence=...)` can clear
an exact old pending selection only with explicit no-dispatch evidence and no
uncertain native effects. It does not request or release a model response.

`inspect_operating_session` and `decision_outcomes` read historical reviews and
selection traces without any former transport module. Preserve old provider
receipts/ledgers/checkpoints. Start direct work in a new directory with explicit
arguments and observed prerequisites; legacy inferred catalogs cannot execute.
The episode baseline is historical context, not today's selected checkpoint.
