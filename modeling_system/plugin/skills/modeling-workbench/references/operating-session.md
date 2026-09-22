# Astra and Jev operation

Use `modeling_system.operating_session.OperatingSession` for continuing modeling.
It composes the existing decision episode, WorkQueue, LaneSelector, controller,
service journal, procedures and qualified private native adapter. It does not
replace the geometric, reference, motion, recovery or retained-method tools.
`operating_protocol()` lists task profiles and routes every capability family.

## Responsibilities and the modeling loop

Astra owns the accepted design, dominant visible defect, scope, hypotheses and
actual image interpretation. It develops a new capability when the current ones
cannot express a useful correction. It does not directly inspect, pose, render,
save or edit Blender. Those operations run inside Jev-selected capabilities or
their fixed prerequisites. Avoid a separate Modeler handoff per action.

Jev chooses the next useful lane, method, region, compatible scope, evidence or
recovery option. Existing code executes the selected capability immediately.
Fixed preservation, capture and persistence steps are ordinary code inside that
capability. Batch independent questions; conditional answers cannot read one
another. Reuse only exact applicable answers. See the official TypeSafe
[state](https://docs.typesafe.ai/concepts/state) and
[fan-out](https://docs.typesafe.ai/patterns/fan-out) contracts. Jev receives text
and structured facts; Astra interprets the actual images.

The loop stays: current situation and accepted identity -> causal question ->
qualified guide/depth/section support -> recoverable correction -> early matched
appearance -> useful-result motion/reopen -> retain result and method. These are
dependencies, not a compulsory complete checklist before every edit. Reuse
unchanged evidence and preserve gains while a remaining defect is unresolved.

The seven lanes distribute work, not scene ownership. Independent immutable
analysis and preparation may overlap through qualified workspace workers;
native effects remain serial. The current generic queue executes handlers
serially and batches independent Jev judgments. No native concurrency or measured
speedup is implied. Astra can submit a scoped review while an unrelated operation
runs; immutable feedback files never overwrite the owner's queue state.

## Bind once, keep existing work

```python
from modeling_system.operating_session import OperatingSession

session = OperatingSession(
    private_queue_directory,
    service=service, episode=active_episode, owner=sole_owner,
    goal={"objective": bounded_modeling_goal},
    observe_context=read_actual_revisions,
    catalog=qualified_catalog,
    handlers=qualified_handlers,
    judge=existing_budgeted_judgments,
    public_projection=reviewed_public_projection,
    report=adapt_existing_receipts,
)
session.run(max_steps=authorized_step_bound, on_status=owner_status_callback)
```

The direct TypeSafe transport, credentials and provider ledger stay with the
bound workspace. Use its current budget; persistence never renews it. The
judgment callback rechecks the exact owner/menu/dependency binding before release.
Status uses the same controller path and native display adapter as before.

An optional `budget=DecisionBudget(...)` configures the queue, selector and
default retained context together. New tasks can attach closed, typed
`decision.arguments` to existing handlers; Jev chooses the operation and its
arguments in one batch. The original task remains immutable while the exact
compiled payload enters the existing journal and handler. See
[typed control](typed-control.md) for the task schema, grounded observation
contracts, staged retrieval and joined decision outcomes. The private transport
must honor the packet's budget rather than keep an older fixed question limit.

An existing settled WorkQueue can be attached at the same directory with the
same owner and goal. Original completed item definitions and receipts stay
unchanged. Attachment retains their historical provenance in the episode; it
does not pretend that they executed through the new runtime. Add contracts to
new tasks. An uncertain effect must be reconciled before continued execution.
Use the real decision episode, not a native feature-context fingerprint. The
historical episode baseline does not select the current Blender checkpoint.

The context callback reads observed revisions and actual lane state. The catalog
supplies bounded qualified operations, never provider-authored code. All Blender
access, including fresh native preflight, stays inside the selected handler and
its atomic native guard. Keep source/pose/side/support, target and loaded adapter
identities in relevant reads; a label or supplied version is not qualification.

## A task carries its modeling relationships

Each existing WorkQueue item adds `workbench`:

```python
{
    "profile": "appearance_edit",
    "capability": "qualified guide-constrained attachment repair",
    "method": "measured support-frame transport, revision 2",
    "native": True,
    "stage": "native_trial",
    "bindings": {
        "identity": ["accepted_identity"],
        "baseline": ["source", "pose"],
        "guide": ["guide", "guide_disposition"],
        "depth": ["depth_support"],
        "sections": ["corresponding_sections"],
        "correspondence": ["surface_binding"],
        "preservation": ["protected_regions"],
        "adapter": ["qualified_adapter", "method_script"]
    },
    "consumes": {"prepare": ["analysis"]}
}
```

Every binding key must exist in the item's exact `reads`; each consumed task
must be a declared `requires` prerequisite. The runtime verifies these
relationships and requires actual output evidence. Native qualification still
has to implement them: actual registered guides, depth and corresponding
sections constrain construction, including smoothing, joins and attachments.
Metadata or a post-hoc distance check does not implement guide fitting.

Profiles cover analysis, inspection, appearance edits, capture, verification,
display, checkpoint, retention, generation, recovery and learning. Read their
current inputs/outputs with `operating_protocol()` rather than copying a stale
schema. A generation capability wraps the existing source-bound claim/reconcile
job and current policy/preflight; it never starts a second provider ledger.
Recovery uses the original transaction/job receipt and later-edit protections.
Save/checkpoint work may run while visual judgment is pending. Local retention
additionally declares `visual_review: candidate_task_id` and a candidate
prerequisite, plus exact reopen evidence. User acceptance remains separate.

## Results are the next decision's inputs

A handler or the session's `report(item, result)` adapter returns:

```python
{
    "checks": {
        "analysis": {"status": "pass", "evidence": [
            {"kind": "file", "path": actual_report_path,
             "sha256": observed_hash, "role": "actual qualification"}
        ]}
    },
    "findings": [
        {"kind": "scope_noop", "scope": "first side",
         "summary": "This construction is already present on the first side."},
        {"kind": "predicted", "scope": "second side",
         "summary": "Local placement changes; actual visual effect still needs review."}
    ]
}
```

Checks are pass/fail/unknown with typed record/workflow/file evidence. Required
outputs missing from a completed execution become `needs evidence`, not a
fabricated success or an automatic rerun. `consumes` tests the named checks.
Output facts and the applicable procedure context enter the existing episode
operation journal. A raw capability result is saved before report/indexing work,
so later bookkeeping failure cannot erase a successful native operation.

Findings have a kind, scope and deliberately public summary. Use measured,
predicted, hypothesis, scope_noop, unsupported, appearance, failure or method.
Keep uncertainty, contradictions and regional applicability explicit. A worse
maximum angle cannot veto a visually useful gain, and an unchanged region
cannot turn another region's gain into a failure. Technical checks, Astra
appearance judgment, user acceptance and method benefit are independent.

The session automatically inserts these findings and current review feedback
into every Jev choice; the private projection cannot accidentally omit them.
Changed source/guide/intent/public facts invalidate affected decisions before
dispatch. Historical findings remain labeled historical. Do not copy private
paths or identity into public summaries. Reports are bounded; when a decision
context exceeds the budget, narrow the active catalog or finish the episode
instead of silently truncating relevant evidence. The full episode retains it.

## Astra feedback, recovery and learning

Use [cooperative intervals](cooperation.md) to expose the exact review inbox and
continue qualified unrelated work while reviewing. The ordinary session API also
supports a finite supervised mailbox wait; no new controller or agent is needed.

After examining actual matched views, bind the judgment to the current candidate:

```python
basis = session.review_basis(candidate_task)
session.record_review(
    candidate_task, expected_basis=basis,
    judgment={"disposition": "useful", "scope": reviewed_scope,
              "reason": observed_visual_change, "next_question": remaining_defect},
    evidence=actual_image_links,
)
```

Useful/rejected/unresolved is a scoped Astra interpretation, never automatic
whole-character approval. A new result, changed guide or changed episode intent
invalidates it. Review submission only writes immutable episode-linked feedback;
the owner consumes it at the next decision boundary. `inspect_operating_session`
and `decision_workspace` expose exact receipts and expansion links without
touching Blender. Update existing episode requirements/context with
`revise_episode`; retain the current human checkpoint authority separately.

`session.recover(task)` repairs a stopped queue/controller from the exact already
returned qualified operation result. It never invokes the capability. If the
capability returned but reporting failed, inspect its `capability-result.json`
and use `session.repair_report(task, corrected_report)`. It retains a separate
report operation and resolves the original failure with exact return evidence;
the failed original record is preserved and the capability is never repeated.
Known deterministic projection refusals before inference clear their own pending
selection. For an older retained refusal, `session.recover_selection` requires
the exact selection fingerprint, observed `confirmed_not_dispatched` with its
basis, and actual typed evidence. Unknown or submitted provider requests keep
their original ledger identity and cannot use this route.

For HTTP 200 with a valid retained answer but failed metadata, use the separate
completed-response route. It validates the exact owner packet, wire request,
model, full answer distributions and usage; it never sends another request.

```python
from modeling_system.provider_recovery import reconcile_completed_response
from modeling_system.controller import fingerprint, read_json

call = existing_call_directory
reconcile_completed_response(call, existing_ledger_directory, session.observe)
pending = read_json(session.directory / "controller" / "controller.json")["selection"]
session.recover_completed_selection(
    expected_selection=fingerprint(pending),
    packet_path=call / "owner-packet.json",
    request_path=call / "decision-1.request.json",
    response_path=call / "decision-1.response.json",
    reconciliation_path=call / "completed-response-reconciliation.json",
)
# Continue through the usual bounded session.run(...) and native guard.
```

Stop the controller before recovery. The ledger helper shares the dispatch lock,
keeps original failure evidence and updates only the already existing attempt.
Already-accounted usage is not added twice; missing accounting can be completed
once from retained token usage. It supplies no extra requests or spending scope.
`session.observe` must provide fresh owner, authority and dependency revisions.
Changed evidence or public context keeps the old response from being released.
The exact pre-call batch includes cached branches, so later cache changes cannot
reinterpret the original questions. Earlier runtimes without that batch need
explicit owner reconciliation from their original records.

Provider, queue and controller writes use the same bounded Windows sharing-error
retry on metadata rename only. A complete fsynced temporary receipt survives
exhaustion; explicitly inspect and supply its path as `response_path` to the
ledger helper. Do not guess among temporary files, repeat HTTP or replay Blender
work. The transport records response and cost metadata together in one write.

Normal Jev use follows existing account credits. A `normal_use` policy may omit
`max_cost_usd` and `max_requests`; installation does not impose a lifetime spending
or request cap. Preserve usage accounting, explicit user limits when supplied,
and the prohibition on purchases/topups. Provider credit or authorization errors
stop normally; a synthetic test budget is not a user's spending instruction.

For transient HTTP429/500/502/503/504/529 or a recorded network timeout/reset,
`provider_recovery.prepare_transient_retry` preserves the original request and
accounts its original reservation once. Missing responses retain an **unknown**
provider outcome; they are never relabeled not-dispatched or successful. The helper
prepares a distinct linked packet. The normal bridge allows two retries after the
original attempt, with backoff and fresh dependency checks. This is a per-decision
failure bound, not a lifetime usage gate. A further failure stops under its existing
identity; invalid answers, exhausted credits and uncertain native effects never
trigger this route.

After a retry succeeds, reconcile its completed response and supply that retry's
`transient-retry.json` as `transient_retry_path` to `recover_completed_selection`.
Both original failure and new answer remain linked. An explicit authority correction
can be rebound with pinned evidence only if every non-authority geometry, guide and
operation dependency still matches; retain both old and fresh selection contexts.
Never silently treat an older pending selection as current under changed authority.

An authority correction can change the applicability labels of retained findings.
Use `session.authority_feedback_rebinding(previous, current, authority_keys)` to
reconstruct the saved feedback with only the old authority values restored. Pass
that proof as `feedback_rebinding` with the explicit rebind. Every finding and check
remains identical. Newly stale reviews leave the current view while their original
records remain retained; no judgment can be added or rewritten. Geometry, guides
and other dependency revisions must still match. A changed feedback hash alone is
not evidence for this exception.

When a handler itself raises after known partial effects and has no
`capability-result.json`, use `failed_task_recovery.reconcile_failed_task` with
the exact operation handle, actual receipt links, a failed workbench report and
`observed.effect_status="resolved_failed"`. Name the completed and unapplied
effects; `uncertain_effects` must be empty. The helper preserves the original raised
outcome, records a failed task and repairs its execution indexes without invoking
the handler. It never fabricates a completed return. After reconciliation, offer
only the remaining work as a new qualified task. A known return instead uses the
existing result/report recovery; unresolved native effects remain blocked.
The earlier terminal503-specific helper remains available for recorded older flows.

Run existing `record_outcome`, `reconcile_episode`, `promote_procedure` and
`retrieve_experience` through the episode as before. Preserve failures and exact
limits. A useful method is learned only with its supported judgment and actual
reuse; logging a successful command does not establish method benefit. New
mechanisms become qualified reusable capabilities, so Astra does less recurring
plumbing while retaining responsibility for likeness and unresolved art questions.
