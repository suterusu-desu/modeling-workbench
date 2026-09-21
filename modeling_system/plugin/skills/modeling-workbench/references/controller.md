# Persistent owner controller

Use `modeling_system.controller.PersistentController` for an ongoing, authorized
modeling sequence. It composes existing native operations and private selectors;
it does not supply a new rig adapter, model account or generation authority.
Run it on the sole native owner's thread. Only the advisory planner runs on a
worker. Required operations and a single eligible operation bypass inference.

The execution relationship is the owning planner -> Jev -> native execution code. The
owner authors objectives, constraints and executable primitives; `select`
returns an operation and the controller immediately calls `execute` in the same
tick. There is no owner-language-model interpretation/approval round trip per
action. "Sole native owner" means one serialized adapter lane, not human/agent
mediation of each edit. Return to high-level reasoning for an actual missing
operation, changed scope, unresolved judgment or failure needing a new mechanism.

```python
from modeling_system.controller import PersistentController, MailboxPlanner

with PersistentController(
    private_run_directory, owner=owner_id,
    observe=adapter.observe, execute=adapter.execute,
    select=adapter.select, initial_plan=current_plan,
    planner=MailboxPlanner(private_planner_directory),
    on_status=adapter.show_status,
) as controller:
    result = controller.run(max_steps=100)
```

`observe()` returns fresh local facts:

- `owner`, `authority_revision`, `stage`, `values` (dependency key to exact
  revision), `active_operations` (actual owner lane), `actions`, optional `done`.
- Each action has `id`, `revision`, nonempty `reads` (key/revision), `writes`
  (subset of reads), optional `required`, `completed`, `blocked`, and private
  `payload`. The adapter authors the finite menu; planner output cannot add code
  or native authority. Revise an action when its method or relevant evidence
  changes, never just to evade a retained failure.
- Optional `inference_budget` reports remaining authorized calls/cost from the
  existing provider ledger, never a fresh allowance. Missing means unknown.
- Optional `planner_reads` narrows a planning request's dependencies to what
  actually determines its objective. Include guide, scope and construction
  dependencies; include mesh revision when reasoning depends on that mesh.
  Omission binds the full measured revision map. `replan_revision` marks new
  owner evidence or a changed question, not an automatic retry counter.

`execute(action, context)` runs the existing bounded native primitive on the owner
lane. Recheck `context['owner']`, `authority_revision` and `expected_values` under
the adapter's native lease immediately before effects; the controller's earlier
check cannot replace that atomic native guard. Preserve registered guide/depth
constraints, original/user edits, checkpoints and independently required reopen
verification. `context['cancelled']` is a cooperative stop event; operations must
keep their existing time/iteration bounds. `context['progress'](**facts)` reports
actual progress on the owner thread, never fabricated percentages.

Return `{'status': 'completed', 'evidence': [...]}` after a settled operation.
`failed` or `no_progress` means effects are known and settled. Exceptions, timeouts
with unknown effects and explicit `needs_reconciliation` block further dispatch.
Keep technical completion separate from visual judgment and user acceptance.

`select(snapshot, eligible_actions, plan)` returns exactly one offered action ID.
Use the workspace's existing compact anonymous projection, provider transport,
budget reservations and receipts. Raw local snapshots/payloads contain private
data and must not be forwarded wholesale to an external service. Required steps
execute in owner menu order; compatible operations may be one bounded native
macro. More than one optional action is a genuine decision requiring the selector.
Missing selection or planning reports a real waiting state rather than guessing.

`planner(request)` receives `request_id`, exact `binding`, `snapshot`, prior
`plan` and a reason. It returns an objective with `authority_revision`, `stage`,
nonempty `reads`, and optional priorities. It is advisory and read-only: no bpy,
scene capture, native dispatch, live adapter access or authority changes from its
worker. Plan/request dependencies are both checked when the response is consumed.
Initial owner-authored plans permit useful work while the replacement is pending.
Changed objectives invalidate an in-flight selection before native dispatch.
Requests are deduplicated for unchanged observations; progress refreshes are
rate-limited. Planner failure leaves an applicable existing plan available.

The mailbox alternative writes `request.json`; an existing planner agent reads
it and atomically writes `response.json` with matching `request_id` and `plan`.
It creates no agent, paid call, scheduled task or permanent background service.
The mailbox is active only while this explicitly invoked controller is open.
The callback alternative supports an already-authorized bounded model transport.

## Visible operation and recovery

Keep the working candidate and registered guide/depth workbench visible in the
shared Blender viewport, at the current useful pose and view. Use `on_status` or
read `status.json` from a Blender draw handler/timer for an overlay containing
objective, actual action, progress, planner pending state, remaining budget and
idle/stopped/error state. The overlay must also expire to disconnected/stale when
its writer disappears; a leftover `running` file is not proof of active execution.
Display status without moving geometry or manufacturing activity.

The reusable `blender_controller_status.py` supplies `install(status_path)` and
`uninstall()` for the sole owner to call in Blender. It draws a POST_PIXEL overlay
and viewport header, expires stale active statuses, and reports unknown budgets.
Load this standalone file through `importlib.util.spec_from_file_location` when
Blender does not have the core Python dependencies. Its timer only reads status
and redraws. Offscreen captures can omit overlays; inspect the actual UI separately.
Report progress during long primitives at least every 30 seconds, or select a
longer explicit display expiry; do not use a stale file as liveness evidence.

`controller.json` persists exact action attempts before native effects;
`events.jsonl` links choices, plan acceptance/discard and operation results.
Repeated completed/failed/no-progress action identities are pruned. `run()` ends
on no useful work, waiting, uncertainty, cancellation or its explicit step bound;
it never busy-polls unchanged state. After new evidence/plan arrival, the owner can
call `tick()` or `run()` again on the same open controller.

The exclusive `controller.lock` prevents a second driver. A crashed process leaves
the lock and unresolved attempt intact: inspect its PID, native state, original
receipts and provider ledger before explicitly reconciling the journal and
removing only that abandoned lock. Clean close releases the lock but does not
clear uncertain operations or renew inference budgets. Reopening resumes retained
failures and completion. Saving native work never waits for planner prose.

Use this directly in normal modeling after proportionate contract checks. Fix
observed problems in use; a separate benchmark/adoption phase is not required.

## Minimize repeated work

Optional `revalidate(action, expected)` replaces the second full observation with
an adapter-qualified check of the selected operation. `expected` includes owner,
authority revision, stage, the union of action/plan/pending-plan reads and action
fingerprint. Return fresh `owner`, `authority_revision`, `stage`, `values`,
`active_operations`, and the fingerprint of the currently authorized action.
Read these from current authoritative state; never echo the expected values or
fill missing dependencies from cached observations. Include the entire required
read union even when the chosen action has fewer dependencies. Changed action
identity, scope, guide, plan or busy lane prevents dispatch. Without a qualified
callback the controller retains its complete second observation. File hashes,
dirty flags and owner tokens alone cannot establish evaluated geometry freshness.
This optional optimization never replaces the atomic native guard in `execute`.

`decision_menu.factor_choices(options, instructions)` compiles compatible
operation families and their conditional targets into one typed request. Each
option has a private `id` and `family`, plus reviewed public `family_description`
and `target_description`. Send only returned `questions` and reviewed public
state; retain `mapping` locally. `resolve_choice(mapping, choices)` resolves only
the selected family's target. Singleton branches omit unnecessary questions; a
fully determined menu needs no inference. Keep family descriptions distinct and
explain the decision criteria in instructions. Transport validation, authorized
budget, dependency binding and native dispatch remain the existing adapter's job.
This removes a sequential operation-then-target request when one is otherwise
needed; it does not make a previously flat single-request menu faster by itself.

Each cycle records `timings_ms` for observation, selection when needed,
revalidation, execution and the cycle through result persistence. The final
status file and `cycle_timing` journal event retain these actual measurements;
the Blender status overlay displays them. Use normal modeling receipts to locate
wasted work before optimizing. Timings are elapsed durations, not proof of better
geometry or an end-to-end speedup. Required and singleton actions still bypass
Jev entirely. Keep one serial native writer and batch only compatible decisions.
