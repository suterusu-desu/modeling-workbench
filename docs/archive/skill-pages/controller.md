# Persistent owner controller

The owner supplies qualified actions, actual observations and explicit priorities.
`PersistentController` executes on the creating owner thread, with one exclusive
controller lock and serial native effects. `WorkQueue` and `OperatingSession`
compose it with dependencies, journaling, preservation and exact review gates.

Observation returns owner, authority_revision, stage, actual dependency values,
active_operations and eligible action definitions. A qualified optional
revalidate callback can check only selected dependencies instead of rebuilding
the whole menu. It must supply fresh facts; it cannot fill missing values from
an old snapshot. Native effects also need an atomic expected-state guard.

An eligible singleton or fixed prerequisite runs directly. Multiple choices
require an explicit owner selector; the queue supplies `OwnerSelection` from an
authorized task order. A choice never creates or widens a capability. Selection
and dependency bindings are rechecked before dispatch. The optional MailboxPlanner
is a local asynchronous owner-plan mailbox, not an inference transport; a plan
prioritizes existing operations and cannot create new native authority.

Attempts land durably before effects. Interrupted selections and uncertain native
attempts stop for explicit receipt reconciliation. Settled effects do not replay
on restart. A failed report cannot erase a completed effect. Unknown outcomes do
not authorize another attempt, even under a new task label.

Use finite max_steps, cancellation and actual on_status progress. The Blender
status adapter can display current operation and returned evidence through the
same qualified native owner. Do not label a completed or stopped job as running.
The controller does not grant unattended-work or generation authorization.

Reuse unchanged evidence and qualified native bootstrap. Fixed saves, capture
bundles and recovery steps belong in code. Session metrics retain observed cycle,
execution, native-stage and review times with missing workload explicitly marked.
No inference service or model credentials participate in this controller.
