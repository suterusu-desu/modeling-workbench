# Owner-controlled work queue

`WorkQueue` executes one authorized catalog across diagnosis, repair, verification,
review, recovery, experience and preparation. `OperatingSession` adds episode
journaling, modeling contracts, preservation, reviews and retained experience.
Use [direct sessions](operating-session.md) for the complete modeling loop.

Tasks bind unique IDs, revision, handler, meaningful description, completion
condition, exact `reads`, covered `writes`, payload and `requires`. Prerequisites
name allowed settled outcomes: completed, failed or no_progress. Cycles, absent
prerequisites and conflicting revisions are errors. Changed data blocks stale
tasks. Generated outputs must be declared prerequisite writes and bound from
actual output receipts, not a before-effect revision.

Eligible singletons and required housekeeping execute directly. For multiple
eligible tasks, provide `task_order=[...]` or a local owner `select` callback.
Only eligible IDs may be selected. No default priority is inferred. A deferral is
`needs_review`, not a failed or uncertain operation. Fixed dependencies continue
within one bounded run; native effects are serial under one owner.

Persist attempts before dispatch. Save exact returned results even if later
reporting fails. Settled definitions do not replay after restart; running or
uncertain attempts stop for reconciliation. Do not evade that stop by changing
task IDs or revisions. Preserve failed and rejected experiments as evidence.
