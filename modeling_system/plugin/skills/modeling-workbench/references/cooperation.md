# Cooperative Astra and Jev intervals

Keep a useful task frontier available to Jev while Astra reviews returned images
or develops a missing capability. Prefer event boundaries over a fixed number of
seconds: review when a candidate comparison is ready, intervene when evidence
conflicts or the qualified catalog cannot proceed. A slow review blocks only work
that consumes that review. Routine choices and known computations stay in Jev
and code respectively.

## One catalog and owner

```python
from modeling_system.cooperation import compose_catalogs

catalog = compose_catalogs(candidate_pipeline, diagnostic_methods, preparation_methods)
# Supply this catalog to the existing OperatingSession and normal provider ledger.
```

Each catalog receives current state and completed results. Keep task IDs unique,
source identities pinned and actual prerequisites declared. Tasks that need an
unreviewed candidate to be accepted must name `visual_review`; immutable analysis
of that candidate can proceed without acceptance. Same-region native work that
depends on source choice must wait. Composition neither clones Blender writers
nor permits activity without a concrete question. Reuse completed evidence.

Useful Jev work includes choosing a discriminating saved-array diagnostic,
qualified fitting method and typed parameters, comparing known causes with
recorded findings, retrieving applicable failures, selecting exact matched review
evidence and routing a failed prerequisite to an offered remedy. Native work uses
the same qualified adapter. Jev cannot generate a missing mathematical mechanism
or interpret images through a text-only judgment interface: Astra supplies that
capability or actual visual interpretation, then returns its findings to the loop.

Lane choices share one TypeSafe batch. If Jev's preferred lane cannot proceed,
the selector can use another actual conditional lane choice from that batch,
ordered by its priority distribution. Each alternative must pass its own method,
argument and grounded-claim checks. Deferrals remain in the decision trace.
An explicit global deferral still stops selection; no retry, confidence threshold
or silent substitute action is invented. Changed evidence invalidates old advice.

## Review while useful work continues

```python
session.run(
    max_steps=authorized_operation_bound,
    feedback_timeout=supervised_wait_seconds,
    on_handoff=publish_private_handoff,
    on_status=owner_status_callback,
)
```

Normal `run(max_steps=...)` also publishes the private `handoff.json` and controller
status. They contain the exact candidate review basis, existing evidence links,
question, tasks awaiting that review and other ready work. The callback should
notify promptly rather than block execution for reasoning. In a supervised worker
process, Astra reads the inbox, opens its actual evidence, and submits the existing
`record_review` while Jev continues unrelated operations. There is no new review
format or second queue. Feedback enters at fresh decision boundaries.

The optional finite `feedback_timeout` bounds total idle waiting for visual review
during this invocation. The default is zero. When no useful work remains, waiting
watches review-file metadata only, without provider calls, native inspection or
repeated context construction. A changed mailbox wakes normal validation; it does
not itself approve an operation. Rejected/unresolved reviews return a reasoning
handoff instead of waiting indefinitely for another review. Stale sources and
uncertain effects retain their ordinary refusal/reconciliation behavior.

Use `request_stop(session_directory, reason=...)` to request a stop at the next
operation boundary, including during an idle review wait. A running effect is not
silently killed or replayed. The request binds this invocation and does not stop a
later separately authorized one. A caller can also supply a cancellation Event.
Keep the process supervised and join or stop it before ending the turn. This API
does not create an automation, scheduler, extra agent or spending authorization.

`cooperative-run.json` reports actual waiting time and operations completed while
review was pending. Those counts demonstrate overlap opportunity, not avoided
Astra tokens, user acceptance or an end-to-end speedup. Session timing and original
operation/provider receipts remain the evidence for performance claims.
