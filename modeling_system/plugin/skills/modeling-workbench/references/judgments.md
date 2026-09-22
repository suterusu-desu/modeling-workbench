# Offload recurring planning judgments

Jev can supply recurring planning judgments as well as action choices. The
reasoning owner establishes objectives, novel mechanisms and visual interpretation.
Use reusable state projections and decision specifications instead of writing a
new menu/controller for every candidate. The generic `modeling_system.judgments`
module compiles and validates decisions; the workspace supplies its authorized
transport, accounting and execution binding. No provider credentials are packaged.

## Useful division of work

| Repeated judgment | Jev input and output | What remains in code or with the reasoning owner |
| --- | --- | --- |
| Next issue | Reviewed observations and offered issues -> priority Choice | Current user scope, measured severity and dependency eligibility filter |
| Retained method | Mechanism evidence, applicable methods and failure conditions -> method Choice | Retrieve exact procedure; qualify support and execute its existing implementation |
| Next observation | Explicit uncertainty, available evidence and capture options -> evidence Choice | Reuse valid evidence; compute exact costs/geometry; do required checks directly |
| Failure response | Settled/uncertain effects and known recovery methods -> recovery Choice | Journal reconciliation and recovery authorization; never retry unknown effects |
| Reasoning route | Available capabilities and missing information -> handler Choice | Return unresolved visual judgment, new mechanisms and contradictory context to the reasoning owner |
| Lesson relevance | A specific recorded lesson and current mechanism -> Noul | No universal probability threshold; caller's reviewed policy decides use |
| Work priority dimensions | Separate descriptions of impact, applicability or information value -> Scores | Combine specified weights in code; scores do not measure mesh displacement or likeness |

Ask only questions that change current work. Fixed prerequisites, arithmetic,
authorization and geometric validity stay deterministic. Batching independent
questions reduces round trips; it does not let one question read another answer.
For dependent choices either ask conditional branches and consume the applicable
ones, or use a necessary second request with fresh relevant evidence.

## Callable contract

```python
from modeling_system.judgments import (
    planning_choice, prepare_judgments, resolve_judgments,
)

decisions = [
    planning_choice('method', applicable_public_method_descriptions),
    planning_choice('evidence', available_public_evidence_options),
    {'id': 'lesson_relevance', 'type': 'noul',
     'instructions': 'Does the supplied lesson address the observed correspondence failure?'},
]
packet = prepare_judgments(reviewed_public_state, decisions, current_binding)
# Existing authorized transport sends only state and questions plus its pinned model.
# It persists the packet/response, enforces budget and validates provider identity.
result = resolve_judgments(packet, validated_provider_answers, fresh_binding)
```

The optional binding field `decision_budget` is a `DecisionBudget.record()`.
The same versioned limits apply to compilation, wire validation and response
validation. Defaults allow 96 useful questions, a 60000-byte request and a
30000-byte state plus longest question. These are conservative local byte
envelopes, not tokenizer measurements or spending authority. See
[typed control](typed-control.md) for conditional argument composition, staged
retrieval, grounded claims and decision-to-outcome inspection.

Candidate rows have `id` (local only) and `description` (reviewed public string or
structured content). `planning_choice` supports `priority`, `method`, `evidence`,
`recovery` and `route`, adding an explicit `needs_astra` alternative. The fixed
name identifies escalation to the reasoning owner; it does not start another
agent. A planning choice is a recommendation within already-authorized scope.
Do not ask Jev to invent candidates that are absent from the available operation
catalog or infer missing anatomy. Retrieve candidate methods from retained
experience and filter their known prerequisites in code before ranking.

Custom decisions use `id`, `type`, `instructions` and either `options` (Choice)
or `criteria` (Score/Noul). Score criteria are 2-10 concrete ordered situations;
Noul optionally supplies both `true` and `false` descriptions. Instructions and
criteria can be structured JSON. Each question must say which state fields and
region it concerns: question IDs themselves are not model context.

Bindings contain `owner`, `authority_revision`, and `dependencies` with exact
nonempty `reads` and covered `writes`. Include every fact that can change the
judgment, plus current objective/menu applicability. `resolve_judgments` refuses
changed bindings and validates full distributions, score legends/expectations
and finite numeric ranges. Native execution still checks its actual lane and
expected geometry immediately before effects. A planning result cannot itself
approve appearance, confer authority or prove a confidence value is correctness.

For an operating-session transport, call `session.judgment_context(binding)` at
each freshness boundary. It returns `binding` and `active_operations` from one
fresh observation and refuses a changed action menu. Use both returned fields
from that observation; a second context scan solely to read the lane repeats
input hashing unnecessarily. This method does not cache state across requests,
release a choice or replace the native execution guard. Compare the returned
binding with the original packet before dispatch and when consuming the result.

Successful dispatch and failure recovery are separate paths. A newly validated,
durably accounted response needs fresh release, not a rewrite through completed
response recovery. Interrupted or uncertain attempts still use their original
recovery receipts and ledger. Measure whole selection, HTTP elapsed, fresh-context
reads and local validation separately. HTTP elapsed includes client/network time;
it is not a server-only inference measurement. Use ordinary work for timings.

The reader has a narrow compatibility rule for observed Choice/Score responses whose
complete probabilities are serialized in hundredths and total 0.99 or 1.01.
It accepts at most 0.01 mass drift only when per-entry rounding intervals of
0.005 admit a unit-mass distribution. Original probabilities, confidence and the
selected maximum remain unchanged; missing options, invalid numbers, larger
drift and inconsistent higher-precision values still fail. For a Score with
rounded probability mass, the score must also be on the hundredth grid and its
rounding interval must intersect an attainable expectation of a unit-mass
distribution inside all probability intervals. Checking mass alone is insufficient.
Original scores and probabilities are never normalized or replaced. Exact legends,
answer sets and native action guards remain required; truly malformed optional
advice is not silently turned into a valid action choice.
The upstream API documents a sum of one, not a rounding guarantee;
this is an explicitly bounded reader accommodation, not upstream conformance.
Recover an already saved response through the original completed-response path
and ledger identity, without repeating inference or rewriting the raw response.

For genuinely invalid typed answers, `invalid_response.prepare_selection_retry`
routes HTTP200 responses to exact-envelope and usage validation before marking
the original attempt `invalid_answer_accounted`. It never creates a selection or
corrects the chosen option. Known reported cost is reconciled once, even if
inputs have since changed or retries are exhausted. Only a fresh, unchanged
decision receives a successor packet, with the original state and questions.
Model, wire identity, unknown usage and credit failures are not invalid-answer
retry candidates. Invalid answers and transient transport failures share two
retries for the same decision with backoff; a restart does not reset that bound.
The workspace transport loops over the returned retry packet and receipt,
checks freshness after waiting, and reuses an already dispatched successor.
An interrupted session can pass this receipt through its existing
`recover_completed_selection(..., transient_retry_path=...)` lineage reader.
Accounting or recovery alone never invokes native work.

Retain probability distributions and score levels alongside the selected value.
An arbitrary .9 confidence gate is not a calibrated Blender criterion. A known
missing prerequisite blocks in code regardless of the answer. A `needs_astra`
result returns the precise question and retained evidence; it is not a repeated
poll or an invitation to create another standing agent.

## Ordinary use and retention

Reuse a completed judgment only while its exact packet, relevant revisions and
authority remain applicable. Do not spend a second request on unchanged evidence
or renew an exhausted development ledger automatically. Keep original uncertain
requests for reconciliation. New provider capacity needs an authorized bounded
run; packaging never supplies spending authority.

Retain the authored question specification, input evidence links, answer,
consuming operation and outcome. Promote useful question/method specifications
as reusable data. Technical checks establish execution; the reasoning owner's
visual review establishes observed artistic benefit. Measure avoided reasoning
work in actual modeling rather than claiming savings from model latency alone.

Primary references: [function calling](https://docs.typesafe.ai/cookbooks/function_calling),
[skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion),
[typed API](https://docs.typesafe.ai/api), and
[known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
