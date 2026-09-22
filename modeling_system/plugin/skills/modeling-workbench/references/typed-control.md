# Typed capability control

Use the existing `OperatingSession`, queue, qualified handlers and provider ledger.
Jev can select an operation and its arguments in one conditional batch. Astra
qualifies reusable mechanisms and reviews actual appearance; it need not prebind
every complete operation variant or write a new controller for each candidate.

## Declare the meaningful choices

An ordinary task may add `decision.arguments`. Each argument supplies a public
semantic question, a path **inside the handler payload**, and closed-set options.
Option `value`, `id` and dependency `reads` remain local. Only deliberately public
questions and descriptions enter the provider request. Include every candidate's
dependency in the task's reads; options do not weaken profile or native guards.

```python
task['decision'] = {'arguments': {
    'region': {
        'question': 'Which supported region contains the dominant observed defect?',
        'path': ['parameters', 'region'],
        'options': [
            {'id': 'first', 'description': 'The supported region with a measured join fold',
             'value': first_region_token, 'reads': {'source': source_revision}},
            {'id': 'second', 'description': 'The supported region with an unresolved closing gap',
             'value': second_region_token, 'reads': {'source': source_revision}},
        ],
    },
    'method': {
        'question': 'Which qualified construction mechanism addresses the current evidence?',
        'path': ['parameters', 'method'],
        'options': qualified_method_options,
    },
}}
```

The existing handler receives the selected exact values in `payload.parameters`.
Register the handler once; reuse it with fresh source/guide/pose tokens. The
capability supplies the geometry algorithm and numerical derivation. Jev does
not generate coordinates, source paths, correspondence or executable code.

`OperatingSession`/`WorkQueue` recognize these declarations directly.
`CapabilityCatalog(existing_catalog)` additionally validates them at catalog
construction. The original template remains the queue definition; the compiled
payload and exact selected-call proof enter the same episode operation. Dynamic
service arguments are signature-checked after binding, before their effects.

Choices include explicit unknown support. Optional arguments can declare a local
`default`; Jev may choose omission, preserving that default rather than guessing.
For a set, use `type='set'`: independent Noul judgments select offered members.
An exact tie is unresolved; empty sets require `allow_empty=True`. No automatic
confidence threshold is introduced. Unknown unused branches do not block the
selected operation. Use `decision.incompatible`, a list of partial argument-ID
dictionaries, to reject prohibited tuples before dispatch. These rules supplement
the existing deterministic qualification; they never grant authority.

Keep current result tokens in normal `CandidatePipeline` predecessor results.
A later stage can use those exact outputs in another typed specification or a
fixed already-selected continuation. Do not re-run preparation to reconstruct a
token that already exists, or turn a new method choice into required housekeeping.

## Choose useful observations and check claims

`decision.observation = observation_contract(...)` adds the named question,
competing hypotheses, the edit each result would change, known cost and existing
evidence to the operation choice. Supply those facts from current qualified
evidence. Unknown cost stays unknown. This is the place to explain why a saved
array comparison suffices or a particular matched view is discriminating; it
does not create a mandatory render or another inference for a fixed check.

For a consequential method claim, use
`grounded_claim(claim, source_text=exact_source, public_excerpt=exact_span)` in
`decision.claims`, keyed by a local name. Code verifies the exact span before
inference; Jev judges supported, contradicted or insufficient. The complete source
and its identity stay local. Explicitly review the excerpt for public projection.
Only the selected operation's claim checks affect its release. Unsupported claims
return a localized unresolved result without native effects. This optional check
does not approve appearance or delay ordinary checkpoint saving.

## Retrieve relevant lessons before choosing an action

`RetainedContext` still projects eligible records before limiting them and retains
whole passages with their caveats. When the broader candidate pool exceeds the
initial context, `LaneSelector` first asks independent relevance and relationship
questions. Packing admits the best fitting whole passage of each applicable
relationship (supporting, contradicting or conditional), in relevance order,
then fills remaining space by relevance. An oversized passage is skipped whole;
a smaller passage of the same relationship can still fit. Neither a large
failure archive nor many supporting reviews get unconditional priority.
A subsequent action request receives these sources in relevance order.

The passage and byte limits stay unchanged. With too little room, some sides
can remain unrepresented: `semantic_coverage.coverage_by_relationship` reports
candidate, selected and omitted counts for every relationship. `excluded` names
budget-omitted eligible passages and the binding passage or byte limit;
unrelated candidates stay out and are counted separately. Complete source
caveats and raw judgments remain intact. This is bounded greedy coverage, not
an exhaustive source search or an override of Jev's action/defer choice.

This second request supplies new evidence; sibling questions cannot consume one
another's answers. Small sufficient context continues through one batch. The
retrieval and final choice each have exact dependency-bound caches. A changed
selection policy also invalidates both caches without rewriting old decisions.
No passage's relevance changes current guide authority.
Historical failures inform mechanisms without universally banning repaired uses.

## One local decision budget

`OperatingSession(..., budget=DecisionBudget(...))` and `LaneSelector` share the
same versioned budget. A standalone judgment may supply its record through
`binding['decision_budget']`. `RetainedContext` and method catalogs also accept
that budget when customized. Compiler and direct transport enforce the same
limits before dispatch. All spending/request authority remains in the existing
workspace ledger.

Defaults are 96 questions, 60,000 encoded request bytes, 30,000 bytes for state
plus the longest question, eight whole context passages within 8,000 bytes, and
up to 32 retrieval candidates within 16,000 bytes. Method and task limits are
128 each. These are local policies, not provider question-count or token limits.
The conservative byte envelope does not report an invented exact token count.
Narrow irrelevant state or adjust explicit limits within the supported envelope;
do not silently truncate a caveat or increase a spending ceiling.

## Outcomes, recovery and learning

Every completed selector result retains its policy/question revision, exact
packet, full answers, applicable cached branches, selected arguments, claim checks
and retrieval evidence under `advice/decisions/`. Observed model/usage metadata is
retained when the workspace judge returns it; missing metadata stays unknown.
The actual episode operation binds that trace. `inspect_operating_session` joins
it to execution and the latest actual scoped review. Rejections and unknown Astra
effort remain visible; no training, calibration or causal savings are inferred.

`decision_outcomes.composite_scores` normalizes each retained Score by its rubric
length and combines explicit nonnegative weights without another provider call.
Keep material violations separate from compensating preferences. Full distributions
remain available: a middle mean need not mean a certain middle outcome. Missing
dimensions stay unknown. Use actual reviewed cases to improve questions before
considering downstream training or threshold tuning.

An interrupted retrieval uses the same saved-packet recovery as an action choice.
Recovering that stage records its known answer without another request; the next
ordinary controller run uses it to select the operation. Unknown attempts remain
under their original ledger identity. No recovered choice directly touches Blender.

Direct transport retains only validated provider request ID and retry timing
headers. Linked transient recovery preserves `not_before` and the original
reservation/debit. `retry_ready` refuses an early dispatch. A long delay raises
`ProviderRetryDeferred` with its eligibility time; retain the stopped state and
resume within existing authority later. Do not implement a long blocking sleep,
hidden SDK retry, automatic wakeup, or reset of the spend ledger.
