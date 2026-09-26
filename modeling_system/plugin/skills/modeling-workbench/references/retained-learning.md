# Learning that reaches the next decision

OperatingSession retains each actual visual review in the workspace's existing
content-addressed store. Later scopes retrieve relevant reviews automatically;
they need no copied history file or custom projector. Exact receipt identity,
candidate basis and evidence stay private. Only the deliberately public judgment
text already used in session feedback enters the owner's observed context.

Begin actual review with `session.begin_review(task)`. Submit the usual scoped
judgment and evidence with `session.record_review`. Its `useful` disposition
means a local visual gain, never user acceptance, current selection or a general
method benefit. A rejected candidate remains a useful counterexample. Review
does not wait for a lesson or completion of the whole character.

When review establishes an actionable mechanism or revises an interpretation,
add a compact `lesson` to the same submission:

```python
session.record_review(task, expected_basis=opened["basis"],
    judgment=actual_visual_judgment, evidence=actual_evidence,
    lesson={
        "mechanism": "Shared guide-material motion",
        "observation": "A previously poor timing result improved after the opposing surface fit changed.",
        "conditions": {"opposing_fit": "whole_surface"},
        "next_use": "Compare the spatial prerequisite before repeating or discarding this timing method.",
        "limits": "Scoped comparison only; other joins and continuous motion remain unresolved.",
    })
```

All lesson text, including conditions, is explicitly authored for the decision
provider. Keep names, paths, account information and private evidence in typed
links. The lesson has a 1600-byte budget; pin extended reasoning and counterexamples
as evidence. For a lesson spanning actual cases, use
`modeling_system.learning.record_lesson(service, lesson=..., evidence=...)`.
It returns an immutable record ID; exact repeated submissions are idempotent.

Conditions describe the observation's circumstances, not universal requirements.
Current context can match, differ or lack those facts; unknown remains unknown.
This comparison does not disqualify an executable method or turn a past failure
into a universal ban. A changed prerequisite can justify reconsideration, but
code does not invent that conclusion. Retain the failure and the new comparison.

Useful distinctions to preserve when actual work supports them:

- Exact landmarks do not establish the intervening surface's guide fit.
- Targets with real guide-triangle ancestry can still have incoherent material
  correspondence. Compare the connected representation and actual appearance.
- Nearest spatial pairs do not establish anatomical attachment.
- A raw carrier target, the support blend and final evaluated positions are
  different objects of analysis. Name which one an observation measures.
- Technical correctness and a local gain can coexist with an unresolved dominant
  visible defect. Carry that next question forward.

These are interpretation aids, not mandatory new tests before every edit. Reuse
observations that already answer the next question.

## Retrieval and migration

`retrieve_experience(query)` returns authority passages first, then matching
records. Each retained review or lesson match leads with `public`: the scoped
review judgment, the lesson and a condition comparison against the supplied
context, exactly the deliberately public projection used in session feedback.
`evidence_count` follows, then the complete record in `excerpt` with its pinned
evidence and private provenance. Read `public` to decide relevance; expand the
record only when its evidence matters.

Authority passages are ordered by source priority, so a highly relevant passage
in a lower-priority bound source (for example a project record of a rejected
method) can fall past the limit. Before a native trial, query with the task's
stated problem and mechanism using `order="relevance"`, which ranks authority by
matched terms with priority as the tie-breaker, and read the top passages and
matching lessons. A read-only copy of another owner's authority can be bound as a
lower-priority source for exactly this recall.

The default query uses the current public situation and offered work descriptions.
An explicit `RetainedContext` can add source passages or custom queries. Dedicated
review/lesson records remain available even if its old projector accepts only a
particular history file. Conditions use `public_state`, or `context(state)`.

Projection precedes candidate limiting, so private records cannot crowd out
usable history. The shared `DecisionBudget` defaults to eight complete context
passages within 8000 serialized bytes, with up to 32 candidates within 16000
bytes available for semantic selection. Coverage distinguishes excluded,
duplicate, unreturned and size-omitted passages. Caveats are never clipped to fit
another passage. Use
`retrieve_experience` or exact retained records to expand missing coverage.

When all candidates fit the initial context, the passages and independent
relevance questions share the action-selection request. Otherwise one cached
evidence-selection request precedes action selection. It packs relevant support,
counterexamples and conditional evidence together, then fills by relevance;
neither successes nor failures can monopolize context just by their relationship.
Whole-passage byte limits can still leave gaps, which are reported by relationship
in the action context. Exact dependency or selection-policy changes invalidate
both stages. The `retained_context` module documents configuration,
coverage and interruption recovery. Relevance is not guide admission, appearance
approval or proof that a mechanism caused improvement.

Adopt old actual reviews once with:

```python
from modeling_system.learning import import_review_history
receipt = import_review_history(service, old_session_directories)
```

The importer verifies original episode facts and leaves old scopes, queues and
receipts unchanged. No native or provider call occurs. Repeating it returns the
same record IDs. Session startup repairs its own interrupted review index.
Corrected interpretations supersede only the same exact review basis in
retrieval; earlier immutable facts remain available. No historical candidate is
reopened, rerun or silently selected.

Formal method promotion remains the separate evidence-backed process in
[method integration](method-integration.md). Automatic recall is not promotion.
