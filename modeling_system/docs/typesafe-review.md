# TypeSafe documentation and workbench review

Reviewed 2026-09-21 against workbench 0.2.26, source commit
`1b484cfeab13507778a2e468e099e123c577800e`.

The useful next step is to let Jev construct more of the executable decision:
select the applicable mechanism, bind qualified arguments, and choose evidence
that resolves a named uncertainty. The existing episode, queue, native lane and
retained learning provide the execution foundation. This document records an
audit and implementation targets; it does not claim those extensions are shipped.

Follow-through: 0.2.27 implements D1-D7 through
[typed capability control](../plugin/skills/modeling-workbench/references/typed-control.md).
The baseline findings below remain a dated audit. See the
[implementation record](../../IMPLEMENTATION-QUEUE.md) for current verification;
source implementation, installed delivery and real modeling use remain distinct.

## Coverage and evidence

The [official index](https://docs.typesafe.ai/llms.txt) contained 110 pages.
All were retrieved into a private, timestamped source snapshot with per-page
hashes. The review covered the conceptual and prompting guidance, primitives,
patterns, all 18 cookbooks, model limitations, HTTP contract, and Python and
JavaScript SDK references and changelogs. Relevant example code was inspected;
the examples were not executed. Repetitive generated reference scaffolding is
not an additional behavioral claim. The legal index was read; its linked legal
agreements were outside this technical review.

Source inspection was checked against 37 passing judgment, transport, queue and
learning tests. Two additional local preflight checks reproduced the question
count and payload refusals below. There were no provider requests or native
operations. These checks establish current software behavior, not modeling
quality, calibration, or saved Astra effort. Raw vendor pages and private
workspace evidence are not distributed with this repository.

## Existing foundation

| Implemented in 0.2.26 | Relevant source | Boundary |
| --- | --- | --- |
| Choice, Score and Noul; structured criteria; complete answer validation | [judgments](../judgments.py) | Typed validity does not establish truth |
| Pinned direct v1 transport, actual usage, durable request binding | [transport](../typesafe_transport.py) | No hidden retries; unknown attempts retain their accounting |
| Batched next-lane and conditional task choices; exact answer reuse | [queue](../work_queue.py) | Questions in a batch cannot consume one another's answers |
| Reusable methods and implementation/parameter bindings | [catalog](../method_catalog.py) | Parameters are currently bound before Jev chooses |
| Fresh scoped findings and visual feedback in one operating session | [session](../operating_session.py) | Astra supplies actual visual interpretation; code reports native facts |
| Automatic review/lesson retention and projected experience retrieval | [learning](../learning.py), [context](../retained_context.py) | Retrieval and retained lessons are not trained prediction or calibration |

## Open implementation targets

These extend the existing control path. They do not reopen the completed
integration gate or require a modeling pause, benchmark campaign, new executor,
or SDK migration. Implement and use each useful slice through the current owner.
Current workspace authority and explicit resource limits remain binding.

### D1. Compose operations and their arguments

`ParameterizedCatalog` removes copied factories, but its bindings already contain
the complete parameter dictionary. A richer menu alone still leaves recurring
assembly to Astra. Add a versioned capability specification declaring semantic
argument roles, eligible values, dependencies, defaults and compatibility rules.
Jev selects a mechanism and its applicable arguments; code resolves local IDs,
validates the whole tuple and compiles the ordinary queue task.

Use conditional questions for method, region, source role, correspondence and
evidence view when their candidate sets are already known. Consume only the
chosen branch. Optional omission must preserve a declared default; missing
required support must remain unknown. Compute coordinates, counts and geometric
fits in code. Do not manufacture open-ended numbers or correspondence through
long chains of choices. See [function calling](https://docs.typesafe.ai/cookbooks/function_calling)
and [value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook).

**Completion evidence:** an ordinary session constructs distinct useful calls
without Astra prebinding each complete variant; omitted, incompatible, stale and
unsupported bindings are handled before effects. The existing journal, recovery,
single native writer and qualified adapter remain the only execution route.

### D2. Make decision budgets explicit and consistent

Current local limits are 12 questions, 11,000 encoded bytes for the prepared
state/questions, and 12,000 bytes for the transport request. Queue relevance
questions use whatever remains of those 12 slots. Retained context permits four
whole passages within 2,400 bytes; method catalogs allow 32 methods and at most
128 generated tasks. These are workbench policies, not documented Jev limits.

The current [model contract](https://docs.typesafe.ai/models) instead documents
64k tokens for state plus all questions and 32k for state plus the longest
question. Byte counts are not token counts. Introduce one versioned decision
budget used by compilation, projection and transport; distinguish provider
constraints from deliberately smaller workspace limits. Expand useful candidate
coverage within authorized resources and report exclusions. Group questions by
shared relevant state rather than filling a large context indiscriminately.

**Completion evidence:** a useful batch exceeding the old count can pass all
layers under an explicit budget; over-budget work stops before dispatch; whole
passage caveats, cost reservations and exact cache invalidation remain intact.
Larger context is capacity, not a quality improvement by itself.

### D3. Choose observations by the uncertainty they resolve

Existing evidence choices need a richer reusable input contract: competing
hypotheses, observed facts, actual visual judgments, missing support, available
observations, and what each possible result would change. Jev can choose the
next discriminating observation or decide that existing evidence suffices.
Code supplies known prerequisites, cost and reusable results. A mandatory save
or fixed freshness check remains code, without another model judgment.

**Completion evidence:** a changed uncertainty produces a different useful
observation; unchanged evidence is reused; unsupported information gain is not
reported as a measured numerical quantity. Use [structured state](https://docs.typesafe.ai/concepts/state)
and [coherent narrow questions](https://docs.typesafe.ai/concepts/how-to-build-with-system-one).

### D4. Retrieve experience before using it to decide

Current context selection is primarily lexical and budgeted before Jev sees it.
The queue's passage relevance Scores share the action-selection batch, so they
cannot improve that same request's selected context. Retain this cheap route
when sufficient. Add staged retrieval when important coverage is missing:
deterministic applicability/privacy filtering, broad compact candidates, semantic
selection, then expansion of the selected exact sources before the action choice.

Keep support, contradiction, changed prerequisites and missing applicability
distinct. A relevant failure is useful evidence, not a universal prohibition.
Historical advice cannot overrule current authority. Include a no-match outcome
and validate the selected method's applicability, not the maximum fit of some
other shortlisted method. Use hierarchy only when catalog size warrants it.
See [skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion),
[reranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe) and
[passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages).

**Completion evidence:** a relevant source outside the old top-four projection
can affect the next choice, while private sources and stale authority stay
excluded. Conflicting source passages retain their qualifiers and provenance.
A second request is used only when the first supplies genuinely new context.

### D5. Check consequential claims against exact evidence

Current hashes and lineage prove which record was used. They do not establish
that a new lesson's wording is supported by that record. Add an optional narrow
support/contradiction/insufficient-evidence judgment when summarizing or promoting
a consequential method claim. First locate the exact source in code, then judge
its meaning. Preserve scope qualifications such as local improvement versus
whole-motion acceptance. See [citation checking](https://docs.typesafe.ai/cookbooks/citation_check).

**Completion evidence:** an overbroad synthetic lesson is flagged while its
supported local claim survives. A semantic check neither approves appearance nor
becomes a blanket extra call for every unchanged receipt.

### D6. Connect judgments to actual reviewed outcomes

Retain question/model versions, pre-decision state, complete distributions,
selected branch and arguments, execution result, actual review and missing
measurements as a joined record. Distinguish bad evidence, poor candidates,
incorrect interpretation, implementation defects and service failures. Improve
question wording from these normal-use failures. Reuse unchanged judgments when
changing explicit ranking weights; do not spend again merely to change policy.

Independent Scores can represent supported benefit, disruption and evidence
coverage. Normalize different rubric lengths before combining compensating
preferences; a material violation must not disappear in an average. A middle
Score can conceal disagreement between extremes. See [composite scoring](https://docs.typesafe.ai/patterns/composite-scoring).

**Completion evidence:** inspect a decision through its later review, including
unused branches and unknown Astra effort, and compare a policy revision offline.
Only accumulated actual labels can support calibration or learned ranking.
If downstream training later becomes useful, separate cases/characters/time
across evaluation splits and keep a held-out set. [Feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery)
is a possible later technique, not a requirement to train a model now.

### D7. Preserve provider recovery guidance

The direct transport records status and timing but drops response headers.
Recovery recognizes transient statuses, including 429 and 529, and uses local
backoff. Retain bounded, validated `x-typesafe-request-id`, `Retry-After` and
`retry-after-ms` fields when supplied. Honor server timing within the existing
linked-attempt and resource policy; represent a future retry time without a long
blocking wait. Preserve the original uncertain attempt and conservative debit.

**Completion evidence:** synthetic responses cover valid, malformed and excessive
retry hints, trace identity, missing responses and restart. No added hidden SDK
retry, credential/error-body logging, presumed idempotency, or native replay.
See [Python retry policy](https://docs.typesafe.ai/sdk/python/api/retries) and
[JavaScript response metadata](https://docs.typesafe.ai/sdk/javascript/api/interfaces/WithResponse).

## Question and API details that matter

- **Meaning must be explicit.** Question IDs are code keys, not model context.
  Include roles and exclusions in instructions/criteria. Choice supports up to
  255 options; Score has 2-10 ordered levels. Keep each Score level self-contained.
  A Noul measures a condition's probability, not graded intensity.
  See [HTTP contract](https://docs.typesafe.ai/api), [Score](https://docs.typesafe.ai/primitives/score)
  and [Noul](https://docs.typesafe.ai/primitives/noul).
- **Use the relevant distribution.** Choice compares the offered alternatives;
  a winner does not establish that any option fits. Noul has no separate
  confidence. Choice/Score confidence summarizes distribution concentration,
  not permission or a calibrated probability of overall workflow correctness.
  Uncertainty among several acceptable choices need not call Astra; unused
  branches should not block the selected one. [Confidence guidance](https://docs.typesafe.ai/confidence)
  requires domain-specific interpretation, not copied demonstration thresholds.
- **Batch only genuine shared-state decisions.** Conditional questions are useful
  when their premises are explicit. They cannot read sibling answers. New
  evidence, candidates or a hierarchy frontier can require another request.
  Preserve that dependency rather than implying same-batch feedback.
  See [fan-out](https://docs.typesafe.ai/patterns/fan-out).
- **Keep exact operations in code.** Jev 1.13 documents weaknesses in arithmetic,
  counting, dates, indirection, negation, long irrelevant state, adversarial text
  and assumed consistency between separate questions. Compute numeric facts,
  name direct relationships, and enforce invariants in code. Semantic injection
  detection is not an authorization boundary. See [known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).
- **Keep the transport deliberately strict.** Current model pinning and exact
  answer-set validation are useful. SDKs add conveniences, not a requirement to
  migrate. Python and JavaScript differ in timeout/retry policy; automatic retries
  must not bypass our ledger. Python's extra request body can override fields,
  so any future SDK adapter must preserve exact wire binding. Debug logging can
  still contain bodies. See [Python usage](https://docs.typesafe.ai/sdk/python/usage)
  and [JavaScript client](https://docs.typesafe.ai/sdk/javascript/api/classes/TypeSafeClient).
- **Respect modality and data boundaries.** Jev currently accepts text/JSON,
  not images or meshes. It can direct observation and use returned measurements
  and authored visual judgments; those are not direct visual perception.
  Customer fine-tuning is not offered. No-training language does not establish
  standard-account zero retention. Keep deliberate public projections and local
  raw assets. These are current [model facts](https://docs.typesafe.ai/models),
  not an independent legal assessment.

## Cookbook disposition

Each indexed cookbook was reviewed. These are mechanisms to adopt selectively,
not 18 new mandatory stages.

| Cookbook | Workbench disposition |
| --- | --- |
| [Noul consistency](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook) | Preserve raw signals and exact-result reuse. Repeatability does not prove accuracy. |
| [Choice consistency](https://docs.typesafe.ai/cookbooks/consistency_choice_cookbook) | Keep defer/no-match. Avoid repeating unchanged requests to obtain a preferred answer. |
| [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions) | Already used; D2 removes accidental limits. Published latency comparison uses serial single requests. |
| [Reranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe) | D4: semantic selection after a broad cheap shortlist; omitted candidates cannot be recovered. |
| [Semantic find](https://docs.typesafe.ai/cookbooks/semantic_find) | D4/D5: select exact source spans and separately allow absence of an answer. |
| [Structure recovery](https://docs.typesafe.ai/cookbooks/autoformat) | Optional evidence-brief assembly from exact spans; preserve source wording and caveats. |
| [Function calling](https://docs.typesafe.ai/cookbooks/function_calling) | D1: typed operation and argument selection with explicit omission/default semantics. |
| [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion) | D4: compact discovery, expand a few, then verify selected applicability. |
| [Entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment) | Conditional metadata role matching; never infer geometric or anatomical correspondence from semantic similarity. |
| [RAG passage classification](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) | D4: distinguish relevant support, contradiction and missing information. |
| [Citation checking](https://docs.typesafe.ai/cookbooks/citation_check) | D5: source existence plus semantic support for consequential claims. |
| [LLM guardrails](https://docs.typesafe.ai/cookbooks/llm_guardrails) | Use only named semantic checks where useful. Deterministic authority/privacy boundaries remain primary. |
| [SDE cascade](https://docs.typesafe.ai/cookbooks/sde_cascade) | Localize a failing field and escalate that gap; no automatic additional paid-model chain. |
| [Date extraction](https://docs.typesafe.ai/cookbooks/date_extraction_cookbook) | Reuse decomposition principle. Dates, ordering and arithmetic already belong in code. |
| [Pre-parsed extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook) | D1/D4: find candidates in code, select semantic roles, copy exact values. |
| [Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification) | D4 when needed: retain plausible branches; aggregate path scores are heuristics, not proven posteriors. |
| [Feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery) | D6 first: collect actual reviewed outcomes. Optional downstream learning follows sufficient evidence. |
| [Classification with confidence](https://docs.typesafe.ai/cookbooks/classification_using_confidence) | Return a supported broader category when exact classification is unresolved; retain uncertainty. |

Several examples use cached older model outputs, external helpers, small synthetic
datasets or domain-specific thresholds. Their mechanisms inform this design;
their reported results do not establish Blender performance. Recheck the current
contract when changing models or transport, and verify improvements through
ordinary useful work. The [design](../../DESIGN.md) and
[implementation queue](../../IMPLEMENTATION-QUEUE.md) track that work separately
from this review snapshot.
