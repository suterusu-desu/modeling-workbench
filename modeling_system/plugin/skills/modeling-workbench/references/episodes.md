# Recoverable decision operations

All signatures are discoverable with `runtime_status()['signatures']`. Each public effectful Python method captures exact arguments and available file inputs before execution. `run_episode_operation` adds the episode relation. Direct legacy calls remain usable and are recorded, but have no episode association unless called through that facade. Raw `wb`, `references`, `ledger`, and native bridge internals are not the public operator surface and are not instrumented.

Use the separately installed package interpreter and an explicitly bound project directory. Replace PROJECT with the absolute private workspace, not an arbitrary worktree missing its evidence. See the workspace setup guide for creating a new environment and character binding.

```python
from modeling_system.service import ModelingService
work = ModelingService(workspace=r"C:/absolute/project")
print(work.runtime_status())
print(work.decision_workspace())  # read-only, lists existing episodes
```

The equivalent CLI is `python -B -m modeling_system --workspace PROJECT decision_workspace`. Supply exact structured arguments with `--input arguments.json`. The saved project's existing `modeling-workspace.json` resolves character, authority, native adapter, evidence and production store. Isolated tests supply `--store TEMP_STORE`; do not use another store to evade a production job identity.

Create an episode over an actual recorded question, not a guessed live state:

```python
episode = work.open_episode(
    question=question_record, owner="actual responsible owner", scope="current authorized question",
    idempotency_key="stable-decision-name",
    context={
      "stage": "guide_qualification", "feature": "actual feature", "mechanism": "mechanism being examined",
      "requirements": [{"id":"r1", "category":"obligation", "text":"actual controlling requirement",
                        "source":"exact instruction or evidence link", "scope":"affected region/phase"}],
      "hypotheses": [{"id":"h1", "explanation":"proposed causal interpretation", "prediction":"what should be observed",
                      "discriminating_observation":"cheap observation that could reject it"}],
      "uncertainties": ["explicit missing target or unresolved judgment"],
      "links": [{"kind":"workflow", "id":existing_job, "role":"actual delivered comparison arm"},
                {"kind":"record", "id":closed_observation, "role":"matching closed pose, distinct from historical question"},
                {"kind":"file", "path":"relative/or/absolute/RESULT.md", "role":"actual owner findings and exclusions"}]})
```

Requirement categories are `obligation`, `soft_goal`, `provisional_constraint`, `unresolved_choice`. A temporary fixed rim/XZ/volume assumption stays provisional. Later repairs may replace demonstrated bad geometry while preserving evidence. Do not invent acceptance or attach old settings as current authority.

`decision_workspace(episode, detail='summary')` joins current workflow dispositions, qualified support/exclusions, current authority changes, authored requirements/hypotheses/uncertainties, pending effects and exact judgment. This is historical and does not contact Blender. List sections give total/shown/deferred counts, selection basis, source revision and coverage. Large link catalogs, completed history, prompts and native trial payloads are deferred. `incomplete_decision_coverage=true` means indispensable context exceeded the 24,000-character summary budget: follow the required reads before deciding; never infer a missing effect was resolved. `snapshot_workspace` deliberately saves a projection; `since=snapshot_id` compares sections. Job completion is distinct from fitting qualification and appearance.

Expansion descriptors contain public `operation` and `arguments`; execute them directly. `read_record(record, path=[])` bounds the immutable root; a path such as `['data','native_trial']` selects retained fields before transfer. Array indices are nonnegative; integer selectors on objects use sorted key order. `offset`/`limit` select slots (at most 64 for containers, 1024 code points for strings); `max_chars` is 2048-12000, default 8000, measured as compact JSON UTF-16 characters. Oversized values provide another bounded descriptor. `shown` counts slots, so inspect `content_complete` and deferred-value descriptors before assuming nested data is complete. `deferred` counts all slots outside this page; `remaining_after_window` counts only subsequent slots.

Use `inspect_operations(episode=episode_id, view='index')` for paged history; `selection='unresolved'` retains active/unindexed/unknown rows. Exact `handle` plus `view='header'`, `'intent'`, `'result'`, `'effects'` or `'resolution'` works without scanning all calls, including an absent result. Lease-only recovery uses `view='leases'` then its exact `view='lease'` descriptor. Index ordering is stable by handle. Returned continuations include `expected_view`; a changed index returns `changed_read_view` with no mixed rows and a refresh descriptor. `decision_workspace(detail='section', section=...)` bounds the named composition. The no-episode default is a paged episode index. Complete legacy reads remain explicit escape hatches: omit `path` for full `read_record`, use `inspect_operations(view='full')`, or workspace `detail='links'/'full'`.

For MCP output, consume one representation:

```javascript
const response = await tools.mcp__modeling_workbench__decision_workspace({episode: episodeId});
const value = response.structuredContent ?? JSON.parse(response.content.find(c => c.type === 'text').text);
text(value);
```

Use the actual discovered tool name. The SDK's text fallback remains available for clients without structured results; do not emit both representations or retrieve a full record/history merely to select a field locally.

Execute an already supported operation with automatic linkage:

```python
result = work.run_episode_operation(episode['episode'], "query_geometry", arguments)
```

MCP/CLI and direct public Python retain the same intent/result facts. An operation has a durable `operation_handle` and `operation_result_path` before later indexing. If the result is returned but indexing fails, recover it using `reconcile_operation(handle)`. If no durable result exists, inspect actual native/provider receipts and supply observed outcome/evidence. Never repeat an uncertain effect merely because a lesson or episode pointer is missing.

Use `revise_episode` with its latest revision for interpretation changes. Arrays/records/jobs retain their own revisions. `reconcile_episode` records separate `character` and `method` objects, each with supported/rejected/unresolved status and reason. Either may remain `None` (pending). Saving and recovery do not depend on this judgment. Closure requires reconciled effects, explicit judgments and the method's [integration disposition](method-integration.md). The character may honestly remain unresolved; closure never supplies user appearance approval.

Episode dispatch validates the callable and argument binding before acquiring a lease. An invalid call returns a typed argument-binding refusal through `execute`/CLI/MCP, with `lease_acquired=false`; no native/provider effect was dispatched. Direct Python raises the corresponding `PreconditionRefusal`. Correct the arguments rather than reconciling a nonexistent call. Once invocation begins, a failed result remains effect-uncertain unless actual typed evidence establishes otherwise.

For an old terminal lease with a null call handle, `inspect_operations(episode=..., view="leases")` and the exact `view="lease"` expose `lease_revision`. The supported legacy recovery is deliberately narrow: only `reconcile_operation` rejected at Python argument binding, with its original immutable `operation` refusal record and terminal completion marker. Call `reconcile_operation(handle=lease_id, episode=episode_id, expected_lease=lease_revision, observed={"effect_status":"confirmed_not_applied", "lease_id":lease_id, "refusal_record":original_operation_record, "basis":"Exact association established from the original saved return and lease evidence"}, evidence_paths=[original_evidence_path])`. The service verifies the refusal, arguments, marker, absence of a linked callable intent and unchanged lease; the operator must establish the exact original receipt association. Missing handles, dead processes and later good Blender states alone are insufficient. Active, linked, mismatched or ambiguous leases remain blocked. No native/provider work is repeated and original records are preserved. Existing operation handles continue to use the normal reconciliation route.

Compact `context.links` rows are explicitly marked display projections; a large catalog uses a non-list deferred marker so it cannot silently replace canonical links with an empty list. Follow `context.expand_links` for exact rows. Append evidence with `revise_episode(episode, revision, {'add_links':[{'kind':'file','path':path,'role':role}]})`; existing pinned links are preserved without re-reading original files. Remove selected links with `{'remove_links':[summary_row['link_id']]}` against the same expected revision; for expanded canonical rows, `link_id` is SHA-256 of the sorted compact canonical JSON, as defined by `episodes.link_id`. These patches may be combined. A stale revision is refused. Complete `{'links': full_links}` replacement remains supported using `detail='links'` or `'full'`, but cannot be mixed with additions/removals. A summary list is refused before the episode update, with a typed validation stage and guidance; the failed operation's input/journal evidence may already have been saved. No native or provider effect occurs in this operation.

Historical anatomy/source evidence must use immutable bytes when its original file is an evolving experiment log. The high-level `integrate_guide(..., anatomy_basis_paths=[...])` already copies these bytes into the store before native dispatch. For an advanced `native_integrate_guide` qualification, prepare the historical input with `revise_episode(..., {'add_links':[{'kind':'file','path':source_path,'sha256':reviewed_sha,'role':'historical anatomy evidence at qualification'}]})`. Read the exact full link via `decision_workspace(..., detail='links')`; its `asset` preserves the original `source`, immutable relative `path`, SHA and byte count. Resolve that path under `runtime_status()['store']`, verify its bytes equal the reviewed SHA, and place this immutable path/SHA into `qualification.provenance.anatomy_basis`. Keep the episode link as original-source lineage. Changed source bytes are refused when the reviewed hash is supplied. A later source append does not change the pinned historical evidence.

This preparation freezes historical meaning only. Current native topology, registration, controls/pose, guide geometry, applicable support and measurement dependencies must still pass their existing freshness/qualification checks. Identical bytes at another locator do not establish ongoing applicability or a new qualification. There is no registry-rebinding shortcut: do not bypass `_apply_proposal`'s registry-mutation refusal or edit a live guide registry to repair provenance. Existing guide recovery remains with the native owner and its guarded transaction.

For semantic reasoning use `record_dependencies`, `register_semantic_graph`, `validate_semantics`, then `semantic_impact` forward/reverse/both. Bind roles and selectors explicitly to recorded topology and construction with source evidence and missing coverage. An invalid semantic mapping stays invalid even when coordinates are fresh. Use supported native feature inspection through the sole owner to establish or repair meaning.

Operation leases distinguish active owner work from interrupted effects. Another operator must not reconcile an active/in-flight call. Closure reserves the same episode lock as invocation start; active or unknown effects block closure. Raw native returns are durably stored below the parent call's `effects/` before trial/episode bookkeeping, so a later ledger failure cannot erase a successful native receipt. An indexing repair of a timeout does not establish whether it applied. Supply actual outcome evidence with explicit `effect_status` only after inspecting the original effect; do not replay it. Omitted character/method components preserve prior judgment. Use `clear_components` to explicitly clear a component.

A failed lease-finalization write must not mask a durable result. The returned handle and independent completion receipt permit metadata repair with `reconcile_operation`; context resets even if bookkeeping fails. Running owners and unresolved native effects remain distinct.

In versions through 0.2.6, appending the destination name to an atomic temporary filename could exceed Windows path limits even when the final completion path was valid. Version 0.2.7 uses a separate short sibling filename. If a returned result carries `lease_retention_status="needs finalization repair"`, inspect its exact handle and original result. Use the fixed installed `reconcile_operation(handle)` directly. For a durable returned result with no unknown nested effect, it recreates a missing completion receipt before finalizing that lease; `completion_repair` reports the receipt and disposition. Repeating this metadata repair preserves the original intent/result and does not repeat the analysis or native/provider effect. Preserve the old error-bearing receipts as history. Unknown effects still require their existing evidence-based reconciliation. A closed episode can legitimately have finalization pending when its durable result already establishes completion; closure alone does not prove that every completion file was written.

`apply_trial` now emits a typed `apply_trial.baseline_precondition` refusal if its baseline guard fails after returned inspection, before proposal dispatch. This known pre-dispatch disposition needs a fresh owner read and explicit rebase, not blind replay or a claim of an uncertain geometry mutation. Existing legacy records retain their original wording; reconcile them from actual receipts and control-flow evidence. Any after-dispatch timeout remains unknown. Concurrent authority changes are legitimate freshness invalidation, even when native content matches.
