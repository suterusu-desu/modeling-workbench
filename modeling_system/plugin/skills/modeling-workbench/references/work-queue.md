# Shared Jev work queue

Use `WorkQueue` and `LaneSelector` across diagnosis, repair, verification, review
preparation, recovery, experience retrieval and preparation. Keep one queue running
across useful dependent operations instead of writing a controller per candidate.
The workspace binds qualified handlers, observed revisions and its provider ledger.

```python
from modeling_system.work_queue import WorkQueue, LaneSelector
selector = LaneSelector(advice_directory, judge=adapter.judge_batch,
                        project=adapter.public_projection)
queue = WorkQueue(run_directory, owner=adapter.owner,
    goal={"objective": "Resolve the observed regional defect using accepted guides"},
    observe_context=adapter.current_context, catalog=adapter.work_catalog,
    handlers=adapter.handlers, select=selector)
result = queue.run(max_steps=12, on_status=adapter.show_status)
```

`current_context()` returns `owner`, `authority_revision`, `values` (current exact
dependency revisions), `active_operations` and optionally `public_state`. It reads
current receipts/files; live Blender inspection belongs inside a native operation.
Native handlers retain atomic scene/owner guards, guide constraints, save/reopen
obligations and affected-region checks. No parallel native writes.

`work_catalog(context, results)` returns at most128 executable items:

```python
{
    "id": "inspect_attachment", "revision": "qualified-handler-v1",
    "lane": "diagnosis", "handler": "native_evidence",
    "description": "Distinguish attachment distortion from guide-supported rim shape",
    "payload": {"operation": "inspect_attachment"},
    "reads": {"scene": "observed-revision", "guide": "qualified-revision"},
    "writes": [], "requires": {},
    "completion_condition": "Saved measurements distinguish the two mechanisms"
}
```

`handlers[name](item, controller_context)` returns a result with status `completed`,
`failed`, `no_progress` or `needs_reconciliation`, and original evidence. `requires`
maps prerequisite IDs to allowed outcomes, such as `{"inspect": ["completed"]}`
or a recovery item's `{"fit": ["failed"]}`. Retain prerequisite definitions in
the catalog, including completed ones. Freeze their original read revisions; do
not rebase completed definitions on every observation. Source changes invalidate
dependent use. A new candidate number does not justify repeating a failed mechanism.

Declare writes when an operation changes a named shared dependency. A generated
target that replaces a shared target slot is such a write; bind dependent work
to the actual completed output revision. A directory write does not implicitly
cover separately named file dependencies. The conflict diagnostic identifies
the key and prerequisite to correct.

Pure `ArrayPreparation` jobs that create new immutable result artifacts can use
`writes=[]`. Read only their real inputs and intent/authority dependencies, not
the growing output directory or every key in the observed state. Otherwise the
job changes its own definition after completion and can be selected repeatedly.
Two independent preparations must not share an output-directory read merely
because their artifacts live under the same parent. Keep each task definition
stable for its input scope; downstream tasks use `requires` and the exact
returned artifact hashes. Real input or authority changes still invalidate use.
Preserve completed results rather than rerunning them to repair catalog metadata.

Every item addresses a named uncertainty or improvement; catalogs are executable,
not wish lists. Unsupported repairs remain blocked while other useful work proceeds.
Image interpretation, ambiguous anatomy and new mechanisms return to the reasoning
owner. Numerical verification and model confidence cannot approve appearance. Offer
shared-view activation only after its visual review/preservation prerequisites are
met. Save/checkpoint housekeeping is fixed code, not gratuitous model inference.

`public_projection(snapshot, actions, plan)` returns `state`, public `descriptions`
for every offered action ID, and optional `lane_facts`. Exclude private paths,
identity and raw payloads. `judge_batch(state, decisions, binding)` uses the
workspace's typed judgment bridge and returns freshly validated `binding` and
normalized `judgments`. Questions choose the next lane and independently choose a
conditional operation in each lane, with an explicit no-applicable-operation option.
They do not see one another's answers. Lane-specific facts are supplied to the
priority question as well as the lane's operation question. Include before/after improvement, preserved
constraints and current target qualification in the reviewed context, rather than
only a remaining defect count. Label historical exclusions separately from a
newly qualified proposal; neither technical improvement nor qualification approves
appearance. Applicable conditional answers are cached
with exact evidence/menu/authority bindings and reused until these change. The complete
last decision, including a deferral, is also reused across restarts when the menu,
plan, authority and reviewed public context are identical. New evidence invalidates it.

Batch ready independent questions. Do not speculate about missing future native
results: consume actual receipts to refresh the catalog. Read-only preparation may
overlap native work only under declared disjoint dependencies and a qualified
adapter; this queue executes handlers serially. Independent question batching is
implemented; concurrent native execution is not.

The controller continues until its finite bound, no eligible work, review deferral,
cancellation or uncertainty. `needs_review` is a normal semantic result and clears
inference-in-flight state. Transient provider failures can use bounded, accounted retries with backoff and
fresh dependencies. Unknown native effects require reconciliation and are never
replayed automatically. `work-queue.json`, controller
events and `last-batch.json` retain actual work, judgments, recommendations and
outcomes. Judge success by useful visible results and avoided owner turns, not
inference count. Lesson selection/proposed updates preserve original evidence and
do not automatically promote a single-character success to a universal method.
