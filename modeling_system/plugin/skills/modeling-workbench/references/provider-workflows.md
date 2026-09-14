# Provider workflows and aggregate cost preview

Use a provider workflow when several generation jobs depend on one another and you want one persisted place that says which exact job sits at which stage, what is still unresolved, and what the remaining known spend is. It composes existing authorities and adds none: `request_reference`/`prepare_guide` prepare jobs, `claim_job` is the only dispatch intent, `reconcile_job` is the only result path, `review_reference` decides image validity, and diagnostic recipes keep their own steps. The workflow never contacts a provider, fetches a price, spends credits, replaces a transport or retries a job.

## Graph contract

A graph is `schema_version: 1`, a `name`, and 1–24 `nodes` with distinct IDs. It forms a bounded acyclic dependency graph:

| Node kind | Binds | Fields |
| --- | --- | --- |
| `job` | an existing `reference_job` handle | optional `expects` (`kind` image/video/mesh, `provider`, `intent_class`), optional `source` naming an upstream job or review node, optional `depends_on` |
| `review` | a `reference_review` record of the source job's output | required `source` naming a job node |
| `diagnostic` | a diagnostic recipe step with a current reusable/accepted result | optional `depends_on` |

`source` and `depends_on` both create dependencies. A job node with a `source` must bind a video or mesh job whose `reviewed_source` descends from that node: the exact bound review when the source is a review node, or a review of the bound job when the source is a job node. Cycles, unknown references, unsupported kinds and oversized graphs are refused before any record exists.

```python
graph = {"schema_version": 1, "name": "Image, review gate, mesh", "nodes": [
    {"id": "image", "kind": "job", "expects": {"kind": "image"}},
    {"id": "gate", "kind": "review", "source": "image"},
    {"id": "mesh", "kind": "job", "expects": {"kind": "mesh"}, "source": "gate"}]}
flow = work.create_provider_workflow(episode=episode_id, graph=graph, idempotency_key="eye-guide-chain-01")
flow = work.bind_provider_nodes(flow["workflow"], flow["revision"], {"image": {"job": image_job}})
```

Each instance belongs to one workspace and episode; the key identifies the instance, so two instances with the same name are independent and a job bound in one is not visible in the other. Identical creation retries recover the same instance. The episode receives a workflow link; if `episode_link_pending` is returned, append it with `revise_episode`.

## Binding and inspection

`bind_provider_nodes(workflow, expected_revision, bindings, reason="")` validates every requested binding and writes them in one compare-and-swap revision; if any binding is invalid nothing is written. Bind upstream nodes before their dependents (one call may do both in graph order). A job can occupy only one node of an instance, so nothing is counted twice. Replacing an existing binding requires a `reason`; the prior binding, its observed status and the replacement identity stay in `history`. Review bindings must be currently usable, and sourced job bindings must currently pass the same review check that `claim_job` applies. A stale `expected_revision` is refused; inspect again before retrying.

`inspect_provider_workflow(workflow)` is read-only and recomputes everything from current records:

- Job nodes show the ledger status of the bound job (`prepared`, `dispatching`, `submitted`, `unknown`, `completed`, `failed`, `cancelled`) or `planned` when unbound, plus `binding_validity`, the current `source_review` validity, `blocked_by`, `dependency_changes`, `ready_for_dispatch` and one compact `next`.
- Review nodes show `usable`, `rejected`, `superseded`, `unusable` or `invalidated` from the current References check; a later rejection blocks every dependent node.
- Diagnostic nodes show `satisfied` or `stale` against the recipe's current step status and result identity.
- `summary` lists ready, busy, blocked, invalidated, planned and new-branch nodes; `next_steps` orders busy reconciliation first, then invalidated bindings, then ready claims, then planning.

A prepared job is not a submission. `ready_for_dispatch` only means the node's dependencies and current reviews hold; the returned `claim_job` step still performs its own policy, review and live preflight checks, and the composition confers no authority. Dispatching, submitted, unknown and completed jobs never become dispatch suggestions; their next step is `reconcile_job` on the same handle or review of actual outputs. Failed and cancelled jobs are terminal: bind a distinct newly prepared job at the node with a reason rather than retrying the same key. Rebinding an upstream node invalidates downstream lineage and dependency snapshots; restoring the original binding makes retained descendants valid again, while other dependency changes are acknowledged by rebinding the dependent node with a reason. Applicability propagates through every ancestor: a completed image and its usable review cannot hide a stale upstream diagnostic. Quotes pin that applicability as well as binding identities; stale ancestors invalidate remaining quoted totals.

## Quotes and cost preview

`quote_provider_node(workflow, expected_revision, node, amount, unit, observed_at, expires_at, source, evidence)` retains observed displayed-price evidence for a bound job node. The amount is exact: an integer, a finite number or a decimal string; NaN, infinity, negatives and booleans are refused. `unit` is a short denomination label such as `USD` or `tripo:credits`; distinct strings are distinct denominations and are never converted or summed together. Timestamps carry explicit timezones; a quote cannot be observed in the future, must expire after both its observation and the present, and is valid for at most 30 days. Evidence links are pinned like episode evidence. The quote pins its basis: the job's intent digest (settings, generation input, authorization), the current review identity and disposition of its reviewed source, the binding identities of every upstream node and the graph revision. Unbound, failed, cancelled or invalidated nodes cannot be quoted.

`preview_provider_cost(workflow)` is read-only and recomputed on every call:

- Per node it reports `cost_role` (`remaining`, `busy`, `completed`, `terminal`, or `none` for review/diagnostic nodes), the current quote status (`known`, `stale` when the basis changed, `expired`, `missing`), the job's own recorded evidence such as displayed credits at preparation and claim, and reasons.
- `aggregate.remaining.known_subtotals` sums known quotes per denomination for prepared and planned nodes only; `unknown_nodes` lists remaining nodes without a known quote and `complete` is false while any exist. An unknown price is never shown as zero.
- Busy and completed nodes are excluded from remaining spend and shown separately with their own quoted subtotals; terminal nodes are excluded from every subtotal.
- `dispatch_authorized` and `prices_fetched` are always false. A preview grants nothing; claim and reconcile remain required.

```python
flow = work.quote_provider_node(flow["workflow"], flow["revision"], "mesh", 30, "tripo:credits",
    observed_at=observed_iso, expires_at=expires_iso, source="Tripo Studio generation panel",
    evidence=[{"kind": "file", "path": panel_screenshot, "role": "displayed price"}])
print(flow["aggregate"]["remaining"])
```

Full records remain accessible: the instance `detail` descriptor reads the ledger revision with bindings, history and quote record IDs; each job row carries an `inspect_workflow` descriptor, each review row a `read_record` descriptor and each diagnostic row an `inspect_recipe` descriptor. Historical completed jobs are read as immutable content while their reviews are checked against the current state.

## Limits

No API route, provider executor, automatic whole-graph execution, retry loop, price lookup, currency conversion or funding is delivered. Cost previews are per instance and rely on operator-recorded quotes. The workflow does not change the generation policy, the workspace binding, native ownership or appearance judgment, and it does not replace episode judgments or [method integration](method-integration.md). See [generation](generation.md) for route selection and recovery of the individual jobs, and [recipes](recipes.md) for diagnostic steps.
