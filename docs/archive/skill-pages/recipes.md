# Reusable offline diagnostic recipes

Use a recipe when a repeated diagnostic has meaningful dependencies and review decisions worth retaining. For a single analysis, direct service calls remain sufficient. Version 1 supports `inspect_control_coverage`, `inspect_target_domain`, `inspect_surface_correspondence` and `analyze_repair`; it cannot mutate Blender or dispatch generation.

## Start and inspect

Use the explicit workspace bootstrap in [episodes](episodes.md), with an active episode. `recipe_template()` returns the bundled `diagnostic-review` template and its content revision without creating records. Supply your own case files using the schemas in [control coverage](control-coverage.md) and [repair analysis](repair-analysis.md). The template contains no character data.

```python
from pathlib import Path
import hashlib

def pinned(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

template = work.recipe_template()["template"]
recipe = work.create_recipe(
    episode=episode_id, template=template,
    bindings={"coverage": pinned(coverage_case), "state": recorded_state,
              "repair": pinned(repair_case)},
    idempotency_key="coverage-repair-study-01")
recipe = work.run_recipe_step(recipe["recipe"], "coverage", recipe["revision"])
```

Each instance belongs to one workspace and episode; the key identifies that instance, not its display name. A different key creates an independent run. Identical creation retries recover the same instance. The episode receives a workflow link; if `episode_link_pending` is returned, append that link through `revise_episode` after inspecting the current episode revision.

`inspect_recipe(recipe_id)` is read-only. It reports exact input content revisions, each step's status and dependencies, review questions, bounded evidence reads and result expansion descriptors. Execute returned `operation`/`arguments` descriptors to read evidence. The instance `detail` exposes pinned inputs, template ID and historical attempts. Since 0.2.5, the default `decision_workspace` includes compact recipe step states and the direct inspection route. Running or uncertain recipe reservations block episode closure even if interruption occurred before an operation lease was created.

## Review, change inputs and recover

Inspect the coverage report and pinned repair input before `scope_review`. The bundled template deliberately invalidates that review when either case changes. Review whether the inputs refer to a justified common problem; the recipe does not manufacture correspondence between them. Supply your judgment and evidence:

```python
recipe = work.review_recipe_step(
    recipe["recipe"], "scope_review", recipe["revision"],
    decision="accepted", reason=diagnostic_reason,
    evidence=[{"kind": "record", "id": analysis_id, "role": "Reviewed coverage and its limits"}])
recipe = work.run_recipe_step(recipe["recipe"], "repair", recipe["revision"])
```

Reviews automatically retain their exact input key and upstream result reads alongside your evidence. Accepted means suitable for the stated diagnostic use. It never grants native or appearance approval. Rejection blocks descendants. A later corrected review appends evidence and invalidates dependent uses while preserving the original judgment. The final bundled review allows an honest diagnostic of missing evidence as well as a candidate. A custom template can require a particular result using `checks`.

`revise_recipe_inputs(recipe_id, expected_revision, bindings)` replaces only named inputs. Each `case` is a JSON file with exact SHA-256; nested file references are retained using the analysis pinning contract. Each `json` input is a retained JSON value. Changed inputs invalidate only dependent results and reviews. Unused navigation inputs do not invalidate analyses. Source-file edits alone do not change the retained input: `live_source_checked=false` is intentional. Rebind with the reviewed new hash when evidence changes. Historical reuse establishes no current Blender freshness.

Call `run_recipe_step` directly; it already uses `run_episode_operation` internally. One ready or stale operation runs per call. Reusable results return without redispatch. Each attempt is reserved before dispatch and records the existing operation handle. Running or uncertain attempts block further recipe mutation. If indexing is interrupted, use `recover_recipe_step(recipe_id, step, expected_revision)`: a matching durable returned result is recovered without rerunning the analysis. Absent or abnormal results stay unresolved and point to the original operation's reconciliation path. Never discard an uncertain instance just to bypass recovery.

Keep the exact instance ID across restarts. A fresh process can restore pinned cases and their assets after original files disappear. `expected_revision` prevents stale writers; inspect again after a conflict. Result applicability includes the template, resolved inputs, upstream result identities, service signature and loaded source revision. A package update conservatively makes older results stale; source compatibility is not guessed.

## Template contract and current limits

A template declares `schema_version: 1`, `name`, `inputs` (`case` or `json`) and 1–24 `steps`. Operation arguments use `{"$input":"name"}` or `{"$step":"id","path":["field"]}`. Analysis `case_path` must reference a declared pinned case directly. `depends_on` adds ordering; references add dependencies automatically. Review nodes require a question and nonempty `evidence_from` list of upstream step references. Optional `checks` compare resolved `value` and `equals`; a failed check blocks the step. Cycles, unsupported operations and unknown references are refused before dispatch.

Use `input_dependencies` on a node to make a reviewed input part of its applicability even when the operation does not consume it. This is especially useful for reviewing a downstream case before allowing its analysis. Graph structure expresses declared dependence; the agent still owns semantic compatibility and evidence interpretation.

No automatic whole-graph execution, retry loop, arbitrary Python node, provider cost estimate, native batch, topology lineage or cross-version compatibility inference is delivered. Close the episode using the existing separate character/method judgments and [method integration](method-integration.md); recipe completion does not replace them.
