# Retain, integrate and use a method

The workbench develops reusable character-modeling capability throughout reference preparation, likeness review, diagnosis, construction, guide qualification, fitting, motion and recovery. Retain a demonstrated gain at the scope actually tested. A proven diagnostic can be supported while the character, solver hypothesis or appearance remains unresolved. Keep failed alternatives as searchable counterexamples. Do not wait for a whole animation or character to finish before integrating a useful method.

Use the existing `record_outcome`, `reconcile_episode`, `promote_procedure` and `retrieve_experience` operations. A prose archive preserves evidence, but an integrated method also has an operational destination and a use check. Save exact inputs, scripts and results immediately; interpretation and integration never gate saving or recovering a candidate.

## From evidence to a usable procedure

During ordinary modeling, [retained learning](retained-learning.md) carries actual
scoped reviews and optional conditional lessons into later decisions automatically.
This keeps interpretation available before formal promotion; it does not supply
the supported-benefit or distinct-reuse evidence required below.

1. Name the observed failure and causal mechanism, the existing approach, the bounded change, and the comparison that supports or rejects the benefit. Separate measurements, construction evidence, artistic choices and untested hypotheses.
2. Retain exact input manifests, code, dependencies, output receipts and counterexamples in the private workspace. Record support, exclusions, protected context, artistic variables, applicability and tests still missing. State the benefit narrowly: for example, grouping repeated missing-target constraints by native point reduces duplicate acquisition nominations without establishing target shape or native fitting quality.
3. Use `record_outcome` to retain separate character and method dispositions. Each outcome revises the question, so pass the current revision from `inspect_situation()["question"]`; an earlier identity is refused before any record is written. Link that outcome and the actual evidence in `reconcile_episode`. `method.status="supported"` refers only to the reason and applicability stated there. Keep unfinished claims in the character result and unresolved list, or a separate experimental method episode.
4. For a supported benefit, call `promote_procedure` with its episode judgment, mechanism-based instruction and conditions, stage, explicit limits, counterexample links, and exact `executable_paths`. Include the replay/input manifest and required dependencies alongside any script. One supported episode establishes `level="local"`; distinct episode reuse is required for `"reusable"`, and cross-character quality still needs actual cross-character evidence.
5. Retrieve by failure mechanism in a fresh process or operator context. Inspect conditions, limits and counterexamples; expand the actual procedure record. Verify and restore its pinned files in an isolated workspace and use the recorded supported operation. Do not execute an arbitrary retrieved script without inspecting its contents and applicability. Record the actual result, tool/runtime identity and remaining differences. No native/provider call is implied by retention or retrieval.
6. Close the episode with an explicit integration disposition in `method.integration`. Partial judgments stay valid while this field is pending. General methods and synthetic regressions can be reviewed into the clean tools repository; character evidence stays private and source-linked. Check source, package installation, loaded runtime and actual operator use separately.

`inspect_situation(question="ordinary question text", live=False)` focuses read-only retrieval around the selected recorded question. It neither selects a new question nor contacts Blender. A 64-character hexadecimal `question` explicitly selects that immutable question record for the read. On a new workspace without geometry, text focus still returns authority/experience and an honest missing-state result.

## Explicit close-out

`reconcile_episode(..., close=True)` requires separate judgments, reconciled operations and a `method.integration` disposition. It does not require character success. Existing historical closed records remain readable. Use one of:

- `implemented`: a narrowly supported benefit has an actual procedure/tool/rule artifact. Require reason, mechanism, scope, artifact links, limits and all four adoption dispositions below. Pending adoption is visible and does not become a claim of live use.
- `experimental`: an executable or operational trial recipe is retained with reason, mechanism, scope, artifacts and limits. Its unsupported claims stay unresolved; it cannot be promoted as a supported procedure.
- `not_generalizable`: explain why this result should not become a reusable method. The judgment's evidence and counterexamples remain retrievable; this is an explicit disposition, not silent loss.

After promoting a procedure, revise the episode judgment to reference it. This preserves the original supporting judgment and avoids a circular reference:

```python
method = {
    "status": "supported",
    "reason": "The measured diagnostic benefit, limited to the tested mechanism",
    "integration": {
        "disposition": "implemented",
        "reason": "Exact local procedure retained; broader adoption remains pending",
        "mechanism": "Observed failure mechanism for future retrieval",
        "scope": "Exact supported diagnostic or modeling step",
        "artifacts": [{"kind": "record", "id": procedure_id, "role": "local procedure"}],
        "limits": ["Unproven character, shape and motion claims remain unresolved"],
        "adoption": {
            "source": {"status": "evidenced", "reason": "Actual retained procedure",
                       "evidence": [{"kind": "record", "id": procedure_id, "role": "procedure source"}]},
            "installed": {"status": "not_applicable", "reason": "Local recipe uses existing tools"},
            "runtime": {"status": "pending", "reason": "Need exact runtime receipt from the use check"},
            "operator": {"status": "pending", "reason": "Need fresh-context retrieval and actual use"}
        }
    }
}
work.reconcile_episode(episode_id, latest_revision, method=method, close=True)
```

Each adoption stage accepts `evidenced`, `pending` or `not_applicable`, with a reason. `evidenced` requires actual typed evidence links; version assertions alone are insufficient. Installation is applicable when a tool/skill package changed. A fresh test process verifies only that process, not the current modeling task or its Blender adapter. `adoption_complete` concerns the declared artifact and scope; it never certifies character quality.

The packaged synthetic regression `modeling_system.test_method_integration` demonstrates the complete route: existing coupled-response diagnosis, separate unresolved character result, local promotion with exact inputs and failed alternative, fresh-interpreter mechanism retrieval, replay through `analyze_repair`, and evidence-linked close-out. It performs no Blender or provider work. Run `python -m unittest modeling_system.test_method_integration` from an installed environment or source checkout.

For a numerical stop, retain three separate observations: solver termination, independently measured constraint feasibility, and justified stationarity residuals. A line-search stop alone is not evidence of impossible geometry or a need for another generated guide. Preserve the exact candidate; when appropriate, test rescaling the same objective and unchanged constraints in a separate bounded trial and measure the actual geometry difference. Never replace this diagnostic with changed constraints or a final clamp. This is a supported diagnostic distinction, not a universal convergence remedy, native fitting result or appearance approval. `operation_context(stage="diagnosis", context={"observed_failure":"optimizer stopped without success"})` exposes the packaged rule in a newly initialized workspace; existing workspace procedure updates need explicit adoption.
