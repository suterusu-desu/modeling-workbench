# Repeated guide fitting and candidate continuation

Use `modeling_system.guide_fitting` for recorded-array preparation and
`CandidatePipeline` for repeated candidate, verification and retention work.
Both compose with the existing OperatingSession. Private adapters supply scene
bindings and native handlers; this is not another executor.

## Guide and material paths

- `triangle_sections(vertices, triangles, position, axis=0)` returns segments,
  source triangle indices and separately identified coplanar triangles. Remap
  indices through the original face selection when cutting a subset.
- `closest_on_segments(point, segments)` and `point_at_coordinate(value, segments,
  reference, axis=1, reference_axis=2)` return a point and selected segment. Select
  the qualified anatomical branch first; proximity does not supply correspondence.
- `section_intersections(first, second, plane_axis=0)` retains segment-pair ancestry
  and collinear overlaps separately. An overlap is not a unique intersection.
- `interpolate_supported(stations, targets, supported, provenance, max_gap=...,
  max_missing=...)` fills bounded interior gaps only. Stations must be ordered with
  direct endpoint support. Supply JSON-compatible source provenance per direct
  station. Interpolated stations retain direct parents, fraction and span; they
  are never labelled direct triangle support.
- `remap_material_path(source, target)` allocates source arclength fractions along
  an already qualified ordered target and returns target-segment ancestry.
- `prepare_section_fit(stations, baseline, targets, supported, provenance, weights,
  max_gap=..., max_missing=..., space=..., frames=..., object_matrix=...)` combines
  bounded interpolation, preservation weights and displacement encoding. Arrays
  may contain multiple corresponding profile points per station. Zero weight
  preserves the baseline exactly. Limits and weights come from qualified support.

Choose coordinate space explicitly. `space="world"` takes the actual affine object
matrix and encodes world displacement as object-local vectors for application
after attachment rotation. `space="attachment"` takes each actual material frame.
These produce different intermediate motion when neighboring frames rotate.
Preserve or compose existing displacement inside the established position owner;
never replace an occupied input or add a second independent position owner.

These functions cannot admit guides, invent correspondence, certify native
realization or approve appearance. Retain output provenance and pinned inputs
privately. Native operations still compare actual geometry, unaffected regions
and matched images under the same atomic owner/state guard.

## Automatic continuation

```python
from modeling_system.candidate_pipeline import CandidatePipeline, capability_task

pipeline = CandidatePipeline(run_directory / "pipeline", revision="qualified-method-v1",
    prepare=adapter.preparation_options,       # optional useful Jev choices
    candidates=adapter.candidate_options,      # genuine qualified alternatives
    verify=adapter.verification_task,          # fixed independent verification
    review=adapter.review_preparation_task,    # optional fixed evidence packaging
    retain=adapter.retention_task)             # fixed reviewed retention
# Pass catalog=pipeline to the existing create_session / OperatingSession.
```

Each factory `(state, previous)` returns normal qualified work items. Initially
`previous` is None; later it contains the exact selected `task` and settled
`result`. Bind actual output paths and hashes. `capability_task` builds a standard
item from explicit profile, semantic bindings, dependency keys, covered writes,
handler and native flag. Use it inside a factory to freeze the first binding.
It also supports diagnosis, experience and recovery lanes; offer only executable
work addressing a named question, with applicability and prior failures in its
public description. Each lane's Jev question addresses that actual purpose.
Factories describe work without native or provider effects. Put actual preparation
inside a qualified queued handler, so its effects and evidence are retained.

The pipeline adds prerequisites and required output checks, freezes materialized
catalogs, and removes unselected variants from eligibility without erasing them.
Failed or uncertain work never triggers a sibling variant automatically. Use the
normal queue for a separately justified recovery with its own original receipts.

By default, actual early image review gates broader verification. Record it through
`session.record_review(candidate_id, expected_basis=session.review_basis(candidate_id),
judgment=..., evidence=...)`, then resume `session.run(max_steps=...)`. This is
existing visual interpretation, not a new user approval. Use `early_review=False`
only when the fixed verification is already necessary for that scope. Retention
always consumes a useful actual review of the final verification or review task.
Fixed follow-ups do not spend inference; candidate and preparation alternatives
remain Jev choices. Single-lane choices retain action/defer without a redundant
lane/defer question; multiple lanes share one independent conditional batch.

Resume the same instance with the same owner, episode and ledger. Do not rebase
frozen dependencies after edits. A new scope gets a new revision and directory.
Include factory code and runtime revisions among the qualified task's read keys,
alongside the real scene, guide, adapter and output dependencies.

For an already completed candidate, `pipeline.adopt_completed_candidate(task,
outcome)` attaches its exact original queue task and outcome before first use.
Do not change its definition to match the new helper, recreate its effect or
relabel its historical runtime. Follow-ups bind the original dependencies plus
separately named new runtime/helper dependencies. The pipeline rejects changed
or missing original outcomes. This entrypoint has no preceding preparation stage.

## Workload accounting

`session.metrics()` and `inspect_session` report controller stage times, native
handler time, outcomes, verified retention tasks and visual review submissions.
Native time includes its fixed prerequisites. Cycle totals overlap component
times; do not add them together.

Record other actual Astra work with `session.record_intervention(kind=..., reason=...,
evidence=..., seconds=...)`. Kinds: `correspondence_preparation`,
`capability_development`, `catalog_preparation`, `recovery`, `evidence_interpretation`.
Omit duration when unknown. These are episode-linked records. Image reviews are
counted automatically. Unreported effort and avoided turns remain unknown;
retention is not user acceptance and no speedup is inferred without a baseline.
