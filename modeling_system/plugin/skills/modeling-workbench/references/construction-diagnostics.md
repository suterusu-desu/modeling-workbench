# Diagnose a constructed surface before fitting

Use these checks when an aggregate objective hides local ridges, a predicted support cap disagrees with native geometry, or section plots appear disconnected. They inspect saved data without calling Blender. The Python helpers below are included in 0.2.12; they add no MCP operations. Verify the selected package and interpreter before using them. Their availability does not establish active-operator adoption or upgrade a running native adapter.

## Local shape and presentation

Pin the original and candidate arrays, topology, units/frame, chart correspondence, selection and viewing conditions. A common resampled chart permits a controlled comparison; it is not native subdivision or an exact native render. Keep actual native triangle sections as separate evidence. Do not compare tail counts across different sampling densities or exclusion masks as if only the surface changed.

```python
from modeling_system.construction_diagnostics import compare_bends
result = compare_bends(before_xyz, after_xyz, same_triangles,
    excluded_edges=reviewed_crop_edges, tagged_edges=chart_transition_edges,
    threshold_degrees=10, objective_before=before_score,
    objective_after=after_score, limit=12)
```

Supply integer vertex-index pairs; an absent edge refuses. The threshold is a diagnostic choice, not a universal crease limit. The paired objective arguments declare a comparable lower-is-better objective. Inspect its contradiction flag, local count/max/p95, tagged versus other edges, and worst added bends. A lower integral can coexist with more severe localized folds. Deliberate folds can also be valid: inspect identity references, exact sections and matched close/whole/profile views before judging form. Changing representation and support constraints together does not isolate either mechanism's effect.

Boundary, nonmanifold, inconsistently wound and explicit excluded edge counts remain visible. An empty measured domain returns null extrema, not proof of smoothness. Degenerate triangles refuse. `worst_added` returns at most 20 examples; `omitted_edges` reports the remaining measured domain. Full pinned arrays remain the source for further selection. Each helper limits inputs to 100,000 samples/triangles and returns an input-array revision; pin parameters and original source provenance alongside it. The revision does not certify live freshness or include authored parameter choices.

An unwelded preview insert has a display perimeter. Identify it separately from real interior geometry; exclude a perimeter only for a declared diagnostic question, and review the actual eventual join. A continuous preview cannot establish native boundary continuity.

For exact cuts, use the existing `geometry.plane_sections(arrays, axis, value, selection)` or the recorded-geometry section query. The existing cutter retains exact vertex-on-plane hits and deduplicates endpoints within each triangle. A plane along an edge produces that segment; an isolated tangent point produces no segment. Coplanar triangles are counted and excluded rather than silently turned into curves. Shared segments can have multiple triangle owners. This function uses an absolute 1e-12 coordinate tolerance; extremely small features or near-plane degeneracies require a declared scale/tolerance study, not automatic gap interpretation. It returns full segments, so retain them privately and use existing bounded record reads for agent summaries. Tests cover exact vertex crossings, on-plane edges, tangency and coplanarity. These planar cuts do not prove global three-dimensional connectivity.

## Verify the actual support calculation first

Read a pinned saved native graph and its input values, including selection, every additive branch, clamping/saturation, blending width, transforms, modifier order and evaluated pose. A remembered formula or one support object can omit an active contribution. Before interpreting candidate displacement, reproduce corresponding saved native outputs with the complete calculation.

```python
from modeling_system.construction_diagnostics import sample_residuals
baseline_check = sample_residuals(saved_native_xyz, predicted_native_xyz,
                                  tolerance=declared_tolerance, limit=12)
```

The tolerance is the maximum absolute XYZ component in the declared common frame. Inspect the maximum, p95, failed count and exact worst sample indices. Empty, nonfinite and mismatched arrays refuse; inputs are never changed. A failed reproduction invalidates the simplified model's reachability interpretation. Preserve it as a counterexample, trace missing graph branches, then retain the corrected reproduction separately. Do not change native clearance or support rules merely to make a candidate fit.

Once reproduction passes, compare a proposed target to its output under that same verified support map using `sample_residuals(target_xyz, mapped_target_xyz, tolerance)`. A difference shows displacement under the sampled map. It does not independently prove that no raw control solution exists. A passing baseline only validates these supplied samples; complete graph attribution, selected/interpolated sample coverage, all intermediate poses and control reachability remain separate. Report native samples and interpolated construction samples separately.

## Retain the limited result

Save exact originals, scripts, graph/source pins, diagnostic parameters and revised artifacts privately. Keep numerical observations, visual judgment, changed mechanisms and causal uncertainty distinct. Use existing `record_outcome` and the [method integration procedure](method-integration.md) to link the diagnostic lesson and any unresolved character result. An improved surface plus a corrected cap can support a bounded trial while full profile, actual native fit, joins and motion remain unresolved. Source regression success does not close those judgments. Coordinate native changes and installation/operator pickup with the sole owner; this procedure does not reopen a completed tools prerequisite.
