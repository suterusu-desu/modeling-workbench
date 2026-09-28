# Measure coverage of a declared source surface

Use `construction_diagnostics.source_coverage` when vertices on source skin may
still form folded cells, leave gaps or cover the source twice. It measures saved
arrays without Blender, fitting, topology changes or a new retention gate. The
same function is registered as the `source_coverage` array preparation operation.

Supply one explicitly qualified UV chart/component and one pose per call. Choose
the source shape and footprint from the actual authority; a UV chart or nearest
point does not establish anatomical correspondence. This diagnostic never finds
correspondence for you. Separate disconnected charts/components into calls and
retain the declared seam scope. No baseline classification is provided.

```python
from modeling_system.construction_diagnostics import source_coverage

report = source_coverage(
    source_positions, source_actual_triangles, source_uv,
    source_components, source_qualified,
    candidate_positions, candidate_actual_triangles,
    vertex_source_triangle_ids, vertex_source_barycentrics,
    footprint_triangle_ids, candidate_domain_triangle_ids,
    source_boundary_edges, candidate_boundary_edges,
    component=0, units='m', frame='shared evaluated mesh frame',
    provenance={
        'source': {'mesh': source_name, 'pose': pose, 'capture_sha256': source_capture_hash},
        'candidate': {'mesh': candidate_name, 'pose': pose, 'capture_sha256': candidate_capture_hash},
        'qualification': qualification_record,
    },
    resolution=128, interior_order=4, boundary_samples=16,
    uv_tolerance=1e-10, area_tolerance=1e-12, distance_tolerance=1e-9,
)
```

`source_positions`/`positions` are finite Nx3; UV is source Nx2 with no implicit
seam or projection. Each source triangle has an integer component label and a
boolean qualification flag. `footprint` and `domain` contain distinct triangle
row indices in the actual supplied arrays. Arrays outside those selections are
retained in input hashes; unqualified source triangles are not used as support.
Correspondence has one triangle ID and three barycentric weights per candidate
vertex. Every vertex used by the domain needs finite weights in [0,1], summing to
one within `uv_tolerance`; that small sum error is normalized. Missing values on
unused vertices are allowed. Negative weights are never clamped or extrapolated.

Boundaries contain complete vertex-index edge pairs for each selected patch,
including holes. They must equal the edges with exactly one selected face owner;
stale, partial or duplicate declarations stay unknown. Face winding is evaluated
relative to the source chart, including a globally mirrored chart. A qualified
source component must be connected, consistently wound, nondegenerate and have
no positive-area UV triangle overlap above `area_tolerance`. Qualification flags
and the provenance record remain caller claims, not authenticated approval.

The report includes:

- **Orientation:** original candidate triangle row IDs with reversed or collapsed
  UV images. A measured reversal is a defect finding, not an unknown measurement.
- **Coverage:** conservative UV-area intervals for gaps and the area covered more
  than once, plus multiplicity counts in certain cells. This is chart area, not
  physical 3D area. The grid spans both footprint and mapped candidate. Cells
  within half a cell diagonal plus UV tolerance of any triangle edge remain
  unresolved, including internal edges. Their area widens both intervals. Thin
  features can therefore remain unresolved even when no cell center detects them.
  `candidate_outside_area_uv_sum` separately sums candidate triangle area outside
  the footprint by numerical clipping (multiplicity weighted, not union area).
- **Boundary drift:** symmetric sampled UV distance to the complete opposite
  boundary segments, with an upper bound from half the largest sample spacing.
  It compares boundary sets, not semantic point-to-point boundary correspondence.
- **Interior deviation:** Euclidean 3D distance to the source at corresponding UV
  samples. Each triangle gets strictly interior integer barycentric stations of
  order `interior_order`: order 4 gives three samples. Maximum and p95 refer only
  to measured samples, with missing/ambiguous counts and affected triangle IDs.
  Unsampled extrema, boundary heights and physical crossings are not established.

Candidate triangle containment is checked against the whole footprint by clipped
area, so an edge spanning a concavity or hole cannot evade the check merely by
having supported vertices. Beyond-footprint, missing, ambiguous or unqualified
correspondence returns `status='unknown'`, with reasons; available partial
measurements remain labelled. Shape/type errors refuse. Numerical overflow also
stays unknown. No `passed` field exists: `status='measured'` means these measurements
were obtained, even when they describe a fold or large gap. Raster coverage always
retains its approximation status and unresolved area; it is never exact coverage.

Choose tolerances in the declared scale. `area_tolerance` is UV area, `uv_tolerance`
is used for UV containment and barycentric roundoff, and `distance_tolerance` is
XYZ agreement of source hits on shared edges. They are numerical qualification
tolerances, not appearance acceptance thresholds. Coincidence below these values
is unresolved at that precision. The floating-point bounds are not interval
arithmetic guarantees. Source pairs, sample work and resolution are bounded;
budget exhaustion stays unknown, with no partial success. Resolution is 8..512,
interior order 3..16, boundary segments per edge 1..128. Select a smaller qualified
domain if a work limit is reached, and keep the exclusions explicit.

Preserve the full report alongside the source/candidate capture files. It records
exact supplied array hashes (shape, dtype and bytes) without reordering actual triangles, all parameters,
caller provenance and a combined input hash. For archive byte checking use the
existing `prepare_arrays` path/sha256/array inputs. Never reuse a rest-pose triangle
array as evidence for another native pose. Invoke this independently per saved
pose, retaining its actual triangle identity. No live freshness, motion between
samples, collision, eye clearance, likeness or owner acceptance is inferred.
