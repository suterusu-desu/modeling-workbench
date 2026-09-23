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

## Material crowding behind persistent creases

When creases persist after a depth or slope fit, especially creases that radiate from a
corner or run across rows of material, measure tangential strain before fitting again:

```python
from modeling_system.construction_diagnostics import compare_stretch
crowding = compare_stretch(rest_xyz, posed_xyz, pose_triangles,
    compressed_below=.5, stretched_above=2., tagged_triangles=corner_rows, limit=12)
```

Each triangle's smallest and largest principal stretch compares its reference and deformed
shape in their own planes; `normal_reversals` counts triangles whose normal turns over. A
surface compressed well below its reference length buckles, and no depth objective removes
that: the material has to move. Compare baseline and candidate on the same triangles and
report `compressed`, the low quantiles and the worst examples. Closing lids and deliberate
folds compress legitimately, so thresholds are diagnostic choices, and a relieved tail does
not approve appearance. `metric_fitting.relax_displacement` is one supported remedy, and
`metric_fitting.rigid_deform` when held material moves or turns a long way; see
[metric fitting](metric-fitting.md). A normal reversal marks a fold only where the material should not
have turned past 90 degrees from its reference.

Count folds with `local_reversals`, not `normal_reversals`, wherever the motion turns material
a long way. `normal_reversals` compares each normal with its reference direction, so a closing
lid margin that rolls smoothly past 90 degrees counts, band after band, while nothing is
folded. `local_reversals` fits each triangle's neighbourhood with its best rotation first (the
local step of as-rigid-as-possible fitting) and counts only triangles flipped against their
own surroundings: pleats, tucks, a vertex pushed through its neighbours. In real use a rolled
margin strip made up most of a corner's reversal count; minimizing that count steered the
corrections toward un-rolling the margin, which reviews saw as a bulge, while the actual
defect was a local crease. A wide flap folded back over a crease is a half turn of that
flap, so it is flagged along the crease rather than across its interior; measure the crease
lines, and any visible pinch, with `compare_bends` and compare the counts by chart region
before choosing what a correction has to change. `construction_diagnostics.local_reversals`
returns the per-triangle flags.

## Select the tessellation for the measured pose

Native quad diagonals can change between poses while vertex identities and polygon loops remain stable. For an exact native triangle comparison, obtain both recordings at the measured pose, validate their triangulation, then derive selected rows from those records:

```python
from modeling_system import geometry
geometry.compare(recorded_before, recorded_after)  # default triangle correspondence
pose_triangles = recorded_before['tri'][selected_triangle_ids]
result = compare_bends(recorded_before['co'], recorded_after['co'], pose_triangles)
```

Selection indices must belong to that pose's pinned triangle array. The default `geometry.compare` refuses unequal triangle layouts; do not evade this by copying old topology into a native record. Its explicit `recorded_polygon_loops` mode can justify a vertex comparison while returning `triangle_layout_equal=False`; that does not validate triangle-index transfer. If within-pose tessellation differs, resolve the correspondence or explicitly choose a common chart for the question.

A deliberate fixed-chart comparison passes its pinned chart to `compare_bends` and records that choice, both native topology revisions and their agreement/disagreement with the chart. It remains a fixed-chart result on posed coordinates. `compare_bends` receives only one chart and cannot verify its native provenance for the caller. Native plane sections use each pose's actual triangle record; a nonplanar quad's alternative diagonals can change both bend angles and section geometry. Preserve the original fixed-chart result when adding an exact-native continuation.

The synthetic changed-diagonal fixture in `test_construction_diagnostics` checks independently known quad angles and section points, native correspondence refusal and the separate fixed-chart result. No pose or anatomical identity is inferred from this fixture.

## Verify the actual support calculation first

Read a pinned saved native graph and its input values, including selection, every additive branch, clamping/saturation, blending width, transforms, modifier order and evaluated pose. A remembered formula or one support object can omit an active contribution. Before interpreting candidate displacement, reproduce corresponding saved native outputs with the complete calculation.

```python
from modeling_system.construction_diagnostics import sample_residuals
baseline_check = sample_residuals(saved_native_xyz, predicted_native_xyz,
                                  tolerance=declared_tolerance, limit=12)
```

The tolerance is the maximum absolute XYZ component in the declared common frame. Inspect the maximum, p95, failed count and exact worst sample indices. Empty, nonfinite and mismatched arrays refuse; inputs are never changed. A failed reproduction invalidates the simplified model's reachability interpretation. Preserve it as a counterexample, trace missing graph branches, then retain the corrected reproduction separately. Do not change native clearance or support rules merely to make a candidate fit.

Once reproduction passes, compare a proposed target to its output under that same verified support map using `sample_residuals(target_xyz, mapped_target_xyz, tolerance)`. A difference shows displacement under the sampled map. It does not independently prove that no raw control solution exists. A passing baseline only validates these supplied samples; complete graph attribution, selected/interpolated sample coverage, all intermediate poses and control reachability remain separate. Report native samples and interpolated construction samples separately.

## Distinguish expected attachment motion from residual error

A fixed anchor need not have a fixed support frame. Trace the recorded binding, supporting cells, frame convention and local offset for the dependent part. Verify baseline reproduction first; compare the candidate against an independently predicted finite-frame response, not merely against its old world coordinates. For a qualified affine frame whose columns are the local basis in the common measurement frame:

```python
expected_before = anchors_before + np.einsum('nij,nj->ni', frames_before, local_offsets)
baseline = sample_residuals(expected_before, observed_before, tolerance)
if not baseline['sampled_match']:
    raise ValueError('Binding/frame prediction does not reproduce the recorded baseline')
expected_after = anchors_after + np.einsum('nij,nj->ni', frames_after, local_offsets)
motion = sample_residuals(observed_before, observed_after, tolerance)
unexplained = sample_residuals(expected_after, observed_after, tolerance)
```

Here `np` is NumPy, arrays have corresponding sample order, and tolerances/units are declared. Use the actual qualified adapter's map when its blending, scale or nonlinear behavior differs from this example. Derive frames from independent supporting geometry and bindings; fitting them to the dependent observations would make the residual circular. Keep the frame/anchor/offset inputs pinned with the output; `sample_residuals` fingerprints positions, not their causal provenance.

Inspect anchor movement under its own declared constraint, supporting-cell/frame changes, expected dependent displacement and unexplained residuals separately. A failed baseline invalidates the predictor. A good baseline with a later residual nominates a binding/support/deformation discrepancy; it is not permission to ignore the difference or increase tolerance to pass. A matched moving-frame response explains the sampled motion, but does not approve its appearance or untested poses. Exact cell/frame freezing is an explicit temporary experimental restriction when justified, not a permanent requirement for every dependent part. Raw transaction fingerprints and rollback checks remain exact; this diagnostic does not weaken their later-edit protection.

The independent fixed-anchor fixture uses a known rotation and hand-specified observations. It detects raw displacement, accepts the independently predicted response, rejects an added unexplained displacement, and rejects a mismatched baseline binding. No native mutation or artistic acceptance is tested.

## Qualify contact samples by their complete source face

A triangle belongs wholly to a qualified material region only when every one of
its vertex identities belongs to that region. A rounded mean station or pair ID
can admit a triangle crossing into an unsupported guard. Preserve per-vertex
membership and report central and transition faces separately. If only part of
a face is qualified, explicitly clip that part and retain its barycentric source
ancestry rather than qualifying the complete face.

For a signed nearest-triangle distance, retain the closest point, barycentrics,
triangle normal/orientation, actual pose topology and projection classification.
A negative sign at a nearest edge or vertex does not by itself establish surface
penetration: the offset can be lateral to a finite patch. An interior negative
witness is a local oriented-surface result, not global inside/outside or swept
collision certification. No interior projection means unknown contact for that
patch, not clearance. Do not omit those unqualified samples from the report.

Compare a candidate and its baseline at corresponding poses with the same
qualification and sampling rule. Separate newly introduced, worsened, inherited
and improved contact residuals. Vertices, edge midpoints and centroids are finite
samples; absence of sampled failures does not certify the triangle interior,
motion between poses or appearance. Endpoint clearance alone does not establish
interior clearance. Use complete triangle contact for a supported projected
question; it still does not provide anatomical or whole-motion acceptance.

## Retain the limited result

Save exact originals, scripts, graph/source pins, diagnostic parameters and revised artifacts privately. Keep numerical observations, visual judgment, changed mechanisms and causal uncertainty distinct. Use existing `record_outcome` and the [method integration procedure](method-integration.md) to link the diagnostic lesson and any unresolved character result. An improved surface plus a corrected cap can support a bounded trial while full profile, actual native fit, joins and motion remain unresolved. Source regression success does not close those judgments. Coordinate native changes and installation/operator pickup with the sole owner; this procedure does not reopen a completed tools prerequisite.
