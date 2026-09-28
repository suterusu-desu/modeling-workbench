# Planar and surface metrics for nonuniform meshes

Use `planar_fem_metric` through `ArrayPreparation`, or import it from
`modeling_system.metric_fitting`, when a qualified fit needs a differential
operator on a nonuniform recorded planar chart. An index-normalized graph
average can move an affine plane on an irregular grid. This helper assembles
piecewise-linear finite-element stiffness and lumped area mass from the actual
chart triangles. It does not choose guides, fitting directions, regularization
strength or native actions.

The deformers on this page (relaxation, shape-preserving deformation, on a
support surface or not, planar re-layout) are bounded clean-up of a surface, not a way to create motion. Several
real-use cases below repaired a per-point blink that was later replaced by a hinged
lid ([build the mechanism first](build-the-mechanism-first.md)): when the same kind
of mark keeps returning in a moving part, change its construction instead.

```python
from scipy.sparse import coo_matrix, diags
from modeling_system.metric_fitting import planar_fem_metric

metric = planar_fem_metric(
    chart, triangles, units="m", frame="qualified orthonormal chart",
    relative_area_tolerance=1e-12,
)
K = coo_matrix((metric["stiffness_values"],
    (metric["stiffness_rows"], metric["stiffness_columns"])),
    shape=metric["stiffness_shape"]).tocsr()
M = diags(metric["lumped_mass"])
```

`chart` is finite `(N,2)` in a declared Euclidean frame with both axes using the
same length unit; `triangles` is `(T,3)` with actual vertex indices. Units and
frame are explicit declarations, not conversions or qualification. A projection
of a curved surface defines a planar fitting metric, not its intrinsic 3D metric.
Retain the transform, source/pose, chart meaning and qualified support privately.

The dimensionless conditioning threshold compares absolute doubled triangle
area to the longest squared edge length. Choose it for the actual input scale
and precision; the example is not an anatomical tolerance. Invalid/duplicate or
degenerate triangles, unused vertices, nonmanifold edges and locally overlapping
adjacent triangles refuse assembly. Mixed input winding is allowed when adjacent
triangles occupy opposite sides of their exact shared edge. Global overlaps,
chart injectivity and anatomical roles are not inferred.

Outputs include sparse stiffness triplets/shape, lumped vertex masses, triangle
areas, exact boundary edges and boundary/interior vertices. `public_metrics`
provides compact counts, conditioning, signed orientation counts and affine
reproduction measurements through the ordinary preparation report. No interior
vertices means an unavailable affine check (`null`), not a successful test.

## Units, residual domain and boundary conditions

`K` is positive-semidefinite stiffness (integrated gradient dot gradient), with
dimensionless entries in a 2D length chart. `M` has length-squared entries;
`M^-1 K` has inverse-length-squared entries. Constant functions have zero weak
load; affine functions have zero weak load at interior vertices, up to arithmetic
error. Boundary rows include boundary flux, so an affine boundary residual is
not evidence of curved geometry. The reported affine defect is interior-only,
in the chart's length unit after applying `K` to centered chart coordinates.

Choose which residual rows and boundary conditions the fit actually means.
For a declared interior-only residual domain `r`, a squared-Laplacian matrix is
`K[r,:].T @ diags(1 / mass[r]) @ K[r,:]`. This is not an automatic boundary
condition or a complete solver: nullspaces still require qualified constraints.
Using every row instead penalizes boundary flux too and can bias a fit with
nonzero boundary slope. Holding positions alone does not prescribe that slope.
The [libigl tutorial](https://libigl.github.io/tutorial/#data-smoothing) explains
this boundary distinction and alternative Hessian boundary treatment; no such
Hessian operator is implemented here.

A squared-Laplacian matrix has inverse-length-squared entries. When combined
with a mass-weighted data residual, its coefficient has length-to-the-fourth
units. There is no default length inferred from mesh indices or elongated guide
edges. Hard guide constraints and held boundary values are another explicit
formulation, with their own feasibility and conditioning requirements.

Obtuse elements can yield signed edge weights; positive semidefinite energy
does not guarantee nonnegative interpolation weights or absence of overshoot.
Preserve direct guide support versus interpolation and the objective's actual
quantity: displacement smoothing and final-surface fitting differ. Metric and
affine tests do not approve target correspondence, native realization or visible
quality. See [preparation](preparation-and-bootstrap.md).

## Surface metric on recorded 3D triangles

A planar chart measures lengths in its projection. Where the surface is steep
or folds relative to that plane, projected distances shrink and the operator
no longer describes the material. `surface_fem_metric(positions, triangles,
units=..., frame=..., relative_area_tolerance=...)` assembles the same outputs
from `(N,3)` positions: each element is the planar element in its own triangle
plane, so stiffness and lumped mass are intrinsic to the recorded surface. It is
also registered for `ArrayPreparation` under the same name.

Use it when a correction smooths or interpolates material over curved or steep
regions, for example redistributing a displacement from rest across a corner,
and use `planar_fem_metric` when the fit is genuinely posed in a qualified chart.
A flat patch in any orientation reproduces the planar metric; an isometric fold
leaves it unchanged. Winding does not affect intrinsic stiffness, so mixed
winding is reported (`inconsistently_wound_edges`) rather than refused;
nonmanifold edges, duplicates and degenerate triangles refuse.

Linear functions of ambient coordinates are harmonic only on a flat patch.
Their interior load is the discrete mean-curvature normal, reported as
`interior_mean_curvature_max` (inverse length), not as a reproduction error.
The metric qualifies no correspondence, tangential material coordinates or
appearance; state which quantity a fit regularizes and which vertices are held.

## Relax crowded material

`relax_displacement(reference, deformed, triangles, held, units=..., frame=...,
relative_area_tolerance=...)` (also an `ArrayPreparation` operation) replaces the
displacement from `reference` of every free vertex in the patch by the field that
minimizes the squared intrinsic Laplacian with all held vertices fixed. Use it when
[crowding](construction-diagnostics.md) is the cause of a crease. The patch may be
a region of a larger mesh; free vertices must be interior to it, and held vertices
keep their exact coordinates. It reports principal-stretch tails before and after.

Choose the held set from evidence, and say why each part is held:

- Attachment-sampled rows (for example a lid rim that a lash samples) exactly,
  without a surrounding ring when that ring is the crowded material.
- Guide-supported or separately fitted material, and returns or pockets that the
  guide does not represent, each with one ring for slope continuity.
- Material the pose leaves static. Otherwise the interpolation spreads the moving
  boundary's motion into skin that should not move.
- The patch edge, with two rings.

The result fits no guide. Moving material across a curved surface changes its depth,
so follow it with a depth fit. In a retained configuration whose depth came from a
coupled whole-region registration, fit that retained surface rather than refitting
the guide locally: a local refit inside the region can add a ring where it meets the
held surroundings. Affine displacements are reproduced exactly on flat patches, and
only approximately on curved ones.

## Move handles and keep the shape

A linear displacement interpolation cannot rotate material. When held handles move a
long way relative to their neighbours, or turn the material (a lid margin carried over
a corner, a sheet that has to bend), it shortens chords, crowds the transition and can
fold it. `rigid_deform(reference, initial, triangles, held, units=..., frame=...,
relative_area_tolerance=..., iterations=50, tolerance=1e-9)` (also an `ArrayPreparation`
operation) is the as-rigid-as-possible alternative: held vertices take their `initial`
positions exactly, and every free vertex keeps the `reference` shape up to local
rotations (intrinsic cotangent weights of the reference surface, alternating best
rotations and a sparse solve, started from the free vertices' `initial` positions).

- Choose a fold-free reference. It is the shape being preserved: the material's rest
  shape when the current pose is itself folded, since preserving a folded pose keeps
  its folds.
- Start from a sensible initial state, for example the linear result; the energy has
  local minima. The report gives iterations, convergence and first and last energy.
- Hold the patch edge and everything whose position is established, as for
  `relax_displacement`. Negative cotangent weights of obtuse elements are clamped to
  a small positive value and counted.
- A hard held boundary between rest-shaped free material and a differently shaped achieved
  surround concentrates the mismatch into a crease along that boundary. Optional soft
  targets (`targets=`, `target_weights=`) pull free vertices toward an achieved shape with
  a weight relative to each vertex's cotangent degree: zero on and near the material being
  replaced, rising with distance from it. A soft pull toward a shape that is itself folded
  brings the fold back, so measure the distance from the folded material, not from a point.
- Optional intervals (`interval_axis=`, `lower=`, `upper=`, `interval_weight=1e3`) keep one
  coordinate of each free vertex inside its own `[lower, upper]` (NaN or an infinite value
  leaves that side open), for example a guide's depth band along the view axis, so the
  relaxed material stays inside the guide during the solve instead of being measured
  against it afterwards. Vertices outside their interval are pulled onto the violated bound
  in that coordinate only, by a degree-scaled penalty re-solved with the rotations until the
  set stops changing; a vertex the solve keeps inside leaves the set, and the active ones
  finish exactly on their bound. The report gives the bounded and active vertices, the
  outside counts and largest violation before and after, the penalty's residual before that
  last step and how often the set changed. Give a vertex that already lies outside the band
  an interval that includes where it is when the relaxation must not drag it (and its
  neighbours) across.
- A guide band can carry the guide's own fold lines in its edges. Hard intervals pin the
  vertices that press against such an edge and trace it, and a vertex held where it is
  beside free neighbours leaves a crease along that boundary. With a small
  `interval_weight` (about 1, as strong as the shape term) and `project_active=False` the
  band is a soft pull that the shape term smooths, so the material follows the band's
  volume. The report's penalty residual then says how far outside the band the solve
  settled.
- Without intervals it fits no guide. Follow it with a guide depth fit, joined to the held
  surroundings.
- A rest reference also undoes the large rotations the pose makes on purpose: a lid margin
  that rolls under as the lid closes comes back as an open-eye bulge if it is free. Hold or
  soft-target material whose posed turn is intended, and keep the free set to the defect.

A twist of an interior handle inside a fixed boundary needs shear, not rotation; there
the rigid solve can leave more compressed triangles than the linear one. Compare the
stretch tails of both. A normal reversal against the reference means a fold only for
material that should not turn past 90 degrees; a closing lid legitimately does, so judge
folds with `local_reversals_before/after` (flips against the local rotation) and creases
with `construction_diagnostics.compare_bends`, not with the reference-relative count.

In real use three separately fitted forward depth corrections met on a closing upper lid beside its inner corner (a
guide-band fit decaying toward the corner, another guide's field growing away from it, an older fit peaking between).
Each correction was smooth, but where they met the lid's bend across its rows gathered into a narrow fold, 11 to 14
degrees per edge against 7 to 9 at rest with the corner side flattened, seen as a soft line up from the corner in
three-quarter view through the middle of the blink. Softening one of the fits or re-timing the skin above made it
worse. What worked was re-laying that patch of each in-between as-rigid-as-possible from rest, with the soft pull toward
the finished shape at 0 on the fold's core and rising with distance from it. Three details mattered:
- The pull also rose toward the held margin rows. A hard held row right below the fold took the mismatch as a
  sharper margin turn.
- The band was a soft interval widened to where each vertex already was. Hard intervals pinned the vertices already
  behind the band and left a crease beside them. A soft interval to the band itself traced its fold lines.
- The re-lay faded out before closure. There the skin stretches flat, and a rest-shaped pull brought back an older line.

When both end poses of a motion are established and only the in-between poses are wrong, build the in-betweens from
the two poses with [motion paths](motion-paths.md) (pace, rolled or hinged paths) instead of fitting further keys.

## Keep material on a support surface

An interval on one coordinate and `planar_relayout` with a front depth map constrain material through a chart: a
height field over one axis or plane. A `radius_map` is a radial chart about a supplied centre and follows some steep
lateral surfaces, but it too needs the target to be single-valued along its rays. Where the target turns away from the
chart (skin above an outer eye corner facing the temple, seen from the front), a view-axis band bounds only the part of
a pleat that points along the view, and surplus material can buckle out of the surface there. `conform_to_surface(
reference, initial, triangles, held, support_positions, support_triangles, units=..., frame=...,
relative_area_tolerance=..., lower=0., upper=0., support_weights=1.)` (also an `ArrayPreparation` / `construct`
operation) is `rigid_deform` with the band measured along the local normal of an arbitrary qualified triangle support,
with no chart and no single-valued requirement; motion along the support is left to the shape term. Repeated buckling
of that kind motivates trying it; it does not diagnose every hood or pleat.

```python
from modeling_system.metric_fitting import conform_to_surface

result = conform_to_surface(rest, start, patch_triangles, held, support_co, support_tri,
                            units="m", frame="object", relative_area_tolerance=1e-9,
                            lower=0.0005, upper=0.002,              # just outside the support, per vertex or one value
                            support_weights=weights,                # 0 frees a vertex from the support
                            project_active=True)                    # False keeps the soft (penalty-only) result
closed = result["deformed"]; report = result["public_metrics"]; result["warnings"]
```

- Offset convention. Each iteration every free vertex with a positive weight gets its exact closest point `c` on the
  support (a complete search, `geometry.closest_points`, not a shortlist) and the normal `n` there: the support's
  area-weighted vertex normals interpolated with the barycentric weights of `c` and normalized. Its offset is
  `n . (x - c)`, positive on the side the support's winding faces, and should lie in `[lower, upper]` (NaN leaves a side
  open). Inside the band material is free, like the intervals. Outside, a penalty (`band_weight`, default 1e3, times its
  weight and cotangent degree) pulls only that normal component, in one coupled XYZ solve.
- The support must be manifold, nondegenerate and consistently wound (it refuses otherwise; rewind it). Choose it and
  its extent yourself; the tool never enlarges it.
- Convergence: closest points, normals and the set outside the band are re-queried after every solve; the solve has
  converged when that set is unchanged, the energy settles (`tolerance`) and the last step moved no vertex further than
  `position_tolerance` (default `distance_tolerance`, itself 1e-6 of the support's size). An energy plateau while
  closest points still slide is not convergence. With every weight zero it is exactly `rigid_deform`.
- Final projection (`project_active`, default): every free vertex still outside its band moves along `n` onto the
  violated bound, re-queried three times, and every measure is taken after that. It applies to every positive weight
  alike, so it does not keep a weight fade: a small weight only makes the penalty weak during the solve. To keep a fade
  (support weight falling off toward a brow), use `project_active=False` and read the soft result and its residuals, or
  give the vertices that should leave the support zero weight. On a curved support the offset re-measured after
  projection can differ slightly from the bound; it is reported.
- The report separates the solve (`converged`, `iterations`, `last_step_max_move`, `band_penalty_residual`,
  `projection_max_move`) from support of the result. Band counts and tails (`band_outside_*`, `band_max_*`,
  `band_p95_after`, `support_distance_max_after`, `beyond_boundary_*`, `supported_after` / `unsupported_after`) count
  free vertices with a positive weight only. Held vertices with a weight are measured separately (`held_on_support`,
  `held_conflicts`, `held_conflict_max`: they keep their exact positions); zero-weight vertices are not measured
  (`unconstrained_free_vertices`). A vertex whose closest point is on the support's open edge and lies past it counts as
  beyond the boundary and unsupported even when its normal offset is zero. Closest points that move between support
  components are counted during the solve (`component_changes_during`), during projection
  (`component_changes_projection`) and from start to end (`component_switches`). `support_facing_before/after` give
  patch triangles facing with / against the support normal, measured only on the `support_facing_measured_triangles`
  of `patch_triangles` whose three vertices have a positive weight (wind the patch like the support). The stretch tails
  and reversals are those of `rigid_deform`. Per-vertex `support_offset`, `band_residual` and `beyond_boundary` arrays
  come back for display.

Hold what is established (margin rows, a landing, the brow) and choose the weights where the material should keep to
the support.

Limits. It cannot create area: surplus material is compressed along the surface (see `smallest_stretch_q01_after` and
`compressed_after`) and a patch with far too much material for the support still needs a construction change. The
closest point is not a correspondence: it does not say which part of a disconnected support a vertex belongs to, and a
normal band does not establish clearance from anything but the support. A zero-weight vertex beside weighted ones is not
constrained but moves with them through the shape term, so it differs from the plain `rigid_deform` result. The
reference must be fold-free, the result depends on the initial positions, and zero facing flips on the measured
triangles do not prove that a sheet is unfolded or free of self-intersection.

## Smooth a region on the mesh

`metric_fitting.smooth_region(positions, triangles, held=..., weights=..., iterations=10, lam=.5, mu=-.53)` (also the
`construct` operation `smooth_region`) is Taubin smoothing on the mesh: each pass moves the weighted points toward the
mean of their edge neighbours and back, which removes lumps without shrinking the surface. `held` points keep their exact
positions (the margins and the row beside them, a lid's inner surface, a mirror-hidden half); `weights` ramp it out
(over rows from the margin, a radius fade, a taper at the corners). Give it the rest and the mechanism's end shape as one
stack, `np.stack([rest, closed])`: the operator is linear, so the motion is smoothed exactly as the shapes are.

In real use, lumps within a few millimetres of a lid margin were out of reach of the front height-field fits (a front
height field must hold still there); smoothing on the mesh with a held set, a row ramp, a radius fade and a corner taper
removed them and was accepted. Smoothing only the rest creased the closing lid, because the closed shape's change no
longer fitted the smoothed rest; the same smoothing on the closed shape fixed it. Without the corner taper the smoothing
squeezed 11 more triangles near the corners during the blink (`corner_compression` 787 against 776); with it, 748. Check
the selection with `behind_front` first: a radius drawn in the front view also takes the back of the head.

## Re-lay collapsed material in a plane

A pose can squeeze a block of material until its rows and columns run parallel (collapsed quads): shading shows a hard
line along it, and relaxing, smoothing or refitting depth around it keeps the line because every one of those works on
the collapsed layout itself. `planar_relayout(positions, triangles, free, plane_axes=..., depth_axis=...,
depth_samples=... or depth_map=..., window=..., cell=...)` (also an `ArrayPreparation` operation) gives the free
vertices the uniform harmonic (Tutte) layout of the patch in a projection plane, every other patch vertex held, then
takes their depth from a smoothed thin-plate field through the `depth_samples` vertices or from a front depth map
(for example a guide surface). The report counts triangles flipped in the plane before and after and, with a
`reference`, the compressed triangles and local folds against it.

- Pick the plane in which the block is not folded (for a closing lid seen from the front, the front view), and check
  `plane_flips_after` is 0.
- Hold everything whose position is established: the block's surroundings, rows an attachment samples, seam or rim
  vertices, and a ring around the block.
- Take depth from the retained surface around the block when that surface came from a coupled registration; a local
  guide refit inside the block adds a ring where it meets the held surroundings. Use `depth_map` when a qualified
  guide surface covers the block.

`layout='rigid'` (with `reference`, for example the rest pose) keeps the reference layout up to local rotations
instead of evening out the spacing: `rigid_deform` of the positions projected into the plane, every non-free patch
vertex held. Use it where the material has to turn, such as the fan of rows around a corner that rotates down as a lid
closes: uniform harmonic weights pull an uneven fan toward even spacing and distort it. With `depth_map`,
`depth_blur` low-passes the map by a Gaussian of that many cells, ignoring empty cells, so a retained surface can give
the depth: its volume stays and its fine corrugation goes.

In real use the inner end of a closed lid had the fan of rows around the canthus left near its rest place while the
margin rows closed: rows folded into a V and buckled into a zigzag across the columns (fine radial lines). A block that
held part of the torn rows rebuilt the tear at its edge; a thin-plate depth through a ring of held samples bulged into
a pad where dense held seam rows met the block; freeing the seam's inner end opened a hole at the corner. What worked:
the whole torn fan free with the seam end held, the rigid layout from the rest fan, depth from the retained closed
surface low-passed over three cells, and the change blended out toward the lid centre, where the untouched rows
continue (a hard block edge there met them in a sharp corner).

In real use a closed lid corner whose material collapsed along an unsupported hook of its guide resisted eleven
repairs (rigid and similar deformation from rest, relaxation, thin-plate depth fills, harmonic 3D layouts, smoothing,
depth-only fairing). A uniform layout of that block in the front view, where it was not folded, with depth from the
surrounding retained skin, removed the line and left the guide depth fit unchanged.

### Check the projected boundary before the layout

Tutte's theorem makes a harmonic layout with positive weights injective under several premises: the domain is a
triangulated disc, its whole boundary is held on a convex polygon (no interior edge joining two boundary vertices on
one straight side), and every interior vertex is free. `projected_boundary(positions, triangles, free=...,
plane_axes=..., units=..., frame=...)` (also an `ArrayPreparation` / `construct` operation) measures the boundary part
of those premises, and reports what it sees of the rest (loops, components, handles, topology), before
`planar_relayout` runs. It does not establish the whole premise. Give it the same positions, triangles, free set and plane as the layout. `plane_basis` (two
orthonormal in-plane axes, shape (2, 3), checked to 1e-9 independently of the geometric tolerance) replaces
`plane_axes` for any other direction.

```python
import numpy as np
from modeling_system.metric_fitting import projected_boundary

n = 7                                                      # a 7 x 7 sheet in the x-z plane, free 3 x 3 block inside
x, z = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
positions = np.c_[x.ravel(), np.zeros(n * n), z.ravel()]
index = np.arange(n * n).reshape(n, n)
a, b, c, d = index[:-1, :-1].ravel(), index[:-1, 1:].ravel(), index[1:, 1:].ravel(), index[1:, :-1].ravel()
triangles = np.r_[np.c_[a, b, c], np.c_[a, c, d]]
free = np.zeros(n * n, bool); free[index[2:5, 2:5].ravel()] = True

report = projected_boundary(positions, triangles, free=free, plane_axes=(0, 2), units="m", frame="object")
report["public_metrics"]      # loops, crossings, touches, overlaps, reflex corners, interior held vertices, tolerance
report["loops"][0]["vertices"], report["loops"][0]["reflex_corners"]
```

- **Domain.** With `free`, the analysed domain is the triangles incident to a free vertex. That is the layout's
  influence domain, in which only the free vertices move, and its boundary is the held boundary they are laid out
  against. Without `free` it is the supplied patch's own perimeter. A larger surrounding patch is not the constrained
  boundary, so pass `free`.
  - Free IDs outside the patch refuse, as they do in `planar_relayout`.
  - `interior_held_vertices` lists held vertices inside that domain (fixed handles); no loop check covers them.
  - `free_on_patch_boundary` lists free vertices that `planar_relayout` would refuse.
- **Loops.** Boundary edges run along their triangle's winding and are traced through the triangle fans, so a pinched
  vertex splits two loops instead of making a figure of eight. Each loop gives, in the supplied indexing:
  - its vertices in order, its edges and each edge's owning triangle, and its component;
  - its signed area (counterclockwise positive in the plane of the first and second axis), perimeter and turning
    number;
  - which side the domain lies on, taken per edge from the owning triangle;
  - the other loops it lies inside (holes). Nesting is decided only between simple loops that neither cross nor touch
    each other; any other pair is listed in `nesting_unresolved_with`.
- **Interactions.** Every boundary edge pair, within and between loops, is classified once:
  - `crossings`: a proper crossing, with its point;
  - `touches`: non-adjacent edges within the tolerance, marked `segments_meet` when they actually meet;
  - `collinear_overlaps`: collinear edges overlapping, with the overlap length;
  - `adjacent_overlaps`: consecutive edges folding back over each other;
  - `collapsed_edges`: edges no longer than the tolerance.
  Ordinary neighbouring edges are never reported. Each entry names both source edges, their owning triangles and
  loops.
- **Corners.** Corners come from the ordered loop only. `domain_angle_degrees` is measured on the domain (material)
  side, and a corner whose vertex lies within the tolerance of its neighbours' chord is straight. At a hole loop the
  material side is outside the hole polygon, so a square hole has four reflex corners.
  - `reflex_corners` are claimed only for a loop that is simple, encloses area (more than the tolerance times its
    perimeter), has the domain on one side and turns once.
  - Otherwise `interior_angles` gives the reason it is undefined, and `local_turns_against_domain` lists the local
    turns instead. Those are a diagnostic, not polygon corners.
- **Topology.** Duplicate triangles, triangles that repeat a vertex, bad indices and an empty domain refuse. The report
  lists instead:
  - nonmanifold and inconsistently wound edges;
  - pinched, branched and ambiguous vertices;
  - vertices with more than one triangle fan, even at an ordinary boundary degree (a closed part touching the patch);
  - open chains;
  - per component, its triangle count and Euler characteristic (1 for a disc).
  `warnings` names a domain with no boundary at all and a boundary with zero span in the plane: neither is a clean
  polygon.
- **Tolerance.** One length in the caller's units decides every near case. `tolerance` gives it directly; otherwise it
  is `relative_tolerance` (default 1e-9) times the projected boundary's bounding-box diagonal. The report states which,
  with units and frame. Positions are translated to the boundary's centre in 3D before projecting, and the
  predicates run in coordinates normalized by the boundary's span, using sign tests rather than products. That
  reduces cancellation and underflow; it cannot recover precision already lost in the supplied positions, and the
  predicates are floating point, not exact. A span outside 1e-150 to 1e150, a tolerance of 1e150 or more, or any
  input that would make a reported value nonfinite refuses. Near cases are reported against
  the tolerance, not decided, and the input is never snapped or changed.

Read the whole report before choosing the layout, not one count:
- The boundary is compatible with the convex-boundary premise, within the declared tolerance, only when all of these
  hold: one loop and one component with Euler characteristic 1; interior angles defined for that loop; no crossings,
  touches, overlaps, collapsed edges or open chains; no nonmanifold, inconsistent, pinched, branched, ambiguous or
  multi-fan entries; no interior held vertices and no free vertices on the patch boundary; no reflex corners; and no
  warnings. Zero reflex corners with undefined interior angles says nothing about the corners. Every decision is made
  within the tolerance, so a clear report is not exact convexity: a turn or crossing smaller than the tolerance is
  classified as straight or as a touch.
- Reflex corners on a simple loop describe a legitimate nonconvex boundary; the harmonic premise is then not met. What
  to do is the caller's choice. No other layout is guaranteed injective there either.
- Crossings mean the held boundary is not a simple curve in this plane, and no injective layout of a disc has that
  boundary in this plane. A different qualified projection, a constructed boundary or a changed domain may each be
  the answer; this report does not choose.
- Several loops or a pinched vertex mean the domain is not one disc, and the premise does not apply as stated.

It approves nothing:
- It is a projection test, not 3D collision: crossing edges can belong to sheets separated in depth.
- A clean outer loop does not qualify interior held handles, a second boundary, the rigid layout or the later depth and
  rest blending. After the layout, check `plane_flips_after` and look at the result.

## Fit explicitly assigned support domains together

When nearby support sheets serve different material regions, a globally closest
foot can select the wrong sheet. Optional assigned mode in `conform_to_surface`
keeps the same coupled endpoint solve while restricting each weighted vertex's
search to one caller-declared domain. It is also available in array preparation:

```python
result = conform_to_surface(reference, initial, actual_triangles, held,
    combined_support_positions, combined_support_triangles,
    units='m', frame='shared evaluated coordinates', relative_area_tolerance=1e-12,
    lower=-0.0002, upper=0.0002, support_weights=weights,
    support_domains=triangle_domain_ids,
    support_assignment=vertex_domain_ids,
    qualified_domains=[10, 20], distance_tolerance=1e-8)
```

The three assignment arguments must be supplied together:

- `support_domains`: one nonnegative integer label per support triangle. Keep
  the actual bounded support triangles, with any exclusions already removed.
- `support_assignment`: one integer domain label per candidate vertex. Every
  weighted face-owning vertex, including a held one, needs a qualified assignment.
  Use `-1` only where no support is consumed (unweighted or outside the patch).
- `qualified_domains`: distinct known domain IDs explicitly qualified by the
  caller for this use. Missing/unqualified assignments refuse before solving.
  This declaration does not authenticate source authority or appearance.

Each consumed domain must be one connected, consistently wound, nondegenerate
support component. Label separate components separately; do not rely on nearest
selection between them. Normals and boundary edges are computed independently
per domain, even if domains share vertices, overlap or use opposite orientations.
Every initial/iterative/final-projection query obeys the same assignment. The
patch remains one coupled ARAP solve, including edges between differently
assigned vertices; there are no sequential independent fits to overwrite one
another. Closest point within a domain is still not anatomical point correspondence.

A transported-section constraint can use this route only when the caller has
actually constructed and qualified its bounded triangulated support surface.
The tool does not invent a ribbon, depth or correspondence from a raw 1D curve.
Keep an inferred section-derived support labelled inferred in the private source
ledger; numeric assignment does not promote it to accepted anatomy. Conflicting
constraints, bad reference folds and unsupported transitions remain modeling work.

The existing band energy and projection are unchanged. They do **not** force
material back inside an open support boundary. Zero normal-band residual may
coexist with positive `beyond_boundary`; the existing `unsupported_after` already
counts those free vertices. Inspect both quantities, convergence and held conflicts.
Assigned mode additionally returns:

- `assigned_support_triangle`: original row index in the supplied support
  triangle array per measured vertex, `-1` for unmeasured vertices.
- `assignment_report`: supported and unknown weighted vertex IDs, separate
  unconstrained free IDs, per-domain escape/unknown lists, and an input-array
  revision. `unknown` includes band residual or boundary escape beyond tolerance,
  including held conflicts. No weighted scope reports `unmeasured`.

`measured` refers only to the declared weighted support checks. It is not a
coverage, collision, eye-clearance, likeness, solve-convergence or retention pass.

## Preserve declared source-cell order during the fit

Assigned ownership alone leaves tangential motion free. A vertex can remain near
its correct support while its connected faces reverse or cross a held collar.
For a qualified disk-shaped source chart, `conform_to_surface` can instead solve
for explicit barycentric correspondence **during** each coupled ARAP step:

```python
result = conform_to_surface(
    reference, initial, triangles, held, support_positions, support_triangles,
    units="m", frame="registered frame", relative_area_tolerance=1e-12,
    support_domains=source_domains, support_assignment=vertex_domains,
    qualified_domains=[0, 1], support_weights=weights,
    lower=lower, upper=upper, distance_tolerance=1e-8,
    ordered_charts=[{
        "support_triangle_ids": qualified_chart_faces,
        "source_uv": source_uv,
        "candidate_triangle_ids": connected_candidate_faces,
        "vertex_source_triangles": declared_vertex_cells,
        "initial_barycentric": declared_initial_barycentrics,
        "minimum_area_ratio": 0.1,
        "boundary_mode": "fixed",
    }],
    ordered_max_iterations=100, iterations=50,
)
```

The numbers above illustrate the API; qualify tolerances and the minimum area
ratio for the intended construction. Each chart dictionary has exactly these
seven fields. Indices refer to the original supplied arrays, not a reindexed
subset:

- `support_triangle_ids` selects one caller-qualified source chart in one
  assigned support domain. `source_uv` is a finite `(len(support_positions),2)`
  chart. The selected source faces must form a consistently oriented manifold
  disk with a simple boundary, nondegenerate UV faces and no UV overlap.
- `candidate_triangle_ids` selects the complete connected candidate disk to
  protect, including its boundary and incident faces within that disk. Every
  vertex in it must have positive support weight and that same qualified owner.
  Separate charts may not share candidate vertices; unlisted triangles,
  cross-chart joins and unconstrained material are explicitly outside the order
  contract. `uncovered_triangle_ids` lists them.
- `vertex_source_triangles` is an integer array of length `len(initial)`. Each
  selected vertex names one actual triangle in the selected source chart: its
  **admissible cell**, not a nearest-point guess. `initial_barycentric` is a
  finite `(len(initial),3)` array; selected rows are nonnegative and sum to one.
  Unselected rows are ignored (use `-1` and finite zero rows).
- `minimum_area_ratio` is strictly between `1e-6` and `1`: every selected UV
  triangle must retain at least that fraction of its positive **initial mapped
  area**. This is a declared compression bound, not an anatomical default.
- `boundary_mode="fixed"` explicitly fixes the entire initial chart boundary
  in UV and XYZ, **including vertices otherwise free in the larger patch**.
  Existing held interior vertices also stay exact. Only chart interiors can
  redistribute. Choose this boundary as part of the construction; do not freeze
  a defective join merely to use this operator.

The initial connected UV disk must lie entirely within the qualified source
chart. Its fixed boundary defines the footprint preserved here; the tool does
not assert that this is the intended whole source footprint. Initial XYZ must
realize the supplied barycentrics within `distance_tolerance`, including held
vertices, and their normal bands must include zero. The small initial XYZ
residual is retained exactly, reported and never silently snapped. A new support
pose therefore needs an explicitly constructed feasible starting correspondence;
passing the old positions against a changed surface is not automatic transport.

Interior vertices vary continuously within their declared source triangles.
Their exact affine source lift replaces the nearest-foot normal penalty there;
unlisted free material retains the existing normal-band/ARAP objective. The
entire patch uses the same cotangent metric, local rotations, coupled XYZ
objective and held positions. A constrained numerical solve changes its search
direction under cell and face-area constraints. An analytic quadratic area
bound also prevents an accepted straight step from crossing a collapsed UV
triangle, even if both endpoints would be positive. Final normal projection
cannot move ordered vertices outside this parameterization.

This is a **bounded source-cell mode**, not a general sliding chart solver.
Cells never change automatically. A vertex at a source triangle corner may have
only a narrow allowed direction; a desired move across that cell edge requires
the caller to revise its qualified correspondence/construction. Material indices
are not frozen in the interior, and no nearest-point or anatomy inference chooses
the cells. Curved triangulated sources work because each selected cell has its
own affine 3D lift. Do not substitute coarse source triangles that change the
qualified shape merely to make larger cells.

Read `ordered_report` together with `public_metrics`, `assignment_report` and
the actual returned geometry:

- `invalid_initial`: a source realization, band or initial orientation conflict;
  no deformation was attempted, `feasible=false`, and the initial state is
  returned with reasons. Malformed or ambiguous chart/topology declarations
  refuse with `ValueError`.
- `blocked`: the local constrained search could not make an admissible useful
  step, or there were no movable variables. The last feasible state is returned;
  it is not a completed requested endpoint or proof of global infeasibility.
- `iteration_limit`: finite work ended without the full convergence criteria.
  `converged` requires successful inner optimization plus the existing outer
  energy, position and active-set criteria. A constrained optimum need not match
  the rest shape or achieve a desired modeling correction.

Reports retain actual source triangle IDs, resulting barycentrics/UV, fixed
boundary IDs, area-limited faces, cell-boundary vertices, inner-solver outcomes,
accepted step fractions, numerical tolerances and a combined input revision.
`feasible=true` describes the declared numerical chart/cell constraints only.
Qualification uses finite clipping and boundary tests in normalized UV, not
exact arithmetic. Each chart is bounded to 4096 source/candidate faces, two
million containment pairs and 200000 overlap pairs; the coupled solve is bounded
to 1500 variables and 1–1000 inner iterations per outer step.

Positive connected faces and a fixed simple disk boundary preserve the declared
piecewise-linear **UV footprint**. Actual 3D candidate edges/faces connect lifted
vertices by straight segments; their interiors need not follow a curved source
across its triangle edges. This does not certify actual 3D coverage, surface
distance, absence of folds/collisions, tangent or bending quality, cross-chart
joins, swept motion or appearance. Continue to measure
[source coverage](source-coverage.md), the actual connected surface and the
visible result. This option neither repairs an already reversed starting map
nor changes native trial/retention requirements. Without `ordered_charts`, the
existing fitter and its convergence behavior are unchanged.
Keep unweighted/inferred regions visible; no support claim applies there. Pin
source/candidate captures, parameters and private qualification evidence alongside
the report. By default assigned mode uses 1e-6 of the smallest consumed domain's
size for its shared distance tolerance; set an explicit tolerance in the declared
units when domains differ in scale. Legacy calls without assignment arguments
retain their previous global support search, defaults and result structure.
