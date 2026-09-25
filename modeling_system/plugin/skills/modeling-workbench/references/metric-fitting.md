# Planar and surface metrics for nonuniform meshes

Use `planar_fem_metric` through `ArrayPreparation`, or import it from
`modeling_system.metric_fitting`, when a qualified fit needs a differential
operator on a nonuniform recorded planar chart. An index-normalized graph
average can move an affine plane on an irregular grid. This helper assembles
piecewise-linear finite-element stiffness and lumped area mass from the actual
chart triangles. It does not choose guides, fitting directions, regularization
strength or native actions.

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
quality. See [preparation](preparation-and-bootstrap.md) and
[connected-patch correspondence](surface-correspondence.md).

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
- It fits no guide. Follow it with a guide depth fit, joined to the held surroundings.
- A rest reference also undoes the large rotations the pose makes on purpose: a lid margin
  that rolls under as the lid closes comes back as an open-eye bulge if it is free. Hold or
  soft-target material whose posed turn is intended, and keep the free set to the defect.

A twist of an interior handle inside a fixed boundary needs shear, not rotation; there
the rigid solve can leave more compressed triangles than the linear one. Compare the
stretch tails of both. A normal reversal against the reference means a fold only for
material that should not turn past 90 degrees; a closing lid legitimately does, so judge
folds with `local_reversals_before/after` (flips against the local rotation) and creases
with `construction_diagnostics.compare_bends`, not with the reference-relative count.

When both end poses of a motion are established and only the in-between poses are wrong, build the in-betweens from
the two poses with [motion paths](motion-paths.md) (pace, rolled or hinged paths) instead of fitting further keys.

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

In real use a closed lid corner whose material collapsed along an unsupported hook of its guide resisted eleven
repairs (rigid and similar deformation from rest, relaxation, thin-plate depth fills, harmonic 3D layouts, smoothing,
depth-only fairing). A uniform layout of that block in the front view, where it was not folded, with depth from the
surrounding retained skin, removed the line and left the guide depth fit unchanged.
