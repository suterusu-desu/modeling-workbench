# Pose guides from generated variants, joined onto an accepted neutral

Use these operations when a pose guide comes from several generated runs (image-to-mesh variants of the same pose, and
of the neutral pose made the same way) and has to constrain a character whose neutral surface is already accepted. They
are `ArrayPreparation` operations and plain Python functions in `modeling_system.guide_synthesis`. They work on front
depth maps (one depth per cell of a regular chart seen along one axis), fit no character and approve no appearance.

## Why a generated pose is not a guide by itself

Independent runs of one pose differ by smooth depth offsets and by local detail, and a generated neutral pose is not
the accepted neutral: in real use four runs of a neutral face differed from the accepted neutral by .005 to .027 before
alignment and by about .0035 in local detail after it, most at the eye corners. Fitting a character to one generated
pose, or to the generated pose with only a smooth correction, imports that generator's detail as if it were the design.
What the runs agree on is the *change* of the pose from their own neutral pose. Carry that change onto the accepted
neutral, and say at every place how far it can be trusted.

## The operations

- `front_depth(positions, triangles, window=..., cell=...)`: the front-most surface depth per cell (a z-buffer of cell
  centres) and the triangle seen in each cell. `front='min'` when smaller depth is nearer the viewer.
- `remove_thin_relief(depth, size=...)`: removes thin parts standing in front of the surface (lash fins, strands, drawn
  strokes the generator sculpted) with a grey closing over a `size`-cell square. Slopes are kept; a form curved toward
  the viewer is flattened by up to its depth curvature times the square's half-diagonal squared, and relief crossing a
  crease is filled with an error up to the slope change times the half-width.
- `stationary_offset(target, source, support, ...)`: a smoothed thin-plate offset through the difference on skin that
  does not move (per-run alignment, or removing a generator's whole-area shift).
- `pose_change(rest_maps, pose_maps, support, reference=..., behind=...)`: each rest run aligned to the accepted
  neutral map on the still support, each pose run to the rest consensus; the change is the median pose minus the median
  rest, and `spread` combines both sets' median absolute deviations. `behind` empties cells that show something through
  a cut in a generated surface (an eye slit) *after* alignment: before it, runs can sit farther from each other than any
  sensible hole threshold.
- `change_band(spread, change, cell=..., position_uncertainty=..., accuracy=...)`: per-cell tolerance. A change is
  carried by the position of the features that make it (a lid margin, a fold); where it varies fast, a small placement
  error is a large depth error, so the slope times the placement uncertainty is part of the band.
- `fit_depth_field(points, values, knots=..., bending=..., weights=..., pins=..., anchors=..., robust=...)` and
  `evaluate_depth_field`: a smooth cubic B-spline field through scattered samples with soft pins, zero anchors and
  Huber reweighting, for joins and corrections.
- `band_excess(values, lower, upper)`: how far each value lies outside its interval, 0 inside. Fitting a correction to
  the excesses (with points already inside holding it at 0) moves only what is outside, and only to the band's edge.
- `height_field_mesh(depth, window=..., cell=..., keep=..., max_step=...)`: the guide as a surface for display and
  fitting; blocks across a cliff are left open.

## Composing a guide (what real use needed)

1. Depth maps of every run in the registered chart, thin relief removed per run (a median of runs whose lash strokes lie
   in different places keeps pieces of all of them). Choose the width from the relief itself: in real use an 11-cell
   square left blocks of a drawn lash band and a 23-cell square removed it while still skin changed at the 99th
   percentile only; a 31-cell square began to remove the brow.
2. `pose_change` on a still support around the moving part, with `behind` for cut slits.
3. If the generator also moves skin the design keeps still (in real use the masks brought the whole eye area, cheek and
   brow included, forward as the eye closed), remove a `stationary_offset` of the change fitted on that still skin.
4. Where the pose moves, use the pose's own surface: adding a *smoothed* change to the accepted neutral leaves a ghost of
   the neutral's features wherever the pose covers them (in real use, the open eye's margin line inside a closing lid).
   The pose's own surface is the neutral plus the change plus the generated neutral's detail relative to the accepted
   one; on still skin keep the accepted neutral plus the change, and blend between them.
5. Hold points the design keeps fixed (a canthus) with a small local join only when the correction is small. A large one
   means the runs put another surface in front of that point (a fold covering a corner, or a remnant of relief): a join
   then digs a pit into the guide. Leave it unpinned, and widen the band to span both depths there.
6. Band: variant spread, slope times placement uncertainty, a stated accuracy, the generated neutral's detail where the
   pose's own surface is used (on skin; where the pose covers what the rest showed, use its typical value on the nearby
   skin, since a generator's different eyeball says nothing about a lid closing over it), and any join correction.

At rest the change is zero, so the guide reproduces the accepted neutral exactly there. A guide built this way is a
declared derivative of its runs: record the runs, the chart, the supports, each width and threshold, the pinned points
and which of them were left unpinned, and qualify it by its band before a correction relies on it. Inside a wide band
the guide does not decide the shape: say so, and choose the shape there by clean geometry and the reference images.

## Using the band in a correction

Measure how much of the corrected region lies inside the band per pose (`band_excess`) before and after, and declare a
preservation cell for it before a native trial. A correction fitted to the excesses leaves in-band material where it is,
so it cannot improve on a shape the guide cannot tell apart; combine it with the construction change that removes the
visible defect, and judge that change with matched renders and the crease diagnostics.
