# Motion paths: hinged parts, clean in-betweens, re-timing

Build a moving part as a mechanism first ([build the mechanism first](build-the-mechanism-first.md)): a lid lands on
the still opposing lid by turning on a hinge and rolls there with one pace. The hinge operations below derive the end
pose from the mechanism and carry attached parts with it. The other operations build clean in-betweens between two
established poses, and re-time or join an inherited motion. All are `ArrayPreparation` operations and plain Python
functions in `modeling_system.motion_paths`; they fit no guide and approve no appearance.

## Why in-betweens go wrong

A blend shape moves every point along its straight chord, all at one pace. Reference characters close a lid with one
shape: the lid moves as one piece at one rate, the corners and the lower lid stay still, and their flat, recessed eyes
leave the straight path clear. Over a convex obstacle close behind the material (a lid over a round eye, a sheet over a
bone) the chord cuts into the obstacle, so the same one-piece closing needs a turn on a hinge. The usual wrong repair
fits extra keys at intermediate phases and stacks correction fields on top; each is fitted separately, so the
in-between sections bend: across a moving band the section turns one way above its edge and the other way at it (an
S), which shading shows as a crease or shelf even when every key is close to its guide. Measure that with
`construction_diagnostics.section_turns` before and after a change (see below).

## Hinge: land a lid and carry what rides on it

`hinge_landing(positions, edge, landing, pivot, axis, weights=None, outside=0., rest=None, corner_blend=0.,
band_smooth=None)` derives a closed pose from the rest
pose. Each `edge` point (the lid margin, ordered or not) turns about the hinge line through `pivot` along `axis` to the
`landing` curve's angle at the same axial position (the opposing margin at rest, sampled densely enough to interpolate
along the axis) and moves to the landing curve's distance from the axis plus `outside`. Every other point turns and
moves out by its weight times the edge's values at its own axial position: 1 on the margin and the lid's inner surface,
falling off above the margin, 0 on what stays. Weight-0 points keep their exact bytes, and no point slides along the
axis. `public_metrics` reports the edge's turn and radial range, the landed gap to the landing polyline and how many
edge points lay beyond the landing curve's axial range (they take its end values). Points past the edge's own ends take
its end point's turn: to bring the turn to 0 at a pinned corner, add the corner to `edge` with weight 0. Build the
in-betweens as one roll to this pose: `path_positions` with the same pivot and axis and one pace row per phase for every
point.

When `positions` is a retained closed body rather than the rest, its corners may be unusable: rims left apart at the
seam's end need a large extra turn right beside a pinned corner (a spike in real use). With `rest` and `corner_blend`
(an axial distance, or one per end, lower axial end first) the moving points near each end take the landing of `rest`,
blended C1 into the landing of `positions` over that distance; `public_metrics['corner_blend']` reports the widths and
the largest change. A C1 fade of the turn toward the corner instead opened a gap at the seam's end.

At a blunt corner the margin rises almost across the axis, so its first points need very different turns within a tiny
axial distance, and the band, which takes the edge's turn at its own axial position, steps there: a column of squeezed
triangles running up from the corner. `band_smooth={'sigma': ..., 'within': ..., 'fade': ..., 'ends': 'low'}` smooths
the band's turn and radial change along the axis (Gaussian, `sigma`) within `within` of the chosen end(s), fading back
over `fade`; the edge keeps its exact landing and pinned edge points count as not turning. In real use sigma .003 within
.015, fade .015 (eye width .11) cut the band's step beside the canthus from about 13 to 4 degrees per .001 of axis,
removed the squeezed column and left the landed margin identical; sigma .006 folded more. It leaves a small zigzag right
at the canthus, where the smoothed band out-turns the margin's first points; capping each band point's turn at its
nearest margin point's turn may remove it (untested). Turning the band from the nearest point along the margin, a
harmonic turn profile and a smoothing that fades in above the corner were tried and did worse or were not chosen.

`hinge_carry(attached, host_reference, host_end, pivot, axis, fraction=1., host_index=None)` moves attached points (lashes,
a seam, a marking) by the hinge change of their host point (the nearest host point at rest unless `host_index` is
given): at fraction f each turns by f times the host's turn, moves out by f times its radial change and along the axis by
f times its axial change. Give the fraction the host's in-betweens use; for a one-pace roll that is the phase itself. The
returned per-point `turn`, `radial` and `axial` values are what a native node group needs; the formula is in the
docstring and on the mechanism page. `seat_distance_change_max` reports how far an attached point leaves its host by
the end pose: rigid carrying holds only as far as neighbouring host points turn alike.

`hinge_change(reference, end, pivot, axis)` gives each point's `turn` (radians, right-handed about the axis, the shorter
way), `radial` and `axial` change between any two poses, for example to realize an existing closed pose as a hinge in a
rig. Points on the axis and half turns are refused.

`hinge_lift(positions, edge, corners, pivot, axis, keep_share=..., up=(0, 0, 1), across=(1, 0, 0), weights=None,
fade=0.)` raises a facing edge's middle on the same hinge, for a design whose lower lid meets the upper part way: each
edge point turns about the axis, keeping its distance from it and its axial position, until its sag below the line
between the two `corners` (measured in the view plane of `up` and `across`, as `line_depth`) is `keep_share` of its rest
sag. `fade` eases the turn to 0 toward the corners along the axis so the band below does not kink there; `weights`
carry the band below the margin and the lid's inner surface. Land the upper margin on the lifted edge with
`hinge_landing` and roll both with one pace: both are turns about the same hinge. Keep the closed line's depth a
measured design choice (`line_depth` on the approved closed drawing).

Measure the result with `construction_diagnostics.closing_edges` ([construction diagnostics](construction-diagnostics.md));
with a rising facing edge, measure against its pose (`against='pose'`).

## Pace: keep when each region moves

`motion_pace(reference, end, samples, sigma=..., cutoff=None, travel_scale=None, minimum_travel=0.)` measures, for an
existing motion sampled at increasing phases, how far each point is along its own reference-to-end chord, clipped to
[0, 1], smoothed over reference-pose neighbours with a travel weight (points that barely move take the pace of the
material around them) and made monotone. It keeps the existing motion's timing while its paths are replaced.

Keep the host's own pace wherever something attached to it follows a schedule of its own (a lash strip that turns on
the lid rim at fixed phases). Re-timing the host, even by smoothing its pace along the band, desynchronizes the
attachment: in real use a rim that lagged by a few percent opened a gap between the lash and the lid, and rows above
the rim that lagged the rim let the eye show through the band. Smooth the pace in radial rows only where no attachment
depends on it, or move the attachment with it. An attachment that rides a hinge moves with its host through
`hinge_carry` at the host's own fraction, which removes this problem.

## Re-time a region of an inherited motion: shared schedules and pace floors

These operations repair the timing of a motion that already exists. Do not construct a moving part with them: a lid
rebuilt from two established poses with its corner floored to close first, and its inner end on a later schedule, was
rejected on sight as a zipper. Its inner, middle and outer thirds had turned .50, .58 and .84 of the way at the same
phase; the hinged rebuild with one pace, .64, .64 and .65.

An existing motion's pace can shear where neighbouring parts keep different timings: one part still at rest while the
part beside it is half way, so the material between them creases or a part stands proud of its neighbour. Re-time a
smoothly weighted region with `schedule_pace(pace, schedule, weights, mode=...)`:

- `mode='blend'` puts the weighted points on one shared `schedule` ((1 - w) pace + w schedule), so the region moves as
  one piece. Take the schedule from the material next to it that already moves right, `shared_schedule(pace, members)`
  (per-phase median, or a weighted mean), or a plain timing `smooth_step(phases, start, end)`.
- `mode='floor'` raises the weighted points to at least the schedule (max(pace, w schedule)): a part of the inherited
  motion that must reach its end pose by a given phase. A per-point onset, `smooth_step(phases[:, None], onset - ramp,
  onset)`, moves a region progressively from one end.

`weights` are the spatial fade, usually `1 - smooth_step(distance, full, zero)` from the part. Points at weight 0 keep
their exact pace; both modes keep every point's pace monotone. In real use, on the per-point lid that was later
replaced by the hinge:

- A floor whose spatial fade spanned only a few rim spacings brought one rim point to its end pose a column before
  the facing rim met it and left a small notch in the open rim; a fade about twice as wide had none.
- A floor applied before a depth field fitted to the in-between positions made the fit solve again against the new
  timing and spread changes into skin that should not move; applied after the fit, the fitted field stayed.
- Where the two sides of a corner meet (upper and lower lid columns running out of a canthus), a schedule shared by
  one side only sheared a slash along the junction: fade it before the junction, or share it across both sides.

Re-timing changes when each point moves, not where: paths, clearance and attachments behave as with any pace, so an
attachment that follows the host's schedule has to move with it.

## Join a rebuilt motion to the existing one

A rebuilt region usually hands over to the existing motion where the existing motion hardly moves the surface: each
point takes `w * rebuilt + (1 - w) * existing`, with `w` from how far the existing motion moves it. `travel_weight(reference,
travel, low=..., high=..., sigma=...)` gives that weight. Where the travel is small it is noisy, and a weight taken from
each point's own travel alternates between neighbours (in real use .05 next to .97 beside a lid's inner corner); any
difference between the two motions there comes out as fine lines. With `sigma` the travel is averaged over the
reference surface first and the weight varies smoothly (`neighbour_jump_p99_before/after` measure it).

Two things follow from the blend. A correction applied to the rebuilt motion is scaled by `w`, so it must already be 0
where `w` falls, or the falling weight turns it into a ramp: anchor correction fields at 0 on the blend edge. And fields
fitted to the rebuilt motion are often global (a smooth field through many points): if smoothing the weight moves the
points they are fitted to, they re-solve everywhere. Fit them with the weight and region they were built with and use
the smoothed weight only to compose the result; points the smoothed weight adds to the region then sit at the existing
motion while fitting. The pace is global in the same way: `motion_pace` scales travel weights by a percentile of the
travel of the points it is given, so adding barely moving points shifts every point's pace slightly. Compute the old
region's pace as before and give the added points theirs at the same `travel_scale`.

Smooth only where the weights alternate. In real use smoothing the weight of every point near a corner, including the
lower lid's inner end and the inner surfaces there, changed how much of that surface the three-quarter view showed;
limited to the upper lid's own skin away from the corner, the fine lines faded (the largest added bend there fell from
14-24 to 3-9 degrees) and the rest stayed as it was.

## Paths: roll over the obstacle, straight beside it

`path_positions(reference, end, pace, pivot=None, axis=None, roll_weight=None)` places every point at its pace.

- No pivot: the chord.
- `pivot` and `axis`: a roll about that hinge line. The angle about the axis and the distance from it are interpolated
  linearly and the position along it moves straight, so the band keeps its distance from the hinge between its two
  end values and gains no sideways drift. Give the axis a closing lid turns about (through the eye's centre, along the
  line between the corners).
- `pivot` alone: slerp of the direction from the pivot. A great-circle route swings points away from the central
  meridian sideways, a drift that looks like the material sliding toward a corner; prefer the axis form.
- `roll_weight`: 1 where the material passes over the obstacle, 0 beside it. A roll about the obstacle's centre swings
  material beside it (a corner beyond the eye's outline) the wrong way; keep chords there and blend smoothly.

Rolling keeps each distance from the hinge between its two end values; it does not know the obstacle's real surface.
Check clearance against the actual obstacle mesh, especially where neighbouring rows move at different paces.

## Swing a part between two established poses

To build a lid, derive its closed pose on the hinge with `hinge_landing` (above). `hinge_motion` serves a different
case: a part whose two poses already exist and must swing between them as one piece.

`hinge_motion(reference, end, members, pivot, pace, base=None, weights=None)` finds the best rotation about a fixed
pivot that takes the members' reference positions to their end positions (travel-weighted), turns every point by the
paced fraction of that angle about the same axis and carries the paced fraction of its own residual, so both end poses
are exact and the piece keeps its shape in between as far as its two poses allow. `weights` blend the hinged positions
into `base` positions (for example `path_positions` output): 1 inside the piece, falling to 0 where it joins material
that moves differently. `rigid_share` reports how much of the piece's travel one rotation explains; a low share means
the residual, carried linearly, dominates.

Use it for a part whose end pose swings it a long way relative to its surroundings (the outer wing of a lid that folds
back at a corner). Per-point paths shear such a part into streaks or a flap, and `relax_displacement` smooths its
displacement but cannot rotate material, so spreading a large swing with it creases the corner. Pick the members from
the part's closed-pose travel, pivot it where it stays attached (the corner), and blend it over a band wide enough that
the joining material bends rather than folds; check with `section_turns`, `compare_stretch` and matched renders.
Read `rigid_share` before relying on the turn: in real use a lid's outer crescent that tucks back in the closed pose
had a share of about 0.05, a slide rather than a swing, and the hinge then mostly gave that part one shared pace
(which removed shear streaks) on near-straight paths. Keep the part's blend away from rows an attachment follows.

## Measure the bend of a moving band

`construction_diagnostics.section_turns(reference, poses, triangles, origin=..., normal=..., min_turn_degrees=4.,
window=3)` cuts the reference surface with a plane, follows the same material into every pose and counts inflections
(changes of turning direction between significant windows of the section) per pose. A band rolling over a round
obstacle keeps its turning direction; an S adds inflections. Compare candidates on the same planes (several along the
band), and read `section_points` to plot the sections. Deliberate creases and folds also turn: counts are diagnostic
choices, not an approval.

## Keep the moving surface outside the obstacle

`keep_clearance(positions, obstacle, centre, clearance, angular_radius_degrees=2., soft=0., envelope='smooth')` keeps
points (one pose or one row per phase) outside a star-shaped obstacle seen from `centre`: a point closer than the
obstacle's envelope in its direction plus its `clearance` moves outward along its own direction, with an optional smooth
ramp (`soft`) so pushed and unpushed neighbours join without a crease. The envelope is measured from the obstacle points
within the angular radius (at least twice their spacing as seen from the centre). The default `smooth` envelope is a
kernel-weighted near-maximum that varies smoothly with direction, and pushes fade out smoothly at the obstacle's edge.
`envelope='max'` (the largest distance in the cone, the 0.2.49 behaviour) steps whenever an obstacle point enters or
leaves the cone and the pushes step with it: in real use that rippled lid skin into fine parallel wrinkles. The result's
`envelope` gives each point's measured envelope. Moving obstacles (an eye that turns during a blink) should be sampled
at the same phase as the surface.

`end_clearance(reference, end, obstacle_reference, obstacle_end, centre, cap=None, angular_radius_degrees=2.,
envelope='smooth')` gives each point the clearance to keep between two established poses: the smaller of its clearances
in the two poses (the obstacle sampled in each), capped at `cap`, and -1 (left alone) where either pose is inside the
envelope or uncovered. Cap it at the margin that matters, for a lid about the smallest clearance of the rows that ride
on the eye. Uncapped, material far from the obstacle has to keep its whole end distance and is pushed sideways wherever
its chord dips toward the centre or the envelope estimate varies: in real use the skin beside an eye's outer corner,
which never came near the eye, was pushed and wrinkled at the end of the blink. Capping at just under the rim rows'
smallest clearance removed the wrinkles, and a screen with the eye drawn showed it did not let the eye through.

Clearance is not a contact certification: check sections, stretch and renders with the obstacle drawn (a view without
it cannot show the obstacle coming through). It also fixes penetration, not every gap that shows the obstacle. In real
use, eye visible at a lid corner came from the opening's own end staying open while the rims' paces differed, not from
the lid passing through the eye: measure the gap between the two rims near the corner through the motion
(`closing_edges`). Where the rims close at different rates the construction is the cause; one pace on a hinge removes
it, and a pace floor on a corner adds a zipper.
