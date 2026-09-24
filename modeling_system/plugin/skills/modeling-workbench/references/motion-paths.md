# Build in-between poses from two established poses

Use these operations when both end poses of a motion are established (an accepted rest and an accepted end pose)
and the in-between poses bend, crease or cut into something behind the moving surface. They are `ArrayPreparation`
operations and plain Python functions in `modeling_system.motion_paths`; they fit no guide and approve no appearance.

## Why in-betweens go wrong

A blend shape moves every point along its straight chord, all at one pace. Reference characters with flat, recessed
eyes close their lids that way without trouble: nothing lies behind the lid's path. When material has to pass over a
convex obstacle close behind it (a lid over a round eye, a sheet over a bone), the chord cuts into the obstacle. The
usual repairs fit extra keys at intermediate phases and stack correction fields on top; each is fitted separately, so
the in-between sections bend: across a moving band the section turns one way above its edge and the other way at it
(an S), which shading shows as a crease or shelf even when every key is close to its guide. Measure that with
`construction_diagnostics.section_turns` before and after a change (see below).

## Pace: keep when each region moves

`motion_pace(reference, end, samples, sigma=..., cutoff=None, travel_scale=None, minimum_travel=0.)` measures, for an
existing motion sampled at increasing phases, how far each point is along its own reference-to-end chord, clipped to
[0, 1], smoothed over reference-pose neighbours with a travel weight (points that barely move take the pace of the
material around them) and made monotone. It keeps the existing motion's timing while its paths are replaced.

Keep the host's own pace wherever something attached to it follows a schedule of its own (a lash strip that turns on
the lid rim at fixed phases). Re-timing the host, even by smoothing its pace along the band, desynchronizes the
attachment: in real use a rim that lagged by a few percent opened a gap between the lash and the lid, and rows above
the rim that lagged the rim let the eye show through the band. Smooth the pace in radial rows only where no attachment
depends on it, or move the attachment with it.

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

## Hinge: move a part as one piece

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

## Measure the bend of a moving band

`construction_diagnostics.section_turns(reference, poses, triangles, origin=..., normal=..., min_turn_degrees=4.,
window=3)` cuts the reference surface with a plane, follows the same material into every pose and counts inflections
(changes of turning direction between significant windows of the section) per pose. A band rolling over a round
obstacle keeps its turning direction; an S adds inflections. Compare candidates on the same planes (several along the
band), and read `section_points` to plot the sections. Deliberate creases and folds also turn: counts are diagnostic
choices, not an approval.
