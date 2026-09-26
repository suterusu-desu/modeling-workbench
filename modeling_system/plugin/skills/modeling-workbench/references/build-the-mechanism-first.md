# Build the mechanism first

Read this before changing how anything moves: a blink, a jaw, a mouth shape, an expression. A moving feature is a
mechanism before it is a set of poses. Build the mechanism the reference avatars use, keep the rest pose on the accepted
neutral, and let the pose guides tune the mechanism's few parameters and check it by overlap. Do not make motion by
fitting every point to a guide at every pose.

## Why

A motion made of per-point paths or keys, each fitted to a pose guide, can sit close to every guide and still look wrong
in motion: each point keeps its own path and its own timing, so the surface morphs between keys instead of moving as a
part. Corrections fitted on top make it worse, because each one is fitted separately. Reference avatars move the same
features as parts: a small band of loops turns or slides as one piece, the corners and the opposing lid stay put, and
one control with one timing curve drives it.

In real use a blink was built for two weeks as about 58,000 per-point cubic paths written after subdivision, with
per-phase guide fields and local re-lays stacked on top. Fifteen retained checkpoints passed every declared numeric
check. The first normal-speed video showed a lid that morphed "like putty", closed from one corner first "like a zipper"
and pulled the lower lid up. Rebuilt as one hinged lid (the whole margin turning about an axis through the eyeball's
centre on one timing, landing on the still lower lid), it was judged better within about an hour, using a few of the
functions below.

## The order

1. **Study the construction in the reference avatars** you were given, before touching the character. Measure, do not
   only look: which loops move, whether they turn or slide and about what, how far the moving band reaches, what stays
   still (corners, the opposing lid, the socket behind the margin), where the closed line lies, how attached parts such
   as lashes ride, how many controls there are and how their timing is shaped. Extract the avatar's shapes as arrays and
   run the checks below on them: the avatars are the passing examples.
2. **State the mechanism** in a few lines: what moves, about what axis or in what direction, what stays still, the one
   timing law, where it lands, what it carries, and which avatar shows it.
3. **Keep the rest pose exactly on the accepted neutral.** The mechanism starts from it; nothing fitted later moves it.
4. **Build the mechanism and derive the end pose from it.** For a lid: land the margin on the opposing lid's rest line by
   turning it about the hinge, and build every in-between as one roll with one pace for every point.
5. **Fit the mechanism's parameters to the pose guides**: the axis, the landing line, the band's falloff, the timing
   curve. Use the guides as overlap and volume checks (orange mesh, cyan guide wire, same coordinates). A pose guide is
   not a per-point target at its phase.
6. **Check it and watch it**: the checks below, then a normal-speed and a slowed video of the whole motion compared with
   the last approved state, before any checkpoint of it is retained.
7. **Fix a motion defect by changing the mechanism**: a parameter, the band, the axis, the landing line, the topology. Do
   not stack a per-phase guide field, a per-point path or a local patch on top.

Static shape (the rest pose, smoothing, joins) is still fitted to the registered guides as the rest of this skill
describes. [Motion paths](motion-paths.md) re-time or join an inherited motion; they do not replace this order.

## Symptoms and their causes

| What you see in motion | Usual cause | Change |
|---|---|---|
| The edge closes from one corner first ("zipper") | per-point pace: corners on their own schedules, pace floors or shared schedules that close a corner early | one pace for the whole margin |
| The part morphs between keys ("putty") | per-point paths, keys at intermediate phases, per-phase guide fields | one roll or slide of the whole band, with the end pose derived from it |
| The opposing lid rises, evenly and early | the closed pose's seam sits above the opposing lid's rest line, so the lower rim is pulled up to meet it | land the moving edge on the opposing lid's rest line; if the design wants a rise, build it into the mechanism (middle only, zero at the corners, same timing) |
| Skin beside the moving part sinks or pulls in ("sucking in") | stacked correction fields fighting each other; retreat left behind by a fitted field | remove the layers; reshape or widen the moving band |
| Lumps, short marks and creases that move around between fixes | local re-lays and patches fitted on top of an inherited motion | rebuild the motion from the mechanism, then clean up once |
| The lid cuts into the eye at mid-blink | a straight rest-to-closed chord over a round eye | roll about the hinge; keep clearance against the real eye surface |
| Lashes separate from the lid or collapse | lashes follow separately stored shapes | carry them on the lid edge with the host's turn |

## The hinged lid

The upper lid turns as one piece about an axis through the eyeball's centre, along the line between the two corners.
The lower lid and the corners stay still unless the design says otherwise. Build it from the accepted rest pose:

```python
import numpy as np
from modeling_system.motion_paths import hinge_landing, path_positions, end_clearance, keep_clearance, hinge_carry
from modeling_system.construction_diagnostics import closing_edges

closed = hinge_landing(rest, margin, rest[lower_margin], eye_centre, axis,
                       weights=band, outside=small_gap)['positions']
pace = np.repeat(phases[:, None], len(rest), axis=1)           # one pace for every point: one timing
poses = path_positions(rest, closed, pace, pivot=eye_centre, axis=axis, roll_weight=over_eye)['positions']
need = end_clearance(rest, closed, eye_hull, eye_hull, eye_centre, cap=rim_clearance)['clearance']
poses = keep_clearance(poses, eye_hull, eye_centre, need, soft=ramp)['positions']
lashes = hinge_carry(lash_rest, rest[margin], closed[margin], eye_centre, axis, fraction=phases)['positions']
report = closing_edges(rest, poses, margin, lower_margin, pivot=eye_centre, axis=axis)
```

- `hinge_landing` turns each margin point about the axis onto the landing line's angle and distance at its own axial
  position, plus `outside` so it rests just in front of the lower margin, and carries the band by its weight: 1 on the
  margin and the lid's inner surface, falling off with rest distance above the margin, 0 on what stays (the lower lid,
  the corners). Nothing slides along the axis. Margin points beyond the landing line's axial range take its end values
  and are counted.
- `path_positions` with a pivot and axis rolls every point from rest to closed at the pace given, so one pace row per
  phase is one timing for the whole lid. `roll_weight` is 1 over the eye and falls to 0 beyond its outline, where skin
  should move on chords.
- `end_clearance` gives each point the clearance to keep, capped near the rim's own; `keep_clearance` keeps the rolled
  lid outside the eye.
- `hinge_carry` moves attached points by the turn, radial and axial change of the host point they sit on, at the same
  fraction, so lashes stay seated instead of following stored shapes. Its per-point `turn`, `radial` and `axial` values
  are what a native node group needs: P(f) = pivot + (s + f axial) axis + R(axis, f turn) r (|r| + f radial) / |r|, with
  s and r the point's axial and radial parts about the axis and R right-handed (Blender's Vector Rotate node, axis-angle
  type, turns right-handed).
- `closing_edges` measures the closing; see the checks below and [construction diagnostics](construction-diagnostics.md).

`hinge_change` gives the same per-point turn, radial and axial values between any two poses, for example to realize an
already built closed pose in a rig.

Found in real use:

- An axis through the two corners themselves turns the lid about 95 degrees, like a visor. Put it through the eyeball's
  centre, along the line between the corners.
- A band turned rigidly to its top keeps its thick fold off the eye and reads as a puffy dome. Let the band's weight fall
  off above the margin (the full turn reached about a fifth of the eye's width up); the inner surface turns with the
  margin at full weight.
- Clearance measured against a smooth envelope of the raw eye points let the iris ring poke through. Keep clearance from
  a max-filtered hull of the eye, about one percent of the eye's width near the margin and more away from it.
- Re-laying the lid as a curtain with a free boundary left pinches at the corners and a U-shaped outline, and
  projecting the lid onto a guide surface copied the guide's facets. Turn the lid; do not re-lay or project it.
- A roll about the eye's centre swings skin beyond the eye's outline the wrong way: roll over the eye, chords beside it.
- With the lower lid still, the closed line follows the lower lid's rest curve. A deep lower lid then closes on a deep U.
  The reference avatars close on a shallow line by raising the middle of the lower margin on the same timing; which
  closed line the character should have is a design decision, measured against the approved closed drawing.
- The attachment's rig value is already the geometric phase: mapping it through the lid's timing curve again ran the
  lash ahead of the lid. A surface-deform binding of the lashes to the skin stood the lash fins upright; carry them on
  the hinge instead.

## Checks that catch a wrong mechanism

Numbers can reject a candidate; they cannot approve one. Run these on the avatars first to see passing values.
[Standard construction checks](construction-checks.md) run all of them from one declaration and guard native trials
with them.

- **One closing rate.** `closing_edges` gives each margin point's closed share of its gap to the opposing rest line and
  its turn share, per pose and per stretch of the margin. A spread of more than about a tenth at mid-motion is a zipper.
  In real use the rejected blink's inner, middle and outer thirds had turned .50, .58 and .84 of the way at the same
  phase; the hinged rebuild .64, .64 and .65.
- **Opposing lid travel.** `facing_travel_share` is the opposing edge's largest travel as a share of the rest opening:
  .53 in the rejected blink, 0 in the rebuild. The reference avatars raise only the middle of the lower margin (7-30 %
  of the opening, zero at the corners, on the same timing), or barely at all.
- **Where it closes.** `seam_to_facing_rest` measures the closed edge's distance from the opposing lid's rest line.
- **One piece.** With the hinge given, `roll_deviation` measures how far the moving edge lies from one roll at the
  median pace. Material on separate paths or pushed by stacked corrections deviates.
- **Band and reach.** How much of the surface moves, and how far above the margin the motion stops. The avatars move
  about 170-190 base vertices per eye, and the motion stops within one or two loops above the margin's band; the
  rejected blink moved four times as many evaluated points as the hinged rebuild.
- **Section bends, crowding and folds.** `section_turns`, `compare_stretch`, `local_reversals` and `compare_bends`
  ([construction diagnostics](construction-diagnostics.md)).
- **Clearance.** The lid, its inner surface included, stays outside the eye at every phase and at the gaze limits.

## Production form

If the result ships as blend shapes, check how many shapes carry the motion. One straight rest-to-closed shape cuts a
round eye at mid-blink. In real use one mid-blink corrective driven at 4s(1-s) by the same control reproduced the built
roll within a few thousandths of the eye's width and kept clearance, for the skin and the carried lash. The avatars ship
one shape per channel, since their eyes are flat and recessed, and shape the timing with the animation curve. Timing is
a curve over a few shapes, not a per-point path.

## Notes from reference avatars

- The lid is the band above the margin: it moves by about the margin's amount, and the motion stops within one or two
  loops above it. The stretch goes into the loop above, where a drawn crease can sit.
- Around the eye the topology is concentric closed quad loops with equal counts and no poles in the moving band. The
  socket pocket behind the margin stays still, so the strip from margin to pocket stretches as the lid closes.
- Lashes are separate strips on the same control, moving about the margin's travel with a small turn about their root.
- The closing corners move a little (a few hundredths of the eye's width), not exactly zero; forcing them to zero is
  not needed.
- Blink and strong eye expressions are not simply added: the avatars ship closed variants of expressions as their own
  shapes and switch the automatic blink off in those states.
- A study is finished when its finding is built: a mechanism, a check, or a recorded rejection with its reason. The
  avatar finding above was measured more than once and not built for two weeks, and the blink kept failing meanwhile.
