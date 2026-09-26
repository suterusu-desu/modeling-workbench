# Construction and failure notes

What real modeling taught, in the form an agent can use before the next edit: what you see, why it happens, what to
do. Read the notes for the feature you are about to change. Add a note when a finding is general and has changed
what gets built; keep scoped evidence in the workspace's own records.

## Motion

**The part morphs between keys ("putty").** Cause: motion made of per-point paths or keys fitted to pose guides one
phase at a time, often written after subdivision. Each point keeps its own path and timing. Remedy: build the mechanism
the reference avatars use (a band turning or sliding as one piece, one timing) and derive the end pose from it.
[Build the mechanism first](build-the-mechanism-first.md). A soft falloff that spreads the motion into the brow is the
same failure seen from above: the avatars' bands stop by about half the eye's width (`band_reach`).

**The edge closes from one corner first ("zipper").** Cause: separate paces for parts of one edge: pace floors, shared
schedules on a corner, an inner end kept late. Remedy: one pace for the whole margin; the `closing_spread` check fails a
spread of more than a tenth between the edge's thirds. [Construction checks](construction-checks.md). The corners
themselves move little (the avatars' .02-.10 of the eye's width) and the margin's travel rises smoothly from each: the
`corner_ramp` check fails a margin at full travel within a fifth of its length from a corner.

**The opposing lid rises, evenly and early.** Cause: a stored closed pose whose seam sits above the opposing lid's rest
line, so the other rim is pulled up to meet it. Remedy: land the moving edge on the opposing rest line (`hinge_landing`);
if the design wants a rise, build it into the mechanism, middle only and on the same timing (`hinge_lift`).

**Skin sinks, lumps and creases move around between fixes.** Cause: correction fields and local re-lays stacked on an
inherited motion, each fitted separately. Remedy: remove the layers and rebuild the motion from the mechanism; clean up
once afterwards. A defect that returns after a patch belongs to the construction.

**A straight blend shape cuts the eye at mid-blink.** Cause: a round eye close behind the lid; the chord from open to
closed passes inside it. Remedy: roll about the eye's centre; as blend shapes, one mid-blink corrective driven at
4s(1-s) usually carries the roll (`carrier_shapes` check).

**Lashes lift off or collapse during the blink.** Cause: attached parts following their own stored shapes. Remedy:
carry them on the moving edge with the host's turn (`hinge_carry`), in the offline build and in the rig. The `lash_*`
checks fail a lash left behind, lagging, flipping on the lid (more than 35°) or stretching.

**Blink plus an expression crosses or stays open.** Cause: blend shapes add; a lower-lid-raising expression plus the
blink puts the upper edge past the lower (the avatars: 3-15 % of the opening), a surprised lid does not close. Remedy:
dedicated closed variants of the strong expressions and the blink switched off while they play, as the avatars ship
them; declare the combinations so `combination_seam` measures each.

**The mouth bends, slides and turns late ("putty mouth").** Cause: the blink's mistake on the mouth: one control
driving an open shape plus guide keys gated to phases of it, and teeth that slide before they turn. Remedy: a rigid jaw
turn about a hinge in front of the ear, skin following it by a weight, a separate upper-lip lift, stored as one
straight shape; guides fit its angle, weight and lift and check the in-between by overlap. The [face audit](face-checks.md)
refuses phase-gated keys and measures the hinge.

**A face shape moves the eyelid.** Cause: a mouth, cheek or brow shape whose falloff reaches the lid. Remedy: each
shape owns one region; the squint and the smile-closed lid are lid shapes on the eye's own rows (`still_travel_share`
on the lid margin).

## Guides and fitting

**Every key sits near its guide and the motion still looks wrong.** Cause: pose guides used as per-point targets at
every phase. Remedy: rest exactly on the accepted neutral; pose guides fit the mechanism's few parameters and check it
by overlap and volume. [Guides and display](guide-and-show.md).

**A registration looks excellent but the guide is placed wrong.** Cause: a free scale collapsing toward the part that
fits best, registration on moving features, or a target remapped where the report does not show it. Remedy: fixed
scale, rigid alignment on stationary anatomy only, and one registration used for display, fitting and every metric
(`register_rigid`).

**Credits spent chasing a "defect" that was the design.** Cause: generating new guides as a reflex, for a question that
was not about the character's identity (an intended convexity chased through many jobs). Remedy: generate only for a
missing identity shape, name the likeness question first, and ask the owner when the design is in doubt.

## Judging and retaining

**Numbers approved what the eye rejected.** Seen in real use: an improvement measured against a shifted target, a field
predicted to 1e-8 that added a crease, "93 % within tolerance" measured on skin that never moves, a registration
reported under a millimetre because of a bug. Remedy: numbers may reject a candidate, never approve it; judge by the
whole motion at normal speed, overlapped on the guide and compared with the last approved state.

**Fifteen checkpoints passed every check on a mechanism the owner then rejected.** Cause: retention ratcheting on the
previous checkpoint with cells that protected its geometry, while nobody watched the motion. Remedy: retain only after
the motion has been watched against the last owner-approved state; construction checks stay absolute, defect counts
may not grow; an unseen checkpoint is an experiment. [Trial, reopen, retain](trial-reopen-retain.md).

**A study was written up and never built.** Cause: treating a finding as a report. Remedy: a study ends in a built
mechanism, a check or a recorded rejection with its reason. [Study the avatars](study-avatars.md).

## Hidden traps

**Geometry nobody sees still moves.** Seen in real use: half of a mirrored source mesh, cut away by the mirror modifier
and never displayed, kept moving on old per-point paths and ended inside the eye. Remedy: set the declaration's region
to the visible feature so `still_outside` catches it, and make hidden material still.

**Clearance reports a penetration that is not there.** Cause: an obstacle measured from a point that is not its centre
(a point on the hinge axis is not the eyeball's centre), so the obstacle is not star-shaped from it. Remedy: measure
from the obstacle's own centroid (the check's default) and confirm any penetration against the surface.

**The lid clears the eye looking ahead and cuts it looking down.** Cause: clearance checked at one gaze; the eye
turns the cornea or iris bulge into the closing lid's path. Remedy: declare the gaze limits in the clearance entry
(`gaze`), so the check is repeated with the eye turned.

**An overlap render shows the guide as a solid.** Cause: a wire display type renders solid. Remedy: a Wireframe
modifier on the guide copy (`overlap_views.py`).

**A trial fails after its work because a record changed.** Cause: files pinned by hash (prose records, installed
package files) edited while the trial ran. Remedy: do not edit pinned records during a trial; upgrade the workbench
between trials.
