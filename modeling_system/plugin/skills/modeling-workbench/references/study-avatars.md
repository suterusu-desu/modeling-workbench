# Study the reference avatars

When reference avatars (rigs of other characters whose motion reads well) are given, learn how they construct a
feature before building it for the character, then build that construction ([build the mechanism first](build-the-mechanism-first.md)).
A study is finished when its finding is built: a mechanism, a check or a recorded rejection with its reason.

## Extract

Run `study_extract.py` as a NativeJob worker on the avatar (a .blend, or an FBX through `import_fbx`):

```json
{"script": ".../modeling_system/study_extract.py", "blender": "...", "input": "avatar.blend",
 "output_root": "study/avatar-name", "object": "Body"}
```

It writes `<object>.npz` and `.json` without changing the avatar: rest positions in world space, topology, every
shape key that moves anything as a world-space delta, key metadata and drivers, vertex-group weights and armature bones.
`study.load_shapes(npz)` reads it back: `rest`, `edges`, `polygons`, `keys` by name, `bones`, `vertex_groups`.

## Find the parts

Pick the lid margin's vertices once (for example the loop where the outer skin turns into the socket) and order them
with `order_loop(vertices, edges)`. `rings(rest, edges, margin, polygons=avatar['polygons'])` numbers the loops outward
over the outer face (+1, +2, ...) and inward over the lid's inner surface and the socket pocket (-1, -2, ...), the two
sides told apart by the faces on either side of the margin (without polygons, by the front view, which misreads the
avatars' pockets: they reach wider than the opening). The corners are
the margin vertices where the blink's vertical motion changes sign. `loop_topology(rest, polygons, margin,
travel=...)` reports whether the margin and the next loops are closed quad loops of one count, their poles, the inner
rings and how many vertices move (by more than a millionth of the largest travel) (the service's `study_blink` adds it with `margin_loop`): a report to compare with
the avatars, not a gate.

## Measure the construction

```python
from modeling_system.study import load_shapes, motion_models, band_profile, blink_report
avatar = load_shapes('study/avatar-name/Body.npz')
rest = avatar['rest']; closed = rest + avatar['keys']['blink']
report = blink_report(rest, closed, upper_margin, lower_margin, corners, pivot=eye_centre, axis=canthal_axis,
                      band=upper_face_vertices)
```

- `motion_models(rest, end, members, axes={...}, guess=...)`: how much of a part's motion one slide, a turn about a
  named axis and the best free axis leave unexplained (share of the travel).
- `band_profile(rest, end, margin, candidates)`: travel as a share of the margin's travel against rest height above
  the margin at the column that moves most, and `reach_share`, where the band stops.
- `blink_report`: moving vertices, the lower lid's travel as a share of the upper's, each corner's travel, the closed
  line's depth and the lower lid's rest depth below the corner line (all in the eye's own width), the upper margin's
  motion models and the band profile.

For the face: `rigid_motion(rest, end)` fits a part's rigid turn (angle, axis, a point on the hinge, residual),
`jaw_weights(positions, delta, R, t)` how much of that turn each vertex follows and how much of its motion is off it,
`viseme_mix(basis, target)` a viseme's nearest non-negative mix of base shapes, and `seam_rings(edges, seam, count)`
ring numbers out from the lip seam. The [face audit](face-checks.md) runs the checks built on them.

Run the same numbers on the character's candidate and put the ones that matter into its declaration
([standard construction checks](construction-checks.md), [face audit](face-checks.md)): the avatars are the passing
examples.

## What three avatars showed

Measured on three commercial anime avatars (eye width w = the distance between the eye's corners):

| Measure | Avatars |
|---|---|
| Upper lid motion unexplained by one slide | 7-16 % |
| Unexplained by a turn about the eye's gaze pivot / by a visor through the corners | 23-34 % / 79-93 % |
| Visible lid carried as one piece | up to .2-.4 w above the margin, motion gone by .53 w |
| Closed line depth below the corner line | .035-.046 w |
| Lower-lid rise at the centre | 7-30 % of the opening, middle only, same timing as the upper lid |
| Corner travel | .02-.10 w |
| Moving vertices per eye | about 170-190 (base mesh) |
| Loops round the opening | margin and the next three closed, one count each (28-37), quads; poles from ring 3 out |
| Lash strips | separate, 90-104 % of the margin travel under them, 16-33° turn, 0.86-1.23x length |
| Shapes per channel | one; timing from the animation curve |

Their eyes are flat and recessed, so a straight slide clears them. A round eye close behind the lid needs the turn
about its centre (and, as blend shapes, one extra mid-blink shape). A character whose lower lid is deep at rest closes
on a deep line unless its lower lid rises; how shallow the closed line should be is decided against the approved
closed drawing (`line_depth`). These are measurements of other characters' construction, not targets for the
character's likeness.
