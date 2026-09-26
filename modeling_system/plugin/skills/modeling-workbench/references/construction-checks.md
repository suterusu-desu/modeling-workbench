# Standard construction checks and trial guards

Declare once what a moving feature is allowed to do, and every candidate is measured the same way: offline with
`checks.run_checks`, and in native trials and retention through the policy `checks.standard_policy` builds. No
per-trial preservation adapter is written. The checks put [build the mechanism first](build-the-mechanism-first.md)
into numbers. Numbers can reject a candidate; they never approve its appearance, so watch the motion as well.

## Poses

Saved poses are arrays named `<phase>::<object>::co` (vertex positions) and `<phase>::<object>::tri` (triangles; only
the rest phase's are needed), in one .npz file or one mapping. Phases are numbers and the lowest, 0, is the rest pose.
`load_states(path)` reads them in phase order. A native trial saves the candidate's poses (`evaluated.npz`) and the
source's (`source-evaluated.npz`) beside the saved candidate file.

## The declaration

A small JSON document. Only `object` and `region` are required; each other field adds its checks. Index lists and
arrays can be inline or `{"path": "file.npz", "key": "name"}` relative to the declaration file.

```json
{
  "object": "Face",
  "region": {"path": "eye-region.npz", "key": "region"},
  "region_label": "left eye area",
  "rest": {"path": "accepted-neutral.npz", "key": "co"},
  "protected": ["Eyeball", "Teeth"],
  "clearance": [{"obstacle": "Eyeball", "minimum": 0.0005}],
  "symmetry": {"axis": 0},
  "closing": {"moving": [...upper margin, ordered...], "facing": [...lower margin, ordered...],
              "pivot": [0.1, -0.2, 0.6], "axis": [0.95, 0.30, 0.10]},
  "carrier": {"tolerance": 0.0007},
  "limits": {"facing_travel_share": 0.25}
}
```

- `region`: vertex indices (or a boolean mask) of the object allowed to move. Everything else must stay at rest.
- `rest`: the accepted neutral positions of the whole object.
- `protected`: objects the candidate must not change (compared with the baseline, or with their own rest).
- `clearance`: obstacles the region must stay outside. `centre` defaults to the obstacle's centroid at each phase,
  which suits an eyeball; the obstacle must be star-shaped from it.
- `symmetry`: the mirror axis (0, 1 or 2) and `plane` (default 0), for an object that holds both sides.
- `closing`: the moving and facing edges, ordered along the edge, and optionally the hinge (`pivot` on the axis,
  `axis`) for the roll check; `parts` (default 3) stretches along the edge.
- `carrier`: the tolerance within which blend shapes must reproduce the motion; `weights` maps phases to shape weights
  when they differ from the phase values.
- `limits`: overrides of the defaults below; `arrays` renames the trial's pose files.

## The checks

| Check | Measures | Default limit |
|---|---|---|
| `rest_identity` | largest distance from the accepted rest at phase 0 | 1e-6 |
| `still_outside` | largest travel of any vertex outside the region, any phase | 1e-6 |
| `protected_unchanged` | largest change of a protected object | 1e-6 |
| `clearance_shortfall` | how far a point that starts outside the obstacle comes inside the smaller of the minimum and its own rest clearance | 0 |
| `folds` | locally reversed region triangles at the worst phase (`local_reversals`) | 0 |
| `reversing_vertices` | movers whose progress along their own rest-to-end chord falls back by more than 2 % | 0 |
| `symmetry` | largest distance between a vertex and its mirrored partner's mirror image | 1e-5 |
| `facing_travel_share` | the facing edge's largest travel, share of the rest opening (`closing_edges`) | .05 |
| `closing_spread` | largest difference in closed share between stretches of the edge, before closure | .1 |
| `seam_share` | the closed edge's median distance from the facing rest line, share of the opening | .05 |
| `roll_deviation_share` | the moving edge's distance from one roll about the hinge, share of the opening | .1 |
| `carrier_shapes` | blend shapes needed: one straight shape, or with a mid shape driven at 4s(1-s) | 2 |

`clearance_shortfall`, `folds` and `reversing_vertices` count defects a baseline can already have. With a baseline, the
limit is an allowance over the baseline's value: a candidate is not failed for what it inherited, the report still shows
the inherited value, and a rebuild is not held to its predecessor's geometry. The other checks describe the construction
itself and stay absolute, so a rebuild of a failed mechanism has to pass them. A design that wants the lower lid to rise
raises `facing_travel_share` in the declaration, where the choice is visible.

Each row reports `observed`, `limit`, `rule`, `status` (pass, fail or unknown), `detail` (where and when, per-stretch
medians, carrier errors and worst vertices) and, with a baseline, `baseline_observed`. A check that cannot be measured
(no triangles saved, a closing spread with one pose) is unknown, and unknown blocks retention.

In real use, on one character's saved native arrays, a blink built as per-point paths with stacked corrections (rejected
on sight) failed every closing check: facing travel .55 of the opening, spread between the lid's thirds .50, seam .47,
roll deviation .27. Its hinged rebuild passed them: 0, .07, .01, .06. Against the rejected blink as baseline, the rebuild's
folds (116 against 171), reversals (29 against 211) and one skin point about .003 inside the eye at closure were
inherited and passed as such. The carrier check found about 2,400 vertices, all outside the hinged band and still on the
old per-point motion, that two blend shapes could not carry within .0007. Both checks ran in about three seconds on a
58,000-point skin.

## Guards for native trials

```python
from modeling_system.checks import standard_policy

policy = standard_policy('eye-declaration.json', 'runtime/sessions/c20-requirements.json',
                         construction='hinge_build.py', authority=['PROJECT.md'], guide='neutral-guide.npz')
```

The result is an ordinary `PreservationPolicy` with one outcome whose cells are the declared checks (the increase-only
ones as `max_increase` cells), bound with the packaged adapter name `standard`: pass it where a hand-written policy was
passed, and bind trial and retention tasks with `adapter='standard'`. Its measurement reads the trial's poses from the
folder of the result's `subject` (at retention, from the measured trial), so the trial must save both pose files there.
The requirements file is written once per declaration; to change the declaration, write a new file. Custom cells
remain possible through a hand-written adapter when a question is not a standard one.

## Limits

- Sampled phases do not certify the motion between them; triangles are the rest phase's.
- Clearance uses the obstacle's envelope seen from its centre. Near the obstacle's outline the envelope is uncertain,
  and a point that crosses it can read a small false shortfall; confirm a reported penetration against the surface.
- How many correction layers are stacked in a region is a property of the native construction and is not measured
  from saved poses.
- The checks do not judge likeness, shading or whether the motion reads as the character's.
