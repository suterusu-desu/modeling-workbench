# Guides and display: register, overlap, show

The owner's loop: an image made from all the character's references, a mesh generated from it, the mesh registered
onto the accepted character, and the result shown overlapped on the live model (the editable character solid orange,
the guide a blue wire cage in the same coordinates, never side by side). This page covers registering the guide, what
each guide may be used for, and every way to show the work: live in Blender, as overlap renders, as a motion video
with several columns and as an overlay on a drawing.

## Register a generated guide

```python
from modeling_system.registration import register_rigid, apply_registration

result = register_rigid(generated_co, neutral_co, reference_triangles=neutral_tri,
                        exclude=[{'min': eye_min, 'max': eye_max, 'mirror': 0}, {'min': mouth_min, 'max': mouth_max}])
registered = result['positions']
```

The generated mesh is scaled by the ratio of heights along the up axis (it must come upright, as image-to-mesh output
of an upright head does; otherwise give `scale`), placed by its bounding box and aligned by rigid trimmed ICP on
stationary anatomy only: moving features are left out with `exclude` boxes in the scene frame (`mirror` covers both
eyes with one box) or a `stationary` mask over the reference vertices. The scale stays fixed during ICP; a free scale
collapses toward whatever fits best. ICP starts point to point and refines point to plane (normals from
`reference_triangles`), so it settles on the reference surface instead of within a vertex spacing of it. The report
gives the distance to the reference surface on the stationary anatomy (how well the two agree where they should) and
on the excluded region (where the guide is meant to differ). `apply_registration` carries the same transform to a
separated part of the generated mesh, such as its lashes.

In real use, on a generated face mask registered onto an accepted neutral, the packaged registration converged in 45
iterations and landed within .0007 (median) of the registration made earlier with a private script that stopped at 60
iterations without converging.

Registration measures nearest distances, not correspondence. A mesh that differs everywhere (a different face)
registers to a compromise: look at the overlap before fitting anything to it.

## What each guide is for

Every guide carries a role in the workspace binding:

- `identity`: the accepted rest the character must match exactly (the neutral).
- `pose_check`: a pose of a moving feature, used to fit the mechanism's few parameters and to check it by overlap and
  volume ([build the mechanism first](build-the-mechanism-first.md)); never per-point targets at its phase.
- `detail`: local form for a region, within its qualified support.

The reference adapter reports the role of the guide on show, and the live header names it.

## Live: the reference adapter and the live bridge

Any character workspace can drive a visible Blender without writing its own adapter. Describe the scene once:

```json
{"working": "Face", "character": ["Face", "Body", "Eyeball.L"],
 "guides": {"Neutral": {"object": "Guide neutral", "role": "identity"},
            "Closed": {"object": "Guide closed", "role": "pose_check", "controls": {"blink": 1.0}}},
 "controls": {"object": "Controls", "properties": ["blink"]},
 "shape_keys": {"object": "Face", "keys": ["Blink"]},
 "restore": {"path": "restore_rig.py", "function": "restore"}}
```

```sh
python -m modeling_system.live bind --workspace /path/to/character --reference reference.json --owner my-owner
python -m modeling_system.live launch --workspace /path/to/character --blender /path/to/blender --file work.blend
```

`bind` pins the installed reference adapter and the restore hook in `modeling-workspace.json`; rebind after editing the
hook or upgrading the package, since stale pins are refused. `launch` starts Blender with an isolated profile, no
startup file and no auto-run scripts, running `live_bridge`, which claims the live file for the owner and serves the
workbench's native bridge on the binding's port. Then, from Python:

```python
from modeling_system.native_bridge import NativeBridge
bridge = NativeBridge('/path/to/character', port=9876)
live = bridge.call('inspect_live', {}, owner='my-owner')['live']
live = bridge.call('set_controls', {'expected_state': live['expected_state'], 'guide': 'Closed'}, owner='my-owner')['live']
live = bridge.call('set_display', {'expected_state': live['expected_state'], 'mode': 'GUIDE_WIRE'}, owner='my-owner')['live']
```

`set_controls` poses the declared custom properties and shape keys and shows one guide as a depth-tested blue wire
(with only `guide` given it applies that guide's own pose). `set_display` colours the working mesh orange and the rest of
the character grey (`GUIDE_WIRE`), or switches to clay or materials, sets parts aside with `hide` and frames the view.
`open_checkpoint` opens an exact saved file and runs the restore hook, `bootstrap` runs the hook on the current file and
`save_checkpoint` saves a new file without overwriting a different one. Every change needs the current
`expected_state`; presentation never changes the character's geometry (the content fingerprint says so). The 3D view
header names the owner, the file, the controls and the guide shown with its role.

The restore hook is where a rig that needs setup after loading (drivers, a private bootstrap) gets it:
`restore(bpy, workspace, config)` in a file inside the workspace. Characters whose rig works as saved need none. A
workspace with a richer private adapter keeps using it; the reference adapter is the default, not a requirement.

`--background` serves without a window (tests, automation) until a shutdown request. Anything that can reach the port
can run code in that Blender: it listens on 127.0.0.1 only.

## Overlap renders

`overlap_views.py` is a NativeJob worker for the isolated runner. It poses the character through the same `reference`
configuration and renders the working mesh solid orange with a guide as a blue wire cage from orthographic views
around a target: one image per step and view, plus `overlaps.json`.

```json
{"script": ".../modeling_system/overlap_views.py", "blender": "...", "input": "work.blend", "output_root": "renders",
 "workspace": "/path/to/character", "reference": {"...": "as above"},
 "steps": [{"label": "open", "controls": {"blink": 0}, "guide": "Neutral"},
           {"label": "closed", "controls": {"blink": 1}, "guide": "guides/closed-registered.npz"}],
 "views": [{"name": "front", "yaw": 0}, {"name": "three-quarter-left", "yaw": 40}],
 "target": [0.1, -0.2, 0.62], "ortho_scale": 0.2, "size": 900}
```

A guide is a name from `reference.guides` or a registered `.npz` (`co`, `tri`). The cage is a Wireframe modifier
about 1.5 pixels thick unless `wire` is given; a wire display type would render solid.

## Motion videos with several columns

`import_motion` takes a manifest whose frames hold, per column, one captured image per view; `export_replay` encodes
normal speed then slowed playback from the nearest captured pose. Columns default to `before` and `after`; list
`"columns": ["A", "B", "C"]` (up to six) with `"labels": {"A": "lower lid still", ...}` to compare variants of one
decision in one video with the same timing.

## Overlay on a drawing

```python
from modeling_system.review_sheets import aligned_overlay
aligned_overlay('render-front.png', 'closed-drawing.png', render_corners, drawn_corners, 'overlay.png',
                crop=(x0, y0, x1, y1), lines=[{'points': drawn_closed_line, 'color': [220, 40, 40]}])
```

The render is mapped into the drawing's frame by the similarity taking two landmarks (for an eye, its two corners)
onto the drawn ones, and the drawing is laid over it, its dark strokes stronger. A whole-head calibration can leave
a feature several pixels off; two local landmarks make that feature coincide, so what differs is the shape, not the
placement. `construction_diagnostics.line_depth` measures a traced drawn line and a model edge the same way (sag below
the corner line as a share of the corner distance), for example the approved closed drawing's closed line against the
model's closed seam.
