# Bake to blend shapes

A motion built as a mechanism ships as blend shapes driven by a clip. `modeling_system.bake` finds the fewest shapes
that carry it, writes the clip curve and round-trips an FBX.

```python
from modeling_system.bake import bake_shapes, clip_curve, write_bake, export_fbx

bake = bake_shapes({'Face': (face_rest, face_poses), 'Upper lash': (lash_rest, lash_poses)}, phases,
                   tolerance=.0007, obstacles={'Face': {'obstacle': eyeball_points}})
clip = clip_curve(bake['drivers'], timing={'close': .083, 'hold': .033, 'open': .217}, fps=60)
write_bake('blink-bake.npz', bake, faces={'Face': face_triangles, 'Upper lash': lash_triangles},
           rest={'Face': face_rest, 'Upper lash': lash_rest})
record = export_fbx('blink-bake.npz', clip, blender=BLENDER, output_root='runtime/bake')
```

From a trial's saved poses in one call (also the service operation `bake_poses`, on the CLI and MCP): the phases and
the rest triangles come from the file, and the bake file, the clip and (with `renderers`) the Unity .anim are
written beside the output.

```python
from modeling_system.bake import bake_poses
report = bake_poses('runtime/trials/blink04/trial-runs/<run>/evaluated.npz', ['Face', 'Upper lash'], tolerance=.0007,
                    output='runtime/bake/blink04.npz')
```

**Shapes.** The poses are sampled at increasing phases, the main shape's weight from 0 (rest) to 1 (the end pose); for
a one-pace hinge that is the geometric phase. The main shape is the end pose, exactly. Correctives are fitted by least
squares to what the main shape misses and are driven by bumps of the same weight that vanish at 0 and 1: a mid
corrective at 4s(1-s), or early and late correctives. Rest and the end pose stay exact and every weight stays within
0..1 (engines commonly clamp blend-shape weights). The smallest set within the tolerance for every object is kept, with
one set of drivers for all objects, so the skin and the lashes carried on the lid play from one clip.

**Report.** Every candidate set's largest error per object and per phase, and with an obstacle the closest clearance of
the baked motion between the sampled phases (a straight shape that cuts the eye shows here). `evaluate(rest, shapes, s,
drivers)` gives the baked positions at any weight.

**Clip.** `clip_curve` keys one eased blink (close, hold, open) at the given frame rate, each shape's weight following
its driver of the main weight. In a [face audit](face-checks.md) declaration, list the correctives under the control's
`correctives` with their drivers, so their bumps are accepted as the bake's and not refused as phase-gated keys.

**Unity clip.** Reference avatars do not blink through the avatar descriptor: they blink from an FX-layer clip on the
shape weights. `write_unity_anim(path, clip, renderers)` writes the clip as a Unity .anim with one `blendShape.<shape>`
curve (weights 0-100) for every shape on every renderer; `renderers` maps each baked object to its
SkinnedMeshRenderer's path under the animated root (`{'Face': 'Body', 'Upper lash': 'Body/Lash'}`). `check_unity_anim`
reads the file back and reports any renderer or shape missing and any key off the clip. `bake_poses` takes the same
`renderers` and writes `<output>.anim` beside the clip, refusing a map that does not name exactly the baked objects.
Checked in Unity 2022.3: the clip imported with its four bindings (two renderers, two shapes), and sampling it on
generated renderers gave the clip's weights at every frame.

**FBX.** `export_fbx` builds the meshes with their shape keys and the clip in a clean Blender, exports an FBX, imports it
back and compares every shape key's positions and every weight curve (`roundtrip.json`). Found while building it:

- Blender's default FBX export (actions per object and NLA strips) drops animation that exists only on shape keys: the
  file has no animation stack at all. The worker exports the scene's animation instead, which keeps the curves.
- Blender's importer shifts imported keys by one frame unless `anim_offset=0`.
- Blender's importer ignores in-between shapes, which Unity plays; correctives as their own shapes driven by the clip
  avoid depending on either.

**Real use.** A hinged blink over a round eye (skin 1,665 moving points, lash 2,742): one shape missed by .0075 (skin)
and .0099 (lash) and cut toward the eye at mid-blink; main plus the mid corrective reached .00099 and .00067; main plus
early and late correctives .00061 and .00004. The tolerance is a production choice (the reference study used .0007,
about .006 of the eye's width): the report shows the cost of each set.
