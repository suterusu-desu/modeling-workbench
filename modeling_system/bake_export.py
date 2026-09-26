"""NativeJob worker for `bake.export_fbx`: meshes with baked shape keys and the blink clip, FBX out and back in.

Runs in the isolated runner's empty Blender (JOB, OUT_DIR). For every object in the bake file it builds a mesh from the
rest positions and faces, adds a shape key per baked shape and an action keying their weights from the clip, then
exports `bake.fbx` (Blender's FBX exporter with the scene's animation baked, so shape-key curves are kept; the default
per-object action export drops them). It clears the scene, imports the
FBX and compares, per object, every shape key's world positions with the baked ones and every weight curve with the
clip. Writes `roundtrip.json` (status passed when positions agree within `JOB['tolerance']` and weights within 1e-4).
"""
import hashlib
import json
from pathlib import Path

import bpy
import numpy as np

tolerance = float(JOB.get('tolerance', 1e-5))
clip = JOB['clip']
with np.load(JOB['bake']) as data:
    arrays = {k: data[k] for k in data.files}
objects = sorted({k.split('::')[0] for k in arrays})
scene = bpy.context.scene
scene.render.fps = int(clip['fps']); scene.frame_start = 0; scene.frame_end = clip['frames'][-1]['frame']
built = {}
for name in objects:
    rest = arrays[f'{name}::rest']; faces = arrays[f'{name}::faces']
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(rest.tolist(), [], faces.tolist()); mesh.update()
    ob = bpy.data.objects.new(name, mesh); scene.collection.objects.link(ob)
    ob.shape_key_add(name='Basis')
    shapes = sorted(k.split('::')[2] for k in arrays if k.startswith(f'{name}::shape::'))
    for shape in shapes:
        block = ob.shape_key_add(name=shape, from_mix=False)
        block.data.foreach_set('co', (rest + arrays[f'{name}::shape::{shape}']).astype(np.float32).ravel())
        block.value = 0.
    keys = mesh.shape_keys; keys.animation_data_create()
    keys.animation_data.action = bpy.data.actions.new(name + ' blink')
    for frame in clip['frames']:
        for shape in shapes:
            block = keys.key_blocks[shape]; block.value = frame['weights'].get(shape, 0.)
            block.keyframe_insert('value', frame=frame['frame'])
    built[name] = {'rest': rest, 'shapes': {s: rest + arrays[f'{name}::shape::{s}'] for s in shapes}}

fbx = OUT_DIR / 'bake.fbx'
# Blender's default preset exports actions per object and NLA strips; animation that exists only on shape keys is
# then dropped (no animation stack at all). Exporting the scene's animation keeps the shape-key curves.
bpy.ops.export_scene.fbx(filepath=str(fbx), object_types={'MESH'}, bake_anim=True, bake_anim_use_all_actions=False,
                         bake_anim_use_nla_strips=False, use_mesh_modifiers=False, add_leaf_bones=False)

for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.actions):     # an empty scene for the import
    for block in list(collection):
        collection.remove(block)
bpy.ops.import_scene.fbx(filepath=str(fbx), anim_offset=0.)            # the importer's default shifts keys by a frame


def world(ob, coords):
    M = np.array(ob.matrix_world); return coords @ M[:3, :3].T + M[:3, 3]


report = {'fbx': {'path': str(fbx), 'sha256': hashlib.sha256(fbx.read_bytes()).hexdigest()}, 'objects': {}}
passed = True
for name, want in built.items():
    ob = bpy.data.objects.get(name)
    row = {'found': ob is not None}
    if ob is None or ob.type != 'MESH' or not ob.data.shape_keys:
        passed = False; report['objects'][name] = row; continue
    n = len(ob.data.vertices); row['vertices'] = n
    if n != len(want['rest']):
        passed = False; row['error'] = 'vertex count changed'; report['objects'][name] = row; continue
    blocks = ob.data.shape_keys.key_blocks
    co = np.empty(n * 3); blocks['Basis'].data.foreach_get('co', co)
    errors = {'Basis': float(np.abs(world(ob, co.reshape(-1, 3)) - want['rest']).max())}
    curves = {}
    action = ob.data.shape_keys.animation_data.action if ob.data.shape_keys.animation_data else None
    for shape, target in want['shapes'].items():
        if shape not in blocks:
            errors[shape] = None; continue
        blocks[shape].data.foreach_get('co', co); errors[shape] = float(np.abs(world(ob, co.reshape(-1, 3)) - target).max())
        curve = None
        if action is not None:
            path = f'key_blocks["{shape}"].value'
            curve = next((c for c in getattr(action, 'fcurves', []) if c.data_path == path), None)
            if curve is None and hasattr(action, 'layers'):                 # layered actions
                for layer in action.layers:
                    for strip in layer.strips:
                        for bag in getattr(strip, 'channelbags', []):
                            curve = curve or next((c for c in bag.fcurves if c.data_path == path), None)
        if curve is None:
            curves[shape] = None
        else:
            curves[shape] = float(max(abs(curve.evaluate(f['frame']) - f['weights'].get(shape, 0.)) for f in clip['frames']))
    row.update(position_error=errors, weight_error=curves)
    ok = all(e is not None and e <= tolerance for e in errors.values()) and all(c is not None and c <= 1e-4 for c in curves.values())
    row['passed'] = ok; passed &= ok
    report['objects'][name] = row
report.update(status='passed' if passed else 'failed', tolerance=tolerance)
(OUT_DIR / 'roundtrip.json').write_text(json.dumps(report, indent=1), encoding='utf-8')

camera = bpy.data.objects.new('bake camera', bpy.data.cameras.new('bake camera'))       # the runner renders one view
scene.collection.objects.link(camera); camera.location = (0., -5., 0.); camera.rotation_euler = (1.5707963, 0., 0.)
scene.camera = camera
