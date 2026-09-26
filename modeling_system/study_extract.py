"""NativeJob worker: read-only extraction of a reference avatar's mesh and shape keys for `study.load_shapes`.

Use as a job's `script` with the isolated runner (`modeling_system.isolated_blender`); `input` is the avatar's .blend
(import an FBX into a .blend first, or set `import_fbx` to its path). JOB keys: `object` (the mesh to read, default
the mesh with the most shape keys), `import_fbx` (optional path imported into the empty file before reading). Writes
`<object>.npz` and `<object>.json` in the output folder: the rest positions in world space (`basis_world`),
polygons (`loops`, `sizes`), `edges`, every shape key that moves anything as a world-space delta from its relative key
(`key0000`...), key metadata and drivers, vertex-group weights for groups with any weight, and armature bones at rest.
Nothing in the avatar is changed.
"""
import json
from pathlib import Path

import bpy
import numpy as np


def extract(job, out_dir):
    if job.get('import_fbx'):
        bpy.ops.import_scene.fbx(filepath=str(Path(job['import_fbx']).resolve(strict=True)))
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    obj = bpy.data.objects[job['object']] if job.get('object') else max(
        meshes, key=lambda o: len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0)
    mesh = obj.data; n = len(mesh.vertices)
    M = np.array(obj.matrix_world); L = M[:3, :3]
    co = np.zeros(n * 3); mesh.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    loops = np.zeros(len(mesh.loops), np.int32); mesh.loops.foreach_get('vertex_index', loops)
    sizes = np.zeros(len(mesh.polygons), np.int32); mesh.polygons.foreach_get('loop_total', sizes)
    edges = np.zeros(len(mesh.edges) * 2, np.int32); mesh.edges.foreach_get('vertices', edges)
    arrays = {'basis_world': co @ L.T + M[:3, 3], 'loops': loops, 'sizes': sizes, 'edges': edges.reshape(-1, 2),
              'matrix_world': M}
    meta = {'object': obj.name, 'vertices': n, 'polygons': len(mesh.polygons), 'keys': [], 'drivers': [], 'bones': []}
    keys = mesh.shape_keys
    if keys:
        cache = {}

        def key_co(block):
            if block.name not in cache:
                values = np.zeros(n * 3); block.data.foreach_get('co', values); cache[block.name] = values.reshape(-1, 3)
            return cache[block.name]
        for i, block in enumerate(keys.key_blocks):
            relative = block.relative_key
            delta = (key_co(block) - key_co(relative)) @ L.T if relative and relative != block else np.zeros((n, 3))
            size = np.linalg.norm(delta, axis=1)
            meta['keys'].append({'index': i, 'name': block.name, 'relative_key': relative.name if relative else None,
                                 'value': block.value, 'slider_min': block.slider_min, 'slider_max': block.slider_max,
                                 'vertex_group': block.vertex_group, 'mute': block.mute,
                                 'max_delta': float(size.max()) if n else 0., 'moved_vertices': int((size > 1e-6).sum())})
            if n and size.max() > 1e-7:
                arrays[f'key{i:04d}'] = delta.astype(np.float32)
        if keys.animation_data:
            for curve in keys.animation_data.drivers:
                meta['drivers'].append({'data_path': curve.data_path, 'type': curve.driver.type,
                                        'expression': curve.driver.expression})
    names = [g.name for g in obj.vertex_groups]; W = np.zeros((n, len(names)), np.float32)
    for v in mesh.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    used = np.flatnonzero(W.max(axis=0) > 0) if len(names) else np.zeros(0, int)
    arrays['vg_weights'] = W[:, used]; meta['vertex_groups'] = [names[i] for i in used]
    for modifier in obj.modifiers:
        if modifier.type == 'ARMATURE' and modifier.object:
            arm = modifier.object; AM = arm.matrix_world
            for bone in arm.data.bones:
                meta['bones'].append({'armature': arm.name, 'name': bone.name,
                                      'parent': bone.parent.name if bone.parent else None,
                                      'head': list(AM @ bone.head_local), 'tail': list(AM @ bone.tail_local)})
    safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in obj.name)
    np.savez_compressed(Path(out_dir) / f'{safe}.npz', **arrays)
    (Path(out_dir) / f'{safe}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    if bpy.context.scene.camera is None:                                    # the runner's closing render needs one
        data = bpy.data.cameras.new('study camera'); cam = bpy.data.objects.new('study camera', data)
        bpy.context.scene.collection.objects.link(cam); cam.location = (0., -5., 1.); cam.rotation_euler = (1.5708, 0., 0.)
        bpy.context.scene.camera = cam
    return meta


if 'JOB' in globals():
    extract(JOB, OUT_DIR)
