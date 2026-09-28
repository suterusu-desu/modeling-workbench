"""Evaluate declared objects at declared poses inside Blender and save them in the standard pose format.

Blender-side helper used by `trial_worker.py` and `reopen_poses.py` (load it by path; it needs only Blender and
NumPy). Poses are {phase: {control: value}}, the controls those of the workspace's `reference` configuration (custom
properties and shape keys). Arrays are named `<phase>::<object>::co` (evaluated world positions, after modifiers)
and `<phase>::<object>::tri` (the evaluated triangles at that pose), the format `checks.run_checks` reads. Each pose
also records ordered polygon connectivity (`loops`, `polygon_starts`, `polygon_lengths`) and `triangle_polygon`.
Polygon connectivity can remain stable while Blender tessellates a deforming quad/ngon differently at each pose.
"""
import importlib.util
from pathlib import Path

import bpy
import numpy as np


def restore(reference, workspace):
    """Run the workspace's restore hook, `restore(bpy, workspace, reference)`, if one is declared."""
    hook = reference.get('restore')
    if not hook:
        return None
    path = (Path(workspace) / hook['path']).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('_modeling_workbench_pose_restore', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, hook.get('function', 'restore'))(bpy, Path(workspace), reference)


def pose(reference, values):
    """Set custom-property and shape-key controls, then update the scene."""
    c = reference.get('controls') or {}; s = reference.get('shape_keys') or {}
    for key, value in (values or {}).items():
        if key in c.get('properties', []):
            ob = bpy.data.objects[c['object']]; ob[key] = float(value); ob.update_tag(refresh={'OBJECT'})
        elif key in s.get('keys', []):
            mesh = bpy.data.objects[s['object']]; mesh.data.shape_keys.key_blocks[key].value = float(value)
            mesh.data.shape_keys.update_tag(); mesh.update_tag(refresh={'DATA'})
        else:
            raise ValueError('Unknown control: ' + key)
    bpy.context.view_layer.update()


def evaluated(name, *, with_topology=False):
    """World positions and native triangles; optionally also their evaluated polygon-connectivity witness.

    The default two-tuple is unchanged. With `with_topology=True`, a third value holds the four witness arrays,
    read from the same evaluated mesh as the positions and triangles, before that temporary mesh is cleared.
    """
    ob = bpy.data.objects[name]; graph = bpy.context.evaluated_depsgraph_get(); ev = ob.evaluated_get(graph)
    mesh = ev.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3); mesh.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
        mesh.calc_loop_triangles()
        tri = np.empty(len(mesh.loop_triangles) * 3, np.int32); mesh.loop_triangles.foreach_get('vertices', tri)
        if with_topology:
            topology = {}
            for field, collection, attribute in (
                    ('loops', mesh.loops, 'vertex_index'), ('polygon_starts', mesh.polygons, 'loop_start'),
                    ('polygon_lengths', mesh.polygons, 'loop_total'),
                    ('triangle_polygon', mesh.loop_triangles, 'polygon_index')):
                topology[field] = np.empty(len(collection), np.int32)
                collection.foreach_get(attribute, topology[field])
    finally:
        ev.to_mesh_clear()
    M = np.array(ob.matrix_world)
    result = co @ M[:3, :3].T + M[:3, 3], tri.reshape(-1, 3)
    return (*result, topology) if with_topology else result


def evaluate(objects, poses, reference):
    """Positions, native triangles and polygon witness for every object at every pose, in the order given."""
    arrays = {}
    for phase, values in poses.items():
        pose(reference, values)
        for name in objects:
            co, tri, topology = evaluated(name, with_topology=True)
            arrays[f'{phase}::{name}::co'] = co; arrays[f'{phase}::{name}::tri'] = tri
            arrays.update({f'{phase}::{name}::{key}': value for key, value in topology.items()})
    return arrays


def save(path, arrays):
    np.savez(Path(path), **arrays)
    return str(path)


def ensure_camera(reference):
    """Give the scene a front orthographic camera on the working object if it has none (the runner renders one)."""
    scene = bpy.context.scene
    if scene.camera is not None:
        return scene.camera
    from mathutils import Vector
    ob = bpy.data.objects[reference['working']]
    corners = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)]); high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, size = (low + high) / 2, max((high - low).x, (high - low).z, 1e-3)
    data = bpy.data.cameras.new('trial camera'); data.type = 'ORTHO'; data.ortho_scale = 1.4 * size
    cam = bpy.data.objects.new('trial camera', data); scene.collection.objects.link(cam)
    cam.location = (centre.x, low.y - 3 * size, centre.z); cam.rotation_euler = (1.5707963, 0., 0.)
    scene.camera = cam
    return cam
