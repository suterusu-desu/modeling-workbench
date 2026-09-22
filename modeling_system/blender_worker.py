"""Executed inside a factory-profile Blender process by isolated_blender.py."""
import array
import hashlib
import json
import math
from pathlib import Path
import sys
import bpy
from mathutils import Vector

JOB = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text(encoding='utf-8'))
OUT_DIR = Path(JOB['output'])
if JOB.get('copied_input'):
    bpy.ops.wm.open_mainfile(filepath=JOB['copied_input'], load_ui=False)
else:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
if JOB.get('scene'):
    bpy.context.window.scene = bpy.data.scenes[JOB['scene']]


def mesh_signature(name):
    obj = bpy.data.objects.get(name)
    if obj is None:
        raise RuntimeError('Protected object missing: ' + name)
    result = hashlib.sha256()
    result.update(repr([list(row) for row in obj.matrix_world]).encode())
    if obj.type == 'MESH':
        coords = array.array('f', [0.0]) * (3 * len(obj.data.vertices))
        obj.data.vertices.foreach_get('co', coords)
        loops = array.array('i', [0]) * len(obj.data.loops)
        obj.data.loops.foreach_get('vertex_index', loops)
        counts = array.array('i', [0]) * len(obj.data.polygons)
        obj.data.polygons.foreach_get('loop_total', counts)
        result.update(coords.tobytes())
        result.update(loops.tobytes())
        result.update(counts.tobytes())
    return result.hexdigest()


protected = {name: mesh_signature(name) for name in JOB.get('protected_objects', [])}
patch = Path(JOB['script'])
exec(compile(patch.read_text(encoding='utf-8-sig'), str(patch), 'exec'), globals())
scene = bpy.context.scene
for name, before in protected.items():
    if mesh_signature(name) != before:
        raise RuntimeError('Protected mesh geometry or transform changed: ' + name)

resolution = int(JOB.get('resolution', 640))
scene.render.resolution_x = resolution
scene.render.resolution_y = resolution
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False
scene.render.use_file_extension = True
scene.render.threads_mode = 'FIXED'
scene.render.threads = int(JOB.get('threads', 2))
scene.frame_set(1)
cameras = JOB.get('cameras') or {'view': scene.camera.name}
previous_engine = scene.render.engine
previous_camera = scene.camera
rendered = []
for label, name in cameras.items():
    scene.camera = bpy.data.objects[name]
    scene.render.filepath = str(OUT_DIR / ('shaded-' + label + '.png'))
    bpy.ops.render.render(write_still=True)
    rendered.append(Path(scene.render.filepath).name)
if JOB.get('clay', True):
    scene.render.engine = 'BLENDER_WORKBENCH'
    shading = scene.display.shading
    shading.light = 'STUDIO'
    shading.color_type = 'SINGLE'
    shading.single_color = (0.52, 0.55, 0.6)
    shading.show_shadows = True
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    shading.curvature_ridge_factor = 1.25
    shading.curvature_valley_factor = 1.0
    shading.background_type = 'WORLD'
    scene.world.color = (0.055, 0.06, 0.075)
    for label, name in cameras.items():
        scene.camera = bpy.data.objects[name]
        scene.render.filepath = str(OUT_DIR / ('clay-' + label + '.png'))
        bpy.ops.render.render(write_still=True)
        rendered.append(Path(scene.render.filepath).name)
scene.render.engine = previous_engine
scene.camera = previous_camera

inventory = {'scene': scene.name, 'blender_version': bpy.app.version_string,
             'protected_geometry_unchanged': list(protected), 'rendered': rendered, 'meshes': []}
for obj in scene.objects:
    if obj.type != 'MESH' or obj.hide_render:
        continue
    mesh = obj.data
    sizes = [len(p.vertices) for p in mesh.polygons]
    bounds = [obj.matrix_world @ Vector(point) for point in obj.bound_box]
    inventory['meshes'].append({'name': obj.name, 'vertices': len(mesh.vertices),
        'polygons': len(sizes), 'quads': sizes.count(4), 'triangles': sum(n - 2 for n in sizes),
        'uv_layers': [x.name for x in mesh.uv_layers],
        'shape_keys': len(mesh.shape_keys.key_blocks) if mesh.shape_keys else 0,
        'bounds': [[min(p[k] for p in bounds) for k in range(3)], [max(p[k] for p in bounds) for k in range(3)]],
        'modifiers': [{'name': m.name, 'type': m.type} for m in obj.modifiers]})
(OUT_DIR / 'inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
bpy.ops.wm.save_as_mainfile(filepath=str(OUT_DIR / 'candidate.blend'), compress=True)
print(json.dumps({'lab_complete': True, 'candidate': str(OUT_DIR / 'candidate.blend')}))
