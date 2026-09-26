"""NativeJob worker: overlap renders of the posed character and a guide in the same coordinates.

Use as a job's `script` with the isolated runner (`modeling_system.isolated_blender`); it runs inside Blender with the
runner's JOB and OUT_DIR. The working mesh renders solid orange, the rest of the character grey and the guide as a blue
wire cage (a Wireframe modifier: a wire display type renders solid), from orthographic cameras around a target. It
never saves the character; the runner keeps its usual candidate copy and inventory.

JOB keys:
- `reference`: the same configuration the reference adapter reads (working, character, controls, shape_keys, guides,
  restore); controls and shape keys pose each step.
- `workspace`: folder that relative `restore` and guide paths resolve against (default: the job's output root).
- `steps`: [{"label": "closed", "controls": {"Blink": 1.0}, "guide": "Closed" | "guide.npz" | null}, ...]; a guide is
  a name from `reference.guides` (an object in the file) or an .npz with `co` and `tri` in scene coordinates.
- `views`: [{"name": "front", "yaw": 0, "pitch": 0}, ...] in degrees about the target (yaw about Z from the front,
  looking along +Y); default front, three-quarter left and profile left.
- `target` (XYZ), `ortho_scale`, `size` (pixels), `wire` (cage thickness; default about 1.5 pixels), `hide_objects`
  (left out of the renders).
Writes `overlap-<label>-<view>.png` for every step and view, and `overlaps.json`.
"""
import importlib.util
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ORANGE, GREY, BLUE = (1., .45, .08, 1.), (.85, .85, .83, 1.), (.15, .45, 1., 1.)
VIEWS = [{'name': 'front', 'yaw': 0.}, {'name': 'three-quarter-left', 'yaw': 40.}, {'name': 'profile-left', 'yaw': 90.}]


def _restore(reference, workspace):
    hook = reference.get('restore')
    if not hook:
        return None
    path = (workspace / hook['path']).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('_modeling_workbench_overlap_restore', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return getattr(module, hook.get('function', 'restore'))(bpy, workspace, reference)


def _pose(reference, values):
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


def _cage(scene, source, name, thickness):
    import numpy as np
    if isinstance(source, str) and source.endswith('.npz'):
        data = np.load(source); co = np.asarray(data['co'], np.float32); tri = np.asarray(data['tri'], np.int32)
        mesh = bpy.data.meshes.new(name); mesh.vertices.add(len(co)); mesh.vertices.foreach_set('co', co.ravel())
        mesh.loops.add(tri.size); mesh.loops.foreach_set('vertex_index', tri.ravel())
        mesh.polygons.add(len(tri)); mesh.polygons.foreach_set('loop_start', np.arange(len(tri), dtype=np.int32) * 3)
        mesh.polygons.foreach_set('loop_total', np.full(len(tri), 3, np.int32)); mesh.update(calc_edges=True)
        mesh.validate(); ob = bpy.data.objects.new(name, mesh)
    else:
        ob = bpy.data.objects[source].copy(); ob.name = name
    scene.collection.objects.link(ob); ob.hide_render = False; ob.color = BLUE
    wire = ob.modifiers.new('overlap wire', 'WIREFRAME'); wire.thickness = float(thickness); wire.use_replace = True
    wire.use_even_offset = False; wire.use_relative_offset = False    # even offset throws spikes at sliver corners
    return ob


def run(job, out_dir):
    scene = bpy.context.scene; reference = job['reference']
    workspace = Path(job.get('workspace') or Path(out_dir).parent).resolve()
    _restore(reference, workspace)
    guides = reference.get('guides', {}); guide_objects = {g['object'] for g in guides.values()}
    character = [reference['working'], *reference.get('character', [])]
    hidden = set(job.get('hide_objects', ()))
    for ob in scene.objects:
        if ob.animation_data:
            for driver in ob.animation_data.drivers:
                if driver.data_path == 'hide_render':
                    driver.mute = True
        ob.hide_render = ob.type == 'MESH' and (ob.name not in character or ob.name in hidden)
        if ob.name in guide_objects:
            ob.hide_render = True
        if ob.name in character:
            ob.color = ORANGE if ob.name == reference['working'] else GREY
    scene.render.engine = 'BLENDER_WORKBENCH'; shading = scene.display.shading
    shading.light = 'STUDIO'; shading.color_type = 'OBJECT'; shading.show_shadows = False; shading.show_cavity = False
    scene.view_settings.view_transform = 'Standard'; scene.render.film_transparent = False
    size = int(job.get('size', 900)); scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    target = Vector(job.get('target') or [0., 0., 0.]); scale = float(job.get('ortho_scale', .3))
    cameras = []
    for view in job.get('views') or VIEWS:
        data = bpy.data.cameras.new('overlap ' + view['name']); data.type = 'ORTHO'; data.ortho_scale = scale
        data.clip_start, data.clip_end = .001, 100.
        cam = bpy.data.objects.new(data.name, data); scene.collection.objects.link(cam)
        front = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).transposed()         # looking along +Y, Z up
        turn = Matrix.Rotation(math.radians(float(view.get('yaw', 0.))), 3, 'Z') @ \
            Matrix.Rotation(math.radians(float(view.get('pitch', 0.))), 3, 'X') @ front
        cam.matrix_world = Matrix.Translation(target + turn @ Vector((0, 0, 10 * scale))) @ turn.to_4x4()
        cameras.append((view['name'], cam))
    rows = []
    for step in job['steps']:
        _pose(reference, step.get('controls'))
        source = step.get('guide')
        if source and not str(source).endswith('.npz'):
            source = guides[source]['object']
        elif source:
            source = str((workspace / source).resolve(strict=True))
        wire = job.get('wire') or 1.5 * scale / size                           # about 1.5 pixels by default
        cage = _cage(scene, source, 'overlap guide ' + step['label'], wire) if source else None
        for name, cam in cameras:
            scene.camera = cam; bpy.context.view_layer.update()
            path = Path(out_dir) / f"overlap-{step['label']}-{name}.png"
            scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)
            rows.append({'step': step['label'], 'view': name, 'controls': step.get('controls'),
                         'guide': step.get('guide'), 'image': path.name})
        if cage is not None:
            bpy.data.objects.remove(cage)
    (Path(out_dir) / 'overlaps.json').write_text(json.dumps(rows, indent=1), encoding='utf-8')
    return rows


if 'JOB' in globals():
    run(JOB, OUT_DIR)
