"""Reference native adapter (blender_json_v1): a small generic operation set for any character workspace.

Blender loads this file by path through `blender_operations` after checking its pinned hash (bind it with
`python -m modeling_system.live bind`). It uses only Blender and the standard library. The workspace binding configures
it under `native_configuration.reference`:

    {"working": "Face",                                  # the editable character mesh, shown solid orange
     "character": ["Face", "Body", "Eyeball"],            # other meshes of the character, shown solid grey
     "guides": {"Neutral": {"object": "Guide neutral", "role": "identity"},
                "Closed": {"object": "Guide closed", "role": "pose_check", "controls": {"blink": 1.0}}},
     "controls": {"object": "Controls", "properties": ["blink"]},        # custom properties, and/or
     "shape_keys": {"object": "Face", "keys": ["Blink"]},                # shape key values
     "restore": {"path": "restore_rig.py", "function": "restore"},       # optional rig hook after a file load
     "scene": "Scene"}

Guide roles say what a guide may be used for: `identity` (the accepted rest the character must match), `pose_check`
(an overlap and volume check of a mechanism at a pose) or `detail` (local form). Operations: inspect_live, bootstrap,
open_checkpoint, set_controls, set_display (GUIDE_WIRE: the working mesh solid orange with one guide overlapped as a
depth-tested blue wire; CLAY; MATERIALS; parts set aside with `hide`; a framed view), save_checkpoint. One owner holds
the live Blender (claimed by `live_bridge`); every state-changing call needs the current expected_state, and
presentation calls never change character geometry. Trusted local code, not a sandbox.
"""
import hashlib
import importlib.util
import json
import math
import time
import uuid
from pathlib import Path

ORANGE, GREY, BLUE = (1., .45, .08, 1.), (.85, .85, .83, 1.), (.1, .45, 1., 1.)
ROLES = ('identity', 'pose_check', 'detail')
SET_ASIDE = 'modeling_workbench_set_aside'
OPERATIONS = ('inspect_live', 'bootstrap', 'open_checkpoint', 'set_controls', 'set_display', 'save_checkpoint')


def _bpy():
    import bpy
    return bpy


def lane_key(workspace):
    return 'modeling-workbench lane::' + str(Path(workspace).resolve())


def _lane(workspace):
    return _bpy().app.driver_namespace.setdefault(lane_key(workspace), {'owner': None, 'session': uuid.uuid4().hex})


def _binding(arguments):
    workspace = Path(arguments['_workspace']).resolve()
    binding = json.loads((workspace / 'modeling-workspace.json').read_text(encoding='utf-8-sig'))
    config = binding.get('native_configuration', {}).get('reference')
    if not isinstance(config, dict) or not config.get('working'):
        raise ValueError('The binding has no native_configuration.reference with a working object')
    for name, guide in config.get('guides', {}).items():
        if not isinstance(guide, dict) or not guide.get('object') or guide.get('role') not in ROLES:
            raise ValueError(f'Guide {name!r} needs an object and a role: {", ".join(ROLES)}')
    return workspace, binding, config


def _scene(config):
    bpy = _bpy()
    return bpy.data.scenes[config['scene']] if config.get('scene') else bpy.context.scene


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _controls(config):
    bpy = _bpy(); out = {}
    c = config.get('controls') or {}
    ob = bpy.data.objects.get(c.get('object', ''))
    for key in c.get('properties', []):
        if ob is not None and key in ob.keys():
            out[key] = float(ob[key])
    s = config.get('shape_keys') or {}
    mesh = bpy.data.objects.get(s.get('object', ''))
    blocks = mesh.data.shape_keys.key_blocks if mesh is not None and mesh.data.shape_keys else {}
    for key in s.get('keys', []):
        if key in blocks:
            out[key] = float(blocks[key].value)
    return out


def _set_controls(config, values):
    bpy = _bpy(); c = config.get('controls') or {}; s = config.get('shape_keys') or {}
    props, keys = set(c.get('properties', [])), set(s.get('keys', []))
    for key, value in values.items():
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Control values must be finite')
        if key in props:
            ob = bpy.data.objects[c['object']]; ob[key] = value; ob.update_tag(refresh={'OBJECT'})
        elif key in keys:
            mesh = bpy.data.objects[s['object']]; mesh.data.shape_keys.key_blocks[key].value = value
            mesh.data.shape_keys.update_tag(); mesh.update_tag(refresh={'DATA'})
        else:
            raise ValueError('Unknown control: ' + key)
    bpy.context.view_layer.update()


def _meshes(config):
    bpy = _bpy(); names = [config['working'], *config.get('character', [])]
    return [bpy.data.objects[n] for n in dict.fromkeys(names) if bpy.data.objects.get(n) is not None]


def _fingerprint(config):
    """Hash of the character content presentation must not change: base coordinates, modifiers and control values."""
    import array
    h = hashlib.sha256()
    for ob in _meshes(config):
        h.update(ob.name.encode())
        if ob.type == 'MESH':
            co = array.array('f', [0.]) * (3 * len(ob.data.vertices)); ob.data.vertices.foreach_get('co', co)
            h.update(co.tobytes())
            if ob.data.shape_keys:
                for block in ob.data.shape_keys.key_blocks:
                    h.update(f'{block.name}|{block.value:.9g}|{block.mute}'.encode())
        for m in ob.modifiers:
            h.update(f'{m.name}|{m.type}|{m.show_viewport}'.encode())
    h.update(json.dumps(_controls(config), sort_keys=True).encode())
    return h.hexdigest()


def _shown_guide(config):
    bpy = _bpy()
    return next((name for name, g in config.get('guides', {}).items()
                 if (ob := bpy.data.objects.get(g['object'])) is not None and not ob.hide_get()), None)


def _hidden(config):
    scene = _scene(config)
    return sorted(n for n in scene.get(SET_ASIDE, []) if (ob := scene.objects.get(n)) is not None and ob.hide_get())


def _live(workspace, binding, config):
    bpy = _bpy(); lane = _lane(workspace); path = bpy.data.filepath
    saved = {'path': path, 'sha256': _file_sha(path)} if path and Path(path).is_file() else None
    guide = _shown_guide(config)
    live = {'owner': lane.get('owner'), 'file': path, 'dirty': bool(bpy.data.is_dirty), 'saved_file': saved,
            'controls': _controls(config), 'guide': guide,
            'guide_role': config['guides'][guide]['role'] if guide else None,
            'guides': {n: g['role'] for n, g in config.get('guides', {}).items()},
            'display': lane.get('display'), 'hidden': _hidden(config), 'content_fingerprint': _fingerprint(config),
            'session': lane.get('session'), 'restored': bool(lane.get('restored')), 'observed_at': time.time(),
            'workspace_owner': binding.get('native_owner')}
    keys = ('owner', 'file', 'dirty', 'saved_file', 'controls', 'guide', 'display', 'hidden', 'content_fingerprint')
    live['expected_state'] = hashlib.sha256(json.dumps({k: live[k] for k in keys}, sort_keys=True).encode()).hexdigest()
    return live


def _require(arguments, workspace, binding, config):
    owner = arguments.get('_owner')
    if not owner or owner != binding.get('native_owner') or _lane(workspace).get('owner') != owner:
        raise PermissionError('This live Blender is not held by the requesting owner')
    live = _live(workspace, binding, config)
    if arguments.get('expected_state') != live['expected_state']:
        raise ValueError('Stale expected_state: the live session changed since it was observed')
    return live


def _restore(workspace, config):
    """Run the workspace's rig hook (for example restoring drivers after a file load). Its file must be pinned among
    the binding's native dependencies, which blender_operations verified before loading this adapter."""
    hook = config.get('restore')
    lane = _lane(workspace)
    if not hook:
        lane['restored'] = True
        return None
    path = (workspace / hook['path']).resolve(strict=True)
    spec = importlib.util.spec_from_file_location('_modeling_workbench_restore_' + _file_sha(path)[:16], path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    result = getattr(module, hook.get('function', 'restore'))(_bpy(), workspace, config)
    lane['restored'] = True
    return result


def _show_guide(config, name):
    bpy = _bpy(); guides = config.get('guides', {})
    if name not in guides:
        raise ValueError('Unknown guide: ' + name)
    for other, g in guides.items():
        ob = bpy.data.objects.get(g['object'])
        if ob is None:
            if other == name:
                raise ValueError(f"Guide object {g['object']!r} is not in the file")
            continue
        ob.hide_set(other != name)
        if other == name:
            ob.display_type = 'WIRE'; ob.color = BLUE; ob.show_in_front = False; ob.show_wire = True


def _set_aside(config, names):
    scene = _scene(config); names = [str(n) for n in names]
    unknown = [n for n in names if scene.objects.get(n) is None]
    if unknown:
        raise ValueError('Not in the scene, cannot set aside: ' + ', '.join(unknown))
    for n in scene.get(SET_ASIDE, []):
        ob = scene.objects.get(n)
        if ob is not None and n not in names:
            ob.hide_set(False)
    for n in names:
        scene.objects[n].hide_set(True)
    scene[SET_ASIDE] = names


def _areas():
    bpy = _bpy()
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                yield area


def _display(workspace, config, mode, view=None):
    bpy = _bpy()
    guides = {g['object'] for g in config.get('guides', {}).values()}
    for ob in _meshes(config):
        if ob.name not in guides:
            ob.color = ORANGE if ob.name == config['working'] else GREY
    for area in _areas():
        shading = area.spaces.active.shading; shading.type = 'SOLID'
        shading.color_type = {'GUIDE_WIRE': 'OBJECT', 'CLAY': 'SINGLE', 'MATERIALS': 'MATERIAL'}[mode]
        if mode == 'CLAY':
            shading.single_color = (.78, .78, .8)
        if view:
            from mathutils import Quaternion, Vector
            r3d = area.spaces.active.region_3d
            if 'rotation' in view: r3d.view_rotation = Quaternion(view['rotation'])
            if 'location' in view: r3d.view_location = Vector(view['location'])
            if 'distance' in view: r3d.view_distance = float(view['distance'])
            if 'perspective' in view: r3d.view_perspective = view['perspective']
            if 'lens' in view: area.spaces.active.lens = float(view['lens'])
        area.tag_redraw()
    _lane(workspace)['display'] = mode


def header(workspace, config, status=None):
    """One line in every 3D view: owner, file, controls and the guide shown with its role."""
    bpy = _bpy(); lane = _lane(workspace)
    if status is not None:
        lane['status'] = status
    guide = _shown_guide(config)
    shown = f"blue wire = {guide} ({config['guides'][guide]['role']})" if guide else 'no guide shown'
    controls = ', '.join(f'{k} {v:.3f}' for k, v in _controls(config).items())
    text = (f"WORKBENCH {lane.get('owner') or 'no owner'}  |  {lane.get('status') or 'live'}  |  "
            f"{Path(bpy.data.filepath).name or 'unsaved'}  |  {controls or 'no controls'}  |  orange = "
            f"{config['working']}, {shown}")
    for area in _areas():
        area.header_text_set(text)


def execute(operation, arguments):
    try:
        workspace, binding, config = _binding(arguments)
        bpy = _bpy()
        if operation == 'inspect_live':
            return {'ok': True, 'result': {'live': _live(workspace, binding, config)}}
        if operation not in OPERATIONS:
            raise ValueError('Operation not implemented by the reference adapter: ' + operation)
        live = _require(arguments, workspace, binding, config)
        result = {}
        if operation == 'bootstrap':
            expected = arguments['expected_file']
            if not live['saved_file'] or live['saved_file']['sha256'] != expected['sha256'] or \
                    Path(live['saved_file']['path']).resolve() != Path(expected['path']).resolve():
                raise ValueError('The live file is not the expected saved file')
            result['restore'] = _restore(workspace, config)
        elif operation == 'open_checkpoint':
            source = Path(arguments['source']['path']).resolve(strict=True)
            if _file_sha(source) != arguments['source']['sha256']:
                raise ValueError('Checkpoint bytes differ from the requested source')
            owner = _lane(workspace)['owner']
            bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=bool(arguments.get('load_ui', False)))
            lane = _lane(workspace); lane.update(owner=owner, display=None, restored=False)
            result['restore'] = _restore(workspace, config)
            result['opened'] = {'path': str(source), 'sha256': arguments['source']['sha256']}
        elif operation == 'set_controls':
            guide = arguments.get('guide')
            values = arguments.get('controls')
            if values is None and guide:
                values = config['guides'].get(guide, {}).get('controls')
            if values:
                _set_controls(config, values)
            if guide:
                _show_guide(config, guide)
        elif operation == 'set_display':
            mode = arguments.get('mode', 'GUIDE_WIRE')
            if mode not in ('GUIDE_WIRE', 'CLAY', 'MATERIALS'):
                raise ValueError('Display mode not implemented by the reference adapter: ' + mode)
            _display(workspace, config, mode, arguments.get('view'))
            if 'hide' in arguments:
                _set_aside(config, arguments['hide'])
        elif operation == 'save_checkpoint':
            current = bpy.data.filepath
            target = Path(arguments.get('path') or current).resolve()
            if arguments.get('expected_file_sha256') and current and _file_sha(current) != arguments['expected_file_sha256']:
                raise ValueError('Current saved file changed')
            if target.exists() and (not current or target != Path(current).resolve()):
                raise FileExistsError('Refusing to overwrite an existing different file: ' + str(target))
            bpy.ops.wm.save_as_mainfile(filepath=str(target), copy=bool(arguments.get('copy', False)), compress=True)
            result['saved'] = {'path': str(target), 'sha256': _file_sha(target), 'label': arguments.get('label')}
        header(workspace, config, arguments.get('label') or operation.replace('_', ' '))
        result['live'] = _live(workspace, binding, config)
        return {'ok': True, 'result': result}
    except Exception as exc:
        return {'ok': False, 'error': {'code': type(exc).__name__, 'message': str(exc),
                                       'recovery': 'Inspect the live session with inspect_live before repeating an effect.'}}
