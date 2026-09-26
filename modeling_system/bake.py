"""Bake a built motion into the fewest blend shapes that carry it, plus the clip curve that drives them.

A motion built as a mechanism (a lid rolled about a hinge, a jaw turned) must ship as blend shapes. One straight shape
carries a slide; a roll over a round eye needs correctives. `bake_shapes` takes each object's rest pose and its poses
at increasing phases (the main shape's weight, 0 open to 1 closed), keeps the end pose exact as the main shape and
fits correctives driven by bump functions of the same weight that vanish at 0 and 1, so rest and end stay exact and
every weight stays within 0..1 (engines commonly clamp blend-shape weights). It tries one shape, then a mid corrective
driven at 4s(1-s), then early and late correctives, and keeps the smallest set within the tolerance for every object,
with the same drivers for all of them (skin and attached lashes play from one clip). It reports each phase's error and,
with an obstacle, the clearance of the baked motion between the sampled phases. `clip_curve` gives the keyframes of an
eased blink, `write_unity_anim` writes it as a Unity .anim for an FX-layer state (`check_unity_anim` reads it back),
and `export_fbx` writes and round-trips an FBX through Blender (`bake_export.py`).

Measured on a hinged blink over a round eye: one shape missed by several hundredths of the eye's width and cut into the
eye at mid-blink; one mid corrective carried it within a few thousandths, lash included.
"""
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib

import numpy as np

BASES = {'main': lambda s: s, 'mid': lambda s: 4 * s * (1 - s),
         'early': lambda s: 6.75 * s * (1 - s) ** 2, 'late': lambda s: 6.75 * s ** 2 * (1 - s)}
SETS = (('main',), ('main', 'mid'), ('main', 'early', 'late'))
TIMING = {'close': .083, 'hold': .033, 'open': .217}          # seconds; eased, as a reference avatar's blink clip


def _phases(phases):
    s = np.asarray(phases, float)
    if s.ndim != 1 or len(s) < 2 or s[0] != 0. or s[-1] != 1. or np.any(np.diff(s) <= 0):
        raise ValueError('Phases must rise from 0 (rest) to 1 (the end pose)')
    return s


def _fit(D, s, names):
    """Main shape = end pose; correctives by least squares on what it misses. Returns shapes and per-phase errors."""
    main = D[-1]; miss = D - s[:, None, None] * main
    shapes = {'main': main}
    extra = [n for n in names if n != 'main']
    if extra:
        B = np.stack([BASES[n](s) for n in extra], axis=1)                 # (phases, correctives)
        coef, *_ = np.linalg.lstsq(B, miss.reshape(len(s), -1), rcond=None)
        for n, c in zip(extra, coef):
            shapes[n] = c.reshape(main.shape)
        miss = miss - np.einsum('pk,knd->pnd', B, coef.reshape(len(extra), *main.shape))
    return shapes, np.linalg.norm(miss, axis=2).max(axis=1)


def evaluate(rest, shapes, s, drivers=None):
    """Positions of a baked object at main-shape weight s; each shape's weight follows its driver (`drivers` maps a
    shape name to its driver, as a bake's `drivers` does; shapes named after a driver need no mapping)."""
    out = np.asarray(rest, float) + 0.
    for n, delta in shapes.items():
        out = out + BASES[(drivers or {}).get(n, n)](float(s)) * delta
    return out


def _clearance(rest, shapes, obstacle, centre, samples, angular_radius_degrees):
    from .motion_paths import keep_clearance
    obs = np.asarray(obstacle, float); c = obs.mean(axis=0) if centre is None else np.asarray(centre, float)
    gap = lambda P: np.linalg.norm(P - c, axis=1) - keep_clearance(P, obs, c, -1.,
                                                                   angular_radius_degrees=angular_radius_degrees)['envelope']
    start = gap(np.asarray(rest, float)); follow = np.isfinite(start) & (start >= 0)
    worst, at = np.inf, None
    for s in samples:
        g = gap(evaluate(rest, shapes, s))[follow]
        g = g[np.isfinite(g)]
        if len(g) and g.min() < worst:
            worst, at = float(g.min()), float(s)
    return {'closest': None if at is None else worst, 'at_weight': at, 'followed_points': int(follow.sum())}


def bake_shapes(objects, phases, *, tolerance, name='blink', obstacles=None, samples=41, max_shapes=3,
                angular_radius_degrees=2.):
    """The fewest shapes that carry every object's motion within `tolerance` (largest vertex distance).

    `objects`: {object: (rest (N, 3), poses (phases, N, 3))}, the poses at the given `phases`. `obstacles`:
    {object: {'obstacle': (M, 3) points, 'centre': optional}} for the clearance report between phases. Returns
    `shapes` ({object: {shape name: delta}}), `drivers` ({shape name: driver}), the chosen set's `per_phase_error` and
    `max_error`, every candidate set's errors and clearance, and whether the chosen set meets the tolerance.
    """
    s = _phases(phases)
    if not objects:
        raise ValueError('At least one object to bake is required')
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('The tolerance must be a positive distance')
    data = {}
    for obj, (rest, poses) in objects.items():
        R = np.asarray(rest, float); P = np.asarray(poses, float)
        if R.ndim != 2 or R.shape[1] != 3 or P.shape != (len(s), *R.shape) or not np.isfinite(P).all():
            raise ValueError(f'Object {obj!r} needs finite poses of every rest vertex at every phase')
        if np.abs(P[0] - R).max() > 1e-9:
            raise ValueError(f'Object {obj!r}: the pose at phase 0 must be the rest pose')
        data[obj] = (R, P - R)
    dense = np.linspace(0., 1., int(samples))
    candidates, chosen = [], None
    for names in SETS:
        if len(names) > max_shapes:
            break
        row = {'drivers': list(names), 'objects': {}}
        fitted = {}
        for obj, (R, D) in data.items():
            shapes, err = _fit(D, s, names); fitted[obj] = shapes
            entry = {'max_error': float(err.max()), 'per_phase_error': dict(zip(map(float, s), map(float, err)))}
            if obstacles and obj in obstacles:
                o = obstacles[obj]
                entry['clearance'] = _clearance(R, shapes, o['obstacle'], o.get('centre'), dense, angular_radius_degrees)
            row['objects'][obj] = entry
        row['max_error'] = max(e['max_error'] for e in row['objects'].values())
        row['within_tolerance'] = row['max_error'] <= tolerance
        candidates.append(row)
        if row['within_tolerance'] and chosen is None:
            chosen = (row, fitted)
    if chosen is None:
        chosen = (candidates[-1], fitted)
    row, fitted = chosen
    label = lambda n: name if n == 'main' else f'{name}_{n}'
    return {'shapes': {obj: {label(n): delta for n, delta in shapes.items()} for obj, shapes in fitted.items()},
            'drivers': {label(n): n for n in row['drivers']}, 'shape_count': len(row['drivers']),
            'within_tolerance': row['within_tolerance'], 'tolerance': float(tolerance), 'max_error': row['max_error'],
            'objects': row['objects'], 'candidates': candidates,
            'objective': 'Main shape = the end pose; correctives fitted by least squares on the remaining motion, driven '
                         'by bumps of the main weight that vanish at 0 and 1',
            'limits': 'Errors at the sampled phases; clearance sampled between them from the obstacle envelope. The '
                      'shipped look still needs the clip played in the target engine.'}


def clip_curve(drivers, *, timing=None, fps=60):
    """Keyframes of one eased blink: close, hold, open. Each shape's weight follows its driver of the main weight."""
    t = {**TIMING, **(timing or {})}
    if min(t['close'], t['open']) <= 0 or t['hold'] < 0 or not 1 <= fps <= 240:
        raise ValueError('Close and open need positive durations, hold a nonnegative one, fps 1..240')
    ease = lambda u: u * u * (3 - 2 * u)
    total = t['close'] + t['hold'] + t['open']; frames = []
    for k in range(int(round(total * fps)) + 1):
        time = min(k / fps, total)
        if time <= t['close']:
            w = ease(time / t['close'])
        elif time <= t['close'] + t['hold']:
            w = 1.
        else:
            w = 1 - ease(min((time - t['close'] - t['hold']) / t['open'], 1.))
        frames.append({'frame': k, 'time': time, 'weight': w,
                       'weights': {shape: float(BASES[d](w)) for shape, d in drivers.items()}})
    return {'fps': fps, 'timing': t, 'frames': frames}


def _unity_float(v):
    return format(float(v), '.9g')


def _unity_curve(keys, indent):
    """A Unity float curve (serializedVersion 2) with free tangents from finite differences, so Unity's Hermite
    interpolation passes through every key with the curve's own slope."""
    t = np.array([k[0] for k in keys]); v = np.array([k[1] for k in keys])
    slope = np.gradient(v, t) if len(t) > 1 else np.zeros(1)
    pad = ' ' * indent; lines = [f'{pad}serializedVersion: 2', f'{pad}m_Curve:']
    for time, value, s in zip(t, v, slope):
        lines += [f'{pad}- serializedVersion: 3', f'{pad}  time: {_unity_float(time)}', f'{pad}  value: {_unity_float(value)}',
                  f'{pad}  inSlope: {_unity_float(s)}', f'{pad}  outSlope: {_unity_float(s)}', f'{pad}  tangentMode: 0',
                  f'{pad}  weightedMode: 0', f'{pad}  inWeight: 0.33333334', f'{pad}  outWeight: 0.33333334']
    return lines + [f'{pad}m_PreInfinity: 2', f'{pad}m_PostInfinity: 2', f'{pad}m_RotationOrder: 4']


def write_unity_anim(path, clip, renderers, *, name=None, shapes=None):
    """Write `clip_curve` output as a Unity AnimationClip (.anim, text): one `blendShape.<shape>` curve in weights
    0-100 for every shape on every renderer, so an FX-layer state plays the blink on skin and lashes alike.

    `renderers` maps each baked object to its SkinnedMeshRenderer's path under the animated root (for example
    {'Face': 'Body', 'Upper lash': 'Body/Lash'}); `shapes` optionally lists each object's shapes (default: every shape
    in the clip). Returns the curves written as [(renderer path, shape)]."""
    if not renderers:
        raise ValueError('At least one renderer path is required')
    if len(set(renderers.values())) != len(renderers) or not all(isinstance(p, str) and p for p in renderers.values()):
        raise ValueError('Each object needs its own renderer path')
    frames = clip['frames']; available = set(frames[0]['weights'])
    curves = []
    for obj, renderer in renderers.items():
        names = list((shapes or {}).get(obj, sorted(available)))
        missing = [s for s in names if s not in available]
        if missing:
            raise ValueError(f'Shapes {missing} of {obj!r} are not in the clip')
        curves += [(renderer, s, [(f['time'], 100. * f['weights'][s]) for f in frames]) for s in names]
    stop = frames[-1]['time']; clip_name = name or Path(path).stem
    float_curves = []
    for renderer, shape, keys in curves:
        float_curves += ['  - curve:'] + _unity_curve(keys, 6) + [
            f'    attribute: blendShape.{shape}', f'    path: {renderer}', '    classID: 137', '    script: {fileID: 0}']
    bindings = []
    for renderer, shape, _ in curves:                     # derived data Unity rebuilds on import; written for players
        bindings += ['    - serializedVersion: 2', f'      path: {zlib.crc32(renderer.encode())}',
                     f"      attribute: {zlib.crc32(f'blendShape.{shape}'.encode())}", '      script: {fileID: 0}',
                     '      typeID: 137', '      customType: 20', '      isPPtrCurve: 0']
    text = ['%YAML 1.1', '%TAG !u! tag:unity3d.com,2011:', '--- !u!74 &7400000', 'AnimationClip:',
            '  m_ObjectHideFlags: 0', '  m_CorrespondingSourceObject: {fileID: 0}', '  m_PrefabInstance: {fileID: 0}',
            '  m_PrefabAsset: {fileID: 0}', f'  m_Name: {clip_name}', '  serializedVersion: 6', '  m_Legacy: 0',
            '  m_Compressed: 0', '  m_UseHighQualityCurve: 1', '  m_RotationCurves: []',
            '  m_CompressedRotationCurves: []', '  m_EulerCurves: []', '  m_PositionCurves: []', '  m_ScaleCurves: []',
            '  m_FloatCurves:'] + float_curves + [
            '  m_PPtrCurves: []', f"  m_SampleRate: {clip['fps']}", '  m_WrapMode: 0', '  m_Bounds:',
            '    m_Center: {x: 0, y: 0, z: 0}', '    m_Extent: {x: 0, y: 0, z: 0}', '  m_ClipBindingConstant:',
            '    genericBindings:'] + bindings + [
            '    pptrCurveMapping: []', '  m_AnimationClipSettings:', '    serializedVersion: 2',
            '    m_AdditiveReferencePoseClip: {fileID: 0}', '    m_AdditiveReferencePoseTime: 0', '    m_StartTime: 0',
            f'    m_StopTime: {_unity_float(stop)}', '    m_OrientationOffsetY: 0', '    m_Level: 0',
            '    m_CycleOffset: 0', '    m_HasAdditiveReferencePose: 0', '    m_LoopTime: 0', '    m_LoopBlend: 0',
            '    m_LoopBlendOrientation: 0', '    m_LoopBlendPositionY: 0', '    m_LoopBlendPositionXZ: 0',
            '    m_KeepOriginalOrientation: 0', '    m_KeepOriginalPositionY: 1', '    m_KeepOriginalPositionXZ: 0',
            '    m_HeightFromFeet: 0', '    m_Mirror: 0', '  m_EditorCurves:'] + float_curves + [
            '  m_EulerEditorCurves: []', '  m_HasGenericRootTransform: 0', '  m_HasMotionFloatCurves: 0',
            '  m_Events: []']
    Path(path).write_bytes(('\n'.join(text) + '\n').encode('utf-8'))
    return [(renderer, shape) for renderer, shape, _ in curves]


def check_unity_anim(path, clip, renderers, *, shapes=None):
    """Read a written .anim's float curves back and compare them with the clip: every (renderer, shape) curve present,
    every key at the clip's time and weight (0-100). Returns `missing`, `unexpected`, `largest_key_error` and `status`."""
    text = Path(path).read_text(encoding='utf-8')
    section = text.split('  m_FloatCurves:\n', 1)[1].split('\n  m_PPtrCurves:', 1)[0]
    found = {}
    for block in re.split(r'(?m)^  - curve:', section)[1:]:
        times = [float(x) for x in re.findall(r'\n\s+time: (\S+)', block)]
        values = [float(x) for x in re.findall(r'\n\s+value: (\S+)', block)]
        attribute = re.search(r'\n    attribute: blendShape\.(.+)', block).group(1)
        renderer = re.search(r'\n    path: (.+)', block).group(1)
        found[(renderer, attribute)] = list(zip(times, values))
    frames = clip['frames']; available = sorted(frames[0]['weights'])
    expected = {(r, s): [(f['time'], 100. * f['weights'][s]) for f in frames]
                for o, r in renderers.items() for s in (shapes or {}).get(o, available)}
    missing = sorted(k for k in expected if k not in found); unexpected = sorted(k for k in found if k not in expected)
    error = max((abs(a[0] - b[0]) + abs(a[1] - b[1]) for k in expected if k in found
                 for a, b in zip(found[k], expected[k])), default=0.)
    counts = [len(found[k]) != len(expected[k]) for k in expected if k in found]
    ok = not missing and not unexpected and not any(counts) and error <= 1e-4
    return {'status': 'passed' if ok else 'failed', 'missing': missing, 'unexpected': unexpected,
            'key_count_mismatch': int(sum(counts)), 'largest_key_error': float(error), 'curves': len(found)}


def write_bake(path, bake, faces, rest):
    """Save a bake for `bake_export.py`: per object its rest positions, faces (triangles) and shape deltas."""
    arrays = {}
    for obj, shapes in bake['shapes'].items():
        arrays[f'{obj}::rest'] = np.asarray(rest[obj], float); arrays[f'{obj}::faces'] = np.asarray(faces[obj], np.int64)
        for shape, delta in shapes.items():
            arrays[f'{obj}::shape::{shape}'] = np.asarray(delta, float)
    np.savez(Path(path), **arrays)
    return str(path)


def export_fbx(bake_file, clip, *, blender, output_root, python=None, timeout=600, tolerance=1e-5):
    """Build the meshes with their shape keys and the clip in a clean Blender, export an FBX with Blender's default
    preset, import it back and compare positions and curves. Returns the round-trip record (`roundtrip.json`)."""
    root = Path(output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    job = {'script': str(Path(__file__).with_name('bake_export.py')), 'blender': str(blender),
           'output_root': str(root / 'runs'), 'bake': str(Path(bake_file).resolve()), 'clip': clip,
           'tolerance': tolerance, 'timeout_seconds': timeout, 'resolution': 64, 'clay': False}
    path = root / 'bake-export-job.json'; path.write_text(json.dumps(job, indent=1), encoding='utf-8')
    done = subprocess.run([str(python or sys.executable), '-m', 'modeling_system.isolated_blender', str(path)],
                          capture_output=True, text=True, timeout=timeout + 60, cwd=str(Path(__file__).resolve().parents[1]))
    receipt = json.loads(done.stdout.strip().splitlines()[-1])
    record = Path(receipt['output']) / 'roundtrip.json' if receipt.get('output') else None
    if receipt.get('status') != 'completed' or not record or not record.is_file():
        return {'status': 'failed', 'receipt': receipt, 'log': done.stdout[-4000:]}
    return {**json.loads(record.read_text(encoding='utf-8')), 'receipt': receipt}


def bake_poses(states, objects, *, tolerance, output, name='blink', obstacles=None):
    """Bake saved poses (a trial's `evaluated.npz`, `<phase>::<object>::co|tri`) and write the bake file for
    `export_fbx`: phases from the file (0 to 1), faces from the rest phase's triangles. Returns the bake report without
    its arrays, the drivers and the bake file."""
    from .checks import load_states
    poses = load_states(states); phases = [float(p) for p in poses]
    data, faces, rest = {}, {}, {}
    for obj in objects:
        stack = np.stack([np.asarray(poses[p][obj]['co'], float) for p in poses])
        data[obj] = (stack[0], stack); rest[obj] = stack[0]
        faces[obj] = np.asarray(poses[next(iter(poses))][obj]['tri'], np.int64)
    bake = bake_shapes(data, phases, tolerance=tolerance, name=name, obstacles=obstacles)
    write_bake(output, bake, faces, rest)
    return {'bake_file': str(Path(output).resolve()), 'drivers': bake['drivers'], 'shape_count': bake['shape_count'],
            'within_tolerance': bake['within_tolerance'], 'max_error': bake['max_error'], 'tolerance': bake['tolerance'],
            'objects': bake['objects'], 'candidates': bake['candidates'], 'limits': bake['limits']}
