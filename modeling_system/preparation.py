"""Reusable recorded-array preparation and effect evidence for method choices."""
from pathlib import Path
import hashlib
import json
import numpy as np
from . import guide_fitting


def active_vertex_coverage(vertex_count, faces, driven):
    faces = [np.asarray(face) for face in faces]
    driven = np.asarray(driven)
    if (type(vertex_count) is not int or vertex_count < 1 or driven.ndim != 1
            or driven.dtype.kind not in 'iu' or any(f.ndim != 1 or len(f) < 3
            or f.dtype.kind not in 'iu' for f in faces)):
        raise ValueError('Actual vertex count, polygon indices and driven indices required')
    active = np.unique(np.concatenate(faces)) if faces else np.array([], dtype=int)
    if any(np.any(a < 0) or np.any(a >= vertex_count) for a in (active, driven)):
        raise ValueError('Coverage index outside actual source topology')
    driven = np.unique(driven)
    missing = np.setdiff1d(active, driven)
    return {'active_count': len(active), 'driven_active_count': len(np.intersect1d(active, driven)),
            'missing_active_indices': missing.tolist(),
            'unused_vertex_count': vertex_count - len(active),
            'driven_unused_count': len(np.setdiff1d(driven, active)),
            'complete_active_coverage': not len(missing)}


def prepared_effect(baseline, target, *, active_indices=None, tolerance):
    baseline, target = np.asarray(baseline, float), np.asarray(target, float)
    if (baseline.shape != target.shape or baseline.ndim != 2 or baseline.shape[1] != 3
            or not np.isfinite(baseline).all() or not np.isfinite(target).all()
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Corresponding finite positions and a declared effect tolerance required')
    indices = np.arange(len(baseline)) if active_indices is None else np.asarray(active_indices)
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not len(indices)
            or np.any(indices < 0) or np.any(indices >= len(baseline))):
        raise ValueError('Actual nonempty affected surface indices required')
    distance = np.linalg.norm(target[indices] - baseline[indices], axis=1)
    return {'sample_count': len(indices), 'tolerance': float(tolerance),
            'maximum': float(distance.max()), 'rms': float(np.sqrt(np.mean(distance**2))),
            'changed_count': int(np.count_nonzero(distance > tolerance)),
            'has_effect_above_tolerance': bool(np.any(distance > tolerance)),
            'limits': 'Prepared displacement only; not native realization or appearance improvement.'}


def compose_correspondence(baseline, moving_before, moving_target, moving_weight, *, preservation=None):
    """Replace only the qualified moving contribution of an existing composition.

    Weights are the recorded continuous correspondence, never a diagnosis-derived
    binary mask. The private capability qualifies their meaning and target guide.
    """
    baseline, before, target = [np.asarray(v, float) for v in (baseline, moving_before, moving_target)]
    weight = np.asarray(moving_weight, float)
    if (baseline.ndim != 2 or baseline.shape[1] != 3 or baseline.shape != before.shape
            or target.shape != baseline.shape or weight.shape != (len(baseline),)
            or not all(np.isfinite(v).all() for v in (baseline, before, target, weight))
            or np.any(weight < 0) or np.any(weight > 1)):
        raise ValueError('Corresponding component positions and original continuous weights required')
    preserve = np.ones(len(baseline)) if preservation is None else np.asarray(preservation, float)
    if preserve.shape != weight.shape or not np.isfinite(preserve).all() or np.any(preserve < 0) or np.any(preserve > 1):
        raise ValueError('Explicit continuous preservation weights required')
    delta = (target - before) * (weight * preserve)[:, None]
    return {'target': baseline + delta, 'world_displacement': delta,
            'preserved_contribution': baseline - before * weight[:, None],
            'moving_weight': weight.copy(), 'appearance_accepted': False}


def pose_correspondence(guide_points, pose_points, pose_values):
    """Compare already corresponding semantic material samples across saved poses."""
    guide, poses, values = [np.asarray(v, float) for v in (guide_points, pose_points, pose_values)]
    if (guide.ndim != 2 or guide.shape[1] != 3 or not len(guide)
            or poses.ndim != 3 or poses.shape[1:] != guide.shape
            or values.shape != (len(poses),) or not len(values)
            or not all(np.isfinite(v).all() for v in (guide, poses, values))):
        raise ValueError('Explicit corresponding samples, saved poses and actual pose values required')
    error = np.linalg.norm(poses - guide, axis=2)
    rms = np.sqrt(np.mean(error**2, axis=1))
    return {'pose_values': values.tolist(), 'rms': rms.tolist(), 'maximum': error.max(axis=1).tolist(),
            'best_sample_index': int(np.argmin(rms)),
            'limits': 'Finite saved correspondences only; smallest residual does not establish pose identity or visual acceptance.'}


ARRAY_OPERATIONS = {'section_fit': guide_fitting.prepare_section_fit,
    'material_path': guide_fitting.remap_material_path, 'compose_correspondence': compose_correspondence,
    'prepared_effect': prepared_effect, 'active_vertex_coverage': active_vertex_coverage,
    'pose_correspondence': pose_correspondence}


def prepare_arrays(operation, *, inputs, parameters=None):
    """Execute a declarative recipe in an existing queued handler.

    Each input is {path, sha256, array}. Paths and keys stay private; exact bytes
    are checked once before loading without pickle. No arbitrary code is run.
    """
    if operation not in ARRAY_OPERATIONS or set(inputs).intersection(parameters or {}):
        raise ValueError('Known array operation and distinct arguments required')
    arrays, archives = {}, {}
    for key, source in inputs.items():
        path = Path(source['path'])
        identity = (str(path.resolve()), source['sha256'])
        if identity not in archives:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != source['sha256']:
                raise ValueError('Prepared-array source changed')
            import io
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                archives[identity] = {name: archive[name] for name in archive.files}
        arrays[key] = archives[identity][source['array']]
    return ARRAY_OPERATIONS[operation](**arrays, **(parameters or {}))


def save_preparation(directory, result):
    """Preserve arrays and JSON provenance as a new immutable preparation output."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(directory / 'arrays.npz', **arrays)
    details = {k: v for k, v in result.items() if k not in arrays}
    (directory / 'result.json').write_text(json.dumps(details, indent=2, allow_nan=False), encoding='utf-8')
    return {name: {'path': str(directory / name),
                   'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
            for name in ('arrays.npz', 'result.json')}


class ArrayPreparation:
    """Queue handler for stored recipes; no new script per prepared variant.

    Payload: operation, inputs, parameters. Optional effect evidence is produced
    by a separate prepared_effect recipe, using the actual active surface.
    """
    def __init__(self, directory):
        self.directory = Path(directory)

    def __call__(self, item, context):
        from .controller import fingerprint, write_json
        payload = item['payload']
        if item['workbench']['profile'] != 'analysis' or item['workbench'].get('native'):
            raise ValueError('Array preparation is an offline analysis capability')
        sources = payload['inputs']
        if any(source['sha256'] not in item['reads'].values() for source in sources.values()):
            raise ValueError('Every prepared-array file must be bound to current task dependencies')
        directory = self.directory / fingerprint({'task': item, 'attempt': context['attempt_key']})[:20]
        try:
            result = prepare_arrays(payload['operation'], inputs=sources, parameters=payload.get('parameters'))
        except (ValueError, KeyError, OSError) as error:
            # No native effects and no output yet. Preserve this failed preparation
            # so a separately offered recovery can be selected without a replay.
            write_json(directory / 'failure.json', {'operation': payload['operation'], 'error': str(error),
                'effects': 'No native access; no prepared output', 'inputs': sources})
            evidence = [{'kind': 'file', 'path': str(directory / 'failure.json'), 'role': 'Preparation failure'}]
            return {'status': 'failed', 'workbench': {'checks': {'analysis': {'status': 'fail', 'evidence': evidence}},
                'findings': [{'kind': 'failure', 'scope': 'qualified preparation',
                              'summary': str(error) if isinstance(error, ValueError) else 'A required saved input could not be loaded; inspect the retained preparation failure.'}]}}
        artifacts = save_preparation(directory, result)
        evidence = [{'kind': 'file', 'path': row['path'], 'role': name} for name, row in artifacts.items()]
        summary = {k: v for k, v in result.items() if k in ('maximum', 'rms', 'changed_count',
            'has_effect_above_tolerance', 'complete_active_coverage', 'active_count', 'driven_active_count',
            'unused_vertex_count', 'best_sample_index')}
        no_progress = (payload.get('stop_on_no_effect') is True
                       and result.get('has_effect_above_tolerance') is False)
        return {'status': 'no_progress' if no_progress else 'completed', 'prepared': artifacts, 'summary': summary,
            'workbench': {'checks': {'analysis': {'status': 'pass', 'evidence': evidence}},
                'findings': [{'kind': 'measured', 'scope': 'qualified preparation',
                    'summary': json.dumps({'operation': payload['operation'], 'result': summary,
                        'limits': 'Saved-array preparation; appearance remains unjudged.'})}]}}
