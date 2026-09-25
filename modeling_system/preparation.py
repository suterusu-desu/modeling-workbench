"""Reusable recorded-array preparation and effect evidence for method choices."""
from pathlib import Path
import hashlib
import json
import numpy as np
from . import guide_fitting
from . import material_operations
from .preparation_contracts import PreparationOperation
from .result_reporting import json_data, compact_summary
from .triangle_contact import projected_triangle_contact
from .metric_fitting import planar_fem_metric, surface_fem_metric, relax_displacement, rigid_deform
from .correction_scope import correction_scope
from .motion_paths import (motion_pace, path_positions, hinge_motion, keep_clearance, end_clearance, schedule_pace,
                           shared_schedule)
from .construction_diagnostics import section_turns


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


ARRAY_OPERATIONS = {'planar_fem_metric': planar_fem_metric, 'surface_fem_metric': surface_fem_metric,
    'relax_displacement': relax_displacement, 'rigid_deform': rigid_deform,
    'correction_scope': correction_scope,
    'section_fit': guide_fitting.prepare_section_fit,
    'material_path': guide_fitting.remap_material_path, 'compose_correspondence': compose_correspondence,
    'prepared_effect': prepared_effect, 'active_vertex_coverage': active_vertex_coverage,
    'pose_correspondence': pose_correspondence,
    'surface_realization': material_operations.surface_realization,
    'fit_landmark_field': material_operations.fit_landmark_field,
    'material_trajectory': material_operations.material_trajectory,
    'attachment_motion': material_operations.attachment_motion,
    'projected_triangle_contact': projected_triangle_contact,
    'motion_pace': motion_pace, 'path_positions': path_positions, 'hinge_motion': hinge_motion, 'keep_clearance': keep_clearance,
    'end_clearance': end_clearance, 'section_turns': section_turns, 'schedule_pace': schedule_pace,
    'shared_schedule': shared_schedule}


def prepare_arrays(operation, *, inputs, parameters=None, operations=None):
    """Execute a declarative recipe in an existing queued handler.

    Each input is {path, sha256, array}. Paths and keys stay private; exact bytes
    are checked once before loading without pickle. No arbitrary code is run.
    """
    registry = {name: PreparationOperation(function) for name, function in ARRAY_OPERATIONS.items()}
    if operations:
        if set(operations).intersection(registry) or any(not isinstance(v, PreparationOperation) for v in operations.values()):
            raise ValueError('Private pure-array operations need unique names and explicit contracts')
        registry.update(operations)
    if operation not in registry or set(inputs).intersection(parameters or {}):
        raise ValueError('Known array operation and distinct arguments required')
    selected = registry[operation]
    # Check names and dependencies before loading potentially large archives.
    selected.bind({**dict.fromkeys(inputs), **(parameters or {})})
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
    return selected(**arrays, **(parameters or {}))


def save_preparation(directory, result):
    """Preserve arrays and JSON provenance as a new immutable preparation output."""
    if not isinstance(result, dict):
        raise ValueError('Preparation must return an array/metadata dictionary')
    directory = Path(directory)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError('Prepared arrays must load without pickle')
    details = json_data({k: v for k, v in result.items() if k not in arrays})
    encoded = json.dumps(details, indent=2, allow_nan=False)
    # Validate serialization before creating the immutable output directory.
    import io
    packed = io.BytesIO()
    np.savez_compressed(packed, **arrays)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'arrays.npz').write_bytes(packed.getvalue())
    (directory / 'result.json').write_text(encoded, encoding='utf-8')
    return {name: {'path': str(directory / name),
                   'sha256': hashlib.sha256((directory / name).read_bytes()).hexdigest()}
            for name in ('arrays.npz', 'result.json')}


class ArrayPreparation:
    """Queue handler for stored recipes; no new script per prepared variant.

    Payload: operation, inputs, parameters. Optional effect evidence is produced
    by a separate prepared_effect recipe, using the actual active surface.
    """
    def __init__(self, directory, *, operations=None):
        self.directory = Path(directory)
        self.operations = dict(operations or {})

    def __call__(self, item, context):
        from .controller import fingerprint, write_json
        payload = item['payload']
        if item['workbench']['profile'] != 'analysis' or item['workbench'].get('native'):
            raise ValueError('Array preparation is an offline analysis capability')
        sources = payload['inputs']
        if any(source['sha256'] not in item['reads'].values() for source in sources.values()):
            raise ValueError('Every prepared-array file must be bound to current task dependencies')
        directory = self.directory / fingerprint({'task': item, 'attempt': context['attempt_key']})[:20]
        if directory.exists():
            raise ValueError('Preserve the existing preparation attempt; inspect its original receipt instead of replaying')
        try:
            result = prepare_arrays(payload['operation'], inputs=sources, parameters=payload.get('parameters'),
                                    operations=self.operations)
            summary = preparation_metrics(result)
            artifacts = save_preparation(directory, result)
        except (ValueError, KeyError, OSError, ImportError, TypeError, IndexError) as error:
            # Qualified pure-array functions have no native or provider effects.
            # Preserve partial files if persistence failed; never overwrite them.
            write_json(directory / 'failure.json', {'operation': payload['operation'], 'error': str(error),
                'effects': 'No native access; preparation did not complete', 'inputs': sources,
                'partial_files': sorted(p.name for p in directory.iterdir()) if directory.exists() else []})
            evidence = [{'kind': 'file', 'path': str(directory / 'failure.json'), 'role': 'Preparation failure'}]
            return {'status': 'failed', 'workbench': {'checks': {'analysis': {'status': 'fail', 'evidence': evidence}},
                'findings': [{'kind': 'failure', 'scope': 'qualified preparation',
                              'summary': 'Preparation failed before a usable output was published. Inspect the retained dependency, array-contract or serialization failure; no native operation occurred.'}]}}
        evidence = [{'kind': 'file', 'path': row['path'], 'role': name} for name, row in artifacts.items()]
        no_progress = (payload.get('stop_on_no_effect') is True
                       and result.get('has_effect_above_tolerance') is False)
        limits = 'Saved-array preparation; appearance remains unjudged. ' + result.get('limits', '')
        try:
            finding = compact_summary(payload['operation'], summary, limits=limits)
        except ValueError:
            # Keep an oversized limitation intact for OperatingSession's report
            # repair path, rather than hiding it or repeating completed math.
            finding = json.dumps({'operation': payload['operation'], 'limits': limits})
        return {'status': 'no_progress' if no_progress else 'completed', 'prepared': artifacts, 'summary': summary,
            'workbench': {'checks': {'analysis': {'status': 'pass', 'evidence': evidence}},
                'findings': [{'kind': 'measured', 'scope': 'qualified preparation',
                    'summary': finding}]}}


def preparation_metrics(result):
    """Legacy metrics plus an explicit, deliberately public operation projection."""
    if not isinstance(result, dict):
        raise ValueError('Preparation must return an array/metadata dictionary')
    summary = {k: v for k, v in result.items() if k in ('maximum', 'rms', 'changed_count',
        'has_effect_above_tolerance', 'complete_active_coverage', 'active_count', 'driven_active_count',
        'unused_vertex_count', 'best_sample_index', 'passed', 'landmark_maximum',
        'orientation_determinant', 'maximum_by_pose', 'relative_motion_maximum',
        'overlap_pairs', 'critical_vertices', 'branch_crossings', 'violating_vertices',
        'minimum_slack', 'coverage_complete', 'fixed_infeasible_rows')}
    public = result.get('public_metrics', {})
    if (not isinstance(public, dict) or any(not isinstance(k, str) or not k.strip() for k in public)
            or set(public).intersection(summary)):
        raise ValueError('public_metrics must be a named mapping without overriding built-in metrics')
    return json_data({**summary, **public})
