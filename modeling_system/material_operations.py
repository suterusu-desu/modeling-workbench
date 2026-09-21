"""Guide-bound material math and one-owner native installation.

Inputs carry correspondence supplied by the workspace. These operations never
infer anatomy, admit a guide, or turn numerical agreement into appearance approval.
"""
import numpy as np


def _points(value):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Nonempty finite corresponding 3D points required')
    return value


def surface_realization(expected, actual, faces, *, tolerance):
    """Compare face-owning geometry; report unused points separately, never as failures."""
    from .preparation import active_vertex_coverage, prepared_effect
    expected, actual = _points(expected), _points(actual)
    if expected.shape != actual.shape:
        raise ValueError('Expected and actual source topology must correspond exactly')
    active_vertex_coverage(len(expected), faces, np.arange(len(expected)))
    active = np.unique(np.concatenate(faces)).astype(int) if len(faces) else np.array([], int)
    if not len(active):
        raise ValueError('No face-owning surface to verify')
    unused = np.setdiff1d(np.arange(len(expected)), active)
    def compare(indices):
        row = prepared_effect(expected, actual, active_indices=indices, tolerance=tolerance)
        return {**row, 'within_tolerance': not row['has_effect_above_tolerance']}
    visible = compare(active)
    return {'surface': visible, 'unused': compare(unused) if len(unused) else None,
            'active_indices': active, 'unused_indices': unused,
            'passed': visible['within_tolerance'],
            'limits': 'Exact supplied source correspondence only. Unused points are reported separately; preservation and appearance require their own checks.'}


def fit_landmark_field(source, source_anchors, guide_anchors, *, method, chart_axes=None):
    """Full affine transport, optionally with an exact thin-plate residual field.

    chart_axes is an explicitly qualified material chart, not a guessed projection.
    The affine component carries 3D thickness; the residual interpolates landmarks.
    """
    source, anchors, guide = map(_points, (source, source_anchors, guide_anchors))
    if anchors.shape != guide.shape or len(anchors) < 4:
        raise ValueError('At least four corresponding guide landmarks required')
    design = np.column_stack((anchors, np.ones(len(anchors))))
    if np.linalg.matrix_rank(design) != 4:
        raise ValueError('Landmarks do not support a full 3D affine field')
    matrix = np.linalg.lstsq(design, guide, rcond=None)[0]
    determinant = float(np.linalg.det(matrix[:3]))
    if determinant <= 0:
        raise ValueError('Guide correspondence reverses or collapses material orientation')
    target = np.column_stack((source, np.ones(len(source)))) @ matrix
    fitted = design @ matrix
    if method == 'affine_thin_plate':
        if (not isinstance(chart_axes, (list, tuple)) or len(chart_axes) != 2
                or len(set(chart_axes)) != 2 or not set(chart_axes) <= {0, 1, 2}):
            raise ValueError('Two distinct qualified material-chart axes required')
        from scipy.interpolate import RBFInterpolator
        chart = anchors[:, chart_axes]
        scale, center = np.ptp(chart, axis=0), chart.mean(axis=0)
        if np.any(scale == 0) or len(np.unique(chart, axis=0)) != len(chart):
            raise ValueError('Distinct noncollapsed material-chart landmarks required')
        uv = (chart - center) / scale
        if np.linalg.matrix_rank(np.column_stack((uv, np.ones(len(uv))))) != 3:
            raise ValueError('Material-chart landmarks are collinear')
        model = RBFInterpolator(uv, guide - fitted, kernel='thin_plate_spline', degree=1, smoothing=0.)
        target += model((source[:, chart_axes] - center) / scale)
        fitted += model(uv)
    elif method != 'affine':
        raise ValueError('Unknown qualified material fitting method')
    return {'target': target, 'fitted_anchors': fitted, 'affine': matrix,
            'landmark_maximum': float(np.linalg.norm(fitted - guide, axis=1).max()),
            'orientation_determinant': determinant, 'method': method,
            'limits': 'Supplied landmarks constrain the field; intervening material is interpolated, not pointwise guide evidence or appearance approval.'}


def material_trajectory(keyframes, knots, poses, *, baseline=None, support=None):
    """One piecewise-linear material law; preserve the existing unselected contribution."""
    frames, knots, poses = np.asarray(keyframes, float), np.asarray(knots, float), np.asarray(poses, float)
    if (frames.ndim != 3 or frames.shape[2] != 3 or frames.shape[1] < 1
            or knots.shape != (len(frames),) or len(knots) < 2
            or poses.ndim != 1 or not len(poses) or np.any(np.diff(knots) <= 0)
            or not all(np.isfinite(v).all() for v in (frames, knots, poses))
            or np.any(poses < knots[0]) or np.any(poses > knots[-1])):
        raise ValueError('Finite corresponding keyframes, increasing knots and supported poses required')
    result = np.repeat(frames[:1], len(poses), axis=0)
    for i, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        t = np.clip((poses - a) / (b - a), 0., 1.)
        result += (frames[i+1] - frames[i])[None, :, :] * t[:, None, None]
    if support is not None or baseline is not None:
        support, baseline = np.asarray(support, float), np.asarray(baseline, float)
        if (support.shape != (frames.shape[1],) or baseline.shape != result.shape
                or not np.isfinite(support).all() or not np.isfinite(baseline).all()
                or np.any(support < 0) or np.any(support > 1)):
            raise ValueError('Actual baseline poses and original continuous support required together')
        result = baseline + (result - baseline) * support[None, :, None]
    return {'target': result, 'knots': knots.copy(), 'poses': poses.copy(),
            'limits': 'Declared finite material law; guide qualification, native realization and visual motion remain separate.'}


def attachment_motion(first, second, poses, *, first_indices, second_indices):
    """Measure separation and relative motion of explicitly paired material samples."""
    first, second, poses = np.asarray(first, float), np.asarray(second, float), np.asarray(poses, float)
    a, b = np.asarray(first_indices), np.asarray(second_indices)
    if (first.ndim != 3 or second.ndim != 3 or first.shape[2] != 3 or second.shape[2] != 3
            or len(first) != len(second) or poses.shape != (len(first),) or len(poses) < 2
            or np.any(np.diff(poses) <= 0) or a.ndim != 1 or b.shape != a.shape or not len(a)
            or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or np.any(a < 0) or np.any(b < 0) or np.any(a >= first.shape[1]) or np.any(b >= second.shape[1])
            or not all(np.isfinite(v).all() for v in (first, second, poses))):
        raise ValueError('Matched poses and explicit valid attachment pairs required')
    gap = second[:, b] - first[:, a]
    distances = np.linalg.norm(gap, axis=2)
    relative = np.diff(gap, axis=0) / np.diff(poses)[:, None, None]
    return {'gap_vectors': gap, 'distances': distances, 'relative_motion': relative,
            'maximum_by_pose': distances.max(axis=1).tolist(),
            'relative_motion_maximum': float(np.linalg.norm(relative, axis=2).max()),
            'limits': 'Supplied material pairing only; separation or correlated timing does not establish the anatomical cause.'}


def install_material_trajectory(bpy, *, ob, control, modifier_name, position_label,
                                control_path, attribute_prefix, keyframes, knots, support,
                                composition='composed_zero_offset'):
    """Controller-only native capability: replace material in its existing position owner.

    Workspace supplies exact object/control/owner bindings. No new modifier, no
    layered correction and no import-time Blender work. Caller saves/reopens and
    checks realization through the ordinary native transaction.
    """
    frames, knots, support = np.asarray(keyframes, float), np.asarray(knots, float), np.asarray(support, float)
    material_trajectory(frames, knots, knots, baseline=frames, support=support)
    if frames.shape[1] != len(ob.data.vertices) or not attribute_prefix or not control_path:
        raise ValueError('Exact source topology and declared native attribute/control bindings required')
    # Resolve all shape and binding errors before copying any native graph.
    control.path_resolve(control_path)
    modifier = ob.modifiers[modifier_name]
    nodes = [n for n in modifier.node_group.nodes
             if n.bl_idname == 'GeometryNodeSetPosition' and n.label == position_label]
    if composition not in ('composed_zero_offset', 'compose_position_offset'):
        raise ValueError('Declare whether the existing owner already composes its Offset')
    if len(nodes) != 1 or (composition == 'composed_zero_offset' and (
            not nodes[0].inputs['Position'].is_linked or nodes[0].inputs['Offset'].is_linked
            or tuple(nodes[0].inputs['Offset'].default_value) != (0., 0., 0.))):
        raise ValueError('Exactly one existing position owner with already composed zero Offset required')
    names = [attribute_prefix + str(i) for i in range(len(frames))] + [attribute_prefix + 'support']
    if any(name in ob.data.attributes for name in names):
        raise ValueError('Material attribute prefix already exists; do not replay this installation')
    matrix = np.asarray(ob.matrix_world, float)
    inverse = np.linalg.inv(matrix)
    local = (frames - matrix[:3, 3]) @ inverse[:3, :3].T
    graph = modifier.node_group.copy()
    modifier.node_group = graph
    owner = next(n for n in graph.nodes if n.bl_idname == 'GeometryNodeSetPosition' and n.label == position_label)
    nd, links = graph.nodes, graph.links
    def attr(name, kind, values):
        data = ob.data.attributes.new(name, kind, 'POINT')
        data.data.foreach_set('vector' if kind == 'FLOAT_VECTOR' else 'value', values.astype(np.float32).ravel())
        node = nd.new('GeometryNodeInputNamedAttribute'); node.data_type = kind
        node.inputs['Name'].default_value = name
        return node.outputs['Attribute']
    def scalar(operation, *values):
        node = nd.new('ShaderNodeMath'); node.operation = operation
        for i, value in enumerate(values):
            if isinstance(value, bpy.types.NodeSocket): links.new(value, node.inputs[i])
            else: node.inputs[i].default_value = float(value)
        return node.outputs[0]
    def vector(operation, a, b):
        node = nd.new('ShaderNodeVectorMath'); node.operation = operation
        links.new(a, node.inputs[0]); links.new(b, node.inputs['Scale'] if operation == 'SCALE' else node.inputs[1])
        return node.outputs['Vector']
    position, offset = owner.inputs['Position'], owner.inputs['Offset']
    old = position.links[0].from_socket if position.is_linked else nd.new('GeometryNodeInputPosition').outputs['Position']
    if composition == 'compose_position_offset':
        if offset.is_linked:
            old = vector('ADD', old, offset.links[0].from_socket)
        elif tuple(offset.default_value) != (0., 0., 0.):
            constant = nd.new('ShaderNodeCombineXYZ')
            for i, value in enumerate(offset.default_value): constant.inputs[i].default_value = value
            old = vector('ADD', old, constant.outputs['Vector'])
    clock = nd.new('ShaderNodeValue')
    driver = clock.outputs[0].driver_add('default_value').driver
    driver.expression = 'v'; variable = driver.variables.new(); variable.name = 'v'; variable.type = 'SINGLE_PROP'
    variable.targets[0].id = control; variable.targets[0].data_path = control_path
    positions = [attr(name, 'FLOAT_VECTOR', co) for name, co in zip(names, local)]
    desired = positions[0]
    for i, (a, b) in enumerate(zip(knots[:-1], knots[1:])):
        t = scalar('MINIMUM', 1., scalar('MAXIMUM', 0., scalar('DIVIDE', scalar('SUBTRACT', clock.outputs[0], a), b-a)))
        desired = vector('ADD', desired, vector('SCALE', vector('SUBTRACT', positions[i+1], positions[i]), t))
    weight = attr(names[-1], 'FLOAT', support)
    links.new(vector('ADD', old, vector('SCALE', vector('SUBTRACT', desired, old), weight)), owner.inputs['Position'])
    for link in list(offset.links): links.remove(link)
    offset.default_value = (0., 0., 0.)
    ob.data.update(); ob.update_tag(refresh={'DATA'}); bpy.context.view_layer.update()
    return {'knots': knots.tolist(), 'vertices': len(support), 'attributes': names,
            'position_owner': position_label, 'composition': composition, 'appearance_accepted': False}
