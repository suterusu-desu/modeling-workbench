"""Complete projected triangle-overlap bounds for qualified saved surfaces.

This measures piecewise-linear contact for supplied poses and directions. It
does not infer anatomy, visibility, guide authority, thickness or swept motion.
"""
import numpy as np
from .preparation_contracts import check_arrays


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _barycentric(points, triangle):
    u, v = triangle[1] - triangle[0], triangle[2] - triangle[0]
    q = np.asarray(points) - triangle[0]
    x, y = _cross(q, v) / _cross(u, v), _cross(u, q) / _cross(u, v)
    return np.stack([1 - x - y, x, y], axis=-1)


def _clip(subject, triangle, tolerance):
    polygon = list(subject)
    sign = 1 if _cross(triangle[1] - triangle[0], triangle[2] - triangle[0]) > 0 else -1
    for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
        if not polygon:
            return np.empty((0, 2))
        result = []
        previous = polygon[-1]
        d_previous = sign * _cross(b - a, previous - a)
        d_previous = 0. if abs(d_previous) <= tolerance else d_previous
        for current in polygon:
            d_current = sign * _cross(b - a, current - a)
            d_current = 0. if abs(d_current) <= tolerance else d_current
            if (d_current >= 0) != (d_previous >= 0):
                result.append(previous + (current - previous) * (d_previous / (d_previous - d_current)))
            if d_current >= 0:
                result.append(current)
            previous, d_previous = current, d_current
        polygon = result
    return np.asarray(polygon)


def projected_triangle_contact(surface, baseline, triangles, obstacle, obstacle_triangles,
        frame, *, margin, tolerance, active_indices=None, bound_mode='preserve_baseline',
        build_constraints=False, displacement_direction=None, displacement_scale=1.,
        projection_tolerance=1e-12, area_tolerance=1e-14, overlap_area_tolerance=1e-16,
        max_witnesses=12):
    """Return full overlap extrema and optional sparse scalar-displacement bounds.

    Frame columns are orthonormal projection axes and increasing clearance depth.
    preserve_baseline requires height >= min(material baseline, obstacle+margin);
    clearance requires height >= obstacle+margin. Constraints apply only while
    projected coordinates, topology and the declared displacement law stay fixed.
    """
    arguments = locals()
    check_arrays(arguments, {k: {'shape': [None, 3], 'kinds': 'iu' if 'triangles' in k else 'fiu'}
                            for k in ('surface', 'baseline', 'triangles', 'obstacle', 'obstacle_triangles')})
    check_arrays(arguments, {'frame': {'shape': [3, 3]}})
    surface, baseline, obstacle, frame = [np.asarray(v, float) for v in (surface, baseline, obstacle, frame)]
    triangles, obstacle_triangles = np.asarray(triangles, int), np.asarray(obstacle_triangles, int)
    if (surface.shape != baseline.shape or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-9, rtol=0)
            or bound_mode not in ('preserve_baseline', 'clearance')):
        raise ValueError('Corresponding source positions, an orthonormal frame and an explicit contact policy required')
    for indices, count in ((triangles, len(surface)), (obstacle_triangles, len(obstacle))):
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError('Contact topology index outside its actual surface')
    for value in (margin, tolerance, projection_tolerance, area_tolerance, overlap_area_tolerance):
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative declared contact tolerances required')
    if type(build_constraints) is not bool or type(max_witnesses) is not int or not 0 <= max_witnesses <= 1000:
        raise ValueError('Explicit constraint mode and bounded witness count required')
    active = np.arange(len(surface)) if active_indices is None else np.asarray(active_indices)
    if (active.ndim != 1 or active.dtype.kind not in 'iu' or not len(active)
            or len(np.unique(active)) != len(active) or np.any(active < 0) or np.any(active >= len(surface))):
        raise ValueError('Unique nonempty qualified active indices required')
    projected, original, other = surface @ frame, baseline @ frame, obstacle @ frame
    projection_preserved = bool(np.allclose(projected[:, :2], original[:, :2], atol=projection_tolerance, rtol=0))
    factor = None
    if build_constraints:
        direction = np.asarray(displacement_direction, float)
        if (direction.shape != (3,) or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1., atol=1e-9, rtol=0)
                or not np.allclose(direction @ frame[:, :2], 0, atol=1e-9, rtol=0)
                or not projection_preserved or type(displacement_scale) not in (int, float)
                or not np.isfinite(displacement_scale)):
            raise ValueError('Constraint displacement must preserve fixed projected coordinates and declare its finite scale')
        factor = float(direction @ frame[:, 2]) * displacement_scale
    obstacle_points = other[obstacle_triangles]
    valid_obstacle = np.abs(_cross(obstacle_points[:, 1, :2] - obstacle_points[:, 0, :2],
                                  obstacle_points[:, 2, :2] - obstacle_points[:, 0, :2])) > area_tolerance
    obstacle_ids = np.flatnonzero(valid_obstacle)
    obstacle_points = obstacle_points[valid_obstacle]
    low, high = obstacle_points[:, :, :2].min(axis=1), obstacle_points[:, :, :2].max(axis=1)
    relevant = np.isin(triangles, active).any(axis=1)
    lookup = np.full(len(surface), -1); lookup[active] = np.arange(len(active))
    rows, columns, values, bounds, witnesses = [], [], [], [], []
    pairs = points = crossings = violations = skipped_surface = fixed_infeasible = 0
    minimum_slack = None
    for triangle_id in np.flatnonzero(relevant):
        ids = triangles[triangle_id]; target = projected[ids]; uv = target[:, :2]
        if abs(_cross(uv[1] - uv[0], uv[2] - uv[0])) <= area_tolerance:
            skipped_surface += 1
            continue
        candidates = np.flatnonzero(((low <= uv.max(axis=0) + projection_tolerance)
                                   & (high >= uv.min(axis=0) - projection_tolerance)).all(axis=1))
        for j in candidates:
            polygon = _clip(uv, obstacle_points[j, :, :2], projection_tolerance)
            if len(polygon) < 3 or abs(np.sum(_cross(polygon, np.roll(polygon, -1, axis=0)))) <= overlap_area_tolerance:
                continue
            sb = _barycentric(polygon, uv); ob = _barycentric(polygon, obstacle_points[j, :, :2])
            old_height, other_height = sb @ original[ids, 2], ob @ obstacle_points[j, :, 2]
            if bound_mode == 'preserve_baseline':
                difference = old_height - other_height - margin
                extra = []
                for k in range(len(polygon)):
                    following = (k + 1) % len(polygon)
                    if difference[k] * difference[following] < 0:
                        extra.append(polygon[k] + (polygon[following] - polygon[k])
                                     * difference[k] / (difference[k] - difference[following]))
                if extra:
                    crossings += len(extra)
                    polygon = np.concatenate([polygon, extra])
                    sb = _barycentric(polygon, uv); ob = _barycentric(polygon, obstacle_points[j, :, :2])
                    old_height, other_height = sb @ original[ids, 2], ob @ obstacle_points[j, :, 2]
            required = other_height + margin
            if bound_mode == 'preserve_baseline':
                required = np.minimum(old_height, required)
            height = sb @ target[:, 2]; slack = height - required
            pairs += 1; points += len(polygon); violations += int(np.count_nonzero(slack < -tolerance))
            local_min = float(slack.min())
            minimum_slack = local_min if minimum_slack is None else min(minimum_slack, local_min)
            if local_min < -tolerance and max_witnesses:
                k = int(slack.argmin())
                witnesses.append({'slack': local_min, 'surface_triangle': int(triangle_id),
                    'obstacle_triangle': int(obstacle_ids[j]), 'surface_indices': ids.tolist(),
                    'surface_weights': sb[k].tolist(), 'obstacle_weights': ob[k].tolist(),
                    'projected_point': polygon[k].tolist()})
                witnesses = sorted(witnesses, key=lambda row: row['slack'])[:max_witnesses]
            if build_constraints:
                for weights, lower_bound in zip(sb, required - old_height):
                    row_id = len(bounds); bounds.append(float(lower_bound)); nonzero = False
                    for vertex, coefficient in zip(ids, weights):
                        if lookup[vertex] >= 0:
                            value = float(coefficient * factor)
                            if value:
                                rows.append(row_id); columns.append(int(lookup[vertex])); values.append(value); nonzero = True
                    fixed_infeasible += int(not nonzero and lower_bound > tolerance)
    skipped_obstacle = int(np.count_nonzero(~valid_obstacle))
    complete = bool(pairs and not skipped_surface and not skipped_obstacle)
    result = {'overlap_pairs': pairs, 'critical_vertices': points, 'branch_crossings': crossings,
        'violating_vertices': violations, 'minimum_slack': minimum_slack, 'worst': witnesses,
        'projection_preserved': projection_preserved, 'coverage_complete': complete,
        'skipped_surface_triangles': skipped_surface, 'skipped_obstacle_triangles': skipped_obstacle,
        'passed': complete and violations == 0, 'bound_mode': bound_mode,
        'limits': 'Piecewise-linear overlap only in the supplied frame and pose. No appearance, visibility, anatomical correspondence or swept-motion acceptance. Degenerate projections and no overlap leave coverage incomplete.'}
    if build_constraints:
        result.update(constraint_rows=np.asarray(rows, dtype=np.int64),
            constraint_columns=np.asarray(columns, dtype=np.int64), constraint_values=np.asarray(values, float),
            constraint_shape=np.asarray([len(bounds), len(active)], dtype=np.int64),
            lower_bound=np.asarray(bounds, float), active_indices=active.copy(), fixed_infeasible_rows=fixed_infeasible)
    return result
