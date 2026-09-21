"""Recorded-array fitting primitives; no native access or anatomical inference.

Callers supply qualified sections, material correspondence and preservation scope.
Interpolated support is labelled as such and never becomes direct triangle contact.
"""
import numpy as np


def _points(value, shape=None):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or (shape and value.shape[-len(shape):] != shape):
        raise ValueError('Finite points with the declared dimensions required')
    return value


def triangle_sections(vertices, triangles, position, *, axis=0, tolerance=1e-12):
    """Cut triangles, retaining original face IDs and explicitly coplanar faces."""
    vertices = _points(vertices, (3,))
    triangles = np.asarray(triangles)
    if (vertices.ndim != 2 or triangles.ndim != 2 or triangles.shape[1] != 3
            or triangles.dtype.kind not in 'iu' or axis not in (0, 1, 2)
            or not np.isfinite(position) or not np.isfinite(tolerance) or tolerance < 0
            or np.any(triangles < 0) or np.any(triangles >= len(vertices))):
        raise ValueError('Valid triangle indices, plane and tolerance required')
    segments, parents, coplanar = [], [], []
    points = vertices[triangles]
    distances = points[:, :, axis] - position
    possible = np.flatnonzero((distances.min(1) <= tolerance) & (distances.max(1) >= -tolerance))
    for face in possible:
        tri, d = points[face], distances[face]
        if np.all(np.abs(d) <= tolerance):
            coplanar.append(int(face))
            continue
        hits = [tri[i].copy() for i in range(3) if abs(d[i]) <= tolerance]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (d[i] < -tolerance and d[j] > tolerance) or (d[j] < -tolerance and d[i] > tolerance):
                hits.append(tri[i] + (tri[j] - tri[i]) * (-d[i] / (d[j] - d[i])))
        unique = []
        for hit in hits:
            if not any(np.linalg.norm(hit - prior) <= tolerance for prior in unique):
                unique.append(hit)
        if len(unique) == 2:
            segments.append(unique)
            parents.append(int(face))
    return {'segments': np.asarray(segments, dtype=float).reshape(-1, 2, 3),
            'triangles': np.asarray(parents, dtype=int), 'coplanar_triangles': coplanar}


def closest_on_segments(point, segments):
    point, segments = _points(point, (3,)), _points(segments, (2, 3))
    if point.shape != (3,) or segments.ndim != 3 or not len(segments):
        raise ValueError('A point and nonempty segment set required')
    a, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    lengths = np.einsum('ij,ij->i', delta, delta)
    weights = np.divide(np.einsum('ij,ij->i', point - a, delta), lengths,
                        out=np.zeros(len(a)), where=lengths > 0).clip(0, 1)
    projected = a + weights[:, None] * delta
    index = int(np.argmin(np.sum((projected - point) ** 2, axis=1)))
    return projected[index], index


def point_at_coordinate(value, segments, reference, *, axis=1, reference_axis=2):
    """Select a section hit using an explicit reference coordinate, never frontmost."""
    segments = _points(segments, (2, 3))
    if axis not in (0, 1, 2) or reference_axis not in (0, 1, 2) or axis == reference_axis:
        raise ValueError('Distinct coordinate and reference axes required')
    if segments.ndim != 3 or not np.isfinite([value, reference]).all():
        raise ValueError('Finite coordinate, reference and segments required')
    choices = []
    for index, (a, b) in enumerate(segments):
        span = b[axis] - a[axis]
        if span == 0:
            if a[axis] != value:
                continue
            refspan = b[reference_axis] - a[reference_axis]
            t = np.clip((reference - a[reference_axis]) / refspan, 0, 1) if refspan else 0.
        else:
            t = (value - a[axis]) / span
            if not 0 <= t <= 1:
                continue
        hit = a + t * (b - a)
        choices.append((abs(hit[reference_axis] - reference), index, hit))
    if not choices:
        return None, None
    _, index, hit = min(choices, key=lambda row: (row[0], row[1]))
    return hit, index


def section_intersections(first, second, *, plane_axis=0, tolerance=1e-10):
    """Return intersections and segment ancestry; retain collinear ambiguity."""
    first, second = _points(first, (2, 3)), _points(second, (2, 3))
    if (first.ndim != 3 or second.ndim != 3 or plane_axis not in (0, 1, 2)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Planar segments and positive tolerance required')
    combined = np.concatenate((first, second)).reshape(-1, 3)
    if len(combined) and np.ptp(combined[:, plane_axis]) > tolerance:
        raise ValueError('Segments must occupy the same section plane')
    axes = [i for i in range(3) if i != plane_axis]
    hits, parents, overlaps = [], [], []
    for i, (p, p1) in enumerate(first):
        v = (p1 - p)[axes]
        for j, (q, q1) in enumerate(second):
            w, diff = (q1 - q)[axes], (q - p)[axes]
            matrix = np.column_stack((v, -w))
            scale = np.linalg.norm(v) * np.linalg.norm(w)
            if scale == 0 or abs(np.linalg.det(matrix)) <= tolerance * scale:
                if scale and abs(v[0] * diff[1] - v[1] * diff[0]) <= tolerance * np.linalg.norm(v):
                    t0, t1 = np.dot(diff, v) / np.dot(v, v), np.dot((q1 - p)[axes], v) / np.dot(v, v)
                    if max(min(t0, t1), 0) <= min(max(t0, t1), 1):
                        overlaps.append([i, j])
                continue
            t, u = np.linalg.solve(matrix, diff)
            if -tolerance <= t <= 1 + tolerance and -tolerance <= u <= 1 + tolerance:
                hits.append(p + np.clip(t, 0, 1) * (p1 - p))
                parents.append([i, j])
    return {'points': np.asarray(hits, dtype=float).reshape(-1, 3),
            'segment_pairs': parents, 'ambiguous_overlaps': overlaps}


def interpolate_supported(stations, targets, supported, provenance, *, max_gap, max_missing):
    """Interpolate only bounded interior gaps between original direct observations."""
    stations = _points(stations)
    targets = np.asarray(targets, dtype=float)
    supported = np.asarray(supported)
    if (stations.ndim != 1 or len(stations) < 2 or np.any(np.diff(stations) <= 0)
            or targets.ndim < 2 or len(targets) != len(stations)
            or supported.shape != stations.shape or supported.dtype.kind != 'b'
            or len(provenance) != len(stations) or not supported[0] or not supported[-1]
            or not np.isfinite(targets[supported]).all()
            or not np.isfinite(max_gap) or max_gap <= 0
            or type(max_missing) is not int or max_missing < 0):
        raise ValueError('Ordered stations, direct endpoint support and explicit finite gap bounds required')
    if any(not provenance[i] for i in np.flatnonzero(supported)):
        raise ValueError('Direct support requires source provenance')
    result, records = targets.copy(), []
    direct = np.flatnonzero(supported)
    for index, station in enumerate(stations):
        if supported[index]:
            records.append({'kind': 'direct', 'station': index, 'parents': provenance[index]})
            continue
        right = int(direct[np.searchsorted(stations[direct], station)])
        left = int(direct[np.searchsorted(stations[direct], station) - 1])
        span = float(stations[right] - stations[left])
        if span > max_gap or right - left - 1 > max_missing:
            raise ValueError('Unsupported gap exceeds explicit interpolation bounds')
        fraction = float((station - stations[left]) / span)
        result[index] = (1 - fraction) * targets[left] + fraction * targets[right]
        records.append({'kind': 'interpolated', 'station': index,
            'parent_stations': [left, right], 'parents': [provenance[left], provenance[right]],
            'fraction': fraction, 'span': span})
    return result, records


def remap_material_path(source, target):
    """Map source arclength fractions to an already qualified ordered target path."""
    source, target = _points(source, (3,)), _points(target, (3,))
    if source.ndim != 2 or target.ndim != 2 or min(len(source), len(target)) < 2:
        raise ValueError('Two ordered nonempty paths required')
    def arc(points):
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        if np.any(lengths <= 0):
            raise ValueError('Repeated material stations require explicit correspondence repair')
        return np.r_[0., np.cumsum(lengths)]
    source_arc, target_arc = arc(source), arc(target)
    fractions = source_arc / source_arc[-1]
    distance = fractions * target_arc[-1]
    left = np.searchsorted(target_arc, distance, side='right').clip(1, len(target) - 1) - 1
    weight = (distance - target_arc[left]) / (target_arc[left + 1] - target_arc[left])
    points = target[left] + weight[:, None] * (target[left + 1] - target[left])
    return {'points': points, 'material_fraction': fractions,
            'target_segments': np.column_stack((left, left + 1)), 'fraction': weight}


def prepare_section_fit(stations, baseline, targets, supported, provenance, weights,
                        *, max_gap, max_missing, space, frames=None, object_matrix=None):
    """Fit corresponding material samples with preserved zero-weight boundaries.

    World-space fields encode object-local vectors after attachment rotation.
    Attachment-space fields encode vectors in each supplied material frame; a
    caller must explicitly choose this different behavior and verify its motion.
    """
    baseline = _points(baseline, (3,))
    fitted, records = interpolate_supported(stations, targets, supported, provenance,
                                           max_gap=max_gap, max_missing=max_missing)
    weights = _points(weights)
    if (baseline.shape != fitted.shape or weights.shape != (len(baseline),)
            or np.any(weights < 0) or np.any(weights > 1)):
        raise ValueError('Corresponding baseline/target samples and bounded station weights required')
    shape = (len(weights),) + (1,) * (baseline.ndim - 1)
    target = baseline + (fitted - baseline) * weights.reshape(shape)
    delta = target - baseline
    if space == 'world':
        matrix = _points(object_matrix)
        if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise ValueError('Explicit affine object transform required for a world field')
        encoded = delta @ np.linalg.inv(matrix[:3, :3]).T
    elif space == 'attachment':
        frames = _points(frames, (3, 3))
        if frames.shape != baseline.shape[:-1] + (3, 3):
            raise ValueError('One nonsingular actual attachment frame per point required')
        encoded = np.linalg.solve(frames, delta[..., None])[..., 0]
    else:
        raise ValueError('Explicit world or attachment coordinate space required')
    return {'target': target, 'world_displacement': delta, 'encoded_displacement': encoded,
            'space': space, 'support': records, 'weights': weights.copy(),
            'appearance_accepted': False}
