"""Pure sampled construction checks; no fitting, native access or target admission."""
from collections import defaultdict
import hashlib
import numpy as np
from .geometry import validate


def _points(value):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not 0 < len(points) <= 100000 or not np.isfinite(points).all():
        raise ValueError('Expected 1..100000 finite XYZ samples in one declared frame')
    return points


def _limit(limit):
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError('Example limit must be 1..20')


def _revision(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(str(array.shape).encode('ascii'))
        h.update(str(array.dtype).encode('ascii'))
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def sample_residuals(expected, actual, tolerance, limit=12):
    """Compare corresponding samples; tolerance is maximum absolute XYZ component."""
    _limit(limit)
    before, after = _points(expected), _points(actual)
    if before.shape != after.shape or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Matching sample shapes and a finite nonnegative tolerance are required')
    residual = np.max(np.abs(after - before), axis=1)
    if not np.isfinite(residual).all():
        raise ValueError('Residual overflow')
    order = np.argsort(-residual, kind='stable')[:limit]
    failures = int(np.count_nonzero(residual > tolerance))
    return dict(input_revision=_revision(before, after), samples=len(before), tolerance=float(tolerance),
                metric='maximum absolute XYZ component in the declared common frame',
                maximum=float(residual.max()), p95=float(np.percentile(residual, 95)),
                over_tolerance=failures, sampled_match=failures == 0,
                worst=[dict(sample=int(i), residual=float(residual[i])) for i in order],
                omitted_samples=max(0, len(before) - limit),
                interpretation='Corresponding samples only; no graph completeness, control reachability, other poses or target admission')


def compare_bends(before, after, triangles, *, excluded_edges=(), tagged_edges=(),
                  threshold_degrees=10., objective_before=None, objective_after=None, limit=12):
    """Compare shared-edge normal angles on an explicitly identical triangle chart.

    Edges are vertex-index pairs. Exclusions and tags are caller evidence, never
    inferred anatomy. Output includes aggregate tails and bounded worst deltas.
    """
    _limit(limit)
    before, after = _points(before), _points(after)
    tri = np.asarray(triangles)
    validate(dict(co=before, tri=tri))
    if before.shape != after.shape or not 0 < len(tri) <= 100000:
        raise ValueError('Matching points and 1..100000 triangles are required')
    if not np.isfinite(threshold_degrees) or not 0 < threshold_degrees < 180:
        raise ValueError('Threshold must be between 0 and 180 degrees')
    tri = tri.astype(np.int64)
    if len({tuple(sorted(t)) for t in tri.tolist()}) != len(tri):
        raise ValueError('Duplicate triangles do not establish surface bends')
    normals = []
    for points in (before, after):
        q = points[tri]
        n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
        length = np.linalg.norm(n, axis=1)
        if not np.isfinite(length).all() or np.any(length == 0):
            raise ValueError('Degenerate or overflowing triangles require explicit repair')
        normals.append(n / length[:, None])
    owners = defaultdict(list)
    for i, t in enumerate(tri.tolist()):
        for a, b in zip(t, t[1:] + t[:1]):
            owners[tuple(sorted((a, b)))].append((i, a < b))

    def edge_set(values):
        result = set()
        for edge in values:
            if len(edge) != 2 or any(type(i) is not int for i in edge):
                raise ValueError('Edges must be pairs of integer vertex indices')
            key = tuple(sorted(edge))
            if key not in owners:
                raise ValueError('Declared edge absent from the chart')
            result.add(key)
        return result

    excluded, tagged = edge_set(excluded_edges), edge_set(tagged_edges)
    edges = []
    counts = dict(boundary=0, nonmanifold=0, inconsistent_winding=0, explicitly_excluded=0)
    for edge, faces in sorted(owners.items()):
        if len(faces) == 1: counts['boundary'] += 1
        elif len(faces) != 2: counts['nonmanifold'] += 1
        elif faces[0][1] == faces[1][1]: counts['inconsistent_winding'] += 1
        elif edge in excluded: counts['explicitly_excluded'] += 1
        else: edges.append((edge, faces[0][0], faces[1][0]))
    angles = []
    for n in normals:
        angles.append(np.array([np.degrees(np.arccos(np.clip(np.dot(n[a], n[b]), -1, 1))) for _, a, b in edges]))
    old, new = angles
    mask = np.array([e in tagged for e, _, _ in edges], dtype=bool)

    def stats(a):
        return dict(count=len(a), maximum=float(a.max()) if len(a) else None,
                    p95=float(np.percentile(a, 95)) if len(a) else None,
                    over_threshold=int(np.count_nonzero(a > threshold_degrees)))

    objective = None
    if objective_before is not None or objective_after is not None:
        if objective_before is None or objective_after is None or not np.isfinite([objective_before, objective_after]).all():
            raise ValueError('Both comparable finite objective values are required')
        objective = dict(before=float(objective_before), after=float(objective_after),
                         lower_is_better_declared=True,
                         improved_with_more_large_bends=bool(objective_after < objective_before and np.count_nonzero(new > threshold_degrees) > np.count_nonzero(old > threshold_degrees)))
    order = np.argsort(-(new - old), kind='stable')[:limit]
    return dict(input_revision=_revision(before, after, tri), threshold_degrees=float(threshold_degrees),
                measured_edges=len(edges), excluded_edge_counts=counts,
                declared_tagged_edges=len(tagged), before=stats(old), after=stats(new),
                tagged=dict(before=stats(old[mask]), after=stats(new[mask])),
                untagged=dict(before=stats(old[~mask]), after=stats(new[~mask])), objective=objective,
                worst_added=[dict(edge=list(edges[i][0]), before=float(old[i]), after=float(new[i]),
                                  delta=float(new[i]-old[i]), tagged=bool(mask[i])) for i in order],
                omitted_edges=max(0, len(edges)-limit),
                interpretation='Unsigned shared-triangle normal angles on this chart; no smoothness acceptance, intended-fold classification or causal attribution')
