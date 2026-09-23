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


def triangle_stretches(reference, deformed, triangles):
    """Principal stretches of each triangle's map from reference to deformed, in the triangles' own planes.

    Returns (largest, smallest, normal_dot): singular values of the 2x2 map between orthonormal
    frames of the corresponding triangles, and the cosine between their unit normals. A collapsed
    deformed triangle gives smallest 0; a degenerate reference triangle refuses.
    """
    before, after = _points(reference), _points(deformed)
    tri = np.asarray(triangles)
    validate(dict(co=before, tri=tri))
    if before.shape != after.shape or not 0 < len(tri) <= 100000:
        raise ValueError('Matching points and 1..100000 triangles are required')
    tri = tri.astype(np.int64)

    def frames(points):
        p = points[tri]; a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        normal = np.cross(a, b); length = np.linalg.norm(normal, axis=1)
        return a, b, normal, length

    a0, b0, n0, l0 = frames(before)
    if not np.isfinite(l0).all() or np.any(l0 <= 1e-300) or np.any(l0 <= 1e-12 * np.maximum(np.sum(a0*a0, 1), np.sum(b0*b0, 1))):
        raise ValueError('Degenerate reference triangles do not define a material metric')
    u = a0 / np.linalg.norm(a0, axis=1, keepdims=True); v = np.cross(n0 / l0[:, None], u)
    reference_2d = np.stack([np.c_[np.sum(a0*u, 1), np.zeros(len(tri))], np.c_[np.sum(b0*u, 1), np.sum(b0*v, 1)]], axis=2)
    a1, b1, n1, l1 = frames(after)
    au = np.linalg.norm(a1, axis=1); safe = np.where(au > 0, au, 1.)
    u1 = a1 / safe[:, None]
    v1 = np.where((l1 > 0)[:, None], np.cross(np.where((l1 > 0)[:, None], n1 / np.where(l1 > 0, l1, 1.)[:, None], 0.), u1), 0.)
    deformed_2d = np.stack([np.c_[np.sum(a1*u1, 1), np.sum(a1*v1, 1)], np.c_[np.sum(b1*u1, 1), np.sum(b1*v1, 1)]], axis=2)
    mapping = deformed_2d @ np.linalg.inv(reference_2d)
    singular = np.linalg.svd(mapping, compute_uv=False)
    cosine = np.where(l1 > 0, np.sum(n0 * n1, 1) / (l0 * np.where(l1 > 0, l1, 1.)), 0.)
    return singular[:, 0], singular[:, 1], cosine


def compare_stretch(reference, deformed, triangles, *, compressed_below=.5, stretched_above=2.,
                    tagged_triangles=(), limit=12):
    """Tangential material strain between two states of the same triangles.

    Creases that persist after a depth or slope fit can be material crowding: a surface
    compressed well below its reference length buckles, and no depth objective removes that.
    Measure the smallest and largest principal stretch per triangle (reference -> deformed),
    the tails under the declared thresholds and triangles whose normal reverses. Thresholds are
    diagnostic choices, not anatomical limits; deliberate folds and closing lids compress.
    """
    _limit(limit)
    if not (np.isfinite(compressed_below) and np.isfinite(stretched_above) and 0 < compressed_below < 1 < stretched_above):
        raise ValueError('Require 0 < compressed_below < 1 < stretched_above')
    largest, smallest, cosine = triangle_stretches(reference, deformed, triangles)
    tri = np.asarray(triangles).astype(np.int64)
    tagged = np.zeros(len(tri), bool)
    if len(tagged_triangles):
        index = np.asarray(tagged_triangles)
        if index.ndim != 1 or index.dtype.kind not in 'iu' or index.min() < 0 or index.max() >= len(tri):
            raise ValueError('Tagged triangles are indices into the supplied triangle rows')
        tagged[index] = True

    def stats(mask):
        if not mask.any():
            return dict(count=0, smallest_stretch_q01=None, smallest_stretch_q05=None, smallest_stretch_minimum=None,
                        largest_stretch_q99=None, compressed=0, stretched=0, normal_reversals=0)
        return dict(count=int(mask.sum()), smallest_stretch_q01=float(np.quantile(smallest[mask], .01)),
                    smallest_stretch_q05=float(np.quantile(smallest[mask], .05)),
                    smallest_stretch_minimum=float(smallest[mask].min()),
                    largest_stretch_q99=float(np.quantile(largest[mask], .99)),
                    compressed=int(np.count_nonzero(smallest[mask] < compressed_below)),
                    stretched=int(np.count_nonzero(largest[mask] > stretched_above)),
                    normal_reversals=int(np.count_nonzero(cosine[mask] < 0)))

    order = np.argsort(smallest, kind='stable')[:limit]
    return dict(input_revision=_revision(_points(reference), _points(deformed), tri),
                thresholds=dict(compressed_below=float(compressed_below), stretched_above=float(stretched_above)),
                all=stats(np.ones(len(tri), bool)), tagged=stats(tagged), untagged=stats(~tagged),
                worst_compressed=[dict(triangle=int(i), vertices=tri[i].tolist(), smallest=float(smallest[i]),
                                       largest=float(largest[i]), normal_cosine=float(cosine[i]), tagged=bool(tagged[i]))
                                  for i in order],
                omitted_triangles=max(0, len(tri) - limit),
                interpretation='Principal stretches of corresponding triangles in their own planes; crowding predicts buckling '
                               'that depth objectives cannot remove, but no appearance, intended-fold or causal judgment')
