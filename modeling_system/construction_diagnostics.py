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
                  threshold_degrees=10., objective_before=None, objective_after=None, limit=12,
                  edge_values=False):
    """Compare shared-edge normal angles on an explicitly identical triangle chart.

    Edges are vertex-index pairs. Exclusions and tags are caller evidence, never
    inferred anatomy. Output includes aggregate tails and bounded worst deltas.
    With `edge_values=True` the result also carries `edge_values`, NumPy arrays over
    every measured edge (for maps and region counts, not for a JSON report): `edges`
    (sorted vertex pairs), unsigned `before`/`after` angles, `before_signed`/
    `after_signed` (degrees, positive where the surface turns away from its winding
    normal side at the edge, a ridge; negative where it turns toward it, a valley)
    and the `tagged` mask.
    """
    _limit(limit)
    if type(edge_values) is not bool:
        raise ValueError('edge_values must be True or False')
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
    values = None
    if edge_values:
        pairs = np.array([e for e, _, _ in edges], dtype=np.int64).reshape(-1, 2)
        first = np.array([a for _, a, _ in edges], dtype=np.int64); second = np.array([b for _, _, b in edges], dtype=np.int64)
        # the vertex of the second face that is not on the edge; with consistent winding its side of the
        # first face's plane tells a valley (toward the normal) from a ridge (away from it)
        opposite = tri[second].sum(1) - pairs.sum(1)
        signed = []
        for points, n in zip((before, after), normals):
            side = np.einsum('ij,ij->i', n[first], points[opposite] - points[pairs[:, 0]]) if len(pairs) else np.zeros(0)
            signed.append(np.where(side > 0, -1., 1.))
        values = dict(edges=pairs, before=old, after=new, before_signed=old * signed[0], after_signed=new * signed[1],
                      tagged=mask.copy())
    result = dict(input_revision=_revision(before, after, tri), threshold_degrees=float(threshold_degrees),
                measured_edges=len(edges), excluded_edge_counts=counts,
                declared_tagged_edges=len(tagged), before=stats(old), after=stats(new),
                tagged=dict(before=stats(old[mask]), after=stats(new[mask])),
                untagged=dict(before=stats(old[~mask]), after=stats(new[~mask])), objective=objective,
                worst_added=[dict(edge=list(edges[i][0]), before=float(old[i]), after=float(new[i]),
                                  delta=float(new[i]-old[i]), tagged=bool(mask[i])) for i in order],
                omitted_edges=max(0, len(edges)-limit),
                interpretation='Unsigned shared-triangle normal angles on this chart; no smoothness acceptance, intended-fold classification or causal attribution')
    if values is not None:
        result['edge_values'] = values
    return result


def edge_values_at_vertices(edges, values, vertex_count):
    """Per-vertex value of largest magnitude among the edges at each vertex (0 where no measured edge).

    Spreads per-edge values (for example `compare_bends(..., edge_values=True)`
    signed angles or their change) onto vertices for a map; a crease line appears
    on the vertices of the edges along it. Presentation only: a vertex map loses
    which edge carried the value.
    """
    pairs = np.asarray(edges, dtype=np.int64); v = np.asarray(values, dtype=np.float64)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or v.shape != (len(pairs),) or not np.isfinite(v).all():
        raise ValueError('Edges must be (N, 2) vertex pairs with one finite value each')
    if type(vertex_count) is not int or vertex_count < 1 or (len(pairs) and (pairs.min() < 0 or pairs.max() >= vertex_count)):
        raise ValueError('Edges must index the declared vertex count')
    ends, both = pairs.ravel(), np.repeat(v, 2)
    largest = np.zeros(vertex_count); np.maximum.at(largest, ends, np.abs(both))
    out = np.zeros(vertex_count); hit = np.abs(both) == largest[ends]
    out[ends[hit]] = both[hit]                              # ties of equal magnitude keep one of their values
    return out


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


def local_reversals(reference, deformed, triangles):
    """Triangles whose deformed normal opposes their neighbourhood's best-fit rotation of the reference normal.

    Each vertex gets the proper rotation that best maps its reference 1-ring edge vectors onto
    the deformed ones (the local step of as-rigid-as-possible fitting); a triangle uses the
    rotation fitted to its three vertices' rings together. Material that turns coherently, a
    closing lid margin rolling past 90 degrees or a patch turned over rigidly, is not flagged:
    only triangles flipped relative to their own surroundings are (pleats, tucks, a vertex
    pushed through its neighbours). A wide flap folded back over a crease is flagged along the
    crease, not in its interior, because a mirror image of a flat patch is a half turn; measure
    the crease lines themselves with `compare_bends`.
    """
    before, after = _points(reference), _points(deformed)
    tri = np.asarray(triangles).astype(np.int64)
    edges = np.unique(np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1), axis=0)
    covariance = np.zeros((len(before), 3, 3))
    for a, b in ((0, 1), (1, 0)):
        i, j = edges[:, a], edges[:, b]
        np.add.at(covariance, i, np.einsum('ki,kj->kij', after[j] - after[i], before[j] - before[i]))
    joint = covariance[tri[:, 0]] + covariance[tri[:, 1]] + covariance[tri[:, 2]]
    u, _, vt = np.linalg.svd(joint)
    fix = np.ones((len(tri), 3)); fix[:, 2] = np.where(np.linalg.det(u @ vt) < 0, -1., 1.)
    rotation = u @ (fix[:, :, None] * vt)
    n0 = np.cross(before[tri[:, 1]] - before[tri[:, 0]], before[tri[:, 2]] - before[tri[:, 0]])
    n1 = np.cross(after[tri[:, 1]] - after[tri[:, 0]], after[tri[:, 2]] - after[tri[:, 0]])
    return np.einsum('ki,ki->k', np.einsum('kij,kj->ki', rotation, n0), n1) < 0


def compare_stretch(reference, deformed, triangles, *, compressed_below=.5, stretched_above=2.,
                    tagged_triangles=(), limit=12):
    """Tangential material strain between two states of the same triangles.

    Creases that persist after a depth or slope fit can be material crowding: a surface
    compressed well below its reference length buckles, and no depth objective removes that.
    Measure the smallest and largest principal stretch per triangle (reference -> deformed),
    the tails under the declared thresholds and triangles whose normal reverses. Thresholds are
    diagnostic choices, not anatomical limits; deliberate folds and closing lids compress.
    `normal_reversals` compares each normal with its reference direction, so material that
    rotates coherently past 90 degrees counts; `local_reversals` counts only triangles flipped
    relative to their own neighbourhood (see `local_reversals`).
    """
    _limit(limit)
    if not (np.isfinite(compressed_below) and np.isfinite(stretched_above) and 0 < compressed_below < 1 < stretched_above):
        raise ValueError('Require 0 < compressed_below < 1 < stretched_above')
    largest, smallest, cosine = triangle_stretches(reference, deformed, triangles)
    tri = np.asarray(triangles).astype(np.int64)
    flipped = local_reversals(reference, deformed, tri)
    tagged = np.zeros(len(tri), bool)
    if len(tagged_triangles):
        index = np.asarray(tagged_triangles)
        if index.ndim != 1 or index.dtype.kind not in 'iu' or index.min() < 0 or index.max() >= len(tri):
            raise ValueError('Tagged triangles are indices into the supplied triangle rows')
        tagged[index] = True

    def stats(mask):
        if not mask.any():
            return dict(count=0, smallest_stretch_q01=None, smallest_stretch_q05=None, smallest_stretch_minimum=None,
                        largest_stretch_q99=None, compressed=0, stretched=0, normal_reversals=0, local_reversals=0)
        return dict(count=int(mask.sum()), smallest_stretch_q01=float(np.quantile(smallest[mask], .01)),
                    smallest_stretch_q05=float(np.quantile(smallest[mask], .05)),
                    smallest_stretch_minimum=float(smallest[mask].min()),
                    largest_stretch_q99=float(np.quantile(largest[mask], .99)),
                    compressed=int(np.count_nonzero(smallest[mask] < compressed_below)),
                    stretched=int(np.count_nonzero(largest[mask] > stretched_above)),
                    normal_reversals=int(np.count_nonzero(cosine[mask] < 0)),
                    local_reversals=int(np.count_nonzero(flipped[mask])))

    order = np.argsort(smallest, kind='stable')[:limit]
    return dict(input_revision=_revision(_points(reference), _points(deformed), tri),
                thresholds=dict(compressed_below=float(compressed_below), stretched_above=float(stretched_above)),
                all=stats(np.ones(len(tri), bool)), tagged=stats(tagged), untagged=stats(~tagged),
                worst_compressed=[dict(triangle=int(i), vertices=tri[i].tolist(), smallest=float(smallest[i]),
                                       largest=float(largest[i]), normal_cosine=float(cosine[i]),
                                       locally_reversed=bool(flipped[i]), tagged=bool(tagged[i]))
                                  for i in order],
                omitted_triangles=max(0, len(tri) - limit),
                interpretation='Principal stretches of corresponding triangles in their own planes; crowding predicts buckling '
                               'that depth objectives cannot remove, but no appearance, intended-fold or causal judgment')


def section_turns(reference, poses, triangles, *, origin, normal, min_turn_degrees=4., window=3, limit=12):
    """Material sections through a moving surface, and how each one bends in every pose.

    Where the reference surface crosses the plane (origin, normal) every crossing keeps its edge and edge parameter, so
    the same material is followed into each pose; crossings chain into sections through shared triangles. Each pose's
    section is projected onto the plane and its signed turning summed over `window` consecutive corners; an inflection
    is a change of turning direction between windows that turn at least `min_turn_degrees`. Material that rolls over a
    round obstacle keeps its turning direction; an S across a moving band (sunk behind its edge, bulged at it) adds
    inflections. Compare poses and candidates on the same sections; thresholds are diagnostic choices, and deliberate
    creases and folds also turn.
    """
    _limit(limit)
    rest = _points(reference); tri = np.asarray(triangles).astype(np.int64)
    states = np.asarray(poses, float)
    if states.ndim == 2: states = states[None]
    if states.ndim != 3 or states.shape[1:] != rest.shape or not np.isfinite(states).all():
        raise ValueError('Poses must be finite positions of every reference point')
    if tri.ndim != 2 or tri.shape[1:] != (3,) or not len(tri) or tri.min() < 0 or tri.max() >= len(rest):
        raise ValueError('Triangles must index the supplied points')
    o, n = np.asarray(origin, float), np.asarray(normal, float)
    if o.shape != (3,) or n.shape != (3,) or not np.isfinite(o).all() or not np.isfinite(n).all() or np.linalg.norm(n) < 1e-12:
        raise ValueError('A finite plane origin and nonzero normal are required')
    if not (np.isfinite(min_turn_degrees) and min_turn_degrees > 0 and int(window) >= 1):
        raise ValueError('Require a positive turn threshold and window >= 1')
    n = n / np.linalg.norm(n)
    side = (rest - o) @ n; side = np.where(side == 0, 1e-300, side)
    crossings, segments = {}, []
    for t, (a, b, c) in enumerate(tri):
        hit = []
        for i, j in ((a, b), (b, c), (c, a)):
            if (side[i] > 0) != (side[j] > 0):
                key = (min(i, j), max(i, j))
                if key not in crossings:
                    p, q = key; crossings[key] = side[p] / (side[p] - side[q])
                hit.append(key)
        if len(hit) == 2: segments.append(tuple(hit))
    ends = {}
    for s, (k0, k1) in enumerate(segments):
        ends.setdefault(k0, []).append(s); ends.setdefault(k1, []).append(s)
    used, chains = set(), []
    for s0 in range(len(segments)):
        if s0 in used: continue
        used.add(s0); chain = list(segments[s0])
        for forward in (True, False):
            while True:
                tip = chain[-1] if forward else chain[0]
                nxt = [s for s in ends.get(tip, ()) if s not in used]
                if not nxt: break
                s = nxt[0]; used.add(s); k0, k1 = segments[s]; far = k1 if k0 == tip else k0
                if forward: chain.append(far)
                else: chain.insert(0, far)
        if len(chain) >= 3: chains.append(chain)
    e1 = np.cross(n, [1., 0., 0.]) if abs(n[0]) < .9 else np.cross(n, [0., 1., 0.]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)

    def bend(points):
        uv = np.c_[(points - o) @ e1, (points - o) @ e2]; d = np.diff(uv, axis=0); keep = np.linalg.norm(d, axis=1) > 1e-12
        d = d[keep]
        if len(d) < 2: return 0, 0., 0.
        turn = np.arctan2(d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0], np.sum(d[:-1] * d[1:], axis=1))
        w = min(int(window), len(turn)); summed = np.convolve(turn, np.ones(w), mode='valid')
        signs = np.sign(summed[np.abs(summed) >= np.radians(min_turn_degrees)])
        flips = int(np.count_nonzero(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        return flips, float(np.degrees(np.abs(turn).sum())), float(np.degrees(np.abs(summed).max(initial=0)))

    all_states = np.concatenate([rest[None], states])
    edge_pairs, params, owner, rows = [], [], [], []
    for c, chain in enumerate(chains):
        pairs = np.array(chain, dtype=np.int64); u = np.array([crossings[k] for k in chain])
        points = (1 - u)[None, :, None] * all_states[:, pairs[:, 0]] + u[None, :, None] * all_states[:, pairs[:, 1]]
        measured = [bend(p) for p in points]
        rows.append(dict(section=c, crossings=len(chain), inflections=[m[0] for m in measured],
                         total_turn_degrees=[m[1] for m in measured], largest_window_turn_degrees=[m[2] for m in measured]))
        edge_pairs.append(pairs); params.append(u); owner.append(np.full(len(chain), c))
    added = [int(sum(max(0, r['inflections'][p] - r['inflections'][0]) for r in rows)) for p in range(1, len(all_states))]
    order = sorted(rows, key=lambda r: -max(r['inflections'][1:] or [0]))[:limit]
    return dict(input_revision=_revision(rest, tri, *states),
                edge_pairs=np.concatenate(edge_pairs) if edge_pairs else np.zeros((0, 2), np.int64),
                edge_parameters=np.concatenate(params) if params else np.zeros(0), section_index=np.concatenate(owner) if owner else np.zeros(0, np.int64),
                section_points=np.stack([(1 - np.concatenate(params))[:, None] * s[np.concatenate(edge_pairs)[:, 0]] +
                                         np.concatenate(params)[:, None] * s[np.concatenate(edge_pairs)[:, 1]] for s in all_states])
                if edge_pairs else np.zeros((len(all_states), 0, 3)),
                sections=len(rows), reference_inflections=int(sum(r['inflections'][0] for r in rows)),
                inflections_per_pose=[int(sum(r['inflections'][p] for r in rows)) for p in range(1, len(all_states))],
                added_inflections_per_pose=added, worst_sections=order, omitted_sections=max(0, len(rows) - limit),
                interpretation='Material sections followed from the reference crossing into each pose, projected on the '
                               'cut plane; inflections count turning-direction changes, not appearance or intent')


def _polyline_gap(points, line):
    a, b = line[:-1], line[1:]; ab = b - a; length2 = np.maximum((ab * ab).sum(1), 1e-300)
    t = np.clip(np.einsum('msk,sk->ms', points[:, None, :] - a[None], ab) / length2[None], 0, 1)
    return np.linalg.norm(points[:, None, :] - (a[None] + t[..., None] * ab[None]), axis=-1).min(axis=1)


def closing_edges(reference, poses, moving, facing, *, pivot=None, axis=None, parts=3):
    """How an edge closes onto a facing edge through a motion, before judging its shading.

    `reference` is the open pose (N, 3); `poses` the motion at increasing phases (P, N, 3), the last one closed;
    `moving` and `facing` index the two edges (for a blink the upper and lower lid margins), each ordered along the
    edge. The questions are the ones a viewer asks of a blink: does the facing side stay where it rests, does the whole
    edge close at one rate, and does it turn like a lid or morph between keys?

    Per pose:
    - `facing_travel_max` and `facing_travel_share`: the largest displacement of a facing-edge point from the reference,
      absolute and as a share of the rest opening (median gap from the moving edge to the facing edge at rest).
    - `closure`: each moving-edge point's closed share of its gap to the facing edge's REST line, 1 - gap / rest gap,
      as min / median / max, the spread between its 10th and 90th percentiles, and the median of each of `parts`
      stretches along the edge. One rate gives equal shares; a corner that closes first (a zipper) shows a high share at
      one end and a wide spread.
    - With `pivot` and `axis` (the hinge the lid should turn on): `turn_share`, each moving point's turn about the axis
      as a share of its turn at the last pose, and `roll_deviation`, how far the moving edge lies from where one roll
      at the median share would put it (`motion_paths.path_positions`). Material that moves on separate paths and
      timings, or is pushed by stacked corrections, deviates; a lid that turns as one piece does not.
    At the last pose `seam_to_facing_rest` gives the moving edge's distance to the facing edge's rest line (0 where the
    closed seam lies on it).
    """
    rest = _points(reference)
    seq = np.asarray(poses, dtype=np.float64)
    if seq.ndim != 3 or seq.shape[1:] != rest.shape or not len(seq) or not np.isfinite(seq).all():
        raise ValueError('Poses must be finite positions of every reference point, at least one pose')
    mv, fc = (np.asarray(x) for x in (moving, facing))
    for name, idx in (('moving', mv), ('facing', fc)):
        if (idx.ndim != 1 or idx.dtype.kind not in 'iu' or len(idx) < 2 or idx.min() < 0 or idx.max() >= len(rest)
                or len(np.unique(idx)) != len(idx)):
            raise ValueError(f'The {name} edge needs at least two distinct ordered point indices')
    if type(parts) is not int or not 1 <= parts <= len(mv):
        raise ValueError('Parts must be an integer from 1 to the number of moving-edge points')
    if (pivot is None) != (axis is None):
        raise ValueError('Give both a pivot and an axis, or neither')
    line = rest[fc]
    gap0 = _polyline_gap(rest[mv], line)
    opening = float(np.median(gap0))
    if opening <= 1e-12:
        raise ValueError('The edges touch at rest; there is no opening to close')
    open_ok = gap0 > 1e-9 * max(opening, 1.)
    stretches = np.array_split(np.arange(len(mv)), parts)
    hinge = None
    if pivot is not None:
        from .motion_paths import hinge_change, path_positions
        total = hinge_change(rest[mv], seq[-1][mv], pivot, axis)['turn']
        turning = np.abs(total) > 1e-6
        hinge = (total, turning, path_positions, hinge_change)

    def stats(values):
        v = values[np.isfinite(values)]
        if not len(v):
            return None
        return {'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max()),
                'spread_p10_p90': float(np.quantile(v, .9) - np.quantile(v, .1)),
                'parts_median': [float(np.median(values[p][np.isfinite(values[p])])) if np.isfinite(values[p]).any()
                                 else None for p in stretches]}
    rows = []
    for g, pose in enumerate(seq):
        travel = np.linalg.norm(pose[fc] - rest[fc], axis=1)
        gap = _polyline_gap(pose[mv], line)
        closure = np.where(open_ok, 1 - gap / np.where(open_ok, gap0, 1), np.nan)
        row = {'pose': g, 'facing_travel_max': float(travel.max()), 'facing_travel_share': float(travel.max() / opening),
               'closure': stats(closure)}
        if hinge is not None:
            total, turning, path_positions, hinge_change = hinge
            turn = hinge_change(rest[mv], pose[mv], pivot, axis)['turn']
            share = np.where(turning, turn / np.where(turning, total, 1), np.nan)
            row['turn_share'] = stats(share)
            m = float(np.clip(np.nanmedian(share), 0, 1)) if np.isfinite(share).any() else 0.
            rolled = path_positions(rest[mv], seq[-1][mv], np.full(len(mv), m), pivot=pivot, axis=axis)['positions']
            dev = np.linalg.norm(pose[mv] - rolled, axis=1)
            row['roll_deviation'] = {'median_share_used': m, 'max': float(dev.max()), 'median': float(np.median(dev)),
                                     'share_of_opening_max': float(dev.max() / opening)}
        rows.append(row)
    rows[-1]['seam_to_facing_rest'] = {'max': float(gap.max()), 'median': float(np.median(gap)),
                                       'share_of_opening_max': float(gap.max() / opening)}
    inner = rows[:-1] if len(rows) > 1 else rows
    summary = {'rest_opening_median': opening,
               'facing_travel_share_max': max(r['facing_travel_share'] for r in rows),
               'closure_spread_max': max((r['closure']['spread_p10_p90'] for r in inner if r['closure']), default=None),
               'seam_to_facing_rest_share_max': rows[-1]['seam_to_facing_rest']['share_of_opening_max']}
    if hinge is not None:
        summary['roll_deviation_share_max'] = max(r['roll_deviation']['share_of_opening_max'] for r in rows)
    return {'poses': rows, 'summary': summary, 'revision': _revision(rest, seq, mv, fc),
            'objective': "Facing-edge travel, the moving edge's closed share of its gap to the facing rest line, and "
                         'optionally its turn about a hinge and distance from one roll, per pose',
            'limits': 'Edge measures only: the lid body between the edges, its shading and the corners need section_turns, '
                      'compare_bends and matched renders. Gaps are to the polyline through the ordered facing points. A '
                      'deliberate difference in rate or a moving facing side is a design choice these numbers cannot '
                      'make.'}
