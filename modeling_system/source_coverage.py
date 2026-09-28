"""Finite source-chart coverage measurements, never a solver or acceptance gate."""
from collections import defaultdict
import hashlib
import json
import numpy as np
from .triangle_contact import _barycentric, _cross, _overlap


def _digest(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def _array(value, width, name, integer=False):
    a = np.asarray(value)
    if (a.ndim != 2 or a.shape[1] != width or not len(a) or len(a) > 1000000
            or a.dtype.kind not in ('iu' if integer else 'fiu') or not np.isfinite(a).all()):
        raise ValueError(f'{name} requires finite nonempty Nx{width} rows')
    return a.astype(np.int64 if integer else float)


def _ids(value, size, name):
    a = np.asarray(value)
    if (a.ndim != 1 or not len(a) or a.dtype.kind not in 'iu' or a.min() < 0
            or a.max() >= size or len(np.unique(a)) != len(a)):
        raise ValueError(f'{name} requires distinct valid indices')
    return a.astype(np.int64)


def _edges(tri):
    owners = defaultdict(list)
    for i, t in enumerate(tri.tolist()):
        for a, b in zip(t, t[1:] + t[:1]):
            owners[tuple(sorted((a, b)))].append((i, a, b))
    boundary = {edge for edge, rows in owners.items() if len(rows) == 1}
    bad = any(len(rows) > 2 or (len(rows) == 2 and rows[0][1:] == rows[1][1:])
              for rows in owners.values())
    return boundary, bad, owners


def _area(tri):
    return _cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) / 2


def _polygon_area(p):
    return abs(float(_cross(p[1:-1] - p[0], p[2:] - p[0]).sum())) / 2 if len(p) >= 3 else 0.


def _distances(points, segments):
    best = np.full(len(points), np.inf)
    for a, b in segments:
        d = b - a
        if d @ d == 0:
            distance = np.linalg.norm(points - a, axis=1)
        else:
            t = np.clip((points - a) @ d / (d @ d), 0, 1)
            distance = np.linalg.norm(points - (a + t[:, None] * d), axis=1)
        best = np.minimum(best, distance)
    return best


def _boundary_measure(a, b, samples):
    def directed(source, target):
        t = np.linspace(0, 1, samples + 1)
        points = (source[:, :1] + t[None, :, None] * (source[:, 1:] - source[:, :1])).reshape(-1, 2)
        measured = float(_distances(points, target).max())
        upper = measured + float(np.linalg.norm(source[:, 1] - source[:, 0], axis=1).max()) / (2 * samples)
        return measured, upper
    ab, ba = directed(a, b), directed(b, a)
    return {'sampled_max_uv': max(ab[0], ba[0]), 'upper_bound_uv': max(ab[1], ba[1]),
            'segments_per_edge': samples, 'metric': 'symmetric distance to opposite boundary segments in UV'}


def source_coverage(source_positions, source_triangles, source_uv, source_components, source_qualified,
                    positions, triangles, correspondence_triangles, correspondence_barycentric,
                    footprint, domain, source_boundary, boundary, *, component, provenance, units, frame,
                    resolution=128, interior_order=4, boundary_samples=16, uv_tolerance=1e-10,
                    area_tolerance=1e-12, distance_tolerance=1e-9, max_pairs=200000):
    """Measure one declared source chart/component at one saved pose.

    Candidate vertex correspondence is explicit source triangle + barycentrics;
    no closest-point inference occurs. Footprint/domain select actual source and
    candidate triangle rows. Boundaries are complete topological edge sets.
    Qualification failures return unknown and null measurements. Raster area
    intervals retain all edge-near cells as unresolved; there is no pass result.
    Provenance records caller claims and array hashes, not authority or freshness.
    """
    S = _array(source_positions, 3, 'source_positions')
    T = _array(source_triangles, 3, 'source_triangles', True)
    UV = _array(source_uv, 2, 'source_uv')
    P = _array(positions, 3, 'positions')
    C = _array(triangles, 3, 'triangles', True)
    if len(UV) != len(S) or T.min() < 0 or T.max() >= len(S) or C.min() < 0 or C.max() >= len(P):
        raise ValueError('Mesh indices or source UV count do not match positions')
    F, D = _ids(footprint, len(T), 'footprint'), _ids(domain, len(C), 'domain')
    SB = _array(source_boundary, 2, 'source_boundary', True)
    CB = _array(boundary, 2, 'boundary', True)
    comps, qualified = np.asarray(source_components), np.asarray(source_qualified)
    refs, weights = np.asarray(correspondence_triangles), np.asarray(correspondence_barycentric)
    if (comps.shape != (len(T),) or comps.dtype.kind not in 'iu' or qualified.shape != (len(T),)
            or qualified.dtype.kind != 'b'):
        raise ValueError('One integer component and boolean qualification flag per source triangle required')
    if (type(component) is not int or not isinstance(units, str) or not units.strip()
            or not isinstance(frame, str) or not frame.strip()):
        raise ValueError('Integer component and nonempty units/frame required')
    for name, value, low, high in [('resolution', resolution, 8, 512), ('interior_order', interior_order, 3, 16),
            ('boundary_samples', boundary_samples, 1, 128), ('max_pairs', max_pairs, 1, 2000000)]:
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in [{low}, {high}]')
    if any(not np.isfinite(v) or v <= 0 for v in (uv_tolerance, area_tolerance, distance_tolerance)):
        raise ValueError('Positive finite numerical tolerances required')
    try:
        provenance = json.loads(json.dumps(provenance, allow_nan=False))
        for key in ('source', 'candidate'):
            row = provenance[key]
            if not row['mesh'] or 'pose' not in row or len(row['capture_sha256']) != 64:
                raise ValueError()
            int(row['capture_sha256'], 16)
        if not provenance['qualification']:
            raise ValueError()
    except (TypeError, KeyError, ValueError):
        raise ValueError('Source/candidate mesh, pose, capture_sha256 and qualification provenance required') from None
    arrays = dict(source_positions=source_positions, source_triangles=source_triangles,
        source_uv=source_uv, source_components=source_components, source_qualified=source_qualified,
        positions=positions, triangles=triangles, footprint=footprint, domain=domain,
        source_boundary=source_boundary, boundary=boundary)
    report = {'status': 'unknown', 'reasons': [], 'provenance': provenance,
        'array_sha256': {k: _digest(v) for k, v in arrays.items()},
        'parameters': dict(component=component, units=units, frame=frame, resolution=resolution,
            interior_order=interior_order, boundary_samples=boundary_samples, uv_tolerance=uv_tolerance,
            area_tolerance=area_tolerance, distance_tolerance=distance_tolerance, max_pairs=max_pairs),
        'orientation': None, 'coverage': None, 'boundary': None, 'deviation': None,
        'limits': 'One caller-qualified UV chart and saved pose only. Raster intervals, sampled interior and boundary '
                  'distances; floating-point tolerances are not exact arithmetic. No collision, eye clearance, '
                  'likeness, source authority, live freshness, swept motion or owner approval. No pass/retention gate.'}
    def finish():
        report['input_sha256'] = hashlib.sha256(json.dumps({k: report[k] for k in
            ('provenance', 'array_sha256', 'parameters')}, sort_keys=True, allow_nan=False).encode()).hexdigest()
        try:
            json.dumps(report, allow_nan=False)
        except ValueError:
            report.update(status='unknown', orientation=None, coverage=None, boundary=None, deviation=None)
            report['reasons'].append('Derived measurements exceed finite numerical range')
        return report
    def unknown(reason):
        report['reasons'].append(reason)
        return finish()
    if (refs.shape != (len(P),) or refs.dtype.kind not in 'iu' or weights.shape != (len(P), 3)
            or weights.dtype.kind not in 'fiu'):
        return unknown('Missing or malformed explicit candidate-vertex correspondence')
    report['array_sha256'].update(correspondence_triangles=_digest(refs), correspondence_barycentric=_digest(weights))
    used = np.unique(C[D])
    if (not np.isfinite(weights[used]).all() or np.any(refs[used] < 0) or np.any(refs[used] >= len(T))
            or np.any(weights[used] < 0) or np.any(weights[used] > 1)
            or np.any(abs(weights[used].sum(axis=1) - 1) > uv_tolerance)):
        return unknown('Missing or beyond-triangle barycentric correspondence; extrapolation is not measured')
    active = np.flatnonzero((comps == component) & qualified)
    if (not len(active) or np.any(~qualified[F]) or np.any(comps[F] != component)
            or np.any(~qualified[refs[used]]) or np.any(comps[refs[used]] != component)):
        return unknown('Unqualified source footprint/correspondence or component switch')
    if len(active)**2 > 25000000:
        return unknown('Source qualification exceeds finite broad-phase budget; select a smaller chart')
    suv = UV[T[active]]
    areas = _area(suv)
    xyz_area = np.linalg.norm(np.cross(S[T[active, 1]] - S[T[active, 0]], S[T[active, 2]] - S[T[active, 0]]), axis=1)
    if (not np.isfinite(areas).all() or not np.isfinite(xyz_area).all() or np.any(abs(areas) <= area_tolerance)
            or np.any(xyz_area == 0) or np.any(np.sign(areas) != np.sign(areas[0]))):
        return unknown('Degenerate, unresolved or inconsistently oriented source chart')
    _, bad, owners = _edges(T[active])
    if bad or len({tuple(sorted(t)) for t in T[active].tolist()}) != len(active):
        return unknown('Nonmanifold, duplicate or inconsistently wound source triangles')
    adjacency = [set() for _ in active]
    for rows in owners.values():
        if len(rows) == 2:
            a, b = rows[0][0], rows[1][0]; adjacency[a].add(b); adjacency[b].add(a)
    reached, pending = set(), [0]
    while pending:
        i = pending.pop()
        if i not in reached:
            reached.add(i); pending.extend(adjacency[i] - reached)
    if len(reached) != len(active):
        return unknown('Declared qualified component contains disconnected triangle sheets')
    checked_pairs = 0
    mins, maxs = suv.min(axis=1), suv.max(axis=1)
    for i in range(len(active)):
        nearby = np.flatnonzero(np.all(maxs[i] >= mins, axis=1) & np.all(maxs >= mins[i], axis=1))
        for j in nearby[nearby > i]:
            checked_pairs += 1
            if checked_pairs > max_pairs:
                return unknown('Source-chart overlap qualification exceeded max_pairs')
            try:
                polygon = _overlap(suv[i], suv[j], uv_tolerance)[0]
            except ValueError:
                return unknown('Numerically unresolved source-chart overlap')
            if _polygon_area(polygon) > area_tolerance:
                return unknown('Ambiguous source UV: qualified triangles have positive-area overlap')
    f_boundary, f_bad, _ = _edges(T[F])
    c_boundary, _, c_owners = _edges(C[D])
    def matches(edges, expected, count):
        return (edges.min() >= 0 and edges.max() < count and all(a != b for a, b in edges)
                and len(edges) == len(expected) and {tuple(sorted(e)) for e in edges.tolist()} == expected)
    if f_bad or any(len(v) > 2 for v in c_owners.values()) or not matches(SB, f_boundary, len(S)) or not matches(CB, c_boundary, len(P)):
        return unknown('Missing, ambiguous or inconsistent complete domain/footprint boundary')
    uv = np.full((len(P), 2), np.nan)
    w = weights[used] / weights[used].sum(axis=1)[:, None]
    uv[used] = np.einsum('ij,ijk->ik', w, UV[T[refs[used]]])
    candidate_uv, footprint_uv = uv[C[D]], UV[T[F]]
    ca = _area(candidate_uv)
    sample_count = len(D) * (interior_order-1) * (interior_order-2) // 2
    if ((len(F) + len(D)) * resolution**2 > 80000000 or len(F) * len(D) > max_pairs
            or sample_count > 1000000 or sample_count * len(F) > 20000000
            or len(SB) * len(CB) * boundary_samples > 20000000):
        return unknown('Requested coverage/interior work exceeds finite measurement budget; select a smaller domain')
    report['orientation'] = {'reversed_triangle_ids': D[ca * np.sign(areas[0]) < -area_tolerance].tolist(),
        'unresolved_triangle_ids': D[abs(ca) <= area_tolerance].tolist()}
    report['boundary'] = _boundary_measure(uv[CB], UV[SB], boundary_samples)
    # Whole triangle containment complements sparse samples, including holes.
    outside = 0.
    for tri, area in zip(candidate_uv, abs(ca)):
        if area <= area_tolerance:
            continue
        covered = 0.
        for corners in footprint_uv:
            if np.any(tri.max(axis=0) < corners.min(axis=0)) or np.any(corners.max(axis=0) < tri.min(axis=0)):
                continue
            try:
                covered += _polygon_area(_overlap(tri, corners, uv_tolerance)[0])
            except ValueError:
                return unknown('Numerically unresolved candidate/footprint intersection')
        outside += max(0., float(area) - covered)
    if outside > area_tolerance:
        report['reasons'].append('Candidate triangle interiors extend beyond the declared footprint')
    # Corresponding UV interior samples, never unconstrained nearest points.
    bary = np.array([[i, j, interior_order-i-j] for i in range(1, interior_order)
                     for j in range(1, interior_order-i)], float) / interior_order
    samples_uv = np.einsum('ij,kjl->kil', bary, candidate_uv).reshape(-1, 2)
    samples_xyz = np.einsum('ij,kjl->kil', bary, P[C[D]]).reshape(-1, 3)
    sample_owner = np.repeat(D, len(bary))
    target = np.zeros_like(samples_xyz); hit = np.zeros(len(target), bool); ambiguous = np.zeros(len(target), bool)
    for t, corners in zip(F, footprint_uv):
        ids = np.flatnonzero(np.all(samples_uv >= corners.min(axis=0)-uv_tolerance, axis=1)
                            & np.all(samples_uv <= corners.max(axis=0)+uv_tolerance, axis=1))
        bs = _barycentric(samples_uv[ids], corners)
        inside = np.all(bs >= -uv_tolerance, axis=1) & np.all(bs <= 1+uv_tolerance, axis=1)
        ids, bs = ids[inside], bs[inside]
        values = bs @ S[T[t]]
        ambiguous[ids] |= hit[ids] & (np.linalg.norm(target[ids] - values, axis=1) > distance_tolerance)
        target[ids] = values; hit[ids] = True
    good = hit & ~ambiguous
    distances = np.hypot.reduce(samples_xyz[good] - target[good], axis=1)
    report['deviation'] = {'samples': len(target), 'measured_samples': int(good.sum()),
        'missing_samples': int((~hit).sum()), 'ambiguous_samples': int(ambiguous.sum()),
        'maximum': float(distances.max()) if len(distances) else None,
        'p95': float(np.percentile(distances, 95)) if len(distances) else None,
        'unmeasured_triangle_ids': np.unique(sample_owner[~good]).tolist(),
        'metric': 'Euclidean XYZ deviation at corresponding UV interior samples; unsampled extrema unresolved'}
    if not good.all():
        report['reasons'].append('Interior correspondence missing, beyond footprint or ambiguous')
    if report['orientation']['unresolved_triangle_ids']:
        report['reasons'].append('Collapsed or numerically unresolved candidate UV triangles')
    # Conservative raster: no cell touched by any mesh edge can certify multiplicity.
    lo = np.minimum(footprint_uv.min(axis=(0, 1)), candidate_uv.min(axis=(0, 1)))
    hi = np.maximum(footprint_uv.max(axis=(0, 1)), candidate_uv.max(axis=(0, 1)))
    step = (hi-lo) / resolution
    if np.any(step <= 0) or not np.isfinite(step).all():
        return unknown('Unresolved footprint extent')
    x, y = np.meshgrid(lo[0]+(np.arange(resolution)+.5)*step[0], lo[1]+(np.arange(resolution)+.5)*step[1])
    points = np.column_stack([x.ravel(), y.ravel()])
    edge_near = np.zeros(len(points), bool)
    counts = []
    radius = float(np.linalg.norm(step)) / 2 + uv_tolerance
    for mesh in (footprint_uv, candidate_uv):
        count = np.zeros(len(points), int)
        for tri in mesh:
            ids = np.flatnonzero(np.all(points >= tri.min(axis=0)-radius, axis=1)
                                & np.all(points <= tri.max(axis=0)+radius, axis=1))
            edges = tri[[[0, 1], [1, 2], [2, 0]]]
            edge_near[ids] |= _distances(points[ids], edges) <= radius
            if abs(_area(tri[None])[0]) > area_tolerance:
                b = _barycentric(points[ids], tri)
                count[ids] += np.all(b >= 0, axis=1)
        counts.append(count)
    footprint_count, multiplicity = counts
    certain = (footprint_count == 1) & ~edge_near
    cell_area = float(np.prod(step)); footprint_area = float(abs(_area(footprint_uv)).sum())
    certified_area = float(certain.sum()) * cell_area
    unresolved = max(0., footprint_area - certified_area)
    gap = float(np.count_nonzero(certain & (multiplicity == 0))) * cell_area
    overlap = float(np.count_nonzero(certain & (multiplicity > 1))) * cell_area
    report['coverage'] = {'method': 'conservative edge-excluded uniform UV raster',
        'resolution': [resolution, resolution], 'cell_size_uv': step.tolist(), 'footprint_area_uv': footprint_area,
        'candidate_outside_area_uv_sum': outside,
        'gap_area_uv_interval': [gap, min(footprint_area, gap+unresolved)],
        'overlap_area_uv_interval': [overlap, min(footprint_area, overlap+unresolved)],
        'unresolved_area_uv': unresolved, 'certain_cells': int(certain.sum()),
        'certain_multiplicity_cells': {str(k): int(np.count_nonzero(certain & (multiplicity == k)))
                                      for k in np.unique(multiplicity[certain])},
        'status': 'approximate_with_unresolved_cells',
        'limits': 'UV area, not physical surface area. All triangle-edge cells and subcell features remain unresolved. '
                  'Bounds assume the numerically qualified chart; no exact-coverage claim.'}
    # A vertex mapping to an adjacent qualified triangle may be outside this footprint.
    vertex_hit = np.zeros(len(used), bool)
    for corners in footprint_uv:
        b = _barycentric(uv[used], corners)
        vertex_hit |= np.all(b >= -uv_tolerance, axis=1) & np.all(b <= 1+uv_tolerance, axis=1)
    if not vertex_hit.all():
        report['reasons'].append('Declared candidate vertices lie beyond the footprint')
    report['status'] = 'unknown' if report['reasons'] else 'measured'
    return finish()
