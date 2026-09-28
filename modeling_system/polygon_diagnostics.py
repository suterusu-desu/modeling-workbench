"""Source-matched polygon shape measurements on each capture's native triangles."""
import hashlib
import json
import numpy as np
from .source_coverage import _digest


_FIELDS = ('co', 'tri', 'loops', 'polygon_starts', 'polygon_lengths', 'triangle_polygon')


def _capture(record):
    from .checks import _polygon_witness
    if not isinstance(record, dict):
        raise ValueError('Each capture must be a recorded array dictionary')
    arrays, hashes = {}, {}
    missing = [key for key in _FIELDS if key not in record]
    for key in _FIELDS:
        if key not in record:
            continue
        a = np.asarray(record[key])
        if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
            raise ValueError('Capture arrays must be finite numeric data: ' + key)
        arrays[key] = a; hashes[key] = _digest(a)
    if missing:
        return None, hashes, 'missing recorded ' + ', '.join(missing)
    co, tri = arrays['co'], arrays['tri']
    if co.ndim != 2 or co.shape[1:] != (3,) or not 0 < len(co) <= 200000:
        raise ValueError('Capture positions require 1..200000 finite XYZ rows')
    if tri.ndim != 2 or tri.shape[1:] != (3,):
        raise ValueError('Capture triangles require integer Mx3 rows')
    if len(tri) > 400000 or arrays['loops'].size > 800000:
        raise ValueError('Capture diagnostic is bounded to 400000 triangles and 800000 loops')
    witness, why = _polygon_witness(arrays, len(co))
    if why:
        return None, hashes, why
    arrays['co'] = co.astype(float)
    return arrays, hashes, None


def _cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def _projected_contact(a, b, c, d, tolerance):
    """Separate strict crossing from near-touch/collinear contact in this plane."""
    ab, cd = b-a, d-c
    o = np.array([_cross(ab, c-a), _cross(ab, d-a), _cross(cd, a-c), _cross(cd, b-c)])
    first, second = tolerance*np.linalg.norm(ab), tolerance*np.linalg.norm(cd)
    if ((o[0] > first and o[1] < -first or o[1] > first and o[0] < -first)
            and (o[2] > second and o[3] < -second or o[3] > second and o[2] < -second)):
        denominator = _cross(ab, cd)
        return 'proper_crossing', (_cross(c-a, cd)/denominator, _cross(c-a, ab)/denominator)
    def on(p, q, r):
        return (abs(_cross(q-p, r-p)) <= tolerance*np.linalg.norm(q-p)
                and np.all(r >= np.minimum(p, q)-tolerance) and np.all(r <= np.maximum(p, q)+tolerance))
    if on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b):
        return 'touch_or_overlap', None
    return None, None


def _shape(record, polygon, native_ids, tolerance):
    start, count = int(record['polygon_starts'][polygon]), int(record['polygon_lengths'][polygon])
    loop = record['loops'][start:start+count].astype(np.int64)
    points = record['co'][loop]
    relative = points-points[0]
    extent = float(np.linalg.norm(np.ptp(relative, axis=0)))
    if not np.isfinite(relative).all() or not np.isfinite(extent) or not np.isfinite(extent**2):
        raise ValueError('Polygon coordinates overflow; use a suitable common frame and units')
    q = relative/extent if extent > 0 else relative.copy()
    centered = q-q.mean(0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    plane_unique = bool(extent > 0 and singular[1] > tolerance and singular[1]-singular[2] > tolerance)
    area_vector = .5*np.cross(q, np.roll(q, -1, axis=0)).sum(0)
    vector_length = float(np.linalg.norm(area_vector))
    area_normal = area_vector/vector_length if vector_length > tolerance else None
    triangles = record['tri'][native_ids].astype(np.int64)
    lookup = {int(v): k for k, v in enumerate(loop)}
    local = np.array([[lookup[int(v)] for v in face] for face in triangles])
    t = q[local]
    cross = np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0])
    double_area = np.linalg.norm(cross, axis=1)
    valid = double_area > tolerance
    normals = np.zeros_like(cross); normals[valid] = cross[valid]/double_area[valid, None]
    edges = {}
    for index, face in enumerate(triangles):
        for a, b in zip(face, np.roll(face, -1)):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(index)
    angles = []
    for edge, owners in sorted(edges.items()):
        if len(owners) != 2:
            continue
        a, b = owners
        dot = float(np.clip(normals[a] @ normals[b], -1, 1)) if valid[a] and valid[b] else None
        angles.append({'edge': list(edge), 'triangle_ids': native_ids[owners].tolist(),
                       'normal_dot': dot, 'degrees': float(np.degrees(np.arccos(dot))) if dot is not None else None})
    contacts = None
    if plane_unique:
        uv = centered @ axes[:2].T
        contacts = []
        for i in range(count):
            for j in range(i+1, count):
                if j == i+1 or (i == 0 and j == count-1):
                    continue
                kind, parameters = _projected_contact(uv[i], uv[(i+1)%count], uv[j], uv[(j+1)%count], tolerance)
                if kind:
                    gap = None
                    if parameters is not None:
                        a, b = parameters
                        gap = float(np.linalg.norm(q[i]+a*(q[(i+1)%count]-q[i])-
                                                  q[j]-b*(q[(j+1)%count]-q[j]))*extent)
                    contacts.append({'loop_edges': [i, j], 'vertex_edges': [[int(loop[i]), int(loop[(i+1)%count])],
                        [int(loop[j]), int(loop[(j+1)%count])]], 'kind': kind,
                        'spatial_gap_at_projected_crossing': gap})
    lengths = np.linalg.norm(np.roll(q, -1, axis=0)-q, axis=1)
    native = []
    for k, triangle_id in enumerate(native_ids):
        native.append({'triangle_id': int(triangle_id), 'vertices': triangles[k].tolist(),
            'area': float(double_area[k]*extent**2/2), 'normalized_double_area': float(double_area[k]),
            'degenerate': bool(not valid[k]), 'normal': normals[k].tolist() if valid[k] else None,
            'dot_polygon_area_normal': float(normals[k] @ area_normal) if valid[k] and area_normal is not None else None})
    measured_angles = [a['degrees'] for a in angles if a['degrees'] is not None]
    return {'loop': loop.tolist(), 'extent': extent, 'edge_lengths': (lengths*extent).tolist(),
        'degenerate_loop_edges': np.flatnonzero(lengths <= tolerance).tolist(),
        'polygon_area_vector': (area_vector*extent**2).tolist(),
        'polygon_area_normal': area_normal.tolist() if area_normal is not None else None,
        'native_triangles': native, 'native_degenerate_triangle_ids': native_ids[~valid].tolist(),
        'native_internal_angles': angles, 'maximum_native_dihedral_degrees': max(measured_angles, default=None),
        'native_dihedral_complete': all(a['degrees'] is not None for a in angles),
        'best_fit_plane': {'status': 'measured' if plane_unique else 'unknown',
            'reason': None if plane_unique else 'collapsed/collinear loop or nonunique least-squares plane normal',
            'normalized_singular_values': singular.tolist(),
            'rms_distance': float(singular[-1]*extent/np.sqrt(count)),
            'maximum_distance': float(np.max(abs(centered @ axes[-1]))*extent) if plane_unique else None,
            'projected_contacts': contacts,
            'proper_crossing': any(c['kind'] == 'proper_crossing' for c in contacts) if contacts is not None else None},
        'self_intersection': {'status': 'not_measured'}}


def compare_polygon_shapes(before, after, polygon_ids, *, units, frame, relative_tolerance):
    """Compare stable recorded loops using each capture's validated native triangles.

    Numeric tolerance scales with each polygon's extent, not an anatomical fold
    threshold. Missing or incompatible loop evidence returns unknown, never a
    triangle-row fallback. Does not modify records, standard checks or geometry.
    """
    if not isinstance(units, str) or not units.strip() or not isinstance(frame, str) or not frame.strip():
        raise ValueError('Declare common units and frame for both captures')
    if (not np.isscalar(relative_tolerance) or not np.isfinite(relative_tolerance)
            or not 0 < relative_tolerance < 1):
        raise ValueError('relative_tolerance must be finite and strictly between zero and one')
    ids = np.asarray(polygon_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 0 < len(ids) <= 4096
            or int(ids.min()) < 0 or int(ids.max()) > np.iinfo(np.int64).max or len(np.unique(ids)) != len(ids)):
        raise ValueError('Select 1..4096 distinct nonnegative polygon IDs')
    ids = ids.astype(np.int64)
    a, ah, aw = _capture(before); b, bh, bw = _capture(after)
    params = {'units': units, 'frame': frame, 'relative_tolerance': float(relative_tolerance), 'polygon_ids': ids.tolist()}
    revision = {'before': ah, 'after': bh, 'parameters': params}
    result = {'status': 'unknown', 'reasons': [], **params, 'array_sha256': {'before': ah, 'after': bh},
        'input_revision': hashlib.sha256(json.dumps(revision, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'polygons': [], 'limits': 'Recorded indexed polygon-loop identity only; no inferred original editable identity '
            'through modifiers or reindexing. Native triangles are independently measured per capture, never paired by '
            'old row IDs. Internal normal angles depend on native tessellation. Best-fit-plane crossings are projection '
            'measurements, not 3D self-intersections; reported crossing gaps can be nonzero. Degenerate normals and '
            'ambiguous planes remain unknown. Area-normal rotation in the common frame is not a fold test. '
            'No inter-polygon collision, anatomical, visible-quality, swept-motion or retention verdict.'}
    if aw or bw:
        result['reasons'] = [{'capture': label, 'reason': why} for label, why in [('before', aw), ('after', bw)] if why]
        return result
    if a['co'].shape != b['co'].shape or any(not np.array_equal(a[k], b[k]) for k in ('loops', 'polygon_starts', 'polygon_lengths')):
        result['reasons'] = [{'reason': 'vertex count or ordered recorded polygon-loop identity differs'}]
        return result
    if ids.max() >= len(a['polygon_starts']):
        raise ValueError('Selected polygon ID is outside the recorded polygon table')
    counts = a['polygon_lengths'][ids].astype(np.int64)
    if np.any(counts > 64) or int(np.sum(counts*(counts-3)//2)) > 2000000:
        raise ValueError('Select polygons with at most 64 corners and at most 2000000 nonadjacent edge pairs')
    native_ids = []
    for record in (a, b):
        owners = record['triangle_polygon'].astype(np.int64)
        order = np.argsort(owners, kind='stable')
        starts = np.r_[0, np.cumsum(np.bincount(owners, minlength=len(a['polygon_starts'])))]
        native_ids.append([order[starts[p]:starts[p+1]] for p in ids])
    for index, polygon in enumerate(ids):
        old = _shape(a, int(polygon), native_ids[0][index], relative_tolerance)
        new = _shape(b, int(polygon), native_ids[1][index], relative_tolerance)
        x, y = old['best_fit_plane']['proper_crossing'], new['best_fit_plane']['proper_crossing']
        transition = ('unknown' if x is None or y is None else 'present_in_both' if x and y else
                      'introduced_in_projection' if y else 'absent_after_in_projection' if x else 'absent_in_both')
        nx, ny = old['polygon_area_normal'], new['polygon_area_normal']
        result['polygons'].append({'polygon_id': int(polygon), 'loop': old['loop'], 'before': old, 'after': new,
            'native_triangle_rows_equal': bool(np.array_equal(a['tri'][native_ids[0][index]], b['tri'][native_ids[1][index]])),
            'projected_crossing_transition': transition,
            'before_after_area_normal_dot': float(np.clip(np.dot(nx, ny), -1, 1)) if nx is not None and ny is not None else None})
    result.update(status='measured', correspondence='identical_ordered_recorded_polygon_loops',
        triangle_layout_equal=bool(np.array_equal(a['tri'], b['tri'])), selected_polygons=len(ids))
    return result
