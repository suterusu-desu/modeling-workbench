"""Standard construction checks from a short declaration, and trial guards generated from it.

A declaration names the moving object, the region allowed to move, the accepted rest, protected objects, clearance
obstacles, symmetry and, for a closing feature, its two edges and hinge. `run_checks` measures a candidate's saved poses
against it and returns one row per check with its observed value, limit and status; a baseline's values are reported
beside them for comparison only, so a rebuild is judged against its limits and not against its predecessor.
`standard_policy` turns the same declaration into a `PreservationPolicy` for native trials and retention, so no
per-trial adapter is written.

Poses are saved as arrays named `<phase>::<object>::co` (vertex positions) and `<phase>::<object>::tri` (triangles),
phases as numbers, the lowest one (0) the rest pose. Numbers can reject a candidate; they never approve its appearance.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from .construction_diagnostics import closing_edges, line_depth, local_reversals, triangle_stretches
from .motion_paths import keep_clearance

LIMITS = {
    'rest_identity': 1e-6,          # largest distance from the accepted rest at phase 0
    'still_outside': 1e-6,          # largest travel of a vertex outside the region at any phase
    'protected_unchanged': 1e-6,    # largest change of a protected object (from the baseline, or from its own rest)
    'clearance_shortfall': 0.,      # how far a point clear of the obstacle at rest comes inside its required clearance:
                                    # with a baseline, allowance over the baseline's shortfall
    'folds': 0.,                    # locally reversed region triangles at the worst phase: allowance on new ones
    'reversing_vertices': 0.,       # moving vertices whose progress along their own chord goes back: allowance on new ones
    'symmetry': 1e-5,               # largest distance between a vertex and its mirrored partner's mirror image
    'facing_travel_share': .05,     # opposing edge travel as a share of the rest opening
    'closing_spread': .1,           # largest difference in closed share between stretches of the edge before closure
    'seam_share': .05,              # closed edge's median distance from the opposing edge, share of the opening
    'roll_deviation_share': .1,     # distance from one roll about the hinge, share of the opening
    'closed_depth_error': .02,      # closed line's depth below the corner line minus the declared target, share of width
    'corner_travel_share': .1,      # largest travel of a closing corner, share of the corner distance
    'corner_ramp': 0.,              # how far the margin's travel reaches 90 % of its most sooner than `min_ramp` of its
                                    # length from either end (a pinched corner)
    'corner_compression': 0.,       # region triangles near a closing corner squeezed below `stretch` (.5) of their rest
                                    # size in some direction at the worst phase: allowance on new ones
    'band_reach': .55,              # height above the margin (share of the width) where the band's travel falls below a tenth
    'lash_travel': 0.,              # how far a lash root's travel leaves .9-1.05 of the travel of the margin under it
    'lash_turn': 35.,               # largest turn of a lash about its root relative to the lid it rides (degrees)
    'lash_length': .25,             # largest change of a lash's root-to-tip length, share of its rest length
    'lash_timing': .1,              # largest difference between a lash root's and its host's share of travel at a phase
    'attachment_timing': .1,        # largest difference between an attached point's and its host's share of travel
    'attachment_still': .02,        # largest travel of an attached point whose host stays still, share of the largest travel
    'combination_seam': .05,        # largest |median signed seam gap| of a declared combination, share of the opening
    'carrier_shapes': 2.,           # blend shapes needed to carry the motion within the declared tolerance
}
# Defects a baseline may already have: with a baseline the limit is an allowance over the baseline's value, so a candidate
# is not failed for what it inherited (the report still shows it) and a rebuild is not held to its predecessor's
# geometry. Rest identity, motion outside the region and the closing checks describe the construction and stay absolute.
INCREASE_ONLY = ('clearance_shortfall', 'folds', 'reversing_vertices', 'corner_compression')
REVERSAL_SHARE = .02                # progress falling by more than this share of the chord counts as moving back
MOVER_SHARE = .2                    # vertices whose end chord exceeds this share of the largest are followed for reversals


POLYGON_WITNESS = ('loops', 'polygon_starts', 'polygon_lengths', 'triangle_polygon')


def load_states(source):
    """Poses from an .npz path or `<phase>::<object>::field` arrays, retaining co, tri and polygon witness fields.
    Returns {phase: {object: {field: array}}}, ordered by phase value; already nested poses are sorted likewise."""
    if isinstance(source, dict) and source and all(isinstance(v, dict) for v in source.values()):
        try:
            return dict(sorted(source.items(), key=lambda item: float(item[0])))
        except ValueError:
            raise ValueError('Phases must be numbers') from None
    data = np.load(source) if isinstance(source, (str, Path)) else source
    keys = data.files if hasattr(data, 'files') else list(data)
    states = {}
    for key in keys:
        parts = key.split('::')
        if len(parts) != 3 or parts[2] not in ('co', 'tri', *POLYGON_WITNESS):
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        if not math.isfinite(value):
            raise ValueError(f'Phase {parts[0]!r} is not a finite number')
        states.setdefault(parts[0], {}).setdefault(parts[1], {})[parts[2]] = np.asarray(data[key])
    if not states:
        raise ValueError('No `<phase>::<object>::co` arrays found')
    return dict(sorted(states.items(), key=lambda item: float(item[0])))


def _load(spec, base, key):
    """An inline list/array, or {'path': .npz relative to the declaration, 'key': array name}."""
    if isinstance(spec, dict):
        path = Path(spec['path'])
        path = path if path.is_absolute() else Path(base or '.') / path
        with np.load(path) as data:
            return np.asarray(data[spec.get('key', key)])
    return np.asarray(spec)


def _indices(spec, base, key, count, what):
    idx = _load(spec, base, key)
    if idx.dtype == bool:
        if idx.shape != (count,):
            raise ValueError(f'The {what} mask needs one value per vertex')
        idx = np.flatnonzero(idx)
    if (idx.ndim != 1 or idx.dtype.kind not in 'iu' or not len(idx) or idx.min() < 0 or idx.max() >= count
            or len(np.unique(idx)) != len(idx)):
        raise ValueError(f'The {what} needs distinct vertex indices of the object')
    return idx.astype(np.int64)


def validate_declaration(declaration):
    """Check a declaration's shape and return a copy. Only `object` and `region` are required."""
    d = deepcopy(declaration)
    if not isinstance(d, dict) or not isinstance(d.get('object'), str) or not d['object'] or 'region' not in d:
        raise ValueError('A declaration needs the moving object and the region allowed to move')
    unknown = set(d) - {'object', 'region', 'region_label', 'rest', 'protected', 'clearance', 'symmetry', 'closing',
                        'carrier', 'limits', 'arrays', 'description', 'band', 'lash', 'combinations', 'attachments',
                        'preserved'}
    if unknown:
        raise ValueError(f'Unknown declaration fields: {sorted(unknown)}')
    if not all((isinstance(p, str) and p) or (isinstance(p, dict) and isinstance(p.get('object'), str) and p['object']
                                              and p.get('against', 'baseline') in ('baseline', 'rest'))
               for p in d.get('protected', [])):
        raise ValueError("Protected objects are object names, or {'object': name, 'against': 'rest' or 'baseline'}")
    for entry in d.get('clearance', []):
        if (not isinstance(entry, dict) or not isinstance(entry.get('obstacle'), str)
                or not _nonnegative(entry.get('minimum', 0.))):
            raise ValueError('Each clearance entry needs an obstacle object and a nonnegative minimum')
        gaze = entry.get('gaze')
        if gaze is not None and (not isinstance(gaze, dict) or len(gaze.get('pivot', [])) != 3 or not all(
                len(r) == 2 and len(r[0]) == 3 and _nonnegative(abs(r[1])) for r in gaze.get('rotations', []))):
            raise ValueError('Gaze needs a pivot and rotations as [[axis x, y, z], degrees] pairs')
    symmetry = d.get('symmetry')
    if symmetry is not None and (not isinstance(symmetry, dict) or symmetry.get('axis') not in (0, 1, 2)):
        raise ValueError('Symmetry needs the mirror axis 0, 1 or 2')
    closing = d.get('closing')
    if closing is not None and (not isinstance(closing, dict) or 'moving' not in closing or 'facing' not in closing
                                or ('pivot' in closing) != ('axis' in closing)):
        raise ValueError('Closing needs the moving and facing edges, and both a pivot and an axis or neither')
    squeeze = (closing or {}).get('corner_compression')
    if squeeze is not None and (not (closing or {}).get('corners') or not (squeeze is True or (
            isinstance(squeeze, dict) and all(_positive(squeeze.get(k, 1.)) for k in ('radius', 'stretch'))))):
        raise ValueError('corner_compression needs the closing corners and optionally a positive radius and stretch')
    if closing is not None and 'corners' in closing and len(closing['corners']) != 2:
        raise ValueError('Closing corners are the two vertices where the edges meet')
    depth = (closing or {}).get('closed_depth')
    if depth is not None and (not isinstance(depth, dict) or not _nonnegative(depth.get('target'))
                              or len(depth.get('corners', (closing or {}).get('corners', []))) != 2):
        raise ValueError('closed_depth needs a nonnegative target share and the two corner vertices')
    for key in ('band', 'lash', 'combinations'):
        if key in d and not closing:
            raise ValueError(f'{key} checks need the closing edges')
    lash = d.get('lash')
    if lash is not None and (not isinstance(lash, dict) or not isinstance(lash.get('object'), str)
                             or len(lash.get('roots', [])) == 0 or len(lash.get('roots', [])) != len(lash.get('tips', []))):
        raise ValueError('The lash check needs the lash object and matching root and tip vertices')
    for entry in d.get('attachments', []):
        if not isinstance(entry, dict) or not isinstance(entry.get('object'), str) or not entry['object']:
            raise ValueError("Each attachment needs its object (and optionally 'host' and 'points')")
    for combo in d.get('combinations', []):
        if not isinstance(combo, dict) or not combo.get('name') or not combo.get('poses'):
            raise ValueError('Each combination needs a name and a poses file whose last phase is the combined closed pose')
    carrier = d.get('carrier')
    if carrier is not None and (not isinstance(carrier, dict) or not _positive(carrier.get('tolerance'))):
        raise ValueError('The carrier check needs a positive tolerance')
    for key, value in d.get('limits', {}).items():
        if key not in LIMITS or not _nonnegative(value):
            raise ValueError(f'Limit {key!r} must be a known check with a nonnegative bound')
    if set(d.get('arrays', {})) - {'candidate', 'baseline'}:
        raise ValueError("Arrays name only the 'candidate' and 'baseline' files")
    if 'preserved' in d:
        _validate_preserved(d['preserved'])
    return d


def _validate_preserved(spec):
    """Selected vertex regions that must match the accepted baseline at declared phases (see `_preserved`)."""
    if not isinstance(spec, dict) or set(spec) - {'units', 'frame', 'tolerance', 'phases', 'regions'}:
        raise ValueError("preserved takes 'units', 'tolerance', 'phases', 'regions' and optionally 'frame'")
    if not isinstance(spec.get('units'), str) or not spec['units'].strip():
        raise ValueError('preserved needs the units of the saved positions, for example "m"')
    if 'frame' in spec and (not isinstance(spec['frame'], str) or not spec['frame'].strip()):
        raise ValueError('preserved frame must be a nonempty description')
    if not _nonnegative(spec.get('tolerance')):
        raise ValueError('preserved needs a finite nonnegative Euclidean distance tolerance in those units')
    phases = spec.get('phases')
    if (not isinstance(phases, list) or not phases
            or not all(type(g) in (int, float) and math.isfinite(g) for g in phases)):
        raise ValueError('preserved needs the phases to compare, as a nonempty list of finite numbers')
    if len({float(g) for g in phases}) != len(phases):
        raise ValueError('preserved phases must be distinct numbers (0 and 0.0 are the same phase)')
    regions = spec.get('regions')
    if not isinstance(regions, list) or not regions:
        raise ValueError('preserved needs at least one region')
    labels = []
    for i, region in enumerate(regions):
        if (not isinstance(region, dict) or set(region) - {'label', 'object', 'vertices'}
                or not isinstance(region.get('object'), str) or not region['object'] or 'vertices' not in region):
            raise ValueError("Each preserved region needs its 'object' and 'vertices' (and optionally a 'label')")
        vertices = region['vertices']
        if not (isinstance(vertices, list) or (isinstance(vertices, dict) and isinstance(vertices.get('path'), str)
                                               and set(vertices) <= {'path', 'key'})):
            raise ValueError("Preserved vertices are an inline index/mask list or {'path': .npz, 'key': array}")
        label = region.get('label', f"{region['object']}[{i}]")
        if not isinstance(label, str) or not label:
            raise ValueError('A preserved region label must be a nonempty string')
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ValueError('Preserved region labels must be distinct')


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _positive(value):
    return _nonnegative(value) and value > 0


def applicable_checks(declaration):
    """The check ids a declaration asks for, in report order."""
    d = validate_declaration(declaration)
    ids = ['rest_identity'] if 'rest' in d else []
    ids += ['still_outside']
    ids += ['protected_unchanged'] if d.get('protected') else []
    ids += ['preserved_regions'] if d.get('preserved') else []
    ids += ['clearance_shortfall'] if d.get('clearance') else []
    ids += ['folds', 'reversing_vertices']
    ids += ['symmetry'] if d.get('symmetry') else []
    if d.get('closing'):
        ids += ['facing_travel_share', 'closing_spread', 'seam_share']
        ids += ['roll_deviation_share'] if 'pivot' in d['closing'] else []
        ids += ['closed_depth_error'] if d['closing'].get('closed_depth') else []
        ids += ['corner_travel_share', 'corner_ramp'] if d['closing'].get('corners') else []
        ids += ['corner_compression'] if d['closing'].get('corner_compression') else []
        ids += ['band_reach'] if d.get('band') is not None else []
        ids += ['lash_travel', 'lash_turn', 'lash_length', 'lash_timing'] if d.get('lash') else []
        ids += ['combination_seam'] if d.get('combinations') else []
    ids += ['attachment_timing', 'attachment_still'] if d.get('attachments') else []
    ids += ['carrier_shapes'] if d.get('carrier') else []
    return ids


def _stack(states, name):
    missing = [g for g, objects in states.items() if name not in objects or 'co' not in objects[name]]
    if missing:
        raise ValueError(f'Object {name!r} missing at phases {missing}')
    P = np.stack([np.asarray(objects[name]['co'], float) for objects in states.values()])
    if P.ndim != 3 or P.shape[2] != 3 or not np.isfinite(P).all():
        raise ValueError(f'Object {name!r} needs finite XYZ positions with the same vertex count at every phase')
    return P


def _rotation(axis, degrees):
    k = np.asarray(axis, float); k = k / np.linalg.norm(k); a = np.radians(float(degrees))
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _obstacle_parts(O0, gap_share):
    """The obstacle's separate parts at rest: points closer than `gap_share` of its extent join, so an eyeball with its
    iris and highlight stays one part and a pair of eyeballs is two. Parts under a tenth of the points are left in the
    nearest major part's company (they do not split the obstacle). Returns the major parts' index arrays."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(O0); reach = gap_share * float(np.linalg.norm(np.ptp(O0, axis=0)))
    pairs = cKDTree(O0).query_pairs(reach, output_type='ndarray') if reach > 0 else np.zeros((0, 2), int)
    _, labels = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)),
                                     directed=False)
    sizes = np.bincount(labels)
    return [np.flatnonzero(labels == j) for j in np.argsort(-sizes) if sizes[j] >= .1 * n]


def _clearance_row(P, points, O, entry, minimum, phases, parts=None, assign=None):
    """Shortfall of the points that start outside the obstacle, each keeping the smaller of the minimum and its own rest
    clearance; points inside the envelope at rest (a socket beside or behind the obstacle) are not followed. With several
    obstacle parts (two eyeballs in one object) each point is measured against its nearest part from that part's centre."""
    gaps = []
    groups = [(np.arange(len(points)), None)] if not parts or len(parts) < 2 else [
        (np.flatnonzero(assign == j), part) for j, part in enumerate(parts)]
    for g in range(len(phases)):
        pts = P[g][points]; gap = np.full(len(points), np.nan)
        for members, part in groups:
            if not len(members):
                continue
            obstacle = O[g] if part is None else O[g][part]
            centre = (np.asarray(entry['centre'], float) if entry.get('centre') is not None else obstacle.mean(axis=0))
            env = keep_clearance(pts[members], obstacle, centre, -1.,
                                 angular_radius_degrees=entry.get('angular_radius_degrees', 2.))['envelope']
            gap[members] = np.linalg.norm(pts[members] - centre, axis=1) - env
        gaps.append(gap)
    gaps = np.array(gaps)                                  # (phases, points); NaN where the obstacle is not behind
    followed = np.isfinite(gaps[0]) & (gaps[0] >= 0)
    need = np.minimum(minimum, gaps[0])
    miss = np.where(followed[None] & np.isfinite(gaps), need[None] - gaps, -np.inf)
    worst = np.unravel_index(np.argmax(miss), miss.shape) if followed.any() else None
    value = max(0., float(miss[worst])) if worst is not None and np.isfinite(miss[worst]) else 0.
    return {'minimum': minimum, 'shortfall': value,
            'at': {'phase': phases[worst[0]], 'vertex': int(points[worst[1]])} if value > 0 else None,
            'closest_followed': float(np.nanmin(np.where(followed[None], gaps, np.nan))) if followed.any() else None,
            'followed_points': int(followed.sum()),
            'inside_at_rest': int(np.count_nonzero(np.isfinite(gaps[0]) & (gaps[0] < 0))),
            'not_in_front_of_obstacle': int(np.count_nonzero(~np.isfinite(gaps[0])))}


def _measure(d, states, base, baseline=None, mover_cut=None):
    """{check id: (observed or None, detail)} for one set of poses."""
    phases = list(states)
    if float(phases[0]) != 0.:
        raise ValueError('The lowest phase must be the rest pose, 0')
    P = _stack(states, d['object']); n = P.shape[1]
    region = _indices(d['region'], base, 'region', n, 'region')
    inside = np.zeros(n, bool); inside[region] = True
    out = {}

    if 'rest' in d:
        rest = _load(d['rest'], base, 'co').astype(float)
        if rest.shape != (n, 3):
            raise ValueError('The accepted rest needs one position per vertex of the object')
        gap = np.linalg.norm(P[0] - rest, axis=1)
        out['rest_identity'] = (float(gap.max()), {'worst_vertex': int(gap.argmax())})

    travel = np.linalg.norm(P - P[0], axis=2)                                   # (phases, vertices)
    outside = travel[:, ~inside]
    out['still_outside'] = (float(outside.max()) if outside.size else 0., {
        'moved_outside': int(np.count_nonzero(outside.max(axis=0) > LIMITS['still_outside'])) if outside.size else 0})

    if d.get('protected'):
        worst, against = {}, {}
        for entry in d['protected']:
            name = entry if isinstance(entry, str) else entry['object']
            still = isinstance(entry, dict) and entry.get('against') == 'rest'   # must not move, whatever the baseline did
            Q = _stack(states, name)
            if baseline is not None and not still:
                B = _stack(baseline, name)
                if B.shape != Q.shape:
                    raise ValueError(f'Protected object {name!r} differs in shape from the baseline')
                worst[name], against[name] = float(np.abs(Q - B).max()), 'baseline'
            else:
                worst[name], against[name] = float(np.abs(Q - Q[0]).max()), 'own rest'
        out['protected_unchanged'] = (max(worst.values()), {'objects': worst, 'against': against})

    if d.get('clearance'):
        rows, shortfall = [], 0.
        for entry in d['clearance']:
            points = region if entry.get('points', 'region') == 'region' else _indices(
                entry['points'], base, 'points', n, 'clearance points')
            O = _stack(states, entry['obstacle']); minimum = float(entry.get('minimum', 0.))
            parts, assign, centre_used = None, None, 'declared' if entry.get('centre') is not None else 'obstacle centroid'
            if entry.get('centre') is None:                # the obstacle's own centroid is only meaningful for one part
                parts = _obstacle_parts(O[0], float(entry.get('part_gap', .05)))
                if len(parts) > 1:
                    from scipy.spatial import cKDTree
                    near = np.stack([cKDTree(O[0][part]).query(P[0][points])[0] for part in parts])
                    assign = near.argmin(axis=0)
                    centre_used = f'nearest of {len(parts)} obstacle parts'
            variants = [('rest gaze', O)]
            gaze = entry.get('gaze')
            for axis_vector, degrees in (gaze or {}).get('rotations', []):
                R = _rotation(axis_vector, degrees); pivot = np.asarray(gaze['pivot'], float)
                variants.append((f'{degrees:+g} deg about {list(axis_vector)}', (O - pivot) @ R.T + pivot))
            worst_row = None
            for label, obstacle in variants:
                row = _clearance_row(P, points, obstacle, entry, minimum, phases, parts, assign)
                row.update(obstacle=entry['obstacle'], gaze=label, centre=centre_used)
                if assign is not None:
                    row['parts'] = [{'points': int(len(part)), 'centre_at_rest': O[0][part].mean(axis=0).tolist(),
                                     'clearance_points': int(np.count_nonzero(assign == j))} for j, part in enumerate(parts)]
                if worst_row is None or row['shortfall'] > worst_row['shortfall']:
                    worst_row = row
            if len(variants) > 1:
                worst_row['gazes_measured'] = [label for label, _ in variants]
            rows.append(worst_row)
            shortfall = max(shortfall, worst_row['shortfall'])
        out['clearance_shortfall'] = (shortfall, {'obstacles': rows})

    tri = states[phases[0]][d['object']].get('tri'); region_tri = None
    if tri is None:
        out['folds'] = (None, {'reason': 'no triangles saved at the rest phase'})
    else:
        tri = np.asarray(tri).astype(np.int64)
        tri = tri[inside[tri].all(axis=1)] if len(tri) else tri; region_tri = tri
        folded = [np.flatnonzero(local_reversals(P[0], P[g], tri)) if len(tri) else np.zeros(0, int) for g in range(len(P))]
        counts = [len(f) for f in folded]
        out['folds'] = (float(max(counts)), {'per_phase': dict(zip(phases, counts)), 'region_triangles': int(len(tri)),
                                             '_folded': {g: {tuple(int(v) for v in tri[i]) for i in f}
                                                         for g, f in zip(phases, folded)}})

    offset = P[:, region] - P[0, region]; chord = offset[-1]; length = np.linalg.norm(chord, axis=1)
    cut = float(mover_cut) if mover_cut is not None else MOVER_SHARE * max(float(length.max()), 1e-300)
    movers = length > cut
    progress = np.einsum('gnk,nk->gn', offset[:, movers], chord[movers]) / np.maximum(length[movers], 1e-300) ** 2
    fall = (progress[:-1] - progress[1:]).max(axis=0) if len(progress) > 1 else np.zeros(int(movers.sum()))
    back = fall > REVERSAL_SHARE
    out['reversing_vertices'] = (float(np.count_nonzero(back)), {
        'followed_movers': int(movers.sum()), 'moved_vertices': int(np.count_nonzero(travel[:, region].max(axis=0) > 0)),
        'mover_cut': cut, 'largest_fall': float(fall.max()) if len(fall) else 0.,
        'examples': region[movers][back][:12].tolist(), '_reversing': set(region[movers][back].tolist())})

    if d.get('symmetry'):
        from scipy.spatial import cKDTree
        axis = d['symmetry']['axis']; plane = float(d['symmetry'].get('plane', 0.))
        mirror = lambda X: np.concatenate([X[..., :axis], 2 * plane - X[..., axis:axis + 1], X[..., axis + 1:]], -1)
        distance, partner = cKDTree(P[0]).query(mirror(P[0][region]))
        tolerance = float(d['symmetry'].get('match_tolerance', 1e-5))
        matched = distance <= tolerance
        diff = np.linalg.norm(mirror(P[:, region[matched]]) - P[:, partner[matched]], axis=2) if matched.any() else np.zeros(1)
        out['symmetry'] = (float(diff.max()), {'unmatched_region_vertices': int(np.count_nonzero(~matched))})

    if d.get('closing'):
        c = d['closing']
        moving_edge = _load(c['moving'], base, 'moving').astype(np.int64)
        facing_edge = _load(c['facing'], base, 'facing').astype(np.int64)
        if len(P) < 2:
            raise ValueError('Closing checks need at least one pose after rest')
        report = closing_edges(P[0], P[1:], moving_edge, facing_edge, pivot=c.get('pivot'), axis=c.get('axis'),
                               parts=int(c.get('parts', 3)), against='pose')
        s = report['summary']; rows = report['poses']
        out['facing_travel_share'] = (s['facing_travel_share_max'], {})
        # One rate: with a hinge, the margin's own turn share per stretch (independent of how the facing edge moves);
        # without one, its closed share of the gap to the facing edge where that edge is at each pose.
        source = 'turn_share' if 'pivot' in c else 'closure'
        parts = {phases[r['pose'] + 1]: r[source]['parts_median'] if r[source] else None for r in rows}
        spreads = [max(p) - min(p) for p in list(parts.values())[:-1] if p and None not in p]
        out['closing_spread'] = (float(max(spreads)) if spreads else None, {
            'measured_on': source, 'parts_median_per_pose': parts, 'pointwise_p10_p90_max': s['closure_spread_max']})
        seam = rows[-1]['seam_to_facing']
        out['seam_share'] = (seam['median'] / s['rest_opening_median'], {
            'rest_opening': s['rest_opening_median'], 'largest_share': seam['share_of_opening_max']})
        if c.get('closed_depth'):
            spec = c['closed_depth']; a, b = (int(v) for v in spec.get('corners', c.get('corners')))
            view = {k: spec[k] for k in ('up', 'across') if k in spec}
            closed = line_depth(P[-1][moving_edge], P[-1][a], P[-1][b], **view)
            rest = line_depth(P[0][facing_edge], P[0][a], P[0][b], **view)
            out['closed_depth_error'] = (abs(closed['depth_share'] - float(spec['target'])), {
                'closed_depth_share': closed['depth_share'], 'target': float(spec['target']),
                'facing_rest_depth_share': rest['depth_share']})
        if 'pivot' in c:
            out['roll_deviation_share'] = (s['roll_deviation_share_max'], {
                'turn_share_parts_median_per_pose': {phases[r['pose'] + 1]: r['turn_share']['parts_median']
                                                     if r['turn_share'] else None for r in report['poses']}})
        if c.get('corners'):
            out.update(_corners(P, moving_edge, [int(v) for v in c['corners']], float(c.get('min_ramp', .2))))
            if c.get('corner_compression'):
                out['corner_compression'] = _corner_compression(P, region_tri, [int(v) for v in c['corners']],
                                                                c['corner_compression'])
        if d.get('band') is not None:
            out['band_reach'] = _band(P, d['band'], c, moving_edge, base, n)
        if d.get('lash'):
            out.update(_lash(states, P, d['lash'], c, moving_edge, base))
        if d.get('combinations'):
            out['combination_seam'] = _combinations(d, P, moving_edge, facing_edge, s['rest_opening_median'], base)

    if d.get('attachments'):
        out.update(_attachments(states, P, d['attachments'], base))

    if d.get('carrier'):
        weights = d['carrier'].get('weights') or {}
        s = np.array([float(weights.get(g, g)) for g in phases])
        if s[0] != 0. or s[-1] != 1. or np.any(np.diff(s) <= 0):
            raise ValueError('Carrier weights must rise from 0 at rest to 1 at the last phase')
        D = P[:, region] - P[0, region]; one = D - s[:, None, None] * D[-1]
        b = 4 * s * (1 - s); mid = np.einsum('g,gnk->nk', b, one) / max(float(b @ b), 1e-300)
        two = one - b[:, None, None] * mid
        r1, r2 = np.linalg.norm(one, axis=2).max(axis=0), np.linalg.norm(two, axis=2).max(axis=0)
        e1, e2 = float(r1.max()), float(r2.max())
        tolerance = float(d['carrier']['tolerance'])
        out['carrier_shapes'] = (1. if e1 <= tolerance else 2. if e2 <= tolerance else 3., {
            'one_shape_error': e1, 'two_shape_error': e2, 'tolerance': tolerance,
            'two_shape_worst_vertices': region[np.argsort(r2)[::-1][:12]].tolist(),
            'vertices_over_tolerance_with_two_shapes': int(np.count_nonzero(r2 > tolerance)),
            'mid_shape': 'driven at 4s(1-s) by the same control', 'three_means': 'more than two'})
    return out


def _corners(P, margin, corners, min_ramp):
    """Corner travel (share of the corner distance) and how gradually the margin's travel rises from each end."""
    a, b = corners; width = float(np.linalg.norm(P[0][b] - P[0][a]))
    travel = np.linalg.norm(P - P[0], axis=2)
    corner = float(max(travel[:, a].max(), travel[:, b].max()) / width)
    t = travel[-1][margin]; step = np.linalg.norm(np.diff(P[0][margin], axis=0), axis=1)
    arc = np.r_[0., np.cumsum(step)] / max(step.sum(), 1e-300)
    high = np.flatnonzero(t >= .9 * t.max()) if t.max() > 0 else np.array([], int)
    ramps = (float(arc[high[0]]), float(1 - arc[high[-1]])) if len(high) else (1., 1.)
    return {'corner_travel_share': (corner, {'width': width, 'corners': corners}),
            'corner_ramp': (max(0., min_ramp - min(ramps)), {'ramp_from_start': ramps[0], 'ramp_from_end': ramps[1],
                                                             'min_ramp': min_ramp})}


def _corner_compression(P, tri, corners, spec):
    """Region triangles within `radius` (share of the corner distance, default .2) of a closing corner whose smaller
    principal stretch falls below `stretch` (default .5), summed over the in-between phases: the column a band's turn
    step squeezes up from a blunt corner shows during the motion. The closed pose is squeezed near the corners by design,
    so its count is reported, not gated (on a real lid the closed counts of a fixed and an unfixed corner were equal)."""
    if tri is None or not len(tri):
        return None, {'reason': 'no region triangles saved at the rest phase'}
    spec = {} if spec is True else spec
    radius, below = float(spec.get('radius', .2)), float(spec.get('stretch', .5))
    a, b = corners; width = float(np.linalg.norm(P[0][b] - P[0][a])); centre = P[0][tri].mean(axis=1)
    near = np.minimum(np.linalg.norm(centre - P[0][a], axis=1), np.linalg.norm(centre - P[0][b], axis=1)) <= radius * width
    T = tri[near]
    between = list(range(1, len(P) - 1)) or [len(P) - 1]          # with only rest and end, the end
    if not len(T):
        return 0., {'triangles_near_corners': 0, '_squeezed': {}}
    squeezed = {}; counts = []
    for g, phase in enumerate(P):
        smallest = triangle_stretches(P[0], phase, T)[1]
        hit = np.flatnonzero(smallest < below); counts.append(len(hit))
        if g in between:
            squeezed[g] = {tuple(int(v) for v in T[i]) for i in hit}
    return float(sum(counts[g] for g in between)), {
        'triangles_near_corners': int(len(T)), 'radius': radius, 'stretch': below, 'per_phase': counts,
        'measured_on': 'the in-between phases (summed)', 'closed_phase': counts[-1], '_squeezed': squeezed}


def _band(P, band, c, margin, base, n):
    """Where the band above the margin stops moving, as a share of the width (study.band_profile at the last pose)."""
    from .study import band_profile
    edge = _indices(band['margin'], base, 'margin', n, 'band margin') if 'margin' in band else margin
    pool = (np.arange(n) if band.get('candidates') is None
            else _indices(band['candidates'], base, 'candidates', n, 'band candidates'))
    width = band.get('width') or (float(np.linalg.norm(P[0][c['corners'][1]] - P[0][c['corners'][0]]))
                                  if c.get('corners') else None)
    view = {k: band[k] for k in ('up', 'across') if k in band}
    profile = band_profile(P[0], P[-1], edge, pool, width=width, **view)
    detail = {'rows': profile['rows'], 'width': profile['width'], 'column': profile['column']}
    if profile['reach_share'] is not None:
        return float(profile['reach_share']), detail
    top = max((r['height_share'] for r in profile['rows']), default=None)
    detail['note'] = ('still moving at the top of the measured points: the band reaches at least there; declare '
                      'candidates reaching higher if that is below the limit')
    return top, dict(detail, at_least=True)


def _lash(states, P, lash, c, margin, base):
    """Lash carriage: root travel against the margin under it, the lash's own turn, length, timing."""
    from scipy.spatial import cKDTree
    L = _stack(states, lash['object'])
    pairs = {}
    for key, count in (('roots', L.shape[1]), ('tips', L.shape[1]), ('host', P.shape[1])):
        if lash.get(key) is not None:                      # pairs, so a root (or host) may serve several lash points
            ids = _load(lash[key], base, key)
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or ids.min() < 0 or ids.max() >= count:
                raise ValueError(f'Lash {key} are vertex indices of their object')
            pairs[key] = ids.astype(np.int64)
    roots, tips = pairs['roots'], pairs['tips']
    if len(tips) != len(roots) or len(pairs.get('host', roots)) != len(roots):
        raise ValueError('Lash roots, tips and hosts pair up one to one')
    if 'host' in pairs:
        host = pairs['host']
    else:
        host = margin[cKDTree(P[0][margin]).query(L[0][roots])[1]]
    rt = np.linalg.norm(L[:, roots] - L[0, roots], axis=2); ht = np.linalg.norm(P[:, host] - P[0, host], axis=2)
    carried = ht[-1] > .1 * max(float(ht[-1].max()), 1e-300)          # roots whose host moves by the end
    if not carried.any():
        none = (None, {'reason': 'the margin under the lash does not move'})
        return dict.fromkeys(('lash_travel', 'lash_turn', 'lash_length', 'lash_timing'), none)
    ratio = rt[-1][carried] / ht[-1][carried]
    travel = max(0., .9 - float(ratio.min()), float(ratio.max()) - 1.05)
    v = L[:, tips] - L[:, roots]; length = np.linalg.norm(v, axis=2)
    stretch = np.abs(length / np.maximum(length[0], 1e-300) - 1)
    angle = lambda a, b: np.degrees(np.arccos(np.clip(np.einsum('...k,...k', a, b) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-300), -1, 1)))
    absolute = angle(v, v[0][None])
    pivot, axis = lash.get('pivot', c.get('pivot')), lash.get('axis', c.get('axis'))
    if pivot is not None:                 # the lid's own roll at the host taken out: what the lash turns on the lid
        from .motion_paths import hinge_change
        own = [np.zeros(len(roots))]
        for g in range(1, len(P)):
            lid = hinge_change(P[0][host], P[g][host], pivot, axis)['turn']
            own.append(angle(np.stack([_rotation(axis, -np.degrees(t)) @ w for t, w in zip(lid, v[g])]), v[0]))
        own = np.stack(own)
    else:
        own = absolute
    share = lambda x: x[:, carried] / x[-1][carried]
    timing = np.abs(share(rt) - share(ht)).max(axis=1)
    phases = list(states)
    return {'lash_travel': (travel, {'ratio_min': float(ratio.min()), 'ratio_median': float(np.median(ratio)),
                                     'ratio_max': float(ratio.max()), 'allowed': [.9, 1.05],
                                     'carried_roots': int(carried.sum())}),
            'lash_turn': (float(own.max()), {'relative_to': 'the lid under the root' if pivot is not None else 'rest',
                                             'median_at_end': float(np.median(own[-1])),
                                             'absolute_max': float(absolute.max())}),
            'lash_length': (float(stretch.max()), {'median_at_end': float(np.median(stretch[-1]))}),
            'lash_timing': (float(timing.max()), {'per_phase': dict(zip(phases, map(float, timing)))})}


def lash_pairs(lash_rest, skin_rest, margin, *, carrier=None, on_margin=.0005, points=None):
    """Root and tip pairs for the `lash` declaration, from rest positions alone (saved poses carry no strands).

    Lash points are grouped by the skin point that carries them: `carrier` (one skin vertex per lash point, the hosts
    the carry used, for example `hinge_carry`'s) or, by default, the nearest vertex of the ordered `margin`. A group's
    root is its point nearest the margin line, if within `on_margin`; every other point of the group is a tip paired
    with it and hosted by the carrier, except points within `on_margin` of the root (a lash needs a length to have a
    direction). Groups with no point on the margin are skipped and counted. `points` limits the pairing to some of the
    lash object's points (one eye of a lash object that holds both); indices stay the object's. Found in real use:
    taking a group's nearest point as the root without the `on_margin` test put a root 3.6 mm off the margin (travel
    ratio 1.055), and pairing each lash point with the nearest on-margin lash point crossed groups (116 degree turns)."""
    from scipy.spatial import cKDTree
    L, S = np.asarray(lash_rest, float), np.asarray(skin_rest, float); M = np.asarray(margin, np.int64)
    if len(M) < 2 or M.min() < 0 or M.max() >= len(S):
        raise ValueError('The margin needs at least two vertices of the skin, in order')
    host = (M[cKDTree(S[M]).query(L)[1]] if carrier is None else np.asarray(carrier, np.int64))
    if host.shape != (len(L),) or host.min() < 0 or host.max() >= len(S):
        raise ValueError('The carrier needs one skin vertex per lash point')
    use = np.zeros(len(L), bool); use[np.arange(len(L)) if points is None else np.asarray(points, np.int64)] = True
    off = np.full(len(L), np.inf); off[use] = np.linalg.norm(L[use] - _polyline_nearest(L[use], S[M]), axis=1)
    roots, tips, hosts, skipped, root_off = [], [], [], 0, []
    for h in np.unique(host[use]):
        group = np.flatnonzero(use & (host == h)); r = group[np.argmin(off[group])]
        if off[r] > on_margin:
            skipped += 1
            continue
        others = group[np.linalg.norm(L[group] - L[r], axis=1) >= on_margin]
        roots += [int(r)] * len(others); tips += others.tolist(); hosts += [int(h)] * len(others); root_off.append(off[r])
    return {'roots': roots, 'tips': tips, 'host': hosts, 'groups': int(len(np.unique(host[use]))),
            'groups_with_root': int(len(root_off)), 'groups_without_root': skipped,
            'root_distance_max': float(max(root_off)) if root_off else None}


def _attachments(states, P, entries, base):
    """Attached objects (lash lines, seams, markings) following the skin under them: each point's host is the nearest
    skin vertex at rest unless given. Carried points (host moving at least a twentieth of the largest travel) must keep
    their host's share of travel at every phase; points on still hosts must stay still."""
    from scipy.spatial import cKDTree
    largest = max(float(np.linalg.norm(P - P[0], axis=2).max()), 1e-300)
    rows, timing, still = [], 0., 0.
    for entry in entries:
        A = _stack(states, entry['object'])
        pts = (np.arange(A.shape[1]) if entry.get('points') is None
               else _indices(entry['points'], base, 'points', A.shape[1], 'attachment points'))
        host = (_load(entry['host'], base, 'host').astype(np.int64) if entry.get('host') is not None
                else cKDTree(P[0]).query(A[0][pts])[1])
        if host.shape != pts.shape or host.min() < 0 or host.max() >= P.shape[1]:
            raise ValueError(f"Attachment {entry['object']!r}: one skin vertex per attached point")
        at = np.linalg.norm(A[:, pts] - A[0, pts], axis=2); ht = np.linalg.norm(P[:, host] - P[0, host], axis=2)
        carried = ht[-1] >= .05 * largest; resting = ht.max(axis=0) <= .01 * largest
        t = float(np.abs(at[:, carried] / np.maximum(at[-1, carried], 1e-300)
                         - ht[:, carried] / ht[-1, carried]).max()) if carried.any() else 0.
        r = float(at[:, resting].max() / largest) if resting.any() else 0.
        rows.append({'object': entry['object'], 'points': int(len(pts)), 'carried': int(carried.sum()),
                     'on_still_hosts': int(resting.sum()), 'timing': t, 'still_travel_share': r,
                     'worst_still_point': int(pts[resting][np.argmax(at[:, resting].max(axis=0))]) if r > 0 else None})
        timing, still = max(timing, t), max(still, r)
    return {'attachment_timing': (timing, {'objects': rows}), 'attachment_still': (still, {'objects': rows,
                                                                                          'largest_travel': largest})}


def _polyline_nearest(points, line):
    a, b = line[:-1], line[1:]; ab = b - a; length2 = np.maximum((ab * ab).sum(1), 1e-300)
    t = np.clip(np.einsum('msk,sk->ms', points[:, None, :] - a[None], ab) / length2[None], 0, 1)
    near = a[None] + t[..., None] * ab[None]; dist = np.linalg.norm(points[:, None, :] - near, axis=-1)
    return near[np.arange(len(points)), dist.argmin(axis=1)]


def _combinations(d, P, moving, facing, opening, base):
    """Signed seam gap (positive still open, negative crossed) of each combined closed pose, share of the opening."""
    up = np.asarray(d['closing'].get('up', (0., 0., 1.)), float); up = up / np.linalg.norm(up)
    rows, worst = [], 0.
    for combo in d['combinations']:
        path = Path(combo['poses']); path = path if path.is_absolute() else Path(base or '.') / path
        Q = _stack(load_states(path), d['object'])[-1]
        gap = Q[moving] - _polyline_nearest(Q[moving], Q[facing])
        signed = np.linalg.norm(gap, axis=1) * np.sign(gap @ up) / opening
        rows.append({'name': combo['name'], 'median_share': float(np.median(signed)), 'min_share': float(signed.min()),
                     'max_share': float(signed.max())})
        worst = max(worst, abs(float(np.median(signed))))
    return worst, {'combinations': rows}


def _phase_table(states, what):
    """{phase value: saved key}; two keys with one value (0 and 0.0) make the phase ambiguous and refuse."""
    table = {}
    for key in states:
        value = float(key)
        if not math.isfinite(value):
            raise ValueError(f'{what} phase {key!r} is not a finite number')
        if value in table:
            raise ValueError(f'{what} poses hold phase {value:g} twice ({table[value]!r} and {key!r}); '
                             'phase identity is ambiguous')
        table[value] = key
    return table


def _topology(states, name):
    """The object's one saved triangle array (every saved copy must agree), or a reason it is unusable."""
    found = [np.asarray(objects[name]['tri']) for objects in states.values()
             if name in objects and 'tri' in objects[name]]
    if not found:
        return None, 'no triangles saved for the object'
    first = found[0]
    if any(t.ndim != 2 or t.shape[1:] != (3,) or t.dtype.kind not in 'iu' for t in found):
        return None, 'saved triangles are not integer (m, 3) rows'
    if any(not len(t) for t in found):
        return None, 'saved triangles are empty'
    # Bounds before narrowing: a huge unsigned index must not wrap into a small or negative one.
    if any(int(t.min()) < 0 or int(t.max()) > np.iinfo(np.int64).max for t in found):
        return None, 'saved triangles hold negative or out-of-range indices'
    if any(t.shape != first.shape or not np.array_equal(t, first) for t in found[1:]):
        return None, 'saved triangles differ between phases'
    return first.astype(np.int64), None


def _polygon_witness(obj, count):
    """Validate ordered polygon loops and their own pose's tessellation. No geometric collision claim is made."""
    if any(key not in obj for key in ('co', 'tri', *POLYGON_WITNESS)):
        return None, 'incomplete polygon witness (positions, native triangles and all four witness arrays required)'
    if np.asarray(obj['co']).shape != (count, 3):
        return None, 'polygon witness positions differ in vertex count or shape'
    arrays = []
    for key in POLYGON_WITNESS:
        a = np.asarray(obj[key])
        if (a.ndim != 1 or a.dtype.kind not in 'iu' or not len(a)
                or int(a.min()) < 0 or int(a.max()) > np.iinfo(np.int64).max):
            return None, f'polygon witness {key} needs nonempty nonnegative integer rows'
        arrays.append(a.astype(np.int64))
    loops, starts, lengths, owners = arrays
    if (len(starts) != len(lengths) or np.any(lengths < 3)
            or sum(map(int, lengths)) != len(loops)
            or not np.array_equal(starts, np.r_[0, np.cumsum(lengths[:-1])])):
        return None, 'polygon witness starts/lengths must partition the loops into polygons of at least three vertices'
    if loops.max() >= count:
        return None, 'polygon witness loops index past the vertices'
    tri, why = _topology({'pose': {'object': obj}}, 'object')
    if why or tri.max() >= count:
        return None, 'polygon witness native triangles: ' + (why or 'indices exceed the vertex count')
    if len(owners) != len(tri) or owners.max() >= len(starts):
        return None, 'polygon witness triangle_polygon must name one in-range polygon per native triangle'
    counts = np.bincount(owners, minlength=len(starts))
    if not np.array_equal(counts, lengths - 2):
        return None, 'polygon witness needs exactly n-2 native triangles per n-vertex polygon'
    order = np.argsort(owners, kind='stable'); at = 0
    for polygon, (start, length) in enumerate(zip(starts, lengths)):
        vertices = list(map(int, loops[start:start + length])); vertex_set = set(vertices)
        if len(vertex_set) != length:
            return None, 'polygon witness repeats a vertex within a polygon'
        boundary = {}
        for a, b in zip(vertices, vertices[1:] + vertices[:1]):
            boundary[min(a, b), max(a, b)] = 1 if a < b else -1
        edges = {}
        for t in tri[order[at:at + counts[polygon]]]:
            a, b, c = map(int, t)
            if len({a, b, c}) != 3 or not {a, b, c} <= vertex_set:
                return None, 'native triangle vertices do not belong to their witnessed polygon'
            for x, y in ((a, b), (b, c), (c, a)):
                key = min(x, y), max(x, y)
                uses, direction = edges.get(key, (0, 0))
                edges[key] = uses + 1, direction + (1 if x < y else -1)
        at += counts[polygon]
        if (any(edges.get(key) != (1, direction) for key, direction in boundary.items())
                or any(value != (2, 0) for key, value in edges.items() if key not in boundary)):
            return None, 'native triangles do not tessellate the witnessed directed polygon boundary'
    return (loops, starts, lengths), None


def _correspondence(candidate, baseline, name, count):
    """Stable recorded polygon loops when supplied, otherwise the strict legacy triangle contract.

    Exact indexed connectivity supports a source-corresponding position comparison, not semantic identity after
    arbitrary reindexing. A partial or bad witness never falls back to weaker evidence.
    """
    captures = [(what, phase, objects[name]) for what, states in (('candidate', candidate), ('baseline', baseline))
                for phase, objects in states.items() if name in objects]
    if any(any(key in obj for key in POLYGON_WITNESS) for _, _, obj in captures):
        first = None
        for what, phase, obj in captures:
            witness, why = _polygon_witness(obj, count)
            if why:
                return None, f'{what} {name!r} phase {phase}: {why}'
            if first is not None and any(not np.array_equal(a, b) for a, b in zip(first, witness)):
                return None, f'{what} {name!r} phase {phase}: ordered polygon connectivity differs'
            first = witness
        return {'kind': 'recorded_polygon_loops', 'captures': len(captures), 'polygons': len(first[1]),
                'triangles': 'phase-native; checked against each capture\'s polygon witness, never replaced'}, None
    tri_c, why_c = _topology(candidate, name); tri_b, why_b = _topology(baseline, name)
    if why_c or why_b:
        return None, f'topology of {name!r}: ' + '; '.join(
            f'{w}: {r}' for w, r in (('candidate', why_c), ('baseline', why_b)) if r)
    if tri_c.shape != tri_b.shape or not np.array_equal(tri_c, tri_b):
        return None, f'candidate and baseline triangles of {name!r} differ'
    if tri_c.max() >= count:
        return None, f'saved triangles of {name!r} index past its vertices'
    return {'kind': 'identical_triangles', 'polygons': 'not recorded; differing tessellation requires new captures'}, None


def _preserved(spec, cand, base_states, base, shown=20):
    """Selected vertex regions must match the accepted baseline at every declared phase: (largest distance, detail).

    The distance is Euclidean, between the candidate's and the baseline's position of the same vertex index at the same
    phase, in the saved arrays' units and frame. Missing baseline, phase, object or correspondence evidence leaves the
    check unmeasured (None, with reasons); a malformed selection or an ambiguous phase refuses.
    """
    tol = float(spec['tolerance']); required = [float(g) for g in spec['phases']]
    name_of = lambda g: repr(g)
    detail = {'units': spec['units'], 'units_source': "the caller's declared label; the arrays carry no units",
              'frame': spec.get('frame', "the supplied captures' common frame; no alignment or conversion is applied"),
              'tolerance': tol, 'distance': 'Euclidean, candidate against baseline, same vertex index, same phase',
              'rule': 'pass when every distance <= tolerance', 'required_phases': required, 'reasons': []}
    if base_states is None:
        detail['reasons'].append('no baseline poses: preserved regions are compared only with an accepted baseline')
        return None, detail
    cand_phases, base_phases = _phase_table(cand, 'Candidate'), _phase_table(base_states, 'Baseline')
    coverage = {'required': required,
                'missing_in_candidate': [g for g in required if g not in cand_phases],
                'missing_in_baseline': [g for g in required if g not in base_phases],
                'extra_in_candidate': sorted(set(cand_phases) - set(required)),
                'extra_in_baseline': sorted(set(base_phases) - set(required)),
                'keys': {name_of(g): {'candidate': cand_phases.get(g), 'baseline': base_phases.get(g)} for g in required}}
    detail['phases'] = coverage
    if coverage['missing_in_candidate'] or coverage['missing_in_baseline']:
        detail['reasons'].append('declared phases are missing: nothing is interpolated or taken from another phase')
        return None, detail
    rows, worst, unmeasured = [], None, False
    for i, region in enumerate(spec['regions']):
        label = region.get('label', f"{region['object']}[{i}]"); name = region['object']
        row = {'label': label, 'object': name}
        stacks, problem = [], None
        for states, table, what in ((cand, cand_phases, 'candidate'), (base_states, base_phases, 'baseline')):
            try:
                raw = [np.asarray(states[table[g]][name]['co']) for g in required]
            except KeyError:
                problem = f'the {what} has no positions of {name!r} at a declared phase'; break
            if any(a.dtype.kind not in 'fiu' for a in raw):
                problem = f'the {what} positions of {name!r} are not real numbers'; break
            if len({a.shape for a in raw}) != 1:
                problem = f'the {what} positions of {name!r} differ in vertex count between phases'; break
            stacks.append(np.stack(raw).astype(float))
        if problem is None:
            C, B = stacks
            if C.ndim != 3 or C.shape[2] != 3 or C.shape != B.shape:
                problem = f'candidate and baseline positions of {name!r} do not correspond in shape'
            elif not (np.isfinite(C).all() and np.isfinite(B).all()):
                problem = f'positions of {name!r} are not all finite'
        if problem is None:
            correspondence, problem = _correspondence(cand, base_states, name, C.shape[1])
            if problem is None:
                row['correspondence'] = correspondence
        if problem is not None:
            row.update(status='unmeasured', reason=problem); rows.append(row); unmeasured = True
            detail['reasons'].append(f'{label}: {problem}')
            continue
        idx = _indices(region['vertices'], base, 'vertices', C.shape[1], f'preserved region {label!r}')
        with np.errstate(over='ignore', invalid='ignore'):
            delta = C[:, idx] - B[:, idx]
            # hypot neither underflows a tiny difference to 0 nor overflows a large finite one by squaring
            D = np.hypot(np.hypot(delta[..., 0], delta[..., 1]), delta[..., 2])      # (phases, selected vertices)
        if not np.isfinite(D).all():
            problem = f'the distance of some {name!r} vertices from the baseline is not representable'
            row.update(status='unmeasured', reason=problem); rows.append(row); unmeasured = True
            detail['reasons'].append(f'{label}: {problem}')
            continue
        g, k = np.unravel_index(int(np.argmax(D)), D.shape)
        over = D > tol; per_vertex = D.max(axis=0); order = np.argsort(-per_vertex)
        row.update(status='measured', vertices=int(len(idx)), max_distance=float(D[g, k]),
                   worst_vertex=int(idx[k]), worst_phase=required[g],
                   per_phase={name_of(p): {'max_distance': float(D[j].max()), 'over_tolerance': int(over[j].sum())}
                              for j, p in enumerate(required)},
                   vertices_over_tolerance=int(np.count_nonzero(per_vertex > tol)))
        examples = [{'vertex': int(idx[v]), 'distance': float(per_vertex[v]), 'phase': required[int(D[:, v].argmax())]}
                    for v in order[:shown] if per_vertex[v] > tol]
        row['worst_examples'] = examples
        row['examples_shown'] = f'{len(examples)} of {row["vertices_over_tolerance"]} vertices over the tolerance'
        rows.append(row)
        if worst is None or row['max_distance'] > worst['max_distance']:
            worst = row
    detail['regions'] = rows
    if worst is not None:
        detail['worst'] = {k: worst[k] for k in ('label', 'object', 'worst_vertex', 'worst_phase', 'max_distance')}
    # A measured excess fails even when another region is unmeasured; otherwise any unmeasured region leaves it unknown.
    if worst is not None and worst['max_distance'] > tol:
        detail['incomplete'] = unmeasured
        return worst['max_distance'], detail
    return (None if unmeasured or worst is None else worst['max_distance']), detail


def _compare_sets(measured, before, shown=50):
    """Which reversing vertices and folded triangles are new against the baseline and which it had; private sets out."""
    rev, fold = measured['reversing_vertices'][1], measured['folds'][1]
    now = rev.pop('_reversing'); folded = fold.pop('_folded', None)
    squeeze = measured.get('corner_compression', (None, {}))[1]; squeezed = squeeze.pop('_squeezed', None)
    had_squeezed = before.get('corner_compression', (None, {}))[1].pop('_squeezed', None) if before else None
    if squeezed is not None and had_squeezed is not None:
        common = [g for g in squeezed if g in had_squeezed]
        new = {g: sorted(squeezed[g] - had_squeezed[g]) for g in common}
        gone = {g: sorted(had_squeezed[g] - squeezed[g]) for g in common}
        squeeze.update(new_per_phase=[len(new[g]) for g in common], relieved_per_phase=[len(gone[g]) for g in common],
                       new_squeezed_triangles=[list(t) for g in common for t in new[g]][:shown],
                       relieved_triangles=[list(t) for g in common for t in gone[g]][:shown])
    if not before:
        return
    was = before['reversing_vertices'][1].pop('_reversing')
    rev.update(mover_cut_from='baseline', new_vs_baseline=len(now - was), new_examples=sorted(now - was)[:shown],
               inherited=len(now & was), no_longer_reversing=len(was - now))
    had = before['folds'][1].pop('_folded', None)
    if folded is not None and had is not None:
        common = [g for g in folded if g in had]
        new = {g: sorted(folded[g] - had[g]) for g in common}; gone = {g: sorted(had[g] - folded[g]) for g in common}
        fold.update(new_per_phase={g: len(v) for g, v in new.items()}, unfolded_per_phase={g: len(v) for g, v in gone.items()},
                    new_folded_triangles={g: [list(t) for t in v[:shown]] for g, v in new.items() if v},
                    unfolded_triangles={g: [list(t) for t in v[:shown]] for g, v in gone.items() if v})


def run_checks(declaration, candidate, baseline=None, *, base=None):
    """Measure a candidate's poses against a declaration.

    `candidate` and `baseline` are .npz paths or mappings in the `<phase>::<object>::co|tri` form (see `load_states`).
    `base` resolves relative array paths in the declaration (usually the declaration file's folder). Returns `checks`
    (id, observed, limit, rule, status, detail and, with a baseline, the baseline's observed value), the overall
    `status` (fail if any check fails, unknown if any could not be measured) and the phases measured. Every check gates
    on its own limit, except the defect counts in INCREASE_ONLY, which with a baseline allow `limit` more than it has.
    """
    d = validate_declaration(declaration)
    cand = load_states(candidate)
    base_states = load_states(baseline) if baseline is not None else None
    limits = {**LIMITS, **d.get('limits', {})}
    before = _measure(d, base_states, base) if base_states is not None else {}
    # With a baseline both runs follow the same movers (the baseline's cut), so a vertex whose path did not change
    # is not counted in one run and not the other when the largest chord changes.
    cut = before['reversing_vertices'][1]['mover_cut'] if before else None
    measured = _measure(d, cand, base, base_states, mover_cut=cut)
    _compare_sets(measured, before)
    if d.get('preserved'):
        # Candidate against baseline at declared phases; never measured on the baseline alone.
        measured['preserved_regions'] = _preserved(d['preserved'], cand, base_states, base)
        limits['preserved_regions'] = float(d['preserved']['tolerance'])
    rows = []
    for key in applicable_checks(d):
        observed, detail = measured[key]
        prior = before[key][0] if key in before and key != 'protected_unchanged' else None
        relative = key in INCREASE_ONLY and base_states is not None
        if observed is None or (relative and prior is None):
            status = 'unknown'
        else:
            status = 'pass' if observed - (prior if relative else 0.) <= limits[key] else 'fail'
            if status == 'pass' and detail.get('at_least'):                   # a lower bound within the limit proves nothing
                status = 'unknown'
        row = {'id': key, 'observed': observed, 'limit': limits[key],
               'rule': 'increase over the baseline' if relative else 'maximum', 'status': status, 'detail': detail}
        if key in before and key != 'protected_unchanged':
            row['baseline_observed'] = prior
        rows.append(row)
    status = 'fail' if any(r['status'] == 'fail' for r in rows) else (
        'unknown' if any(r['status'] == 'unknown' for r in rows) else 'pass')
    return {'status': status, 'checks': rows, 'phases': list(cand), 'object': d['object'],
            'objective': 'Standard construction checks of saved poses against a declaration: accepted rest, motion '
                         'kept inside its region, protected objects, preserved regions, clearance, folds, reversals, symmetry, how an '
                         'edge closes and how many blend shapes carry the motion',
            'limits': 'Numbers can reject a candidate, never approve its appearance. Baseline values are shown for '
                      'comparison and do not gate. Existing fold checks track rest triangles; sampled phases do not certify the '
                      'motion between them.'}


# --- Guards for native trials and retention -----------------------------------------------------------------------

def standard_cells(declaration):
    """PreservationPolicy cells for the checks a declaration asks for (one per check, over all phases). The defect counts
    in INCREASE_ONLY are judged against the trial's baseline poses, which the trial must therefore save."""
    d = validate_declaration(declaration)
    limits = {**LIMITS, **d.get('limits', {})}
    if d.get('preserved'):
        limits['preserved_regions'] = float(d['preserved']['tolerance'])
    region = d.get('region_label') or f"{d['object']} region"
    cells = [{'id': key, 'region': region, 'pose': 'all phases', 'metric': key,
              'rule': {'max_increase' if key in INCREASE_ONLY else 'maximum': limits[key]}}
             for key in applicable_checks(d)]
    for cell in cells:
        if cell['id'] == 'preserved_regions':               # its own regions and phases, not the moving region's
            spec = d['preserved']
            names = [f"{r['object']} {r.get('label', r['object'] + f'[{i}]')}" for i, r in enumerate(spec['regions'])]
            cell['region'] = 'preserved: ' + '; '.join(names)
            cell['pose'] = 'declared phases ' + ', '.join(repr(float(g)) for g in spec['phases'])
    return cells


def _array_refs(d, base):
    from .preservation import file_ref
    refs = {}
    specs = [('region', d['region']), ('rest', d.get('rest'))]
    for i, region in enumerate((d.get('preserved') or {}).get('regions', [])):
        specs.append(('preserved:' + region.get('label', f"{region['object']}[{i}]"), region['vertices']))
    for key, spec in specs:
        if isinstance(spec, dict):
            path = Path(spec['path']); refs[key] = file_ref(path if path.is_absolute() else Path(base) / path)
    return refs


def standard_policy(declaration, requirements, *, construction, authority, guide, evidence=()):
    """A PreservationPolicy whose outcome cells and measurement come from a declaration file.

    `declaration` is the JSON declaration's path; `requirements` the path the requirements document is written to (a
    new path per trial family: an existing file must already hold the same document). `construction` is the file that
    builds the candidate, `authority` the project files that govern it, `guide` the registered guide the work is checked
    against. The trial's result `subject` must be a saved file whose folder holds the candidate's poses (`evaluated.npz`)
    and, when present, the baseline's (`source-evaluated.npz`); the declaration's `arrays` renames them.
    """
    from .preservation import PreservationPolicy, file_ref
    from . import construction_diagnostics, motion_paths
    path = Path(declaration).resolve(strict=True)
    d = validate_declaration(json.loads(path.read_text(encoding='utf-8')))
    declaration_ref = file_ref(path)
    constraints = {'declaration': declaration_ref, **_array_refs(d, path.parent)}
    doc = {'schema_version': 1, 'construction': file_ref(construction),
           'authority': [file_ref(a) for a in ([authority] if isinstance(authority, (str, Path)) else authority)],
           'outcomes': [{'id': 'standard_checks',
                         'description': d.get('description') or 'Standard construction checks from the declaration',
                         'baseline': constraints.get('rest', declaration_ref), 'guide': file_ref(guide),
                         'evidence': [declaration_ref, *[file_ref(e) for e in evidence]],
                         'constraints': constraints, 'cells': standard_cells(d)}]}
    target = Path(requirements)
    text = json.dumps(doc, indent=2) + '\n'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != doc:
            raise ValueError(f'{target} already holds different requirements; use a new path for a new declaration')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')
    sources = [file_ref(Path(__file__)), file_ref(Path(construction_diagnostics.__file__)),
               file_ref(Path(motion_paths.__file__))]
    return PreservationPolicy(file_ref(target), {'standard': {'sources': sources, 'coverage': coverage,
                                                              'bind': bind, 'measure': measure}})


def coverage(item, document):
    return {'outcomes': [row['id'] for row in document['outcomes']], 'evidence': [document['construction']]}


def bind(item, constraints):
    inputs = {row['id']: row['constraints'] for row in constraints['outcomes']}
    payload = deepcopy(item.get('payload', {}))
    if 'job' in payload:
        payload['job']['required_inputs'] = inputs
        hashes = payload['job'].setdefault('dependency_hashes', {})
        for group in inputs.values():
            for ref in group.values():
                hashes[ref['path']] = ref['sha256']
    else:
        payload['required_inputs'] = inputs
    return {'payload': payload, 'consumed': inputs}


def measure(item, result, constraints):
    from .preservation import checked, file_ref
    binding = item.get('workbench', {}).get('preservation', {})
    if binding.get('stage') == 'prepare':
        raise ValueError('Standard checks measure saved native poses, not a preparation')
    upstream = binding.get('upstream') or {}
    if item.get('workbench', {}).get('profile') == 'retention' and upstream.get('kind') == 'evaluated_output':
        folder = checked(upstream['subject']).parent        # retention carries the measured candidate forward
    else:
        folder = checked(result['subject']).parent
    rows, evidence = {}, []
    for row in constraints['outcomes']:
        declaration = checked(row['constraints']['declaration'])
        d = validate_declaration(json.loads(declaration.read_text(encoding='utf-8')))
        names = {'candidate': 'evaluated.npz', 'baseline': 'source-evaluated.npz', **d.get('arrays', {})}
        cand, base = folder / names['candidate'], folder / names['baseline']
        report = run_checks(d, cand, base if base.exists() else None, base=declaration.parent)
        evidence += [file_ref(cand)] + ([file_ref(base)] if base.exists() else [])
        cells = {}
        for cell, check in zip(row['cells'], report['checks']):
            if cell['id'] != check['id']:
                raise ValueError('Declared cells and measured checks differ; the declaration changed')
            cells[cell['id']] = {k: cell[k] for k in ('region', 'pose', 'metric')}
            cells[cell['id']].update(observed=check['observed'], baseline=check.get('baseline_observed'))
            if check['id'] == 'preserved_regions':            # where the worst distance is, and what was compared
                cells[cell['id']]['detail'] = deepcopy(check['detail'])
        rows[row['id']] = {'baseline': row['baseline'], 'guide': row['guide'], 'cells': cells}
    return {'subject': result['subject'], 'construction': constraints['construction'], 'kind': 'evaluated_output',
            'evidence': evidence, 'outcomes': rows}
