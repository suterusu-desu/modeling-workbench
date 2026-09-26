"""Study how a reference avatar constructs a moving feature, before building it for the character.

Extract the avatar's shapes with the `study_extract.py` NativeJob worker (rest positions, every shape key as a world
delta, topology, vertex groups and bones), load them with `load_shapes`, and measure the feature's construction:
which loop is the margin (`order_loop`, `rings`), whether the loops around it are closed quad loops
(`loop_topology`), which simple motion explains it (`motion_models`: one slide, a turn about a named axis, the best
free axis), how far the moving band reaches (`band_profile`) and, for a blink, the numbers the construction checks use
(`blink_report`: moving vertices, opposing-lid travel, corner travel, closed-line depth). For the face: whether a part
turns rigidly and where its hinge is (`rigid_motion`), how much of a jaw turn the skin follows (`jaw_weights`), which
mix of base shapes a viseme is (`viseme_mix`) and rings counted out from the lip seam (`seam_rings`); the face
construction audit (`face_checks`) runs the checks. The avatars are the passing examples: run the same checks on them
and on the character. Measurements, not targets: the character's own references govern its likeness.
"""
import json
from collections import deque
from pathlib import Path

import numpy as np

from .construction_diagnostics import line_depth


def load_shapes(npz, meta=None):
    """An extraction from `study_extract.py`: {'rest', 'edges', 'polygons', 'keys': {name: delta}, 'bones',
    'vertex_groups': {name: weights}, 'object'}."""
    path = Path(npz)
    info = json.loads(Path(meta or path.with_suffix('.json')).read_text(encoding='utf-8'))
    with np.load(path) as data:
        rest = data['basis_world'].astype(float)
        keys = {k['name']: data[f"key{k['index']:04d}"].astype(float) for k in info['keys']
                if f"key{k['index']:04d}" in data.files}
        sizes, loops = data['sizes'], data['loops']
        weights = data['vg_weights'] if 'vg_weights' in data.files else np.zeros((len(rest), 0))
        edges = data['edges'].astype(np.int64)
    return {'object': info.get('object'), 'rest': rest, 'edges': edges,
            'polygons': np.split(loops, np.cumsum(sizes)[:-1]), 'keys': keys, 'bones': info.get('bones', []),
            'vertex_groups': {name: weights[:, i] for i, name in enumerate(info.get('vertex_groups', []))},
            'key_info': {k['name']: k for k in info['keys']}}


def _neighbours(edges, count):
    nb = [set() for _ in range(count)]
    for a, b in np.asarray(edges, int):
        nb[a].add(b); nb[b].add(a)
    return nb


def order_loop(vertices, edges):
    """Order a closed loop of vertex ids by walking its edges (for example a lid margin picked in the avatar)."""
    members = [int(v) for v in vertices]; inside = set(members)
    nb = _neighbours(edges, max(max(members), int(np.max(edges))) + 1)
    order, previous = [members[0]], None
    while True:
        step = [v for v in nb[order[-1]] if v in inside and v != previous and v not in order]
        if not step:
            break
        previous = order[-1]; order.append(min(step))
    if len(order) != len(members) or order[0] not in nb[order[-1]]:
        raise ValueError('The vertices do not form one closed loop along the edges')
    return np.array(order)


def _inside(points, polygon):
    x, y = points[:, 0], points[:, 1]; hit = np.zeros(len(points), bool)
    for i in range(len(polygon)):
        (x1, y1), (x2, y2) = polygon[i], polygon[(i + 1) % len(polygon)]
        crosses = (y1 > y) != (y2 > y)
        at = x1 + (y - y1) * (x2 - x1) / np.where(y2 - y1 == 0, 1e-30, y2 - y1)
        hit ^= crosses & (x < at)
    return hit


def _pocket_side(polygons, margin, count):
    """Vertices on the smaller side of the margin loop (the lid's inner surface and the pocket), found by flooding the
    faces without crossing the margin's edges; None when the margin does not split the faces into two sides."""
    M = [int(v) for v in margin]; cut = {frozenset(e) for e in zip(M, M[1:] + M[:1])}
    faces = [[int(v) for v in f] for f in polygons]; by_edge = {}
    for i, f in enumerate(faces):
        for e in zip(f, f[1:] + f[:1]):
            by_edge.setdefault(frozenset(e), []).append(i)
    sides = []
    for e in cut:
        for start in by_edge.get(e, []):
            if any(start in side for side in sides):
                continue
            side, stack = {start}, [start]
            while stack:
                f = faces[stack.pop()]
                for e2 in zip(f, f[1:] + f[:1]):
                    if frozenset(e2) in cut:
                        continue
                    for g in by_edge[frozenset(e2)]:
                        if g not in side:
                            side.add(g); stack.append(g)
            sides.append(side)
    if len(sides) != 2:
        return None
    small = min(sides, key=len); edge_ids = set(M)
    pocket = np.zeros(count, bool)
    pocket[[v for i in small for v in faces[i] if v not in edge_ids]] = True
    return pocket


def rings(positions, edges, margin, *, up=(0., 0., 1.), across=(1., 0., 0.), outward=12, inward=8, polygons=None):
    """Ring number of every vertex reached from the margin loop: 0 on it, +k steps outward over the outer face, -k steps
    inward (the lid's inner surface and the socket pocket). With `polygons` the two sides are the faces on either side
    of the margin (the smaller one inward); without, the inward side is inside the margin's outline in the view plane,
    which misreads a pocket reaching wider than the opening."""
    P = np.asarray(positions, float); M = np.asarray(margin, int)
    inside = _pocket_side(polygons, M, len(P)) if polygons is not None else None
    if inside is None:
        view = np.c_[P @ np.asarray(across, float), P @ np.asarray(up, float)]
        inside = _inside(view, view[M])
    nb = _neighbours(edges, len(P))
    ring = {int(v): 0 for v in M}; queue = deque(int(v) for v in M)
    while queue:
        v = queue.popleft(); k = ring[v]
        for u in nb[v]:
            if u in ring:
                continue
            if k >= 0 and not inside[u] and k < outward:
                ring[u] = k + 1; queue.append(u)
            elif k <= 0 and inside[u] and -k < inward:
                ring[u] = k - 1; queue.append(u)
    out = np.full(len(P), np.nan)
    out[list(ring)] = list(ring.values())
    return out


def loop_topology(positions, polygons, margin, *, up=(0., 0., 1.), across=(1., 0., 0.), loops=4, inward=8,
                  travel=None):
    """Whether the margin and the next loops outward are closed quad loops of one count, with no poles among them.

    `polygons` are vertex-id lists (as `load_shapes` returns), `margin` the ordered margin loop. For each ring from the
    margin out to `loops - 1`: its vertex count, whether it is one closed loop along the edges, the quad share of the
    faces between it and the next ring and its poles (vertices off the mesh boundary with other than four edges). Also
    the rings found inward (the lid's inner surface and the socket pocket) and, with `travel` per vertex, how many
    vertices move. The avatars studied: margin and the next three loops closed with one count each (28-37), all quads,
    poles only where loops merge into brow, cheek or nose (ring 3 and beyond), 170-190 moving skin vertices per eye.
    A report, not a gate.
    """
    P = np.asarray(positions, float); faces = [[int(v) for v in f] for f in polygons]
    uses = {}
    for f in faces:
        for a, b in zip(f, f[1:] + f[:1]):
            uses[(min(a, b), max(a, b))] = uses.get((min(a, b), max(a, b)), 0) + 1
    edges = np.array(sorted(uses), int).reshape(-1, 2); nb = _neighbours(edges, len(P))
    boundary = {v for e, count in uses.items() if count == 1 for v in e}
    ring = rings(P, edges, margin, up=up, across=across, outward=loops, inward=inward, polygons=faces)
    rows = []
    for k in range(loops):
        members = [int(v) for v in np.flatnonzero(ring == k)]; inside = set(members)
        closed = len(members) >= 3 and all(len(nb[v] & inside) == 2 for v in members)
        if closed:                                           # one loop, not several
            seen, stack = {members[0]}, [members[0]]
            while stack:
                for u in nb[stack.pop()] & inside - seen:
                    seen.add(u); stack.append(u)
            closed = len(seen) == len(members)
        between = [f for f in faces if {ring[v] for v in f} == {k, k + 1}]
        rows.append({'ring': k, 'vertices': len(members), 'closed_loop': bool(closed),
                     'quad_share_to_next': float(np.mean([len(f) == 4 for f in between])) if between else None,
                     'poles': [v for v in members if v not in boundary and len(nb[v]) != 4]})
    inner = []
    for k in range(1, inward + 1):
        count = int(np.count_nonzero(ring == -k))
        if not count:
            break
        inner.append({'ring': -k, 'vertices': count})
    counts = [r['vertices'] for r in rows]
    report = {'loops': rows, 'closed_loops_of_one_count': all(r['closed_loop'] for r in rows) and len(set(counts)) == 1,
              'poles_in_loops': sum(len(r['poles']) for r in rows), 'inner_rings': inner,
              'limits': 'Topology of the rings around the margin; a report to compare with the avatars, not a gate.'}
    if travel is not None:
        t = np.asarray(travel, float)
        report['moved_vertices'] = int(np.count_nonzero(t > 1e-6 * max(float(t.max()), 1e-300)))
    return report


def _turn_residual(X, Y, point, direction):
    a = np.asarray(direction, float); a = a / np.linalg.norm(a); vx, vy = X - point, Y - point
    sx, sy = vx @ a, vy @ a
    rx = np.linalg.norm(vx - np.outer(sx, a), axis=1); ry = np.linalg.norm(vy - np.outer(sy, a), axis=1)
    return np.sqrt((ry - rx) ** 2 + (sy - sx) ** 2)


def motion_models(rest, end, members, *, axes=None, guess=None):
    """How much of a part's motion each simple model cannot explain, as a share of its travel (travel-weighted RMS).

    Every member keeps its own amount of motion. `slide`: one shared direction (a curtain); the residual is the part
    off that direction. Each of `axes` ({name: (point, direction)}, for example a turn about the eye's centre or about
    the line through the corners): the residual is the change of distance from the axis and of position along it. With
    `guess` (point, direction) the best free axis is fitted from it. The avatars studied closed their lids best as a
    slide (7-16 % unexplained), their eyes being flat and recessed; a round eye needs the turn.
    """
    X, Y = np.asarray(rest, float)[members], np.asarray(end, float)[members]; D = Y - X
    travel = np.linalg.norm(D, axis=1); w = travel ** 2
    if w.sum() <= 0:
        raise ValueError('The members do not move')
    rms = float(np.sqrt((w * travel ** 2).sum() / w.sum()))
    share = lambda r: float(np.sqrt((w * r ** 2).sum() / w.sum()) / rms)
    C = (D * w[:, None]).T @ D; u = np.linalg.eigh(C)[1][:, -1]
    out = {'members': int(len(X)), 'rms_travel': rms,
           'slide': {'residual_share': share(np.linalg.norm(D - np.outer(D @ u, u), axis=1)), 'direction': u.tolist()}}
    for name, (point, direction) in (axes or {}).items():
        out[name] = {'residual_share': share(_turn_residual(X, Y, np.asarray(point, float), direction))}
    if guess is not None:
        from scipy.optimize import least_squares
        p0, a0 = np.asarray(guess[0], float), np.asarray(guess[1], float) / np.linalg.norm(guess[1])
        start = np.r_[p0, np.arccos(np.clip(a0[2], -1, 1)), np.arctan2(a0[1], a0[0])]

        def unpack(q):
            return q[:3], np.array([np.sin(q[3]) * np.cos(q[4]), np.sin(q[3]) * np.sin(q[4]), np.cos(q[3])])

        def residual(q):
            p, a = unpack(q); vx, vy = X - p, Y - p; sx, sy = vx @ a, vy @ a
            rx = np.linalg.norm(vx - np.outer(sx, a), axis=1); ry = np.linalg.norm(vy - np.outer(sy, a), axis=1)
            return np.r_[(ry - rx) * np.sqrt(w), (sy - sx) * np.sqrt(w)]
        p, a = unpack(least_squares(residual, start, x_scale='jac').x)
        out['free_axis'] = {'residual_share': share(_turn_residual(X, Y, p, a)), 'point': p.tolist(), 'direction': a.tolist()}
    out['limits'] = 'Residual shares of simple models, not an appearance judgment; a band that also stretches shows residual.'
    return out


def band_profile(rest, end, margin, candidates, *, up=(0., 0., 1.), across=(1., 0., 0.), width=None, column=None,
                 halfwidth=None, bins=12):
    """Travel of the material above a margin as a share of the margin's travel, against rest height above the margin.

    Points of `candidates` within `halfwidth` (default a sixth of `width`) of the `column` position across the view
    (default where the margin travels most) are binned by rest height above the margin (shares of `width`, default the
    margin's extent across). `reach` is the first height where the travel share falls below .1: where the band stops.
    The avatars carried the visible lid as one piece up to .2-.4 of the eye's width and stopped within .53.
    """
    P0, P1 = np.asarray(rest, float), np.asarray(end, float); M = np.asarray(margin, int)
    u, x = np.asarray(up, float) / np.linalg.norm(up), np.asarray(across, float) / np.linalg.norm(across)
    travel = np.linalg.norm(P1 - P0, axis=1)
    width = float(width or np.ptp(P0[M] @ x))
    col = float(column if column is not None else (P0[M] @ x)[np.argmax(travel[M])])
    half = float(halfwidth or width / 6)
    near_margin = M[np.abs(P0[M] @ x - col) <= half]
    if not len(near_margin):
        raise ValueError('No margin point near the column')
    base, reference = float((P0[near_margin] @ u).mean()), float(travel[near_margin].mean())
    C = np.asarray(candidates, int); C = C[np.abs(P0[C] @ x - col) <= half]
    height = (P0[C] @ u - base) / width; share = travel[C] / max(reference, 1e-300)
    keep = height >= 0; height, share = height[keep], share[keep]
    edges = np.linspace(0, max(float(height.max()) if len(height) else 0., 1e-9), bins + 1); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (height >= lo) & (height <= hi)
        if inside.any():
            rows.append({'height_share': float(height[inside].mean()), 'travel_share': float(share[inside].mean()),
                         'points': int(inside.sum())})
    reach = next((r['height_share'] for r in rows if r['travel_share'] < .1), None)
    return {'rows': rows, 'reach_share': reach, 'width': width, 'column': col, 'margin_travel': reference}


def blink_report(rest, end, upper_margin, lower_margin, corners, *, pivot=None, axis=None, up=(0., 0., 1.),
                 across=(1., 0., 0.), band=None):
    """The construction numbers of one blink shape, in the eye's own width.

    `rest`, `end` are the open and closed positions, the margins ordered loops or rows, `corners` the two vertices where
    the lids meet. Reports the moving vertex count, the lower lid's largest travel as a share of the upper's, each
    corner's travel, the closed line's depth below the corner line and the lower lid's rest depth (as `line_depth`),
    slide and hinge residuals of the upper margin (with `pivot` and `axis`) and, with `band` candidates, the band profile.
    """
    P0, P1 = np.asarray(rest, float), np.asarray(end, float)
    up_m, lo_m, (a, b) = np.asarray(upper_margin, int), np.asarray(lower_margin, int), (int(corners[0]), int(corners[1]))
    travel = np.linalg.norm(P1 - P0, axis=1)
    width = float(np.linalg.norm(P0[b] - P0[a]))
    closed = line_depth(P1[up_m], P1[a], P1[b], up=up, across=across)
    lower_rest = line_depth(P0[lo_m], P0[a], P0[b], up=up, across=across)
    axes = {'hinge': (pivot, axis)} if pivot is not None else None
    report = {'width': width, 'moved_vertices': int(np.count_nonzero(travel > 1e-6 * max(travel.max(), 1e-300))),
              'lower_to_upper_travel': float(travel[lo_m].max() / max(travel[up_m].max(), 1e-300)),
              'corner_travel_share': [float(travel[a] / width), float(travel[b] / width)],
              'closed_depth_share': closed['depth_share'], 'lower_rest_depth_share': lower_rest['depth_share'],
              'upper_motion': motion_models(P0, P1, up_m, axes=axes)}
    if band is not None:
        report['band'] = band_profile(P0, P1, up_m, band, up=up, across=across, width=width)
    return report


def rigid_motion(rest, end):
    """The rigid turn that best carries a part from `rest` to `end` (least squares): `rotation`, `translation`,
    `angle_degrees`, `axis`, a point on the hinge axis (`hinge_point`, None under half a degree) and the RMS distance of
    the part from the rigid result (`residual_rms`). The avatars' lower teeth turned 7-9.4 degrees rigidly (residual
    .0005-.006 of the mouth width) about an axis 2.4 mouth widths behind the lips."""
    X, Y = np.asarray(rest, float), np.asarray(end, float)
    cx, cy = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - cx).T @ (Y - cy))
    R = Vt.T @ np.diag([1., 1., np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    t = cy - R @ cx
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); axis = w / max(np.linalg.norm(w), 1e-300)
    point = None
    if angle > .5:
        point = np.linalg.lstsq(np.eye(3) - R, t - axis * (axis @ t), rcond=None)[0]
    return {'rotation': R, 'translation': t, 'angle_degrees': angle, 'axis': axis, 'hinge_point': point,
            'residual_rms': float(np.sqrt(((X @ R.T + t - Y) ** 2).sum(1).mean()))}


def jaw_weights(positions, delta, rotation, translation):
    """How much of a jaw turn each vertex follows: w = d.p / |p|^2 against the turn's displacement p = R x + t - x, and
    the share of its motion off that direction (|d - w p| / |d|). The avatars' skin below the lip followed the jaw by a
    weight falling from the chin (median .10-.23, top decile over .58), with a median off-share of .11-.23."""
    P, D = np.asarray(positions, float), np.asarray(delta, float)
    p = P @ np.asarray(rotation, float).T + np.asarray(translation, float) - P
    w = (D * p).sum(1) / np.maximum((p * p).sum(1), 1e-300)
    off = np.linalg.norm(D - w[:, None] * p, axis=1) / np.maximum(np.linalg.norm(D, axis=1), 1e-300)
    return w, off


def viseme_mix(basis, target):
    """The non-negative mix of base shapes (`basis`: {name: delta}) nearest a viseme's delta, and the relative residual
    |d - B w| / |d|. Most of the avatars' visemes were exact mixes of five vowel shapes."""
    from scipy.optimize import nnls
    names = list(basis); B = np.stack([np.asarray(basis[k], float).ravel() for k in names], 1)
    d = np.asarray(target, float).ravel(); norm = float(np.linalg.norm(d))
    if norm == 0:
        return {'weights': {}, 'relative_residual': 0., 'empty': True}
    w, _ = nnls(B, d)
    return {'weights': {k: float(x) for k, x in zip(names, w) if x > 1e-6},
            'relative_residual': float(np.linalg.norm(d - B @ w) / norm), 'empty': False}


def seam_rings(edges, seam, count, *, limit=64):
    """Ring number of every vertex by edge steps from the seam vertices (0 on the seam), up to `limit` rings; others
    NaN. Used to report how a lip shape's motion falls off ring by ring from the seam outward."""
    nb = _neighbours(edges, count); ring = np.full(count, np.nan)
    front = [int(v) for v in seam]; ring[front] = 0
    for k in range(1, limit + 1):
        nxt = {u for v in front for u in nb[v] if np.isnan(ring[u])}
        if not nxt:
            break
        ring[list(nxt)] = k; front = list(nxt)
    return ring
