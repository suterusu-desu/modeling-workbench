"""Register a generated guide mesh onto the accepted character on stationary anatomy.

A generated mesh comes in its generator's frame and scale. Registration brings it into the scene: a scale (by default
the ratio of heights along the up axis), then a rigid trimmed ICP against the accepted neutral using only anatomy that
does not change between the two (moving features such as eye openings, lids and the mouth are excluded). The scale is
kept fixed during ICP: a free scale collapses toward the region that fits best. The report gives the remaining distance
on the stationary anatomy and on the excluded region, which is where the guide is meant to differ.
"""
import numpy as np
from scipy.spatial import cKDTree


def _points(value, what):
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError(f'The {what} needs at least three finite XYZ points')
    return points


def _rigid(source, target):
    """Best rotation and translation taking source to target (least squares, no scale, no reflection)."""
    ms, mt = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((target - mt).T @ (source - ms))
    fix = np.eye(3); fix[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ fix @ vt
    return rotation, mt - rotation @ ms


def _rotation(omega):
    angle = float(np.linalg.norm(omega))
    if angle < 1e-15:
        return np.eye(3)
    k = omega / angle; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _plane_step(source, target, normals):
    """One linearized point-to-plane step: the small rotation and translation minimizing sum(((R p + t - q) . n)^2)."""
    J = np.c_[np.cross(source, normals), normals]; r = np.einsum('ij,ij->i', source - target, normals)
    x, *_ = np.linalg.lstsq(J, -r, rcond=None)
    R = _rotation(x[:3])
    return R, x[3:]


def reference_normals(reference, triangles=None, neighbours=12):
    """Unit normals of the reference vertices: area-weighted from triangles when given, else from a local plane fit."""
    T = _points(reference, 'reference mesh')
    if triangles is not None:
        tri = np.asarray(triangles).astype(np.int64)
        face = np.cross(T[tri[:, 1]] - T[tri[:, 0]], T[tri[:, 2]] - T[tri[:, 0]])
        n = np.zeros_like(T)
        for k in range(3):
            np.add.at(n, tri[:, k], face)
    else:
        _, near = cKDTree(T).query(T, k=min(int(neighbours), len(T)))
        local = T[near] - T[near].mean(axis=1, keepdims=True)
        n = np.linalg.svd(local, full_matrices=False)[2][:, 2]
    length = np.linalg.norm(n, axis=1)
    return np.where(length[:, None] > 0, n / np.maximum(length, 1e-300)[:, None], 0.)


def inside_boxes(points, boxes):
    """True where a point lies in any box {'min': xyz, 'max': xyz, 'mirror': axis (optional)}; a mirrored box also
    covers its reflection through 0 on that axis (for example both eyes from one box)."""
    P = np.asarray(points, float); hit = np.zeros(len(P), bool)
    for box in boxes:
        lo, hi = np.asarray(box['min'], float), np.asarray(box['max'], float)
        if lo.shape != (3,) or hi.shape != (3,) or np.any(lo > hi):
            raise ValueError('Each box needs min <= max in three coordinates')
        hit |= np.all((P >= lo) & (P <= hi), axis=1)
        if box.get('mirror') is not None:
            Q = P.copy(); Q[:, int(box['mirror'])] *= -1
            hit |= np.all((Q >= lo) & (Q <= hi), axis=1)
    return hit


def register_rigid(moving, reference, *, reference_triangles=None, up_axis=2, scale='height', initial='bounds',
                   stationary=None, exclude=(), trim=.9, max_iterations=100, tolerance=1e-10):
    """Bring `moving` (a generated mesh's vertices) onto `reference` (the accepted neutral's vertices, scene frame).

    `scale` is 'height' (the ratio of the two meshes' extents along `up_axis`, which assumes the generated mesh comes
    upright, as image-to-mesh output of an upright head does), a number, or 1. `initial` 'bounds'
    first moves the scaled mesh so its bounding-box centre meets the reference's; 'none' trusts the incoming placement.
    Stationary anatomy is where registration may look: `stationary` is a mask over the reference vertices (a moving
    point counts where its nearest reference vertex is stationary), and `exclude` boxes in the scene frame remove
    moving features. Each ICP step pairs the stationary moving points with their nearest reference points, drops the
    worst (1 - trim) of the pairs and solves the rigid motion: first point to point, then point to plane (distance along
    the reference normal, from `reference_triangles` when given), which converges onto the reference surface instead of
    stopping within a vertex spacing of it. Returns `positions`, `scale`, `rotation`, `translation` (positions =
    rotation @ (scale * moving) + translation), `iterations`, `converged` and distances to the reference surface
    (along its normals at the nearest vertex).
    """
    M, T = _points(moving, 'moving mesh'), _points(reference, 'reference mesh')
    if up_axis not in (0, 1, 2) or not 0 < trim <= 1 or int(max_iterations) < 1:
        raise ValueError('Give an up axis 0-2, a trim in (0, 1] and at least one iteration')
    if scale == 'height':
        extent = np.ptp(M[:, up_axis])
        if extent <= 0:
            raise ValueError('The moving mesh has no height along the up axis')
        s = float(np.ptp(T[:, up_axis]) / extent)
    else:
        s = 1. if scale is None else float(scale)
        if not np.isfinite(s) or s <= 0:
            raise ValueError('Scale must be positive')
    keep = np.ones(len(T), bool) if stationary is None else np.asarray(stationary, bool)
    if keep.shape != (len(T),) or not keep.any():
        raise ValueError('The stationary mask needs one value per reference vertex and at least one stationary vertex')
    P = M * s; rotation, translation = np.eye(3), np.zeros(3)
    if initial == 'bounds':
        shift = (T.min(0) + T.max(0)) / 2 - (P.min(0) + P.max(0)) / 2
        P = P + shift; translation = shift
    elif initial != 'none':
        raise ValueError("initial must be 'bounds' or 'none'")
    tree = cKDTree(T); normals = reference_normals(T, reference_triangles); converged = False; coarse = True
    for iteration in range(1, int(max_iterations) + 1):
        distance, nearest = tree.query(P)
        use = keep[nearest] & ~inside_boxes(P, exclude)
        if use.sum() < 6:
            raise ValueError('Fewer than six stationary points to register on; check the mask and boxes')
        src, dst, nrm = P[use], T[nearest[use]], normals[nearest[use]]
        d = distance[use] if coarse else np.abs(np.einsum('ij,ij->i', src - dst, nrm))
        best = d <= np.quantile(d, trim)
        R, t = _rigid(src[best], dst[best]) if coarse else _plane_step(src[best], dst[best], nrm[best])
        P = P @ R.T + t; rotation, translation = R @ rotation, R @ translation + t
        small = np.abs(R - np.eye(3)).max() < tolerance and np.abs(t).max() < tolerance
        if coarse and (small or np.abs(R - np.eye(3)).max() < 1e-4 and np.abs(t).max() < 1e-4 * np.ptp(T, axis=0).max()):
            coarse = False                                          # close enough: refine onto the surface
        elif not coarse and small:
            converged = True
            break
    _, nearest = tree.query(P)
    distance = np.abs(np.einsum('ij,ij->i', P - T[nearest], normals[nearest]))
    still = keep[nearest] & ~inside_boxes(P, exclude)
    q = lambda values, p: float(np.quantile(values, p)) if len(values) else None
    return {'positions': P, 'scale': s, 'rotation': rotation, 'translation': translation,
            'iterations': iteration, 'converged': converged,
            'public_metrics': {'scale': s, 'iterations': iteration, 'converged': converged,
                               'stationary_points': int(still.sum()), 'excluded_points': int((~still).sum()),
                               'stationary_distance_median': q(distance[still], .5),
                               'stationary_distance_p95': q(distance[still], .95),
                               'excluded_distance_median': q(distance[~still], .5),
                               'excluded_distance_p95': q(distance[~still], .95),
                               'whole_distance_p95': q(distance, .95)},
            'objective': 'Fixed scale, then rigid trimmed alignment on stationary anatomy, point to point then point to '
                         'plane',
            'limits': 'Nearest-point distances, not correspondence; a mesh that differs everywhere (a different face) '
                      'registers to a compromise. Review the overlap before fitting anything to the guide.'}


def apply_registration(points, result):
    """Apply a registration result to other points of the same generated mesh (for example a separated part)."""
    P = np.asarray(points, float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError('Points must be XYZ rows')
    return (P * result['scale']) @ np.asarray(result['rotation']).T + np.asarray(result['translation'])
