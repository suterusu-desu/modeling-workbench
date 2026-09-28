"""Metric-aware finite elements on supplied recorded geometry: planar charts and 3D triangle surfaces."""
import numpy as np


def planar_fem_metric(chart, triangles, *, units, frame, relative_area_tolerance):
    """Assemble positive stiffness and lumped area mass; never select fit targets.

    Both chart axes use the same declared length unit in a Euclidean frame.
    Return sparse triplets as ordinary arrays for the preparation/NPZ route.
    Affine reproduction is checked on interior vertices, not boundary flux rows.
    """
    from scipy.sparse import coo_matrix

    points, tri = np.asarray(chart), np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'fiu'
            or not 3 <= len(points) <= 100000 or not np.isfinite(points).all()
            or tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu'
            or not 1 <= len(tri) <= 200000 or np.any(tri < 0) or np.any(tri >= len(points))):
        raise ValueError('Finite 2D chart and bounded valid integer triangles required')
    if (not isinstance(units, str) or not units.strip()
            or not isinstance(frame, str) or not frame.strip()
            or type(relative_area_tolerance) not in (int, float)
            or not np.isfinite(relative_area_tolerance) or not 0 < relative_area_tolerance < 1):
        raise ValueError('Explicit chart units, frame and dimensionless relative-area tolerance required')
    points, tri = points.astype(float), tri.astype(np.int64)
    if len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri):
        raise ValueError('Duplicate triangles do not define a planar metric')
    p = points[tri]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        area2 = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        longest2 = np.maximum.reduce([np.sum(a*a, axis=1), np.sum(b*b, axis=1), np.sum((b-a)**2, axis=1)])
        relative_area = np.abs(area2) / longest2
    if (not np.isfinite(relative_area).all() or not np.isfinite(area2).all()
            or np.any(relative_area <= relative_area_tolerance)):
        raise ValueError('Degenerate or ill-conditioned chart triangle under the declared area tolerance')

    # Exact edge identities distinguish the boundary from interior weak loads.
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold chart edge; qualify a manifold planar patch')
    sides = np.tile(np.sign(area2), 3) * np.where(edges[:, 0] < edges[:, 1], 1, -1)
    side_sum = np.bincount(inverse, weights=sides, minlength=len(unique))
    if np.any((counts == 2) & (side_sum != 0)):
        raise ValueError('Adjacent chart triangles overlap across a shared edge')
    boundary_edges = unique[counts == 1]
    boundary = np.unique(boundary_edges)
    interior = np.setdiff1d(np.arange(len(points)), boundary)

    gradients = np.stack([
        np.stack([p[:, 1, 1]-p[:, 2, 1], p[:, 2, 0]-p[:, 1, 0]], axis=1),
        np.stack([p[:, 2, 1]-p[:, 0, 1], p[:, 0, 0]-p[:, 2, 0]], axis=1),
        np.stack([p[:, 0, 1]-p[:, 1, 1], p[:, 1, 0]-p[:, 0, 0]], axis=1)
    ], axis=1) / area2[:, None, None]
    area = np.abs(area2) / 2
    element = area[:, None, None] * np.einsum('tik,tjk->tij', gradients, gradients)
    stiffness = coo_matrix((element.ravel(),
        (np.repeat(tri, 3, axis=1).ravel(), np.tile(tri, (1, 3)).ravel())),
        shape=(len(points), len(points))).tocsr()
    stiffness.eliminate_zeros()
    mass = np.bincount(tri.ravel(), weights=np.repeat(area / 3, 3), minlength=len(points))
    if (not np.isfinite(stiffness.data).all() or not np.isfinite(mass).all()
            or np.any(mass <= 0) or not np.isfinite(area.sum())):
        raise ValueError('Finite stiffness and positive mass required for every supplied vertex')
    # Centering avoids introducing a large arbitrary translation into the test.
    affine_load = stiffness @ (points - points[0])
    constant_load = stiffness @ np.ones(len(points))
    if not np.isfinite(affine_load).all() or not np.isfinite(constant_load).all():
        raise ValueError('Metric diagnostic overflow')
    sparse = stiffness.tocoo()
    metrics = {'vertices': len(points), 'triangles': len(tri),
        'chart_area': float(area.sum()), 'length_unit': units,
        'interior_vertices': len(interior), 'boundary_vertices': len(boundary),
        'minimum_relative_area': float(relative_area.min()),
        'constant_stiffness_defect': float(np.max(np.abs(constant_load))),
        'affine_interior_defect': float(np.max(np.abs(affine_load[interior]))) if len(interior) else None,
        'orientation_counts': [int(np.count_nonzero(area2 > 0)), int(np.count_nonzero(area2 < 0))],
        'positive_offdiagonal_entries': int(np.count_nonzero((sparse.row != sparse.col) & (sparse.data > 0)))}
    return {'stiffness_rows': sparse.row.copy(), 'stiffness_columns': sparse.col.copy(),
        'stiffness_values': sparse.data.copy(), 'stiffness_shape': list(stiffness.shape),
        'lumped_mass': mass, 'triangle_areas': area, 'boundary_edges': boundary_edges,
        'boundary_vertices': boundary, 'interior_vertices': interior,
        'interior_affine_load': affine_load[interior], 'public_metrics': metrics,
        'metric': {'type': 'euclidean_planar', 'frame': frame, 'length_unit': units,
            'stiffness_units': 'dimensionless', 'mass_units': 'length^2',
            'mass_inverse_stiffness_units': 'length^-2',
            'squared_laplacian_matrix_units': 'length^-2',
            'affine_interior_defect_units': 'length', 'constant_stiffness_defect_units': 'dimensionless',
            'relative_area_tolerance': float(relative_area_tolerance),
            'sign': 'Positive semidefinite stiffness: integrated gradient dot gradient'},
        'limits': 'Supplied Euclidean 2D chart only; no intrinsic 3D metric, anatomical qualification, global overlap test or target admission. Boundary stiffness rows include flux. Interior affine reproduction and edge orientation do not approve appearance or ensure nonnegative interpolation weights.'}


def surface_fem_metric(positions, triangles, *, units, frame, relative_area_tolerance):
    """Assemble the intrinsic P1 stiffness and lumped area mass of a 3D triangle surface.

    Each element is the planar element in its own triangle plane, so the operator
    measures lengths along the recorded surface instead of in a projection; a planar
    chart of a steep or curved region distorts that metric. Same outputs as
    planar_fem_metric. Linear functions of ambient coordinates are harmonic only on
    a flat patch: their interior load is the discrete mean-curvature normal, reported
    as curvature, not as an error.
    """
    from scipy.sparse import coo_matrix

    points, tri = np.asarray(positions), np.asarray(triangles)
    if (points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'fiu'
            or not 3 <= len(points) <= 100000 or not np.isfinite(points).all()
            or tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu'
            or not 1 <= len(tri) <= 200000 or np.any(tri < 0) or np.any(tri >= len(points))):
        raise ValueError('Finite 3D positions and bounded valid integer triangles required')
    if (not isinstance(units, str) or not units.strip()
            or not isinstance(frame, str) or not frame.strip()
            or type(relative_area_tolerance) not in (int, float)
            or not np.isfinite(relative_area_tolerance) or not 0 < relative_area_tolerance < 1):
        raise ValueError('Explicit position units, frame and dimensionless relative-area tolerance required')
    points, tri = points.astype(float), tri.astype(np.int64)
    if len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri):
        raise ValueError('Duplicate triangles do not define a surface metric')
    p = points[tri]
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
        normal = np.cross(a, b); area2 = np.linalg.norm(normal, axis=1)
        longest2 = np.maximum.reduce([np.sum(a*a, axis=1), np.sum(b*b, axis=1), np.sum((b-a)**2, axis=1)])
        relative_area = area2 / longest2
    if (not np.isfinite(relative_area).all() or not np.isfinite(area2).all()
            or np.any(relative_area <= relative_area_tolerance)):
        raise ValueError('Degenerate or ill-conditioned surface triangle under the declared area tolerance')

    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold surface edge; qualify a manifold patch')
    # Winding does not change intrinsic stiffness; mixed winding is reported, not refused.
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    inconsistent = int(np.count_nonzero((counts == 2) & (np.bincount(inverse, weights=direction, minlength=len(unique)) != 0)))
    boundary_edges = unique[counts == 1]
    boundary = np.unique(boundary_edges)
    interior = np.setdiff1d(np.arange(len(points)), boundary)

    # Local orthonormal element frames: the planar element in each triangle's own plane.
    u = a / np.linalg.norm(a, axis=1, keepdims=True)
    v = np.cross(normal / area2[:, None], u)
    local = np.stack([np.zeros((len(tri), 2)),
                      np.stack([np.sum(a*u, axis=1), np.zeros(len(tri))], axis=1),
                      np.stack([np.sum(b*u, axis=1), np.sum(b*v, axis=1)], axis=1)], axis=1)
    gradients = np.stack([
        np.stack([local[:, 1, 1]-local[:, 2, 1], local[:, 2, 0]-local[:, 1, 0]], axis=1),
        np.stack([local[:, 2, 1]-local[:, 0, 1], local[:, 0, 0]-local[:, 2, 0]], axis=1),
        np.stack([local[:, 0, 1]-local[:, 1, 1], local[:, 1, 0]-local[:, 0, 0]], axis=1)
    ], axis=1) / area2[:, None, None]
    area = area2 / 2
    element = area[:, None, None] * np.einsum('tik,tjk->tij', gradients, gradients)
    stiffness = coo_matrix((element.ravel(),
        (np.repeat(tri, 3, axis=1).ravel(), np.tile(tri, (1, 3)).ravel())),
        shape=(len(points), len(points))).tocsr()
    stiffness.eliminate_zeros()
    mass = np.bincount(tri.ravel(), weights=np.repeat(area / 3, 3), minlength=len(points))
    if (not np.isfinite(stiffness.data).all() or not np.isfinite(mass).all()
            or np.any(mass <= 0) or not np.isfinite(area.sum())):
        raise ValueError('Finite stiffness and positive mass required for every supplied vertex')
    curvature_load = stiffness @ (points - points[0])
    constant_load = stiffness @ np.ones(len(points))
    if not np.isfinite(curvature_load).all() or not np.isfinite(constant_load).all():
        raise ValueError('Metric diagnostic overflow')
    sparse = stiffness.tocoo()
    # K x = 2 H n A at interior vertices (lumped area A): report the discrete mean curvature |H|.
    mean_curvature = (np.linalg.norm(curvature_load[interior], axis=1) / (2 * mass[interior])) if len(interior) else None
    metrics = {'vertices': len(points), 'triangles': len(tri),
        'surface_area': float(area.sum()), 'length_unit': units,
        'interior_vertices': len(interior), 'boundary_vertices': len(boundary),
        'minimum_relative_area': float(relative_area.min()),
        'constant_stiffness_defect': float(np.max(np.abs(constant_load))),
        'interior_mean_curvature_max': float(mean_curvature.max()) if len(interior) else None,
        'inconsistently_wound_edges': inconsistent,
        'positive_offdiagonal_entries': int(np.count_nonzero((sparse.row != sparse.col) & (sparse.data > 0)))}
    return {'stiffness_rows': sparse.row.copy(), 'stiffness_columns': sparse.col.copy(),
        'stiffness_values': sparse.data.copy(), 'stiffness_shape': list(stiffness.shape),
        'lumped_mass': mass, 'triangle_areas': area, 'boundary_edges': boundary_edges,
        'boundary_vertices': boundary, 'interior_vertices': interior,
        'interior_curvature_load': curvature_load[interior], 'public_metrics': metrics,
        'metric': {'type': 'intrinsic_surface_p1', 'frame': frame, 'length_unit': units,
            'stiffness_units': 'dimensionless', 'mass_units': 'length^2',
            'mass_inverse_stiffness_units': 'length^-2',
            'squared_laplacian_matrix_units': 'length^-2',
            'interior_mean_curvature_units': 'length^-1', 'constant_stiffness_defect_units': 'dimensionless',
            'relative_area_tolerance': float(relative_area_tolerance),
            'sign': 'Positive semidefinite stiffness: integrated surface gradient dot gradient'},
        'limits': 'Recorded piecewise-linear surface only; no smooth-surface, anatomical or correspondence qualification. Boundary stiffness rows include flux. Ambient-coordinate loads measure discrete curvature, not error. Obtuse elements can give signed weights; reproduction checks and orientation do not approve appearance.'}


def relax_displacement(reference, deformed, triangles, held, *, units, frame, relative_area_tolerance,
                       compressed_below=.5):
    """Replace the free vertices' displacement by the minimum-bending interpolation of the held set.

    The displacement deformed - reference of every free vertex is replaced by the field that
    minimizes the squared intrinsic surface Laplacian (reference metric, interior rows) with every
    held vertex fixed. Crowded material spreads into the smoothest field the held boundary allows.
    No guide is fitted: moving material across a curved surface changes its depth, so follow this
    with a guide or retained-surface depth fit, and hold material the pose leaves static or motion
    spreads into it. Free vertices must be interior to the supplied patch; hold its boundary.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    from .construction_diagnostics import compare_stretch

    rest, posed, tri = np.asarray(reference), np.asarray(deformed), np.asarray(triangles)
    mask = np.asarray(held)
    if (rest.shape != posed.shape or rest.ndim != 2 or rest.shape[1:] != (3,) or not np.isfinite(posed).all()
            or mask.shape != (len(rest),) or mask.dtype.kind not in 'biu'):
        raise ValueError('Corresponding finite reference/deformed positions and one held flag per vertex required')
    if not (np.isfinite(compressed_below) and 0 < compressed_below < 1):
        raise ValueError('Require 0 < compressed_below < 1')
    if tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri) or tri.min() < 0 or tri.max() >= len(rest):
        raise ValueError('Patch triangles must index the supplied vertices')
    # The patch may be a region of a larger mesh: work on its own vertices, report in the input indexing.
    n = len(rest); tri = tri.astype(np.int64); used = np.unique(tri)
    local = np.full(n, -1, dtype=np.int64); local[used] = np.arange(len(used))
    metric = surface_fem_metric(rest[used], local[tri], units=units, frame=frame, relative_area_tolerance=relative_area_tolerance)
    held_local = mask.astype(bool)[used]
    free = np.flatnonzero(~held_local); fixed = np.flatnonzero(held_local)
    if not len(free) or not len(fixed):
        raise ValueError('Both free and held vertices are required in the patch')
    boundary = np.zeros(len(used), bool); boundary[np.asarray(metric['boundary_vertices'], dtype=np.int64)] = True
    if boundary[free].any():
        raise ValueError('Free vertices must be interior to the patch; hold its boundary (two rings keep slope continuity)')
    stiffness = coo_matrix((metric['stiffness_values'], (metric['stiffness_rows'], metric['stiffness_columns'])),
                           shape=metric['stiffness_shape']).tocsr()
    rows = np.asarray(metric['interior_vertices'], dtype=np.int64)
    bending = (stiffness[rows, :].T @ diags(1 / metric['lumped_mass'][rows]) @ stiffness[rows, :]).tocsr()
    displacement = (posed[used] - rest[used]).astype(float); relaxed_displacement = displacement.copy()
    system = bending[free][:, free].tocsc(); coupling = bending[free][:, fixed]
    for k in range(3):
        relaxed_displacement[free, k] = spsolve(system, -(coupling @ displacement[fixed, k]))
    if not np.isfinite(relaxed_displacement).all():
        raise ValueError('Relaxation solve did not produce finite displacements; qualify the held set')
    # Only free vertices receive recomputed positions; held and unused ones keep their exact bytes.
    relaxed = posed.astype(float).copy(); relaxed[used[free]] = rest[used[free]] + relaxed_displacement[free]
    delta = relaxed - posed
    before = compare_stretch(rest, posed, tri, compressed_below=compressed_below, limit=1)['all']
    after = compare_stretch(rest, relaxed, tri, compressed_below=compressed_below, limit=1)['all']
    free, fixed = used[free], used[fixed]
    metrics = {'free_vertices': int(len(free)), 'held_vertices': int(len(fixed)),
               'max_displacement_change': float(np.linalg.norm(delta, axis=1).max()),
               'smallest_stretch_q01_before': before['smallest_stretch_q01'], 'smallest_stretch_q01_after': after['smallest_stretch_q01'],
               'smallest_stretch_q05_before': before['smallest_stretch_q05'], 'smallest_stretch_q05_after': after['smallest_stretch_q05'],
               'compressed_before': before['compressed'], 'compressed_after': after['compressed'],
               'normal_reversals_before': before['normal_reversals'], 'normal_reversals_after': after['normal_reversals'],
               'local_reversals_before': before['local_reversals'], 'local_reversals_after': after['local_reversals']}
    return {'delta': delta, 'relaxed': relaxed, 'free_vertices': free, 'held_vertices': fixed, 'public_metrics': metrics,
            'objective': 'Squared intrinsic Laplacian of the displacement from reference on interior rows; held vertices exact',
            'limits': 'Construction repair of material distribution only: no guide, depth, anatomical or appearance qualification. '
                      'Depth changes wherever material slides over curvature; the held set is an explicit owner choice.'}


def _arap_setup(rest, start, tri, mask, units, frame, relative_area_tolerance):
    """The as-rigid-as-possible core shared by rigid_deform and conform_to_surface: the patch's own vertices, free and
    held sets, clamped intrinsic cotangent edge weights of the reference and their Laplacian."""
    from scipy.sparse import coo_matrix

    n = len(rest); tri = tri.astype(np.int64); used = np.unique(tri)
    local = np.full(n, -1, dtype=np.int64); local[used] = np.arange(len(used))
    metric = surface_fem_metric(rest[used], local[tri], units=units, frame=frame, relative_area_tolerance=relative_area_tolerance)
    held_local = mask.astype(bool)[used]
    free = np.flatnonzero(~held_local); fixed = np.flatnonzero(held_local)
    if not len(free) or not len(fixed):
        raise ValueError('Both free and held vertices are required in the patch')
    boundary = np.zeros(len(used), bool); boundary[np.asarray(metric['boundary_vertices'], dtype=np.int64)] = True
    if boundary[free].any():
        raise ValueError('Free vertices must be interior to the patch; hold its boundary')
    # Cotangent edge weights are the negated off-diagonal P1 stiffness entries of the reference surface.
    rows, cols, values = (np.asarray(metric[k]) for k in ('stiffness_rows', 'stiffness_columns', 'stiffness_values'))
    off = rows != cols
    i, j, w = rows[off].astype(np.int64), cols[off].astype(np.int64), -values[off].astype(float)
    positive = w[w > 0]
    floor = 1e-3 * float(positive.mean()) if len(positive) else 1e-12
    clamped = int((w < floor).sum()); w = np.maximum(w, floor)
    m = len(used); p = rest[used].astype(float); x = start[used].astype(float).copy()
    degree = np.bincount(i, weights=w, minlength=m)
    laplacian = (coo_matrix((-w, (i, j)), shape=(m, m)) + coo_matrix((degree, (np.arange(m), np.arange(m))), shape=(m, m))).tocsr()
    return tri, used, free, fixed, i, j, w, clamped, m, p, x, degree, laplacian


def _arap_rotations(i, j, w, edges, y, m):
    """Per-vertex best rotations of the reference edges onto the current ones (reflections removed)."""
    covariance = np.zeros((m, 3, 3))
    np.add.at(covariance, i, w[:, None, None] * edges[:, :, None] * (y[i] - y[j])[:, None, :])
    u, _, vt = np.linalg.svd(covariance)
    r = np.einsum('nji,nkj->nik', vt, u)
    flip = np.linalg.det(r) < 0
    if flip.any():
        u = u.copy(); u[flip, :, -1] *= -1; r[flip] = np.einsum('nji,nkj->nik', vt[flip], u[flip])
    return r


def _arap_right(i, j, w, edges, r, m):
    right = np.zeros((m, 3)); np.add.at(right, i, .5 * w[:, None] * np.einsum('nab,nb->na', r[i] + r[j], edges))
    return right


def rigid_deform(reference, initial, triangles, held, *, units, frame, relative_area_tolerance,
                 iterations=50, tolerance=1e-9, compressed_below=.5, targets=None, target_weights=None,
                 interval_axis=None, lower=None, upper=None, interval_weight=1e3, project_active=True):
    """As-rigid-as-possible deformation of a patch: free vertices keep the reference shape up to local rotations.

    Held vertices take their `initial` positions exactly (moved handles and preserved material alike). Every free
    vertex minimizes the as-rigid-as-possible energy (Sorkine and Alexa 2007) with the intrinsic cotangent weights
    of the reference surface, alternating per-vertex best rotations with a sparse global solve, starting from its
    `initial` position. Unlike the linear interpolation of relax_displacement, material may rotate, so large handle
    moves bend the patch instead of folding or creasing it. The reference must itself be fold-free: preserving a
    folded shape preserves its folds. Negative cotangent weights (obtuse elements) are clamped to a small positive
    value and counted. Free vertices must be interior to the supplied patch; hold its boundary.

    Optional soft targets add target_weights[i] * degree_i * |x_i - targets[i]|^2 for free vertices, where degree_i is
    the vertex's summed cotangent weight, so a weight of 1 pulls about as strongly as the local shape term. Use them to
    keep an achieved shape away from a correction without the seam a hard held boundary makes: zero weight near the
    material being replaced, rising with distance from it.

    Optional intervals keep each free vertex's coordinate along `interval_axis` (0, 1 or 2) inside
    [lower[i], upper[i]] (NaN or an infinite value leaves that side open), for example a guide's depth band. The
    vertices outside their interval form an active set pulled onto the violated bound by a penalty of
    interval_weight * degree_i in that coordinate only, re-solved with the rotations until the set stops changing; a
    vertex the solve keeps inside leaves the set, and the other two coordinates are never pulled. With
    project_active (the default) the active vertices finish exactly on their bound (the penalty alone leaves them
    outside by about force / weight). A small interval_weight without projection makes the band a soft pull that the
    shape term smooths: the material follows the band's volume instead of tracing a steep edge of it.
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import factorized
    from .construction_diagnostics import compare_stretch

    rest, start, tri = np.asarray(reference), np.asarray(initial), np.asarray(triangles)
    mask = np.asarray(held)
    if (rest.shape != start.shape or rest.ndim != 2 or rest.shape[1:] != (3,) or not np.isfinite(start).all()
            or not np.isfinite(rest).all() or mask.shape != (len(rest),) or mask.dtype.kind not in 'biu'):
        raise ValueError('Corresponding finite reference/initial positions and one held flag per vertex required')
    if not (int(iterations) >= 1 and np.isfinite(tolerance) and tolerance >= 0 and 0 < compressed_below < 1):
        raise ValueError('Require iterations >= 1, a finite nonnegative tolerance and 0 < compressed_below < 1')
    if tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri) or tri.min() < 0 or tri.max() >= len(rest):
        raise ValueError('Patch triangles must index the supplied vertices')
    if (targets is None) != (target_weights is None):
        raise ValueError('Soft targets need both targets and target_weights')
    if targets is not None:
        targets, target_weights = np.asarray(targets, float), np.asarray(target_weights, float)
        if (targets.shape != rest.shape or target_weights.shape != (len(rest),) or not np.isfinite(targets).all()
                or not np.isfinite(target_weights).all() or (target_weights < 0).any()):
            raise ValueError('Soft targets need one finite position and one finite nonnegative weight per vertex')
    axis = None
    if interval_axis is not None or lower is not None or upper is not None:
        if interval_axis is None or lower is None or upper is None or isinstance(interval_axis, bool) or interval_axis not in (0, 1, 2):
            raise ValueError('Intervals need interval_axis 0, 1 or 2 and one lower and one upper bound per vertex')
        axis = int(interval_axis)
        low, high = np.asarray(lower, float), np.asarray(upper, float)
        if low.shape != (len(rest),) or high.shape != (len(rest),):
            raise ValueError('Intervals need interval_axis 0, 1 or 2 and one lower and one upper bound per vertex')
        low, high = np.where(np.isnan(low), -np.inf, low), np.where(np.isnan(high), np.inf, high)
        if (low > high).any() or (low == np.inf).any() or (high == -np.inf).any():
            raise ValueError('Each interval needs lower <= upper')
        if not (np.isfinite(interval_weight) and interval_weight > 0):
            raise ValueError('A finite positive interval_weight is required')
        if not isinstance(project_active, bool):
            raise ValueError('project_active must be True or False')
    tri, used, free, fixed, i, j, w, clamped, m, p, x, degree, laplacian = _arap_setup(
        rest, start, tri, mask, units, frame, relative_area_tolerance)
    soft = np.zeros(m) if targets is None else target_weights[used] * degree
    goal = np.zeros((m, 3)) if targets is None else targets[used]
    solve = factorized((laplacian + diags(soft))[free][:, free].tocsc()); coupling = laplacian[free][:, fixed]
    edges = p[i] - p[j]
    energies = []
    is_free = np.zeros(m, bool); is_free[free] = True
    if axis is not None:
        low_l, high_l = low[used], high[used]
        bounded = is_free & (np.isfinite(low_l) | np.isfinite(high_l))
        penalty = float(interval_weight) * degree
    active = np.zeros(m, bool); bound = np.zeros(m); solve_axis, active_solved = None, None

    def outside(v):
        return bounded & ((v < low_l) | (v > high_l))

    def violation(v):
        return np.where(bounded, np.maximum(np.maximum(low_l - v, v - high_l), 0.), 0.)

    def rotations(y):
        return _arap_rotations(i, j, w, edges, y, m)

    def energy(y, r):
        value = float((w * np.sum(((y[i] - y[j]) - np.einsum('nab,nb->na', r[i], edges)) ** 2, axis=1)).sum()
                      + (soft[free] * np.sum((y[free] - goal[free]) ** 2, axis=1)).sum())
        return value + (float((penalty[active] * (y[active, axis] - bound[active]) ** 2).sum()) if active.any() else 0.)

    if axis is not None:
        before_violation = violation(x[:, axis])
        active = outside(x[:, axis]); bound = np.where(x[:, axis] < low_l, low_l, np.where(active, high_l, 0.))
    converged, set_changes = False, 0
    for _ in range(int(iterations)):
        r = rotations(x)
        right = _arap_right(i, j, w, edges, r, m)
        right += soft[:, None] * goal
        for k in range(3):
            if k == axis and active.any():
                pulled = np.where(active, penalty, 0.)
                if active_solved is None or not np.array_equal(active_solved, active):
                    solve_axis = factorized((laplacian + diags(soft + pulled))[free][:, free].tocsc()); active_solved = active.copy()
                x[free, k] = solve_axis(right[free, k] + (pulled * bound)[free] - coupling @ x[fixed, k])
            else:
                x[free, k] = solve(right[free, k] - coupling @ x[fixed, k])
        energies.append(energy(x, rotations(x)))
        changed = False
        if axis is not None:
            # Vertices the penalty holds stay slightly outside; those the solve keeps inside leave the set.
            now = outside(x[:, axis]); changed = not np.array_equal(now, active)
            set_changes += int(changed); active = now
            bound = np.where(x[:, axis] < low_l, low_l, np.where(active, high_l, 0.))
        if (not changed and len(energies) > 1
                and abs(energies[-2] - energies[-1]) <= tolerance * max(energies[-2], 1e-300)):
            converged = True; break
    if not np.isfinite(x).all():
        raise ValueError('Rigid deformation did not produce finite positions; qualify the held set')
    if axis is not None:
        penalty_residual = float(violation(x[:, axis])[active].max()) if active.any() else 0.
        if project_active:
            x[active, axis] = bound[active]
    # Only free vertices receive recomputed positions; held and unused ones keep their exact bytes.
    deformed = start.astype(float).copy(); deformed[used[free]] = x[free]
    before = compare_stretch(rest, start, tri, compressed_below=compressed_below, limit=1)['all']
    after = compare_stretch(rest, deformed, tri, compressed_below=compressed_below, limit=1)['all']
    free_ids, fixed_ids = used[free], used[fixed]
    metrics = {'free_vertices': int(len(free_ids)), 'held_vertices': int(len(fixed_ids)), 'iterations': len(energies),
               'converged': converged, 'energy_first': energies[0], 'energy_last': energies[-1], 'clamped_weights': clamped,
               'soft_target_vertices': int((soft[free] > 0).sum()),
               'max_position_change': float(np.linalg.norm(deformed - start, axis=1).max()),
               'compressed_before': before['compressed'], 'compressed_after': after['compressed'],
               'stretched_before': before['stretched'], 'stretched_after': after['stretched'],
               'smallest_stretch_q01_before': before['smallest_stretch_q01'], 'smallest_stretch_q01_after': after['smallest_stretch_q01'],
               'normal_reversals_before': before['normal_reversals'], 'normal_reversals_after': after['normal_reversals'],
               'local_reversals_before': before['local_reversals'], 'local_reversals_after': after['local_reversals']}
    if axis is not None:
        after_violation = violation(x[:, axis])
        metrics.update({'interval_axis': axis, 'interval_vertices': int(bounded.sum()), 'interval_active': int(active.sum()),
                        'interval_outside_before': int((before_violation > 0).sum()), 'interval_outside_after': int((after_violation > 0).sum()),
                        'interval_max_violation_before': float(before_violation.max()),
                        'interval_max_violation_after': float(after_violation.max()),
                        'interval_penalty_residual': penalty_residual, 'interval_set_changes': set_changes,
                        'interval_projected': project_active})
    return {'delta': deformed - start, 'deformed': deformed, 'free_vertices': free_ids, 'held_vertices': fixed_ids,
            'energies': energies, 'public_metrics': metrics,
            'objective': 'As-rigid-as-possible energy of the reference shape with intrinsic cotangent weights, plus optional '
                         'degree-scaled soft position targets and interval penalties on one axis; held vertices exact',
            'limits': 'Shape preservation of the chosen reference only: no guide, depth, anatomical or appearance qualification '
                      'beyond the supplied intervals, which bound one coordinate and fit nothing inside them. '
                      'A folded reference keeps its folds; local minima depend on the initial positions; the held set is an explicit owner choice.'}


def _qualify_support(positions, triangles, relative_area_tolerance):
    """An oriented support surface: nondegenerate, manifold, consistently wound triangles; its components, area-weighted
    vertex normals and open-boundary edges and vertices."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    co, tri = np.asarray(positions), np.asarray(triangles)
    if (co.ndim != 2 or co.shape[1:] != (3,) or co.dtype.kind not in 'fiu' or not np.isfinite(co).all()
            or tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri)
            or tri.min() < 0 or tri.max() >= len(co)):
        raise ValueError('A finite support surface with triangles indexing its positions is required')
    co, tri = co.astype(float), tri.astype(np.int64)
    if len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri):
        raise ValueError('Duplicate support triangles')
    a, b = co[tri[:, 1]] - co[tri[:, 0]], co[tri[:, 2]] - co[tri[:, 0]]
    normal = np.cross(a, b); area2 = np.linalg.norm(normal, axis=1)
    longest2 = np.maximum.reduce([np.sum(a*a, 1), np.sum(b*b, 1), np.sum((b-a)**2, 1)])
    if np.any(area2 <= relative_area_tolerance * longest2):
        raise ValueError('Degenerate support triangle under the declared area tolerance')
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Nonmanifold support edge; a signed offset needs a manifold support')
    direction = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any((counts == 2) & (np.bincount(inverse.ravel(), weights=direction, minlength=len(unique)) != 0)):
        raise ValueError('Inconsistently wound support: a signed offset needs one side of the surface; rewind it')
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(co), len(co)))
    _, vertex_component = connected_components(graph, directed=False)
    component = np.unique(vertex_component[tri[:, 0]], return_inverse=True)[1].ravel()
    vertex_normal = np.zeros_like(co)
    for k in range(3):
        np.add.at(vertex_normal, tri[:, k], normal)
    boundary_edges = unique[counts == 1]
    boundary_vertex = np.zeros(len(co), bool); boundary_vertex[boundary_edges.ravel()] = True
    edge_key = set((boundary_edges[:, 0] * len(co) + boundary_edges[:, 1]).tolist())
    return {'positions': co, 'triangles': tri, 'face_normal': normal / area2[:, None], 'vertex_normal': vertex_normal,
            'component': component, 'components': int(component.max()) + 1, 'boundary_vertex': boundary_vertex,
            'boundary_edge_key': edge_key, 'size': float(np.linalg.norm(np.ptp(co[np.unique(tri)], axis=0)))}


def _support_query(support, points):
    """Closest support point of each point, the offset along the interpolated support normal there (the winding's side
    is positive), the distance across the support's open boundary and the support component."""
    from .geometry import closest_points
    hit = closest_points(points, support['positions'], support['triangles'])
    t = hit['triangles']; bary = hit['barycentric']; corners = support['triangles'][t]
    normal = np.einsum('nk,nkd->nd', bary, support['vertex_normal'][corners])
    length = np.linalg.norm(normal, axis=1)
    face = support['face_normal'][t]
    normal = np.where(length[:, None] > 1e-12, normal / np.maximum(length, 1e-300)[:, None], face)
    offset_vector = np.asarray(points, float) - hit['points']
    offset = np.sum(offset_vector * normal, axis=1)
    # On the open boundary when the closest point is on a boundary edge (one barycentric zero) or boundary vertex.
    zero = bary <= 1e-9; nzero = zero.sum(axis=1)
    on_boundary = np.zeros(len(t), bool)
    at_vertex = nzero == 2
    if at_vertex.any():
        on_boundary[at_vertex] = support['boundary_vertex'][corners[at_vertex, np.argmax(bary[at_vertex], axis=1)]]
    on_edge = np.flatnonzero(nzero == 1)
    if len(on_edge):
        k = np.argmax(zero[on_edge], axis=1)
        e0, e1 = corners[on_edge, (k + 1) % 3], corners[on_edge, (k + 2) % 3]
        key = np.minimum(e0, e1) * len(support['positions']) + np.maximum(e0, e1)
        on_boundary[on_edge] = np.fromiter((q in support['boundary_edge_key'] for q in key.tolist()), bool, len(key))
    across = np.linalg.norm(offset_vector - face * np.sum(offset_vector * face, axis=1)[:, None], axis=1)
    return {'closest': hit['points'], 'distance': hit['distances'], 'normal': normal, 'offset': offset,
            'beyond': np.where(on_boundary, across, 0.), 'component': support['component'][t], 'triangle': t}


def conform_to_surface(reference, initial, triangles, held, support_positions, support_triangles, *, units, frame,
                       relative_area_tolerance, lower=0., upper=0., support_weights=1., band_weight=1e3, iterations=50,
                       tolerance=1e-9, compressed_below=.5, project_active=True, distance_tolerance=None,
                       position_tolerance=None):
    """As-rigid-as-possible deformation of a patch whose free vertices are kept on (or within an offset band of) an
    arbitrary oriented triangle support surface, measured along the support's own normal.

    The shape term, held and free sets and the report are those of rigid_deform. Each iteration every free vertex with
    a positive support weight is matched to its exact closest point on the support (complete search) and the support's
    area-weighted vertex normals interpolated there; its signed offset along that normal (positive on the side the
    support's winding faces) should lie in [lower, upper]. A vertex outside is pulled onto the violated bound by a penalty
    band_weight * support_weight * degree on that normal component only (a 3x3 block n n^T in one coupled XYZ solve), so
    its position along the surface is left to the shape term. Closest points, normals and the active set are re-queried
    after every solve. The solve has converged when the set stops changing, the energy settles (relative `tolerance`)
    and no vertex moved further than `position_tolerance` in the last step (default: `distance_tolerance`, itself by
    default 1e-6 of the support's size); with no weighted free vertex it is exactly rigid_deform, criteria included.
    With project_active every free vertex still outside its band, whatever its positive weight, then moves onto the
    violated bound along the normal, re-queried three times; every measure is taken after that projection. Convergence
    of the solve and support of the result are reported separately.

    Unlike rigid_deform's intervals on one coordinate, the constraint does not weaken where the support turns away
    from an axis. It cannot create area: surplus material is compressed along the surface (reported), and a patch that
    is pushed past the support's open boundary slides off it (reported as beyond the boundary, not as supported).
    """
    from scipy.sparse import coo_matrix, kron, identity
    from scipy.sparse.linalg import factorized
    from .construction_diagnostics import compare_stretch

    rest, start, tri = np.asarray(reference), np.asarray(initial), np.asarray(triangles)
    mask = np.asarray(held)
    if (rest.shape != start.shape or rest.ndim != 2 or rest.shape[1:] != (3,) or not np.isfinite(start).all()
            or not np.isfinite(rest).all() or mask.shape != (len(rest),) or mask.dtype.kind not in 'biu'):
        raise ValueError('Corresponding finite reference/initial positions and one held flag per vertex required')
    if not (int(iterations) >= 1 and np.isfinite(tolerance) and tolerance >= 0 and 0 < compressed_below < 1):
        raise ValueError('Require iterations >= 1, a finite nonnegative tolerance and 0 < compressed_below < 1')
    if tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri) or tri.min() < 0 or tri.max() >= len(rest):
        raise ValueError('Patch triangles must index the supplied vertices')
    n = len(rest)
    low, high = (np.broadcast_to(np.asarray(v, float), (n,)).copy() for v in (lower, upper))
    weight = np.broadcast_to(np.asarray(support_weights, float), (n,)).copy()
    low, high = np.where(np.isnan(low), -np.inf, low), np.where(np.isnan(high), np.inf, high)
    if (low > high).any() or (low == np.inf).any() or (high == -np.inf).any():
        raise ValueError('Each offset band needs lower <= upper')
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Support weights must be finite and nonnegative')
    if not (np.isfinite(band_weight) and band_weight > 0) or not isinstance(project_active, bool):
        raise ValueError('A finite positive band_weight and a boolean project_active are required')
    support = _qualify_support(support_positions, support_triangles, relative_area_tolerance)
    tol = 1e-6 * support['size'] if distance_tolerance is None else float(distance_tolerance)
    step_tol = tol if position_tolerance is None else float(position_tolerance)
    if not (np.isfinite(tol) and tol >= 0 and np.isfinite(step_tol) and step_tol >= 0):
        raise ValueError('Finite nonnegative distance and position tolerances are required')

    tri, used, free, fixed, i, j, w, clamped, m, p, x, degree, laplacian = _arap_setup(
        rest, start, tri, mask, units, frame, relative_area_tolerance)
    low_l, high_l, weight_l = low[used], high[used], weight[used]
    is_free = np.zeros(m, bool); is_free[free] = True
    constrained = is_free & (weight_l > 0)
    watched = weight_l > 0                          # held ones are measured for conflicts, never moved
    kappa = float(band_weight) * weight_l * degree
    edges = p[i] - p[j]
    solve = factorized(laplacian[free][:, free].tocsc()); coupling = laplacian[free][:, fixed]
    stiffness3 = kron(laplacian[free][:, free], identity(3), format='csr')
    position_in_free = np.full(m, -1, np.int64); position_in_free[free] = np.arange(len(free))

    def query(y, which):
        out = {'offset': np.full(m, np.nan), 'normal': np.zeros((m, 3)), 'closest': np.full((m, 3), np.nan),
               'distance': np.full(m, np.nan), 'beyond': np.zeros(m), 'component': np.full(m, -1)}
        ids = np.flatnonzero(which)
        if len(ids):
            q = _support_query(support, y[ids])
            for key in out:
                out[key][ids] = q[key]
        return out

    def residual(q):
        return np.where(watched, np.nan_to_num(np.maximum(np.maximum(low_l - q['offset'], q['offset'] - high_l), 0.)), 0.)

    def constraint(q):
        # Active: outside the band along the normal. The penalty's plane: n . y = n . closest + violated bound.
        active = constrained & ((q['offset'] < low_l) | (q['offset'] > high_l))
        bound = np.where(q['offset'] < low_l, low_l, high_l)
        level = np.where(active, np.sum(q['normal'] * q['closest'], axis=1) + np.where(active, bound, 0.), 0.)
        return active, level

    def energy(y, r, q, active, level):
        value = float((w * np.sum(((y[i] - y[j]) - np.einsum('nab,nb->na', r[i], edges)) ** 2, axis=1)).sum())
        if active.any():
            value += float((kappa[active] * (np.sum(q['normal'][active] * y[active], axis=1) - level[active]) ** 2).sum())
        return value

    q = query(x, watched); first_query = q
    before_residual, before_beyond = residual(q), q['beyond'].copy()
    initial_component = q['component'].copy()
    active, level = constraint(q)
    energies, converged, set_changes, component_changes, last_move = [], False, 0, 0, 0.
    for _ in range(int(iterations)):
        r = _arap_rotations(i, j, w, edges, x, m)
        right = _arap_right(i, j, w, edges, r, m)
        previous = x.copy()
        if active.any():
            # Coupled XYZ solve: the normal penalty is a 3x3 block that leaves the tangential directions free.
            ids = np.flatnonzero(active); at = position_in_free[ids]
            block = kappa[ids, None, None] * q['normal'][ids, :, None] * q['normal'][ids, None, :]
            rows = (3 * at[:, None, None] + np.arange(3)[None, :, None]).repeat(3, axis=2)
            cols = (3 * at[:, None, None] + np.arange(3)[None, None, :]).repeat(3, axis=1)
            system = stiffness3 + coo_matrix((block.ravel(), (rows.ravel(), cols.ravel())), shape=stiffness3.shape)
            load = right[free] - coupling @ x[fixed]
            load[at] += (kappa[ids] * level[ids])[:, None] * q['normal'][ids]
            x[free] = factorized(system.tocsc())(load.ravel()).reshape(-1, 3)
        else:
            for k in range(3):
                x[free, k] = solve(right[free, k] - coupling @ x[fixed, k])
        energies.append(energy(x, _arap_rotations(i, j, w, edges, x, m), q, active, level))
        last_move = float(np.linalg.norm(x - previous, axis=1).max())
        fresh = query(x, watched)
        component_changes += int(np.count_nonzero(constrained & (fresh['component'] != q['component'])))
        q = fresh
        now, level = constraint(q)
        changed = not np.array_equal(now, active); set_changes += int(changed); active = now
        # Closest points can keep sliding on an energy plateau: with a support the last step must be small too.
        if (not changed and len(energies) > 1
                and abs(energies[-2] - energies[-1]) <= tolerance * max(energies[-2], 1e-300)
                and (not constrained.any() or last_move <= step_tol)):
            converged = True; break
    if not np.isfinite(x).all():
        raise ValueError('Surface-constrained deformation did not produce finite positions; qualify the held set')
    penalty_residual = float(residual(q)[constrained].max()) if constrained.any() else 0.
    projection_changes, projection_moves = 0, 0.
    if project_active:
        for _ in range(3):
            active_now, _level = constraint(q)
            if not active_now.any():
                break
            bound = np.where(q['offset'] < low_l, low_l, high_l)
            step = ((bound - q['offset'])[active_now])[:, None] * q['normal'][active_now]
            x[active_now] += step; projection_moves = max(projection_moves, float(np.linalg.norm(step, axis=1).max()))
            fresh = query(x, watched)
            projection_changes += int(np.count_nonzero(constrained & (fresh['component'] != q['component'])))
            q = fresh
    deformed = start.astype(float).copy(); deformed[used[free]] = x[free]
    after = query(x, watched)
    after_residual = residual(after)

    # Facing is measured only on patch triangles whose three vertices have a positive weight (held ones included),
    # against the mean support normal at their closest points; the others are not measured.
    loc = np.full(n, -1, np.int64); loc[used] = np.arange(m); tri_local = loc[tri]
    measured = watched[tri_local].all(axis=1)

    def facing(y, qq):
        if not measured.any():
            return [0, 0]
        t = tri_local[measured]
        tn = np.cross(y[t[:, 1]] - y[t[:, 0]], y[t[:, 2]] - y[t[:, 0]])
        dot = np.sum(tn * qq['normal'][t].sum(axis=1), axis=1)
        return [int(np.count_nonzero(dot > 0)), int(np.count_nonzero(dot <= 0))]

    stretch_before = compare_stretch(rest, start, tri, compressed_below=compressed_below, limit=1)['all']
    stretch_after = compare_stretch(rest, deformed, tri, compressed_below=compressed_below, limit=1)['all']
    unsupported_free = constrained & ((after_residual > tol) | (after['beyond'] > tol))
    held_watch = watched & ~is_free
    held_conflict = held_watch & ((after_residual > tol) | (after['beyond'] > tol))
    free_ids, fixed_ids = used[free], used[fixed]
    c = constrained

    def tail(values, mask_):
        return (float(values[mask_].max()), float(np.percentile(values[mask_], 95))) if mask_.any() else (0., 0.)
    band_max_after, band_p95_after = tail(after_residual, c)
    switched = int(np.count_nonzero(c & (after['component'] != initial_component)))
    metrics = {'free_vertices': int(len(free_ids)), 'held_vertices': int(len(fixed_ids)),
               'support_vertices': int(c.sum()), 'unconstrained_free_vertices': int((is_free & ~c).sum()),
               'iterations': len(energies), 'converged': converged, 'energy_first': energies[0], 'energy_last': energies[-1],
               'last_step_max_move': last_move, 'position_tolerance': step_tol, 'clamped_weights': clamped,
               'band_active': int(active.sum()), 'projection_max_move': projection_moves,
               'band_set_changes': set_changes, 'band_penalty_residual': penalty_residual, 'band_projected': project_active,
               'distance_tolerance': tol,
               'band_outside_before': int(np.count_nonzero(c & (before_residual > tol))),
               'band_outside_after': int(np.count_nonzero(c & (after_residual > tol))),
               'band_max_before': float(before_residual[c].max()) if c.any() else 0.,
               'band_max_after': band_max_after, 'band_p95_after': band_p95_after,
               'support_distance_max_after': float(np.nanmax(after['distance'][c])) if c.any() else 0.,
               'beyond_boundary_before': int(np.count_nonzero(c & (before_beyond > tol))),
               'beyond_boundary_after': int(np.count_nonzero(c & (after['beyond'] > tol))),
               'beyond_boundary_max_after': float(after['beyond'][c].max()) if c.any() else 0.,
               'unsupported_after': int(unsupported_free.sum()), 'supported_after': int((c & ~unsupported_free).sum()),
               'held_on_support': int(held_watch.sum()), 'held_conflicts': int(held_conflict.sum()),
               'held_conflict_max': float(np.maximum(after_residual, after['beyond'])[held_conflict].max()) if held_conflict.any() else 0.,
               'support_components': support['components'], 'component_switches': switched,
               'component_changes_during': component_changes, 'component_changes_projection': projection_changes,
               'patch_triangles': int(len(tri)), 'support_facing_measured_triangles': int(measured.sum()),
               'support_facing_before': facing(np.asarray(start, float)[used], first_query),
               'support_facing_after': facing(x, after),
               'max_position_change': float(np.linalg.norm(deformed - start, axis=1).max()),
               'compressed_before': stretch_before['compressed'], 'compressed_after': stretch_after['compressed'],
               'stretched_before': stretch_before['stretched'], 'stretched_after': stretch_after['stretched'],
               'smallest_stretch_q01_before': stretch_before['smallest_stretch_q01'],
               'smallest_stretch_q01_after': stretch_after['smallest_stretch_q01'],
               'normal_reversals_before': stretch_before['normal_reversals'], 'normal_reversals_after': stretch_after['normal_reversals'],
               'local_reversals_before': stretch_before['local_reversals'], 'local_reversals_after': stretch_after['local_reversals']}
    warnings = []
    if switched or component_changes or projection_changes:
        warnings.append('Closest points moved between disconnected support components; the support does not say which '
                        'part a vertex belongs to')
    if metrics['beyond_boundary_after']:
        warnings.append('Vertices lie beyond the support\'s open boundary: their normal offset does not make them supported')
    if held_conflict.any():
        warnings.append('Held vertices lie off their support band; they keep their exact positions')
    if not converged:
        warnings.append('Iteration limit reached before the band set, energy and positions settled')
    if measured.sum() < len(tri):
        warnings.append('Support facing measured on %d of %d patch triangles (the rest have a vertex with zero weight)'
                        % (int(measured.sum()), len(tri)))
    offset = np.full(n, np.nan); offset[used] = after['offset']
    band = np.zeros(n); band[used] = after_residual
    beyond = np.zeros(n); beyond[used] = after['beyond']
    return {'delta': deformed - start, 'deformed': deformed, 'free_vertices': free_ids, 'held_vertices': fixed_ids,
            'energies': energies, 'support_offset': offset, 'band_residual': band, 'beyond_boundary': beyond,
            'public_metrics': metrics, 'warnings': warnings,
            'objective': 'As-rigid-as-possible energy of the reference shape with intrinsic cotangent weights, plus a '
                         'degree-scaled penalty on the offset along the support\'s interpolated normal outside [lower, upper] '
                         'at each weighted free vertex\'s exact closest support point, re-queried every solve; held vertices exact',
            'limits': 'Keeps material on the supplied support only: no guide, anatomical or appearance qualification, and no '
                      'correspondence (the closest point is not a semantic match). It cannot create area; surplus is '
                      'compressed along the surface. A folded reference keeps its folds; the support facing counts and '
                      'reversals are measures on this result, not a proof of an unfolded or self-intersection-free sheet.'}


def planar_relayout(positions, triangles, free, *, plane_axes=(0, 2), depth_axis=1, depth_samples=None, depth_map=None,
                    window=None, cell=None, depth_smoothing=0., reference=None, compressed_below=.5, layout='harmonic',
                    depth_blur=0., iterations=60):
    """Re-lay collapsed material in a projection plane and give it depth from its surroundings or a guide.

    With layout='harmonic' the free vertices take the uniform harmonic (Tutte) layout of the patch in the plane of
    `plane_axes`, every other patch vertex held where it is: each free vertex at the mean of its patch neighbours. With
    layout='rigid' they keep the `reference` layout (for example the rest pose) up to local rotations instead:
    `rigid_deform` of the positions projected into the plane, every other patch vertex held; use it where the material
    has to turn (a fan around a corner) and uniform weights would even out its uneven spacing. Their depth along
    `depth_axis` then comes either from a smoothed thin-plate field through the depth of the `depth_samples` vertices
    (for example the retained surface around the block) or from a front depth map over `window`/`cell` (for example a
    guide surface, or the retained surface itself), optionally low-passed by a Gaussian of `depth_blur` cells.
    Use it where a pose has squeezed material into parallel rows or columns (collapsed quads) and the projection plane
    shows that block unfolded. Free vertices must be interior to the patch; hold the rows an attachment samples.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    from .construction_diagnostics import compare_stretch

    P, tri = np.asarray(positions, float), np.asarray(triangles)
    axes = (int(plane_axes[0]), int(plane_axes[1]), int(depth_axis))
    if P.ndim != 2 or P.shape[1] != 3 or not np.isfinite(P).all() or sorted(axes) != [0, 1, 2]:
        raise ValueError('Finite (N, 3) positions and three distinct axes required')
    if tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri) or tri.min() < 0 or tri.max() >= len(P):
        raise ValueError('Patch triangles must index the supplied positions')
    mask = np.zeros(len(P), bool)
    free_in = np.asarray(free)
    if free_in.dtype == bool:
        if free_in.shape != (len(P),):
            raise ValueError('A boolean free mask needs one flag per position')
        mask = free_in.copy()
    else:
        if free_in.dtype.kind not in 'iu' or free_in.ndim != 1 or (len(free_in) and (free_in.min() < 0 or free_in.max() >= len(P))):
            raise ValueError('Free vertices must be a boolean mask or valid indices')
        mask[free_in] = True
    if (depth_samples is None) == (depth_map is None):
        raise ValueError('Give exactly one depth source: depth_samples or depth_map')
    if layout not in ('harmonic', 'rigid'):
        raise ValueError("layout must be 'harmonic' or 'rigid'")
    if layout == 'rigid' and reference is None:
        raise ValueError('A rigid layout needs the reference positions whose layout it keeps')
    if not (np.isfinite(depth_blur) and depth_blur >= 0):
        raise ValueError('A nonnegative depth_blur (cells) is required')
    tri = tri.astype(np.int64); used = np.unique(tri)
    if not mask.any() or not np.isin(np.flatnonzero(mask), used).all():
        raise ValueError('Free vertices must belong to the patch')
    edges = np.sort(np.r_[tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if (counts > 2).any():
        raise ValueError('Nonmanifold patch edges refuse a harmonic layout')
    boundary = np.zeros(len(P), bool); boundary[np.unique(unique[counts == 1])] = True
    if (mask & boundary).any():
        raise ValueError('Free vertices must be interior to the patch; hold its boundary')
    n = len(P); i, j = unique[:, 0], unique[:, 1]
    A = coo_matrix((np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()
    degree = np.asarray(A.sum(1)).ravel()
    L = (coo_matrix((degree, (np.arange(n), np.arange(n))), shape=(n, n)) - A).tocsr()
    f = np.flatnonzero(mask); h = np.setdiff1d(used, f)
    out = P.copy()
    rigid_metrics = {}
    if layout == 'harmonic':
        for k in axes[:2]:
            out[f, k] = spsolve(L[f][:, f].tocsc(), -(L[f][:, h] @ P[h, k]))
    else:
        ref = np.asarray(reference, float)
        if ref.shape != P.shape or not np.isfinite(ref).all():
            raise ValueError('The reference must correspond to the positions')
        R2, I2 = ref.copy(), P.copy(); R2[:, axes[2]] = 0.; I2[:, axes[2]] = 0.
        held = np.zeros(n, bool); held[h] = True
        rd = rigid_deform(R2, I2, tri, held, units='chart units', frame='projection plane',
                          relative_area_tolerance=1e-12, iterations=int(iterations))
        for k in axes[:2]:
            out[f, k] = np.asarray(rd['deformed'], float)[f, k]
        rigid_metrics = {'rigid_' + k: v for k, v in rd['public_metrics'].items()
                         if k in ('iterations', 'converged', 'energy_first', 'energy_last', 'clamped_weights')}
    plane = out[f][:, list(axes[:2])]
    if depth_samples is not None:
        from scipy.interpolate import RBFInterpolator
        s = np.asarray(depth_samples)
        smask = np.zeros(n, bool)
        if s.dtype == bool:
            if s.shape != (n,):
                raise ValueError('A boolean sample mask needs one flag per position')
            smask = s.copy()
        else:
            smask[s.astype(np.int64)] = True
        if (smask & mask).any():
            raise ValueError('Depth samples must not include the re-laid vertices')
        if smask.sum() < 3:
            raise ValueError('At least three depth samples are required')
        field = RBFInterpolator(P[smask][:, list(axes[:2])], P[smask, axes[2]], kernel='thin_plate_spline',
                                smoothing=float(depth_smoothing) * int(smask.sum()), degree=1)
        out[f, axes[2]] = field(plane)
        source = {'depth_source': 'samples', 'depth_samples': int(smask.sum())}
    else:
        from scipy.ndimage import map_coordinates
        D = np.asarray(depth_map, float); win = np.asarray(window, float)
        if D.ndim != 2 or win.shape != (4,) or not (cell is not None and np.isfinite(cell) and cell > 0):
            raise ValueError('A 2D depth map with its window (a0, a1, b0, b1) and cell size is required')
        ci = (plane[:, 0] - win[0]) / cell - .5; cj = (plane[:, 1] - win[2]) / cell - .5
        V = np.where(np.isfinite(D), D, 0.); W = np.isfinite(D).astype(float)
        if depth_blur > 0:                     # low-pass that ignores empty cells (normalized convolution)
            from scipy.ndimage import gaussian_filter
            num, den = gaussian_filter(V * W, depth_blur), gaussian_filter(W, depth_blur)
            V = np.where(W > 0, num / np.maximum(den, 1e-12), 0.)
        v = map_coordinates(V, [cj, ci], order=1, cval=0.); w = map_coordinates(W, [cj, ci], order=1, cval=0.)
        if (w < .99).any():
            raise ValueError('The depth map does not cover every re-laid vertex')
        out[f, axes[2]] = v / w
        source = {'depth_source': 'map', 'depth_blur_cells': float(depth_blur)}
    if not np.isfinite(out).all():
        raise ValueError('Re-layout did not produce finite positions; qualify the held set')

    def flips(Q):
        a = Q[tri][:, :, axes[0]]; b = Q[tri][:, :, axes[1]]
        s2 = (a[:, 1] - a[:, 0]) * (b[:, 2] - b[:, 0]) - (a[:, 2] - a[:, 0]) * (b[:, 1] - b[:, 0])
        major = 1. if (s2 > 0).sum() >= (s2 < 0).sum() else -1.
        return int(np.count_nonzero(s2 * major < 0))
    metrics = {'free_vertices': int(len(f)), 'held_vertices': int(len(h)), **source, 'layout': layout, **rigid_metrics,
               'plane_flips_before': flips(P), 'plane_flips_after': flips(out),
               'max_position_change': float(np.linalg.norm(out - P, axis=1).max())}
    if reference is not None:
        ref = np.asarray(reference, float)
        if ref.shape != P.shape:
            raise ValueError('The reference must correspond to the positions')
        before = compare_stretch(ref, P, tri, compressed_below=compressed_below, limit=1)['all']
        after = compare_stretch(ref, out, tri, compressed_below=compressed_below, limit=1)['all']
        metrics.update({'compressed_before': before['compressed'], 'compressed_after': after['compressed'],
                        'smallest_stretch_q01_before': before['smallest_stretch_q01'], 'smallest_stretch_q01_after': after['smallest_stretch_q01'],
                        'local_reversals_before': before['local_reversals'], 'local_reversals_after': after['local_reversals']})
    return {'relaid': out, 'delta': out - P, 'free_vertices': f, 'held_vertices': h, 'public_metrics': metrics,
            'objective': ('Uniform harmonic (Tutte) layout' if layout == 'harmonic' else 'As-rigid-as-possible layout of the reference')
                         + ' of the free vertices in the projection plane with the rest of the patch held; depth from a '
                           'smoothed thin-plate field through the samples, or from a (low-passed) depth map',
            'limits': 'The plane must show the block unfolded and its held surroundings must be sound: a uniform layout '
                      'evens out spacing, it does not know anatomy. The free material loses its own relief: its depth is '
                      'that of the field. No guide is fitted unless the depth map is a guide.'}


def projected_boundary(positions, triangles, *, units, frame, free=None, plane_axes=(0, 2), plane_basis=None,
                       tolerance=None, relative_tolerance=1e-9):
    """Report the boundary loops of a triangle patch as seen in a projection plane, before a planar re-layout.

    With `free` (the mask or indices given to planar_relayout) the domain is the triangles incident to a free vertex:
    the influence domain of the layout, in which only the free vertices move. Its boundary is the held boundary those
    vertices are laid out against; held vertices inside that domain are listed separately (no loop check covers them).
    Without `free` the supplied patch's own boundary is analysed. Boundary edges are directed along their triangle's winding and traced into loops through the triangle
    fans, so a pinched vertex splits loops instead of joining a figure of eight. Every edge pair is classified once
    (proper crossing, touch, collinear overlap, adjacent backtracking) and corners are measured from the ordered loop.
    One length `tolerance` (default `relative_tolerance` times the projected boundary's bounding-box diagonal) decides
    every near case. Predicates run on coordinates centred in 3D and normalized by the boundary's span, with sign
    tests; a span outside 1e-150 to 1e150 refuses. A projection test only: it reports geometry and approves nothing.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    P, tri = np.asarray(positions), np.asarray(triangles)
    if (P.ndim != 2 or P.shape[1:] != (3,) or P.dtype.kind not in 'fiu' or not np.isfinite(P).all()
            or tri.ndim != 2 or tri.shape[1:] != (3,) or tri.dtype.kind not in 'iu' or not len(tri)
            or tri.min() < 0 or tri.max() >= len(P)):
        raise ValueError('Finite (N, 3) positions and triangles indexing them are required')
    if not isinstance(units, str) or not units.strip() or not isinstance(frame, str) or not frame.strip():
        raise ValueError('Explicit units and frame are required')
    P, tri = P.astype(float), tri.astype(np.int64)
    if ((tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])).any():
        raise ValueError('A triangle repeats a vertex; remove index-degenerate triangles')
    if len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri):
        raise ValueError('Duplicate triangles do not define a boundary')
    basis_tolerance = 1e-9                           # orthonormality of plane_basis: absolute, dimensionless
    if plane_basis is not None:
        basis = np.asarray(plane_basis, float)
        if (basis.shape != (2, 3) or not np.isfinite(basis).all()
                or np.abs(basis @ basis.T - np.eye(2)).max() > basis_tolerance):
            raise ValueError('plane_basis must be two orthonormal in-plane axes, shape (2, 3), within 1e-9')
        plane = {'basis': basis.tolist(), 'basis_tolerance': basis_tolerance}
    else:
        axes = list(plane_axes)
        if (len(axes) != 2 or not all(isinstance(a, (int, np.integer)) and not isinstance(a, bool) for a in axes)
                or axes[0] == axes[1] or not all(a in (0, 1, 2) for a in axes)):
            raise ValueError('plane_axes must be two distinct integer axes among 0, 1 and 2')
        axes = [int(a) for a in axes]; basis = np.eye(3)[axes]; plane = {'axes': axes}
    if not (np.isfinite(relative_tolerance) and 0 < relative_tolerance < 1):
        raise ValueError('Require 0 < relative_tolerance < 1')
    if tolerance is not None and not (np.isfinite(tolerance) and 0 <= tolerance < 1e150):
        raise ValueError('A nonnegative tolerance below 1e150 (a length in the given units) is required')

    n = len(P); used = np.zeros(n, bool); used[tri.ravel()] = True
    free_mask = None
    if free is not None:
        f = np.asarray(free)
        free_mask = np.zeros(n, bool)
        if f.dtype == bool:
            if f.shape != (n,):
                raise ValueError('A boolean free mask needs one flag per position')
            free_mask[:] = f
        else:
            if f.dtype.kind not in 'iu' or f.ndim != 1 or (len(f) and (f.min() < 0 or f.max() >= n)):
                raise ValueError('Free vertices must be a boolean mask or valid indices')
            free_mask[f] = True
        if not free_mask.any() or (free_mask & ~used).any():
            raise ValueError('Free vertices must belong to the patch (as for planar_relayout)')
        domain_ids = np.flatnonzero(free_mask[tri].any(axis=1))
    else:
        domain_ids = np.arange(len(tri))
    D = tri[domain_ids]

    def undirected(t):
        e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        owner = np.tile(np.arange(len(t)), 3)
        unique, inverse, counts = np.unique(np.sort(e, axis=1), axis=0, return_inverse=True, return_counts=True)
        return e, owner, unique, inverse.ravel(), counts

    e, owner, unique, inverse, counts = undirected(D)
    forward = np.where(e[:, 0] < e[:, 1], 1, -1)
    direction_sum = np.bincount(inverse, weights=forward, minlength=len(unique))
    nonmanifold = unique[counts > 2]
    inconsistent = unique[(counts == 2) & (direction_sum != 0)]
    boundary_rows = np.flatnonzero(counts[inverse] == 1)
    half = {(int(a), int(b)): int(domain_ids[o]) for (a, b), o in zip(e, owner)}
    boundary = {(int(e[r, 0]), int(e[r, 1])): int(domain_ids[owner[r]]) for r in boundary_rows}
    third = {}
    for t in domain_ids.tolist():
        a, b, c = (int(v) for v in tri[t])
        third[(t, a, b)] = c; third[(t, b, c)] = a; third[(t, c, a)] = b

    # Components: triangles joined through shared edges (a pinched vertex does not join them).
    pairs = []
    order = np.argsort(inverse, kind='stable'); starts = np.r_[0, np.cumsum(counts)[:-1]]
    rows_sorted = owner[order]
    for k in np.flatnonzero(counts > 1):
        group = rows_sorted[starts[k]:starts[k] + counts[k]]
        pairs += [(group[0], g) for g in group[1:]]
    pairs = np.array(pairs, np.int64).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(D), len(D)))
    _, triangle_component = connected_components(graph, directed=False)
    # Fans: a vertex's incident triangles joined through the edges at that vertex. More than one fan is a nonmanifold
    # vertex whatever its boundary degree (for example a closed part touching the patch at one vertex).
    corner = {(int(r), int(v)): 3 * int(r) + k for r in range(len(D)) for k, v in enumerate(D[r])}
    links = []
    for k in np.flatnonzero(counts > 1):
        group = rows_sorted[starts[k]:starts[k] + counts[k]]
        for v in unique[k]:
            links += [(corner[(int(group[0]), int(v))], corner[(int(g), int(v))]) for g in group[1:]]
    links = np.array(links, np.int64).reshape(-1, 2)
    _, fan = connected_components(coo_matrix((np.ones(len(links)), (links[:, 0], links[:, 1])),
                                             shape=(3 * len(D), 3 * len(D))), directed=False)
    vertex_fans = {}
    for (r, v), node in corner.items():
        vertex_fans.setdefault(v, set()).add(int(fan[node]))
    multi_fan = sorted(v for v, fans in vertex_fans.items() if len(fans) > 1)
    component_detail = []
    for cid in range(int(triangle_component.max()) + 1):
        rows = D[triangle_component == cid]
        edges_c = np.unique(np.sort(np.r_[rows[:, [0, 1]], rows[:, [1, 2]], rows[:, [2, 0]]], axis=1), axis=0)
        component_detail.append({'component': cid, 'triangles': int(len(rows)),
                                 'euler_characteristic': int(len(np.unique(rows)) - len(edges_c) + len(rows))})

    # Loops: follow each boundary half-edge to the next one around its end vertex through the triangle fan.
    outgoing = {}
    for (a, b) in boundary:
        outgoing.setdefault(a, []).append((a, b))
    degree = np.zeros(n, np.int64)
    for (a, b) in boundary:
        degree[a] += 1; degree[b] += 1
    pinched = np.flatnonzero((degree >= 4) & (degree % 2 == 0))
    branched = np.flatnonzero(degree % 2 == 1)
    ambiguous = set()

    def successor(h):
        a, b = h
        options = outgoing.get(b, [])
        if len(options) == 1:
            return options[0]
        t = boundary[h]; c = third[(t, a, b)]
        for _ in range(len(D) + 1):                     # rotate about b: (b, c) in t, then its twin's triangle
            if (b, c) in boundary:
                return (b, c)
            twin = half.get((c, b))
            if twin is None or twin == t:
                break
            t, c = twin, third[(twin, c, b)]
        ambiguous.add(b)
        return options[0] if options else None

    used_half, loops, chains = set(), [], []
    for start in sorted(boundary):
        if start in used_half:
            continue
        path, h = [], start
        while h is not None and h not in used_half:
            used_half.add(h); path.append(h); h = successor(h)
        (loops if h == start else chains).append(path)

    all_boundary = sorted({v for path in loops + chains for h in path for v in h})
    # Translate in 3D before projecting (large offsets would otherwise cost precision), then measure the plane span.
    anchor = P[all_boundary] if all_boundary else P[np.unique(D)]
    centre3 = (anchor.min(axis=0) + anchor.max(axis=0)) / 2
    Q = (P - centre3) @ basis.T
    centre = centre3 @ basis.T                        # plane coordinates of the local origin, for reported points
    if all_boundary:
        box = Q[all_boundary]; diagonal = float(np.hypot(*(box.max(axis=0) - box.min(axis=0))))
    else:
        diagonal = 0.
    if diagonal and not 1e-150 < diagonal < 1e150:
        raise ValueError('The projected boundary spans %g units: outside the supported range 1e-150 to 1e150; '
                         'rescale the positions' % diagonal)
    tolerance_given = float(tolerance) if tolerance is not None else float(relative_tolerance) * diagonal
    # Predicates run in coordinates normalized by the boundary's span; lengths and areas are reported in caller units.
    unit = diagonal if diagonal else 1.
    Q = Q / unit
    tol = tolerance_given / unit

    # Edge table over loops and open chains: (vertex a, vertex b, owner triangle, loop or chain id, position).
    edges = []
    for kind, groups in (('loop', loops), ('chain', chains)):
        for g, path in enumerate(groups):
            for k, (a, b) in enumerate(path):
                edges.append((a, b, boundary[(a, b)], kind, g, k, len(path)))
    E = len(edges)
    A = np.array([Q[x[0]] for x in edges]).reshape(-1, 2); B = np.array([Q[x[1]] for x in edges]).reshape(-1, 2)
    length = np.linalg.norm(B - A, axis=1)
    collapsed = length <= tol

    def cross2(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def side(a, b, p):
        # signed distance of p from the line a -> b (left positive); None when the edge is collapsed
        ab = b - a; ln = float(np.hypot(*ab))
        return None if ln <= tol or ln == 0 else float(cross2(ab, p - a)) / ln

    def point_segment(p, a, b):
        ab = b - a; ln2 = float(ab @ ab)
        t = 0. if ln2 == 0 else min(max(float((p - a) @ ab) / ln2, 0.), 1.)
        return float(np.linalg.norm(p - (a + t * ab)))

    def ident(i):
        a, b, t, kind, g, k, _ = edges[i]
        return {'edge': [a, b], 'triangle': t, kind: g, 'position': k}

    def plane_point(p):
        return (p * unit + centre).tolist()

    lo = np.minimum(A, B) - tol; hi = np.maximum(A, B) + tol
    vertices_of = [set(x[:2]) for x in edges]
    crossings, touches, overlaps, backtracks = [], [], [], []
    involved = {}                                     # (kind, id) -> set of issue names

    def mark(i, name):
        involved.setdefault(edges[i][3:5], set()).add(name)

    for i in range(E):
        if i + 1 >= E:
            break
        near = np.flatnonzero((lo[i + 1:, 0] <= hi[i, 0]) & (hi[i + 1:, 0] >= lo[i, 0])
                              & (lo[i + 1:, 1] <= hi[i, 1]) & (hi[i + 1:, 1] >= lo[i, 1])) + i + 1
        for j in near:
            shared = vertices_of[i] & vertices_of[j]
            if shared:                                   # adjacent edges: only folding back is an issue
                if collapsed[i] or collapsed[j]:
                    continue
                s = next(iter(shared))
                u = B[i] if edges[i][0] == s else A[i]; w = B[j] if edges[j][0] == s else A[j]
                if point_segment(u, Q[s], w) <= tol or point_segment(w, Q[s], u) <= tol:
                    backtracks.append({'vertex': s, 'edges': [ident(i), ident(j)]})
                    mark(i, 'adjacent_overlap'); mark(j, 'adjacent_overlap')
                continue
            d1, d2 = side(A[i], B[i], A[j]), side(A[i], B[i], B[j])
            d3, d4 = side(A[j], B[j], A[i]), side(A[j], B[j], B[i])
            proper = (None not in (d1, d2, d3, d4) and (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)
                      and min(abs(d1), abs(d2), abs(d3), abs(d4)) > tol)
            if proper:
                r = d1 / (d1 - d2)
                crossings.append({'edges': [ident(i), ident(j)], 'point': plane_point(A[j] + r * (B[j] - A[j]))})
                mark(i, 'crossing'); mark(j, 'crossing'); continue
            c1, c2 = cross2(B[i] - A[i], A[j] - A[i]), cross2(B[i] - A[i], B[j] - A[i])
            c3, c4 = cross2(B[j] - A[j], A[i] - A[j]), cross2(B[j] - A[j], B[i] - A[j])
            meets = np.sign(c1) * np.sign(c2) <= 0 and np.sign(c3) * np.sign(c4) <= 0 and not (c1 == c2 == 0 and max(
                point_segment(A[j], A[i], B[i]), point_segment(B[j], A[i], B[i]),
                point_segment(A[i], A[j], B[j]), point_segment(B[i], A[j], B[j])) > 0)
            distance = 0. if meets else min(point_segment(A[j], A[i], B[i]), point_segment(B[j], A[i], B[i]),
                                            point_segment(A[i], A[j], B[j]), point_segment(B[i], A[j], B[j]))
            if distance > tol:
                continue
            flat = (None not in (d1, d2) and max(abs(d1), abs(d2)) <= tol) or \
                   (None not in (d3, d4) and max(abs(d3), abs(d4)) <= tol)
            span = 0.
            if flat and not (collapsed[i] and collapsed[j]):
                k = i if length[i] >= length[j] else j; o = j if k == i else i
                axis = (B[k] - A[k]) / length[k]
                t0, t1 = sorted([float((A[o] - A[k]) @ axis), float((B[o] - A[k]) @ axis)])
                span = min(t1, float(length[k])) - max(t0, 0.)
            if span > tol:
                overlaps.append({'edges': [ident(i), ident(j)], 'overlap_length': span * unit})
                mark(i, 'collinear_overlap'); mark(j, 'collinear_overlap')
            else:
                touches.append({'edges': [ident(i), ident(j)], 'distance': distance * unit,
                                'segments_meet': bool(meets)})
                mark(i, 'touch'); mark(j, 'touch')
    collapsed_list = [dict(ident(i), length=float(length[i]) * unit) for i in np.flatnonzero(collapsed)]
    for i in np.flatnonzero(collapsed):
        mark(i, 'collapsed_edge')

    loop_reports = []
    for g, path in enumerate(loops):
        verts = [a for a, _ in path]
        X = Q[verts]; m = len(verts)
        area = float(0.5 * np.sum(cross2(X, np.roll(X, -1, axis=0))))
        perimeter = float(np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).sum())
        sides = []
        for (a, b) in path:
            s = side(Q[a], Q[b], Q[third[(boundary[(a, b)], a, b)]])
            sides.append(0 if s is None or abs(s) <= tol else (1 if s > 0 else -1))
        sides = np.array(sides)
        left, right = int((sides > 0).sum()), int((sides < 0).sum())
        majority = 1 if left >= right else -1
        turns, deviation = [], []
        for k in range(m):
            p, v, q = X[k - 1], X[k], X[(k + 1) % m]
            e1, e2 = v - p, q - v
            ok = np.hypot(*e1) > tol and np.hypot(*e2) > tol
            turns.append(float(np.degrees(np.arctan2(cross2(e1, e2), e1 @ e2))) if ok else None)
            chord = q - p; cl = float(np.hypot(*chord))
            deviation.append(float(abs(cross2(chord, v - p))) / cl if cl > tol else float(np.hypot(*e1)))
        defined_turns = None not in turns
        turning = round(sum(turns) / 360.) if defined_turns else None
        issues = sorted(involved.get(('loop', g), set()))
        own = [x for x in crossings + touches + overlaps + backtracks
               if all(y.get('loop') == g for y in x['edges'])]
        reasons = []
        if own or any(collapsed[i] for i in range(E) if edges[i][3:5] == ('loop', g)):
            reasons.append('the loop is not simple in this projection')
        if set(verts) & set(pinched.tolist()) or set(verts) & ambiguous:
            reasons.append('the loop passes a pinched vertex')
        if abs(area) <= tol * perimeter:              # area threshold: the tolerance times the perimeter (length^2)
            reasons.append('the loop encloses no area beyond the tolerance')
        if right and left or (sides == 0).any():
            reasons.append('the domain lies on both sides of the loop, or on none, somewhere along it')
        if turning not in (1, -1):
            reasons.append('the turning number is not +1 or -1')
        report = {'loop': g, 'vertices': verts, 'edges': [list(h) for h in path],
                  'triangles': [boundary[h] for h in path], 'component': int(triangle_component[
                      int(np.flatnonzero(domain_ids == boundary[path[0]])[0])]),
                  'signed_area': area * unit ** 2, 'area_tolerance': tol * perimeter * unit ** 2, 'orientation': 'counterclockwise' if area > tol * perimeter else
                  ('clockwise' if area < -tol * perimeter else 'degenerate'), 'perimeter': perimeter * unit,
                  'turning_number': turning, 'domain_side_edges': {'left': left, 'right': right,
                                                                   'degenerate': int((sides == 0).sum())},
                  'issues': issues, 'simple': not own and 'collapsed_edge' not in issues and not
                  (set(verts) & (set(pinched.tolist()) | ambiguous))}
        if not reasons:
            # The angle on the domain (material) side; at a hole loop that is outside the hole polygon.
            interior = [180. - t * majority for t in turns]
            corners = [{'vertex': verts[k], 'domain_angle_degrees': interior[k], 'deviation': deviation[k] * unit}
                       for k in range(m) if deviation[k] > tol]
            report.update(interior_angles='defined on the domain side',
                          reflex_corners=sorted([c for c in corners if c['domain_angle_degrees'] > 180],
                                                key=lambda c: -c['domain_angle_degrees']),
                          convex_corners=sum(1 for c in corners if c['domain_angle_degrees'] < 180),
                          straight_corners=m - len(corners))
        else:
            local = [{'vertex': verts[k], 'turn_degrees': turns[k], 'deviation': deviation[k] * unit}
                     for k in range(m) if turns[k] is not None and deviation[k] > tol and turns[k] * majority < 0]
            report.update(interior_angles='undefined: ' + '; '.join(reasons), reflex_corners=None,
                          local_turns_against_domain=sorted(local, key=lambda c: -abs(c['turn_degrees'])))
        loop_reports.append(report)

    def inside(p, poly):
        x, y = p; hit = False
        for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
        return hit
    # Nesting only between simple loops that neither cross nor touch each other; otherwise it is left unresolved.
    contact = set()
    for x in crossings + touches + overlaps + backtracks:
        ids = [y.get('loop') for y in x['edges']]
        if None not in ids and ids[0] != ids[1]:
            contact.add(frozenset(ids))
    simple_polys = {r['loop']: Q[r['vertices']] for r in loop_reports if r['interior_angles'].startswith('defined')}
    for r in loop_reports:
        g = r['loop']
        if g in simple_polys:
            others = [o for o in simple_polys if o != g and frozenset((g, o)) not in contact]
            r['inside_loops'] = [o for o in others if inside(Q[r['vertices'][0]], simple_polys[o])]
            r['nesting_unresolved_with'] = sorted(o for o in range(len(loops))
                                                  if o != g and (o not in simple_polys or frozenset((g, o)) in contact))
        else:
            r['inside_loops'] = None
            r['nesting_unresolved_with'] = sorted(o for o in range(len(loops)) if o != g)
    chain_reports = [{'chain': g, 'vertices': [path[0][0]] + [b for _, b in path], 'edges': [list(h) for h in path],
                      'triangles': [boundary[h] for h in path]} for g, path in enumerate(chains)]

    patch_boundary = set()
    if free_mask is not None:
        _, _, all_unique, _, all_counts = undirected(tri)
        patch_boundary = set(all_unique[all_counts == 1].ravel().tolist())
    domain_vertices = np.unique(D)
    on_loops = set(all_boundary)
    interior_held = ([int(v) for v in domain_vertices if not free_mask[v] and v not in on_loops]
                     if free_mask is not None else [])
    free_on_boundary = sorted(int(v) for v in np.flatnonzero(free_mask) if v in patch_boundary) if free_mask is not None else []
    defined = [r for r in loop_reports if r['reflex_corners'] is not None]
    warnings = []
    if not loops and not chains:
        warnings.append('The domain has no boundary: it is closed, so there is no held loop to report')
    elif diagonal == 0.:
        warnings.append('The projected boundary has zero span: every boundary edge collapses in this plane')
    if multi_fan:
        warnings.append('Vertices with more than one triangle fan: the domain is not a manifold surface there')
    metrics = {'domain': 'triangles incident to a free vertex (only free vertices move)' if free_mask is not None
               else 'the supplied patch',
               'domain_triangles': int(len(D)), 'components': int(triangle_component.max()) + 1,
               'loops': len(loops), 'open_chains': len(chains), 'boundary_edges': E,
               'crossings': len(crossings), 'touches': len(touches), 'collinear_overlaps': len(overlaps),
               'adjacent_overlaps': len(backtracks), 'collapsed_edges': int(collapsed.sum()),
               'pinched_vertices': int(len(pinched)), 'branched_vertices': int(len(branched)),
               'multi_fan_vertices': len(multi_fan),
               'ambiguous_vertices': len(ambiguous), 'nonmanifold_edges': int(len(nonmanifold)),
               'inconsistent_edges': int(len(inconsistent)),
               'simple_loops': sum(1 for r in loop_reports if r['simple']),
               'loops_with_interior_angles': len(defined),
               'reflex_corners': sum(len(r['reflex_corners']) for r in defined),
               'local_turns_against_domain': sum(len(r.get('local_turns_against_domain', [])) for r in loop_reports),
               'interior_held_vertices': len(interior_held), 'free_on_patch_boundary': len(free_on_boundary),
               'tolerance': tolerance_given, 'tolerance_source': 'given' if tolerance is not None else 'relative to the boundary size',
               'relative_tolerance': float(relative_tolerance), 'boundary_diagonal': diagonal,
               'units': units, 'frame': frame}
    result = {'public_metrics': metrics, 'plane': plane, 'warnings': warnings, 'components': component_detail,
            'loops': loop_reports, 'open_chains': chain_reports, 'multi_fan_vertices': multi_fan,
            'crossings': crossings, 'touches': touches, 'collinear_overlaps': overlaps, 'adjacent_overlaps': backtracks,
            'collapsed_edges': collapsed_list, 'pinched_vertices': pinched.tolist(), 'branched_vertices': branched.tolist(),
            'ambiguous_vertices': sorted(ambiguous), 'nonmanifold_edges': nonmanifold.tolist(),
            'inconsistent_edges': inconsistent.tolist(), 'interior_held_vertices': interior_held,
            'free_on_patch_boundary': free_on_boundary,
            'conventions': 'Plane coordinates are (positions @ basis.T): first axis right, second up; counterclockwise '
                           'is positive area. Edges run along their triangle\'s winding. A turn is positive to the left; '
                           'the interior angle is measured on the side where the owning triangles lie. Every near case '
                           'uses the one length tolerance.',
            'limits': 'A projection test of the analysed domain\'s boundary, not 3D collision: projected crossings can be '
                      'separate sheets. No crossings and few reflex corners do not make a layout injective: that also '
                      'depends on the whole graph, held interior handles, every boundary, the rigid layout and later '
                      'depth or rest blending. Every decision (crossing, touch, straight or reflex corner) is made '
                      'within the declared tolerance in floating point: a clear report means compatible within that '
                      'tolerance, not exact convexity or simplicity.'}

    def finite(value):
        if isinstance(value, float):
            return np.isfinite(value)
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return True
    if not finite(result):
        raise ValueError('The report would contain nonfinite values: coordinates or tolerance outside the supported range')
    return result


def smooth_region(positions, triangles, *, held=None, weights=None, iterations=10, lam=.5, mu=-.53):
    """Taubin smoothing of a mesh region: each pass moves every point toward the mean of its edge neighbours by `lam`
    times its weight, then back by `mu`, which smooths lumps without shrinking the surface.

    `positions` is one shape (N, 3) or a stack (K, N, 3) smoothed by the same operator: give the rest and the
    mechanism's end shape together. Smoothing only the rest leaves the end shape's change fitted to the old rest, and the
    closing lid creases (found in real use). `held` points do not move (margins and the row beside them, a lid's inner
    surface, a mirror-hidden half); `weights` (0..1 per point, default 1) ramp the smoothing out, for example over rows
    from the margin, a radius fade and a taper at the corners. The operator is linear, so the stack's differences (the
    motion) are smoothed exactly as the shapes are.
    """
    X = np.asarray(positions, float); single = X.ndim == 2
    X = X[None] if single else X
    T = np.asarray(triangles)
    if X.ndim != 3 or X.shape[2] != 3 or not np.isfinite(X).all():
        raise ValueError('Finite (N, 3) positions or a (K, N, 3) stack are required')
    n = X.shape[1]
    if T.ndim != 2 or T.shape[1] != 3 or T.dtype.kind not in 'iu' or not len(T) or T.min() < 0 or T.max() >= n:
        raise ValueError('Triangles must index the supplied positions')
    if not (int(iterations) >= 0 and np.isfinite(lam) and np.isfinite(mu) and lam > 0 > mu and -mu > lam):
        raise ValueError('Taubin smoothing needs lam > 0, mu < -lam and a nonnegative number of iterations')
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if w.shape != (n,) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError('Weights must be one value in [0, 1] per point')
    w = w.copy()
    if held is not None:
        w[np.asarray(held, np.int64)] = 0.
    from scipy.sparse import coo_matrix, diags
    edges = np.unique(np.sort(np.r_[T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1), axis=0)
    A = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                   shape=(n, n)).tocsr()
    degree = np.asarray(A.sum(axis=1)).ravel()
    mean = diags(1 / np.maximum(degree, 1)) @ A                       # mean of the edge neighbours
    W = w * (degree > 0)
    out = X.copy()
    for k in range(len(out)):
        Y = out[k]
        for _ in range(int(iterations)):
            Y = Y + (lam * W)[:, None] * (mean @ Y - Y)
            Y = Y + (mu * W)[:, None] * (mean @ Y - Y)
        out[k] = Y
    moved = np.linalg.norm(out - X, axis=2)
    result = out[0] if single else out
    return {'positions': result,
            'public_metrics': {'points': int(n), 'smoothed_points': int(np.count_nonzero(W > 0)),
                               'held_points': int(np.count_nonzero(W == 0)), 'largest_move': float(moved.max()),
                               'median_move_of_smoothed': float(np.median(moved[:, W > 0])) if (W > 0).any() else 0.},
            'objective': 'Taubin smoothing (lam, mu) of the weighted points toward their edge neighbours, the same linear '
                         'operator on every shape of the stack',
            'limits': 'Smooths geometry, it does not know likeness: check it against the guide and the drawing, and smooth '
                      'the construction\'s end shape with the same weights. Held and zero-weight points keep their exact '
                      'positions.'}
