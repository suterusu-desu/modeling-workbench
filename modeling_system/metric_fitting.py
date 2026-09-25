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
    from scipy.sparse import coo_matrix, diags
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
        covariance = np.zeros((m, 3, 3))
        np.add.at(covariance, i, w[:, None, None] * edges[:, :, None] * (y[i] - y[j])[:, None, :])
        u, _, vt = np.linalg.svd(covariance)
        r = np.einsum('nji,nkj->nik', vt, u)
        flip = np.linalg.det(r) < 0
        if flip.any():
            u = u.copy(); u[flip, :, -1] *= -1; r[flip] = np.einsum('nji,nkj->nik', vt[flip], u[flip])
        return r

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
        right = np.zeros((m, 3)); np.add.at(right, i, .5 * w[:, None] * np.einsum('nab,nb->na', r[i] + r[j], edges))
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
