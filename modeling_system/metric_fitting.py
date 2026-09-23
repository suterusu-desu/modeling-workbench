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
               'normal_reversals_before': before['normal_reversals'], 'normal_reversals_after': after['normal_reversals']}
    return {'delta': delta, 'relaxed': relaxed, 'free_vertices': free, 'held_vertices': fixed, 'public_metrics': metrics,
            'objective': 'Squared intrinsic Laplacian of the displacement from reference on interior rows; held vertices exact',
            'limits': 'Construction repair of material distribution only: no guide, depth, anatomical or appearance qualification. '
                      'Depth changes wherever material slides over curvature; the held set is an explicit owner choice.'}
