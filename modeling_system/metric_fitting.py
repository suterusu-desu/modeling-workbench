"""Metric-aware planar finite elements on supplied recorded geometry."""
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
