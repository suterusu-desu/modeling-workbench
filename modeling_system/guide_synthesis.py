"""Pose guides from several generated variants, joined onto an accepted neutral surface (recorded arrays only).

A generator run is not repeatable at the scale of small features: independent runs of the same pose differ by smooth
offsets and by local detail, and a generated neutral pose is not the accepted neutral. These helpers keep what the
variants agree on as a *change* from their own neutral pose, carry that change onto the accepted neutral surface, and
say how far it can be trusted at every place. Everything works on front depth maps: one depth value per cell of a
regular chart seen along one axis. No anatomy, correspondence or appearance is inferred here; the caller chooses
windows, supports, pinned points and tolerances from its own evidence.
"""
import warnings
import numpy as np


def _window(window, cell):
    window = np.asarray(window, float)
    if window.shape != (4,) or not np.isfinite(window).all() or not window[1] > window[0] or not window[3] > window[2]:
        raise ValueError('Window (a0, a1, b0, b1) with a0 < a1 and b0 < b1 required')
    if not (np.isfinite(cell) and cell > 0):
        raise ValueError('Positive cell size required')
    shape = (int(round((window[3] - window[2]) / cell)), int(round((window[1] - window[0]) / cell)))
    if min(shape) < 2 or shape[0] * shape[1] > 4_000_000:
        raise ValueError('Window must hold between 2x2 and 4,000,000 cells')
    return window, float(cell), shape


def _grid(window, cell, shape):
    a = window[0] + (np.arange(shape[1]) + .5) * cell
    b = window[2] + (np.arange(shape[0]) + .5) * cell
    return np.meshgrid(a, b)


def _front(front):
    if front not in ('min', 'max'):
        raise ValueError("front must be 'min' (smaller depth is nearer) or 'max'")
    return front


def _map(value, shape=None, name='map'):
    value = np.asarray(value, float)
    if value.ndim != 2 or (shape is not None and value.shape != tuple(shape)):
        raise ValueError(f'{name}: a 2D map of the chart shape is required')
    return value


def front_depth(positions, triangles, *, window, cell, depth_axis=1, chart_axes=(0, 2), front='min'):
    """Front-most surface depth per chart cell (a z-buffer of cell centres).

    `window` is (a0, a1, b0, b1) in the two chart axes; cells are `cell` wide. `front='min'` keeps the smallest value
    along `depth_axis` (the surface nearest a viewer looking toward increasing depth). Empty cells are NaN. The
    triangle index map records which triangle is seen in each cell (-1 where empty).
    """
    positions, triangles = np.asarray(positions, float), np.asarray(triangles)
    window, cell, shape = _window(window, cell)
    front = _front(front)
    axes = (int(chart_axes[0]), int(chart_axes[1]), int(depth_axis))
    if (positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all() or sorted(axes) != [0, 1, 2]):
        raise ValueError('Finite (N, 3) positions and three distinct axes required')
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.dtype.kind not in 'iu' or not len(triangles)
            or triangles.min() < 0 or triangles.max() >= len(positions)):
        raise ValueError('Triangles must index the supplied positions')
    sign = 1. if front == 'min' else -1.
    P = positions[triangles][:, :, list(axes)]
    A, B, D = P[:, :, 0], P[:, :, 1], sign * P[:, :, 2]
    keep = np.flatnonzero((A.max(1) > window[0]) & (A.min(1) < window[1]) & (B.max(1) > window[2]) & (B.min(1) < window[3]))
    depth = np.full(shape, np.inf); owner = np.full(shape, -1, np.int64)
    for t in keep:
        a, b, d = A[t], B[t], D[t]
        i0 = max(int(np.floor((a.min() - window[0]) / cell - .5)), 0); i1 = min(int(np.ceil((a.max() - window[0]) / cell - .5)), shape[1] - 1)
        j0 = max(int(np.floor((b.min() - window[2]) / cell - .5)), 0); j1 = min(int(np.ceil((b.max() - window[2]) / cell - .5)), shape[0] - 1)
        if i1 < i0 or j1 < j0:
            continue
        det = (b[1] - b[2]) * (a[0] - a[2]) + (a[2] - a[1]) * (b[0] - b[2])
        if abs(det) < 1e-18:
            continue
        ga, gb = np.meshgrid(window[0] + (np.arange(i0, i1 + 1) + .5) * cell, window[2] + (np.arange(j0, j1 + 1) + .5) * cell)
        l0 = ((b[1] - b[2]) * (ga - a[2]) + (a[2] - a[1]) * (gb - b[2])) / det
        l1 = ((b[2] - b[0]) * (ga - a[2]) + (a[0] - a[2]) * (gb - b[2])) / det
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-9) & (l1 >= -1e-9) & (l2 >= -1e-9)
        value = np.where(inside, l0 * d[0] + l1 * d[1] + l2 * d[2], np.inf)
        sub = depth[j0:j1 + 1, i0:i1 + 1]; nearer = value < sub
        sub[nearer] = value[nearer]; owner[j0:j1 + 1, i0:i1 + 1][nearer] = t
    covered = np.isfinite(depth)
    depth = np.where(covered, sign * depth, np.nan)
    return {'depth': depth, 'triangle_index': owner,
            'public_metrics': {'cells': int(depth.size), 'covered_share': float(covered.mean()), 'triangles_in_window': int(len(keep))},
            'objective': 'Front-most interpolated depth at each cell centre (a z-buffer); no surface fitting',
            'limits': 'Sampled at cell centres: features narrower than a cell can be missed, and a surface seen edge-on '
                      'is represented only where its front triangles cover cell centres.'}


def remove_thin_relief(depth, *, size, front='min'):
    """Remove thin parts that stand in front of the surface (lash fins, strands, fused strokes).

    A grey closing (for `front='min'`; an opening for `'max'`) with a `size` x `size` cell square removes every part
    nearer than its surroundings that is narrower than `size` cells in some direction, and keeps broad forms. Empty
    cells stay empty. Choose `size` just above the relief to remove: a broad convex form narrower than it (a rolled
    margin, a thin fold) is removed too, so check the `removed` map where such forms matter. Slopes are kept exactly
    (except within half the square of the map's border, where the filter reflects);
    a form curved toward the viewer is flattened by up to its depth curvature times the square's half-diagonal squared
    (a lid of radius .06 under an 11-cell square of .0005: about .0002), and relief crossing a crease or fold edge is
    filled with an error up to the change of slope there times the square's half-width.
    """
    from scipy import ndimage
    depth = _map(depth, name='depth'); front = _front(front)
    if type(size) is not int or size < 3 or size % 2 == 0:
        raise ValueError('An odd integer size of 3 cells or more required')
    ok = np.isfinite(depth)
    if not ok.any():
        raise ValueError('The depth map has no covered cells')
    if front == 'min':
        filled = np.where(ok, depth, np.nanmax(depth)); result = ndimage.grey_closing(filled, size=(size, size))
    else:
        filled = np.where(ok, depth, np.nanmin(depth)); result = ndimage.grey_opening(filled, size=(size, size))
    result = np.where(ok, result, np.nan); removed = np.abs(result - depth)
    return {'depth': result, 'removed': removed,
            'public_metrics': {'cells_changed': int(np.count_nonzero(removed[ok] > 0)), 'max_removed': float(np.nanmax(removed)),
                               'median_removed_where_changed': float(np.median(removed[ok][removed[ok] > 0])) if np.any(removed[ok] > 0) else 0.},
            'objective': f'Grey {"closing" if front == "min" else "opening"} with a {size}-cell square',
            'limits': 'Morphological: removes every thin forward part, wanted or not; it does not know what a lash is.'}


def stationary_offset(target, source, support, *, window, cell, stride=4, smoothing=1e-3, max_samples=6000):
    """Smooth offset field that makes `source` match `target` where the surface is known not to move.

    Fits a smoothed thin-plate spline to (target - source) on `support` cells (every `stride`-th), and evaluates it
    everywhere. Use it to remove the smooth depth offsets between generator runs (support: skin the pose leaves still)
    or a generator's whole-area shift. `smoothing` is per sample, in squared depth units.
    """
    from scipy.interpolate import RBFInterpolator
    window, cell, shape = _window(window, cell)
    target, source = _map(target, shape, 'target'), _map(source, shape, 'source')
    support = np.asarray(support)
    if support.shape != shape or support.dtype != bool:
        raise ValueError('A boolean support map of the chart shape is required')
    if type(stride) is not int or stride < 1 or not (np.isfinite(smoothing) and smoothing >= 0):
        raise ValueError('Positive integer stride and nonnegative smoothing required')
    A, B = _grid(window, cell, shape)
    ok = support & np.isfinite(target) & np.isfinite(source)
    grid = np.zeros(shape, bool); grid[::stride, ::stride] = True
    use = ok & grid
    count = int(use.sum())
    if count < 10:
        raise ValueError('Fewer than 10 supported samples: widen the support or lower the stride')
    if count > max_samples:
        raise ValueError(f'{count} samples exceed max_samples={max_samples}: raise the stride')
    field = RBFInterpolator(np.c_[A[use], B[use]], (target - source)[use], kernel='thin_plate_spline',
                            smoothing=smoothing * count, degree=1)
    offset = field(np.c_[A.ravel(), B.ravel()]).reshape(shape)
    before = np.abs(target - source)[ok]; after = np.abs(target - source - offset)[ok]
    return {'offset': offset,
            'public_metrics': {'support_samples': count, 'support_mad_before': float(np.median(before)),
                               'support_mad_after': float(np.median(after))},
            'objective': 'Smoothed thin-plate spline through (target - source) on the declared still support',
            'limits': 'Assumes the declared support does not move; a moving part inside it is averaged into the offset.'}


def pose_change(rest_maps, pose_maps, support, *, window, cell, reference=None, stride=4, smoothing=1e-3, behind=None,
                front='min'):
    """Consensus change of a pose from its own generated neutral pose, with the variants' disagreement.

    `rest_maps` (k0, H, W) are generated variants of the neutral pose, `pose_maps` (k1, H, W) of the pose, all in the
    same registered chart. Each rest variant is aligned to `reference` (the accepted neutral map, if given) and each
    pose variant to the rest consensus by `stationary_offset` on `support`; the change is the pose consensus minus the
    rest consensus (medians). `pose_spread` is the median absolute deviation of the pose variants from their consensus,
    `rest_spread` the same for the rest variants; `spread` combines both in quadrature (the change's own uncertainty).
    `behind` screens holes after alignment: a cell more than `behind` behind the reference (the rest consensus when no
    reference is given) shows something through a cut in the generated surface and becomes empty. Generated runs can
    sit far from each other before alignment, so screen here rather than before.
    """
    window, cell, shape = _window(window, cell)
    front = _front(front)
    if behind is not None and not (np.isfinite(behind) and behind > 0):
        raise ValueError('A positive hole screen distance is required when given')

    def screen(m, ref):
        if behind is None or ref is None:
            return m
        gap = (m - ref) if front == 'min' else (ref - m)
        with np.errstate(invalid='ignore'):
            return np.where(gap > behind, np.nan, m)
    rest_maps, pose_maps = np.asarray(rest_maps, float), np.asarray(pose_maps, float)
    if rest_maps.ndim != 3 or pose_maps.ndim != 3 or rest_maps.shape[1:] != shape or pose_maps.shape[1:] != shape:
        raise ValueError('Stacks of maps of the chart shape required')
    if len(rest_maps) < 1 or len(pose_maps) < 1:
        raise ValueError('At least one rest and one pose variant required')
    reference = None if reference is None else _map(reference, shape, 'reference')
    rest = []
    for m in rest_maps:
        rest.append(m if reference is None else screen(m + stationary_offset(reference, m, support, window=window, cell=cell,
                                                                             stride=stride, smoothing=smoothing)['offset'], reference))
    rest = np.stack(rest)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)          # all-empty cells stay NaN
        rest_consensus = np.nanmedian(rest, axis=0)
        pose = np.stack([screen(m + stationary_offset(rest_consensus, m, support, window=window, cell=cell, stride=stride,
                                                      smoothing=smoothing)['offset'],
                                reference if reference is not None else rest_consensus) for m in pose_maps])
        pose_consensus = np.nanmedian(pose, axis=0)
        rest_spread = np.nanmedian(np.abs(rest - rest_consensus[None]), axis=0)
        pose_spread = np.nanmedian(np.abs(pose - pose_consensus[None]), axis=0)
    change = pose_consensus - rest_consensus
    spread = np.sqrt(np.nan_to_num(rest_spread) ** 2 + np.nan_to_num(pose_spread) ** 2)
    spread[~np.isfinite(change)] = np.nan
    finite = np.isfinite(change)
    return {'change': change, 'spread': spread, 'rest_spread': rest_spread, 'pose_spread': pose_spread,
            'rest_consensus': rest_consensus, 'pose_consensus': pose_consensus,
            'rest_count': np.isfinite(rest).sum(0), 'pose_count': np.isfinite(pose).sum(0),
            'public_metrics': {'rest_variants': int(len(rest_maps)), 'pose_variants': int(len(pose_maps)),
                               'finite_share': float(finite.mean()),
                               'spread_median': float(np.median(spread[finite])) if finite.any() else None},
            'objective': 'Median change of the pose variants from the rest variants after smooth still-support alignment',
            'limits': 'Where the variants place a feature differently the change is blurred and the spread large; a '
                      'change measured on a generated neutral is not yet placed on the accepted neutral\'s features.'}


def _bspline_basis(points, box, knots):
    from scipy import sparse
    counts = [int(np.ceil((box[1] - box[0]) / knots)) + 3, int(np.ceil((box[3] - box[2]) / knots)) + 3]
    parts = []
    for axis, low, n in ((0, box[0], counts[0]), (1, box[2], counts[1])):
        u = np.clip((points[:, axis] - low) / knots, 0, n - 3 - 1e-9); i = np.floor(u).astype(int); t = u - i
        b = np.stack([(1 - t) ** 3, 3 * t ** 3 - 6 * t ** 2 + 4, -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1, t ** 3], 1) / 6.
        parts.append((i, b))
    (ia, ba), (ib, bb) = parts
    rows = np.repeat(np.arange(len(points)), 16)
    cols = ((ib[:, None, None] + np.arange(4)[None, None, :]) * counts[0] + ia[:, None, None] + np.arange(4)[None, :, None]).reshape(-1)
    vals = (ba[:, :, None] * bb[:, None, :]).reshape(-1)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(points), counts[0] * counts[1])), counts


def _bending(counts):
    from scipy import sparse
    na, nb = counts
    d2 = lambda n: sparse.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    d1 = lambda n: sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    I = sparse.identity
    rows = [sparse.kron(I(nb), d2(na)), sparse.kron(d2(nb), I(na)), np.sqrt(2) * sparse.kron(d1(nb), d1(na))]
    return sparse.vstack(rows).tocsr()


def fit_depth_field(points, values, *, knots, bending, weights=None, box=None, pins=None, pin_values=None,
                    pin_weight=1e3, anchors=None, anchor_weight=1., robust=None, iterations=4, evaluate_at=None):
    """Smooth depth field through scattered samples: a cubic B-spline over a regular knot grid in a 2D chart.

    Minimizes sum(w_i (F(p_i) - v_i)^2) / sum(w_i) + bending * mean(second differences of the coefficients)^2 with
    optional soft pins (points whose value is prescribed, weighted `pin_weight` times the mean sample weight),
    zero-valued anchors (weight `anchor_weight` times the mean sample weight each) and Huber reweighting at `robust`
    (value units). `knots` is the knot spacing in chart units: features shorter than about two knots cannot be
    represented. The coefficients, box and spacing are returned so the field can be evaluated elsewhere with
    `evaluate_depth_field`.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    points, values = np.asarray(points, float), np.asarray(values, float)
    if points.ndim != 2 or points.shape[1] != 2 or values.shape != (len(points),) or not len(points):
        raise ValueError('Nonempty (n, 2) chart points and n values required')
    if not (np.isfinite(points).all() and np.isfinite(values).all()):
        raise ValueError('Finite sample points and values required')
    if not (np.isfinite(knots) and knots > 0 and np.isfinite(bending) and bending >= 0):
        raise ValueError('Positive knot spacing and nonnegative bending weight required')
    w = np.ones(len(points)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(points),) or not np.isfinite(w).all() or w.min() < 0 or not w.sum() > 0:
        raise ValueError('Nonnegative finite sample weights with a positive sum required')
    extra = [np.zeros((0, 2))]; extra_v = [np.zeros(0)]; extra_w = [np.zeros(0)]
    mean_w = w.sum() / max(np.count_nonzero(w), 1)
    if pins is not None:
        pins = np.asarray(pins, float); pv = np.zeros(len(pins)) if pin_values is None else np.asarray(pin_values, float)
        if pins.ndim != 2 or pins.shape[1] != 2 or pv.shape != (len(pins),) or not np.isfinite(pins).all() or not np.isfinite(pv).all():
            raise ValueError('Finite (m, 2) pins and m pin values required')
        extra += [pins]; extra_v += [pv]; extra_w += [np.full(len(pins), pin_weight * mean_w)]
    if anchors is not None:
        anchors = np.asarray(anchors, float)
        if anchors.ndim != 2 or anchors.shape[1] != 2 or not np.isfinite(anchors).all():
            raise ValueError('Finite (m, 2) anchors required')
        extra += [anchors]; extra_v += [np.zeros(len(anchors))]; extra_w += [np.full(len(anchors), anchor_weight * mean_w)]
    allp = np.concatenate([points, *extra]); allv = np.concatenate([values, *extra_v])
    fixed_w = np.concatenate([w, *extra_w]); n_samples = len(points)
    if box is None:
        box = (allp[:, 0].min(), allp[:, 0].max() + 1e-12, allp[:, 1].min(), allp[:, 1].max() + 1e-12)
    box = np.asarray(box, float)
    if box.shape != (4,) or not box[1] > box[0] or not box[3] > box[2]:
        raise ValueError('Box (a0, a1, b0, b1) with positive extent required')
    if ((allp[:, 0] < box[0]) | (allp[:, 0] > box[1]) | (allp[:, 1] < box[2]) | (allp[:, 1] > box[3])).any():
        raise ValueError('Every sample, pin and anchor must lie inside the box')
    Bm, counts = _bspline_basis(allp, box, knots)
    if counts[0] * counts[1] > 250_000:
        raise ValueError('Too many coefficients: use a larger knot spacing or a smaller box')
    Pen = _bending(counts); PtP = (Pen.T @ Pen) / Pen.shape[0]
    robust_w = np.ones(len(allp)); huber = None if robust is None else float(robust)
    if huber is not None and not huber > 0:
        raise ValueError('A positive robust threshold is required when given')
    scale = 1. / fixed_w[:n_samples].sum()
    for _ in range(int(iterations) if huber is not None else 1):
        W = fixed_w * robust_w * scale
        A = (Bm.T @ sparse.diags(W) @ Bm + bending * PtP).tocsc()
        coefficients = spsolve(A, Bm.T @ (W * allv))
        if huber is None:
            break
        r = np.abs(Bm @ coefficients - allv)
        robust_w = np.where(r > huber, huber / np.maximum(r, 1e-300), 1.); robust_w[n_samples:] = 1.
    fitted = Bm @ coefficients
    residual = fitted[:n_samples] - values
    pin_error = None
    if pins is not None and len(pins):
        pin_error = float(np.abs(fitted[n_samples:n_samples + len(pins)] - allv[n_samples:n_samples + len(pins)]).max())
    result = {'coefficients': coefficients.reshape(counts[1], counts[0]), 'box': box, 'knots': float(knots),
              'fitted': fitted[:n_samples], 'residual': residual,
              'public_metrics': {'samples': int(n_samples), 'coefficients': int(counts[0] * counts[1]),
                                 'residual_median_abs': float(np.median(np.abs(residual))),
                                 'residual_p90_abs': float(np.quantile(np.abs(residual), .9)),
                                 'downweighted_samples': int(np.count_nonzero(robust_w[:n_samples] < 1)),
                                 'pin_max_abs_error': pin_error},
              'objective': 'Weighted least squares of a cubic B-spline field with a second-difference bending penalty, '
                           'soft pins and zero anchors, optional Huber reweighting',
              'limits': 'A smooth field: it cannot represent steps or features shorter than about two knots, and '
                        'extrapolates weakly where no samples, pins or anchors constrain it.'}
    if evaluate_at is not None:
        result['evaluated'] = evaluate_depth_field(result['coefficients'], box, knots, evaluate_at)['values']
    return result


def evaluate_depth_field(coefficients, box, knots, points):
    """Values of a field returned by `fit_depth_field` at chart points (clamped to its box)."""
    coefficients, points, box = np.asarray(coefficients, float), np.asarray(points, float), np.asarray(box, float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Finite (n, 2) chart points required')
    clamped = np.c_[np.clip(points[:, 0], box[0], box[1]), np.clip(points[:, 1], box[2], box[3])]
    Bm, counts = _bspline_basis(clamped, box, float(knots))
    if tuple(coefficients.shape) != (counts[1], counts[0]):
        raise ValueError('Coefficients do not match the box and knot spacing')
    return {'values': Bm @ coefficients.reshape(-1)}


def change_band(spread, change=None, *, cell=None, position_uncertainty=0., accuracy=0.):
    """Per-cell tolerance of a transferred change: the variants' spread, the change's own slope times how far its
    features may be misplaced, and a documented accuracy, combined in quadrature.

    A change is carried by the position of the features that make it (a lid margin, a fold). Where it varies fast,
    a small placement error of those features is a large depth error; `position_uncertainty` (chart units) turns the
    slope of `change` into that error.
    """
    spread = _map(spread, name='spread')
    if not (np.isfinite(position_uncertainty) and position_uncertainty >= 0 and np.isfinite(accuracy) and accuracy >= 0):
        raise ValueError('Nonnegative position uncertainty and accuracy required')
    slope_term = np.zeros_like(spread)
    if position_uncertainty > 0:
        if change is None or cell is None or not (np.isfinite(cell) and cell > 0):
            raise ValueError('The change map and cell size are required with a position uncertainty')
        change = _map(change, spread.shape, 'change')
        filled = np.where(np.isfinite(change), change, np.nanmedian(change))
        ga, gb = np.gradient(filled, cell)
        slope = np.hypot(ga, gb); slope[~np.isfinite(change)] = np.nan
        slope_term = slope * position_uncertainty
    band = np.sqrt(np.nan_to_num(spread) ** 2 + np.nan_to_num(slope_term) ** 2 + accuracy ** 2)
    band[~np.isfinite(spread)] = np.nan
    ok = np.isfinite(band)
    return {'band': band, 'slope_term': slope_term,
            'public_metrics': {'band_median': float(np.median(band[ok])) if ok.any() else None,
                               'band_p90': float(np.quantile(band[ok], .9)) if ok.any() else None},
            'objective': 'sqrt(spread^2 + (|grad change| * position_uncertainty)^2 + accuracy^2) per cell',
            'limits': 'The components are the caller\'s evidence; a band says how far the guide can be trusted, not '
                      'what shape is right inside it.'}


def band_excess(values, lower, upper):
    """How far each value lies outside its [lower, upper] interval (negative below, positive above, 0 inside)."""
    values, lower, upper = (np.asarray(a, float) for a in (values, lower, upper))
    if not (values.shape == lower.shape == upper.shape) or np.any(upper < lower):
        raise ValueError('Matching values and intervals with lower <= upper required')
    excess = np.where(values < lower, values - lower, np.where(values > upper, values - upper, 0.))
    ok = np.isfinite(excess)
    return {'excess': excess, 'public_metrics': {'inside_share': float(np.mean(excess[ok] == 0)) if ok.any() else None,
                                                 'outside': int(np.count_nonzero(excess[ok])),
                                                 'max_abs_excess': float(np.abs(excess[ok]).max()) if ok.any() else None}}


def height_field_mesh(depth, *, window, cell, keep=None, depth_axis=1, chart_axes=(0, 2), max_step=None):
    """Triangle mesh of a depth map: a vertex at every kept, covered cell centre, two triangles per 2x2 block of them.

    Blocks whose depth differs by more than `max_step` across the block (a cliff where one surface hides another) are
    left open. Use it to display or fit a synthesized guide as a surface.
    """
    window, cell, shape = _window(window, cell)
    depth = _map(depth, shape, 'depth')
    axes = (int(chart_axes[0]), int(chart_axes[1]), int(depth_axis))
    if sorted(axes) != [0, 1, 2]:
        raise ValueError('Three distinct axes required')
    ok = np.isfinite(depth)
    if keep is not None:
        keep = np.asarray(keep)
        if keep.shape != shape or keep.dtype != bool:
            raise ValueError('A boolean keep map of the chart shape is required')
        ok &= keep
    index = -np.ones(shape, np.int64); index[ok] = np.arange(int(ok.sum()))
    A, B = _grid(window, cell, shape)
    positions = np.zeros((int(ok.sum()), 3)); positions[:, axes[0]] = A[ok]; positions[:, axes[1]] = B[ok]; positions[:, axes[2]] = depth[ok]
    q = np.stack([index[:-1, :-1], index[:-1, 1:], index[1:, 1:], index[1:, :-1]], -1).reshape(-1, 4)
    full = (q >= 0).all(1)
    if max_step is not None:
        d = np.stack([depth[:-1, :-1], depth[:-1, 1:], depth[1:, 1:], depth[1:, :-1]], -1).reshape(-1, 4)[full]
        full[np.flatnonzero(full)] = (d.max(1) - d.min(1)) <= max_step
    q = q[full]
    triangles = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int64)
    return {'positions': positions, 'triangles': triangles, 'vertex_index': index,
            'public_metrics': {'vertices': int(len(positions)), 'triangles': int(len(triangles)),
                               'open_blocks': int(np.count_nonzero(~full))},
            'limits': 'A front height field: undersides, returns and anything hidden from the chart view are absent.'}
