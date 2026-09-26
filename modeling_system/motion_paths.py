"""Motion built as a mechanism: a part turned on a hinge, clean in-between paths, pace and coherent pieces.

Build a moving feature as the mechanism reference characters use before fitting any pose: a lid is one piece turning
on a hinge with one timing, landing on the still opposing lid. The hinge operations derive the end pose from the
mechanism and carry attached parts with it:

- `hinge_landing`: an edge (a lid margin) turned about a hinge axis onto a landing curve (the opposing margin at rest)
  at its own axial position, with a weighted band of material carried by the same turn; nothing slides along the axis.
- `hinge_carry`: attached material (lashes) moved by the hinge change of the host point it sits on, at the host's
  fraction, with the per-point turn, radial and axial values a native rig needs.
- `hinge_change`: each point's turn about the axis, change of distance from it and slide along it between two poses.

A blend shape moves every point along its straight chord from a reference pose to an end pose, all at the same pace.
That is enough when nothing lies behind the moving surface. Where material passes over a convex obstacle (a lid over a
round eye) the chord cuts into it, and keys fitted separately at intermediate phases plus correction fields stacked on
top bend the in-between sections instead (an S across a moving band: sunk behind its edge, bulged at it). These
operations construct in-between poses from the two end poses themselves:

- `motion_pace`: when each point moves, measured as progress along its own chord in an existing motion, smoothed over
  the surface and weighted by travel, so points that barely move take the pace of the material around them.
- `path_positions`: where each point is at its pace: on its chord, or rolled about a hinge axis (angle and distance
  interpolated, no sideways drift) where the material passes over a round obstacle.
- `hinge_motion`: a part that must move as one piece (a flap swinging back about a corner) turns rigidly about a fixed
  pivot by the best rotation between its two poses, carrying its non-rigid residual linearly.
- `keep_clearance` and `end_clearance`: points stay outside an obstacle behind them (an eye behind a lid) by at least
  the clearance they have in their two poses, capped at the margin that matters, with a smooth envelope and push.
- `schedule_pace`, `shared_schedule` and `smooth_step`: re-time a smoothly weighted region of an inherited motion,
  either onto one shared schedule (so a region moves as one piece instead of shearing against its neighbours) or with a
  pace floor (a part that must reach its end pose by a given phase), with temporal and spatial fades. They repair the
  timing of a motion that already exists; they do not construct one. A lid given per-region paces (a corner closing
  first) closes like a zipper; build it on a hinge with one pace instead.
- `travel_weight`: how much of a rebuilt motion each point takes where it joins the existing one, from the existing
  motion's travel, smoothed over the surface so barely moving neighbours do not alternate.

None of them fits a guide or judges appearance. `hinge_landing` derives its end pose from the hinge; the other
operations take both end poses as established.
"""
import numpy as np


def _pair(reference, end):
    rest, final = np.asarray(reference, float), np.asarray(end, float)
    if (rest.ndim != 2 or rest.shape[1:] != (3,) or rest.shape != final.shape or not len(rest)
            or not np.isfinite(rest).all() or not np.isfinite(final).all()):
        raise ValueError('Corresponding finite reference and end positions required')
    return rest, final


def _pace(pace, n):
    a = np.asarray(pace, float)
    if a.ndim not in (1, 2) or a.shape[-1] != n or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Pace must be finite values in [0, 1], one per point (optionally per phase)')
    return a


def motion_pace(reference, end, samples, *, sigma, cutoff=None, travel_scale=None, minimum_travel=0.):
    """Progress of every point along its own reference -> end chord in an existing motion, smoothed and monotone.

    `samples` holds the existing motion's positions at increasing phases, shape (phases, points, 3); the first and last
    phases are taken as 0 and 1. The raw pace of a point is the projection of its displacement onto its own chord,
    divided by the chord length squared, clipped to [0, 1]. Each point's pace is then replaced by a Gaussian average
    (`sigma`, neighbours within `cutoff`, default 3 sigma, found in the reference pose) in which every neighbour counts
    with min(1, travel / travel_scale)^2: points that barely move take the pace of the material around them, and a
    point whose own travel is below `minimum_travel` contributes nothing. travel_scale defaults to the 90th percentile
    travel. The result is made monotone (running maximum over phases), with the first phase 0 and the last 1.
    """
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    rest, final = _pair(reference, end)
    seq = np.asarray(samples, float)
    if seq.ndim != 3 or seq.shape[1:] != rest.shape or len(seq) < 2 or not np.isfinite(seq).all():
        raise ValueError('Samples must be finite positions of every point at two or more phases')
    if not (np.isfinite(sigma) and sigma > 0) or (cutoff is not None and not (np.isfinite(cutoff) and cutoff > 0)):
        raise ValueError('A positive smoothing sigma (and cutoff) is required')
    if not (np.isfinite(minimum_travel) and minimum_travel >= 0):
        raise ValueError('Minimum travel must be finite and nonnegative')
    chord = final - rest; travel = np.linalg.norm(chord, axis=1); length2 = travel ** 2
    moving = travel > max(minimum_travel, 1e-12)
    raw = np.einsum('gnk,nk->gn', seq - rest, chord) / np.where(moving, length2, 1)
    overshoot = int(np.count_nonzero(((raw < -1e-9) | (raw > 1 + 1e-9)) & moving))
    raw = np.where(moving, np.clip(raw, 0, 1), 0.)
    scale = float(travel_scale) if travel_scale is not None else float(np.quantile(travel[moving], .9)) if moving.any() else 1.
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError('Travel scale must be positive')
    weight = np.where(moving, np.minimum(1., travel / scale) ** 2, 0.)
    reach = 3 * sigma if cutoff is None else float(cutoff)
    pairs = cKDTree(rest).query_pairs(reach, output_type='ndarray')
    i = np.r_[pairs[:, 0], pairs[:, 1], np.arange(len(rest))]; j = np.r_[pairs[:, 1], pairs[:, 0], np.arange(len(rest))]
    kernel = np.exp(-.5 * (np.linalg.norm(rest[i] - rest[j], axis=1) / sigma) ** 2)
    W = coo_matrix((kernel * weight[j], (i, j)), shape=(len(rest), len(rest))).tocsr()
    total = np.asarray(W.sum(axis=1)).ravel(); covered = total > 1e-12
    smoothed = np.empty_like(raw)
    fallback = (raw * weight).sum(axis=1) / max(weight.sum(), 1e-12)
    for g in range(len(raw)):
        smoothed[g] = np.where(covered, (W @ raw[g]) / np.where(covered, total, 1), fallback[g])
    monotone = np.maximum.accumulate(np.clip(smoothed, 0, 1), axis=0); monotone[0] = 0; monotone[-1] = 1
    metrics = {'phases': int(len(seq)), 'points': int(len(rest)), 'moving_points': int(moving.sum()),
               'uncovered_points': int((~covered).sum()), 'overshoot_samples_clipped': overshoot,
               'travel_scale': scale, 'smoothing_change_max': float(np.abs(smoothed - raw).max()),
               'monotone_change_max': float(np.abs(monotone - np.clip(smoothed, 0, 1)).max())}
    return {'pace': monotone, 'raw_pace': raw, 'travel_weight': weight, 'public_metrics': metrics,
            'objective': 'Projection of each point displacement on its own reference-end chord, travel-weighted Gaussian '
                         'average over reference-pose neighbours, running maximum over phases',
            'limits': 'Timing only: it keeps when the existing motion moves each region, not its path or shape, and no '
                      'guide, contact or appearance qualification. Coupled attachments that follow their own schedule '
                      'need their host to keep that schedule.'}


def path_positions(reference, end, pace, *, pivot=None, axis=None, roll_weight=None):
    """Positions at the given pace on clean paths between two established poses.

    `pace` is one value in [0, 1] per point, or one row per phase. Without a pivot every point lies on its chord,
    (1 - a) reference + a end. With a pivot and an `axis` the path rolls about that hinge line: the angle about the axis
    and the distance from it are interpolated linearly and the position along it moves straight, so a band turning over
    a round obstacle keeps its distance from the hinge between the two end values instead of cutting the chord through
    the obstacle, and gains no sideways drift. With a pivot alone the direction from the pivot is interpolated on the
    sphere (slerp); a great-circle route swings points off the central meridian sideways, so give the hinge axis when
    the motion has one. `roll_weight` (one value in [0, 1] per point, default 1) blends the rolled path with the chord:
    keep chords (0) for material beside the obstacle, where a roll about its centre swings it the wrong way. Pace 0 and
    1 reproduce the two poses exactly.
    """
    rest, final = _pair(reference, end)
    a = _pace(pace, len(rest)); single = a.ndim == 1; a = np.atleast_2d(a)
    chord = (1 - a)[..., None] * rest + a[..., None] * final
    if pivot is None:
        if roll_weight is not None or axis is not None:
            raise ValueError('A roll weight or axis needs a pivot')
        out = chord; deviation = np.zeros(len(a)); inside = 0
    else:
        centre = np.asarray(pivot, float)
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError('A finite 3D pivot is required')
        blend = np.ones(len(rest)) if roll_weight is None else np.asarray(roll_weight, float)
        if blend.shape != (len(rest),) or not np.isfinite(blend).all() or (blend < 0).any() or (blend > 1).any():
            raise ValueError('Roll weights must be one value in [0, 1] per point')
        v0, v1 = rest - centre, final - centre
        if axis is not None:
            k = np.asarray(axis, float)
            if k.shape != (3,) or not np.isfinite(k).all() or np.linalg.norm(k) < 1e-12:
                raise ValueError('A finite nonzero hinge axis is required')
            k = k / np.linalg.norm(k); h0, h1 = v0 @ k, v1 @ k
            p0, p1 = v0 - h0[:, None] * k, v1 - h1[:, None] * k
            r0, r1 = np.linalg.norm(p0, axis=1), np.linalg.norm(p1, axis=1)
            degenerate = (r0 < 1e-12) | (r1 < 1e-12)
            e = p0 / np.where(degenerate, 1, r0)[:, None]; f = np.cross(k, e)
            turn = np.arctan2(np.sum(p1 * f, 1), np.sum(p1 * e, 1))
            if np.any(~degenerate & (np.abs(turn) > np.pi - 1e-6)):
                raise ValueError('A half turn about the axis has no unique roll; move the pivot')
            phi = a * turn; radius = (1 - a) * r0 + a * r1; height = (1 - a) * h0 + a * h1
            rolled = centre + height[..., None] * k + radius[..., None] * (np.cos(phi)[..., None] * e + np.sin(phi)[..., None] * f)
        else:
            r0, r1 = np.linalg.norm(v0, axis=1), np.linalg.norm(v1, axis=1)
            degenerate = (r0 < 1e-12) | (r1 < 1e-12)
            u0 = v0 / np.where(degenerate, 1, r0)[:, None]; u1 = v1 / np.where(degenerate, 1, r1)[:, None]
            omega = np.arccos(np.clip((u0 * u1).sum(1), -1, 1))
            if np.any(~degenerate & (omega > np.pi - 1e-6)):
                raise ValueError('Opposite directions about the pivot have no unique roll; move the pivot')
            so = np.sin(omega); straight = degenerate | (so < 1e-9); s = np.where(straight, 1, so)
            c0 = np.where(straight, 1 - a, np.sin((1 - a) * omega) / s); c1 = np.where(straight, a, np.sin(a * omega) / s)
            d = c0[..., None] * u0 + c1[..., None] * u1; d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-300)
            rolled = centre + ((1 - a) * r0 + a * r1)[..., None] * d
        rolled = np.where(degenerate[None, :, None], chord, rolled)
        out = blend[None, :, None] * rolled + (1 - blend)[None, :, None] * chord
        deviation = np.linalg.norm(out - chord, axis=-1).max(axis=1)
        # chord points that come closer to the pivot (or hinge line) than both end poses: what the roll prevents
        rel = chord - centre
        near = np.linalg.norm(rel - (rel @ k)[..., None] * k, axis=-1) if axis is not None else np.linalg.norm(rel, axis=-1)
        inside = int(np.count_nonzero(near < np.minimum(r0, r1)[None, :] - 1e-12))
    metrics = {'phases': int(len(a)), 'points': int(len(rest)), 'max_deviation_from_chord_per_phase': deviation.tolist(),
               'chord_points_closer_to_pivot_than_both_ends': inside}
    return {'positions': out[0] if single else out, 'public_metrics': metrics,
            'objective': 'Chord interpolation, or a roll about a hinge axis (angle, distance and axial position '
                         'interpolated) or about a pivot (slerp), blended per point',
            'limits': 'Rolling keeps each distance from the hinge between its two end values; it does not know the '
                      "obstacle's actual surface, so check clearance against the real obstacle mesh. No guide, contact "
                      'or appearance qualification.'}


def hinge_motion(reference, end, members, pivot, pace, *, base=None, weights=None):
    """Move a part as one rigid piece about a fixed pivot, then blend it into the surrounding motion.

    The best rotation about `pivot` taking the `members`' reference positions to their end positions (Kabsch with the
    pivot fixed, members weighted by travel) defines an axis and angle. At pace a every point turns by a times that
    angle about the same axis through the pivot and carries a times its own residual (end minus the rotated reference),
    so pace 0 and 1 reproduce both poses exactly and the piece keeps its shape in between as far as its two poses allow.
    `pace` is one value per point, or one row per phase (usually one value for the whole piece). `weights` (one value
    in [0, 1] per point) blend the hinged positions with `base` positions of the same shape (default: the chords): 1
    inside the piece, falling to 0 where it joins material that moves differently.
    """
    rest, final = _pair(reference, end)
    idx = np.asarray(members)
    if idx.ndim != 1 or idx.dtype.kind not in 'iu' or len(idx) < 3 or idx.min() < 0 or idx.max() >= len(rest):
        raise ValueError('At least three member indices of the supplied points are required')
    centre = np.asarray(pivot, float)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError('A finite 3D pivot is required')
    a = _pace(pace, len(rest)); single = a.ndim == 1; a = np.atleast_2d(a)
    X, Y = rest[idx] - centre, final[idx] - centre
    w = np.linalg.norm(final[idx] - rest[idx], axis=1); w = w if w.sum() > 1e-12 else np.ones(len(idx))
    H = (X * w[:, None]).T @ Y; U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.]); R = Vt.T @ D @ U.T
    angle = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle > 1e-12:
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.linalg.norm(axis) < 1e-9:                                   # half turn: axis from the symmetric part
            vals, vecs = np.linalg.eigh(R + R.T); axis = vecs[:, np.argmax(vals)]
        axis = axis / np.linalg.norm(axis)
    else:
        axis = np.array([1., 0., 0.])
    residual = final - (rest - centre) @ R.T - centre
    v = rest - centre; kv = np.cross(axis, v); kd = v @ axis
    out = np.empty((len(a), len(rest), 3))
    for g in range(len(a)):                                                 # Rodrigues turn by pace * angle, per point
        c, s = np.cos(a[g] * angle)[:, None], np.sin(a[g] * angle)[:, None]
        out[g] = v * c + kv * s + axis[None] * kd[:, None] * (1 - c) + centre + a[g][:, None] * residual
    if base is None:
        base_arr = (1 - a)[..., None] * rest + a[..., None] * final
    else:
        base_arr = np.asarray(base, float)
        base_arr = base_arr[None] if base_arr.ndim == 2 else base_arr
        if base_arr.shape != out.shape or not np.isfinite(base_arr).all():
            raise ValueError('Base positions must match the points (and phases)')
    blend = np.ones(len(rest)) if weights is None else np.asarray(weights, float)
    if blend.shape != (len(rest),) or not np.isfinite(blend).all() or (blend < 0).any() or (blend > 1).any():
        raise ValueError('Weights must be one value in [0, 1] per point')
    result = blend[None, :, None] * out + (1 - blend)[None, :, None] * base_arr
    travel = np.linalg.norm(final[idx] - rest[idx], axis=1)
    member_residual = np.linalg.norm(residual[idx], axis=1)
    metrics = {'members': int(len(idx)), 'angle_degrees': float(np.degrees(angle)), 'axis': axis.tolist(),
               'residual_rms': float(np.sqrt(np.mean(member_residual ** 2))), 'residual_max': float(member_residual.max()),
               'member_travel_rms': float(np.sqrt(np.mean(travel ** 2))),
               'rigid_share': float(1 - np.sqrt(np.mean(member_residual ** 2)) / max(np.sqrt(np.mean(travel ** 2)), 1e-300))}
    return {'positions': result[0] if single else result, 'rotation': R, 'residual': residual, 'public_metrics': metrics,
            'objective': 'Travel-weighted best rotation about the fixed pivot; pace-scaled turn plus pace-scaled residual, '
                         'blended with the base motion',
            'limits': 'A piece whose two poses are far from rigid (low rigid_share) is mostly carried by its residual, '
                      'i.e. linearly. The members, pivot and blend weights are owner choices; no guide, contact or '
                      'appearance qualification.'}


def _envelope(directions, obstacle, centre, angular_radius_degrees, envelope):
    """Obstacle envelope distance per unit direction (NaN where uncovered) and a coverage fade in [0, 1]."""
    from scipy.spatial import cKDTree

    if envelope not in ('smooth', 'max'):
        raise ValueError("Envelope must be 'smooth' or 'max'")
    vo = obstacle - centre; ro = np.linalg.norm(vo, axis=1); keep = ro > 1e-12
    uo = vo[keep] / ro[keep, None]; ro = ro[keep]; tree = cKDTree(uo)
    chord = 2 * np.sin(np.radians(angular_radius_degrees) / 2)
    env = np.full(len(directions), np.nan); fade = np.zeros(len(directions))
    if envelope == 'max':
        for k, hits in enumerate(tree.query_ball_point(directions, chord)):
            if hits: env[k] = ro[hits].max(); fade[k] = 1.
        return env, fade
    top = ro.max()

    def sums(u, skip_self=False):
        m = cKDTree(u).sparse_distance_matrix(tree, chord, output_type='coo_matrix')
        w = (1 - (m.data / chord) ** 2) ** 3                               # compact C2 kernel over the cone
        if skip_self: w = np.where(m.row == m.col, 0., w)
        return np.bincount(m.row, w, len(u)), np.bincount(m.row, w * (ro[m.col] / top) ** 16, len(u))

    reference = float(np.median(sums(uo, skip_self=True)[0]))              # the obstacle's own interior coverage
    S, M = sums(directions); ok = S > 1e-12
    env[ok] = top * (M[ok] / S[ok]) ** (1 / 16)                            # weighted power mean: a smooth near-maximum
    t = np.clip(S / max(.25 * reference, 1e-300), 0, 1); fade = t * t * (3 - 2 * t)
    return env, fade


def keep_clearance(positions, obstacle, centre, clearance, *, angular_radius_degrees=2., soft=0., envelope='smooth'):
    """Keep moving points outside a star-shaped obstacle seen from `centre` (an eye from its middle).

    The obstacle's outer envelope in a direction is measured from the obstacle points within `angular_radius_degrees`
    of that direction (isotropic; choose it at least twice the obstacle's vertex spacing as seen from the centre). With
    `envelope='smooth'` (default) it is a weighted power mean (p = 16, a smooth near-maximum) of their distances under a
    compact C2 kernel, so it varies smoothly with direction, and pushes fade out smoothly where the obstacle's coverage
    (kernel sum) falls below a quarter of its interior value, at its edge. `envelope='max'` takes the largest distance,
    which steps whenever an obstacle point enters or leaves the cone: pushes then step across the surface too, which in
    real use rippled lid skin pressed toward an eye. A point closer to the centre than the envelope plus its
    `clearance` (one value per point or a scalar; a negative value leaves the point alone) moves outward along its own
    direction from the centre. With `soft` > 0 the push ramps in smoothly over that width (none `soft` outside the
    limit, full `soft` inside it), so pushed and unpushed neighbours join without a crease. Directions with no obstacle
    point nearby are left alone. `envelope` in the result is the measured envelope distance per point (NaN where
    uncovered); `end_clearance` turns it into the clearance to keep between two established poses.
    """
    pts = np.asarray(positions, float); obs = np.asarray(obstacle, float); c = np.asarray(centre, float)
    single = pts.ndim == 2; P = pts[None] if single else pts
    if P.ndim != 3 or P.shape[-1] != 3 or not np.isfinite(P).all():
        raise ValueError('Finite point positions (optionally per phase) are required')
    if obs.ndim != 2 or obs.shape[1:] != (3,) or len(obs) < 4 or not np.isfinite(obs).all():
        raise ValueError('At least four finite obstacle points are required')
    if c.shape != (3,) or not np.isfinite(c).all():
        raise ValueError('A finite 3D centre is required')
    if not (np.isfinite(angular_radius_degrees) and 0 < angular_radius_degrees <= 30 and np.isfinite(soft) and soft >= 0):
        raise ValueError('Require 0 < angular_radius_degrees <= 30 and soft >= 0')
    need = np.broadcast_to(np.asarray(clearance, float), P.shape[1:2]).copy()
    if not np.isfinite(need).all():
        raise ValueError('Clearance must be finite')
    v = P - c; r = np.linalg.norm(v, axis=-1); u = v / np.maximum(r, 1e-300)[..., None]
    env = np.full(r.shape, np.nan); fade = np.zeros(r.shape)
    for g in range(len(P)):
        env[g], fade[g] = _envelope(u[g], obs, c, angular_radius_degrees, envelope)
    covered = np.isfinite(env) & (need[None] >= 0)
    depth = np.where(covered, np.nan_to_num(env) + need[None] - r, -np.inf)  # > 0: inside the required distance
    if soft > 0:
        push = np.where(depth > soft, depth, np.where(depth > -soft, (depth + soft) ** 2 / (4 * soft), 0.))
    else:
        push = np.maximum(depth, 0.)
    push = push * fade
    out = P + (push / np.maximum(r, 1e-300))[..., None] * v
    metrics = {'phases': int(len(P)), 'points': int(P.shape[1]), 'envelope': envelope,
               'pushed_per_phase': [int(k) for k in (push > 0).sum(axis=1)],
               'max_push_per_phase': [float(m) for m in push.max(axis=1)], 'uncovered_points': int((~np.isfinite(env)).sum())}
    return {'positions': out[0] if single else out, 'push': push[0] if single else push,
            'envelope': env[0] if single else env, 'public_metrics': metrics,
            'objective': 'Radial distance from the centre kept at least the obstacle envelope plus clearance, with an '
                         'optional C1 ramp',
            'limits': 'Star-shaped obstacle seen from the centre; the angular radius is an owner choice; pushes are radial '
                      'and per point, so check the result with section and stretch diagnostics and renders. No guide, '
                      'contact certification or appearance qualification.'}


def end_clearance(reference, end, obstacle_reference, obstacle_end, centre, *, cap=None, angular_radius_degrees=2.,
                  envelope='smooth'):
    """The clearance each point should keep between two established poses, for `keep_clearance`.

    A point's clearance in a pose is its distance from `centre` minus the obstacle envelope in its direction (the
    obstacle sampled in that pose). The requirement is the smaller of its two clearances, capped at `cap`; a point
    inside the envelope or uncovered in either pose gets -1 (left alone). Cap it at the margin that matters (for a lid,
    about the smallest clearance of the rows that ride on the eye): uncapped, material far from the obstacle has to
    keep its whole end distance, so it is pushed wherever its chord dips toward the centre or the envelope estimate
    varies, which in real use wrinkled the skin beside an eye corner that never came near the eye.
    """
    rest, final = _pair(reference, end)
    if cap is not None and not (np.isfinite(cap) and cap >= 0):
        raise ValueError('The cap must be finite and nonnegative')
    k = dict(angular_radius_degrees=angular_radius_degrees, envelope=envelope)
    env0 = keep_clearance(rest, obstacle_reference, centre, -1., **k)['envelope']
    env1 = keep_clearance(final, obstacle_end, centre, -1., **k)['envelope']
    c = np.asarray(centre, float)
    c0 = np.linalg.norm(rest - c, axis=1) - env0; c1 = np.linalg.norm(final - c, axis=1) - env1
    ok = np.isfinite(c0) & np.isfinite(c1) & (c0 > 0) & (c1 > 0)
    need = np.minimum(np.nan_to_num(c0), np.nan_to_num(c1))
    capped = int(np.count_nonzero(ok & (need > cap))) if cap is not None else 0
    if cap is not None: need = np.minimum(need, cap)
    need = np.where(ok, need, -1.)
    metrics = {'points': int(len(rest)), 'required': int(ok.sum()), 'capped': capped, 'cap': cap, 'envelope': envelope,
               'inside_or_uncovered': int((~ok).sum()),
               'required_quantiles': [float(q) for q in np.quantile(need[ok], [.1, .5, .9])] if ok.any() else []}
    return {'clearance': need, 'reference_clearance': c0, 'end_clearance': c1, 'public_metrics': metrics,
            'objective': "Smaller of each point's clearances in its two established poses, capped",
            'limits': 'The cap is an owner choice about the margin that matters; envelopes are measured as in '
                      'keep_clearance. No contact certification or appearance qualification.'}


def smooth_step(values, start, end):
    """0 at or below `start`, 1 at or above `end`, and the C1 smoothstep 3t^2 - 2t^3 between (arrays broadcast).

    Use it for temporal schedules (phases from start to end) and spatial fades (1 - smooth_step(distance, full, zero)).
    A fade narrower than the spacing of the material it crosses acts as a hard edge.
    """
    v, a, b = (np.asarray(x, float) for x in (values, start, end))
    if not (np.isfinite(v).all() and np.isfinite(a).all() and np.isfinite(b).all()) or np.any(b <= a):
        raise ValueError('Finite values with start < end are required')
    t = np.clip((v - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def travel_weight(reference, travel, *, low, high, sigma=0., cutoff=None):
    """How much of a rebuilt motion each point takes, from how far the existing motion moves it.

    `travel` is each point's travel in the existing motion (for example its largest displacement from `reference`
    over the motion). The weight is smooth_step(travel, low, high): 0 for points that barely move (they keep the
    existing motion), 1 for points that clearly move (they take the rebuilt one). Where the travel is small it is noisy,
    and neighbours then alternate between keeping and replacing the motion; any difference between the two motions
    comes out as fine creases. With `sigma` the travel is first averaged over reference-pose neighbours (Gaussian,
    within `cutoff`, default 3 sigma), so the weight varies smoothly across the surface. `neighbour_jump_p99_*`
    compares the weights of neighbours within sigma before and after.
    """
    from scipy.spatial import cKDTree

    rest = np.asarray(reference, float); t = np.asarray(travel, float)
    if (rest.ndim != 2 or rest.shape[1:] != (3,) or t.shape != (len(rest),) or not len(rest)
            or not np.isfinite(rest).all() or not np.isfinite(t).all()):
        raise ValueError('Finite reference positions and one travel value per point required')
    if not (np.isfinite(low) and np.isfinite(high) and high > low) or not (np.isfinite(sigma) and sigma >= 0):
        raise ValueError('Finite low < high and a nonnegative sigma required')
    if cutoff is not None and not (np.isfinite(cutoff) and cutoff > 0):
        raise ValueError('A positive cutoff is required when given')
    raw = smooth_step(t, low, high); smoothed = t.copy(); pairs = None
    if sigma > 0:
        tree = cKDTree(rest)
        for i, nb in enumerate(tree.query_ball_point(rest, 3 * sigma if cutoff is None else float(cutoff))):
            k = np.exp(-.5 * (np.linalg.norm(rest[nb] - rest[i], axis=1) / sigma) ** 2)
            smoothed[i] = (k * t[nb]).sum() / k.sum()
        pairs = tree.query_pairs(sigma, output_type='ndarray')
    weight = smooth_step(smoothed, low, high)

    def jump(w):
        return float(np.quantile(np.abs(w[pairs[:, 0]] - w[pairs[:, 1]]), .99)) if pairs is not None and len(pairs) else None
    metrics = {'points': int(len(rest)), 'weighted_points': int(np.count_nonzero(weight > 0)),
               'full_points': int(np.count_nonzero(weight >= 1)), 'max_change_by_smoothing': float(np.abs(weight - raw).max()),
               'neighbour_jump_p99_before': jump(raw), 'neighbour_jump_p99_after': jump(weight)}
    return {'weight': weight, 'raw_weight': raw, 'smoothed_travel': smoothed, 'public_metrics': metrics,
            'objective': "smooth_step of each point's travel, optionally Gaussian-averaged over reference-pose neighbours",
            'limits': 'The blend of a rebuilt motion into an existing one is only as good as the two motions agree where '
                      'they meet; a smooth weight removes the alternation, not a real difference between them.'}


def _pace_table(pace):
    a = np.asarray(pace, float)
    if a.ndim != 2 or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError('Pace must be (phases, points) values in [0, 1]')
    return a


def shared_schedule(pace, members, *, weights=None, statistic='median'):
    """One schedule for a region: the pace of its member points combined per phase, made monotone.

    `pace` is (phases, points) as from `motion_pace`; `members` indexes (or masks) the points whose timing the region
    should share, for example the part of a moving band next to the region that already moves the way it should.
    'median' takes the per-phase median; 'mean' the per-phase mean weighted by `weights` (one per member, for example
    travel). The result is a running maximum over phases.
    """
    a = _pace_table(pace)
    m = np.asarray(members)
    idx = np.flatnonzero(m) if m.dtype == bool and m.shape == (a.shape[1],) else m.astype(np.int64).ravel()
    if not len(idx) or idx.min() < 0 or idx.max() >= a.shape[1]:
        raise ValueError('Members must select at least one of the points')
    if statistic == 'median':
        common = np.median(a[:, idx], axis=1)
    elif statistic == 'mean':
        w = np.ones(len(idx)) if weights is None else np.asarray(weights, float).ravel()
        if w.shape != (len(idx),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError('Weights must be nonnegative, one per member, not all zero')
        common = a[:, idx] @ w / w.sum()
    else:
        raise ValueError("statistic must be 'median' or 'mean'")
    return np.maximum.accumulate(common)


def schedule_pace(pace, schedule, weights, *, mode='blend'):
    """Re-time a smoothly weighted region of an existing pace.

    This repairs the timing of an inherited motion. It is not a way to construct a moving part: a lid built from
    per-region paces, a corner floored to close first, closes like a zipper. Build a lid as one piece on a hinge with
    one pace (`hinge_landing`, then `path_positions` with one pace for every point).

    `pace` is (phases, points), monotone per point (as from `motion_pace`). `schedule` is a monotone timing in [0, 1]
    per phase, shared (phases,) or per point (phases, points): `shared_schedule` of the region's neighbours,
    `smooth_step(phases, start, end)`, or a per-point onset `smooth_step(phases[:, None], onset - ramp, onset)`.
    `weights` (points,) in [0, 1] are the spatial fade: 1 where the region takes the schedule, 0 where it keeps its own
    pace.

    - 'blend': (1 - w) pace + w schedule. The region moves as one piece on the schedule. Use it where material shears
      because neighbouring parts keep different timings, for example a corner that stays put until late while the part
      beside it is already half way.
    - 'floor': max(pace, w schedule). The region is at least as far as the schedule. Use it where a part of an
      inherited motion must reach its end pose by a given phase without delaying anything already ahead.

    Both keep every point's pace monotone. Points with weight 0 keep their exact values.
    """
    a = _pace_table(pace)
    if np.any(np.diff(a, axis=0) < -1e-12):
        raise ValueError('Pace must be monotone over phases for every point')
    sch = np.asarray(schedule, float)
    if sch.shape == (len(a),):
        sch = np.broadcast_to(sch[:, None], a.shape)
    if sch.shape != a.shape or not np.isfinite(sch).all() or (sch < 0).any() or (sch > 1).any():
        raise ValueError('The schedule must be values in [0, 1] per phase, shared or per point')
    if np.any(np.diff(sch, axis=0) < -1e-12):
        raise ValueError('The schedule must be monotone over phases')
    w = np.asarray(weights, float)
    if w.shape != (a.shape[1],) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError('Weights must be one value in [0, 1] per point')
    if mode == 'blend':
        out = (1 - w[None]) * a + w[None] * sch
    elif mode == 'floor':
        out = np.maximum(a, w[None] * sch)
    else:
        raise ValueError("mode must be 'blend' or 'floor'")
    out = np.where(w[None] > 0, out, a)
    change = np.abs(out - a)
    metrics = {'phases': int(len(a)), 'points': int(a.shape[1]), 'mode': mode,
               'weighted_points': int(np.count_nonzero(w > 0)), 'fully_weighted_points': int(np.count_nonzero(w >= 1)),
               'changed_points': int(np.count_nonzero(change.max(axis=0) > 1e-12)), 'max_change': float(change.max())}
    return {'pace': out, 'public_metrics': metrics,
            'objective': "Weighted blend onto, or floor at, a monotone schedule of each point's pace",
            'limits': 'Timing only: paths, shapes and fitted fields are unchanged, so a field fitted to the old in-between '
                      'positions no longer matches them (re-time after such fits, or refit them). A fade narrower than the '
                      'material spacing acts as a hard edge; where two regions meet, share or fade the schedule across the '
                      'junction. Attachments that follow the host stay in sync only if they move with it. No guide or '
                      'appearance qualification.'}


def _wrap(angle):
    return (np.asarray(angle, float) + np.pi) % (2 * np.pi) - np.pi


def _hinge_frame(pivot, axis, toward):
    """Hinge line through `pivot` along unit `axis`, with a reference direction perpendicular to it pointing toward the
    given material (so the angle's branch cut lies behind the hinge, away from the moving material)."""
    centre, k = np.asarray(pivot, float), np.asarray(axis, float)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise ValueError('A finite 3D pivot is required')
    if k.shape != (3,) or not np.isfinite(k).all() or np.linalg.norm(k) < 1e-12:
        raise ValueError('A finite nonzero hinge axis is required')
    k = k / np.linalg.norm(k)
    d = np.asarray(toward, float).reshape(-1, 3).mean(axis=0) - centre
    d = d - (d @ k) * k
    if np.linalg.norm(d) < 1e-12:
        d = np.cross(k, [1., 0., 0.] if abs(k[0]) < .9 else [0., 1., 0.])
    e1 = d / np.linalg.norm(d)
    return centre, k, e1, np.cross(k, e1)


def _hinge_coordinates(points, frame):
    centre, k, e1, e2 = frame
    v = np.asarray(points, float) - centre; s = v @ k; rad = v - s[..., None] * k
    return s, np.linalg.norm(rad, axis=-1), np.arctan2(rad @ e2, rad @ e1)


def _hinge_place(s, radius, angle, frame):
    centre, k, e1, e2 = frame
    return (centre + np.asarray(s)[..., None] * k + (radius * np.cos(angle))[..., None] * e1
            + (radius * np.sin(angle))[..., None] * e2)


def _off_axis(radius, what):
    if np.any(np.asarray(radius) < 1e-12):
        raise ValueError(f'{what} on the hinge axis has no turn; move the pivot or leave the point out')


def hinge_change(reference, end, pivot, axis):
    """Each point's change between two poses in hinge coordinates about the line through `pivot` along `axis`.

    `turn` is the angle about the axis (radians, right-handed about `axis`, the shorter way round), `radial` the change
    of the point's distance from the axis and `axial` its slide along it. Moving a point by a fraction f of all three is
    a roll about the hinge (as `path_positions` with a pivot and axis) and reproduces both poses at f = 0 and 1. These
    per-point values are what a native rig needs to carry the same motion (`hinge_carry`).
    """
    rest, final = _pair(reference, end)
    frame = _hinge_frame(pivot, axis, rest)
    s0, r0, t0 = _hinge_coordinates(rest, frame); s1, r1, t1 = _hinge_coordinates(final, frame)
    _off_axis(np.minimum(r0, r1), 'A point')
    turn = _wrap(t1 - t0)
    if np.any(np.abs(turn) > np.pi - 1e-6):
        raise ValueError('A half turn about the axis has no unique direction; move the pivot')
    metrics = {'points': int(len(rest)), 'turn_degrees_min': float(np.degrees(turn.min())),
               'turn_degrees_max': float(np.degrees(turn.max())), 'radial_min': float((r1 - r0).min()),
               'radial_max': float((r1 - r0).max()), 'axial_abs_max': float(np.abs(s1 - s0).max())}
    return {'turn': turn, 'radial': r1 - r0, 'axial': s1 - s0, 'public_metrics': metrics,
            'objective': 'Hinge coordinates (axial position, distance from the axis, right-handed angle about it) of both '
                         'poses and their differences',
            'limits': 'Describes the change about the chosen hinge; material that does not turn about it has large '
                      'radial or axial parts. No guide, contact or appearance qualification.'}


def _polyline_distance(points, line):
    """Distance from each point to a polyline through `line` in its given order."""
    a, b = line[:-1], line[1:]; ab = b - a; length2 = np.maximum((ab * ab).sum(1), 1e-300)
    t = np.clip(np.einsum('msk,sk->ms', points[:, None, :] - a[None], ab) / length2[None], 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)


def hinge_landing(positions, edge, landing, pivot, axis, *, weights=None, outside=0.):
    """Close an edge onto a landing curve by turning it about a hinge, carrying a band of material with it.

    `edge` indexes the points of `positions` that must land (a lid's margin); `landing` holds points of the curve they
    close onto (the facing margin where it rests), densely enough to interpolate along the axis. An edge point at axial
    position s turns about the hinge line to the landing curve's angle at the same s and moves to the landing curve's
    distance from the axis there plus `outside`, so it lies just outside the landing curve instead of cutting behind it
    or standing off it. Edge points beyond the landing curve's axial range take its end values (counted in the metrics).
    Every point then turns by its weight times the edge's turn at its own axial position and moves out by its weight
    times the edge's radial change there. `weights` (one per point in [0, 1]; default 1 on the edge, 0 elsewhere) are
    the band that closes with the edge: 1 on the edge and the material behind it (a lid's inner surface), falling off
    with distance from the edge, 0 on what stays (the facing side, the corners). Points keep their axial positions, so
    nothing slides sideways.

    Returns the closed `positions`, each point's full `turn` (radians, right-handed about the axis) and `radial` change,
    and the edge's own values: build the in-betweens as one roll to this pose (`path_positions` with the same pivot and
    axis and one pace for every point) and carry attachments with `hinge_carry`.
    """
    P = np.asarray(positions, float)
    if P.ndim != 2 or P.shape[1:] != (3,) or not len(P) or not np.isfinite(P).all():
        raise ValueError('Finite positions are required')
    idx = np.asarray(edge)
    if (idx.ndim != 1 or idx.dtype.kind not in 'iu' or len(idx) < 2 or len(np.unique(idx)) != len(idx)
            or idx.min() < 0 or idx.max() >= len(P)):
        raise ValueError('At least two distinct edge indices of the supplied points are required')
    L = np.asarray(landing, float)
    if L.ndim != 2 or L.shape[1:] != (3,) or len(L) < 2 or not np.isfinite(L).all():
        raise ValueError('At least two finite landing points are required')
    if not np.isfinite(outside):
        raise ValueError('The outside margin must be finite')
    w = np.zeros(len(P)) if weights is None else np.asarray(weights, float)
    if weights is None:
        w[idx] = 1.
    if w.shape != (len(P),) or not np.isfinite(w).all() or (w < 0).any() or (w > 1).any():
        raise ValueError('Weights must be one value in [0, 1] per point')
    frame = _hinge_frame(pivot, axis, np.r_[P[idx], L])
    s, rho, th = _hinge_coordinates(P, frame); sl, rl, tl = _hinge_coordinates(L, frame)
    _off_axis(rl, 'A landing point'); _off_axis(rho[idx], 'An edge point')
    o = np.argsort(sl, kind='stable'); sl, rl, tl = sl[o], rl[o], np.unwrap(tl[o])
    se, re, te = s[idx], rho[idx], th[idx]
    edge_turn = _wrap(np.interp(se, sl, tl) - te)
    edge_radial = np.interp(se, sl, rl) + float(outside) - re
    if np.any(np.abs(edge_turn) > np.pi - 1e-6):
        raise ValueError('An edge point would turn half way round; move the pivot')
    beyond = int(np.count_nonzero((se < sl[0] - 1e-12) | (se > sl[-1] + 1e-12)))
    oe = np.argsort(se, kind='stable')
    turn = w * np.interp(s, se[oe], edge_turn[oe]); radial = w * np.interp(s, se[oe], edge_radial[oe])
    turn[idx] = w[idx] * edge_turn; radial[idx] = w[idx] * edge_radial
    moved = w > 0
    out = P.copy()
    out[moved] = _hinge_place(s[moved], rho[moved] + radial[moved], th[moved] + turn[moved], frame)
    landed = idx[w[idx] >= 1]
    gap = _polyline_distance(out[landed], L[o]) if len(landed) else np.zeros(0)
    metrics = {'points': int(len(P)), 'edge_points': int(len(idx)), 'moved_points': int(moved.sum()),
               'edge_points_beyond_landing_range': beyond,
               'edge_turn_degrees_min': float(np.degrees(edge_turn.min())),
               'edge_turn_degrees_max': float(np.degrees(edge_turn.max())),
               'edge_radial_min': float(edge_radial.min()), 'edge_radial_max': float(edge_radial.max()),
               'landed_gap_to_landing_line_max': float(gap.max()) if len(gap) else None,
               'landed_gap_to_landing_line_median': float(np.median(gap)) if len(gap) else None}
    return {'positions': out, 'turn': turn, 'radial': radial, 'edge_turn': edge_turn, 'edge_radial': edge_radial,
            'public_metrics': metrics,
            'objective': "Each edge point turned about the hinge onto the landing curve's angle and distance (plus the "
                         'outside margin) at its own axial position; every point takes its weight times the edge values '
                         'at its axial position',
            'limits': 'The landing curve, hinge, outside margin and band weights are construction choices; the landed '
                      'gap is measured to the landing polyline, not against a surface. Build the in-betweens as one roll '
                      'and check clearance against the real obstacle. No guide or appearance qualification.'}


def hinge_carry(attached, host_reference, host_end, pivot, axis, *, fraction=1., host_index=None):
    """Carry attached material rigidly with the hinge motion of the host points it sits on.

    Lashes on a lid margin, a marking or a seam on a moving part: each attached point takes the turn, radial and axial
    change (`hinge_change`) of its host point (`host_index`, default its nearest host point in the reference pose) and
    applies them to its own hinge coordinates. At fraction f it turns by f times the turn about the axis, moves out by f
    times the radial change and along the axis by f times the axial change, so it keeps its seat on the host through the
    motion instead of following separately stored shapes. `fraction` is one number or one value per phase: the same
    fraction the host's in-betweens use (for a one-rate hinge, the geometric phase).

    The per-point `turn`, `radial` and `axial` are what a native rig needs: with s and r the point's axial and radial
    parts about the axis, P(f) = pivot + (s + f axial) axis + R(axis, f turn) r (|r| + f radial) / |r|, R a right-handed
    rotation (Blender's Vector Rotate node, axis-angle type, turns right-handed).
    """
    A = np.asarray(attached, float)
    if A.ndim != 2 or A.shape[1:] != (3,) or not len(A) or not np.isfinite(A).all():
        raise ValueError('Finite attached positions are required')
    host_rest, host_final = _pair(host_reference, host_end)
    if host_index is None:
        from scipy.spatial import cKDTree
        _, j = cKDTree(host_rest).query(A)
    else:
        j = np.asarray(host_index)
        if j.shape != (len(A),) or j.dtype.kind not in 'iu' or j.min() < 0 or j.max() >= len(host_rest):
            raise ValueError('One host index per attached point is required')
    f = np.asarray(fraction, float); single = f.ndim == 0; f = np.atleast_1d(f)
    if f.ndim != 1 or not np.isfinite(f).all():
        raise ValueError('The fraction must be one finite number or one per phase')
    change = hinge_change(host_rest, host_final, pivot, axis)
    frame = _hinge_frame(pivot, axis, host_rest)
    turn, radial, axial = change['turn'][j], change['radial'][j], change['axial'][j]
    s, rho, th = _hinge_coordinates(A, frame)
    _off_axis(rho, 'An attached point')
    out = _hinge_place(s[None] + f[:, None] * axial[None], rho[None] + f[:, None] * radial[None],
                       th[None] + f[:, None] * turn[None], frame)
    seat0 = np.linalg.norm(A - host_rest[j], axis=1)
    carried = _hinge_place(s + axial, rho + radial, th + turn, frame)
    seat1 = np.linalg.norm(carried - host_final[j], axis=1)
    metrics = {'attached_points': int(len(A)), 'host_points_used': int(len(np.unique(j))),
               'seat_distance_max': float(seat0.max()), 'seat_distance_change_max': float(np.abs(seat1 - seat0).max()),
               'turn_degrees_min': float(np.degrees(turn.min())), 'turn_degrees_max': float(np.degrees(turn.max()))}
    return {'positions': out[0] if single else out, 'turn': turn, 'radial': radial, 'axial': axial, 'host_index': j,
            'public_metrics': metrics,
            'objective': "Each attached point moved by the fraction of its host point's turn about the hinge, change of "
                         'distance from it and slide along it',
            'limits': 'Rigid only as far as neighbouring host points turn alike; the seat change measures how far an '
                      'attached point leaves its host by the end pose. The host motion must itself be a roll about the '
                      'same hinge at the same fraction. No contact or appearance qualification.'}
