"""Face construction audit: the checks every face shape, control and combination must pass, from one declaration.

A face ships as single-frame blend shapes. Studied on three commercial avatars, clean faces follow the same rules:
each shape owns one region and leaves the lid margin still (C1); a control moves every vertex along one straight line
and never drives keys gated to a phase of it (C2, C10); the mouth opens by a rigid jaw turn about an axis behind and
above the lips (C3) that the skin follows by a weight falling from the chin (C4); lip shapes die out before a boundary
and not along a topology seam (C5); visemes are mixes of a few base shapes, in the right slots, with PP closing the
lips and FF touching the upper teeth (C6); left and right halves add up to the whole with a hard split for eyes and
brows and a feathered one for the mouth (C7); allowed combinations keep the lips from crossing and the teeth behind
the lips (C8); shapes are stored against the basis and expression clips set every shape (C9). `run_face_audit` measures
a declaration over study extractions (`study_extract.py`, read with `study.load_shapes`) and returns one row per check
and subject. Numbers can reject a construction; they never approve its appearance.
"""
import json
import math
from pathlib import Path

import numpy as np

from .bake import BASES
from .study import jaw_weights, load_shapes, rigid_motion, seam_rings, viseme_mix

LIMITS = {
    'region_outside_share': .05,     # share of a shape's skin motion (summed travel) outside its declared region
    'still_travel_share': .02,       # largest travel of a set that must stay still, share of the shape's largest skin travel
    'path_deviation': .05,           # largest distance of a control's evaluated path from its straight chord, share of travel
    'path_reversals': 0.,            # moving vertices whose progress along their chord falls back by more than 2 %
    'phase_gated_keys': 0.,          # keys a control drives with a weight not proportional to it (bumps, late ramps)
    'jaw_rigid_residual': .01,       # RMS distance of a rigid part from its best rigid turn, share of the mouth width
    'jaw_hinge_position': 0.,        # how much less far behind / above the lips the hinge is than declared (mouth widths)
    'jaw_skin_residual': .25,        # median share of the moving skin's motion below the lips off the jaw turn
    'jaw_weight_monotone': .1,       # largest rise of the binned median jaw weight going away from the chin
    'jaw_drags_upper_face': .1,      # share of the moving skin above the lips following the jaw turn by more than a quarter
    'upper_lip_share_error': .05,    # |upper lip's share of the opening - the declared target|
    'falloff_beyond_boundary': .05,  # largest skin travel farther from the seam than the boundary, share of the largest
    'falloff_on_seam': .05,          # largest travel on a declared topology seam, share of the shape's largest
    'viseme_slot_mapping': 0.,       # viseme slots playing a key named for another slot
    'pp_gap': .01,                   # widest lip gap at PP along the mouth (closest pair at each place), mouth widths
    'ff_gap': .02,                   # closest approach of the lower lip (its seam, or a declared set) to the upper teeth at FF
    'lr_sum_error': .03,             # largest |L + R - both|, share of both's largest travel
    'lr_midline': 0.,                # a hard split's feather beyond .02 mouth widths; a feathered split's shortfall from .1
    'lr_crease': .05,                # largest fall of the left share across the midline going toward the left side
    'lips_cross': .01,               # how far the lips pass through each other, mouth widths (plus a declared press)
    'teeth_behind_lips': 0.,         # how far teeth come in front of the lips, mouth widths
    'single_frame_keys': 0.,         # keys stored against another key instead of the basis
    'clip_full_state': 0.,           # declared expression shapes a clip leaves unset
}
HARD_FEATHER, SOFT_FEATHER = .02, .1          # mouth widths
GATED = .02                                   # a key's weight off proportion by more than this share is phase-gated
VISEME_ORDER = ('sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'ih', 'oh', 'ou')   # VRChat's slots
FIELDS = {'skin', 'objects', 'sets', 'lips', 'unit', 'up', 'forward', 'across', 'shapes', 'controls', 'jaw', 'visemes',
          'pairs', 'teeth', 'combinations', 'clips', 'expression_shapes', 'limits', 'description'}


def _array(spec, base, key):
    if isinstance(spec, dict):
        path = Path(spec['path']); path = path if path.is_absolute() else Path(base or '.') / path
        with np.load(path) as data:
            return np.asarray(data[spec.get('key', key)])
    return np.asarray(spec)


def _unitv(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class _Face:
    """The declaration's objects, vertex sets and frame, with poses from key weights."""

    def __init__(self, d, base, objects):
        self.d = d; self.objects = {}
        for name, spec in d['objects'].items():
            if objects and name in objects:
                self.objects[name] = objects[name]
                continue
            path = Path(spec['path'] if isinstance(spec, dict) else spec)
            self.objects[name] = load_shapes(path if path.is_absolute() else Path(base or '.') / path)
        if d['skin'] not in self.objects:
            raise ValueError('The skin must be one of the objects')
        self.skin = self.objects[d['skin']]; self.P = np.asarray(self.skin['rest'], float); n = len(self.P)
        self.sets = {}
        for name, spec in d.get('sets', {}).items():
            idx = _array(spec, base, name)
            idx = np.flatnonzero(idx) if idx.dtype == bool else idx.astype(np.int64)
            if idx.ndim != 1 or not len(idx) or idx.min() < 0 or idx.max() >= n:
                raise ValueError(f'Set {name!r} needs vertex indices of the skin')
            self.sets[name] = idx
        self.up, self.forward, self.across = (_unitv(d.get(k, v)) for k, v in
                                              (('up', (0., 0., 1.)), ('forward', (0., -1., 0.)), ('across', (1., 0., 0.))))
        known = {k for o in self.objects.values() for k in o['keys']}
        self.known = known
        lips = d.get('lips')
        if lips:
            self.upper, self.lower = self.vertices(lips['upper']), self.vertices(lips['lower'])
            seam = np.r_[self.upper, self.lower]
            self.centre = self.P[seam].mean(0)
            self.unit = float(d.get('unit') or np.ptp(self.P[seam] @ self.across))
            from scipy.spatial import cKDTree
            dist, j = cKDTree(self.P[self.lower]).query(self.P[self.upper])
            ok = dist <= float(lips.get('pair_distance', .05)) * self.unit      # facing points of the two lips
            self.pairs = (self.upper[ok], self.lower[j[ok]])
            self.rest_gap = (self.P[self.pairs[0]] - self.P[self.pairs[1]]) @ self.up / self.unit
        else:
            self.upper = self.lower = None
            self.centre = self.P.mean(0); self.unit = float(d.get('unit') or 1.); self.pairs = None
        if not self.unit > 0:
            raise ValueError('The unit (mouth width) must be positive')

    def gaps(self, Q):
        """Lip gap of each facing pair along `up` in mouth widths (negative: crossed); a pair already crossed at rest
        (rows that interleave) counts from its rest value."""
        up, lo = self.pairs
        return (Q[up] - Q[lo]) @ self.up / self.unit - np.minimum(self.rest_gap, 0.)

    def vertices(self, spec):
        if isinstance(spec, str):
            if spec not in self.sets:
                raise ValueError(f'Unknown set {spec!r}')
            return self.sets[spec]
        return np.asarray(spec, np.int64)

    def key(self, obj, name):
        o = self.objects[obj]
        return np.asarray(o['keys'][name], float) if name in o['keys'] else np.zeros_like(o['rest'], dtype=float)

    def pose(self, obj, weights):
        o = self.objects[obj]; out = np.asarray(o['rest'], float) + 0.
        for name, w in weights.items():
            if name in o['keys']:
                out = out + float(w) * np.asarray(o['keys'][name], float)
        return out

    def need(self, name, where):
        if name not in self.known:
            raise ValueError(f'{where}: no object has a shape named {name!r}')


def validate_face_declaration(declaration):
    d = json.loads(json.dumps(declaration))
    if not isinstance(d, dict) or not d.get('skin') or not isinstance(d.get('objects'), dict):
        raise ValueError('A face declaration needs the skin object and the objects (study extractions)')
    unknown = set(d) - FIELDS
    if unknown:
        raise ValueError(f'Unknown face declaration fields: {sorted(unknown)}')
    for key, value in d.get('limits', {}).items():
        if key not in LIMITS or type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Limit {key!r} must be a known check with a nonnegative bound')
    needs_lips = [k for k in ('jaw', 'visemes', 'combinations') if d.get(k)] + [
        f'shape {k} boundary' for k, v in d.get('shapes', {}).items() if 'boundary' in v]
    if needs_lips and not d.get('lips'):
        raise ValueError(f'{needs_lips[0]} needs the lips (upper and lower seam vertices)')
    for name, c in d.get('controls', {}).items():
        s = np.asarray(c.get('samples', []), float)
        if len(s) < 3 or s[0] != 0 or np.any(np.diff(s) <= 0) or not all(
                len(w) == len(s) for w in c.get('weights', {}).values()) or not c.get('weights'):
            raise ValueError(f'Control {name!r} needs rising samples from 0 and one weight per sample for each key')
        for key, driver in c.get('correctives', {}).items():
            if key not in c['weights'] or driver not in set(BASES) - {'main'}:
                raise ValueError(f'Control {name!r}: a corrective is one of its keys with a bake driver '
                                 f'({", ".join(sorted(set(BASES) - {"main"}))})')
    for pair in d.get('pairs', []):
        if pair.get('split') not in ('hard', 'feathered') or not all(pair.get(k) for k in ('both', 'left', 'right')):
            raise ValueError("Each left/right pair needs both, left, right and split 'hard' or 'feathered'")
    return d


def _row(key, subject, observed, detail, limits, extra=0.):
    limit = limits[key] + extra
    status = 'unknown' if observed is None else ('pass' if observed <= limit + 1e-12 else 'fail')
    return {'id': key, 'subject': subject, 'observed': observed, 'limit': limit, 'status': status, 'detail': detail}


def _regions(F, limits):
    rows = []
    for name, spec in F.d.get('shapes', {}).items():
        F.need(name, 'shapes'); lim = {**limits, **spec.get('limits', {})}
        travel = np.linalg.norm(F.key(F.d['skin'], name), axis=1); top = float(travel.max())
        if 'region' in spec:
            inside = np.zeros(len(travel), bool); inside[F.vertices(spec['region'])] = True
            share = float(travel[~inside].sum() / travel.sum()) if top > 0 else None
            rows.append(_row('region_outside_share', name, share, {'largest_outside': float(travel[~inside].max() / top)
                                                                   if top > 0 else None}, lim))
        for still in spec.get('still', []):
            moved = float(travel[F.vertices(still)].max()) if top > 0 else None
            rows.append(_row('still_travel_share', f'{name} / {still}', None if moved is None else moved / top,
                             {'largest_travel': moved, 'shape_largest_travel': top}, lim))
        if 'boundary' in spec or spec.get('seams'):
            from scipy.spatial import cKDTree
            seam = np.r_[F.upper, F.lower] if F.upper is not None else None
            if 'boundary' in spec:
                distance = cKDTree(F.P[seam]).query(F.P)[0] / F.unit
                far = distance > float(spec['boundary'])
                ring = seam_rings(_edges(F.skin), seam, len(F.P), limit=24)
                per_ring = [float(travel[ring == k].max() / top) if top > 0 and np.any(ring == k) else None
                            for k in range(int(np.nanmax(ring)) + 1)] if np.isfinite(ring).any() else []
                reach = float(distance[travel > .05 * top].max()) if top > 0 else None
                rows.append(_row('falloff_beyond_boundary', name, float(travel[far].max() / top) if top > 0 and far.any()
                                 else (0. if top > 0 else None),
                                 {'boundary': float(spec['boundary']), 'reach_5_percent': reach,
                                  'largest_per_ring_from_seam': per_ring}, lim))
            for seam_set in spec.get('seams', []):
                on = float(travel[F.vertices(seam_set)].max() / top) if top > 0 else None
                rows.append(_row('falloff_on_seam', f'{name} / {seam_set}', on, {}, lim))
    return rows


def _edges(obj):
    if len(obj.get('edges', [])):
        return np.asarray(obj['edges'], np.int64)
    pairs = [(f[i], f[(i + 1) % len(f)]) for f in obj['polygons'] for i in range(len(f))]
    return np.unique(np.sort(np.asarray(pairs, np.int64), axis=1), axis=0)


def _controls(F, limits):
    rows = []
    for name, c in F.d.get('controls', {}).items():
        s = np.asarray(c['samples'], float); weights = {k: np.asarray(w, float) for k, w in c['weights'].items()}
        for k in weights:
            F.need(k, f'control {name}')
        declared = c.get('correctives', {})          # baked correctives: bumps of the main weight (bake.BASES)
        gated = {}
        for k, w in weights.items():
            if k in declared:
                expected = BASES[declared[k]](s); off = np.abs(w - expected).max()
                if off > GATED * max(float(np.abs(expected).max()), 1e-300):
                    gated[k] = {'declared_driver': declared[k], 'largest_off_driver': float(off)}
                continue
            end = float(np.interp(1., s, w)) if s[-1] >= 1 else float(w[-1] / s[-1])
            off = np.abs(w - end * s).max()
            if off > GATED * max(abs(end), float(np.abs(w).max()), 1e-300):
                gated[k] = {'weight_at_full': end, 'largest_off_proportion': float(off)}
        pose = lambda chosen: np.stack([np.concatenate([F.pose(obj, {k: w[i] for k, w in chosen.items()})
                                                        for obj in F.objects]) for i in range(len(s))])
        full = _paths(pose(weights))
        straight = _paths(pose({k: w for k, w in weights.items() if k not in declared})) if declared else full
        detail = {'moving_vertices': straight['movers'], 'share_over_10_percent': straight['share_over_10_percent'],
                  'overshoot': straight['overshoot']}
        if declared:          # the arc a declared hinge corrective carries is the construction: reported, not gated
            detail.update(measured_without=sorted(declared), arc_with_correctives=full['deviation'],
                          arc_share_over_10_percent=full['share_over_10_percent'])
        rows.append(_row('path_deviation', name, straight['deviation'], detail, limits))
        rows.append(_row('path_reversals', name, full['reversals'], {}, limits))
        rows.append(_row('phase_gated_keys', name, float(len(gated)), {'keys': gated}, limits))
    return rows


def _paths(X):
    """Distance of each moving vertex's path (poses X: samples, vertices, 3) from its straight chord, share of its
    travel, and vertices whose progress along the chord falls back by more than 2 %."""
    chord = X[-1] - X[0]; length = np.linalg.norm(chord, axis=1)
    movers = length > .2 * max(float(length.max()), 1e-300)
    if not movers.any():
        return {'deviation': 0., 'reversals': 0., 'movers': 0, 'share_over_10_percent': 0., 'overshoot': 0.}
    off = X[:, movers] - X[0, movers]; c = chord[movers]; L = length[movers]
    progress = np.einsum('gnk,nk->gn', off, c) / L ** 2
    deviation = (np.linalg.norm(off - progress[..., None] * c[None], axis=2) / L).max(axis=0)
    fall = (progress[:-1] - progress[1:]).max(axis=0)
    return {'deviation': float(deviation.max()), 'reversals': float(np.count_nonzero(fall > .02)),
            'movers': int(movers.sum()), 'share_over_10_percent': float(np.mean(deviation > .1)),
            'overshoot': float(progress.max() - 1)}


def _jaw(F, limits):
    j = F.d.get('jaw')
    if not j:
        return [], None
    F.need(j['shape'], 'jaw'); rows = []; fits = {}
    for obj in j.get('rigid', []):
        X = np.asarray(F.objects[obj]['rest'], float); fit = rigid_motion(X, X + F.key(obj, j['shape'])); fits[obj] = fit
        rows.append(_row('jaw_rigid_residual', obj, fit['residual_rms'] / F.unit,
                         {'angle_degrees': fit['angle_degrees']}, limits))
    report = {}
    first = fits[j['rigid'][0]] if j.get('rigid') else None
    if first is not None:
        report['angle_degrees'] = first['angle_degrees']
        if first['hinge_point'] is None:
            rows.append(_row('jaw_hinge_position', 'hinge', None, {'reason': 'the part does not turn'}, limits))
        else:
            h = first['hinge_point'] - F.centre
            behind, above = float(-h @ F.forward / F.unit), float(h @ F.up / F.unit)
            need_b, need_a = float(j.get('min_behind', 1.5)), float(j.get('min_above', 0.))
            X = np.asarray(F.objects[j['rigid'][0]]['rest'], float) - first['hinge_point']
            radius = float(np.linalg.norm(X - np.outer(X @ first['axis'], first['axis']), axis=1).mean())
            report.update(behind=behind, above=above, chord_sag_at_half=radius * (1 - math.cos(math.radians(
                first['angle_degrees']) / 2)) / F.unit)
            rows.append(_row('jaw_hinge_position', 'hinge', max(0., need_b - behind, need_a - above),
                             {'behind': behind, 'above': above, 'min_behind': need_b, 'min_above': need_a}, limits))
        D = F.key(F.d['skin'], j['shape']); travel = np.linalg.norm(D, axis=1)
        w, off = jaw_weights(F.P, D, first['rotation'], first['translation'])
        height = (F.P - F.centre) @ F.up / F.unit
        surface = np.zeros(len(F.P), bool)                    # the outer skin (a mouth bag or teeth in the same mesh
        surface[F.vertices(j['surface']) if j.get('surface') is not None else np.arange(len(F.P))] = True   # left out)
        moving = surface & (travel > .05 * travel.max()); big = travel > .2 * travel.max()
        below, above_ = moving & (height < -.02), moving & (height > .02)
        rows.append(_row('jaw_skin_residual', j['shape'], float(np.median(off[below & big])) if (below & big).any()
                         else None, {'weight_median_below': float(np.median(w[below])) if below.any() else None}, limits))
        chin = int(j['chin']) if j.get('chin') is not None else int(np.flatnonzero(below)[np.argmax(travel[below])])
        dist = np.linalg.norm(F.P - F.P[chin], axis=1)[below]; wb = w[below]
        edges = np.quantile(dist, np.linspace(0, 1, 7)) if len(dist) >= 12 else None
        medians = [float(np.median(wb[(dist >= a) & (dist <= b)])) for a, b in zip(edges[:-1], edges[1:])] \
            if edges is not None else []
        rise = max([b - a for a, b in zip(medians, medians[1:])], default=0.)
        rows.append(_row('jaw_weight_monotone', j['shape'], max(0., rise) if medians else None,
                         {'median_weight_by_distance_from_chin': medians, 'chin': chin}, limits))
        rows.append(_row('jaw_drags_upper_face', j['shape'], float(np.mean(w[above_] > .25)) if above_.any() else 0.,
                         {'moving_above': int(above_.sum())}, limits))
        up = float((D[F.upper] @ F.up).max()); down = float(-(D[F.lower] @ F.up).min())
        share = up / (up + down) if up + down > 0 else None
        report['upper_lip_share'] = share
        if j.get('upper_lip_share') is not None:
            rows.append(_row('upper_lip_share_error', j['shape'], None if share is None else
                             abs(share - float(j['upper_lip_share'])), {'share': share,
                                                                        'target': float(j['upper_lip_share'])}, limits))
    return rows, report


def _slot(name):
    n = name.lower()
    n = n[4:] if n.startswith('vrc.') else n
    return n[2:] if n.startswith('v_') else n


def _visemes(F, limits):
    v = F.d.get('visemes')
    if not v:
        return [], None
    rows = []; report = {}
    slots = v.get('slots', {})
    if isinstance(slots, list):                   # the descriptor's list, in VRChat's slot order
        if len(slots) != len(VISEME_ORDER):
            raise ValueError(f'A viseme slot list has the {len(VISEME_ORDER)} slots {VISEME_ORDER}')
        slots = dict(zip(VISEME_ORDER, slots))
    wrong = {slot: key for slot, key in slots.items() if _slot(key) != slot.lower()}
    if slots:
        rows.append(_row('viseme_slot_mapping', 'slots', float(len(wrong)), {'mismatched': wrong}, limits))
    if v.get('basis'):
        for k in v['basis']:
            F.need(k, 'viseme basis')
        allk = lambda k: np.concatenate([F.key(o, k) for o in F.objects])
        basis = {k: allk(k) for k in v['basis']}
        report['mixes'] = {slot: viseme_mix(basis, allk(key)) for slot, key in slots.items() if key in F.known}
    if v.get('pp'):
        F.need(v['pp'], 'PP')
        gap = F.gaps(F.pose(F.d['skin'], {v['pp']: 1.}))            # the closest pair at each place along the mouth
        place = np.round((F.P[F.pairs[0]] - F.centre) @ F.across / F.unit / .05).astype(int)
        closest = {int(b): float(gap[place == b].min()) for b in np.unique(place)}
        open_at = max(closest, key=closest.get)
        rows.append(_row('pp_gap', v['pp'], closest[open_at], {'smallest_gap': float(gap.min()),
                                                               'widest_at': open_at * .05}, limits))
    if v.get('ff'):
        F.need(v['ff'], 'FF')
        from scipy.spatial import cKDTree
        lower_lip = F.vertices(v['lower_lip']) if v.get('lower_lip') is not None else F.lower
        lip = F.pose(F.d['skin'], {v['ff']: 1.})[lower_lip]; teeth = F.pose(v['upper_teeth'], {v['ff']: 1.})
        rows.append(_row('ff_gap', v['ff'], float(cKDTree(teeth).query(lip)[0].min() / F.unit), {}, limits))
    return rows, report


def _pairs(F, limits):
    rows = []
    for pair in F.d.get('pairs', []):
        for k in ('both', 'left', 'right'):
            F.need(pair[k], 'pair')
        B, L, R = (F.key(F.d['skin'], pair[k]) for k in ('both', 'left', 'right'))
        subject = pair['both']
        top = float(np.linalg.norm(B, axis=1).max())
        rows.append(_row('lr_sum_error', subject, float(np.linalg.norm(L + R - B, axis=1).max() / top) if top > 0
                         else None, {}, limits))
        m = np.linalg.norm(L + R, axis=1); s = m > .1 * m.max()
        l, r = np.linalg.norm(L[s], axis=1), np.linalg.norm(R[s], axis=1)
        share = l / np.maximum(l + r, 1e-300); x = (F.P[s] - F.centre) @ F.across / F.unit
        if np.corrcoef(x, share)[0, 1] < 0:
            x = -x                                                        # left toward +x from here on
        mid = (share > .1) & (share < .9)
        if pair['split'] == 'hard':                   # vertices sharing the shape: only on the midline itself
            width = float(np.ptp(x[mid])) if mid.any() else 0.
            rows.append(_row('lr_midline', subject, max(0., width - HARD_FEATHER), {
                'split': 'hard', 'feather_width': width, 'feather_vertices': int(mid.sum())}, limits))
            continue
        profile = _share_profile(x, share)            # (x, median left share) in bins of .01 mouth widths
        width = _crossing(profile, .9) - _crossing(profile, .1) if len(profile) > 1 else 0.
        rows.append(_row('lr_midline', subject, max(0., SOFT_FEATHER - width), {
            'split': 'feathered', 'feather_width': width, 'feather_vertices': int(mid.sum())}, limits))
        span = [(px, ps) for px, ps in profile if abs(px) <= max(width, SOFT_FEATHER)]
        fall = max([a[1] - b[1] for a, b in zip(span, span[1:])], default=0.)
        rows.append(_row('lr_crease', subject, max(0., fall), {'left_share_across_midline': span}, limits))
    return rows


def _share_profile(x, share, step=.01):
    bins = np.round(x / step).astype(int)
    return [(float(b * step), float(np.median(share[bins == b]))) for b in np.unique(bins)]


def _crossing(profile, level):
    """Where the left share first reaches `level`, interpolated between bins."""
    for (x0, s0), (x1, s1) in zip(profile, profile[1:]):
        if s0 < level <= s1:
            return x0 + (level - s0) / (s1 - s0) * (x1 - x0)
    return profile[0][0] if profile[0][1] >= level else profile[-1][0]


def _combinations(F, limits):
    rows = []
    teeth = F.d.get('teeth', [])
    for combo in F.d.get('combinations', []):
        for k in combo['weights']:
            F.need(k, f"combination {combo['name']}")
        Q = F.pose(F.d['skin'], combo['weights']); gap = F.gaps(Q)
        rows.append(_row('lips_cross', combo['name'], max(0., float(-gap.min())), {'smallest_gap': float(gap.min())},
                         limits, extra=float(combo.get('press', 0.))))
        if teeth:                        # against the front of the skin over each tooth point, where skin covers it
            from scipy.spatial import cKDTree
            view = lambda X: np.c_[X @ F.across, X @ F.up]
            around = np.flatnonzero(np.linalg.norm(F.P - F.centre, axis=1) <= .6 * F.unit)
            tree = cKDTree(view(Q[around])); front = (Q[around] - F.centre) @ F.forward; worst = None
            for obj in teeth:
                T = F.pose(obj, combo['weights'])
                for point, near in zip(T, tree.query_ball_point(view(T), .1 * F.unit)):
                    if near:
                        ahead = float(((point - F.centre) @ F.forward - front[near].max()) / F.unit)
                        worst = ahead if worst is None else max(worst, ahead)
            rows.append(_row('teeth_behind_lips', combo['name'], None if worst is None else max(0., worst),
                             {'closest_to_the_lips': worst}, limits))
    return rows


def _production(F, limits):
    rows = []
    for name, obj in F.objects.items():
        info = obj.get('key_info') or {}
        if not info:
            continue
        ordered = sorted(info.values(), key=lambda k: k['index'])
        shapes = {k['name'] for k in ordered[1:] if k.get('relative_key') not in (None, k['name'])}   # not the basis
        stacked = [k['name'] for k in ordered if k['name'] in shapes and k['relative_key'] in shapes]
        rows.append(_row('single_frame_keys', name, float(len(stacked)), {'relative_to_another_key': stacked}, limits))
    shapes = F.d.get('expression_shapes', [])
    for clip in F.d.get('clips', []):
        missing = [k for k in shapes if k not in clip['weights']]
        rows.append(_row('clip_full_state', clip['name'], float(len(missing)), {'unset': missing}, limits))
    return rows


def run_face_audit(declaration, *, base=None, objects=None):
    """Audit a face construction. `declaration` is a mapping or a JSON path (`base` defaults to its folder);
    `objects` optionally supplies loaded extractions by name ({'rest', 'keys', 'edges', 'polygons', 'key_info'}).
    Returns `checks` (id, subject, observed, limit, status, detail), the overall `status` (fail if any row fails,
    unknown if any could not be measured), the `unit` (mouth width) and a `report` (jaw angle, hinge position, chord
    sag, upper-lip share, viseme mixes)."""
    if isinstance(declaration, (str, Path)):
        base = base or Path(declaration).parent
        declaration = json.loads(Path(declaration).read_text(encoding='utf-8'))
    d = validate_face_declaration(declaration)
    limits = {**LIMITS, **d.get('limits', {})}
    F = _Face(d, base, objects)
    rows = _regions(F, limits) + _controls(F, limits)
    jaw_rows, jaw = _jaw(F, limits); vis_rows, vis = _visemes(F, limits)
    rows += jaw_rows + vis_rows + _pairs(F, limits) + _combinations(F, limits) + _production(F, limits)
    status = 'fail' if any(r['status'] == 'fail' for r in rows) else (
        'unknown' if any(r['status'] == 'unknown' for r in rows) else 'pass')
    return {'status': status, 'checks': rows, 'unit': F.unit, 'report': {'jaw': jaw, 'visemes': vis},
            'objective': 'Face construction audit: regions, straight single-control paths, a jaw that turns with '
                         'the skin following by a weight, bounded falloff, visemes from a basis, left/right splits, '
                         'combinations and production form',
            'limits': 'Numbers can reject a construction, never approve its appearance. Paths are sampled at the '
                      'declared control values; region and seam sets are the declaration\'s.'}
