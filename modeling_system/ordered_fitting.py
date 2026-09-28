"""Caller-declared source cells inside the coupled surface fit (no native IO)."""
import numpy as np
from .source_coverage import _edges, _ids, _area, _polygon_area, _digest
from .triangle_contact import _cross, _overlap


def _disk(tri, uv, name):
    """A consistently wound manifold disk with a simple boundary in this UV map."""
    boundary, bad, owners = _edges(tri)
    vertices = np.unique(tri)
    if (bad or not boundary or len(vertices) - len(owners) + len(tri) != 1
            or len(np.unique(np.sort(tri, axis=1), axis=0)) != len(tri)):
        raise ValueError(name + ' must be a wound manifold disk')
    graph = {int(v): set() for v in vertices}
    for a, b in owners:
        graph[a].add(b); graph[b].add(a)
    reached, pending = set(), [int(vertices[0])]
    while pending:
        v = pending.pop()
        if v not in reached:
            reached.add(v); pending.extend(graph[v] - reached)
    if len(reached) != len(vertices):
        raise ValueError(name + ' must be connected')
    boundary_vertices = set(v for e in boundary for v in e)
    # Edge manifoldness alone permits pinched vertex links.
    for v in vertices:
        link = {}
        for face in tri[np.any(tri == v, axis=1)]:
            a, b = face[face != v]
            link.setdefault(int(a), set()).add(int(b)); link.setdefault(int(b), set()).add(int(a))
        seen, queue = set(), [next(iter(link))]
        while queue:
            a = queue.pop()
            if a not in seen:
                seen.add(a); queue.extend(link[a] - seen)
        degrees = [len(a) for a in link.values()]
        if (len(seen) != len(link) or any(d not in (1, 2) for d in degrees)
                or degrees.count(1) != (2 if v in boundary_vertices else 0)):
            raise ValueError(name + ' has a nonmanifold vertex link')
    edge_list = sorted(boundary)
    for k, (a, b) in enumerate(edge_list):
        p, q = uv[a], uv[b]; d = q - p
        if np.linalg.norm(d) <= 1e-12:
            raise ValueError(name + ' has a collapsed boundary edge')
        for c, e in edge_list[k+1:]:
            r, s = uv[c], uv[e]
            common = set((a, b)) & set((c, e))
            if common:
                v = next(iter(common)); other = b if a == v else a; other2 = e if c == v else c
                u, w = uv[other]-uv[v], uv[other2]-uv[v]
                if abs(_cross(u, w)) <= 1e-12 and u @ w > 0:
                    raise ValueError(name + ' has a backtracking boundary')
                continue
            cross = _cross(d, s-r)
            if abs(cross) > 1e-12:
                t, h = _cross(r-p, s-r)/cross, _cross(r-p, d)/cross
                contact = -1e-12 <= t <= 1+1e-12 and -1e-12 <= h <= 1+1e-12
            else:
                contact = (abs(_cross(r-p, d)) <= 1e-12 and
                    max(min((r-p) @ d, (s-p) @ d), 0) <= min(max((r-p) @ d, (s-p) @ d), d @ d)+1e-12)
            if contact:
                raise ValueError(name + ' has a crossing or touching boundary')
    return np.array(sorted(boundary_vertices), dtype=np.int64)


def _no_overlap(corners, name):
    lo, hi = corners.min(1), corners.max(1)
    pairs = 0
    for i, face in enumerate(corners):
        near = np.flatnonzero(np.all(hi[i] >= lo, axis=1) & np.all(hi >= lo[i], axis=1))
        for j in near[near > i]:
            pairs += 1
            if pairs > 200000:
                raise ValueError(name + ' qualification exceeds 200000 pairs')
            if _polygon_area(_overlap(face, corners[j], 1e-12)[0]) > 1e-12:
                raise ValueError(name + ' has overlapping chart faces')


def _safe_fraction(a, b, c):
    """First root of c+b*t+a*t*t on (0,1], staying in the feasible component."""
    result = 1.
    for aa, bb, cc in zip(a, b, c):
        if abs(aa) < 1e-15:
            roots = [-cc / bb] if bb < 0 else []
        else:
            discriminant = bb*bb - 4*aa*cc
            if discriminant < 0:
                roots = []
            else:
                # Stable quadratic formula, including a zero constant.
                q = -.5 * (bb + np.copysign(np.sqrt(discriminant), bb))
                roots = [q/aa, cc/q] if q else [0.]
        for root in roots:
            if 0 < root <= result:
                result = max(0., .99 * float(root))
        if cc <= 0 and (bb < 0 or (abs(bb) < 1e-15 and aa < 0)):
            result = 0.
    return result


class OrderedFit:
    """Affine barycentric cells plus nonlinear connected-face orientation constraints.

    The sparse XYZ objective remains the caller's existing coupled ARAP global
    step. SLSQP changes its search direction under the constraints; analytic
    area roots additionally keep every accepted straight advancement feasible.
    """
    def __init__(self, charts, rest, start, triangles, used, free, held, source, source_triangles,
                 labels, assignment, qualified, low, high, weight, tolerance, max_iterations):
        from scipy.sparse import coo_matrix
        if not isinstance(charts, (list, tuple)) or not charts:
            raise ValueError('ordered_charts requires a nonempty list of explicit chart declarations')
        if type(max_iterations) is not int or not 1 <= max_iterations <= 1000:
            raise ValueError('ordered_max_iterations must be an integer in [1, 1000]')
        self.max_iterations = max_iterations
        self.charts, self.reasons, self.steps = [], [], []
        self.blocked = False
        self.initial = np.asarray(start, float)[used].copy()
        self.used, self.free = used, free
        self.triangles = np.asarray(triangles)
        self.scale = float(np.linalg.norm(np.ptp(rest[used], axis=0)))
        self.tolerance = tolerance
        self.owner = np.full(len(start), -1, int)
        loc = np.full(len(start), -1, int); loc[used] = np.arange(len(used))
        n = len(start)
        for index, spec in enumerate(charts):
            required = {'support_triangle_ids', 'source_uv', 'candidate_triangle_ids', 'vertex_source_triangles',
                        'initial_barycentric', 'minimum_area_ratio', 'boundary_mode'}
            if not isinstance(spec, dict) or set(spec) != required or spec['boundary_mode'] != 'fixed':
                raise ValueError('Each ordered chart needs exactly the documented fields and boundary_mode="fixed"')
            st = _ids(spec['support_triangle_ids'], len(source_triangles), 'support_triangle_ids')
            ct = _ids(spec['candidate_triangle_ids'], len(triangles), 'candidate_triangle_ids')
            if len(st) > 4096 or len(ct) > 4096 or len(st)*len(ct) > 2000000:
                raise ValueError('Select a smaller ordered chart (4096 faces, 2000000 containment pairs)')
            vertices = np.unique(triangles[ct]); src_vertices = np.unique(source_triangles[st])
            if np.any(self.owner[vertices] >= 0):
                raise ValueError('Ordered charts may not share candidate vertices; declare one chart across a seam')
            self.owner[vertices] = index
            suv = np.asarray(spec['source_uv']); refs = np.asarray(spec['vertex_source_triangles'])
            bary = np.asarray(spec['initial_barycentric'])
            ratio = spec['minimum_area_ratio']
            if (suv.shape != (len(source), 2) or suv.dtype.kind not in 'fiu' or not np.isfinite(suv).all()
                    or refs.shape != (n,) or refs.dtype.kind not in 'iu'
                    or bary.shape != (n, 3) or bary.dtype.kind not in 'fiu' or not np.isfinite(bary).all()
                    or not np.isin(refs[vertices], st).all()
                    or type(ratio) not in (int, float) or not np.isfinite(ratio) or not 1e-6 < ratio < 1):
                raise ValueError('Finite source UV, explicit per-vertex source cells/barycentrics and area ratio in (1e-6,1) required')
            domain = np.unique(labels[st])
            if (len(domain) != 1 or domain[0] not in qualified or np.any(assignment[vertices] != domain[0])
                    or np.any(weight[vertices] <= 0)):
                raise ValueError('Every ordered vertex, including boundary/held, needs the same qualified support owner and positive weight')
            if np.any(low[vertices] > 0) or np.any(high[vertices] < 0):
                self.reasons.append({'chart': index, 'reason': 'source-cell mode requires the normal band to include zero'})
            b = bary[vertices].astype(float)
            if np.any(b < 0) or np.any(b > 1) or np.any(abs(b.sum(1)-1) > 1e-12):
                raise ValueError('Initial barycentric coordinates must lie inside their declared source cells and sum to one')
            b /= b.sum(1)[:, None]
            uv_origin = suv[src_vertices].min(0)
            uv_scale = float(np.linalg.norm(np.ptp(suv[src_vertices], axis=0)))
            if not np.isfinite(uv_scale) or uv_scale <= 0:
                raise ValueError('Nonzero finite source chart extent required')
            uv_source = (suv.astype(float)-uv_origin)/uv_scale
            source_faces = uv_source[source_triangles[st]]
            source_area = _area(source_faces)
            if np.any(abs(source_area) <= 1e-12) or np.any(np.sign(source_area) != np.sign(source_area[0])):
                raise ValueError('Source chart must have nondegenerate consistently oriented faces')
            _disk(source_triangles[st], uv_source, 'Source chart')
            _no_overlap(source_faces, 'Source chart')
            cells = source_triangles[refs[vertices]]
            uv = np.zeros((n, 2)); uv[vertices] = np.einsum('nk,nkd->nd', b, uv_source[cells])
            boundary = _disk(triangles[ct], uv, 'Candidate chart')
            areas = _area(uv[triangles[ct]]) * np.sign(source_area[0])
            if np.any(areas <= 1e-12):
                self.reasons.append({'chart': index, 'reason': 'initial connected correspondence is reversed or collapsed',
                                     'triangle_ids': ct[areas <= 1e-12].tolist()})
            else:
                _no_overlap(uv[triangles[ct]], 'Candidate chart')
                lo, hi = source_faces.min(1), source_faces.max(1)
                for face_id, face in zip(ct, uv[triangles[ct]]):
                    near = np.flatnonzero(np.all(hi >= face.min(0), axis=1) & np.all(lo <= face.max(0), axis=1))
                    covered = sum(_polygon_area(_overlap(face, source_faces[j], 1e-12)[0]) for j in near)
                    if abs(covered-abs(float(_area(face[None])[0]))) > 1e-10:
                        self.reasons.append({'chart': index, 'reason': 'initial connected face leaves the source chart',
                                             'triangle_id': int(face_id)})
                        break
            lifted = np.einsum('nk,nkd->nd', b, source[cells])
            error = np.linalg.norm(start[vertices]-lifted, axis=1)
            if np.any(error > tolerance):
                self.reasons.append({'chart': index, 'reason': 'initial positions disagree with declared source correspondence',
                    'vertices': vertices[error > tolerance].tolist(), 'maximum_distance': float(error.max()),
                    'held_vertices': vertices[(error > tolerance) & held[vertices].astype(bool)].tolist()})
            mutable = ~(held[vertices].astype(bool) | np.isin(vertices, boundary))
            self.charts.append(dict(vertices=vertices, ct=ct, uv=uv, cells=cells, bary=b, refs=refs[vertices].copy(),
                uv_source=uv_source, uv_origin=uv_origin, uv_scale=uv_scale, sign=float(np.sign(source_area[0])),
                area=areas, ratio=float(ratio), boundary=boundary, mutable=mutable, maximum_initial_distance=float(error.max()),
                hashes={k: _digest(spec[k]) for k in required if k not in ('minimum_area_ratio', 'boundary_mode')}))
        rows, columns, values, lower, upper = [], [], [], [], []
        free_lookup = {int(used[f]): k for k, f in enumerate(free)}
        self.variables = 0
        for chart in self.charts:
            chart['columns'] = np.full((len(chart['vertices']), 2), -1, int)
            for k in np.flatnonzero(chart['mutable']):
                v = chart['vertices'][k]; col = self.variables; self.variables += 2
                chart['columns'][k] = (col, col+1)
                cell = source[chart['cells'][k]]
                jac = (cell[1:] - cell[0]).T
                for a in range(3):
                    for b in range(2):
                        rows.append(3*free_lookup[int(v)]+a); columns.append(col+b); values.append(jac[a, b])
                lower.extend(-chart['bary'][k, 1:]); upper.extend(1-chart['bary'][k, 1:])
        for k, f in enumerate(free):
            if self.owner[used[f]] < 0:
                for a in range(3):
                    rows.append(3*k+a); columns.append(self.variables); values.append(self.scale)
                    self.variables += 1; lower.append(-np.inf); upper.append(np.inf)
        if self.variables > 1500:
            raise ValueError('Ordered fitting is bounded to 1500 variables; select a smaller coupled patch')
        self.map = coo_matrix((values, (rows, columns)), shape=(3*len(free), self.variables)).tocsr()
        self.z = np.zeros(self.variables)
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.base = self.initial[free].ravel().copy()
        self.initial_valid = not self.reasons

    def coordinates(self, chart, z):
        uv = chart['uv'].copy(); bary = chart['bary'].copy()
        for k in np.flatnonzero(chart['mutable']):
            d = z[chart['columns'][k]]
            bary[k, 1:] += d; bary[k, 0] -= d.sum()
            cell = chart['uv_source'][chart['cells'][k]]
            uv[chart['vertices'][k]] += d @ (cell[1:]-cell[0])
        return uv, bary

    def query(self, positions, which, domains, assignment, out):
        """Use the declared cell and current barycentrics, never infer a nearest foot."""
        local = np.full(len(self.owner), -1, int); local[self.used] = np.arange(len(self.used))
        for chart in self.charts:
            ids = local[chart['vertices']]; selected = which[ids]
            if not selected.any():
                continue
            _, bary = self.coordinates(chart, self.z); bary = bary[selected]
            ids = ids[selected]
            part = domains[int(assignment[chart['vertices'][0]])]
            face_ids = np.searchsorted(part['original_triangle_ids'], chart['refs'][selected])
            corners = part['triangles'][face_ids]
            closest = np.einsum('nk,nkd->nd', bary, part['positions'][corners])
            normal = np.einsum('nk,nkd->nd', bary, part['vertex_normal'][corners])
            lengths = np.linalg.norm(normal, axis=1)
            normal = np.where(lengths[:, None] > 1e-12, normal/np.maximum(lengths, 1e-300)[:, None],
                              part['face_normal'][face_ids])
            offset = positions[ids]-closest
            out['closest'][ids] = closest; out['normal'][ids] = normal
            out['offset'][ids] = np.sum(offset*normal, axis=1); out['distance'][ids] = np.linalg.norm(offset, axis=1)
            out['triangle'][ids] = chart['refs'][selected]
            out['component'][ids] = assignment[chart['vertices'][selected]]

    def constraints(self, z, jacobian=False):
        result, jac = [], []
        for chart in self.charts:
            uv, bary = self.coordinates(chart, z)
            for k in np.flatnonzero(chart['mutable']):
                result.append(bary[k, 0])
                if jacobian:
                    row = np.zeros(self.variables); row[chart['columns'][k]] = -1.; jac.append(row)
            faces = self.triangles[chart['ct']]
            result.extend((_area(uv[faces])*chart['sign']/chart['area']-chart['ratio']).tolist())
            if jacobian:
                vertex_lookup = {int(v): k for k, v in enumerate(chart['vertices'])}
                for face, area in zip(faces, chart['area']):
                    p = uv[face]
                    gradients = .5*np.array([[p[1, 1]-p[2, 1], p[2, 0]-p[1, 0]],
                        [p[2, 1]-p[0, 1], p[0, 0]-p[2, 0]], [p[0, 1]-p[1, 1], p[1, 0]-p[0, 0]]])
                    row = np.zeros(self.variables)
                    for v, g in zip(face, gradients):
                        k = vertex_lookup[int(v)]
                        if chart['mutable'][k]:
                            cell = chart['uv_source'][chart['cells'][k]]
                            row[chart['columns'][k]] += (cell[1:]-cell[0]) @ g * chart['sign']/area
                    jac.append(row)
        return np.asarray(jac) if jacobian else np.asarray(result)

    def advance(self, target):
        target = np.clip(target, self.lower, self.upper).copy()
        # Remove only simplex roundoff from a solver proposal before finding the
        # common feasible step; an outward 1e-17 at one cell corner must not lock
        # every other variable in the coupled patch.
        for chart in self.charts:
            for k in np.flatnonzero(chart['mutable']):
                cols = chart['columns'][k]
                b = chart['bary'][k, 1:] + target[cols]
                if 1 < b.sum() <= 1+1e-10:
                    b *= np.nextafter(1./b.sum(), 0.)
                    target[cols] = b-chart['bary'][k, 1:]
        d = target-self.z
        c = self.constraints(self.z)
        end = self.constraints(target); mid = self.constraints(self.z+.5*d)
        a = 2*(end+c-2*mid); b = end-c-a
        # A stricter allowance than the final 1e-10 constraint check avoids
        # treating arithmetic noise at an exactly active constraint as motion.
        fraction = _safe_fraction(a, b, c+1e-12)
        for k in range(self.variables):
            if d[k] > 0 and np.isfinite(self.upper[k]):
                fraction = min(fraction, max(0., (self.upper[k]-self.z[k])/d[k]))
            elif d[k] < 0 and np.isfinite(self.lower[k]):
                fraction = min(fraction, max(0., (self.lower[k]-self.z[k])/d[k]))
        return self.z + fraction*d, float(fraction)

    def solve(self, system, load, x):
        from scipy.optimize import minimize
        if not self.variables:
            self.blocked = True
            self.steps.append({'solver_success': False, 'reason': 'no movable variables under declared cells and boundary'})
            return x.copy(), False
        matrix = (self.map.T @ system @ self.map).toarray()/self.scale**2
        vector = np.asarray(self.map.T @ (load.ravel()-system @ self.base))/self.scale**2
        if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
            self.blocked = True
            self.steps.append({'solver_success': False, 'reason': 'nonfinite reduced objective'})
            return x.copy(), False
        def objective(z):
            return .5*float(z @ matrix @ z)-float(vector @ z), matrix @ z-vector
        solved = minimize(objective, self.z, jac=True, method='SLSQP',
            bounds=list(zip(self.lower, self.upper)),
            constraints={'type': 'ineq', 'fun': self.constraints, 'jac': lambda z: self.constraints(z, True)},
            options={'maxiter': self.max_iterations, 'ftol': 1e-13})
        old = self.z.copy(); old_energy = objective(old)[0]
        finite = np.isfinite(solved.x).all()
        proposed, fraction = self.advance(solved.x) if finite else (old, 0.)
        feasible = (np.isfinite(proposed).all() and np.min(self.constraints(proposed)) >= -1e-10
            and np.all(proposed >= self.lower-1e-12) and np.all(proposed <= self.upper+1e-12))
        new_energy = objective(proposed)[0] if feasible else np.inf
        accepted = feasible and new_energy <= old_energy + 1e-12*max(1., abs(old_energy))
        if accepted:
            self.z = proposed
        move = float(np.linalg.norm(self.map @ (self.z-old), ord=np.inf))
        settled = bool(solved.success and accepted and fraction == 1.)
        self.blocked = not settled and move <= self.tolerance
        self.steps.append({'solver_success': bool(solved.success), 'solver_status': int(solved.status),
            'solver_message': str(solved.message), 'solver_iterations': int(solved.nit),
            'accepted': bool(accepted), 'fraction': fraction, 'maximum_coordinate_move': move,
            'minimum_constraint_slack': float(self.constraints(self.z).min()),
            'objective_before': float(old_energy), 'objective_after': float(objective(self.z)[0])})
        result = x.copy(); result[self.free] = (self.base+self.map @ self.z).reshape(-1, 3)
        return result, settled

    def report(self, converged):
        rows = []
        for index, chart in enumerate(self.charts):
            uv, bary = self.coordinates(chart, self.z)
            ratios = _area(uv[self.triangles[chart['ct']]])*chart['sign']/chart['area'] if self.initial_valid else None
            rows.append({'chart': index, 'candidate_triangle_ids': chart['ct'].tolist(),
                'vertices': chart['vertices'].tolist(), 'source_triangle_ids': chart['refs'].tolist(),
                'barycentric': bary.tolist(), 'uv': (uv[chart['vertices']]*chart['uv_scale']+chart['uv_origin']).tolist(),
                'fixed_boundary_vertices': chart['boundary'].tolist(), 'minimum_area_ratio': chart['ratio'],
                'minimum_area_ratio_after': float(ratios.min()) if ratios is not None else None,
                'area_limited_triangle_ids': chart['ct'][ratios <= chart['ratio']+1e-8].tolist() if ratios is not None else [],
                'cell_boundary_vertices': chart['vertices'][bary.min(1) <= 1e-8].tolist(),
                'maximum_initial_source_distance': chart['maximum_initial_distance'], 'array_sha256': chart['hashes']})
        status = ('invalid_initial' if not self.initial_valid else 'blocked' if self.blocked else
                  'converged' if converged else 'iteration_limit')
        return {'status': status, 'feasible': self.initial_valid, 'reasons': self.reasons,
            'numerics': {'normalized_uv_area_tolerance': 1e-12, 'normalized_containment_area_tolerance': 1e-10,
                'constraint_slack_tolerance': 1e-10, 'barycentric_tolerance': 1e-12,
                'advancement_slack_allowance': 1e-12,
                'source_distance_tolerance': self.tolerance, 'maximum_inner_iterations': self.max_iterations},
            'charts': rows, 'steps': self.steps,
            'uncovered_triangle_ids': np.flatnonzero(~np.isin(np.arange(len(self.triangles)),
                np.concatenate([c['ct'] for c in self.charts]))).tolist(),
            'limits': 'Declared source cells and fixed initial UV footprint only, within floating-point tolerances. '
                'Interior barycentrics can redistribute; source cells never change automatically. Positive connected UV '
                'faces are enforced along accepted straight advances. Actual 3D triangle interiors, cross-chart joins, '
                'collisions, appearance and swept motion require separate checks. A blocked local solve is not a proof '
                'of global infeasibility. Initial source residuals within distance_tolerance are retained exactly.'}
