"""Offline coupled XYZ diagnosis within explicit, finite linear-model boundaries."""
import io
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .store import atomic_write, digest

DOMAINS = ('source_state_id', 'pose', 'frame', 'units', 'dependency_fingerprint')


def _number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(name + ' must be finite')
    if positive and value <= 0:
        raise ValueError(name + ' must be positive')
    return float(value)


def _issue(issues, code, subject, reason, action, **details):
    issues.append(dict(code=code, subject=subject, reason=reason, next_action=action, **details))


def _pin(service, value, folder, assets):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value.get('sha256'):
            path = Path(value['path'])
            path = path if path.is_absolute() else folder / path
            key = (str(path.resolve()), value['sha256'])
            if key not in assets:
                asset = service.store.blob(path)
                if asset['sha256'] != value['sha256']:
                    raise ValueError('Repair input changed: ' + str(path))
                assets[key] = asset
            asset = assets[key]
            return dict(value, asset=asset)
        return {k: _pin(service, v, folder, assets) for k, v in value.items()}
    if isinstance(value, list):
        return [_pin(service, v, folder, assets) for v in value]
    return value


def _load(service, reference):
    with np.load(service.store.resolve_blob(reference['asset']), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _matrix(z, n, k):
    if np.asarray(z['shape']).tolist() != [3*n, k]:
        raise ValueError('Response CSR shape must be (3*N, K), point-major XYZ')
    if z['indices'].dtype.kind not in 'iu' or z['indptr'].dtype.kind not in 'iu':
        raise ValueError('Integer CSR indices required')
    if (z['indptr'].shape != (3*n+1,) or z['indptr'][0] != 0
            or np.any(np.diff(z['indptr']) < 0) or z['indptr'][-1] != len(z['data'])
            or z['indices'].shape != z['data'].shape or np.any(z['indices'] < 0)
            or np.any(z['indices'] >= k) or not np.isfinite(z['data']).all()):
        raise ValueError('Invalid finite CSR response')
    matrix = sparse.csr_matrix((z['data'], z['indices'], z['indptr']), shape=(3*n, k))
    matrix.sum_duplicates()
    return matrix


def _rows(case, issues):
    rows = []
    seen = set()
    for kind, entries in (('fit', case.get('targets', [])), ('preserve', case.get('preservation', []))):
        for row in entries:
            name = row.get('id')
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError('Every constraint needs a unique nonempty id')
            seen.add(name)
            if not row.get('region') or not row.get('terms'):
                raise ValueError('Constraint needs region and exact XYZ terms: ' + name)
            for term in row['terms']:
                idx = term['output']; direction = np.asarray(term['direction'], dtype=float)
                if type(idx) is not int or not 0 <= idx < len(case['outputs']):
                    raise ValueError('Constraint output is outside declared coverage')
                if direction.shape != (3,) or not np.isfinite(direction).all() or not np.any(direction):
                    raise ValueError('Constraint needs a finite nonzero XYZ direction')
            detail = dict(region=row['region'], outputs=[case['outputs'][t['output']] for t in row['terms']],
                          evidence=row.get('evidence', []), requested=row)
            if kind == 'fit':
                support = row.get('support', {})
                reasons = []
                if support.get('status') != 'qualified' or row.get('lower') is None or row.get('upper') is None:
                    reasons.append(('missing_target', 'Target form or usable surface support is unqualified',
                                    'Inspect the missing supported surface; retain the artistic target question'))
                if support.get('excluded'):
                    reasons.append(('excluded_target', 'The selected target belongs to an excluded region',
                                    'Choose supported interior correspondence without reviving excluded relief'))
                if support.get('correspondence') != 'qualified':
                    reasons.append(('missing_correspondence', 'Material/target correspondence is unqualified',
                                    'Resolve surface ownership and correspondence before fitting'))
                if support.get('visibility') not in ('visible', 'not_required'):
                    reasons.append(('visibility_uncertain', 'This target is not established by the supplied visible surface',
                                    'Use a view or section that observes this surface rather than projecting through an occluder'))
                registration = support.get('registration', {})
                if registration.get('status') != 'qualified' or 'error_bound' not in registration:
                    reasons.append(('registration_uncertain', 'Registration sensitivity is not bounded',
                                    'Measure alignment sensitivity against stationary context before assigning target displacement'))
                if not support.get('evidence'):
                    reasons.append(('missing_target_evidence', 'Qualification has no retained evidence',
                                    'Supply the measured support and its qualification evidence'))
                for code, reason, action in reasons:
                    _issue(issues, code, name, reason, action, **detail)
                if reasons:
                    continue
            else:
                if (not row.get('evidence') or not row.get('reason')
                        or (row.get('lower') is None and not row.get('lower_unbounded'))
                        or (row.get('upper') is None and not row.get('upper_unbounded'))
                        or row.get('status', 'qualified') != 'qualified'):
                    _issue(issues, 'constraint_uncertain', name, 'Preservation constraint lacks qualified bounds or its basis',
                           'Retain why this relationship must be preserved; distinguish preference from physical evidence', **detail)
                    continue
            lo = -np.inf if kind == 'preserve' and row.get('lower_unbounded') else _number(row['lower'], name+'.lower')
            hi = np.inf if kind == 'preserve' and row.get('upper_unbounded') else _number(row['upper'], name+'.upper')
            if not np.isfinite(lo) and not np.isfinite(hi):
                raise ValueError('A preservation row must have at least one finite bound')
            scale = _number(row['scale'], name+'.scale', True)
            if lo > hi:
                raise ValueError('Constraint lower bound exceeds upper bound: ' + name)
            if kind == 'fit':
                uncertainty = _number(registration['error_bound'], name+'.registration.error_bound')
                if uncertainty < 0:
                    raise ValueError('Registration error bound must be nonnegative')
                lo += uncertainty; hi -= uncertainty
                if lo > hi:
                    _issue(issues, 'registration_uncertain', name, 'Measured registration uncertainty consumes the target tolerance',
                           'Improve alignment evidence or revisit target tolerance explicitly', **detail)
                    continue
            rows.append(dict(row, kind=kind, effective_lower=lo, effective_upper=hi, scale=scale))
    if not case.get('targets'):
        _issue(issues, 'missing_target', 'targets', 'No intended target constraints are supplied',
               'Identify the specific missing form before optimizing controls')
    if not case.get('preservation'):
        _issue(issues, 'constraint_uncertain', 'preservation', 'No explicit preservation constraints are supplied',
               'Name and measure the protected relationships across the influenced region')
    return rows


def _operators(rows, z, matrix):
    rr, cc, vv = [], [], []
    for i, row in enumerate(rows):
        for term in row['terms']:
            for axis, value in enumerate(term['direction']):
                rr.append(i); cc.append(3*term['output']+axis); vv.append(value)
    measure = sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), matrix.shape[0]))
    baseline = measure @ z['baseline_xyz'].ravel()
    return measure, (measure @ matrix).tocsr(), baseline


def _target_conflicts(rows, outputs, issues):
    """Detect contradictory bounds on the same physical measurement, independent of controls."""
    seen = {}
    for row in rows:
        if row['kind'] != 'fit': continue
        terms = {}
        for term in row['terms']:
            for axis, value in enumerate(term['direction']):
                key = (term['output'], axis)
                terms[key] = terms.get(key, 0.) + value
        terms = sorted((key,value) for key,value in terms.items() if value)
        if not terms: continue
        factor = terms[0][1]
        identity = tuple((key, value/factor) for key,value in terms)
        lo, hi = sorted((row['effective_lower']/factor, row['effective_upper']/factor))
        prior = seen.setdefault(identity, dict(lower=lo, upper=hi, rows=[]))
        if max(lo, prior['lower']) > min(hi, prior['upper']):
            _issue(issues, 'contradictory_targets', row['id'], 'Qualified sources demand disjoint ranges for the same physical measurement',
                   'Reconcile target authority, source interpretation and registration; more controls cannot resolve these stated bounds',
                   region=row['region'], outputs=[outputs[t['output']] for t in row['terms']],
                   evidence=row.get('evidence', []), conflicting_rows=[*prior['rows'], row['id']])
        prior.update(lower=max(lo,prior['lower']), upper=min(hi,prior['upper']))
        prior['rows'].append(row['id'])


def _phase(a, low, high, scales, bounds, deadline):
    # One nonnegative normalized slack makes this phase-I problem feasible.
    k = a.shape[1]
    if not len(low):
        return dict(status='solved', slack=0., delta=np.zeros(k), solver_status=0)
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        return dict(status='unresolved', reason='Analysis time budget exhausted')
    scaled = sparse.diags(1/scales) @ a
    upper = np.isfinite(high); lower = np.isfinite(low)
    rhs = np.r_[high[upper]/scales[upper], -low[lower]/scales[lower]]
    if not np.isfinite(scaled.data).all() or not np.isfinite(rhs).all():
        return dict(status='unresolved', reason='Constraint normalization overflow; rescale the bounded problem')
    augmented = sparse.hstack([sparse.vstack([scaled[upper], -scaled[lower]]), -np.ones((len(rhs), 1))], format='csr')
    result = linprog(np.r_[np.zeros(k), 1.], A_ub=augmented, b_ub=rhs,
                     bounds=[*bounds, (0, None)], method='highs',
                     options={'time_limit': remaining, 'maxiter': 10000})
    if result.status != 0 or result.x is None:
        return dict(status='unresolved', solver_status=int(result.status), reason=result.message,
                    meaning='No infeasibility conclusion from optimizer failure or nonconvergence')
    delta = result.x[:k]
    measured_slack = max(0., float(np.max(np.r_[(a@delta-high)/scales, (low-a@delta)/scales])))
    if abs(measured_slack-result.x[-1]) > 1e-6:
        return dict(status='unresolved', reason='Phase-I solution failed independent slack verification')
    return dict(status='solved', slack=measured_slack, delta=delta, solver_status=0,
                meaning='Optimal normalized violation within supplied finite linear model; not nonlinear anatomical infeasibility')


def _candidate(a, baseline, rows, bounds, deadline, fallback):
    fit = np.array([r['kind'] == 'fit' for r in rows]); f = int(fit.sum()); k = a.shape[1]
    if deadline <= time.monotonic():
        return fallback, dict(status='feasible_phase_I', refinement='time budget exhausted')
    low = np.array([r['effective_lower'] for r in rows])-baseline
    high = np.array([r['effective_upper'] for r in rows])-baseline
    center = (low[fit]+high[fit])/2; af = a[fit]
    zero = sparse.csr_matrix((len(rows), f+k))
    identity = sparse.eye(k, format='csr'); fitid = sparse.eye(f, format='csr')
    upper = np.isfinite(high); lower = np.isfinite(low)
    parts = [sparse.hstack([a[upper], zero[upper]]), sparse.hstack([-a[lower], zero[lower]]),
             sparse.hstack([af, -fitid, sparse.csr_matrix((f, k))]),
             sparse.hstack([-af, -fitid, sparse.csr_matrix((f, k))]),
             sparse.hstack([identity, sparse.csr_matrix((k, f)), -identity]),
             sparse.hstack([-identity, sparse.csr_matrix((k, f)), -identity])]
    rhs = np.r_[high[upper], -low[lower], center, -center, np.zeros(2*k)]
    scale = np.array([r['scale'] for r in rows])[fit]
    span = np.maximum(np.max(np.abs(np.asarray(bounds)), axis=1), 1e-12)
    objective = np.r_[np.zeros(k), 1/scale, 1e-6/span]
    result = linprog(objective, A_ub=sparse.vstack(parts, format='csr'), b_ub=rhs,
                     bounds=[*bounds, *([(0, None)]*(f+k))], method='highs',
                     options={'time_limit': max(.001, deadline-time.monotonic()), 'maxiter': 10000})
    if result.status != 0 or result.x is None:
        return fallback, dict(status='feasible_phase_I', refinement='unresolved', solver_status=int(result.status))
    return result.x[:k], dict(status='optimized', objective='normalized absolute fit residual plus 1e-6 normalized control change')


def _nominations(case, issues):
    context = case.get('acquisition', {})
    result = []
    groups = {}
    for index, issue in sorted(enumerate(issues), key=lambda item:(-item[1].get('normalized_violation', 0), item[0])):
        # Group coordinate rows only when cause, region AND physical outputs agree.
        surface = tuple(sorted((o['object'], o['native_vertex']) for o in issue.get('outputs', [])))
        key = (issue['code'], issue.get('region'), surface or issue['subject'])
        groups.setdefault(key, []).append((index, issue))
    ordered = list(groups.values())
    first = {}; remainder = []
    for group in ordered:
        code = group[0][1]['code']
        if code not in first: first[code] = group
        else: remainder.append(group)
    selected = (list(first.values()) + remainder)[:12]
    for group in selected:
        issue = group[0][1]
        subjects = list(dict.fromkeys(item['subject'] for _, item in group))
        target_gap = issue['code'] == 'missing_target'
        requests = context.get('existing_requests', [])
        views = [v for v in context.get('views', []) if set(subjects).intersection(v.get('subjects', []))]
        visible = [v for v in views if v.get('visibility') == 'visible']
        preferred = min(visible, key=lambda v:(v.get('new_capture_required', True), v.get('pixel_area', float('inf')))) if visible else None
        ready = (target_gap and preferred and context.get('identity_evidence') and context.get('stationary_context')
                 and context.get('new_form_question') and not requests)
        result.append(dict(cause=issue['code'], subject=issue['subject'], region=issue.get('region'),
            related_subjects=subjects, diagnostic_indices=[i for i, _ in group], diagnostic_count=len(group),
            phase=case.get('state', {}).get('pose'), surface=issue.get('outputs', []),
            evidence=issue.get('evidence', []), uncertainty=issue['reason'], next_action=issue['next_action'],
            candidate_views=views, preferred_view=preferred,
            view_selection_basis='Prefer an existing recorded visible view, then the smallest supplied informative region; never use an occluded profile as target evidence',
            stationary_context=context.get('stationary_context', []),
            identity_evidence=context.get('identity_evidence', []), video_phase=context.get('video_phase'),
            native_motion_baselines=context.get('native_motion_baselines'),
            smallest_informative_measurement='Select the supplied view/section that reveals this named form or support; preserve stationary context',
            existing_requests=requests, new_form_question=context.get('new_form_question') if target_gap else None,
            generation='conditional_on_reviewed_capture_and_existing_authorization' if ready else 'not_justified_by_this_diagnostic',
            dispatch_authorized=False,
            limits='More pixels or denser geometry alone do not supply missing form. Different poses need distinct guide jobs. Transport input caps do not cap retained authority.'))
    represented = sum(n['diagnostic_count'] for n in result)
    return result, dict(total_diagnostics=len(issues), shown=len(result), total_groups=len(groups),
        represented_diagnostics=represented, deferred=max(0, len(issues)-represented),
        deferred_groups=max(0, len(groups)-len(result)),
        selection='Group identical cause/region/physical outputs; one group per category, then largest normalized violation and input order; all exact diagnostic indices retained')


def _evaluate(rows, values, baseline, outputs):
    result = []
    for i, row in enumerate(rows):
        violation = max(row['effective_lower']-values[i], values[i]-row['effective_upper'], 0.)
        result.append(dict(id=row['id'], region=row['region'], kind=row['kind'], value=float(values[i]),
            baseline=float(baseline[i]), lower=row['effective_lower'] if np.isfinite(row['effective_lower']) else None,
            upper=row['effective_upper'] if np.isfinite(row['effective_upper']) else None,
            violation=float(violation), normalized_violation=float(violation/row['scale']),
            outputs=[outputs[t['output']] for t in row['terms']], evidence=row.get('evidence', [])))
    return result


def analyze(service, case_path, proposed_delta=None, max_seconds=20):
    max_seconds = _number(max_seconds, 'max_seconds', True)
    if max_seconds > 45:
        raise ValueError('Offline analysis budget is at most 45 seconds')
    deadline = time.monotonic()+max_seconds
    path = Path(case_path).resolve(); original = service.store.blob(path)
    raw = json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig'))
    assets = {}; case = _pin(service, raw, path.parent, assets)
    if case.get('schema_version') != 1 or not case.get('question'):
        raise ValueError('Version-1 repair case with explicit question required')
    outputs = case.get('outputs', []); controls = case.get('controls', [])
    n, k = len(outputs), len(controls)
    if not 0 < n <= 250000 or not 0 < k <= 5000:
        raise ValueError('Bounded repair supports 1..250000 outputs and 1..5000 scalar controls')
    ids = [(o['object'], o['native_vertex']) for o in outputs]
    if len(set(ids)) != n or any(type(i) is not int or i < 0 for _, i in ids):
        raise ValueError('Distinct object/native vertex identities required')
    if len({c['id'] for c in controls}) != k or any(not c.get('meaning') for c in controls):
        raise ValueError('Distinct named scalar controls with actual mechanism/coordinate meanings required')
    issues = []; rows = _rows(case, issues); response = case.get('response', {})
    _target_conflicts(rows, outputs, issues)
    for requirement in case.get('unresolved_requirements', []):
        _issue(issues, 'constraint_uncertain', requirement['id'], requirement['reason'],
               'Establish this required relationship before treating a local fit as a complete repair',
               region=requirement.get('region'), evidence=requirement.get('evidence', []))
    if len(rows) > 100000:
        raise ValueError('Repair constraint count exceeds bounded analysis scope')
    for layer in case.get('layers', []):
        complete = layer.get('complete_for_controls', [])
        if (not isinstance(complete, list) or set(complete) != {c['id'] for c in controls}
                or layer.get('missing') or not layer.get('evidence')):
            _issue(issues, 'incomplete_influence', layer['name'], 'Dependent layer influence is unqualified or incomplete',
                   'Expand or measure this dependent layer before using a face-only fit', coverage=layer)
    declared = {l['name'] for l in case.get('layers', [])}
    if not case.get('required_layers'):
        _issue(issues, 'incomplete_influence', 'required_layers', 'Required dependent-layer inventory is unspecified',
               'Declare all layers required by this control scope, including those still unmeasured')
    for name in set(case.get('required_layers', [])) - declared:
        _issue(issues, 'incomplete_influence', name, 'Required dependent layer is missing from coverage',
               'Measure or explicitly retain this missing layer before a candidate can be complete')
    for name in {o['layer'] for o in outputs} - declared:
        _issue(issues, 'incomplete_influence', name, 'Output layer has no influence coverage declaration',
               'Supply complete-control ancestry and boundary evidence for this layer')
    if not case.get('layers'):
        _issue(issues, 'incomplete_influence', 'layers', 'No dependent-layer inventory supplied',
               'Inventory all driven geometry and unmeasured layers')
    for domain in DOMAINS:
        if domain not in case.get('state', {}) or response.get(domain) != case['state'].get(domain):
            _issue(issues, 'response_outside_validity', domain, 'Response and requested analysis domain differ or are unspecified',
                   'Rebind or measure response at the exact recorded state/pose/domain',
                   qualified=response.get(domain), requested=case.get('state', {}).get(domain))
    if (response.get('kind') not in ('exact_linear', 'local_derivative')
            or response.get('validity_status') != 'validated' or not response.get('validation_evidence')):
        _issue(issues, 'response_outside_validity', 'response', 'Final coupled response or finite validity is unqualified',
               'Measure a coupled local response including downstream support; do not copy a fixed-XZ scalar clamp')
    z = _load(service, response['arrays']) if response.get('arrays') else None
    numerical = dict(status='not_run'); predicted = None; delta = None; evaluations = []; effects = []; baseline_checks = []
    if z is None:
        _issue(issues, 'response_outside_validity', 'response.arrays', 'No coupled XYZ response was supplied',
               'Export the response or retain this diagnosis as missing evidence')
    else:
        if z['baseline_xyz'].shape != (n, 3) or not np.isfinite(z['baseline_xyz']).all():
            raise ValueError('Finite baseline_xyz (N,3) required')
        matrix = _matrix(z, n, k)
        bounds = None
        if any(key not in z for key in ('lower_delta', 'upper_delta')):
            _issue(issues, 'response_outside_validity', 'control_ranges', 'Nonzero qualified control ranges are not supplied',
                   'Measure a finite local range; do not replace unknown ranges with invented zero bounds')
        else:
            if any(z[key].shape != (k,) or not np.isfinite(z[key]).all() for key in ('lower_delta', 'upper_delta')):
                raise ValueError('Finite per-control delta bounds required when supplied')
            if np.any(z['lower_delta'] > 0) or np.any(z['upper_delta'] < 0):
                raise ValueError('Qualified delta bounds must include the baseline zero')
            bounds = list(zip(z['lower_delta'], z['upper_delta']))
        verified = set()
        for reference in case.get('native_baselines', []):
            selected = [i for i, o in enumerate(outputs) if o['object'] == reference['object']]
            xyz = _load(service, reference['file'])[reference['xyz_key']]
            native_ids = [outputs[i]['native_vertex'] for i in selected]
            if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
                    or any(i >= len(xyz) for i in native_ids)):
                raise ValueError('Invalid independently captured baseline coordinates or mapping')
            error = float(np.max(np.abs(xyz[native_ids]-z['baseline_xyz'][selected]))) if selected else 0.
            if (reference.get('frame') != case['state'].get('frame')
                    or reference.get('source_state_id') != case['state'].get('source_state_id') or error > 3e-7):
                _issue(issues, 'response_outside_validity', reference['object'], 'Response baseline differs from captured state/domain',
                       'Export and independently validate the exact native baseline', maximum_error=error)
            else:
                verified.update(selected)
        if len(verified) != n:
            _issue(issues, 'response_outside_validity', 'baseline', 'Independent native baseline coverage is incomplete',
                   'Provide exact captured coordinates for every output', missing_outputs=n-len(verified))
        _, a, base = _operators(rows, z, matrix)
        baseline_checks = _evaluate(rows, base, base, outputs)
        low = np.array([r['effective_lower'] for r in rows])-base
        high = np.array([r['effective_upper'] for r in rows])-base
        scales = np.array([r['scale'] for r in rows]); fit = np.array([r['kind'] == 'fit' for r in rows], dtype=bool)
        valid_response = not any(i['code'] == 'response_outside_validity' for i in issues)
        if proposed_delta is not None and valid_response:
            delta = np.asarray(proposed_delta, dtype=float)
            if delta.shape != (k,) or not np.isfinite(delta).all():
                raise ValueError('proposed_delta must contain exactly K finite scalar values')
            invalid = (delta < z['lower_delta']) | (delta > z['upper_delta'])
            if invalid.any():
                _issue(issues, 'response_outside_validity', 'proposed_delta', 'Proposed controls exceed measured finite ranges',
                       'Reduce the step or remeasure the response; do not extrapolate', controls=[controls[i] for i in np.flatnonzero(invalid)])
                delta = None
            else:
                numerical = dict(status='supplied_candidate_screen', solver='not invoked')
        elif proposed_delta is None and rows and valid_response:
            phases = {}
            for name, mask in (('fit', fit), ('preservation', ~fit), ('joint', np.ones(len(rows), dtype=bool))):
                phases[name] = _phase(a[mask], low[mask], high[mask], scales[mask], bounds, deadline)
            numerical = dict(status='diagnosed', phases={name:{key:value for key,value in p.items() if key != 'delta'} for name,p in phases.items()})
            if any(p['status'] != 'solved' for p in phases.values()):
                _issue(issues, 'solver_unresolved', 'fit', 'A numerical phase did not finish reliably',
                       'Inspect scaling and solver status or use a justified larger bounded analysis; do not infer infeasibility')
            elif phases['joint']['slack'] <= 1e-7:
                delta, refinement = _candidate(a, base, rows, bounds, deadline, phases['joint']['delta'])
                numerical.update(status='candidate', refinement=refinement)
            else:
                delta = phases['joint']['delta']
                if any(i['code'] == 'contradictory_targets' for i in issues):
                    code, reason, action = ('contradictory_targets', 'Target bounds conflict independently of the control model',
                                            'Resolve the named source/registration conflict before changing control authority')
                elif phases['fit']['slack'] > 1e-7:
                    code, reason, action = ('insufficient_controls', 'Qualified controls/ranges cannot satisfy the stated fit within this linear model',
                                            'Inspect reachable motion at the named rows; test different control ownership or validity rather than inventing a target')
                else:
                    code, reason, action = ('conflicting_constraints', 'Fit is attainable alone but the stated preservation constraints cannot be met jointly in this linear model',
                                            'Inspect the named conflicting relationships and their evidence; do not silently relax preservation')
                _issue(issues, code, 'linear_problem', reason, action, phases=numerical['phases'])
                numerical['status'] = 'diagnostic_trial_only'
        if delta is not None and (np.any(delta < z['lower_delta']) or np.any(delta > z['upper_delta'])):
            _issue(issues, 'solver_unresolved', 'control_ranges', 'Returned controls failed independent validity-range verification',
                   'Inspect numerical tolerances; do not extrapolate outside the qualified range')
            delta = None
        if delta is not None:
            values = a@delta+base; predicted = (matrix@delta).reshape(n, 3)+z['baseline_xyz']
            evaluations = _evaluate(rows, values, base, outputs)
            failed = [e for e in evaluations if e['normalized_violation'] > 1e-7]
            for failure in failed:
                _issue(issues, 'candidate_constraint_violation', failure['id'], 'Rechecked diagnostic trial violates this '+failure['kind']+' constraint',
                       'Inspect this residual and its preservation/target basis before any native proposal',
                       **{key:value for key,value in failure.items() if key not in ('id','kind')})
            displacement = predicted-z['baseline_xyz']
            for layer in declared:
                selection = np.array([i for i,o in enumerate(outputs) if o['layer'] == layer], dtype=int)
                if not len(selection): continue
                norms = np.linalg.norm(displacement[selection], axis=1)
                order = selection[np.argsort(-norms)[:6]]
                effects.append(dict(layer=layer, outputs=len(selection), maximum_displacement=float(norms.max()),
                                    worst=[dict(output=outputs[i], delta_xyz=displacement[i].tolist()) for i in order]))
    nominations, nomination_coverage = _nominations(case, issues)
    numerical_candidate = predicted is not None and numerical.get('status') in ('candidate', 'supplied_candidate_screen')
    disposition = 'candidate' if numerical_candidate and not issues else 'needs_evidence_or_revision'
    prediction_asset = None
    if predicted is not None:
        buffer = io.BytesIO(); np.savez_compressed(buffer, predicted_xyz=predicted, control_delta=delta)
        data = buffer.getvalue(); location = service.store.root/'derived'/(digest(data)+'.npz')
        atomic_write(location, data); prediction_asset = service.store.blob(location)
    payload = dict(schema_version=1, question=case['question'], state=case['state'], case=case,
        case_source=original, disposition=disposition, numerical=numerical, diagnostics=issues,
        constraints=evaluations, baseline_constraints=baseline_checks, layer_effects=effects, prediction=prediction_asset,
        acquisition_nominations=nominations,
        nomination_coverage=nomination_coverage,
        coverage=dict(outputs=n, controls=k, requested_targets=len(case.get('targets', [])),
                      evaluated_constraints=len(evaluations), qualified_constraints=len(rows), layers=case.get('layers', [])),
        native_ready=False, user_appearance_acceptance=False,
        limits=['Offline recorded analysis; no live freshness, native comparison or appearance acceptance.',
                'Local coupled linear response does not establish nonlinear finite-step or whole-blink correctness.',
                'Layer completeness and registration ranges are qualified only by supplied evidence.',
                'Every candidate still requires owner-controlled native comparison and the existing guide/depth workflow.'])
    record = service.store.put('repair_analysis', payload)
    # Keep heavy identities, constraint details and full arrays behind immutable reads.
    categories = {}
    representatives = []
    for issue in issues:
        if issue['code'] not in categories:
            representatives.append(issue)
        categories[issue['code']] = categories.get(issue['code'], 0)+1
    selected_issues = representatives + [issue for issue in issues if issue not in representatives]
    summary_issues = [{k:v for k,v in issue.items() if k in ('code','subject','reason','next_action','region')} for issue in selected_issues[:12]]
    nomination_summary = []
    for index, nomination in enumerate(nominations[:8]):
        view = nomination['preferred_view']
        nomination_summary.append(dict(
            **{key:nomination[key] for key in ('cause','subject','region','phase','diagnostic_count','next_action','generation','dispatch_authorized')},
            preferred_view={key:view[key] for key in ('observation','role','pixel','visibility','new_capture_required') if key in view} if view else None,
            detail=dict(operation='read_record', arguments=dict(record=record, path=['acquisition_nominations',index], limit=10, max_chars=8000))))
    return dict(analysis=record, disposition=disposition, question=case['question'], numerical_status=numerical['status'],
        diagnostics=summary_issues, diagnostic_count=len(issues), diagnostic_categories=categories,
        diagnostics_selection='One representative per category, then input order; complete category counts and exact records retained',
        diagnostics_deferred=max(0,len(issues)-12),
        nominations=nomination_summary, nominations_deferred=max(0,len(nominations)-len(nomination_summary)),
        prediction=prediction_asset, native_ready=False, user_appearance_acceptance=False,
        coverage={k:v for k,v in payload['coverage'].items() if k != 'layers'},
        reads={key:dict(operation='read_record', arguments=dict(record=record, path=[key], limit=10, max_chars=8000))
               for key in ('diagnostics','constraints','baseline_constraints','layer_effects','acquisition_nominations','nomination_coverage','case','numerical','limits')})
