"""Local versus coupled correction feasibility for a qualified linear response.

This is recorded-array preparation, never permission or a Blender solve. Rows
must represent actual constrained outcomes across the relevant regions/poses.
The caller supplies response validity, connected scope, guide bounds and units.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack, hstack, eye


def correction_scope(response, residual, tolerance, local_controls, coupled_controls,
                     control_radius, *, units, qualification, numerical_tolerance):
    """Find a bounded local witness, expanding controls only if locally infeasible.

Require abs(residual + response @ step) <= tolerance, with no changes outside
the allowed controls and per-control trust radii. Both scopes use EXACTLY the
same guide bounds. Minimize absolute control movement; this is a feasibility
witness, not an aesthetic objective or a nonlinear impossibility certificate.
"""
    matrix = csr_matrix(response, dtype=float)
    residual, tolerance, radius = [np.asarray(v, float) for v in (residual, tolerance, control_radius)]
    m, n = matrix.shape
    if (not m or not n or residual.shape != (m,) or tolerance.shape != (m,) or radius.shape != (n,)
            or not all(np.isfinite(v).all() for v in (matrix.data, residual, tolerance, radius))
            or np.any(tolerance < 0) or np.any(radius < 0) or not units or not qualification
            or type(numerical_tolerance) not in (int, float) or not np.isfinite(numerical_tolerance)
            or not 1e-10 <= numerical_tolerance <= 1e-7):
        raise ValueError('Qualified finite response, explicit guide bounds, trust radii and numerical tolerance required')
    def indices(value):
        value = np.asarray(value)
        if (value.ndim != 1 or value.dtype.kind not in 'iu' or len(np.unique(value)) != len(value)
                or np.any(value < 0) or np.any(value >= n)):
            raise ValueError('Unique actual control indices required')
        return value
    local, coupled = indices(local_controls), indices(coupled_controls)
    if not len(local) or not set(local) <= set(coupled):
        raise ValueError('Coupled scope must include the local controls')
    # Auxiliary t bounds abs(step); sparse construction avoids a dense square matrix.
    zero = csr_matrix((m, n)); identity = eye(n, format='csr')
    constraints = vstack([hstack([matrix, zero]), hstack([-matrix, zero]),
                          hstack([identity, -identity]), hstack([-identity, -identity])], format='csr')
    upper = np.concatenate([tolerance-residual, tolerance+residual, np.zeros(2*n)])
    objective = np.concatenate([np.zeros(n), np.ones(n)])
    attempts, steps = {}, {}
    def solve(name, allowed):
        limits = np.zeros(n); limits[allowed] = radius[allowed]
        answer = linprog(objective, A_ub=constraints, b_ub=upper,
            bounds=list(zip(-limits, limits))+[(0., None)]*n, method='highs',
            options={'primal_feasibility_tolerance':numerical_tolerance,
                     'dual_feasibility_tolerance':numerical_tolerance})
        row = {'status':'infeasible_in_linear_model' if answer.status == 2 else 'unknown',
               'solver_status':int(answer.status), 'allowed_control_count':len(allowed)}
        if answer.status == 0 and answer.x is not None and np.isfinite(answer.x).all():
            step = answer.x[:n]; predicted = residual + matrix @ step
            error = float(max(0., np.max(np.abs(predicted)-tolerance), np.max(np.abs(step)-limits)))
            # Numerical allowance is reported separately; never widen guide bounds.
            row.update(maximum_numerical_violation=error,
                       active_bound_rows=np.flatnonzero(tolerance-np.abs(predicted) <= numerical_tolerance).tolist())
            if error <= numerical_tolerance:
                row['status']='feasible_in_linear_model'; steps[name]=step
        attempts[name]=row
        return row['status']
    local_status = solve('local', local); selected = None
    if local_status == 'feasible_in_linear_model': selected='local'
    elif local_status == 'infeasible_in_linear_model' and set(coupled) != set(local):
        if solve('coupled', coupled) == 'feasible_in_linear_model': selected='coupled'
    status = selected+'_feasible' if selected else (
        'infeasible_in_declared_scopes' if all(r['status']=='infeasible_in_linear_model' for r in attempts.values()) else 'unknown')
    result = {'status':status, 'selected_scope':selected, 'attempts':attempts, 'units':units,
        'qualification':qualification, 'numerical_tolerance':float(numerical_tolerance),
        'tolerance':tolerance.copy(), 'appearance_accepted':False,
        'public_metrics':{'correction_scope':status, 'scopes_tested':list(attempts)},
        'limits':'Qualified linear response and declared sampled bounds only. Final evaluated motion and whole-region visual review remain required. Solver failure is unknown, not impossibility.'}
    if selected:
        result.update(control_step=steps[selected], predicted_residual=residual+matrix@steps[selected])
    return result
