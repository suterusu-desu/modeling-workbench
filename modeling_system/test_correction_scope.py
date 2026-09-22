from pathlib import Path
import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .correction_scope import correction_scope
from .preparation import prepare_arrays


class CorrectionScopeTests(unittest.TestCase):
    def setUp(self):
        self.data = {'response':np.array([[1.,0,0],[1.,1.,0],[0.,0.,1.]]),
            'residual':np.array([-1.,0.,0.]), 'tolerance':np.array([0.,.05,0.]),
            'local_controls':np.array([0]), 'coupled_controls':np.array([0,1]),
            'control_radius':np.array([2.,2.,2.])}
        self.parameters = {'units':'synthetic lengths','qualification':'Known affine coupled surface through the sampled motion',
                           'numerical_tolerance':1e-9}

    def solve(self): return correction_scope(**self.data, **self.parameters)

    def test_coupled_correction_restores_surrounding_shape_under_same_bounds(self):
        result=self.solve()
        self.assertEqual(result['status'],'coupled_feasible')
        self.assertEqual(result['attempts']['local']['status'],'infeasible_in_linear_model')
        np.testing.assert_allclose(result['control_step'],[1.,-.95,0.],atol=1e-12)
        np.testing.assert_array_equal(result['tolerance'],self.data['tolerance'])
        self.assertLessEqual(abs(result['predicted_residual'][1]),.05+1e-12)
        self.assertFalse(result['appearance_accepted'])

    def test_within_guide_tolerance_local_correction_does_not_reopen_surroundings(self):
        self.data['response'][1,0]=.02
        result=self.solve();self.assertEqual(result['status'],'local_feasible')
        self.assertEqual(list(result['attempts']),['local'])
        np.testing.assert_allclose(result['control_step'],[1.,0.,0.],atol=1e-12)

    def test_insufficient_connected_scope_does_not_loosen_bounds_or_move_outside(self):
        self.data['control_radius'][1]=.2
        result=self.solve();self.assertEqual(result['status'],'infeasible_in_declared_scopes')
        self.assertNotIn('control_step',result)
        np.testing.assert_array_equal(result['tolerance'],[0.,.05,0.])

    def test_solver_failure_is_unknown_and_does_not_trigger_scope_expansion(self):
        with patch('modeling_system.correction_scope.linprog',return_value=SimpleNamespace(status=1,x=None)) as solve:
            result=self.solve()
        self.assertEqual(result['status'],'unknown');self.assertEqual(solve.call_count,1)

    def test_solver_success_flag_cannot_hide_a_violating_witness(self):
        with patch('modeling_system.correction_scope.linprog',return_value=SimpleNamespace(status=0,x=np.zeros(6))):
            self.assertEqual(self.solve()['status'],'unknown')

    def test_expansion_cannot_omit_local_controls_or_invent_tolerance(self):
        self.data['coupled_controls']=np.array([1])
        with self.assertRaises(ValueError):self.solve()
        self.data['coupled_controls']=np.array([0,1]);self.data['tolerance'][1]=-1
        with self.assertRaises(ValueError):self.solve()

    def test_existing_pinned_array_preparation_route_returns_the_same_witness(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'qualified.npz';np.savez(path,**self.data)
            sha=hashlib.sha256(path.read_bytes()).hexdigest()
            inputs={key:{'path':str(path),'sha256':sha,'array':key} for key in self.data}
            result=prepare_arrays('correction_scope',inputs=inputs,parameters=self.parameters)
            self.assertEqual(result['status'],'coupled_feasible')
            inputs['residual']['sha256']='changed'
            with self.assertRaisesRegex(ValueError,'source changed'):
                prepare_arrays('correction_scope',inputs=inputs,parameters=self.parameters)


if __name__=='__main__':unittest.main()
