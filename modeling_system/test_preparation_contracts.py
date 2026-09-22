import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from .preparation import ArrayPreparation, prepare_arrays, save_preparation
from .preparation_contracts import PreparationOperation
from .result_reporting import compact_summary, json_data


class PreparationContractTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def handler(self, operation, inputs=None, parameters=None):
        handler = ArrayPreparation(self.root/'out', operations={'qualified': operation})
        sources = inputs or {}
        item = {'reads': {k: v['sha256'] for k, v in sources.items()},
                'workbench': {'profile': 'analysis'},
                'payload': {'operation': 'qualified', 'inputs': sources, 'parameters': parameters or {}}}
        return handler(item, {'attempt_key': 'one'})

    def test_missing_dependency_and_arguments_fail_before_loading_inputs(self):
        source = {'path': str(self.root/'absent.npz'), 'sha256': 'absent', 'array': 'frame'}
        operation = PreparationOperation(lambda frame: {}, requires=('missing_preparation_dependency_1729',))
        with patch('modeling_system.preparation_contracts.importlib.util.find_spec', return_value=None):
            with self.assertRaisesRegex(ValueError, 'dependency unavailable'):
                prepare_arrays('qualified', inputs={'frame': source}, operations={'qualified': operation})
        with self.assertRaisesRegex(ValueError, 'argument contract'):
            prepare_arrays('prepared_effect', inputs={'unknown': source})

    def test_two_column_frame_is_actionable_without_dispatch(self):
        called = []
        op = PreparationOperation(lambda frame: called.append(1), arrays={'frame': {'shape': [3, 3]}})
        result = self.handler(op, parameters={'frame': [[1, 0], [0, 1], [0, 0]]})
        self.assertEqual(result['status'], 'failed'); self.assertEqual(called, [])
        failure, = (self.root/'out').glob('*/failure.json')
        self.assertIn('frame: expected shape [3, 3], received [3, 2]', json.loads(failure.read_text())['error'])

    def test_expected_pure_math_errors_return_settled_failure(self):
        for number, error in enumerate((ModuleNotFoundError('dependency'), IndexError('shape'), TypeError('scalar'))):
            def fail(): raise error
            handler = ArrayPreparation(self.root/str(number), operations={'qualified': PreparationOperation(fail)})
            result = handler({'reads': {}, 'workbench': {'profile': 'analysis'},
                              'payload': {'operation': 'qualified', 'inputs': {}}}, {'attempt_key': 'one'})
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['workbench']['checks']['analysis']['status'], 'fail')

    def test_numpy_scalars_and_nested_arrays_survive_full_detail(self):
        result = {'array': np.arange(3), 'count': np.int32(3), 'nested': {'okay': np.bool_(True), 'values': np.array([.1, .2])}}
        files = save_preparation(self.root/'saved', result)
        actual = json.loads(Path(files['result.json']['path']).read_text())
        self.assertEqual(actual, {'count': 3, 'nested': {'okay': True, 'values': [.1, .2]}})
        with np.load(files['arrays.npz']['path'], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['array'], result['array'])

    def test_nonfinite_json_or_object_arrays_are_explicit_failures_before_output(self):
        for value in ({'metric': np.float64(np.nan)}, {'object': np.array([{}], object)}):
            with self.assertRaises(ValueError): save_preparation(self.root/'invalid', value)
            self.assertFalse((self.root/'invalid').exists())
        with self.assertRaises(TypeError): json_data({1: 'ambiguous key'})
        with self.assertRaises(TypeError): json_data(object())

    def test_large_metrics_stay_private_and_complete_limits_stay_public(self):
        limits = 'Only supplied poses; this cannot approve appearance.'
        result = self.handler(PreparationOperation(lambda: {
            'rms': np.arange(4000).tolist(), 'passed': np.bool_(True), 'limits': limits}))
        self.assertEqual(result['status'], 'completed')
        summary = result['workbench']['findings'][0]['summary']
        self.assertLessEqual(len(summary.encode()), 1500)
        self.assertIn(limits, summary)
        self.assertEqual(json.loads(summary)['omitted_metric_count'], 1)
        detail = json.loads(Path(result['prepared']['result.json']['path']).read_text())
        self.assertEqual(len(detail['rms']), 4000)
        with self.assertRaises(ValueError): compact_summary('op', {}, limits='x'*2000)

    def test_changed_input_is_still_rejected_and_native_handler_is_not_admitted(self):
        path = self.root/'input.npz'; np.savez(path, frame=np.eye(3))
        source = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'array': 'frame'}
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            prepare_arrays('qualified', inputs={'frame': source}, operations={'qualified': PreparationOperation(lambda frame: {})})
        with self.assertRaisesRegex(ValueError, 'explicit contracts'):
            prepare_arrays('qualified', inputs={}, operations={'qualified': lambda: {}})

    def test_repeated_attempt_cannot_rerun_or_write_into_immutable_output(self):
        called = []
        operation = PreparationOperation(lambda: (called.append(True) or {'count': 1}))
        self.handler(operation)
        before = {p: p.read_bytes() for p in (self.root/'out').rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'existing preparation attempt'): self.handler(operation)
        self.assertEqual(called, [True])
        self.assertEqual(before, {p: p.read_bytes() for p in (self.root/'out').rglob('*') if p.is_file()})


if __name__ == '__main__': unittest.main()
