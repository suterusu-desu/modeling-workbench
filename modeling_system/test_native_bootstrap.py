import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from .native_bootstrap import StageTimings, phase_summary, load_pinned_modules, begin_scoped_feature, SCOPE_ROLES


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.file = self.root/'scope.json'; self.file.write_text('{"measured": true}')
        ref = {'path': str(self.file), 'sha256': hashlib.sha256(self.file.read_bytes()).hexdigest()}
        self.checks = {name: {'status': 'pass', 'evidence': [ref]} for name in SCOPE_ROLES}
        self.calls = []
        self.api = SimpleNamespace(refresh=self.refresh,
            bindings=lambda: SimpleNamespace(require_valid=lambda: self.calls.append('binding')),
            require_fresh=lambda record: self.calls.append('fresh'))

    def refresh(self, feature, refresh_depth):
        self.calls.append(('refresh', feature, refresh_depth))
        return {'feature': feature, 'fingerprint': 'actual synthetic state'}

    def begin(self, **kwargs):
        return begin_scoped_feature(self.api, 'region', 'observed defect', 'qualified mechanism',
                                    self.root/'entry', scope_check=lambda record: self.checks, **kwargs)

    def test_scoped_setup_keeps_binding_freshness_and_all_exact_checks(self):
        result = self.begin(protected=['held material'])
        self.assertEqual(self.calls, [('refresh', 'region', False), 'binding', 'fresh'])
        self.assertEqual(result['scoped_checks'], self.checks)
        self.assertNotIn('scene_record', result)
        self.assertTrue((self.root/'entry/feature-entry.json').is_file())
        with self.assertRaisesRegex(ValueError, 'existing feature entry'): self.begin()

    def test_missing_failed_or_changed_scope_evidence_refuses_setup(self):
        self.checks.pop('sections')
        with self.assertRaisesRegex(ValueError, 'every modeling dependency'): self.begin()
        self.checks['sections'] = {'status': 'unknown', 'evidence': []}
        with self.assertRaisesRegex(ValueError, 'passing evidence'): self.begin()
        self.checks['sections'] = self.checks['source']
        self.file.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'input changed'): self.begin()
        self.assertNotIn('fresh', self.calls)

    def test_explicit_binding_repair_still_requires_fresh_feature(self):
        result = self.begin(repair_invalid_bindings=True)
        self.assertNotIn('binding', self.calls); self.assertIn('fresh', self.calls)
        self.assertTrue(result['repair_invalid_bindings'])

    def test_phase_receipt_retains_actual_failed_stage(self):
        ticks = iter([0., 1., 1.25, 2., 2.75, 3.])
        timer = StageTimings(self.root/'native-stages.json', clock=lambda: next(ticks))
        with timer.measure('driver_restore'): pass
        with self.assertRaisesRegex(ValueError, 'failure'):
            with timer.measure('scoped_checks'): raise ValueError('failure')
        record = timer.finish()
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(phase_summary(record), {'driver_restore': 250., 'scoped_checks': 750.})
        self.assertEqual(json.loads(timer.path.read_text()), record)
        with self.assertRaises(ValueError): StageTimings(timer.path)

    def test_incomplete_or_invalid_timings_are_not_success(self):
        timer = StageTimings(self.root/'native-stages.json')
        with timer.measure('load'):
            with self.assertRaises(ValueError):
                with timer.measure('nested'): pass
        record = timer.finish(); record['stages'].append({'name': 'unfinished', 'status': 'running', 'elapsed_ms': 0.})
        self.assertNotIn('unfinished', phase_summary(record))
        record['stages'][0]['elapsed_ms'] = float('nan')
        with self.assertRaises(ValueError): phase_summary(record)

    def test_all_modules_pinned_before_any_import_and_exact_bytes_executed(self):
        source = self.root/'module.py'; source.write_text('VALUE = 42\n')
        marker = 'synthetic_workbench_bootstrap_module'
        self.addCleanup(lambda: sys.modules.pop(marker, None))
        ref = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_pinned_modules({marker: ref, marker+'_bad': {**ref, 'sha256': 'wrong'}})
        self.assertNotIn(marker, sys.modules)
        modules = load_pinned_modules({marker: ref})
        self.assertEqual(modules[marker].VALUE, 42)
        with self.assertRaisesRegex(ValueError, 'fresh explicit'): load_pinned_modules({marker: ref})


if __name__ == '__main__': unittest.main()
