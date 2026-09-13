"""Coverage failures that accurate forward predictions cannot detect."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scipy import sparse
from .service import ModelingService


class ControlCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = ModelingService(store=self.root/'store')
        # p0 has a second omitted ancestor; p1 can only be reached by it.
        matrix = sparse.csr_matrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
        sparse.save_npz(self.root/'ancestry.npz', matrix)
        domain = dict(source_state_id='fixture', pose={'bend': 1}, frame='local',
                      units='unit', dependency_fingerprint='topology-and-stack')
        self.case = dict(schema_version=1, question='Does this new patch have omitted ancestors?', state=domain,
            outputs=[dict(id='p'+str(i), meaning='fixture native vertex '+str(i)) for i in range(3)],
            controls=[dict(id='c'+str(i), meaning='fixture raw control '+str(i)) for i in range(3)],
            ancestry=dict(domain, kind='structural_dependency',
                arrays=dict(path='ancestry.npz', sha256=hashlib.sha256((self.root/'ancestry.npz').read_bytes()).hexdigest()),
                evidence=['independent exact synthetic topology'],
                coverage=dict(universe_complete=True, complete_for_outputs=['p0','p1','p2'], missing=[])),
            targets=['p0','p1'], selected_controls=['c0','c2'], allowed_controls=['c0'],
            exclusions=[dict(control='c2',reason='keep neighbor fixed',interpretation='protected',evidence=['fixture intent'])])

    def run_case(self, expected=None):
        path = self.root/'case.json'; path.write_text(json.dumps(self.case), encoding='utf-8')
        result = self.service.inspect_control_coverage(str(path), expected or self.case['state'])
        return result, self.service.store.get(result['analysis'], 'control_coverage')

    def test_complete_forward_prediction_can_hide_omitted_ancestor(self):
        result, full = self.run_case()
        self.assertEqual(result['status'], 'omitted_ancestors')
        self.assertEqual(full['omitted_ancestors'], ['c1'])
        self.assertEqual(result['summary']['targets_without_allowed_ancestors'], 1)
        self.assertEqual(full['targets'][1]['allowed_ancestors'], [])
        self.assertFalse(result['native_ready'])
        self.assertEqual(result['controllability'], 'not_established')

    def test_selected_control_matrix_cannot_claim_full_reverse_coverage(self):
        self.case['ancestry']['coverage'].update(universe_complete=False, missing=['unexported raw controls'])
        result, full = self.run_case()
        self.assertEqual(result['status'], 'incomplete_ancestry')
        self.assertEqual(result['summary']['incomplete_targets'], 2)
        self.assertEqual(full['omitted_ancestors'], ['c1'])

    def test_expanded_selection_does_not_prove_shape_quality(self):
        self.case['selected_controls'].append('c1'); self.case['allowed_controls'].append('c1')
        result, full = self.run_case()
        self.assertEqual(result['status'], 'selected_covers_declared_target_ancestry')
        self.assertEqual(result['controllability'], 'not_established')
        self.assertEqual(full['targets'][1]['allowed_ancestors'], ['c1'])

    def test_excluded_omitted_control_is_not_permission_to_edit(self):
        self.case['exclusions'].append(dict(control='c1',reason='provisional window',
            interpretation='trial_restriction',evidence=['fixture scope']))
        result, full = self.run_case()
        self.assertEqual(full['omitted_ancestors'], ['c1'])
        self.assertEqual(full['unclassified_omitted_ancestors'], [])
        self.assertEqual(result['summary']['targets_without_allowed_ancestors'], 1)
        self.assertEqual(full['case']['exclusions'][1]['interpretation'], 'trial_restriction')

    def test_recorded_state_drift_is_refused(self):
        changed = copy.deepcopy(self.case['state']); changed['pose']['bend'] = .5
        with self.assertRaisesRegex(ValueError, 'expected recorded state'):
            self.run_case(changed)

    def test_reverse_complete_does_not_claim_target_domain_complete(self):
        result, full = self.run_case()
        self.assertTrue(result['summary']['reverse_coverage_complete'])
        self.assertEqual(result['summary']['target_scope_status'], 'unknown')
        self.case['target_scope'] = dict(status='complete_for_question')
        with self.assertRaisesRegex(ValueError, 'Complete target scope'):
            self.run_case()
        self.case['target_scope'] = dict(status='declared_subset', missing=['other supported points not surveyed'])
        result, full = self.run_case()
        self.assertEqual(result['summary']['target_scope_status'], 'declared_subset')

    def test_changed_arrays_and_unknown_ids_are_refused(self):
        self.case['targets'] = ['not-a-point']
        with self.assertRaisesRegex(ValueError, 'unknown IDs'):
            self.run_case()
        self.case['targets'] = ['p0']; (self.root/'ancestry.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            self.run_case()

    def test_derivative_and_unexplained_restriction_are_refused(self):
        self.case['ancestry']['kind'] = 'local_derivative'
        with self.assertRaisesRegex(ValueError, 'zero derivative'):
            self.run_case()
        self.case['ancestry']['kind'] = 'structural_dependency'; self.case['exclusions'] = []
        with self.assertRaisesRegex(ValueError, 'explicit exclusion'):
            self.run_case()

    def test_no_ancestors_and_unverified_rows_remain_distinct(self):
        sparse.save_npz(self.root/'ancestry.npz', sparse.csr_matrix((3,3), dtype=bool))
        self.case['ancestry']['arrays']['sha256'] = hashlib.sha256((self.root/'ancestry.npz').read_bytes()).hexdigest()
        self.case['ancestry']['coverage']['complete_for_outputs'] = ['p0']
        result, full = self.run_case()
        self.assertTrue(full['targets'][0]['complete']); self.assertFalse(full['targets'][1]['complete'])
        self.assertEqual(result['status'], 'incomplete_ancestry')

    def test_full_record_survives_fresh_process_without_original_inputs(self):
        result, full = self.run_case()
        (self.root/'ancestry.npz').unlink(); (self.root/'case.json').unlink()
        script = ('import json; from modeling_system.service import ModelingService; '
                  's=ModelingService(store='+repr(str(self.root/'store'))+'); '
                  'r=s.store.get('+repr(result['analysis'])+',"control_coverage"); '
                  's.store.resolve_blob(r["case"]["ancestry"]["arrays"]["asset"]); '
                  'print(json.dumps(r["omitted_ancestors"]))')
        completed = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(completed.stdout), ['c1'])


if __name__ == '__main__':
    unittest.main()
