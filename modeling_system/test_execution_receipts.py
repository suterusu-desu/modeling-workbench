"""Compact results, durable receipts and launcher receipts remain separate facts; nothing is replayed."""
import json
from pathlib import Path
import tempfile
import unittest
from .execution_receipts import inspect
from .service import ModelingService
from .store import canonical, digest
from .worker_contract import WorkerContract


def ref(path):
    return dict(path=str(path), sha256=digest(Path(path).read_bytes()))


class ExecutionReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.s = ModelingService(self.root, store=self.root / 'store')
        meta = lambda meaning: dict(meaning=meaning, evidence=['synthetic fixture'])
        self.native = self.root / 'native'; self.native.mkdir()
        self.candidate = self.native / 'candidate.bin'; self.candidate.write_bytes(b'saved candidate bytes')
        self.receipt_path = self.native / 'receipt.json'
        self.receipt = dict(status='applied_trial', operation='apply_proposal', label='synthetic trial', changed_objects=['Fixture mesh'],
                            candidate=dict(ref(self.candidate), bytes=21, role='saved candidate'),
                            receipt_path=str(self.receipt_path), after=dict(native_content_hash='abc'))
        self.write(self.receipt_path, self.receipt)
        self.record = self.s.store.put('operation', dict(status='applied_trial', operation='native_apply_proposal', candidate=self.receipt['candidate']))
        self.result_path = self.root / 'apply-result.json'
        self.result = dict(status='applied_trial', receipt_path=str(self.receipt_path), operation='native_apply_proposal',
                           expanded_fields=dict(candidate='read_record(operation_record).candidate', live='read_record(operation_record).live'),
                           detail_available=True, operation_record=self.record)
        self.write(self.result_path, self.result)
        self.adapter = dict(id='synthetic-adapter', meaning='Owner-verified result contract of the synthetic adapter', evidence=['synthetic contract review'],
                            statuses={'applied_trial': dict(effect='applied', meaning='Trial applied; candidate and rollback saved'),
                                      'failed': dict(effect='failed', meaning='Adapter reported failure'),
                                      'refused': dict(effect='not_applied', meaning='Precondition refused before mutation'),
                                      'uncertain_execution': dict(effect='unknown', meaning='Transport lost after dispatch')},
                            receipt=dict(required=True, reference=['receipt_path']),
                            identity_checks=[dict(id='status_agrees', result=['status'], receipt=['status']),
                                             dict(id='operation_is_apply', receipt=['operation'], equals='apply_proposal')],
                            artifacts=[{'id': 'candidate', 'in': 'receipt', 'path': ['candidate']}],
                            deferred=['expanded_fields'], record_reference=['operation_record'])
        self.case = dict(schema_version=1, question='Did the trial apply and is its receipt verifiable?',
                         operation=dict(meta('Native trial application through the bound adapter'), kind='native_operation', name='native_apply_proposal'),
                         result=ref(self.result_path), receipt=ref(self.receipt_path), adapter=self.adapter)

    def write(self, path, value):
        Path(path).write_bytes(canonical(value)); return Path(path)

    def run_case(self, case=None):
        case_path = self.write(self.root / 'case.json', case or self.case)
        result = self.s.inspect_execution_receipt(str(case_path))
        return result, self.s.store.get(result['analysis'], 'execution_receipt')

    def repin(self):
        self.write(self.receipt_path, self.receipt); self.write(self.result_path, self.result)
        self.case['receipt'] = ref(self.receipt_path); self.case['result'] = ref(self.result_path)

    def test_operation_specific_status_expands_the_receipt_and_verifies_candidate_bytes(self):
        r, p = self.run_case()
        self.assertEqual(r['disposition'], 'verified_consistent'); self.assertTrue(r['success_established'])
        self.assertEqual(p['effect']['disposition'], 'applied'); self.assertFalse(p['effect']['independent_proof'])
        self.assertEqual([a['status'] for a in p['artifacts']], ['verified'])
        self.assertEqual({i['id']: i['outcome'] for i in p['identity']},
                         {'receipt_reference': 'agree', 'status_agrees': 'agree', 'operation_is_apply': 'agree'})
        self.assertEqual(p['deferred_fields'], ['candidate', 'live'])
        self.assertEqual(p['record']['status'], 'present_in_this_store')
        self.assertFalse(r['native_ready']); self.assertFalse(r['appearance_accepted']); self.assertFalse(r['safe_to_replay'])
        self.assertEqual(p['process']['disposition'], 'not_applicable'); self.assertEqual(p['required_outputs']['disposition'], 'not_applicable')
        self.assertEqual(r['links']['receipt']['sha256'], self.case['receipt']['sha256'])
        out = self.s.execute('read_record', r['reads']['artifacts']['arguments']); self.assertIn('verified', json.dumps(out))
        out = self.s.execute('read_record', r['reads']['operation_record']['arguments']); self.assertIn('applied_trial', json.dumps(out))
        self.assertTrue(any('deferred' in a for a in r['next_actions']))

    def test_configured_unknown_scope_cannot_establish_success(self):
        for field,spec in [('required_outputs',{'in':'result','directory':['absent'],'names':['inventory.json']}),
                           ('source_preservation',{'in':'result','unchanged':['absent'],'before':['before'],'after':['after']})]:
            case=json.loads(json.dumps(self.case)); case['adapter'][field]=spec
            with self.subTest(field=field):
                result,payload=self.run_case(case)
                self.assertEqual(result['disposition'],'incomplete')
                self.assertFalse(result['success_established'])

    def test_changed_candidate_bytes_are_a_mismatch_not_success(self):
        self.candidate.write_bytes(b'regenerated candidate bytes')
        r, p = self.run_case()
        self.assertEqual(r['disposition'], 'needs_reconciliation'); self.assertFalse(r['success_established'])
        self.assertEqual(p['artifacts'][0]['status'], 'mismatch'); self.assertEqual(p['effect']['disposition'], 'applied')
        self.assertTrue(any('stale' in a for a in r['next_actions'])); self.assertFalse(r['safe_to_replay'])
        self.candidate.unlink()
        r, p = self.run_case()
        self.assertEqual(p['artifacts'][0]['status'], 'missing'); self.assertEqual(r['disposition'], 'incomplete')

    def test_unrecognized_conflicting_and_missing_receipts_stay_unknown(self):
        del self.adapter['statuses']['applied_trial']
        r, p = self.run_case()
        self.assertEqual(p['effect']['disposition'], 'unknown'); self.assertEqual(r['disposition'], 'needs_reconciliation')
        self.assertIn('unrecognized', p['effect']['reasons'][0])
        self.adapter['statuses']['applied_trial'] = dict(effect='applied', meaning='applied')
        self.receipt['status'] = 'failed'; self.repin()
        r, p = self.run_case()
        self.assertEqual(p['effect']['disposition'], 'unknown'); self.assertTrue(any('conflicting' in x for x in p['effect']['reasons']))
        self.assertEqual(p['artifacts'][0]['status'], 'verified')
        self.receipt['status'] = 'applied_trial'; self.repin(); self.case.pop('receipt')
        r, p = self.run_case()
        self.assertEqual(p['effect']['disposition'], 'unknown'); self.assertEqual(r['disposition'], 'needs_reconciliation')
        self.assertTrue(any('replay' in a for a in r['next_actions'])); self.assertIsNone(r['links']['receipt'])
        self.assertEqual([a['status'] for a in p['artifacts']], ['unreferenced'])

    def test_result_pointing_at_another_receipt_disagrees(self):
        other = self.write(self.native / 'other-receipt.json', dict(self.receipt, label='other'))
        self.result['receipt_path'] = str(other); self.repin()
        r, p = self.run_case()
        self.assertEqual(next(i for i in p['identity'] if i['id'] == 'receipt_reference')['outcome'], 'disagree')
        self.assertEqual(r['disposition'], 'needs_reconciliation')
        self.result.pop('receipt_path'); self.repin()
        r, p = self.run_case()
        self.assertEqual(next(i for i in p['identity'] if i['id'] == 'receipt_reference')['outcome'], 'missing')
        self.assertEqual(p['effect']['disposition'], 'unknown')

    def test_unknown_mapped_status_and_failed_effects_stay_unresolved(self):
        self.result['status'] = self.receipt['status'] = 'uncertain_execution'; self.repin()
        r, p = self.run_case()
        self.assertEqual(p['effect']['disposition'], 'unknown'); self.assertEqual(r['disposition'], 'needs_reconciliation')
        self.result['status'] = self.receipt['status'] = 'failed'; self.repin()
        r, p = self.run_case()
        self.assertEqual(r['disposition'], 'failed'); self.assertFalse(r['success_established'])
        self.assertEqual(p['artifacts'][0]['status'], 'verified'); self.assertFalse(r['safe_to_replay'])
        self.assertTrue(any('Nothing here clears the failure' in a for a in r['next_actions']))

    def runner_fixture(self, status='failed', returncode=0, stamp='stamp'):
        run = self.root / 'runs' / stamp; run.mkdir(parents=True)
        source = self.root / 'working.blend'; source.write_bytes(b'working file bytes'); sha = digest(source.read_bytes())
        (run / 'arrays.npz').write_bytes(b'arrays'); (run / 'report.json').write_bytes(b'{"checks": "all exact"}')
        receipt = dict(source=str(source), source_sha256_before=sha, script_sha256=digest(b'script'), output=str(run), status=status,
                       returncode=returncode, source_sha256_after=sha, source_unchanged=True, elapsed_seconds=3.1)
        path = self.write(run / 'receipt.json', receipt)
        adapter = dict(id='synthetic-launcher', meaning='Isolated launcher receipt contract', evidence=['launcher source review'],
                       statuses={'completed': dict(effect='completed', meaning='exit zero, source unchanged and completion file present'),
                                 'failed': dict(effect='failed', meaning='launcher classification failed'),
                                 'running': dict(effect='unknown', meaning='not finished')},
                       receipt=dict(required=False),
                       process={'in': 'result', 'returncode': ['returncode'], 'success_codes': [0]},
                       required_outputs={'in': 'result', 'directory': ['output'], 'names': ['inventory.json']},
                       source_preservation={'in': 'result', 'unchanged': ['source_unchanged'], 'before': ['source_sha256_before'], 'after': ['source_sha256_after']})
        case = dict(schema_version=1, question='Did the isolated worker run satisfy its launcher contract?',
                    operation=dict(kind='worker_run', name='reopen_worker', meaning='Independent reopen verification worker', evidence=['synthetic']),
                    result=ref(path), adapter=adapter)
        return run, path, case

    def test_exit_zero_without_required_inventory_is_incomplete_not_complete(self):
        run, path, case = self.runner_fixture()
        r, p = self.run_case(case)
        self.assertEqual(p['process']['disposition'], 'exit_ok'); self.assertEqual(p['required_outputs']['missing'], ['inventory.json'])
        self.assertEqual(p['source_preservation']['disposition'], 'unchanged'); self.assertEqual(p['effect']['disposition'], 'failed')
        self.assertEqual(r['disposition'], 'incomplete'); self.assertFalse(r['success_established'])
        self.assertTrue(any('fabricate' in a for a in r['next_actions']))
        self.assertEqual(p['completion_manifest']['disposition'], 'absent')
        run2, path2, claimed = self.runner_fixture(status='completed', stamp='claimed')
        r2, p2 = self.run_case(claimed)
        self.assertEqual(r2['disposition'], 'incomplete'); self.assertFalse(r2['success_established'])
        self.assertEqual(p2['effect']['disposition'], 'completed')

    def test_independent_verification_is_linked_without_altering_the_original_receipt(self):
        run, path, case = self.runner_fixture(); before = digest(path.read_bytes())
        plain = self.run_case(case)[0]
        verification = self.write(self.root / 'verified-offline.json', dict(status='verified offline', original_receipt=ref(path),
                                                                              artifacts=[dict(ref(run / 'arrays.npz'), role='independently produced output')]))
        case['independent_verification'] = [dict(ref(verification), role='Offline verification of produced outputs')]
        r, p = self.run_case(case)
        self.assertEqual(r['disposition'], plain['disposition']); self.assertEqual(r['disposition'], 'incomplete')
        self.assertFalse(r['success_established']); self.assertEqual(r['summary']['effect'], 'failed')
        self.assertEqual(p['independent_verification'][0]['role'], 'Offline verification of produced outputs')
        self.assertIn('does not alter', p['independent_verification'][0]['bearing'])
        self.assertEqual(digest(path.read_bytes()), before); self.assertEqual(len(r['links']['independent_verification']), 1)

    def test_contract_manifest_completes_a_launcher_run_and_later_changes_are_detected(self):
        run, path, case = self.runner_fixture(status='completed')
        contract = WorkerContract(run, completion_file='inventory.json', required_artifacts=['arrays.npz', 'report.json'], required_checks=['exact'],
                                  protected_sources=[dict(path=str(self.root / 'working.blend'), sha256=digest(b'working file bytes'), role='launcher input')])
        contract.preflight(); contract.check('exact', True); contract.finalize(extra={'status': 'reopened_exact'})
        r, p = self.run_case(case)
        self.assertEqual(p['required_outputs']['disposition'], 'complete'); self.assertEqual(p['completion_manifest']['disposition'], 'verified')
        self.assertEqual(p['completion_manifest']['source'], 'required_output')
        self.assertEqual(r['disposition'], 'verified_consistent'); self.assertTrue(r['success_established'])
        (run / 'arrays.npz').write_bytes(b'changed later')
        r, p = self.run_case(case)
        self.assertEqual(p['completion_manifest']['disposition'], 'mismatch'); self.assertEqual(r['disposition'], 'needs_reconciliation')
        (run / 'inventory.json').write_bytes(b'{"status": "hand written"}'); (run / 'arrays.npz').write_bytes(b'arrays')
        r, p = self.run_case(case)
        self.assertEqual(p['required_outputs']['disposition'], 'complete')
        self.assertEqual(p['completion_manifest']['disposition'], 'not_contract_manifest')
        self.assertEqual(r['disposition'], 'consistent_unverified'); self.assertFalse(r['success_established'])

    def test_pinned_manifest_receipt_is_evaluated_and_owner_status_needs_no_mapping(self):
        run, path, case = self.runner_fixture(status='completed')
        contract = WorkerContract(run, completion_file='inventory.json', required_artifacts=['arrays.npz'], required_checks=[], protected_sources=[])
        contract.preflight(); contract.finalize(extra={'status': 'source_inventoried'})
        case['receipt'] = ref(run / 'inventory.json'); case['adapter']['status_path'] = {'receipt': None}
        r, p = self.run_case(case)
        self.assertEqual(p['completion_manifest']['source'], 'receipt'); self.assertEqual(r['disposition'], 'verified_consistent')
        case['adapter'].pop('status_path')
        r, p = self.run_case(case)
        self.assertEqual(p['effect']['disposition'], 'unknown'); self.assertIn('unrecognized receipt status', p['effect']['reasons'][0])

    def test_exit_failure_and_changed_source_are_separate_from_status(self):
        run, path, case = self.runner_fixture(status='failed', returncode=7)
        r, p = self.run_case(case)
        self.assertEqual(p['process']['disposition'], 'exit_failed'); self.assertEqual(r['disposition'], 'failed')
        receipt = json.loads(path.read_bytes())
        receipt.update(status='completed', source_sha256_after='0' * 64, source_unchanged=False, returncode=0)
        self.write(path, receipt); case['result'] = ref(path); (run / 'inventory.json').write_bytes(b'{}')
        r, p = self.run_case(case)
        self.assertEqual(p['source_preservation']['disposition'], 'changed'); self.assertEqual(r['disposition'], 'needs_reconciliation')
        receipt.update(source=None, source_sha256_before=None, source_sha256_after=None, source_unchanged=True)
        self.write(path, receipt); case['result'] = ref(path)
        r, p = self.run_case(case)
        self.assertEqual(p['source_preservation']['disposition'], 'not_applicable'); self.assertEqual(r['disposition'], 'consistent_unverified')
        receipt.pop('returncode'); self.write(path, receipt); case['result'] = ref(path)
        r, p = self.run_case(case)
        self.assertEqual(p['process']['disposition'], 'unknown'); self.assertEqual(r['disposition'], 'needs_reconciliation')

    def test_bad_cases_are_refused_before_any_record(self):
        mutations = [lambda c: c.update(schema_version=2), lambda c: c['adapter'].pop('statuses'),
                     lambda c: c['adapter']['statuses'].update(odd=dict(effect='done', meaning='x')),
                     lambda c: c['adapter'].update(required_outputs={'in': 'result', 'directory': ['output'], 'names': ['../inventory.json']}),
                     lambda c: c['adapter'].update(identity_checks=[dict(id='x', result='status')]),
                     lambda c: c['adapter'].update(identity_checks=[dict(id='x', result=['status'])]),
                     lambda c: c['operation'].update(kind='other'), lambda c: c.pop('result'),
                     lambda c: c['adapter'].update(artifacts=[{'id': 'a', 'in': 'elsewhere', 'path': ['candidate']}]),
                     lambda c: c['adapter'].update(process={'in': 'result', 'returncode': ['returncode'], 'success_codes': ['0']}),
                     lambda c: c.update(independent_verification=[dict(path=str(self.result_path), sha256=digest(self.result_path.read_bytes()))])]
        for mutate in mutations:
            case = json.loads(canonical(self.case)); mutate(case)
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.run_case(case)
        self.assertEqual(list(self.s.store.records(['execution_receipt'])), [])
        self.write(self.receipt_path, dict(self.receipt, label='edited after the case pinned it'))
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            self.run_case()

    def test_pinned_documents_survive_later_changes_and_reads_are_bounded(self):
        r, p = self.run_case(); self.result_path.write_bytes(b'{}')
        self.assertNotEqual(self.s.store.resolve_blob(p['case']['result']['asset']).read_bytes(), b'{}')
        self.assertEqual(p['documents']['result']['status'], 'applied_trial')
        out = self.s.execute('read_record', r['reads']['effect']['arguments']); self.assertIn('applied', json.dumps(out))
        self.assertEqual(self.s.read_record(r['analysis'])['limits'], p['limits'])


if __name__ == '__main__':
    unittest.main()
