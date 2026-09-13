"""A worker cannot exit complete without satisfying its declared completion contract."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .store import digest
from .worker_contract import ContractConflict, ContractError, ContractFailure, WorkerContract, verify_manifest


class WorkerContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.out = self.root / 'run'; self.out.mkdir()
        self.source = self.root / 'pinned-source.bin'; self.source.write_bytes(b'pinned source bytes')
        self.source_sha = digest(self.source.read_bytes())

    def contract(self, **overrides):
        options = dict(completion_file='inventory.json', required_artifacts=['arrays.npz', 'report.json'],
                       required_checks=['source_verified', 'geometry_exported'],
                       protected_sources=[dict(path=str(self.source), sha256=self.source_sha, role='pinned source')],
                       label='synthetic worker')
        options.update(overrides)
        return WorkerContract(self.out, **options)

    def produce(self, contract, arrays=b'arrays', report=b'{"ok": true}'):
        (self.out / 'arrays.npz').write_bytes(arrays); (self.out / 'report.json').write_bytes(report)
        contract.check('source_verified', True, detail={'sha256': self.source_sha}); contract.check('geometry_exported', True)

    def test_missing_or_invalid_declarations_are_caught_before_work(self):
        for bad in [dict(completion_file=None), dict(required_artifacts=None), dict(required_artifacts=[]),
                    dict(required_checks=None), dict(protected_sources=None), dict(required_artifacts=['inventory.json']),
                    dict(required_artifacts=['a.npz', 'a.npz']), dict(failure_file='inventory.json'),
                    dict(required_checks=['x', 'x']), dict(label=''),
                    dict(protected_sources=[dict(path='relative.bin', sha256=self.source_sha)]),
                    dict(protected_sources=[dict(path=str(self.source), sha256='abc')]),
                    dict(required_artifacts=[dict(name='copy.bin', sha256='ABC')]),
                    dict(required_artifacts=[dict(name='copy.bin', extra=1)])]:
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                self.contract(**bad)
        self.assertEqual(list(self.out.iterdir()), [])
        with self.assertRaises(ContractError):
            WorkerContract(self.root / 'absent', completion_file='inventory.json', required_artifacts=['a'],
                           required_checks=[], protected_sources=[]).preflight()

    def test_path_escapes_are_refused_at_declaration(self):
        for name in ['../escape.npz', '/abs.npz', 'C:/abs.npz', 'a\\b.npz', 'a/../b.npz', '.', '', 'x/./y', 'trailing. ',
                     'a:b', 'a/b/c/d/e/f/g/h/i', 'n' * 201]:
            with self.subTest(name=name), self.assertRaises(ContractError):
                self.contract(required_artifacts=[name])
        with self.assertRaises(ContractError):
            self.contract(completion_file='nested/inventory.json')
        contract = self.contract(); contract.preflight()
        with self.assertRaises(ContractError):
            contract.artifact('../outside.bin')
        with self.assertRaises(ContractError):
            contract.artifact('inventory.json')

    def test_linked_artifact_inside_output_is_refused(self):
        outside = self.root / 'outside.bin'; outside.write_bytes(b'outside')
        contract = self.contract(); contract.preflight()
        try:
            os.symlink(outside, self.out / 'arrays.npz')
        except (OSError, NotImplementedError):
            self.skipTest('Symbolic link creation is not permitted in this environment')
        (self.out / 'report.json').write_bytes(b'{}'); contract.check('source_verified', True); contract.check('geometry_exported', True)
        with self.assertRaises(ContractFailure) as caught:
            contract.finalize()
        self.assertEqual([f['kind'] for f in caught.exception.report['failures']], ['invalid_artifact_path'])
        self.assertFalse((self.out / 'inventory.json').exists())

    def test_preflight_verifies_sources_and_writes_a_report_for_changed_or_missing_sources(self):
        report = self.contract().preflight()
        self.assertEqual(report['sources'][0]['status'], 'verified'); self.assertFalse(report['existing_completion_file'])
        self.source.write_bytes(b'changed')
        with self.assertRaises(ContractFailure) as caught:
            self.contract().preflight()
        self.assertEqual(caught.exception.report['failures'][0]['kind'], 'changed_source')
        self.assertTrue(Path(caught.exception.report_path).is_file()); self.assertFalse((self.out / 'inventory.json').exists())
        with self.assertRaises(ContractFailure) as caught:
            self.contract(protected_sources=[dict(path=str(self.root / 'absent.bin'), sha256=self.source_sha)]).preflight()
        self.assertEqual(caught.exception.report['failures'][0]['kind'], 'missing_source')
        self.assertTrue((self.out / 'contract-failure-2.json').is_file())

    def test_finalize_refuses_missing_artifacts_unrecorded_and_failed_checks(self):
        contract = self.contract(); contract.preflight()
        (self.out / 'arrays.npz').write_bytes(b'arrays'); contract.check('source_verified', True)
        with self.assertRaises(ContractFailure) as caught:
            contract.finalize()
        self.assertEqual(sorted(f['kind'] for f in caught.exception.report['failures']), ['missing_artifact', 'unrecorded_check'])
        self.assertFalse((self.out / 'inventory.json').exists()); self.assertTrue((self.out / 'contract-failure.json').is_file())
        (self.out / 'report.json').write_bytes(b'{}'); contract.check('geometry_exported', False, detail='count mismatch')
        with self.assertRaises(ContractError):
            contract.check('geometry_exported', True)
        with self.assertRaises(ContractError):
            contract.check('other', 1)
        with self.assertRaises(ContractFailure) as caught:
            contract.finalize()
        self.assertEqual([f['kind'] for f in caught.exception.report['failures']], ['failed_check'])
        self.assertEqual(caught.exception.report['checks']['geometry_exported']['details'], ['count mismatch'])
        self.assertTrue((self.out / 'contract-failure-2.json').is_file())
        self.assertFalse((self.out / 'inventory.json').exists())
        soft = self.contract(); soft.check('source_verified', True)
        outcome = soft.finalize(strict=False)
        self.assertEqual(outcome['status'], 'failed'); self.assertTrue(Path(outcome['path']).is_file())

    def test_finalize_refuses_a_source_changed_after_preflight(self):
        contract = self.contract(); contract.preflight(); self.produce(contract)
        self.source.write_bytes(b'changed after preflight')
        with self.assertRaises(ContractFailure) as caught:
            contract.finalize()
        self.assertEqual([f['kind'] for f in caught.exception.report['failures']], ['changed_source'])
        self.assertEqual(caught.exception.report['artifacts']['arrays.npz']['status'], 'verified')
        self.assertFalse((self.out / 'inventory.json').exists())
        unchecked = self.contract(); self.produce(unchecked)
        with self.assertRaises(ContractFailure):
            unchecked.finalize()  # without preflight the same verification still happens

    def test_expected_artifact_hash_binds_a_copied_source(self):
        contract = self.contract(required_artifacts=['arrays.npz', 'report.json',
                                                     dict(name='source-copy.bin', sha256=self.source_sha, role='copied pinned source')])
        contract.preflight(); self.produce(contract); (self.out / 'source-copy.bin').write_bytes(b'different bytes')
        with self.assertRaises(ContractFailure) as caught:
            contract.finalize()
        self.assertEqual([f['kind'] for f in caught.exception.report['failures']], ['artifact_hash_mismatch'])
        (self.out / 'source-copy.bin').write_bytes(self.source.read_bytes())
        result = contract.finalize(extra={'status': 'source_inventoried'})
        self.assertEqual(result['manifest']['artifacts']['source-copy.bin']['sha256'], self.source_sha)
        self.assertEqual(result['manifest']['artifacts']['source-copy.bin']['role'], 'copied pinned source')

    def test_success_manifest_is_written_last_and_repeats_idempotently(self):
        contract = self.contract(); contract.preflight(); self.produce(contract)
        contract.artifact('extra.txt', role='log'); (self.out / 'extra.txt').write_bytes(b'log')
        result = contract.finalize(extra={'status': 'exported', 'case': 'synthetic'})
        path = self.out / 'inventory.json'
        self.assertTrue(path.is_file()); self.assertEqual(result['path'], str(path)); self.assertFalse(result['repeat'])
        manifest = json.loads(path.read_bytes())
        self.assertEqual(manifest['kind'], 'worker_completion'); self.assertEqual(manifest['contract_status'], 'satisfied')
        self.assertEqual(manifest['status'], 'exported'); self.assertEqual(manifest['label'], 'synthetic worker')
        self.assertEqual(manifest['artifacts']['arrays.npz']['sha256'], digest(b'arrays'))
        self.assertFalse(manifest['artifacts']['extra.txt']['required'])
        self.assertEqual(manifest['sources'][0]['sha256'], self.source_sha)
        self.assertEqual(manifest['checks']['source_verified']['details'], [{'sha256': self.source_sha}])
        self.assertTrue(manifest['contract']['preflight_run'])
        before = path.read_bytes()
        again = contract.finalize(extra={'status': 'exported', 'case': 'synthetic'})
        self.assertTrue(again['repeat']); self.assertEqual(path.read_bytes(), before)
        self.assertEqual(verify_manifest(path)['disposition'], 'verified')
        with self.assertRaises(ContractError):
            contract.finalize(extra={'artifacts': {}})
        with self.assertRaises(ContractError):
            contract.finalize(extra={'value': float('nan')})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['arrays.npz', 'extra.txt', 'inventory.json', 'report.json'])

    def test_repeat_with_a_different_outcome_preserves_the_existing_manifest(self):
        contract = self.contract(); contract.preflight(); self.produce(contract); contract.finalize()
        path = self.out / 'inventory.json'; before = path.read_bytes()
        second = self.contract(); second.preflight(); self.produce(second, arrays=b'regenerated arrays')
        with self.assertRaises(ContractConflict) as caught:
            second.finalize()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(caught.exception.report['failures'][0]['kind'], 'conflicting_existing_manifest')
        self.assertTrue((self.out / 'contract-failure.json').is_file())
        report = verify_manifest(path)
        self.assertEqual(report['disposition'], 'mismatch'); self.assertEqual(report['artifacts']['arrays.npz'], 'mismatch')

    def test_verify_manifest_refuses_traversal_and_recognizes_unrelated_files(self):
        target = self.out / 'inventory.json'
        target.write_text('{"status": "completed"}', encoding='utf-8')
        self.assertEqual(verify_manifest(target)['disposition'], 'not_contract_manifest')
        (self.root / 'outside.bin').write_bytes(b'outside')
        forged = dict(schema_version=1, kind='worker_completion', contract_status='satisfied', completion_file='inventory.json',
                      contract=dict(required_artifacts=[dict(name='../outside.bin')], required_checks=['never']),
                      artifacts={'../outside.bin': dict(path='../outside.bin', sha256=digest(b'outside'))}, checks={}, sources=[])
        target.write_text(json.dumps(forged), encoding='utf-8')
        report = verify_manifest(target)
        self.assertEqual(report['disposition'], 'mismatch'); self.assertEqual(report['artifacts']['../outside.bin'], 'invalid_name')
        self.assertFalse(report['checks_all_passed'])
        self.assertEqual(verify_manifest(self.out / 'absent.json')['disposition'], 'absent')
        target.write_text('not json', encoding='utf-8')
        self.assertEqual(verify_manifest(target)['disposition'], 'invalid')

    def test_protected_sources_are_independent_of_the_launcher_input(self):
        second = self.root / 'imported.fbx'; second.write_bytes(b'fbx bytes'); sha = digest(second.read_bytes())
        contract = self.contract(protected_sources=[dict(path=str(self.source), sha256=self.source_sha, role='launcher input'),
                                                    dict(path=str(second), sha256=sha, role='imported asset')])
        contract.preflight(); self.produce(contract); result = contract.finalize()
        self.assertEqual([s['role'] for s in result['manifest']['sources']], ['launcher input', 'imported asset'])
        self.assertEqual(result['manifest']['sources'][1]['sha256'], sha)
        self.assertEqual(result['manifest']['contract']['protected_sources'][1]['path'], str(second))
        none = self.contract(protected_sources=[]); self.assertEqual(none.preflight()['sources'], [])
        relative = self.contract(protected_sources=[dict(path='imported.fbx', sha256=sha)], source_root=self.root)
        self.assertEqual(relative.preflight()['sources'][0]['status'], 'verified')

    def test_fail_writes_a_report_and_no_manifest(self):
        contract = self.contract(); contract.preflight(); self.produce(contract)
        with self.assertRaises(ContractFailure) as caught:
            contract.fail('pose differs from the pinned expectation', detail={'control': 0.5})
        self.assertEqual(caught.exception.report['failures'][0]['kind'], 'worker_failure')
        self.assertEqual(caught.exception.report['stage'], 'worker')
        self.assertFalse((self.out / 'inventory.json').exists())

    def test_finalize_requires_preflight(self):
        contract=self.contract(); self.produce(contract)
        with self.assertRaises(ContractFailure) as caught:contract.finalize()
        self.assertIn('missing_preflight',[f['kind'] for f in caught.exception.report['failures']])
        self.assertFalse((self.out/'inventory.json').exists())

    def test_manifest_declared_contract_cannot_be_silently_weakened(self):
        contract=self.contract(); contract.preflight(); self.produce(contract); contract.finalize()
        path=self.out/'inventory.json'; original=json.loads(path.read_bytes())
        mutations=[lambda v:v.update(sources=[]),lambda v:v.update(sources=5),
            lambda v:v.update(completion_file='different.json'),
            lambda v:v['contract'].update(preflight_run=False),
            lambda v:v['contract'].update(required_checks=[{}]),
            lambda v:v['contract'].update(required_artifacts=5),
            lambda v:v['contract']['required_artifacts'][0].update(sha256='0'*64)]
        for mutate in mutations:
            value=json.loads(json.dumps(original)); mutate(value); path.write_text(json.dumps(value))
            with self.subTest(value=value):self.assertEqual(verify_manifest(path)['disposition'],'mismatch')

    def test_concurrent_completion_is_never_overwritten(self):
        contract=self.contract(); contract.preflight(); self.produce(contract)
        real_link=os.link; completion=self.out/'inventory.json'; competing=b'{"other":"worker"}'
        def race(src,dst):
            if Path(dst)==completion:completion.write_bytes(competing)
            return real_link(src,dst)
        with patch('modeling_system.worker_contract.os.link',side_effect=race):
            with self.assertRaises(ContractConflict):contract.finalize()
        self.assertEqual(completion.read_bytes(),competing)

    def test_module_loads_standalone_without_the_package(self):
        spec = importlib.util.spec_from_file_location('standalone_worker_contract', Path(__file__).with_name('worker_contract.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        contract = module.WorkerContract(self.out, completion_file='inventory.json', required_artifacts=['arrays.npz'],
                                         required_checks=[], protected_sources=[])
        contract.preflight()
        (self.out / 'arrays.npz').write_bytes(b'a')
        self.assertEqual(contract.finalize()['status'], 'completed')
        self.assertEqual(verify_manifest(self.out / 'inventory.json')['disposition'], 'verified')


if __name__ == '__main__':
    unittest.main()
