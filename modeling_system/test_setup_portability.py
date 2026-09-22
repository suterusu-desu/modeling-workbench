"""Installation and handoff boundaries on machines without the author's setup."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from .credentials import credential_status, read_key
from .init_workspace import initialize
from .prepare_plugin import prepare
from .provider_dispatch import initialize_ledger, dispatch_many
from .provider_recovery import budget_limits
from .setup_check import inspect_setup
from .skill_bundle import install_skills, verify_dependencies


class SetupPortabilityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_workspace_carries_complete_skills_after_relocation(self):
        original = self.root/'original'
        result = initialize(original, 'Synthetic character')
        self.assertEqual(result['skills'], ['modeling-workbench', 'typesafe-ai'])
        moved = self.root/'moved workspace'
        shutil.move(str(original), moved)
        self.assertTrue((moved/'.agents/skills/typesafe-ai/LICENSE').is_file())
        self.assertTrue((moved/'.agents/skills/modeling-workbench/references/cooperation.md').is_file())
        self.assertNotIn(str(original), (moved/'modeling-workspace.json').read_text(encoding='utf-8'))
        with patch.dict(os.environ, {}, clear=True):
            report = inspect_setup(moved)
        self.assertTrue(report['core_ready'], report)
        self.assertFalse(report['jev']['credentials']['available'])
        self.assertEqual(report['native']['status'], 'unconfigured')
        self.assertFalse(report['native']['live_verified'])

    def test_bundled_upstream_integrity_and_local_edits_are_preserved(self):
        self.assertEqual(verify_dependencies()['status'], 'verified')
        target = self.root/'skills'
        install_skills(target)
        skill = target/'typesafe-ai/SKILL.md'
        skill.write_text('Local customization', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            install_skills(target)
        self.assertEqual(skill.read_text(encoding='utf-8'), 'Local customization')

    def test_credentials_are_explicit_and_diagnostics_never_read_them(self):
        keyfile = self.root/'credential.txt'
        keyfile.write_text('TYPESAFE_API_KEY="fixture-secret"', encoding='utf-8')
        with patch.dict(os.environ, {'TYPESAFE_API_KEY_FILE': str(keyfile),
                                     'TYPESAFE_API_KEY': 'unused-fixture'}, clear=True):
            with patch.object(Path, 'read_text', side_effect=AssertionError('No secret reads')):
                status = credential_status()
            self.assertEqual(status, {'source': 'TYPESAFE_API_KEY_FILE', 'available': True})
            self.assertNotIn(str(keyfile), json.dumps(status))
            self.assertEqual(read_key(), 'fixture-secret')
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(Path, 'read_text', side_effect=AssertionError('No machine fallback')):
            with self.assertRaises(ValueError):
                read_key()

    def test_plugin_contains_both_skills_and_preserves_configuration(self):
        workspace = self.root/'workspace'; initialize(workspace, 'Synthetic')
        plugin = prepare(self.root/'modeling-workbench', workspace)
        self.assertTrue((plugin/'skills/typesafe-ai/SKILL.md').is_file())
        (plugin/'.mcp.json').write_text('local configuration', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            prepare(plugin, workspace)
        self.assertEqual((plugin/'.mcp.json').read_text(encoding='utf-8'), 'local configuration')

    @unittest.skipIf(os.name == 'nt', 'POSIX venv symlink behavior')
    def test_python_symlink_keeps_selected_virtual_environment(self):
        python = self.root/'venv/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        workspace = self.root/'workspace'; initialize(workspace, 'Synthetic')
        plugin = prepare(self.root/'plugin', workspace, python)
        command = json.loads((plugin/'.mcp.json').read_text())['mcpServers']['modeling-workbench']['command']
        self.assertEqual(command, str(python.absolute()))
        self.assertNotEqual(command, str(python.resolve()))

    def test_ledger_requires_authority_and_cannot_reset_accounting(self):
        ledger = self.root/'ledger'
        with self.assertRaises(ValueError):
            initialize_ledger(ledger, authority='')
        self.assertFalse(ledger.exists())
        record = initialize_ledger(ledger, authority='Authorized synthetic work')
        self.assertEqual(budget_limits(record), (float('inf'), float('inf')))
        before = (ledger/'budget.json').read_bytes()
        with self.assertRaises(FileExistsError):
            initialize_ledger(ledger, authority='Do not reset')
        self.assertEqual((ledger/'budget.json').read_bytes(), before)

    def test_missing_credentials_do_not_reserve_or_dispatch_a_request(self):
        ledger = self.root/'ledger'; initialize_ledger(ledger, authority='Synthetic work')
        packet = self.root/'packet.json'
        packet.write_text(json.dumps({'state': {'observation': 'synthetic'},
            'questions': {'q': {'type': 'noul', 'instructions': 'Is support available?'}},
            'local_binding': {'owner': 'owner', 'authority_revision': 'scope',
                'dependencies': {'reads': {'source': 'v1'}, 'writes': []}}}))
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                dispatch_many([packet], ledger, lambda: {'values': {'source': 'v1'}, 'active_operations': []})
        self.assertEqual(json.loads((ledger/'budget.json').read_text())['attempts'], [])
        self.assertFalse((ledger/'dispatch.lock').exists())


if __name__ == '__main__':
    unittest.main()
