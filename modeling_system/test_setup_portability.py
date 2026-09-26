"""Installation and handoff boundaries on machines without the author's setup."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from .init_workspace import initialize
from .prepare_plugin import prepare
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
        self.assertEqual(result['skills'], ['modeling-workbench'])
        moved = self.root/'moved workspace'
        shutil.move(str(original), moved)
        self.assertFalse((moved/'.agents/skills/typesafe-ai').exists())
        self.assertTrue((moved/'.agents/skills/modeling-workbench/references/build-the-mechanism-first.md').is_file())
        binding = (moved/'modeling-workspace.json').read_text(encoding='utf-8')
        for path in (original, original.resolve()):
            for locator in (str(path), json.dumps(str(path))[1:-1], path.as_posix()):
                self.assertNotIn(locator, binding)
        with patch.dict(os.environ, {}, clear=True):
            report = inspect_setup(moved)
        self.assertTrue(report['core_ready'], report)
        self.assertEqual(report['inference_dependencies'], [])
        self.assertEqual(report['native']['status'], 'unconfigured')
        self.assertFalse(report['native']['live_verified'])

    def test_bundled_guidance_and_local_edits_are_preserved(self):
        self.assertEqual(verify_dependencies()['status'], 'verified')
        target = self.root/'skills'
        install_skills(target)
        skill = target/'modeling-workbench/SKILL.md'
        skill.write_text('Local customization', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            install_skills(target)
        self.assertEqual(skill.read_text(encoding='utf-8'), 'Local customization')


    def test_plugin_contains_complete_skill_and_preserves_configuration(self):
        workspace = self.root/'workspace'; initialize(workspace, 'Synthetic')
        plugin = prepare(self.root/'modeling-workbench', workspace)
        self.assertTrue((plugin/'skills/modeling-workbench/SKILL.md').is_file())
        from . import __version__
        manifest = json.loads((plugin/'.codex-plugin/plugin.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['version'], __version__)                    # the one version source
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




if __name__ == '__main__':
    unittest.main()
