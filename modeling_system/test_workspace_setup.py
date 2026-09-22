"""Private workspace isolation, portable source export and explicit adapter boundaries."""
import json,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .init_workspace import initialize
from .service import ModelingService
from .store import digest,canonical


class WorkspaceSetupTests(unittest.TestCase):
    def test_two_characters_recover_independently_with_distinct_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); results=[]
            for label,scale,object_name in [('Robot',1.,'Shell'),('Creature',3.,'Surface')]:
                work=root/label; report=initialize(work,label); s=ModelingService(workspace=work)
                self.assertTrue(all((work/f).is_dir() for f in report['folders']))
                self.assertEqual(s.binding['native_adapter'],'unconfigured')
                a={'co':np.array([[0.,0,0],[scale,0,0],[0,scale,0]]),'tri':np.array([[0,1,2]],dtype=np.int32)}
                arrays=work/'originals/geometry.npz'; np.savez_compressed(arrays,**a)
                state={'objects':[{'name':object_name,'type':'MESH','arrays':str(arrays),'geometry_hash':digest(canonical({k:digest(v.tobytes()) for k,v in a.items()})),'source':{}}],
                    'controls':{},'selected_guide':None,'references':{}}
                path=work/'observations/state.json'; path.write_text(json.dumps({'state':state,'state_id':digest(canonical(state)),'coverage':[{'included':True,'object':object_name,'scope':'one synthetic surface'}]}))
                imported=s.import_scene(str(path)); q=s.open_question('Inspect form',imported['state'],'whole','Establish baseline')
                e=s.open_episode(q['question'],'synthetic owner','recorded only',{'stage':'orientation'},'first')
                recovered=ModelingService(workspace=work); view=recovered.decision_workspace(e['episode'])
                self.assertEqual(view['character'],label)
                self.assertEqual(recovered.wb.arrays(imported['state'],object_name)['co'][1,0],scale)
                results.append((e['episode'],s.store.root))
            self.assertNotEqual(results[0],results[1])
            self.assertEqual(ModelingService(workspace=root/'Robot').decision_workspace()['total'],1)
            with self.assertRaises(FileExistsError):initialize(root/'Robot','Overwrite')

    def test_tools_export_cannot_include_private_binding_or_history(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); work=root/'private'; initialize(work,'PRIVATE_IDENTITY_MARKER')
            (work/'PROJECT.md').write_text('PRIVATE_HISTORY_MARKER')
            s=ModelingService(workspace=work)
            with patch('modeling_system.portable.evidence_manifest',side_effect=AssertionError('Must not inspect private evidence')):
                out=s.export_package(str(root/'tools.zip'))
            self.assertEqual(out['kind'],'tools_only')
            with zipfile.ZipFile(root/'tools.zip') as z:
                self.assertNotIn('modeling-workspace.json',z.namelist())
                self.assertFalse(any('bound-authority' in n or 'evidence-manifest' in n for n in z.namelist()))
                # Markers occur in this test source; inspect binding/history-bearing files separately.
                manifest=json.loads(z.read('package-manifest.json')); self.assertFalse(manifest['workspace_data_included'])
                self.assertFalse(any(n.startswith(('evidence/','external-evidence/')) for n in z.namelist()))

    def test_unlisted_package_file_cannot_enter_tools_export(self):
        from . import distribution
        import shutil
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; source.mkdir()
            actual=Path(distribution.__file__).parent
            selected=json.loads((actual/'distribution-files.json').read_text())['files']
            for name in selected:
                dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(actual/name,dest)
            (source/'private-notes.json').write_text('PRIVATE_HISTORY_MARKER')
            with patch.object(distribution,'__file__',str(source/'distribution.py')):
                distribution.export_tools(root/'tools.zip')
            with zipfile.ZipFile(root/'tools.zip') as z:
                self.assertNotIn('modeling_system/private-notes.json',z.namelist())

    def test_adapter_dependency_changes_are_refused_before_execution(self):
        from .blender_operations import execute
        with tempfile.TemporaryDirectory() as d:
            work=Path(d); initialize(work,'Synthetic')
            entry=work/'runtime/adapter.py'; entry.write_text("def execute(operation, arguments):\n    return {'ok':True,'result':{'scope':'synthetic'}}\n")
            binding=json.loads((work/'modeling-workspace.json').read_text())
            binding.update(native_adapter='blender_json_v1',native_configuration={'entrypoint':{'path':'runtime/adapter.py','sha256':digest(entry.read_bytes())},'operations':['inspect_live']})
            (work/'modeling-workspace.json').write_text(json.dumps(binding))
            self.assertTrue(execute('inspect_live',{'_workspace':str(work)})['ok'])
            entry.write_text('raise AssertionError("Do not execute changed code")')
            self.assertEqual(execute('inspect_live',{'_workspace':str(work)})['error']['code'],'adapter_changed')
            self.assertEqual(execute('set_controls',{'_workspace':str(work),'expected_state':'x','controls':{}})['error']['code'],'adapter_unconfigured')

    def test_plugin_uses_explicit_interpreter_and_workspace(self):
        from .prepare_plugin import prepare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); initialize(root/'project','Synthetic')
            dest=prepare(root/'plugin',root/'project')
            config=json.loads((dest/'.mcp.json').read_text())['mcpServers']['modeling-workbench']
            self.assertEqual(config['args'][-1],str((root/'project').resolve()))
            self.assertTrue(Path(config['command']).is_file())
            self.assertNotIn('Scripts',config['args'][0])
            with self.assertRaises(ValueError):prepare(root/'other')
