"""Portable binding/evidence integrity and installed-process provenance tests."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from .service import ModelingService
from . import test_decisions as fixtures

class PortableTests(unittest.TestCase):
    def setUp(self):
        fixtures.EpisodeTests.setUp(self)
        (self.root/'modeling-workspace.json').write_text(json.dumps({'schema_version':1,'character':'Example Character','store':'store',
            'native_adapter':'unconfigured','authority':[],'evidence':[{'path':str(self.image),'role':'fixture evidence','required':True}]}))
        self.s=ModelingService(workspace=self.root,store=self.s.store.root,native=self.s.native)
    tearDown=fixtures.EpisodeTests.tearDown

    def test_export_relocates_complete_episode_without_hidden_native_access(self):
        e=fixtures.EpisodeTests.episode(self)
        operation=self.s.run_episode_operation(e['episode'],'query_geometry',{'state':self.state,'object_name':'Face','query':'bounds','parameters':{}})
        out=self.root/'package.zip'
        r=self.s.export_package(str(out),e['episode'],True)
        self.assertTrue(out.is_file())
        dest=self.root/'relocated';dest.mkdir()
        with zipfile.ZipFile(out) as z:
            self.assertIn('package-manifest.json',z.namelist())
            self.assertIn('modeling_system/plugin/skills/modeling-workbench/references/episodes.md',z.namelist())
            z.extractall(dest)
        new=ModelingService(workspace=dest)
        view=new.decision_workspace(e['episode'])
        self.assertEqual(view['context']['requirements'][0]['category'],'provisional_constraint')
        self.assertEqual(view['character'],'Example Character')
        self.assertIsNone(new.store.current())
        situation=new.inspect_situation()
        self.assertEqual(situation['question'],new.ledger.read(e['episode'])['intent']['question'])
        self.assertEqual(situation['selection_basis']['episode'],e['episode'])
        self.assertIsNone(new.store.current())
        npz=new.wb.arrays(view['baseline']['state'],'Face')
        self.assertEqual(len(npz['co']),3)
        with self.assertRaisesRegex(ValueError,'unconfigured'):
            new.native.call('inspect_live',{})
        self.assertEqual(new.binding['native_adapter'],'unconfigured')
        self.assertIn(operation['operation_handle'],[c['handle'] for c in new.inspect_operations(e['episode'])['calls']])
        ready=new.package_readiness(manifest_path=str(dest/'evidence-manifest.json'))
        self.assertTrue(ready['usable_for_recorded_analysis'],str(ready['missing']))
        embedded=json.loads((dest/'evidence-manifest.json').read_text())
        original=next(iter(embedded['assets'].values()))
        Path(original['locator']).write_bytes(b'original workspace is no longer authoritative for relocated package')
        self.assertTrue(new.package_readiness(manifest_path=str(dest/'evidence-manifest.json'))['usable_for_recorded_analysis'])
        with self.assertRaises(FileExistsError):self.s.export_package(str(out),e['episode'],True)

    def test_unbound_other_character_reports_missing_semantics_not_character_assumptions(self):
        project=self.root/'another';project.mkdir()
        (project/'modeling-workspace.json').write_text(json.dumps({'schema_version':1,'character':'Example','store':'records','native_adapter':'unconfigured','authority':[],
             'evidence':[{'path':'private.blend','role':'native geometry','required':True,'access':'private'}]}))
        s=ModelingService(workspace=project)
        r=s.package_readiness()
        self.assertEqual(r['character'],'Example');self.assertEqual(len(r['missing']),1)
        self.assertFalse(r['usable_for_recorded_analysis'])
        self.assertIn('unproven',r['transfer'])
        self.assertEqual(s.decision_workspace()['total'],0)

    def test_manifest_hash_change_is_not_fresh_evidence(self):
        e=fixtures.EpisodeTests.episode(self)
        m=self.s.evidence_manifest(e['episode'])
        path=self.root/'manifest.json';path.write_text(json.dumps(m))
        asset=next(iter(m['assets'].values()))
        Path(asset['locator']).write_bytes(b'changed stored evidence')
        r=self.s.package_readiness(manifest_path=str(path))
        self.assertTrue(r['changed']);self.assertFalse(r['usable_for_recorded_analysis'])

    def test_cache_mismatch_refuses_launch_and_does_not_accept_label(self):
        cache=self.root/'cache';(cache/'.codex-plugin').mkdir(parents=True)
        (cache/'.codex-plugin/plugin.json').write_text(json.dumps({'name':'modeling-workbench','version':'current-trust-me'}))
        (cache/'.mcp.json').write_text(json.dumps({'mcpServers':{'modeling-workbench':{'command':'must-not-run','args':[]}}}))
        result=self.s.verify_installation(str(cache))
        self.assertEqual(result['status'],'mismatch');self.assertTrue(result['mismatches'])
        self.assertNotIn('probe',result)

    def test_relocated_policy_remains_bound_without_authorizing_dispatch(self):
        policy=self.root/'policy.json';policy.write_text(json.dumps({'tripo':{'mode':'Smart Mesh','model':'P2.0'}}))
        binding=json.loads((self.root/'modeling-workspace.json').read_text());binding['authority']=[{'path':'policy.json','role':'provider policy'}]
        (self.root/'modeling-workspace.json').write_text(json.dumps(binding))
        original=ModelingService(workspace=self.root,store=self.s.store.root)
        path=self.root/'policy-export.zip';original.export_package(str(path),include_evidence=True)
        destination=self.root/'policy-destination'
        with zipfile.ZipFile(path) as z:z.extractall(destination)
        moved=ModelingService(workspace=destination)
        result=moved.select_generation_route()
        self.assertEqual(result['model'],'P2.0');self.assertFalse(result['dispatch_authorized'])
        self.assertTrue(moved.policy_path.is_file())

    def test_unindexed_return_can_be_exported_for_recovery_without_replaying(self):
        e=fixtures.EpisodeTests.episode(self);put=self.s.store.put
        def fail_index(kind,*args,**kwargs):
            if kind=='operation_fact':raise OSError('Index unavailable')
            return put(kind,*args,**kwargs)
        with patch.object(self.s.store,'put',fail_index):
            r=self.s.run_episode_operation(e['episode'],'query_geometry',dict(state=self.state,object_name='Face',query='bounds',parameters={}))
        path=self.root/'recover.zip';manifest=self.s.export_package(str(path),e['episode'],True)['manifest']
        self.assertEqual(manifest['missing'],[]);self.assertTrue(manifest['pending_operations'])
        destination=self.root/'recover'
        with zipfile.ZipFile(path) as z:z.extractall(destination)
        moved=ModelingService(workspace=destination)
        recovered=moved.reconcile_operation(r['operation_handle'])
        self.assertEqual(recovered['outcome']['status'],'returned')

    def test_exported_full_links_round_trip_with_all_original_workspace_access_blocked(self):
        e=fixtures.EpisodeTests.episode(self);archive=self.root/'full-links.zip'
        self.s.export_package(str(archive),e['episode'],True)
        destination=self.root/'relocated'
        with zipfile.ZipFile(archive) as z:z.extractall(destination)
        moved=ModelingService(workspace=destination);attempts=[]
        original_is_file=Path.is_file;original_read_bytes=Path.read_bytes;original_open=Path.open
        def guard(path):
            resolved=path.resolve()
            if resolved.is_relative_to(self.root) and not resolved.is_relative_to(destination):
                attempts.append(str(resolved));raise PermissionError('Original workspace access blocked')
        def is_file(path):guard(path);return original_is_file(path)
        def read_bytes(path):guard(path);return original_read_bytes(path)
        def open_path(path,*args,**kwargs):guard(path);return original_open(path,*args,**kwargs)
        with patch.object(Path,'is_file',is_file),patch.object(Path,'read_bytes',read_bytes),patch.object(Path,'open',open_path):
            for detail in ('links','full'):
                view=moved.decision_workspace(e['episode'],detail=detail)
                links=view['context']['links']
                result=moved.execute('revise_episode',{'episode':e['episode'],'expected_revision':view['revision'],'patch':{'links':links}})
                self.assertEqual(result['status'],'active',result)
                self.assertEqual(moved.decision_workspace(e['episode'],detail='links')['context']['links'],links)
        self.assertEqual(attempts,[])

    def test_authority_relocation_is_distinct_from_actual_content_change(self):
        path=self.root/'scope.md';path.write_text('Original scope')
        binding=json.loads((self.root/'modeling-workspace.json').read_text())
        binding['authority']=[{'id':'scope','path':'scope.md','role':'current scope'}]
        (self.root/'modeling-workspace.json').write_text(json.dumps(binding))
        e=fixtures.EpisodeTests.episode(self)
        out=self.root/'authority.zip';self.s.export_package(str(out),e['episode'],True)
        dest=self.root/'authority-destination'
        with zipfile.ZipFile(out) as z:z.extractall(dest)
        moved=ModelingService(workspace=dest)
        view=moved.decision_workspace(e['episode'])
        self.assertEqual(view['authority_changed_since_entry'],[])
        self.assertEqual(len(view['authority_location_rebindings']),1)
        (dest/'bound-authority/scope.md').write_text('Changed scope')
        view=moved.decision_workspace(e['episode'])
        self.assertEqual(len(view['authority_changed_since_entry']),1)
        self.assertEqual(view['authority_changed_since_entry'][0]['comparison'],'content changed')

if __name__=='__main__':unittest.main()
