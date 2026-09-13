"""A fresh process retrieves and uses a narrowly supported diagnostic method."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path
from . import test_decisions as fixtures
from . import test_repair_analysis as repair_fixtures
from .store import canonical, digest


class MethodIntegrationTests(unittest.TestCase):
    setUp=fixtures.EpisodeTests.setUp
    tearDown=fixtures.EpisodeTests.tearDown
    ep=fixtures.EpisodeTests.episode

    def test_text_focus_reads_current_question_without_state_changes(self):
        e=self.ep();question=self.s.ledger.read(e['episode'])['intent']['question']
        def snapshot():return {str(p):digest(p.read_bytes()) for p in self.s.store.root.rglob('*') if p.is_file()}
        before=snapshot()
        from unittest.mock import patch
        with patch.object(self.s,'_native_call',side_effect=AssertionError('No native inspection')):
            result=self.s.execute('inspect_situation',{'question':'Diagnose repeated missing target constraints','live':False})
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['question'],question)
        self.assertIn('experience',result)
        self.assertEqual(snapshot(),before)
        self.assertEqual(self.s.inspect_situation(question)['question'],question)
        # A syntactically explicit but absent record stays an error, not a focus.
        self.assertEqual(self.s.execute('inspect_situation',{'question':'f'*64})['status'],'failed')

    def test_new_workspace_accepts_focus_without_inventing_geometry(self):
        from .service import ModelingService
        service=ModelingService(workspace=self.root,store=self.root/'empty-store')
        result=service.inspect_situation('Understand unsupported reconstructed surfaces')
        self.assertEqual(result['status'],'needs evidence')
        self.assertIn('experience',result);self.assertNotIn('state',result)

    def test_method_close_requires_integration_without_blocking_save_or_open_judgments(self):
        e=self.ep()
        args=dict(episode=e['episode'],expected_revision=e['revision'],character={'status':'unresolved','reason':'Appearance pending'},
            method={'status':'supported','reason':'Diagnostic separates missing evidence from an invalid solve'},
            evidence=[{'kind':'file','path':str(self.image),'role':'synthetic evidence'}],applicability='Synthetic diagnostic only')
        opened=self.s.reconcile_episode(**args)
        self.assertEqual(opened['method_integration']['disposition'],'pending')
        with self.assertRaisesRegex(ValueError,'Method close-out needs integration'):
            self.s.reconcile_episode(e['episode'],opened['revision'],close=True)
        self.assertEqual(self.s.ledger.read(e['episode'])['status'],'active')

    def test_evidenced_adoption_needs_receipts_and_hypothesis_is_not_implemented(self):
        from .learning import integration_disposition
        method=dict(status='supported',reason='Narrow diagnostic benefit',integration=dict(disposition='implemented',
            reason='Procedure retained',mechanism='missing target',scope='diagnosis',limits=['No native claim'],
            artifacts=[{'kind':'file','path':str(self.image),'role':'fixture procedure'}],
            adoption={stage:{'status':'evidenced','reason':'claimed'} for stage in ('source','installed','runtime','operator')}))
        with self.assertRaisesRegex(ValueError,'actual receipt links'):integration_disposition(self.s,method)
        method['status']='unresolved'
        with self.assertRaisesRegex(ValueError,'narrowly supported'):integration_disposition(self.s,method)

    def test_supported_diagnostic_promotes_retrieves_and_runs_with_character_unresolved(self):
        # Use the existing coupled XYZ analyzer. This demonstrates a diagnostic
        # gain, not a new solver or successful native fitting/appearance result.
        fixture=repair_fixtures.RepairTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        target=fixture.case['targets'][0];target['support']['status']='missing'
        other=copy.deepcopy(target);other.update(id='fit-Y',terms=[dict(output=0,direction=[0,1,0])])
        fixture.case['targets'].append(other)
        case=fixture.root/'case.json';case.write_bytes(canonical(fixture.case))
        result=self.s.analyze_repair(str(case));analysis=self.s.store.get(result['analysis'])
        self.assertEqual(len(analysis['acquisition_nominations']),1)
        self.assertEqual(analysis['acquisition_nominations'][0]['related_subjects'],['fit-X','fit-Y'])
        self.assertFalse(result['native_ready'])
        e=self.ep();question=self.s.ledger.read(e['episode'])['intent']['question']
        method=dict(status='supported',reason='Group repeated missing target constraints for the same native point while preserving exact diagnostic rows')
        character=dict(status='unresolved',reason='No native geometry, appearance or motion tested')
        outcome=self.s.record_outcome(question,character,method,[result['analysis']], 'Missing qualified target diagnosis; synthetic two-row evidence')
        evidence=[{'kind':'record','id':result['analysis'],'role':'exact diagnostic result'},
                  {'kind':'record','id':outcome['outcome'],'role':'separate character/method result'}]
        judgment=self.s.reconcile_episode(e['episode'],e['revision'],character=character,method=method,
            evidence=evidence,applicability='Repeated missing-target rows share object and native vertex')
        failed=fixture.root/'counterexample.json';failed.write_bytes(canonical(dict(
            alternative='One acquisition request per scalar constraint',status='rejected for this scope',
            reason='Both fixture constraints name the same unsupported native point; duplicate requests add no target evidence',
            limits='Distinct vertices and independent failure causes must stay separate')))
        recipe=fixture.root/'replay.json';recipe.write_bytes(canonical(dict(operation='analyze_repair',case='case.json',
            expected_disposition='needs_evidence_or_revision',expected_groups=1,exact_subjects=['fit-X','fit-Y'])))
        promoted=self.s.promote_procedure('group-missing-targets',[judgment['judgment']],method['reason'],['diagnosis'],
            conditions={'mechanism':'repeated missing target constraints','response_kind':'exact_linear'},
            limits=['Grouping is diagnostic only; never qualifies target shape or authorizes dispatch',
                    'Protected context: all influence layers and per-row provenance stay available',
                    'Artistic variables and depth are unresolved; missing/occluded support remains excluded'],
            counterexamples=[{'kind':'file','path':str(failed),'role':'failed alternative and applicability boundary'}],
            executable_paths=[str(recipe),str(case),str(fixture.root/'response.npz'),str(fixture.root/'native.npz')])
        self.assertEqual(promoted['level'],'local')
        # New interpreter, no conversation or prior object state: discover by
        # mechanism, inspect applicability, restore pinned inputs, use existing op.
        program='''import json,sys
from pathlib import Path
from modeling_system.service import ModelingService
s=ModelingService(workspace=sys.argv[1],store=sys.argv[2])
found=s.retrieve_experience('repeated missing target constraints',20,{'mechanism':'repeated missing target constraints','response_kind':'exact_linear'})
row=next(r for r in found['matches'] if r['excerpt'].get('procedure_id')=='group-missing-targets')
assert row['applicability']['status']=='applicable'
p=s.read_record(row['record']);assert p['level']=='local' and p['appearance_acceptance']=='not implied'
dest=Path(sys.argv[3]);dest.mkdir()
for link in p['executable']:
    (dest/Path(link['path']).name).write_bytes(s.store.resolve_blob(link['asset']).read_bytes())
recipe=json.loads((dest/'replay.json').read_bytes())
assert recipe['operation']=='analyze_repair'
r=s.execute(recipe['operation'],{'case_path':str(dest/recipe['case'])})
a=s.read_record(r['analysis'])
assert r['disposition']==recipe['expected_disposition'] and not r['native_ready']
assert len(a['acquisition_nominations'])==recipe['expected_groups']
assert a['acquisition_nominations'][0]['related_subjects']==recipe['exact_subjects']
Path(sys.argv[4]).write_text(json.dumps({'procedure':row['record'],'result':r,'runtime':s.runtime_status(),'scope':'fresh-process synthetic diagnostic only'}))
'''
        report=self.root/'fresh-use.json'
        run=subprocess.run([sys.executable,'-c',program,str(self.root),str(self.s.store.root),str(self.root/'replay'),str(report)],
            cwd=Path(__file__).resolve().parent.parent,capture_output=True,text=True,timeout=30)
        self.assertEqual(run.returncode,0,run.stderr)
        receipt=json.loads(report.read_bytes());self.assertEqual(receipt['procedure'],promoted['procedure'])
        proof={'kind':'file','path':str(report),'role':'fresh-process retrieval, actual operation result and observed runtime'}
        method['integration']=dict(disposition='implemented',reason='Local executable recipe recovered and used through existing tools',
            mechanism='repeated missing target constraints',scope='diagnostic grouping only',limits=promoted['limits'],
            artifacts=[{'kind':'record','id':promoted['procedure'],'role':'local retained procedure'}],
            adoption={'source':{'status':'evidenced','reason':'Retained procedure verified by fresh reader','evidence':[proof]},
                'installed':{'status':'not_applicable','reason':'Private local procedure uses existing analyzer; no app package switch'},
                'runtime':{'status':'evidenced','reason':'Actual isolated runtime receipt','evidence':[proof]},
                'operator':{'status':'evidenced','reason':'Fresh synthetic reader found and used exact recipe','evidence':[proof]}})
        closed=self.s.reconcile_episode(e['episode'],judgment['revision'],method=method,close=True)
        final=self.s.read_record(closed['judgment'])
        self.assertEqual(closed['status'],'closed')
        self.assertEqual(final['character']['status'],'unresolved')
        self.assertTrue(final['method']['integration']['adoption_complete'])
        # General reuse remains a distinct-evidence requirement.
        with self.assertRaisesRegex(ValueError,'distinct episodes'):
            self.s.promote_procedure('group-missing-targets',[judgment['judgment']]*2,method['reason'],['diagnosis'],{},['Synthetic scope'],level='reusable')


if __name__=='__main__':unittest.main()
