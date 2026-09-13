"""Actual offline analyses composed through recoverable recipe steps."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from . import test_decisions as episode_cases
from . import test_control_coverage as coverage_cases
from . import test_repair_analysis as repair_cases
from .service import ModelingService
from .store import canonical, digest


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.base=episode_cases.EpisodeTests(); self.base.setUp(); self.addCleanup(self.base.tearDown)
        self.s=self.base.s; self.episode=self.base.episode()['episode']; self.root=self.base.root
        self.coverage=coverage_cases.ControlCoverageTests(); self.coverage.setUp(); self.addCleanup(self.coverage.doCleanups)
        self.repair=repair_cases.RepairTests(); self.repair.setUp(); self.addCleanup(self.repair.doCleanups)
        self.cp=self.coverage.root/'case.json'; self.cp.write_bytes(canonical(self.coverage.case))
        self.rp=self.repair.root/'case.json'; self.rp.write_bytes(canonical(self.repair.case))
        self.template=dict(schema_version=1,name='Diagnostic review',inputs={'coverage':'case','state':'json','repair':'case','view_note':'json'},steps=[
            dict(id='coverage',kind='operation',operation='inspect_control_coverage',arguments={'case_path':{'$input':'coverage'},'expected_state':{'$input':'state'}}),
            dict(id='scope_review',kind='review',question='Are these inputs useful for this diagnostic?',evidence_from=[{'$step':'coverage','path':['analysis']}]),
            dict(id='repair',kind='operation',operation='analyze_repair',arguments={'case_path':{'$input':'repair'}},depends_on=['scope_review']),
            dict(id='result_review',kind='review',question='Is the candidate diagnostic useful?',evidence_from=[{'$step':'repair','path':['analysis']}],
                 checks=[dict(value={'$step':'repair','path':['disposition']},equals='candidate')])])
        self.bindings=dict(coverage=self.ref(self.cp),state=self.coverage.case['state'],repair=self.ref(self.rp),view_note='initial view')

    def ref(self,path):return dict(path=str(path),sha256=digest(path.read_bytes()))
    def create(self,key='one'):
        return self.s.create_recipe(self.episode,self.template,self.bindings,key)
    def step(self,state,name):return self.s.run_recipe_step(state['recipe'],name,state['revision'])
    def rows(self,state):return {r['id']:r for r in state['steps']}
    def review(self,state,decision='accepted',reason='Synthetic diagnostic use only'):
        return self.s.review_recipe_step(state['recipe'],'scope_review',state['revision'],decision,reason,
            [{'kind':'file','path':str(self.base.image),'role':'synthetic review evidence'}])
    def calls(self):return list((self.s.store.root/'calls').glob('*/intent.json'))

    def test_real_steps_are_linked_and_reuse_has_no_second_dispatch(self):
        state=self.create();self.assertEqual(self.rows(state)['repair']['status'],'blocked')
        state=self.step(state,'coverage');n=len(self.calls())
        again=self.step(state,'coverage');self.assertEqual(len(self.calls()),n)
        self.assertEqual(self.rows(again)['coverage']['status'],'reusable')
        handle=self.rows(state)['coverage']['operation_handle']
        intent=json.loads((self.s.store.root/'calls'/handle/'intent.json').read_bytes())
        self.assertEqual(intent['episode'],self.episode)
        self.assertEqual(intent['operation'],'inspect_control_coverage')
        state=self.review(state);state=self.step(state,'repair')
        self.assertEqual(self.rows(state)['result_review']['status'],'awaiting_review')
        record=self.s.store.get(self.rows(state)['repair']['result'],'recipe_step_result')
        self.assertEqual(record['disposition'],'candidate');self.assertFalse(record['native_ready'])

    def test_pending_and_rejected_review_block_downstream(self):
        state=self.step(self.create(),'coverage')
        with self.assertRaisesRegex(ValueError,'not ready'):self.step(state,'repair')
        state=self.review(state,'rejected')
        self.assertEqual(self.rows(state)['repair']['blocked_by'],['scope_review'])
        with self.assertRaisesRegex(ValueError,'not ready'):self.step(state,'repair')

    def test_changed_review_invalidates_downstream_use(self):
        state=self.step(self.review(self.step(self.create(),'coverage')),'repair')
        state=self.review(state,'rejected','Later diagnostic correction')
        self.assertEqual(self.rows(state)['repair']['status'],'blocked')
        state=self.review(state,'accepted','Revised interpretation with explicit evidence')
        self.assertEqual(self.rows(state)['repair']['status'],'stale')

    def test_changed_input_invalidates_affected_steps_only(self):
        state=self.step(self.review(self.step(self.create(),'coverage')),'repair')
        state=self.s.revise_recipe_inputs(state['recipe'],state['revision'],{'view_note':'different navigation'})
        self.assertEqual(self.rows(state)['coverage']['status'],'reusable')
        self.assertEqual(self.rows(state)['repair']['status'],'reusable')
        self.coverage.case['question']='A changed target-domain question';self.cp.write_bytes(canonical(self.coverage.case))
        state=self.s.revise_recipe_inputs(state['recipe'],state['revision'],{'coverage':self.ref(self.cp)})
        self.assertEqual(self.rows(state)['coverage']['status'],'stale')
        self.assertEqual(self.rows(state)['scope_review']['status'],'blocked')
        state=self.step(state,'coverage')
        self.assertEqual(self.rows(state)['scope_review']['status'],'stale')

    def test_unsupported_targets_cannot_pass_result_gate(self):
        self.repair.case['targets'][0]['support']['status']='missing';self.rp.write_bytes(canonical(self.repair.case))
        self.bindings['repair']=self.ref(self.rp)
        state=self.step(self.review(self.step(self.create(),'coverage')),'repair')
        self.assertEqual(self.rows(state)['result_review']['status'],'blocked')
        with self.assertRaisesRegex(ValueError,'not ready'):
            self.s.review_recipe_step(state['recipe'],'result_review',state['revision'],'accepted','Override',
                [{'kind':'file','path':str(self.base.image),'role':'fixture'}])

    def test_crash_after_dispatch_recovers_durable_result_without_replay(self):
        state=self.create()
        with patch('modeling_system.recipes._finish',side_effect=OSError('recipe index interrupted')):
            with self.assertRaises(OSError):self.step(state,'coverage')
        state=self.s.inspect_recipe(state['recipe']);self.assertEqual(self.rows(state)['coverage']['status'],'needs_recovery')
        n=len(self.calls())
        with patch.object(self.s,'_run_episode_operation',side_effect=AssertionError('No replay')):
            state=self.s.recover_recipe_step(state['recipe'],'coverage',state['revision'])
        self.assertEqual(self.rows(state)['coverage']['status'],'reusable')
        # Only the recovery fact itself is added, never another analysis.
        self.assertEqual(len(self.calls()),n+1)

    def test_missing_result_stays_uncertain_and_blocks_revision(self):
        state=self.create()
        with patch.object(self.s,'execute',side_effect=RuntimeError('interrupted before result')):
            with self.assertRaises(RuntimeError):self.step(state,'coverage')
        state=self.s.inspect_recipe(state['recipe'])
        recovery=self.s.recover_recipe_step(state['recipe'],'coverage',state['revision'])
        self.assertEqual(self.rows(recovery)['coverage']['status'],'needs_recovery')
        with self.assertRaisesRegex(RuntimeError,'uncertain'):
            self.s.revise_recipe_inputs(state['recipe'],state['revision'],{'view_note':'change'})

    def test_bad_graphs_refused_before_analysis(self):
        for mutate in (lambda t:t['steps'][0].update(operation='native_inspect_live'),
                       lambda t:t['steps'][0].update(depends_on=['repair']),
                       lambda t:t['steps'][1].update(evidence_from=[{'$step':'coverage','path':['absent']}]),
                       lambda t:t['steps'][0]['arguments'].update(case_path=str(self.cp)),
                       lambda t:t['steps'][0]['arguments'].update(case_path={'$input':'view_note'}),
                       lambda t:t['steps'][1].update(evidence_from=['unrelated literal'])):
            t=copy.deepcopy(self.template);mutate(t)
            with patch.object(self.s,'_run_episode_operation',side_effect=AssertionError('Must not dispatch')):
                with self.assertRaises((ValueError,TypeError)):
                    self.s.create_recipe(self.episode,t,self.bindings,'bad')

    def test_instances_are_scoped_and_inspection_is_read_only(self):
        one=self.create('one');two=self.create('two');self.assertNotEqual(one['recipe'],two['recipe'])
        self.step(one,'coverage');self.assertEqual(self.rows(self.s.inspect_recipe(two['recipe']))['coverage']['status'],'ready')
        before={str(p):digest(p.read_bytes()) for p in self.s.store.root.rglob('*') if p.is_file()}
        self.s.execute('inspect_recipe',{'recipe':two['recipe']})
        after={str(p):digest(p.read_bytes()) for p in self.s.store.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        other=ModelingService(workspace=self.root/'other',store=self.s.store.root)
        with self.assertRaisesRegex(ValueError,'different workspace'):other.inspect_recipe(one['recipe'])

    def test_bundled_review_tracks_downstream_input_and_exact_evidence(self):
        self.template=self.s.recipe_template()['template']
        self.bindings.pop('view_note')
        state=self.review(self.step(self.create(),'coverage'))
        review=self.s.store.get(self.rows(state)['scope_review']['result'],'recipe_step_result')
        self.assertEqual(review['basis']['upstream_evidence'][0]['arguments']['record'],self.rows(state)['coverage']['result'])
        self.repair.case['question']='A different repair question';self.rp.write_bytes(canonical(self.repair.case))
        state=self.s.revise_recipe_inputs(state['recipe'],state['revision'],{'repair':self.ref(self.rp)})
        self.assertEqual(self.rows(state)['coverage']['status'],'reusable')
        self.assertEqual(self.rows(state)['scope_review']['status'],'stale')
        self.assertEqual(self.rows(state)['repair']['status'],'blocked')

    def test_stale_revision_cannot_dispatch(self):
        state=self.create();old=state['revision']
        state=self.s.revise_recipe_inputs(state['recipe'],old,{'view_note':'changed'})
        with patch.object(self.s,'_run_episode_operation',side_effect=AssertionError('Must not dispatch')):
            with self.assertRaisesRegex(RuntimeError,'changed'):
                self.s.run_recipe_step(state['recipe'],'coverage',old)

    def test_default_workspace_exposes_recipe_steps_and_inspection_route(self):
        state=self.create()
        row=next(w for w in self.s.decision_workspace(episode=self.episode)['workflows'] if w['handle']==state['recipe'])
        self.assertEqual(row['next_read'],dict(operation='inspect_recipe',arguments=dict(recipe=state['recipe'])))
        self.assertEqual(row['steps'][0]['status'],'ready')

    def test_recipe_reserved_before_lease_blocks_episode_closure(self):
        state=self.create()
        with patch.object(self.s,'_run_episode_operation',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.step(state,'coverage')
        view=self.s.decision_workspace(episode=self.episode)
        row=next(w for w in view['workflows'] if w['handle']==state['recipe'])
        self.assertEqual(row['status'],'needs_reconciliation')
        with self.assertRaisesRegex(RuntimeError,'Close requires reconciled'):
            self.s.reconcile_episode(self.episode,view['revision'],
                character={'status':'unresolved','reason':'Offline only'},
                method={'status':'unresolved','reason':'Interrupted recipe'},
                evidence=[{'kind':'file','path':str(self.base.image),'role':'fixture'}],
                applicability='Synthetic reserved attempt',close=True)

    def test_selected_step_does_not_materialize_unrelated_case(self):
        state=self.create()
        item=self.s.ledger.read(state['recipe'])
        pinned=item['data']['bindings']['repair']['pinned']
        def first_asset(value):
            if isinstance(value,dict):
                if value.get('asset'):return value['asset']
                for v in value.values():
                    found=first_asset(v)
                    if found:return found
            elif isinstance(value,list):
                for v in value:
                    found=first_asset(v)
                    if found:return found
        asset=first_asset(pinned)
        self.s.store.resolve_blob(asset).unlink()
        state=self.step(state,'coverage')
        self.assertEqual(self.rows(state)['coverage']['status'],'reusable')
        state=self.review(state)
        with self.assertRaises(FileNotFoundError):self.step(state,'repair')

    def test_dispatch_reservation_failure_does_not_execute(self):
        state=self.create()
        original=self.s.ledger.update
        def fail_handle(handle,revision,status,data):
            if handle==state['recipe'] and any(a.get('handle') for rows in data.get('attempts',{}).values() for a in rows):
                raise OSError('Handle could not be retained')
            return original(handle,revision,status,data)
        with patch.object(self.s.ledger,'update',side_effect=fail_handle),patch.object(self.s,'execute',side_effect=AssertionError('Must not dispatch')):
            with self.assertRaisesRegex(OSError,'Handle'):self.step(state,'coverage')
        self.assertEqual(self.rows(self.s.inspect_recipe(state['recipe']))['coverage']['status'],'failed')

    def test_fresh_process_resumes_from_pinned_inputs_after_originals_disappear(self):
        state=self.create();self.cp.unlink();(self.coverage.root/'ancestry.npz').unlink()
        script=('import json,sys; sys.path.insert(0,'+repr(str(Path(__file__).resolve().parents[1]))+'); from modeling_system.service import ModelingService; '
            's=ModelingService(workspace='+repr(str(self.s.workspace))+',store='+repr(str(self.s.store.root))+'); '
            'r=s.inspect_recipe('+repr(state['recipe'])+'); '
            'print(json.dumps(s.run_recipe_step(r["recipe"],"coverage",r["revision"])))')
        result=subprocess.run([sys.executable,'-I','-c',script],capture_output=True,text=True,check=True)
        self.assertEqual(self.rows(json.loads(result.stdout))['coverage']['status'],'reusable')


if __name__=='__main__':unittest.main()
