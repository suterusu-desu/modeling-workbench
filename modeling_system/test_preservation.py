"""Synthetic evaluated-motion failures through the real session and pipeline."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .controller import fingerprint, read_json, write_json
from .preservation import PreservationPolicy, file_ref
from .operating_session import OperatingSession, PROFILES
from . import test_decisions as fixtures


def coverage(item, document):
    construction = read_json(document['construction']['path'])
    return {'outcomes': construction['affected'], 'evidence': [document['construction']]}


def bind(item, constraints):
    consumed = {r['id']: r['constraints'] for r in constraints['outcomes']}
    return {'payload': {**item.get('payload', {}), 'constraint_inputs': consumed}, 'consumed': consumed}


def measure(item, result, constraints):
    actual = read_json(result['subject']['path'])
    measured = {}
    for row in constraints['outcomes']:
        baseline = read_json(row['baseline']['path'])
        measured[row['id']] = {'baseline': row['baseline'], 'guide': row['guide'], 'cells': {
            c['id']: {k:c[k] for k in ('region','pose','metric')} | {
                'observed': actual['values'].get(c['pose']), 'baseline': baseline['values'].get(c['pose'])}
            for c in row['cells']}}
    return {'subject': result['subject'], 'construction': constraints['construction'],
        'kind': 'prepared_output' if constraints['stage'] == 'prepare' else 'evaluated_output',
        'evidence': [result['subject']], 'outcomes': measured}


def unconsumed(item, constraints):
    return {'payload': item.get('payload', {}), 'consumed': {}}


def arbitrary_subject(item, result, constraints):
    other = deepcopy(result); other['subject'] = constraints['outcomes'][0]['baseline']
    return measure(item, other, constraints)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.EpisodeTests(); self.base.setUp(); self.addCleanup(self.base.tearDown)
        self.root, self.service = self.base.root, self.base.s
        self.episode = self.base.episode()['episode']
        self.baseline = self.write('baseline.json', {'values': {'open':0., 'half':0.1, 'closed':0.}})
        self.guide = self.write('guide.json', {'geometry': 'qualified synthetic guide'})
        self.construction = self.write('construction.json', {'affected':['shell'], 'law':'old + support*(desired-old)'})
        self.authority = self.write('authority.json', {'required':'Keep achieved shape through motion'})
        self.outcome = {'id':'shell','description':'Preserve the surrounding evaluated surface through the affected motion',
            'baseline':self.baseline,'guide':self.guide,'evidence':[self.authority],
            'constraints':{'baseline':self.baseline,'guide':self.guide},
            'cells':[{'id':p,'region':'shell','pose':p,'metric':'penetration','rule':{'max_increase':0.}}
                     for p in ('open','half','closed')]}
        self.doc={'schema_version':1,'construction':self.construction,'authority':self.authority,
            'public_construction':{'law':'Zero support retains old moving positions'},'outcomes':[self.outcome]}
        self.adapters={'fit':{'sources':[file_ref(__file__)],'coverage':coverage,'bind':bind,'measure':measure}}
        self.policy=self.make_policy()
        self.state={'owner':'native owner','authority_revision':'scope1','active_operations':[],
            'values':{'source':'s1','guide':'g1','adapter':'a1'},'public_state':{'goal':'Synthetic supported correction'}}
        self.ran=[];self.batches=[]

    def write(self,name,value):
        path=self.root/name;write_json(path,value);return file_ref(path)

    def make_policy(self):
        return PreservationPolicy(self.write('required-outcomes.json',self.doc),self.adapters)

    def item(self,key,profile='analysis'):
        return {'id':key,'revision':'1','lane':'repair' if profile=='appearance_edit' else 'diagnosis',
            'handler':'run','description':'Synthetic supported operation','completion_condition':'Real output and source-bound checks',
            'reads':deepcopy(getattr(self,'state',{}).get('values',{'source':'s1','guide':'g1','adapter':'a1'})),
            'writes':[],'requires':{},'payload':{},'workbench':{'profile':profile,'capability':'synthetic fit',
                'method':'full evaluated shape','bindings':{k:['source'] for k in PROFILES[profile]['inputs']}}}

    def result(self,name='candidate.json',values=None):
        return {'status':'completed','subject':self.write(name,{'values': values or {'open':0.,'half':0.1,'closed':0.}})}

    def assess(self,values):
        task=self.policy.bind_task(self.item('fit','appearance_edit'),stage='apply',adapter='fit')
        prepared,consumption=self.policy.prepare(task,{})
        result=self.result(values=values)
        result['preservation']=self.policy.assess(task,result,consumption)
        return task,result

    def test_mid_motion_collateral_failure_survives_exact_endpoints_and_local_gain(self):
        task,result=self.assess({'open':0.,'half':0.2,'closed':0.})
        self.assertEqual([c['status'] for c in result['preservation']['checks']],['pass','fail','pass'])
        self.assertEqual(result['preservation']['status'],'fail')

    def test_missing_closed_observation_is_unknown_not_zero(self):
        _,result=self.assess({'open':0.,'half':0.05})
        self.assertEqual(result['preservation']['status'],'unknown')
        self.assertEqual(result['preservation']['checks'][-1]['status'],'unknown')

    def test_changed_or_missing_pins_are_named_all_at_once_before_work(self):
        self.assertEqual(self.policy.stale_references(),[])
        Path(self.construction['path']).write_text('{"edited during the trial":true}')
        Path(self.authority['path']).unlink()
        stale={Path(s['path']).name:s for s in self.policy.stale_references()}
        self.assertEqual(set(stale),{'construction.json','authority.json'})
        self.assertIsNone(stale['authority.json']['actual'])
        self.assertNotEqual(stale['construction.json']['actual'],stale['construction.json']['pinned'])
        with self.assertRaisesRegex(ValueError,r'source changed: .*construction\.json \(changed\).*authority\.json \(missing\)'):
            self.policy.fresh()
        from .preservation import checked
        with self.assertRaisesRegex(ValueError,r'changed: .*construction\.json \(pinned '):checked(self.construction)
        with self.assertRaisesRegex(ValueError,r'missing: .*authority\.json'):checked(self.authority)

    def test_preview_uses_the_policy_rules_on_offline_measurements_and_is_not_evidence(self):
        def rows(values,base=(0.,.1,0.)):
            return {'shell':{'cells':{p:{'region':'shell','pose':p,'metric':'penetration','observed':v,'baseline':b}
                for p,v,b in zip(('open','half','closed'),values,base) if v is not None}}}
        failing=self.policy.preview(rows((0.,.2,0.)))
        self.assertEqual([c['status'] for c in failing['checks']],['pass','fail','pass'])
        self.assertEqual((failing['status'],failing['kind'],failing['evidence']),('fail','preview',None))
        self.assertEqual(self.policy.preview(rows((0.,.05,None)))['status'],'unknown')
        self.assertEqual(self.policy.preview(rows((0.,.1,0.)))['status'],'pass')
        wrong=rows((0.,.1,0.));wrong['shell']['baseline']=self.guide     # a row naming another baseline is not this outcome
        self.assertEqual(self.policy.preview(wrong)['status'],'unknown')
        with self.assertRaisesRegex(ValueError,'outcomes of this policy'):self.policy.preview(rows((0.,.1,0.)),outcomes=['other'])
        _,result=self.assess({'open':0.,'half':0.2,'closed':0.})       # the native judgment is the same rule
        self.assertEqual([c['status'] for c in result['preservation']['checks']],[c['status'] for c in failing['checks']])

    def test_forged_numbers_and_arbitrary_existing_subject_fail_recomputation(self):
        task,result=self.assess({'open':0.,'half':0.2,'closed':0.})
        altered=deepcopy(result['preservation']);altered['measurements']['outcomes']['shell']['cells']['half']['observed']=0.
        with self.assertRaises(ValueError):self.policy.verify_assessment(altered,task,result)
        self.adapters['fit']['measure']=arbitrary_subject
        self.policy=self.make_policy()
        task=self.policy.bind_task(self.item('fit','appearance_edit'),stage='apply',adapter='fit')
        _,consumption=self.policy.prepare(task,{})
        with self.assertRaisesRegex(ValueError,'actual operation result'):
            self.policy.assess(task,result,consumption)

    def test_live_source_or_stale_construction_invalidates_bound_requirements(self):
        task,_=self.assess({'open':0.,'half':0.1,'closed':0.})
        Path(self.construction['path']).write_text('{"old construction":true}')
        with self.assertRaises(ValueError):self.policy.preflight(task,{})
        self.assertNotEqual(task['reads'],{**task['reads'],**self.policy.context_values()})

    def test_missing_constraints_refuse_before_actual_handler(self):
        self.adapters['fit']['bind']=unconsumed
        self.policy=self.make_policy()
        task=self.policy.bind_task(self.item('fit','appearance_edit'),stage='apply',adapter='fit')
        with self.assertRaisesRegex(ValueError,'omitted required'):self.policy.prepare(task,{})

    def test_unbound_appearance_is_blocked_while_diagnosis_remains_usable(self):
        with self.assertRaises(ValueError):self.policy.preflight(self.item('edit','appearance_edit'),{})
        self.assertIsNone(self.policy.preflight(self.item('diagnose'),{}))

    def test_known_support_blend_fails_until_real_fitter_consumes_full_position_constraint(self):
        # Equal endpoint controls and support=0 preserve the old moving field,
        # not the established final surface. This fixture computes the effect.
        old=[0.,0.4,0.];desired=[0.,0.1,0.];support=[1.,0.,1.]
        bad=[a+s*(b-a) for a,b,s in zip(old,desired,support)]
        _,result=self.assess(dict(zip(('open','half','closed'),bad)))
        self.assertEqual(result['preservation']['status'],'fail')
        task=self.policy.bind_task(self.item('fit','appearance_edit'),stage='apply',adapter='fit')
        prepared,consumption=self.policy.prepare(task,{})
        # The synthetic fitter reads the injected constraint data as solve inputs.
        fixed=read_json(prepared['payload']['constraint_inputs']['shell']['baseline']['path'])['values']
        result=self.result('constrained.json',fixed)
        assessment=self.policy.assess(task,result,consumption)
        self.assertEqual(assessment['status'],'pass');self.assertEqual(assessment['appearance_acceptance'],'not implied')

    def test_retention_requires_actual_prior_result_and_rejection_overrides_pass(self):
        task,result=self.assess({'open':0.,'half':0.05,'closed':0.})
        retain=self.item('retain','retention');retain['payload']['candidate']=result['subject'];retain['requires']={'fit':['completed']}
        retain=self.policy.bind_task(retain,stage='retain',adapter='fit',previous={'task':task,'result':result})
        prior={'fit':{'task':task,'result':result}}
        self.policy.preflight(retain,prior)
        with self.assertRaises(ValueError):self.policy.preflight(retain,{})
        bad=deepcopy(retain);bad['payload']['candidate']=self.baseline
        with self.assertRaises(ValueError):self.policy.preflight(bad,prior)
        self.doc['rejections']=[{'subject':result['subject'],'evidence':self.authority,'reason':'User rejects the whole result'}]
        self.policy=self.make_policy()
        task,result=self.assess({'open':0.,'half':0.05,'closed':0.})
        retain=self.policy.bind_task(retain,stage='retain',adapter='fit',previous={'task':task,'result':result})
        with self.assertRaisesRegex(ValueError,'user rejection'):
            self.policy.preflight(retain,{'fit':{'task':task,'result':result}})

    def session(self,catalog,handler):
        return OperatingSession(self.root/'queue',service=self.service,episode=self.episode,owner='native owner',
            goal={'objective':'Preserve supported form'},observe_context=lambda:deepcopy(self.state),catalog=catalog,
            handlers={'run':handler},preservation=self.policy)

    def test_session_overrides_self_reported_pass_and_keeps_completed_effect(self):
        task=self.policy.bind_task(self.item('edit','appearance_edit'),stage='apply',adapter='fit')
        def handler(item,context):
            self.ran.append(item)
            result=self.result(values={'open':0.,'half':0.3,'closed':0.})
            link={'kind':'file','path':str(self.base.image),'role':'synthetic view'}
            result['workbench']={'checks':{k:{'status':'pass','evidence':[link]} for k in PROFILES['appearance_edit']['outputs']},'findings':[]}
            return result
        session=self.session(lambda state,results:[task],handler)
        observed=session.observe()['observations']
        session.run(max_steps=1)
        entry=read_json(session.path)['results']['edit']
        self.assertEqual(entry['status'],'completed')
        self.assertEqual(entry['result']['workbench']['checks']['preservation']['status'],'fail')
        self.assertIn('constraint_inputs',self.ran[0]['payload'])
        self.assertIn('required_preservation',observed)
        self.assertNotIn(self.root.as_posix(),json.dumps(observed['required_preservation']))

    def test_workspace_requirement_cannot_be_forgotten_but_does_not_block_harmless_work(self):
        tasks=[self.item('edit','appearance_edit'),self.item('read')]
        session=OperatingSession(self.root/'queue',service=self.service,episode=self.episode,owner='native owner',
            goal={'objective':'Scoped work'},observe_context=lambda:deepcopy(self.state),catalog=lambda s,r:tasks,
            handlers={'run':lambda item,context:None},require_preservation=True)
        state=session.observe()
        self.assertEqual([a['id'] for a in state['actions']],['read'])
        self.assertIn('preservation',state['observations']['blocked']['edit'])

    def test_native_job_handoff_must_pin_consumed_solver_inputs(self):
        task=self.item('fit','appearance_edit');task['workbench']['native']=True
        task['payload']['job']={'dependency_hashes':{}}
        task=self.policy.bind_task(task,stage='apply',adapter='fit')
        with self.assertRaisesRegex(ValueError,'Native worker'):
            self.policy.prepare(task,{})
        task['payload']['job']['dependency_hashes']={r['path']:r['sha256'] for r in self.outcome['constraints'].values()}
        prepared,consumption=self.policy.prepare(task,{})
        self.assertEqual(prepared['payload']['constraint_inputs'],consumption['consumed'])

if __name__=='__main__':unittest.main()
