"""Synthetic evaluated-motion failures through the real session and pipeline."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .controller import fingerprint, read_json, write_json
from .preservation import PreservationPolicy, file_ref
from .operating_session import OperatingSession, PROFILES
from .candidate_pipeline import CandidatePipeline
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

    def test_pipeline_carries_outcomes_and_allows_verification_to_resolve_missing_measurements(self):
        pipeline=CandidatePipeline(self.root/'pipeline',revision='1',candidates=lambda s,p:[self.item('fit','appearance_edit')],
            verify=lambda s,p:[self.item('verify','verification')],retain=lambda s,p:[self.item('retain','retention')],
            early_review=False,preservation=self.policy,preservation_adapter='fit')
        first=pipeline(self.state,{})[0]
        result={'status':'completed','preservation':{'policy':self.policy.revision,'status':'unknown','evidence':[]}}
        outcomes={'fit':{'task':first,'definition':fingerprint(first),'status':'completed','result':result}}
        second=pipeline(self.state,outcomes)[-1]
        self.assertEqual(second['workbench']['preservation']['outcomes'],['shell'])
        self.assertNotIn('preservation',second['workbench']['consumes']['fit'])
        self.policy.preflight(second,outcomes)

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

    def test_full_pipeline_keeps_constraints_through_save_review_and_retention(self):
        def follow(key,profile):
            def factory(state,previous):
                task=self.item(key,profile)
                if previous:task['payload']['candidate']=previous['result']['subject']
                return [task]
            return factory
        pipeline=CandidatePipeline(self.root/'pipeline',revision='full',
            prepare=follow('prepare','analysis'),candidates=follow('fit','appearance_edit'),
            verify=follow('verify','verification'),retain=follow('retain','retention'),early_review=False,
            preservation=self.policy,preservation_adapter='fit')
        def handler(item,context):
            self.ran.append(item['id'])
            inputs=item['payload']['constraint_inputs']['shell']
            values=read_json(inputs['baseline']['path'])['values']
            if item['id']=='verify':result={'status':'completed','subject':item['payload']['candidate']}
            else:result=self.result(item['id']+'.json',values)
            evidence={'kind':'file','path':result['subject']['path'],'role':'Saved synthetic result'}
            result['workbench']={'checks':{name:{'status':'pass','evidence':[evidence]}
                for name in PROFILES[item['workbench']['profile']]['outputs']},'findings':[]}
            return result
        session=self.session(pipeline,handler)
        session.run(max_steps=5)
        self.assertEqual(self.ran,['prepare','fit','verify'])
        basis=session.review_basis('verify')
        session.record_review('verify',expected_basis=basis,
            judgment={'disposition':'useful','scope':'synthetic region','reason':'Reviewed saved fixture','next_question':'Retain fixture'},
            evidence=[{'kind':'file','path':str(self.base.image),'role':'Actual synthetic review evidence'}])
        session.run(max_steps=1)
        self.assertEqual(self.ran,['prepare','fit','verify','retain'])
        saved=read_json(session.path)['results']['retain']['result']
        self.assertEqual(saved['preservation']['status'],'pass')
        self.assertEqual(saved['workbench']['checks']['preservation']['status'],'pass')


if __name__=='__main__':unittest.main()
