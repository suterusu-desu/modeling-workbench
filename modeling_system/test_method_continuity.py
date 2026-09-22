from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from . import test_decisions as fixtures
from .controller import read_json
from .learning import record_method_experience, import_method_experiences, experience_rows
from .service import ModelingService, UnconfiguredNative
from .work_queue import LaneSelector


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.base=fixtures.EpisodeTests();self.base.setUp();self.addCleanup(self.base.tearDown)
        self.root,self.service=self.base.root,self.base.s
        self.evidence=self.root/'source.txt';self.evidence.write_text('Exact failed broader shape with a local numerical gain.')
        self.row={'lesson':{'mechanism':'Local smoothing breaks the connected transition',
            'observation':'A local numeric gain worsened the surrounding evaluated shape.',
            'conditions':{'construction':'blended transition'},
            'next_use':'Constrain the established surrounding surface through intermediate motion.',
            'limits':'Demonstrated scoped failure; no automatic aesthetic or all-motion claim.'},
            'evidence':[{'kind':'file','path':str(self.evidence),'role':'Exact failure evidence'}],
            'knowledge_status':'demonstrated','candidate_disposition':'rejected',
            'interpretation':'Supported failure mechanism from the reviewed evidence.',
            'classification':'recurring_failure','authority':'historical_curator'}

    def test_import_is_idempotent_and_failure_knowledge_is_not_candidate_success(self):
        a=import_method_experiences(self.service,[self.row]);b=import_method_experiences(self.service,[self.row])
        self.assertEqual(a,b);key=a['records'][0]
        row=self.service.store.get(key,'modeling_experience')
        self.assertEqual(row['public']['knowledge_status'],'demonstrated')
        self.assertEqual(row['public']['candidate_disposition'],'rejected')
        self.assertEqual(len(list(self.service.store.records(('modeling_experience',)))),1)
        self.assertNotIn(str(self.root),json.dumps(row['public']))

    def test_changed_interpretation_requires_explicit_correction_and_keeps_original(self):
        original=record_method_experience(self.service,**self.row)
        corrected=deepcopy(self.row);corrected['lesson']['limits']='Only the stated transition and observations were demonstrated.'
        with self.assertRaises(ValueError):record_method_experience(self.service,**corrected)
        key=record_method_experience(self.service,**corrected,supersedes=[original])
        self.assertNotEqual(key,original);self.assertTrue(self.service.store.get(original,'modeling_experience'))
        self.assertEqual([k for k,v in experience_rows(self.service.store)],[key])
        self.assertEqual(record_method_experience(self.service,**corrected,supersedes=[original]),key)

    def test_model_or_reviewer_summary_cannot_supersede_user_rejection(self):
        original=record_method_experience(self.service,**{**self.row,'authority':'user'})
        correction={**self.row,'authority':'reviewer','candidate_disposition':'scoped_success'}
        with self.assertRaisesRegex(ValueError,'user authority'):
            record_method_experience(self.service,**correction,supersedes=[original])

    def test_restart_retrieves_conditional_failure_with_exact_sources(self):
        key=record_method_experience(self.service,**self.row)
        service=ModelingService(self.root,self.service.store.root,native=UnconfiguredNative())
        result=service.retrieve_experience('Local smoothing breaks connected transition with numerical gain',
            limit=8,context={'construction':'blended transition'})
        self.assertIn(key,[row.get('record') for row in result['matches']])
        rows=experience_rows(service.store)
        self.assertTrue(rows[0][1]['evidence']);self.assertEqual(rows[0][1]['public']['classification'],'recurring_failure')

    def test_unreviewed_hypothesis_cannot_become_supported_through_classification(self):
        row={**self.row,'knowledge_status':'unverified','classification':'supported_lesson'}
        with self.assertRaises(ValueError):record_method_experience(self.service,**row)
        with self.assertRaises(Exception):record_method_experience(self.service,**self.row,review_fact='not-a-review')

    def test_actual_review_is_saved_before_learning_and_closeout_resumes_without_replay(self):
        from .test_operating_session import OperatingTests
        fixture=OperatingTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        fixture.items=[fixture.item('inspect')]
        session=fixture.session();session.run(max_steps=1)
        evidence=[{'kind':'file','path':str(fixture.base.image),'role':'Actual reviewed evidence'}]
        rejected={**self.row,'evidence':evidence,'authority':'reviewer'}
        with self.assertRaisesRegex(ValueError,'rejected review'):
            session.record_review('inspect',expected_basis=session.review_basis('inspect'),
                judgment={'disposition':'rejected','scope':'Affected evaluated region',
                    'reason':'The actual comparison shows a surrounding regression.',
                    'next_question':'Repair the demonstrated transition mechanism.'},
                evidence=evidence,experience={**rejected,'candidate_disposition':'scoped_success'})
        saved=list((session.directory/'reviews').glob('*.json'))
        self.assertEqual(len(saved),1)
        fact=read_json(saved[0])['operation_fact']
        restarted=fixture.session()
        key=restarted.record_review_experience(fact,**rejected)
        self.assertEqual(restarted.record_review_experience(fact,**rejected),key)
        retained=fixture.service.store.get(key,'modeling_experience')
        self.assertEqual(retained['review_fact'],fact)
        self.assertEqual(retained['public']['candidate_disposition'],'rejected')
        self.assertEqual(fixture.ran,['inspect'])


class MethodReasoningTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.calls=[];self.mode='missing'
        self.actions=[{'id':'fit','lane':'repair','reads':{'source':'v1'},'writes':[],
            'decision':{'method_checks':{'method':'Does the recorded mechanism apply here?',
                'coverage':'Does preservation cover the influenced motion?', 'remedies':['inspect']}}},
            {'id':'inspect','lane':'diagnosis','reads':{'source':'v1'},'writes':[]}]
        self.snapshot={'owner':'owner','authority_revision':'current'}
        self.plan={'reads':{'source':'v1'},'writes':[]}

    def judge(self,state,questions,binding):
        self.calls.append(deepcopy(questions));answers={}
        for q in questions:
            key=q['id'];choice=q['options'][0]['id']
            if key=='next_lane':choice='repair'
            if key.endswith('_method'):choice=self.mode
            if key.endswith('_remedy'):choice='inspect'
            answers[key]={'choice':choice}
        return {'binding':binding,'judgments':answers}

    def selector(self):
        return LaneSelector(self.root,judge=self.judge,project=lambda state,actions,plan:{
            'state':{'required_preservation':{'outcome':'Surrounding evaluated form through motion'},
                     'retained_experience':{'passages':[{'content':'A failed support blend leaves old moving skin.'}]}},
            'descriptions':{'fit':'Qualified guided fit','inspect':'Inspect actual upstream output dependencies'}})

    def test_same_batch_routes_only_unmet_prerequisite_without_another_inference(self):
        selector=self.selector();self.assertEqual(selector(self.snapshot,self.actions,self.plan),'inspect')
        self.assertEqual(len(self.calls),1)
        trace=read_json(self.root/'last-batch.json')
        self.assertEqual(trace['method_route']['proposed'],'fit')
        self.assertEqual(trace['method_route']['selected'],'inspect')
        self.assertEqual(trace['method_checks']['fit']['unmet']['method']['choice'],'missing')

    def test_exact_cached_answers_survive_restart_but_changed_input_rejudges(self):
        self.selector()(self.snapshot,self.actions,self.plan)
        self.selector()(self.snapshot,self.actions,self.plan)
        self.assertEqual(len(self.calls),1)
        for action in self.actions:action['reads']['source']='v2'
        self.selector()(self.snapshot,self.actions,self.plan)
        self.assertEqual(len(self.calls),2)

    def test_ready_method_executes_and_unsupported_method_yields_to_other_chosen_lane(self):
        self.mode='ready';self.assertEqual(self.selector()(self.snapshot,self.actions,self.plan),'fit')
        self.actions[0]['decision']['method_checks']['remedies']=[];self.mode='unknown'
        result=self.selector()(self.snapshot,self.actions,self.plan)
        self.assertEqual(result,'inspect')
        trace=read_json(self.root/'last-batch.json')
        self.assertEqual(trace['method_checks']['fit']['status'],'unmet')
        self.assertEqual(trace['cooperation']['deferred_lanes'][0]['task'],'fit')

    def test_method_remedy_resolves_actual_eligible_task_without_fixed_task_id(self):
        self.actions[0]['decision']['method_checks'].update(remedies=[], remedy_methods=['dependency_diagnosis'])
        self.actions[1]['method_choice'] = 'dependency_diagnosis'
        self.assertEqual(self.selector()(self.snapshot,self.actions,self.plan),'inspect')
        self.assertEqual(len(self.calls),1)

    def test_unavailable_method_remedy_keeps_other_real_lane_choice(self):
        self.actions[0]['decision']['method_checks'].update(remedies=[], remedy_methods=['unavailable'])
        result = self.selector()(self.snapshot,self.actions,self.plan)
        self.assertEqual(result,'inspect')
        self.assertIsNone(read_json(self.root/'last-batch.json')['method_route'])
        self.assertFalse(any(q['id'].endswith('_remedy') for q in self.calls[0]))

    def test_same_batch_names_the_missing_prerequisite_and_routes_its_remedy(self):
        self.actions[0]['decision']['method_checks']['prerequisites']={'graph':'Current evaluated dependency observation is missing.'}
        original=self.judge
        def judge(state,questions,binding):
            result=original(state,questions,binding)
            for q in questions:
                if q['id'].endswith('_prerequisite'):result['judgments'][q['id']]={'choice':'graph'}
            return result
        selector=self.selector();selector.judge=judge
        self.assertEqual(selector(self.snapshot,self.actions,self.plan),'inspect')
        trace=read_json(self.root/'last-batch.json')
        self.assertEqual(trace['method_checks']['fit']['unmet_prerequisite'],'graph')
        self.assertEqual(len(self.calls),1)

    def test_named_reason_disagreement_never_waives_missing_condition(self):
        from .method_reasoning import resolve_method
        report=resolve_method({'a':'method','b':'prerequisite'}, {'a':{'choice':'missing'},'b':{'choice':'__none__'}})
        self.assertEqual(report['status'],'unmet');self.assertFalse(report['prerequisite_consistent'])
        self.assertIsNone(report['unmet_prerequisite'])

    def test_named_condition_can_block_an_otherwise_ready_method(self):
        from .method_reasoning import resolve_method
        report=resolve_method({'a':'method','b':'prerequisite'}, {'a':{'choice':'ready'},'b':{'choice':'qualified_targets'}})
        self.assertEqual(report['status'],'unmet');self.assertEqual(report['unmet_prerequisite'],'qualified_targets')

    def test_remedy_links_alone_do_not_add_an_abstract_approval_gate(self):
        self.actions[0]['decision']['method_checks']={'remedies':['inspect']}
        self.assertEqual(self.selector()(self.snapshot,self.actions,self.plan),'fit')
        self.assertFalse(any(q['id'].startswith('method_') for q in self.calls[0]))


if __name__=='__main__':unittest.main()
