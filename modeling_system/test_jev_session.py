"""Focused normal-use, transient retry and authority-rebind tests. No real HTTP."""
from copy import deepcopy
from pathlib import Path
import json,unittest
from unittest.mock import patch
from .jev_session import create_session,rebind_pending_authority,recover_transient_selection
from . import planning_batch as batch
from . import provider_dispatch as dispatch
from modeling_system import test_decisions as fixtures,typesafe_transport as api
from modeling_system.controller import fingerprint,read_json,write_json
from modeling_system.judgments import prepare_judgments
from modeling_system.test_provider_recovery import completed_call
from modeling_system.provider_recovery import ProviderRetryDeferred

class FocusedTests(unittest.TestCase):
 def setUp(self):
  self.fixture=fixtures.EpisodeTests();self.fixture.setUp();self.addCleanup(self.fixture.tearDown)
  self.root=self.fixture.root;self.service=self.fixture.s;self.episode=self.fixture.episode()['episode']
  self.ledger=self.root/'ledger';self.ledger.mkdir()
  write_json(self.ledger/'budget.json',{'policy':{'mode':'normal_use','spending':'existing_account_credits','authority':'user'},'attempts':[],'known_cost_usd':.75,'carry_in_requests':200,'status':'ready'})
  self.state={'owner':'native owner','authority_revision':'old','values':{'input':'v1','authority':'old'},'active_operations':[]}
  self.items=[{'id':key,'lane':'diagnosis','revision':'v1','handler':'run','description':key,'reads':{'input':'v1','authority':'old'},'writes':[],'requires':{},'completion_condition':'Actual evidence','workbench':{'profile':'analysis','capability':'synthetic diagnosis','method':'source-bound','bindings':{}}} for key in ('first','second')]
  self.wire=[];self.ran=[];self.mode='success'
  test=self
  class Connection:
   def __init__(self,*args,**kwargs):pass
   def request(self,method,path,body,headers):self.body=json.loads(body);test.wire.append(self.body)
   def getresponse(self):
    if test.mode=='always_timeout' or test.mode=='timeout_once' and len(test.wire)==1:raise TimeoutError('synthetic provider timeout')
    answers={}
    for key,q in self.body['questions'].items():
     ids=list(q['criteria']);answers[key]={'type':'choice','choice':ids[0],'probabilities':{k:float(k==ids[0])for k in ids},'confidence':1.}
    payload={'model':api.MODEL,'usage':{'input_tokens':100,'output_tokens':20},'answers':answers}
    if test.mode=='always_invalid' or test.mode=='invalid_once' and len(test.wire)==1:
     first=next(iter(answers.values()));ids=list(first['probabilities'])
     first['probabilities']={key:(.49 if key==ids[0] else .51 if key==ids[1] else 0.) for key in ids}
    class Reply:
     status=402 if test.mode=='credits_exhausted' else 529 if test.mode=='rate_limit_long' else 200
     def getheader(self,name):return {'retry-after':'120','x-request-id':'fixture-request'}.get(name.lower()) if test.mode=='rate_limit_long' else None
     def read(self):return json.dumps(payload).encode()
    return Reply()
   def close(self):pass
  original=dispatch.dispatch_many
  def wrapped(paths,folder,reader):return original(paths,folder,reader,_key_reader=lambda:'fixture-token')
  clock={'now':1700000000.}
  for context in (patch.object(api.http.client,'HTTPSConnection',Connection),patch.object(batch,'dispatch_many',wrapped),patch('time.time',lambda:clock['now']),patch('time.sleep',lambda seconds:clock.update(now=clock['now']+seconds))):
   context.start();self.addCleanup(context.stop)

 def session(self):
  def handler(item,context):
   self.ran.append(item['id'])
   return {'status':'completed','workbench':{'checks':{'analysis':{'status':'pass','evidence':[{'kind':'file','path':str(self.fixture.image),'role':'synthetic evidence'}]}},'findings':[]}}
  return create_session(self.root/'queue',service=self.service,episode=self.episode,owner='native owner',goal={'objective':'Resolve supported scope'},observe_context=lambda:deepcopy(self.state),catalog=lambda *a:deepcopy(self.items),handlers={'run':handler},public_projection=lambda snapshot,actions,plan:{'state':{'goal':'Resolve scope'},'descriptions':{a['id']:a['description']for a in actions}},ledger_directory=self.ledger)

 def test_fresh_success_releases_without_recovery_or_duplicate_lane_reads(self):
  s=self.session()
  with patch.object(batch,'reconcile_completed_response',side_effect=AssertionError('Healthy dispatch must not run repair')):
   s.run(max_steps=1)
  self.assertEqual(self.ran,['first']);self.assertEqual(len(self.wire),1)
  self.assertFalse((self.ledger/'call-1/completed-response-reconciliation.json').exists())
  ledger=read_json(self.ledger/'budget.json')
  self.assertEqual(ledger['status'],'ready')
  self.assertAlmostEqual(ledger['known_cost_usd'],.7500042)
  judgment,= (s.directory/'judgments').glob('*.judgments.json')
  self.assertEqual(read_json(judgment)['bridge_timings']['fresh_context_reads'],7)

 def test_changed_dependency_after_response_refuses_release_without_native_effect(self):
  s=self.session();original=batch.dispatch_many
  def changed(*args,**kwargs):
   result=original(*args,**kwargs)
   self.state['values']['input']='new source'
   return result
  with patch.object(batch,'dispatch_many',changed):
   result=s.run(max_steps=1)
  self.assertEqual(result['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])

 def test_changed_active_lane_after_response_refuses_release(self):
  s=self.session();original=batch.dispatch_many
  def changed(*args,**kwargs):
   result=original(*args,**kwargs)
   self.state['active_operations']=[{'id':'different work','writes':['input']}]
   return result
  with patch.object(batch,'dispatch_many',changed):
   result=s.run(max_steps=1)
  self.assertEqual(result['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])

 def test_timeout_recovers_automatically_using_credits_without_lifetime_cap(self):
  self.mode='timeout_once';s=self.session();s.run(max_steps=1)
  self.assertEqual(len(self.wire),2);self.assertEqual(self.ran,['first'])
  ledger=read_json(self.ledger/'budget.json')
  self.assertEqual(ledger['attempts'][0]['provider_outcome'],'unknown')
  self.assertAlmostEqual(ledger['known_cost_usd'],.75+dispatch.RESERVE_USD+.0000042)
  self.assertNotIn('max_requests',ledger['policy']);self.assertNotIn('max_cost_usd',ledger['policy'])

 def test_three_timeouts_stop_without_retry_loop_or_native_effect(self):
  self.mode='always_timeout';s=self.session()
  self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),3);self.assertEqual(self.ran,[])
  with self.assertRaises(ValueError):recover_transient_selection(s,self.ledger)
  self.assertEqual(len(self.wire),3)

 def test_actual_credit_failure_stops_immediately(self):
  self.mode='credits_exhausted';s=self.session()
  self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])

 def test_invalid_answer_retries_same_questions_and_executes_once(self):
  self.mode='invalid_once';s=self.session();s.run(max_steps=1)
  self.assertEqual(len(self.wire),2);self.assertEqual(self.ran,['first'])
  self.assertEqual(self.wire[0],self.wire[1])
  ledger=read_json(self.ledger/'budget.json')
  self.assertEqual(ledger['attempts'][0]['status'],'invalid_answer_accounted')
  self.assertAlmostEqual(ledger['known_cost_usd'],.75+.0000042*2)
  self.assertFalse((self.ledger/'call-1/selection.json').exists())

 def test_three_invalid_answers_stop_without_native_effect_and_all_costs_are_known(self):
  self.mode='always_invalid';s=self.session()
  self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),3);self.assertEqual(self.ran,[])
  ledger=read_json(self.ledger/'budget.json')
  self.assertEqual(ledger['status'],'ready')
  self.assertTrue(all(a['status']=='invalid_answer_accounted' for a in ledger['attempts']))
  self.assertAlmostEqual(ledger['known_cost_usd'],.75+.0000042*3)
  with self.assertRaises(ValueError):recover_transient_selection(s,self.ledger)
  self.assertEqual(len(self.wire),3)

 def test_pending_invalid_response_recovers_without_changing_original_scope(self):
  s=self.session()
  original=batch.resolve_packet
  def interrupted(*args,**kwargs):
   from modeling_system.invalid_response import prepare_selection_retry
   with patch('modeling_system.invalid_response.prepare_selection_retry',side_effect=ValueError('old reader cannot recover HTTP200')):
    return original(*args,**kwargs)
  self.mode='invalid_once'
  with patch.object(batch,'resolve_packet',side_effect=interrupted):
   self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])
  result=recover_transient_selection(s,self.ledger)
  self.assertEqual(result['provider_retry_lineage']['kind'],'invalid_answer')
  self.assertEqual(self.ran,[])
  s.run(max_steps=1);self.assertEqual(self.ran,['first'])
  self.assertEqual(len(self.wire),2)

 def test_typed_batch_exceeds_old_question_limit_through_installed_bridge(self):
  item=self.items[0];item['payload']={}
  item['decision']={'arguments':{f'argument_{i}':{
   'question':f'Which qualified option applies to role {i}?','path':['parameters',str(i)],
   'options':[{'id':'first','description':'First qualified option','value':f'private-value-{i}'},
              {'id':'second','description':'Second qualified option','value':f'other-private-value-{i}'}]}
   for i in range(14)}}
  s=self.session();s.run(max_steps=1)
  self.assertEqual(self.ran,['first']);self.assertEqual(len(self.wire),1)
  self.assertEqual(len(self.wire[0]['questions']),15)
  self.assertNotIn('private-value',json.dumps(self.wire))
  trace=read_json(s.selector.directory/'last-batch.json')
  self.assertEqual(trace['call']['arguments']['argument_13'],'private-value-13')
  self.assertEqual(trace['model'],api.MODEL);self.assertEqual(trace['usage']['input_tokens'],100)

 def test_long_provider_hint_remains_pending_without_extra_request(self):
  self.mode='rate_limit_long';s=self.session()
  self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])
  retry=read_json(self.ledger/'call-1/transient-retry.json')
  self.assertEqual(retry['not_before'],1700000120.)
  with self.assertRaises(ProviderRetryDeferred):recover_transient_selection(s,self.ledger)
  self.assertEqual(len(self.wire),1)

 def test_existing_timeout_rebinds_only_authority_and_resumes_original_choice(self):
  s=self.session()
  # Real queues already contain findings bound to the previous authority. Their
  # applicability changes even though their content and native inputs do not.
  history=deepcopy(self.items[0]);history['id']='history';self.items.insert(0,history)
  s.record['results']['history']={'definition':fingerprint(history),'status':'completed','result':{'workbench':{
   'basis':{'input':'v1','authority':'old'},'findings':[{'kind':'measured','scope':'surface','summary':'Retained measured support'}],
   'checks':{'analysis':{'status':'pass'}},'qualification':'No visual acceptance'}}}
  write_json(s.path,s.record)
  def old_timeout(state,questions,binding):
   packet=prepare_judgments(state,questions,binding);ledger,call=completed_call(self.root,packet)
   (call/'decision-1.response.json').unlink()
   value=read_json(ledger/'budget.json');value['policy']={'mode':'normal_use','spending':'existing_account_credits','authority':'user'}
   value['known_cost_usd']=.75;value['attempts'][0].update(error_type='TimeoutError',estimated_cost_usd=0.)
   write_json(ledger/'budget.json',value)
   raise TimeoutError('old owner attempt')
  s.selector.judge=old_timeout
  self.assertEqual(s.run(max_steps=1)['status'],'needs_reconciliation')
  self.state['authority_revision']='new';self.state['values']['authority']='new'
  evidence=self.root/'new-authority.json';write_json(evidence,{'authority':'actual user correction'})
  rebound=rebind_pending_authority(s,authority_keys=['authority'],evidence=[evidence])
  self.assertEqual(rebound['feedback_rebinding']['before']['findings'][0]['applicability'],'current inputs')
  self.assertEqual(rebound['feedback_rebinding']['after']['findings'][0]['applicability'],'historical; inputs changed')
  result=recover_transient_selection(s,self.ledger)
  self.assertEqual(result['provider_retry_lineage']['previous_provider_outcome'],'unknown')
  self.assertEqual(len(self.wire),1);self.assertEqual(self.ran,[])
  s.run(max_steps=1);self.assertEqual(self.ran,['first'])
  fresh=self.session();observed=fresh.observe()
  self.assertEqual([a['id']for a in observed['actions']],['second'])
  self.assertEqual(len(read_json(self.ledger/'budget.json')['attempts']),2)

if __name__=='__main__':unittest.main()
