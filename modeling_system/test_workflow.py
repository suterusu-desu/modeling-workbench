"""Failure-oriented tests for durable jobs, exact-view lineage and recoverable trials."""
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from .service import ModelingService
from .store import canonical,digest
from .ledger import Conflict
from .native_bridge import NativeBridgeError
from . import geometry


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.s=ModelingService(store=self.root/'store');self.image=self.root/'preview.png';Image.new('RGB',(200,100),'gray').save(self.image)
        self.a={'co':np.array([[0.,0,0],[2,0,0],[0,2,0]]),'tri':np.array([[0,1,2]],np.int32),'triangle_component':np.array([0])}
        p=self.root/'native.npz';np.savez_compressed(p,**self.a)
        data={'objects':[{'name':'Face','type':'MESH','arrays':str(p),'geometry_hash':digest(canonical({k:digest(v.tobytes()) for k,v in self.a.items()})),'source':{}}],
              'controls':{'blink':0},'selected_guide':'Neutral','references':{}}
        self.sid=digest(canonical(data));record=self.root/'native.json';record.write_text(json.dumps(dict(state=data,state_id=self.sid,coverage=[])))
        self.state=self.s.import_scene(str(record))['state']
        self.meta={'scene_state_id':self.sid,'image_sha256':digest(self.image.read_bytes()),'scene':'Test',
                   'view_matrix':np.eye(4).tolist(),'view_projection_matrix':np.eye(4).tolist(),
                   'native_resolution':[1000,800],'resolution':[200,100],'crop':[400,350,200,100]}
        self.ob=self.s.record_observation(self.state,str(self.image),self.meta)['observation']

    def tearDown(self):self.temp.cleanup()

    def request(self,kind='image',review=None,key='one'):
        return self.s.request_reference(self.ob,[str(self.image)],'Close the eye',kind,'fixture transport',{},
                                        {'scope':'local fixture; no dispatch','source':'unit test'},key,review)

    def completed(self,kind='image',review=None,key='one',path=None):
        job=self.request(kind,review,key);claim=self.s.claim_job(job['job'],job['revision'])
        return self.s.reconcile_job(job['job'],claim['revision'],'completed',output_paths=[str(path or self.image)],receipt={'test_fixture':True})

    def reviewed(self):
        job=self.completed();return self.s.review_reference(job['handle'],0,'useful','eye',10,'Fixture observation','unchanged','matched','closed',[])['review']

    def tripo_policy(self):
        path=self.root/'MESH-GENERATION.json'
        path.write_text(json.dumps({'tripo':{'mode':'Smart Mesh','model':'P2.0'}}),encoding='utf-8')
        self.s.references.generation_policy=path

    def tripo_request(self, review, key='tripo', mode='Smart Mesh', model='P2.0', provider='Tripo Studio'):
        return self.s.prepare_guide(review,provider,{'mode':mode,'model':model,'displayed_generation_credits':65},
            {'scope':'fixture only','source':'unit test'},key)

    def test_tripo_wrong_mode_or_model_cannot_prepare(self):
        review=self.reviewed();self.tripo_policy()
        for mode,model,provider in [('HD Model','v3.1','Tripo Studio'),('Smart Mesh','P1.0','Tripo Studio'),
                                    ('Smart Mesh','P2.0','Tripo HD Model v3.1'),('','', 'Tripo Studio')]:
            with self.subTest(mode=mode,model=model,provider=provider),self.assertRaisesRegex(ValueError,'Smart Mesh / P2.0'):
                self.tripo_request(review,mode=mode,model=model,provider=provider)
        self.assertFalse(any(j['intent']['kind']=='mesh' for j in self.s.ledger.list()))

    def test_tripo_claim_requires_fresh_matching_panel_and_preserves_failed_request(self):
        review=self.reviewed();self.tripo_policy();job=self.tripo_request(review)
        good={'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':65,'observed_balance':100,
              'observed_at':datetime.now(timezone.utc).isoformat(),'source':'observed Tripo generation panel'}
        stale=(datetime.now(timezone.utc)-timedelta(minutes=11)).isoformat()
        for preflight in [None,dict(good,mode='HD Model'),dict(good,model='P1.0'),dict(good,displayed_generation_credits=100),
                          dict(good,observed_balance=60),dict(good,observed_at=stale),dict(good,observed_at='invalid')]:
            with self.subTest(preflight=preflight),self.assertRaises(ValueError):
                self.s.claim_job(job['job'],job['revision'],preflight)
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        claimed=self.s.claim_job(job['job'],job['revision'],good)
        self.assertEqual(claimed['status'],'dispatching');self.assertEqual(claimed['data']['provider_preflight'],good)

    def test_tripo_policy_is_rechecked_for_legacy_prepared_job_and_can_cancel_without_claim(self):
        review=self.reviewed();self.s.references.generation_policy=None
        job=self.tripo_request(review,mode='HD Model',model='v3.1');self.tripo_policy()
        with self.assertRaisesRegex(ValueError,'Smart Mesh / P2.0'):
            self.s.claim_job(job['job'],job['revision'])
        cancelled=self.s.reconcile_job(job['job'],job['revision'],'cancelled',receipt={'reason':'wrong mode; never dispatched'})
        self.assertEqual(cancelled['status'],'cancelled');self.assertNotIn('submission_intent',cancelled['data'])
        self.assertEqual([h['status'] for h in cancelled['history']],['prepared'])

    def reviewed_view(self, key, color):
        path=self.root/(key+'.png');Image.new('RGB',(200,100),color).save(path)
        job=self.completed(key=key,path=path)
        return self.s.review_reference(job['handle'],0,'useful','eye',10,'Fixture view','unchanged','matched','closed',[])['review']

    def multi_view_policy(self, minimum=2):
        path=self.root/'MESH-GENERATION.json'
        path.write_text(json.dumps({'tripo':{'mode':'Smart Mesh','model':'P2.0',
            'multi_view_requirement':{'required':True,'minimum_distinct_views':minimum}}}),encoding='utf-8')
        self.s.references.generation_policy=path

    def multi_view_request(self, views, key='mv', **extra):
        return self.s.prepare_guide(views['front'],'Tripo Studio',{'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':100,
            'views':views,**extra},{'scope':'fixture only','source':'unit test'},key)

    def test_single_image_mesh_is_refused_when_the_policy_requires_views(self):
        front,other=self.reviewed_view('front','red'),self.reviewed_view('other','blue');self.multi_view_policy()
        with self.assertRaisesRegex(ValueError,'never generated from a single image'):
            self.tripo_request(front)
        with self.assertRaisesRegex(ValueError,'never generated from a single image'):
            self.multi_view_request({'front':front})
        with self.assertRaisesRegex(ValueError,'its own image'):
            self.multi_view_request({'front':front,'left':front})
        with self.assertRaisesRegex(ValueError,'front/left/right/back'):
            self.multi_view_request({'front':front,'top':other})
        with self.assertRaisesRegex(ValueError,'front slot'):
            self.s.prepare_guide(front,'Tripo Studio',{'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':100,
                'views':{'front':other,'left':front}},{'scope':'fixture only','source':'unit test'},'mv')
        self.multi_view_policy(minimum=3)
        with self.assertRaisesRegex(ValueError,'at least 3'):
            self.multi_view_request({'front':front,'left':other})
        with self.assertRaisesRegex(ValueError,'mesh generation'):
            self.s.request_reference(self.ob,[str(self.image)],'Close the eye','image','fixture transport',{'views':{'front':front}},
                                     {'scope':'fixture','source':'unit test'},'img-views')
        self.assertFalse(any(j['intent']['kind']=='mesh' for j in self.s.ledger.list()))

    def test_multi_view_mesh_binds_every_slot_and_the_live_panel(self):
        views={slot:self.reviewed_view(slot,color) for slot,color in
               (('front','red'),('left','green'),('right','blue'),('back','white'))}
        self.multi_view_policy()
        job=self.multi_view_request(views,output_settings={'topology':'Quad','polycount':25000})
        self.assertEqual(sorted(job['transport']['view_slots']),sorted(views))
        self.assertEqual(len(set(job['transport']['input_paths'])),4)
        bound=self.s.inspect_workflow(job['job'])['intent']['view_inputs']
        slots={slot:entry['source']['sha256'] for slot,entry in bound.items()}
        self.assertEqual(len(set(slots.values())),4)
        good={'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':100,'observed_balance':1000,
              'observed_at':datetime.now(timezone.utc).isoformat(),'source':'observed Tripo multi-view panel',
              'slot_sha256':slots,'topology':'Quad','polycount':25000}
        swapped=dict(slots,left=slots['right'],right=slots['left'])
        for preflight in [{k:v for k,v in good.items() if k!='slot_sha256'},dict(good,slot_sha256=swapped),
                          dict(good,slot_sha256={k:v for k,v in slots.items() if k!='back'}),
                          dict(good,topology='Triangle'),dict(good,polycount=5000),dict(good,polycount='25000')]:
            with self.subTest(preflight=preflight),self.assertRaises(ValueError):
                self.s.claim_job(job['job'],job['revision'],preflight)
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        claimed=self.s.claim_job(job['job'],job['revision'],good)
        self.assertEqual(claimed['status'],'dispatching');self.assertEqual(claimed['data']['provider_preflight']['slot_sha256'],slots)

    def test_a_view_rejected_after_preparation_blocks_the_claim(self):
        views={slot:self.reviewed_view(slot,color) for slot,color in (('front','red'),('left','green'))}
        self.multi_view_policy();job=self.multi_view_request(views)
        left=self.s.store.get(views['left'],'reference_review')
        self.s.review_reference(left['job'],0,'rejected','eye',10,'Wrong pose on closer review','n/a','n/a','n/a',['pose'])
        good={'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':100,'observed_balance':1000,
              'observed_at':datetime.now(timezone.utc).isoformat(),'source':'observed Tripo multi-view panel',
              'slot_sha256':{s:e['source']['sha256'] for s,e in self.s.inspect_workflow(job['job'])['intent']['view_inputs'].items()}}
        with self.assertRaises(ValueError):
            self.s.claim_job(job['job'],job['revision'],good)
        self.assertEqual(self.s.inspect_workflow(job['job'])['status'],'prepared')

    def test_single_image_job_prepared_before_the_rule_cannot_be_claimed_after_it(self):
        review=self.reviewed();self.tripo_policy();job=self.tripo_request(review)
        self.multi_view_policy()
        good={'mode':'Smart Mesh','model':'P2.0','displayed_generation_credits':65,'observed_balance':100,
              'observed_at':datetime.now(timezone.utc).isoformat(),'source':'observed Tripo generation panel'}
        with self.assertRaisesRegex(ValueError,'never generated from a single image'):
            self.s.claim_job(job['job'],job['revision'],good)
        cancelled=self.s.reconcile_job(job['job'],job['revision'],'cancelled',receipt={'reason':'single image; never dispatched'})
        self.assertEqual(cancelled['status'],'cancelled')

    def test_job_idempotency_and_uncertain_dispatch_never_duplicate(self):
        first=self.request();same=self.request();self.assertTrue(same['reused']);self.assertEqual(first['job'],same['job'])
        claimed=self.s.claim_job(first['job'],first['revision'])
        unknown=self.s.reconcile_job(first['job'],claimed['revision'],'unknown')
        with self.assertRaises(Conflict):self.s.claim_job(first['job'],unknown['revision'])
        with self.assertRaises(Conflict):self.s.reconcile_job(first['job'],first['revision'],'completed',output_paths=[str(self.image)])
        resumed=ModelingService(store=self.root/'store').inspect_workflow(first['job']);self.assertEqual(resumed['status'],'unknown')
        with self.assertRaises(Conflict):self.request('video',self.reviewed(),'one')

    def comparison_policy(self):
        self.tripo_policy()
        self.comparison_id='synthetic-source-comparison'
        self.comparison=dict(status='authorized',provider_id='tripo',mode='HD Model',model='v3.1',
            source_sha256=digest(self.image.read_bytes()),idempotency_key='tripo:synthetic-source-comparison:hd',
            authorization={'scope':'Synthetic test only; no real dispatch','source':'unit test'})
        policy={'tripo':{'mode':'Smart Mesh','model':'P2.0','authorized_comparisons':{self.comparison_id:self.comparison},
            'transport':{'selected':'Tripo Studio website'},
            'selection_preferences':{'exact_view_enlarged_local_detail':{'mode':'HD Model','model':'v3.1','status':'test preference'}}}}
        self.s.references.generation_policy.write_text(json.dumps(policy),encoding='utf-8')

    def comparison_request(self, review, settings=None, key=None, provider='Tripo Studio'):
        return self.s.prepare_guide(review,provider,
            dict({'mode':'HD Model','model':'v3.1','comparison_id':self.comparison_id,
                  'displayed_generation_credits':30,'topology':'triangle','polycount':200000},**(settings or {})),
            {'scope':'synthetic comparison; no provider dispatch','source':'unit test'},
            key or self.comparison['idempotency_key'])

    def comparison_preflight(self):
        return {'mode':'HD Model','model':'v3.1','source_sha256':digest(self.image.read_bytes()),
                'displayed_generation_credits':30,'observed_balance':100,
                'observed_at':datetime.now(timezone.utc).isoformat(),'source':'synthetic panel fixture; no UI observation'}

    def local_reconstruction_policy(self,review):
        self.comparison_policy();path=self.s.references.generation_policy;policy=json.loads(path.read_text())
        self.local_id='fixture-local-guide'
        self.local={'status':'authorized','intent_class':'local_guide_reconstruction','provider_id':'tripo',
            'use_case':'exact_view_enlarged_local_detail','mode':'HD Model','model':'v3.1','transport':'Tripo Studio website',
            'source_image_job':self.s.store.get(review,'reference_review')['job'],'review_id':review,
            'source_sha256':digest(self.image.read_bytes()),'idempotency_key':'tripo:fixture-local-guide:hd-v3.1',
            'output_settings':{'tier':'Ultra','topology':'Triangle','polycount':2000000,'ai_complete':False,'texture':False,'parts':False},
            'max_existing_credit_cost':30,'authorization':{'scope':'Existing bounded eye-guide fixture only; no actual dispatch','source':'Explicit synthetic authorization'}}
        policy['tripo']['transport']={'selected':'Tripo Studio website'}
        policy['tripo']['authorized_local_reconstructions']={self.local_id:self.local};path.write_text(json.dumps(policy))
        self.s.policy_path=path

    def local_reconstruction_request(self,review,settings=None,key=None,provider='Tripo Studio',authorization=None):
        options=dict(mode='HD Model',model='v3.1',intent_class='local_guide_reconstruction',reconstruction_id=self.local_id,
            use_case='exact_view_enlarged_local_detail',transport='Tripo Studio website',displayed_generation_credits=30,**self.local['output_settings'])
        options.update(settings or {})
        return self.s.prepare_guide(review,provider,options,authorization or self.local['authorization'],key or self.local['idempotency_key'])

    def local_reconstruction_preflight(self):
        return dict(mode='HD Model',model='v3.1',source_sha256=self.local['source_sha256'],displayed_generation_credits=30,
            observed_balance=100,observed_at=datetime.now(timezone.utc).isoformat(),source='Inert panel fixture',**self.local['output_settings'])

    def test_local_reconstruction_retains_ordinary_intent_and_recovers_same_unknown_job(self):
        review=self.reviewed();self.local_reconstruction_policy(review);job=self.local_reconstruction_request(review)
        intent=self.s.inspect_workflow(job['job'])['intent']
        self.assertEqual(intent['intent_class'],'local_guide_reconstruction');self.assertNotIn('comparison_authorization',intent)
        self.assertEqual(intent['reconstruction_authorization']['policy'],self.local)
        prepared_context=self.s.store.get(job['decision_context'])
        self.assertEqual(prepared_context['context']['intent_class'],'local_guide_reconstruction')
        self.assertEqual(prepared_context['route_selection']['model'],'v3.1')
        claimed=self.s.claim_job(job['job'],job['revision'],self.local_reconstruction_preflight())
        claim_context=self.s.store.get(claimed['decision_context'])
        self.assertEqual(claim_context['context']['reconstruction_id'],self.local_id)
        self.assertEqual(claim_context['route_selection']['model'],'v3.1')
        unknown=self.s.reconcile_job(job['job'],claimed['revision'],'unknown',receipt={'fixture_only':True,'actual_credits_spent':0})
        self.s=ModelingService(workspace=self.root,store=self.root/'store')
        same=self.local_reconstruction_request(review);self.assertEqual(same['job'],job['job']);self.assertEqual(same['status'],'unknown')
        with self.assertRaises(Conflict):self.s.claim_job(job['job'],unknown['revision'],self.local_reconstruction_preflight())
        with self.assertRaises(ValueError):self.local_reconstruction_request(review,key='retry')
        resolved=self.s.reconcile_job(job['job'],unknown['revision'],'completed',provider_id='fake-existing-job',output_paths=[str(self.root/'native.npz')])
        with self.assertRaises(Conflict):self.s.claim_job(job['job'],resolved['revision'],self.local_reconstruction_preflight())
        self.assertEqual(len([j for j in self.s.ledger.list() if j['intent']['kind']=='mesh']),1)

    def test_local_reconstruction_cannot_relax_general_comparison_route_or_cost_scope(self):
        review=self.reviewed();self.local_reconstruction_policy(review)
        cases=[({'reconstruction_id':None},None,'Tripo Studio'),({'reconstruction_id':'unknown'},None,'Tripo Studio'),
            ({'intent_class':None},None,'Tripo Studio'),({'intent_class':'model_comparison'},None,'Tripo Studio'),
            ({'comparison_id':self.comparison_id},None,'Tripo Studio'),({'use_case':'general'},None,'Tripo Studio'),
            ({'mode':'Smart Mesh','model':'P2.0'},None,'Tripo Studio'),({'transport':'Tripo API'},None,'Tripo Studio'),
            ({},None,'Tripo API'),({'displayed_generation_credits':31},None,'Tripo Studio'),
            ({'displayed_generation_credits':True},None,'Tripo Studio'),({'topology':'Quad'},None,'Tripo Studio'),
            ({'texture':0},None,'Tripo Studio'),({},self.comparison['idempotency_key'],'Tripo Studio')]
        for settings,key,provider in cases:
            with self.subTest(settings=settings,key=key,provider=provider),self.assertRaises(ValueError):
                self.local_reconstruction_request(review,settings,key,provider)
        self.assertFalse(any(j['intent']['kind']=='mesh' for j in self.s.ledger.list()))
        with self.assertRaises(ValueError):self.local_reconstruction_request(review,authorization={'scope':'broader grant','source':'caller'})
        self.assertEqual(self.tripo_request(review)['status'],'prepared')
        comparison=self.comparison_request(review);self.assertEqual(comparison['status'],'prepared')
        self.assertNotIn('reconstruction_authorization',self.s.inspect_workflow(comparison['job'])['intent'])

    def test_local_reconstruction_claim_rechecks_policy_and_exact_live_preflight(self):
        review=self.reviewed();self.local_reconstruction_policy(review);job=self.local_reconstruction_request(review)
        good=self.local_reconstruction_preflight()
        for preflight in [None,dict(good,source_sha256='0'*64),dict(good,topology='Quad'),dict(good,polycount=100000),
                dict(good,texture=True),dict(good,parts=0),dict(good,displayed_generation_credits=31),dict(good,observed_balance=29),
                dict(good,observed_at=(datetime.now(timezone.utc)-timedelta(minutes=11)).isoformat())]:
            with self.subTest(preflight=preflight),self.assertRaises(ValueError):self.s.claim_job(job['job'],job['revision'],preflight)
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        path=self.s.references.generation_policy;original=json.loads(path.read_text())
        for field,value in [('status','pending_source_review'),('review_id','0'*64),('source_image_job','0'*64),
                ('source_sha256','0'*64),('idempotency_key','different'),('max_existing_credit_cost',20),
                ('authorization',{'scope':'changed scope','source':'changed source'})]:
            changed=deepcopy(original);changed['tripo']['authorized_local_reconstructions'][self.local_id][field]=value
            path.write_text(json.dumps(changed))
            with self.subTest(field=field),self.assertRaises((ValueError,FileNotFoundError)):self.s.claim_job(job['job'],job['revision'],good)
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        path.write_text(json.dumps(original));self.assertEqual(self.s.claim_job(job['job'],job['revision'],good)['status'],'dispatching')

    def test_local_reconstruction_binds_review_identity_and_actual_input_bytes(self):
        review=self.reviewed();self.local_reconstruction_policy(review)
        other=self.completed(key='same-bytes-different-image-job')
        other_review=self.s.review_reference(other['handle'],0,'useful','eye',0,'Synthetic','matched','matched','closed',[])['review']
        with self.assertRaisesRegex(ValueError,'source/review identity'):self.local_reconstruction_request(other_review)
        job=self.local_reconstruction_request(review);Path(job['transport']['input_paths'][0]).write_bytes(b'changed prepared input')
        with self.assertRaisesRegex(ValueError,'integrity'):self.s.claim_job(job['job'],job['revision'],self.local_reconstruction_preflight())
        self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])

    def test_authorized_comparison_prepares_and_claims_actual_same_source(self):
        review=self.reviewed();self.comparison_policy()
        prepared=self.comparison_request(review,provider='Tripo HD Model v3.1',settings={'mode':'HD_Model'})
        self.assertEqual(digest(Path(prepared['transport']['input_paths'][0]).read_bytes()),self.comparison['source_sha256'])
        self.assertEqual(prepared['status'],'prepared')
        self.assertEqual(prepared['transport']['submission'],'not submitted by preparation')
        claimed=self.s.claim_job(prepared['job'],prepared['revision'],self.comparison_preflight())
        self.assertEqual(claimed['status'],'dispatching')
        self.assertEqual(claimed['intent']['comparison_authorization'],{'comparison_id':self.comparison_id,'policy':self.comparison})
        # A fake transport supplies an output without invoking any provider/native adapter.
        completed=self.s.reconcile_job(prepared['job'],claimed['revision'],'completed',provider_id='fake-hd-job',
            output_paths=[str(self.root/'native.npz')],receipt={'fake_transport':True,'credits_spent':0})
        self.assertEqual(completed['status'],'completed')
        self.assertEqual(completed['data']['provider_id'],'fake-hd-job')

    def test_comparison_rejects_unrelated_or_missing_scope_route_and_retry_key(self):
        review=self.reviewed();self.comparison_policy()
        before=list(self.s.ledger.root.glob('*.json'))
        cases=[({'comparison_id':'other'},None,'Tripo Studio'),({'comparison_id':None},None,'Tripo Studio'),
               ({'comparison_id':''},None,'Tripo Studio'),({'comparison_id':True},None,'Tripo Studio'),
               ({'model':'v3.0'},None,'Tripo Studio'),({'mode':'Smart Mesh','model':'P2.0'},None,'Tripo Studio'),
               ({},'another-hd-job','Tripo Studio'),({},None,'Meshy'),
               ({'provider_id':'tripo'},None,'Meshy'),({'provider_id':'meshy'},None,'Tripo Studio'),
               ({},None,'Tripo Smart Mesh P2.0')]
        for settings,key,provider in cases:
            with self.subTest(settings=settings,key=key,provider=provider),self.assertRaises(ValueError):
                self.comparison_request(review,settings,key,provider)
            self.assertEqual(list(self.s.ledger.root.glob('*.json')),before)
        # An arbitrary client assertion cannot authorize HD without a configured comparison.
        with self.assertRaises(ValueError):
            self.s.prepare_guide(review,'Tripo Studio',{'mode':'HD Model','model':'v3.1','allow_hd':True,
                'source_sha256':self.comparison['source_sha256']},{'scope':'allow HD','source':'caller'},'bypass')

    def test_comparison_rejects_different_reviewed_bytes_despite_asserted_hash(self):
        self.comparison_policy()
        other=self.root/'other.png';Image.new('RGB',(200,100),'blue').save(other)
        output=self.completed(key='other-source',path=other)
        review=self.s.review_reference(output['handle'],0,'useful','eye',0,'Synthetic','matched','matched','closed',[])['review']
        before=list(self.s.ledger.root.glob('*.json'))
        with self.assertRaisesRegex(ValueError,'authorized comparison source'):
            self.comparison_request(review,{'source_sha256':self.comparison['source_sha256']})
        self.assertEqual(list(self.s.ledger.root.glob('*.json')),before)

    def test_comparison_preparation_checks_bytes_before_creating_ledger_job(self):
        review=self.reviewed();self.comparison_policy()
        source=self.s.ledger.read(self.s.store.get(review,'reference_review')['job'])['data']['outputs'][0]
        self.s.store.resolve_blob(source).write_bytes(b'changed bytes behind original hash')
        before=list(self.s.ledger.root.glob('*.json'))
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):
            self.comparison_request(review)
        self.assertEqual(list(self.s.ledger.root.glob('*.json')),before)

    def test_comparison_claim_rechecks_bytes_before_mutating_ledger(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review)
        Path(job['transport']['input_paths'][0]).write_bytes(b'corrupted after preparation')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):
            self.s.claim_job(job['job'],job['revision'],self.comparison_preflight())
        self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])

    def test_comparison_claim_rechecks_current_policy_and_snapshot(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review)
        path=self.s.references.generation_policy;original=json.loads(path.read_text())
        variants=[]
        for key,value in [('status','revoked'),('source_sha256','0'*64),('model','v3.0'),
                          ('idempotency_key','another-key'),('authorization',{'scope':'different','source':'changed policy'})]:
            policy=deepcopy(original);policy['tripo']['authorized_comparisons'][self.comparison_id][key]=value
            variants.append(policy)
        policy=deepcopy(original);policy['tripo'].pop('authorized_comparisons');variants.append(policy)
        for policy in variants:
            path.write_text(json.dumps(policy),encoding='utf-8')
            with self.subTest(policy=policy),self.assertRaises(ValueError):
                self.s.claim_job(job['job'],job['revision'],self.comparison_preflight())
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
            with self.assertRaises((ValueError,Conflict)):
                self.comparison_request(review)
        path.unlink()
        with self.assertRaisesRegex(ValueError,'unavailable'):
            self.s.claim_job(job['job'],job['revision'],self.comparison_preflight())
        self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        path.write_text(json.dumps(original),encoding='utf-8')
        self.assertEqual(self.s.claim_job(job['job'],job['revision'],self.comparison_preflight())['status'],'dispatching')

    def test_comparison_wrong_or_stale_panel_never_mutates_ledger(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review);good=self.comparison_preflight()
        variants=[None,dict(good,mode='Smart Mesh',model='P2.0'),dict(good,model='v3.0'),dict(good,source=''),
                  dict(good,source_sha256='0'*64),dict(good,source_sha256=None),dict(good,displayed_generation_credits=65),
                  dict(good,observed_balance=29),dict(good,observed_balance=True),dict(good,observed_at='invalid'),
                  dict(good,source=None),dict(good,observed_at=None),dict(good,observed_balance=float('inf')),
                  dict(good,observed_at=datetime.now().isoformat()),
                  dict(good,observed_at=(datetime.now(timezone.utc)-timedelta(minutes=11)).isoformat()),
                  dict(good,observed_at=(datetime.now(timezone.utc)+timedelta(minutes=1)).isoformat())]
        for preflight in variants:
            with self.subTest(preflight=preflight),self.assertRaises(ValueError):
                self.s.claim_job(job['job'],job['revision'],preflight)
            self.assertEqual(self.s.inspect_workflow(job['job'])['revision'],job['revision'])
        self.assertEqual(self.s.claim_job(job['job'],job['revision'],good)['status'],'dispatching')

    def test_comparison_unknown_job_recovered_after_restart_never_replayed(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review)
        same=self.comparison_request(review);self.assertTrue(same['reused']);self.assertEqual(same['job'],job['job'])
        with self.assertRaises(Conflict):self.comparison_request(review,{'polycount':100000})
        claim=self.s.claim_job(job['job'],job['revision'],self.comparison_preflight())
        unknown=self.s.reconcile_job(job['job'],claim['revision'],'unknown',receipt={'fake_transport':'response lost'})
        self.s=ModelingService(workspace=self.root,store=self.root/'store')
        same=self.comparison_request(review);self.assertTrue(same['reused']);self.assertEqual(same['status'],'unknown')
        with self.assertRaises(Conflict):self.s.claim_job(job['job'],unknown['revision'],self.comparison_preflight())
        with self.assertRaises(ValueError):self.comparison_request(review,key='retry')
        completed=self.s.reconcile_job(job['job'],unknown['revision'],'completed',provider_id='recovered-fake-job',
            output_paths=[str(self.root/'native.npz')],receipt={'recovered':True})
        self.assertEqual(completed['status'],'completed')
        with self.assertRaises(Conflict):self.s.claim_job(job['job'],completed['revision'],self.comparison_preflight())
        self.assertEqual(len([j for j in self.s.ledger.list() if j['intent']['kind']=='mesh']),1)

    def test_comparison_cancellation_does_not_release_another_hd_dispatch(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review)
        cancelled=self.s.reconcile_job(job['job'],job['revision'],'cancelled',receipt={'reason':'fixture never dispatched'})
        self.assertNotIn('submission_intent',cancelled['data'])
        self.assertEqual(self.comparison_request(review)['status'],'cancelled')
        with self.assertRaises(Conflict):self.s.claim_job(job['job'],cancelled['revision'],self.comparison_preflight())
        with self.assertRaises(ValueError):self.comparison_request(review,key='replacement')

    def test_comparison_claim_rejects_unbound_legacy_or_wrong_job_identity(self):
        review=self.reviewed();self.comparison_policy();job=self.comparison_request(review)
        intent=self.s.inspect_workflow(job['job'])['intent']
        for binding in ('absent','wrong-job','wrong-source'):
            forged=deepcopy(intent)
            if binding=='absent':forged.pop('comparison_authorization')
            if binding=='wrong-source':forged['generation_input']=dict(forged['generation_input'],sha256='0'*64)
            old=self.s.ledger.create('reference_job',forged,'legacy-'+binding)
            with self.subTest(binding=binding),self.assertRaises(ValueError):
                self.s.claim_job(old['handle'],old['revision'],self.comparison_preflight())
            self.assertEqual(self.s.inspect_workflow(old['handle'])['revision'],old['revision'])

    def test_comparison_does_not_change_ordinary_p2_or_legacy_hd_rules(self):
        review=self.reviewed();self.comparison_policy()
        p2=self.tripo_request(review,key='higher-poly-p2',model='P2.0 - Preview')
        self.assertNotIn('comparison_authorization',self.s.inspect_workflow(p2['job'])['intent'])
        good={'mode':'Smart Mesh','model':'P2.0 - Preview','displayed_generation_credits':65,'observed_balance':100,
              'observed_at':datetime.now(timezone.utc).isoformat(),'source':'synthetic panel'}
        self.assertEqual(self.s.claim_job(p2['job'],p2['revision'],good)['status'],'dispatching')
        # Same source alone is insufficient to retroactively permit any old HD preparation.
        path=self.s.references.generation_policy;self.s.references.generation_policy=None
        old=self.tripo_request(review,key='old-hd',mode='HD Model',model='v3.1')
        self.s.references.generation_policy=path
        with self.assertRaisesRegex(ValueError,'Smart Mesh / P2.0'):
            self.s.claim_job(old['job'],old['revision'],self.comparison_preflight())
        self.assertEqual(self.s.inspect_workflow(old['job'])['revision'],old['revision'])

    def test_zoomed_projection_subtracts_crop_without_rescaling_geometry(self):
        points=self.s.project_points(self.ob,[[0,0,0],[.8,0,0]])['points']
        self.assertEqual(points[0]['pixel'],[100,50]);self.assertTrue(points[0]['inside_crop']);self.assertFalse(points[1]['inside_crop'])
        hit=self.s.query_pixel(self.ob,[100,50],['Face'])['hits'][0]
        np.testing.assert_allclose(hit['point'],[0,0,0]);self.assertEqual(hit['object'],'Face')

    def test_capture_state_mismatch_and_image_corruption_fail(self):
        with self.assertRaises(ValueError):self.s.record_observation(self.state,str(self.image),dict(self.meta,scene_state_id='wrong'))
        ob=self.s.store.get(self.ob,'observation');self.s.store.resolve_blob(ob['image']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.request()

    def test_matching_depth_is_pinned_and_mixed_state_diagnostics_refused(self):
        depth=self.root/'surface.npz';np.savez_compressed(depth,depth=np.ones((2,2)))
        bundle={'state_id':self.sid,'diagnostics':{'surface':{'state_id':self.sid,'path':str(depth)}}}
        pinned=self.s._pin_diagnostics(bundle)['surface']['stored']['path']
        depth.write_bytes(b'changed source')
        with np.load(self.s.store.resolve_blob(pinned)) as arrays:np.testing.assert_array_equal(arrays['depth'],np.ones((2,2)))
        bundle['diagnostics']['surface']['state_id']='other'
        with self.assertRaises(ValueError):self.s._pin_diagnostics(bundle)

    def test_video_requires_reviewed_same_view_and_disables_audio(self):
        with self.assertRaises(ValueError):self.request('video')
        review=self.reviewed();job=self.request('video',review,'video')
        self.assertFalse(job['transport']['settings']['generate_audio']);self.assertEqual(len(job['transport']['input_paths']),1)
        self.assertEqual(self.s.inspect_workflow(job['job'])['status'],'prepared')

    def test_rejected_likeness_blocks_old_review_and_prepared_dispatch(self):
        review=self.reviewed();source=self.s.store.get(review,'reference_review')
        prepared=self.request('mesh',review,'waiting-mesh')
        self.s.review_reference(source['job'],0,'rejected','whole image',0,'User says this is not the character','off model','same view','closed',[])
        with self.assertRaises(ValueError):self.s.claim_job(prepared['job'],prepared['revision'])
        with self.assertRaises(ValueError):self.request('mesh',review,'stale-positive-review')
        with self.assertRaises(ValueError):self.s.review_reference(source['job'],0,'useful','eye',10,'Agent tries to revive image','same','same','closed',[])
        self.assertEqual(self.s.inspect_workflow(prepared['job'])['status'],'prepared')

    def test_later_rejection_blocks_previously_qualified_guide_installation(self):
        review=self.reviewed();source=self.s.store.get(review,'reference_review')
        job=self.completed('mesh',review,'qualified-mesh',self.root/'native.npz')
        result=self.s.qualify_guide(job['handle'],0,self.state,self.a['co'].tolist(),self.a['co'].tolist(),'Closed','eye','Closed','partial_target',.001,
            [{'meaning':'skin','selection':{'component':0},'correspondence_evidence':'fixture'}],{'path':'fixture','sha256':'fixture'},review)
        self.s.review_reference(source['job'],0,'rejected','whole image',0,'Later rejection','off model','same','closed',[])
        class NoNative:
            def call(self,*args,**kwargs):raise AssertionError('Rejected target must never reach native mutation')
        self.s.native=NoNative()
        with self.assertRaises(ValueError):self.s.integrate_guide(result['qualification'],'state','owner','Test','eyes',{},[])

    def test_registration_preserves_source_and_rejects_false_pose_or_anchors(self):
        review=self.reviewed();job=self.completed('mesh',review,'mesh',self.root/'native.npz')
        kwargs=dict(job=job['handle'],output_index=0,native_state=self.state,source_anchors=self.a['co'].tolist(),
                    native_anchors=(self.a['co']*2+[1,2,3]).tolist(),pose='Closed',region='eye',source_pose='Closed',role='partial_target',
                    maximum_anchor_error=.0001,support=[{'meaning':'skin','selection':{'component':0},'correspondence_evidence':'fixture exact triangle'}],
                    expected_registry={'path':'fixture','sha256':'fixture'},review=review)
        result=self.s.qualify_guide(**kwargs)
        with np.load(self.s.store.resolve_blob(result['registered'])) as arrays:np.testing.assert_allclose(arrays['co'],self.a['co']*2+[1,2,3])
        with np.load(self.root/'native.npz') as arrays:np.testing.assert_array_equal(arrays['co'],self.a['co'])
        with self.assertRaises(ValueError):self.s.qualify_guide(**dict(kwargs,source_pose='Neutral'))
        with self.assertRaises(ValueError):self.s.qualify_guide(**dict(kwargs,source_anchors=[[0,0,0],[1,0,0],[2,0,0]]))
        with self.assertRaises(ValueError):self.s.qualify_guide(**dict(kwargs,support=[{'meaning':'skin','selection':{'component':9},'correspondence_evidence':'none'}]))
        self.s.reject_guide(job['handle'],'Actual target skin is absent',[str(self.image)])
        with self.assertRaisesRegex(ValueError,'guide was rejected'):self.s.qualify_guide(**kwargs)
        with self.assertRaisesRegex(ValueError,'guide was rejected'):self.s.integrate_guide(result['qualification'],'unused','unused','Test','eyes',{},[])

    def test_high_level_integration_freezes_historical_anatomy_but_keeps_current_guards(self):
        from .store import digest
        review=self.reviewed();job=self.completed('mesh',review,'immutable-anatomy',self.root/'native.npz')
        registry={'path':'fixture-current-registry','sha256':'fixture-current-hash'}
        q=self.s.qualify_guide(job['handle'],0,self.state,self.a['co'].tolist(),self.a['co'].tolist(),'Closed','eye','Closed','partial_target',.001,
            [{'meaning':'skin','selection':{'component':0},'correspondence_evidence':'fixture'}],registry,review)
        source=self.root/'anatomy.md';source.write_text('Historical construction evidence');expected=digest(source.read_bytes());received=[]
        class Native:
            def call(self,operation,arguments,**kwargs):received.append(arguments);return {'status':'completed','fixture_only':True}
        self.s.native=Native()
        self.s.integrate_guide(q['qualification'],'current-native-token','owner','Fixture','eyes',{'blink':1},[str(source)])
        native=received[0];pinned=native['qualification']['provenance']['anatomy_basis'][0]
        source.write_text('Later experiment log')
        self.assertNotEqual(pinned['path'],str(source));self.assertEqual(digest(Path(pinned['path']).read_bytes()),expected)
        self.assertEqual(native['expected_state'],'current-native-token');self.assertEqual(native['expected_registry'],registry)
        self.assertEqual(native['qualification']['pose'],{'blink':1})
        self.assertEqual(native['qualification']['registered_npz']['sha256'],q['registered']['sha256'])

    def test_ray_retains_surface_ownership_and_material_paths(self):
        arrays=dict(self.a,triangle_material=np.array([2]))
        hit=geometry.ray_hits(arrays,[.5,.5,2],[0,0,-1],{'material':2})
        self.assertAlmostEqual(hit['hits'][0]['distance'],2)
        self.assertEqual(geometry.ray_hits(arrays,[.5,.5,2],[0,0,1])['hits'],[])
        self.assertEqual(geometry.material_path(arrays,[0,1])['length'],2)

    def test_lost_trial_response_requires_reconciliation_and_preserves_history(self):
        sid=self.sid
        class Lost:
            def call(self,operation,*a,**k):
                if operation=='inspect_live':return {'expected_state':'state','geometry_state_id':sid}
                raise NativeBridgeError('uncertain_execution','response lost','inspect receipt')
        q=self.s.open_question('Eye repair',self.state,'eye','Preserve eye volume')['question']
        e=self.s.begin_experiment(q,'Test','Compare native surface',[],'zero paid calls','trial')
        proposal={'kind':'script','label':'test','feature':'eyes','mismatch':'fixture','mechanism':'fixture',
                  'constraints':[{'path':str(self.image),'sha256':digest(self.image.read_bytes())}],
                  'allowed_objects':['Face'],'script':{'path':str(self.image),'sha256':digest(self.image.read_bytes())}}
        proposed=self.s.propose_change(e['handle'],e['revision'],proposal,[str(self.image)])
        self.s.native=Lost()
        with self.assertRaises(NativeBridgeError):self.s.apply_trial(e['handle'],proposed['revision'],'state','owner')
        current=self.s.inspect_workflow(e['handle']);self.assertEqual(current['status'],'needs_reconciliation')
        with self.assertRaises(Conflict):self.s.apply_trial(e['handle'],current['revision'],'state','owner')
        self.assertEqual(self.s.store.get(proposed['revision'],'workflow')['status'],'proposed')

    def test_live_token_cannot_authorize_a_proposal_from_a_different_baseline(self):
        class Changed:
            def call(self,operation,*a,**k):
                if operation!='inspect_live':raise AssertionError('Mutation must not run')
                return {'expected_state':'fresh-token','geometry_state_id':'different-geometry'}
        q=self.s.open_question('Eye repair',self.state,'eye','Preserve volume')['question']
        e=self.s.begin_experiment(q,'Test','Compare',[],'zero','stale-trial');self.s.native=Changed()
        with self.assertRaises(Conflict):self.s.apply_trial(e['handle'],e['revision'],'fresh-token','owner')
        self.assertEqual(self.s.inspect_workflow(e['handle'])['status'],'prepared')

    def test_motion_mismatched_views_and_reference_phase_are_explicit(self):
        other=self.s.record_observation(self.state,str(self.image),dict(self.meta,crop=[200,350,200,100]))['observation']
        result=self.s.compare_motion([self.ob],[other],[{'phase':'closing','control':.5}],reference_mapping=[{'source_time':.2,'basis':'manual'}])
        self.assertEqual(result['rows'][0]['view_disagreements'],['crop']);self.assertEqual(result['coverage']['continuous_motion'],'finite samples only')

    def test_live_query_refreshes_dirty_scene_and_rejects_intervening_edit(self):
        sid=self.sid;path=str(self.root/'native.json')
        class Live:
            def __init__(self,change_after=None):self.calls=0;self.change_after=change_after;self.refreshed=False
            def call(self,operation,args):
                self.calls+=1
                if args.get('refresh_scene'):self.refreshed=True
                changed=self.change_after is not None and self.calls>self.change_after
                return {'expected_state':'new' if changed else 'same','geometry_state_id':None if changed else sid,
                        'scene_freshness':{'status':'dirty' if changed else 'current','record_path':path}}
        self.s.native=Live()
        result=self.s.query_live_geometry('Face','bounds',{})
        self.assertTrue(self.s.native.refreshed);self.assertEqual(result['summary']['vertices'],3)
        self.s.native=Live(change_after=2)
        with self.assertRaises(Conflict):self.s.query_live_geometry('Face','bounds',{})

    def test_evaluated_nonmesh_surface_is_queryable(self):
        path=self.root/'native.json';record=json.loads(path.read_text())
        record['state']['objects'][0]['type']='CURVE'
        record['state']['objects'][0]['evaluation']='evaluated curve surface'
        record['state_id']=digest(canonical(record['state']));path.write_text(json.dumps(record))
        state=self.s.import_scene(str(path));self.assertEqual(state['mesh_objects'],1)
        result=self.s.query_geometry(state['state'],'Face','bounds',{})
        self.assertEqual(result['summary']['vertices'],3)

    def test_mcp_real_schemas_and_calls(self):
        from .mcp_server import ModelingMCP
        async def run():
            m=ModelingMCP(self.s);tools=await m.list_tools();capture=next(t for t in tools if t.name=='native_capture_view')
            self.assertIn('crop',capture.input_schema['properties']);self.assertIn('owner',capture.input_schema['required'])
            result=await m.call_tool('query_geometry',dict(state=self.state,object_name='Face',query='nearest',parameters={'point':[.5,.5,3]}))
            self.assertEqual(result.structured_content['summary']['distance'],3)
            bad=await m.call_tool('query_geometry',dict(state='bad',object_name='Face',query='bounds',parameters={}))
            self.assertEqual(bad.structured_content['status'],'failed')
            self.assertTrue(bad.is_error)
        asyncio.run(run())


if __name__=='__main__':unittest.main()
