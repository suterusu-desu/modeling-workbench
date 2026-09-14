"""Persisted provider-workflow composition over actual jobs, reviews and recipe steps; no dispatch or pricing."""
import copy
from datetime import datetime, timedelta, timezone
import unittest
from unittest.mock import patch
from . import provider_workflow as pw
from . import test_decisions as episode_cases
from .ledger import Conflict
from .service import ModelingService
from .store import canonical, digest


class ProviderWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.base=episode_cases.EpisodeTests(); self.base.setUp(); self.addCleanup(self.base.tearDown)
        self.s=self.base.s; self.root=self.base.root; self.image=self.base.image; self.ob=self.base.ob
        self.episode=self.base.episode()['episode']
        self.graph=dict(schema_version=1,name='Image, review gate, mesh',nodes=[
            dict(id='image',kind='job',expects={'kind':'image'}),
            dict(id='gate',kind='review',source='image'),
            dict(id='mesh',kind='job',expects={'kind':'mesh'},source='gate')])

    def create(self,key='one',graph=None):return pw.create(self.s,self.episode,graph or self.graph,key)
    def job(self,kind='image',key='img',review=None,credits=None):
        settings={'displayed_generation_credits':credits} if credits is not None else {}
        return self.s.request_reference(self.ob,[str(self.image)],'Close the eye',kind,'fixture transport',settings,
            {'scope':'local fixture; no dispatch','source':'unit test'},key,review)
    def finish(self,job,status='completed'):
        claimed=self.s.claim_job(job['job'],job['revision'])
        if status=='completed':
            return self.s.reconcile_job(job['job'],claimed['revision'],'completed',output_paths=[str(self.image)],receipt={'test_fixture':True})
        return self.s.reconcile_job(job['job'],claimed['revision'],status,receipt={'test_fixture':True})
    def review(self,handle):
        return self.s.review_reference(handle,0,'useful','eye',10,'Fixture observation','unchanged','matched','closed',[])['review']
    def bind(self,state,bindings,reason=''):return pw.bind(self.s,state['workflow'],state['revision'],bindings,reason)
    def quote(self,state,node,amount,unit='fixture:credits',**overrides):
        now=datetime.now(timezone.utc)
        args=dict(observed_at=(now-timedelta(seconds=5)).isoformat(),expires_at=(now+timedelta(hours=1)).isoformat(),
                  source='synthetic displayed price; no provider contact',
                  evidence=[{'kind':'file','path':str(self.image),'role':'synthetic price evidence'}])
        args.update(overrides)
        return pw.quote(self.s,state['workflow'],state['revision'],node,amount,unit,**args)
    def rows(self,state):return {r['id']:r for r in state['nodes']}
    def snapshot(self):return {str(p):digest(p.read_bytes()) for p in self.s.store.root.rglob('*') if p.is_file()}
    def lineage(self):
        """Completed reviewed image, bound gate and a prepared mesh job descending from that review."""
        image=self.finish(self.job());review=self.review(image['handle']);mesh=self.job('mesh','mesh-one',review)
        state=self.bind(self.create(),{'image':{'job':image['handle']},'gate':{'review':review},'mesh':{'job':mesh['job']}})
        return state,image,review,mesh

    def test_bad_graphs_are_refused_before_any_record(self):
        before=sorted(self.s.ledger.root.glob('*.json'))
        mutations=[lambda g:g['nodes'][0].update(depends_on=['mesh']),lambda g:g['nodes'][1].update(source='mesh'),
                   lambda g:g['nodes'][2].update(source='missing'),lambda g:g['nodes'][2].update(expects={'kind':'image'}),
                   lambda g:g['nodes'][0].update(kind='provider'),lambda g:g['nodes'].append(dict(id='image',kind='job')),
                   lambda g:g.update(schema_version=2),lambda g:g['nodes'][0].update(expects={'model':'v3.1'}),
                   lambda g:g['nodes'].append(dict(id='d',kind='diagnostic',source='image')),
                   lambda g:g['nodes'][1].update(expects={'kind':'image'}),lambda g:g.update(nodes=[]),
                   lambda g:g['nodes'].append(dict(id='d',kind='diagnostic',expects=None)),lambda g:g['nodes'][0].update(expects=None)]
        for mutate in mutations:
            graph=copy.deepcopy(self.graph);mutate(graph)
            with self.subTest(graph=graph),self.assertRaises(ValueError):self.create('bad',graph)
        with self.assertRaises(ValueError):self.create('',self.graph)
        self.assertEqual(sorted(self.s.ledger.root.glob('*.json')),before)

    def test_prepared_binding_is_not_submission_and_preview_grants_nothing(self):
        prepared=self.job(credits=65);state=self.create()
        self.assertTrue(any(l['id']==state['workflow'] for l in self.s.decision_workspace(self.episode,detail='links')['context']['links'] if l['kind']=='workflow'))
        state=self.bind(state,{'image':{'job':prepared['job']}})
        row=self.rows(state)['image']
        self.assertEqual(row['status'],'prepared');self.assertTrue(row['ready_for_dispatch'])
        self.assertEqual(row['next'][0]['operation'],'claim_job');self.assertIn('grants no authority',row['next'][0]['reason'])
        self.assertEqual(self.rows(state)['mesh']['status'],'planned');self.assertEqual(self.rows(state)['mesh']['blocked_by'],['gate'])
        self.assertEqual(state['summary']['ready_for_dispatch'],['image']);self.assertFalse(state['dispatch_authorized'])
        self.assertLess(len(canonical(state)),8000)
        before=self.snapshot()
        cost=pw.preview(self.s,state['workflow']);pw.inspect(self.s,state['workflow'])
        self.assertEqual(self.snapshot(),before)
        self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{});self.assertFalse(cost['aggregate']['remaining']['complete'])
        self.assertEqual(cost['aggregate']['remaining']['unknown_nodes'],['image','mesh'])
        image=next(r for r in cost['nodes'] if r['id']=='image')
        self.assertEqual(image['quote']['status'],'missing');self.assertEqual(image['cost_role'],'remaining')
        self.assertEqual(image['job_evidence']['displayed_generation_credits_at_preparation'],65)
        self.assertFalse(cost['dispatch_authorized']);self.assertFalse(cost['prices_fetched'])
        state=self.quote(state,'image',65)
        self.assertEqual(state['aggregate']['remaining']['known_subtotals'],{'fixture:credits':'65'})
        self.assertEqual(state['aggregate']['remaining']['unknown_nodes'],['mesh']);self.assertFalse(state['aggregate']['remaining']['complete'])
        quote=self.s.store.get(state['quote'],'provider_quote')
        self.assertEqual(quote['basis']['job'],prepared['job']);self.assertEqual(quote['basis']['intent'],digest(canonical(self.s.inspect_workflow(prepared['job'])['intent'])))
        self.assertEqual(quote['evidence'][0]['sha256'],digest(self.image.read_bytes()))
        job=self.s.inspect_workflow(prepared['job']);self.assertEqual(job['status'],'prepared');self.assertNotIn('submission_intent',job['data'])

    def test_busy_unknown_and_completed_jobs_never_become_dispatch_suggestions(self):
        prepared=self.job(credits=65);state=self.quote(self.bind(self.create(),{'image':{'job':prepared['job']}}),'image',65)
        claimed=self.s.claim_job(prepared['job'],prepared['revision'])
        cost=pw.preview(self.s,state['workflow']);image=next(r for r in cost['nodes'] if r['id']=='image')
        self.assertEqual((image['status'],image['cost_role']),('dispatching','busy'))
        self.assertEqual(cost['aggregate']['busy']['quoted_subtotals'],{'fixture:credits':'65'})
        self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{});self.assertNotIn('image',cost['aggregate']['remaining']['nodes'])
        self.assertEqual(image['next'][0]['operation'],'reconcile_job');self.assertEqual(cost['next_steps'][0]['node'],'image')
        unknown=self.s.reconcile_job(prepared['job'],claimed['revision'],'unknown',receipt={'fixture':'response lost'})
        cost=pw.preview(self.s,state['workflow']);image=next(r for r in cost['nodes'] if r['id']=='image')
        self.assertEqual((image['status'],image['cost_role']),('unknown','busy'))
        self.assertEqual(image['next'][0],dict(operation='reconcile_job',arguments=dict(job=prepared['job'],expected_revision=unknown['revision']),reason=image['next'][0]['reason']))
        self.s.reconcile_job(prepared['job'],unknown['revision'],'completed',output_paths=[str(self.image)],receipt={'recovered':True})
        cost=pw.preview(self.s,state['workflow']);image=next(r for r in cost['nodes'] if r['id']=='image')
        self.assertEqual(image['cost_role'],'completed');self.assertEqual(cost['aggregate']['completed']['quoted_subtotals'],{'fixture:credits':'65'})
        self.assertEqual(image['next'][0]['operation'],'review_reference');self.assertEqual(image['quote']['status'],'known')
        self.assertFalse(any(step.get('operation')=='claim_job' for step in cost['next_steps']))
        inspected=pw.inspect(self.s,state['workflow']);rows=self.rows(inspected)
        self.assertEqual(inspected['summary']['ready_for_dispatch'],[]);self.assertTrue(rows['image']['satisfied'])
        self.assertEqual(rows['gate']['status'],'planned');self.assertEqual(rows['gate']['next'][0]['operation'],'review_reference')
        self.assertEqual(rows['mesh']['blocked_by'],['gate'])

    def test_source_rejection_blocks_downstream_and_stales_its_quote(self):
        state,image,review,mesh=self.lineage();rows=self.rows(state)
        self.assertEqual(rows['gate']['status'],'usable');self.assertTrue(rows['mesh']['ready_for_dispatch'])
        self.assertEqual(rows['mesh']['source_review'],dict(review=review,status='usable'))
        state=self.quote(state,'mesh',30)
        self.assertEqual(state['aggregate']['remaining']['known_subtotals'],{'fixture:credits':'30'});self.assertTrue(state['aggregate']['remaining']['complete'])
        self.assertEqual(state['aggregate']['completed']['unquoted_nodes'],['image'])
        self.s.review_reference(image['handle'],0,'rejected','whole image',0,'Not the character','off model','same view','closed',[])
        cost=pw.preview(self.s,state['workflow']);rows={r['id']:r for r in cost['nodes']}
        self.assertEqual(rows['gate']['status'],'rejected');self.assertFalse(rows['mesh']['ready_for_dispatch'])
        self.assertEqual(rows['mesh']['quote']['status'],'stale');self.assertIn('review',rows['mesh']['quote']['reasons'][0])
        self.assertEqual(cost['aggregate']['remaining']['unknown_nodes'],['mesh']);self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{})
        inspected=pw.inspect(self.s,state['workflow'])
        self.assertIn('gate',inspected['summary']['invalidated']);self.assertIn('mesh',inspected['summary']['blocked'])
        self.assertEqual(self.rows(inspected)['mesh']['source_review']['status'],'rejected')
        with self.assertRaises(ValueError):self.s.claim_job(mesh['job'],mesh['revision'])
        other=self.bind(self.create('two'),{'image':{'job':image['handle']}})
        with self.assertRaisesRegex(ValueError,'rejected'):self.bind(other,{'gate':{'review':review}})
        self.assertEqual(self.rows(pw.inspect(self.s,other['workflow']))['gate']['status'],'planned')

    def test_rebinding_upstream_invalidates_descendants_and_keeps_history(self):
        state,image,review,mesh=self.lineage();state=self.quote(state,'mesh',30)
        other=self.finish(self.job(key='img-two'))
        with self.assertRaisesRegex(ValueError,'reason'):self.bind(state,{'image':{'job':other['handle']}})
        state=self.bind(state,{'image':{'job':other['handle']}},'Second image proposal replaces the first')
        rows=self.rows(state)
        self.assertEqual(rows['image']['job'],other['handle']);self.assertEqual(rows['image']['history'],1)
        self.assertEqual(rows['gate']['status'],'invalidated');self.assertEqual(rows['mesh']['binding_validity'],'invalidated')
        self.assertFalse(rows['mesh']['ready_for_dispatch']);self.assertEqual(sorted(state['summary']['invalidated']),['gate','mesh'])
        history=self.s.ledger.read(state['workflow'])['data']['history']
        self.assertEqual(history[0]['previous']['job'],image['handle']);self.assertEqual(history[0]['observed_status'],'completed')
        self.assertEqual(history[0]['replaced_by'],{'job':other['handle']})
        cost=pw.preview(self.s,state['workflow']);mesh_row=next(r for r in cost['nodes'] if r['id']=='mesh')
        self.assertEqual(mesh_row['quote']['status'],'stale');self.assertEqual(cost['aggregate']['remaining']['unknown_nodes'],['mesh'])
        with self.assertRaisesRegex(ValueError,'invalidated'):self.quote(state,'mesh',30)
        state=self.bind(state,{'image':{'job':image['handle']}},'Return to the reviewed first proposal')
        rows=self.rows(state)
        self.assertEqual(rows['gate']['status'],'usable');self.assertEqual(rows['mesh']['binding_validity'],'current');self.assertTrue(rows['mesh']['ready_for_dispatch'])
        cost=pw.preview(self.s,state['workflow'])
        self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{'fixture:credits':'30'});self.assertEqual(rows['image']['history'],2)

    def test_dependency_change_requires_explicit_acknowledgement(self):
        graph=dict(schema_version=1,name='Sequential images',nodes=[dict(id='first',kind='job',expects={'kind':'image'}),
            dict(id='second',kind='job',expects={'kind':'image'},depends_on=['first'])])
        first=self.finish(self.job(key='first'));second=self.job(key='second')
        state=self.bind(self.create('sequence',graph),{'first':{'job':first['handle']},'second':{'job':second['job']}})
        self.assertTrue(self.rows(state)['second']['ready_for_dispatch']);state=self.quote(state,'second',10)
        replacement=self.finish(self.job(key='first-replacement'))
        state=self.bind(state,{'first':{'job':replacement['handle']}},'Replaced the completed first image')
        row=self.rows(state)['second']
        self.assertEqual(row['dependency_changes'],['first']);self.assertFalse(row['ready_for_dispatch']);self.assertEqual(row['binding_validity'],'current')
        self.assertEqual(next(r for r in pw.preview(self.s,state['workflow'])['nodes'] if r['id']=='second')['quote']['status'],'stale')
        state=self.bind(state,{'second':{'job':second['job']}},'Acknowledged the replaced upstream image')
        self.assertTrue(self.rows(state)['second']['ready_for_dispatch']);self.assertNotIn('dependency_changes',self.rows(state)['second'])
        self.assertEqual(self.rows(state)['second']['history'],1)

    def test_instances_are_isolated_and_a_job_binds_at_one_node_only(self):
        prepared=self.job();one=self.create('one');two=self.create('two')
        self.assertNotEqual(one['workflow'],two['workflow']);self.assertFalse(one['reused'])
        same=self.create('one');self.assertEqual(same['workflow'],one['workflow']);self.assertTrue(same['reused'])
        with self.assertRaises(Conflict):self.create('one',dict(self.graph,name='Different graph under the same key'))
        one=self.bind(one,{'image':{'job':prepared['job']}})
        self.assertEqual(self.rows(pw.inspect(self.s,two['workflow']))['image']['status'],'planned')
        graph=dict(schema_version=1,name='Two image slots',nodes=[dict(id='a',kind='job',expects={'kind':'image'}),dict(id='b',kind='job',expects={'kind':'image'})])
        pair=self.create('pair',graph)
        with self.assertRaisesRegex(ValueError,'charged twice'):self.bind(pair,{'a':{'job':prepared['job']},'b':{'job':prepared['job']}})
        pair=self.bind(pair,{'a':{'job':prepared['job']}})
        with self.assertRaisesRegex(ValueError,'charged twice'):self.bind(pair,{'b':{'job':prepared['job']}})
        self.assertEqual(self.rows(pw.inspect(self.s,pair['workflow']))['b']['status'],'planned')
        other=ModelingService(workspace=self.root/'other',store=self.s.store.root)
        with self.assertRaisesRegex(ValueError,'different workspace'):pw.inspect(other,one['workflow'])
        with self.assertRaisesRegex(ValueError,'Expected provider workflow'):pw.inspect(self.s,prepared['job'])
        fresh=ModelingService(workspace=self.s.workspace,store=self.s.store.root)
        self.assertEqual(self.rows(pw.inspect(fresh,one['workflow']))['image']['job'],prepared['job'])

    def test_bindings_are_atomic_and_stale_revisions_are_refused(self):
        prepared=self.job();done=self.finish(self.job(key='img-done'));mesh=self.job('mesh','mesh-one',self.review(done['handle']))
        state=self.create();old=state['revision']
        with self.assertRaisesRegex(ValueError,'does not satisfy'):self.bind(state,{'image':{'job':mesh['job']}})
        with self.assertRaisesRegex(ValueError,'Bind source node gate'):self.bind(state,{'image':{'job':prepared['job']},'mesh':{'job':mesh['job']}})
        with self.assertRaisesRegex(ValueError,'known named nodes'):self.bind(state,{'unknown':{'job':prepared['job']}})
        with self.assertRaisesRegex(ValueError,'reference job'):self.bind(state,{'image':{'job':self.episode}})
        with self.assertRaisesRegex(ValueError,'Unknown job'):self.bind(state,{'image':{'job':'0'*64}})
        with self.assertRaises(ValueError):self.bind(state,{'image':{'handle':prepared['job']}})
        item=self.s.ledger.read(state['workflow']);self.assertEqual(item['revision'],old);self.assertEqual(item['data']['bindings'],{})
        state=self.bind(state,{'image':{'job':prepared['job']}});self.assertNotEqual(state['revision'],old)
        with self.assertRaises(Conflict):pw.bind(self.s,state['workflow'],old,{'image':{'job':prepared['job']}},'stale writer')
        with self.assertRaises(Conflict):self.quote(dict(state,revision=old),'image',1)
        self.assertEqual(self.s.ledger.read(state['workflow'])['revision'],state['revision'])

    def test_cost_units_stay_separate_and_invalid_quotes_are_refused(self):
        graph=dict(schema_version=1,name='Two providers',nodes=[dict(id='a',kind='job'),dict(id='b',kind='job')])
        a=self.job(key='a');b=self.job(key='b')
        state=self.bind(self.create('units',graph),{'a':{'job':a['job']},'b':{'job':b['job']}})
        state=self.quote(state,'a','12.50',unit='USD');state=self.quote(state,'b',65,unit='tripo:credits')
        self.assertEqual(state['aggregate']['remaining']['known_subtotals'],{'USD':'12.5','tripo:credits':'65'})
        self.assertTrue(state['aggregate']['remaining']['complete'])
        state=self.quote(state,'a','0.25',unit='USD')
        self.assertEqual(state['aggregate']['remaining']['known_subtotals'],{'USD':'0.25','tripo:credits':'65'})
        self.assertEqual(next(r for r in state['nodes'] if r['id']=='a')['quote']['history'],2)
        two=self.bind(self.create('shared-unit',graph),{'a':{'job':self.job(key='c')['job']},'b':{'job':self.job(key='d')['job']}})
        two=self.quote(self.quote(two,'a','0.1',unit='USD'),'b','0.2',unit='USD')
        self.assertEqual(two['aggregate']['remaining']['known_subtotals'],{'USD':'0.3'})
        for amount in (float('nan'),float('inf'),-1,'-0.5',True,None,'abc','1e30','',[65]):
            with self.subTest(amount=amount),self.assertRaises(ValueError):self.quote(state,'a',amount,unit='USD')
        for unit in ('','US Dollars','$',7,None):
            with self.subTest(unit=unit),self.assertRaises(ValueError):self.quote(state,'a',1,unit=unit)
        now=datetime.now(timezone.utc)
        for overrides in [dict(observed_at='yesterday'),dict(observed_at=now.replace(tzinfo=None).isoformat()),
                          dict(observed_at=(now+timedelta(minutes=5)).isoformat()),dict(expires_at=(now-timedelta(seconds=1)).isoformat()),
                          dict(expires_at=(now+timedelta(days=40)).isoformat()),dict(evidence=[]),dict(source=''),
                          dict(evidence=[{'kind':'record','id':'0'*64,'role':'absent'}])]:
            with self.subTest(overrides=overrides),self.assertRaises((ValueError,FileNotFoundError)):self.quote(state,'a',1,unit='USD',**overrides)
        with self.assertRaisesRegex(ValueError,'job nodes'):self.quote(self.create(),'gate',1)
        with self.assertRaisesRegex(ValueError,'unbound'):self.quote(self.create(),'image',1)
        self.assertEqual(self.s.ledger.read(state['workflow'])['revision'],state['revision'])
        with patch('modeling_system.provider_workflow._now',return_value=now+timedelta(hours=2)):
            cost=pw.preview(self.s,state['workflow'])
        self.assertEqual({r['id']:r['quote']['status'] for r in cost['nodes']},{'a':'expired','b':'expired'})
        self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{});self.assertEqual(cost['aggregate']['remaining']['unknown_nodes'],['a','b'])
        self.assertFalse(cost['aggregate']['remaining']['complete'])

    def test_exact_decimal_precision_is_independent_of_global_context(self):
        from decimal import localcontext
        value='999999999999999999.123456789012'
        with localcontext() as context:
            context.prec=6
            self.assertEqual(pw._text(pw._amount(value)),value)
            self.assertEqual(pw._sum(value,value),'1999999999999999998.246913578024')
        self.assertEqual(pw._text(pw._amount('0e-999999999')),'0')
        with self.assertRaises(ValueError):pw._amount('0'*10000)
        graph=dict(schema_version=1,name='Exact prices',nodes=[dict(id='a',kind='job'),dict(id='b',kind='job')])
        state=self.bind(self.create('decimal',graph),{'a':{'job':self.job(key='decimal-a')['job']},'b':{'job':self.job(key='decimal-b')['job']}})
        state=self.quote(self.quote(state,'a',value,'USD'),'b',value,'USD')
        self.assertEqual(state['aggregate']['remaining']['known_subtotals']['USD'],'1999999999999999998.246913578024')

    def test_stale_ancestor_crosses_completed_job_and_review_gate(self):
        self.graph['nodes'].insert(0,dict(id='diagnostic',kind='diagnostic'))
        self.graph['nodes'][1]['depends_on']=['diagnostic']
        current=dict(recipe='synthetic-recipe',revision='synthetic-revision',steps=[dict(id='check',status='reusable',result='retained-result')])
        with patch('modeling_system.recipes.inspect_recipe',side_effect=lambda *args:current):
            state,image,review,mesh=self.lineage()
            state=self.bind(state,{'diagnostic':{'recipe':'synthetic-recipe','step':'check'},'image':{'job':image['handle']}},'Attach diagnostic dependency')
            state=self.quote(state,'mesh','20')
            current['steps'][0]['status']='stale'
            before=self.snapshot();after=pw.inspect(self.s,state['workflow']);cost=pw.preview(self.s,state['workflow'])
            self.assertEqual(before,self.snapshot())
            rows=self.rows(after)
            self.assertFalse(rows['image']['satisfied']);self.assertFalse(rows['gate']['satisfied']);self.assertFalse(rows['mesh']['ready_for_dispatch'])
            self.assertEqual(rows['mesh']['blocked_by'],['gate'])
            self.assertEqual(self.rows(cost)['mesh']['quote']['status'],'stale')
            self.assertFalse(cost['aggregate']['remaining']['complete']);self.assertEqual(cost['aggregate']['remaining']['unknown_nodes'],['mesh'])
            with self.assertRaisesRegex(ValueError,'not usable'):self.quote(after,'mesh','20')

    def test_failed_job_needs_a_distinct_new_branch(self):
        prepared=self.job(credits=65);state=self.quote(self.bind(self.create(),{'image':{'job':prepared['job']}}),'image',65)
        self.finish(prepared,'failed')
        state=pw.inspect(self.s,state['workflow']);row=self.rows(state)['image']
        self.assertEqual(row['status'],'failed');self.assertFalse(row['ready_for_dispatch']);self.assertEqual(state['summary']['needs_new_branch'],['image'])
        self.assertIn('distinct',row['next'][0]['action'])
        cost=pw.preview(self.s,state['workflow']);image=next(r for r in cost['nodes'] if r['id']=='image')
        self.assertEqual(image['cost_role'],'terminal');self.assertEqual(cost['aggregate']['terminal']['nodes'],['image'])
        self.assertEqual(cost['aggregate']['remaining']['known_subtotals'],{});self.assertEqual(cost['aggregate']['busy']['quoted_subtotals'],{})
        self.assertEqual(cost['aggregate']['completed']['quoted_subtotals'],{})
        with self.assertRaisesRegex(ValueError,'new branch'):self.quote(state,'image',65)
        with self.assertRaises(Conflict):self.s.claim_job(prepared['job'],self.s.inspect_workflow(prepared['job'])['revision'])
        with self.assertRaisesRegex(ValueError,'reason'):self.bind(state,{'image':{'job':self.job(key='img-retry',credits=65)['job']}})
        state=self.bind(state,{'image':{'job':self.job(key='img-retry',credits=65)['job']}},'Explicit new branch after provider failure')
        row=self.rows(state)['image']
        self.assertEqual(row['status'],'prepared');self.assertEqual(row['history'],1);self.assertTrue(row['ready_for_dispatch'])
        history=self.s.ledger.read(state['workflow'])['data']['history'][0]
        self.assertEqual((history['previous']['job'],history['observed_status']),(prepared['job'],'failed'))
        self.assertEqual(next(r for r in pw.preview(self.s,state['workflow'])['nodes'] if r['id']=='image')['quote']['status'],'stale')

    def test_diagnostic_recipe_output_gates_a_job_and_input_changes_invalidate_it(self):
        from .test_recipes import RecipeTests
        r=RecipeTests();r.setUp();self.addCleanup(r.doCleanups)
        recipe=r.create()
        graph=dict(schema_version=1,name='Diagnostic gate',nodes=[dict(id='coverage',kind='diagnostic'),
            dict(id='image',kind='job',expects={'kind':'image'},depends_on=['coverage'])])
        state=pw.create(r.s,r.episode,graph,'gated')
        with self.assertRaisesRegex(ValueError,'reusable or accepted'):
            pw.bind(r.s,state['workflow'],state['revision'],{'coverage':{'recipe':recipe['recipe'],'step':'coverage'}})
        with self.assertRaisesRegex(ValueError,'Unknown recipe step'):
            pw.bind(r.s,state['workflow'],state['revision'],{'coverage':{'recipe':recipe['recipe'],'step':'absent'}})
        recipe=r.step(recipe,'coverage')
        state=pw.bind(r.s,state['workflow'],state['revision'],{'coverage':{'recipe':recipe['recipe'],'step':'coverage'}})
        rows=self.rows(state)
        self.assertEqual(rows['coverage']['status'],'satisfied');self.assertEqual(rows['coverage']['result'],r.rows(recipe)['coverage']['result'])
        self.assertNotIn('blocked_by',rows['image']);self.assertEqual(rows['image']['next'][0]['operation'],'request_reference')
        r.coverage.case['question']='A changed coverage question';r.cp.write_bytes(canonical(r.coverage.case))
        r.s.revise_recipe_inputs(recipe['recipe'],recipe['revision'],{'coverage':r.ref(r.cp)})
        before={str(p):digest(p.read_bytes()) for p in r.s.store.root.rglob('*') if p.is_file()}
        rows=self.rows(pw.inspect(r.s,state['workflow']))
        self.assertEqual({str(p):digest(p.read_bytes()) for p in r.s.store.root.rglob('*') if p.is_file()},before)
        self.assertEqual(rows['coverage']['status'],'stale');self.assertEqual(rows['coverage']['step_status'],'stale')
        self.assertEqual(rows['image']['blocked_by'],['coverage']);self.assertEqual(rows['image']['next'][0]['action'],'wait')


if __name__=='__main__':unittest.main()
