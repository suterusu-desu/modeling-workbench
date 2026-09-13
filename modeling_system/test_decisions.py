"""Decision/recovery contracts using temporary stores; never provider/native calls."""
import json
from pathlib import Path
import tempfile
import unittest
from .service import ModelingService
from .store import Store
from .experience import retrieve, passages, applicability


class DecisionTests(unittest.TestCase):
    def test_context_lookups_write_nothing_and_retention_is_explicit(self):
        from .store import digest
        with tempfile.TemporaryDirectory() as d:
            from .init_workspace import initialize
            root=Path(d)/'project'; initialize(root,'Synthetic')
            policy={'tripo':{'mode':'Smart Mesh','model':'P2.0','selection_preferences':{'exact_view_enlarged_local_detail':{'mode':'HD Model','model':'v3.1','status':'test preference'}}}}
            (root/'generation-policy.json').write_text(json.dumps(policy))
            s=ModelingService(workspace=root)
            def snapshot():return {str(p.relative_to(s.store.root)):(digest(p.read_bytes()),p.stat().st_mtime_ns) for p in s.store.root.rglob('*') if p.is_file()}
            before=snapshot()
            direct=s.operation_context('generation',{'use_case':'exact_view_enlarged_local_detail'})
            route=s.select_generation_route('exact_view_enlarged_local_detail')
            transport=s.execute('operation_context',{'stage':'recovery'})
            self.assertEqual(snapshot(),before)
            self.assertEqual(direct['retention']['status'],'not retained')
            self.assertNotIn('context_record',direct);self.assertNotIn('operation_record',transport)
            self.assertFalse(route['dispatch_authorized'])
            captured=s.capture_operation_context('generation',{'use_case':'exact_view_enlarged_local_detail'})
            self.assertIn('operation_handle',captured);self.assertIn('operation_fact',captured)
            self.assertTrue(s.store.get(captured['context_record'])['pinned_authority'])
            self.assertNotEqual(snapshot(),before)

    def test_late_decisive_passage_is_not_heading_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p=root/'lessons.md'
            p.write_text('# Guide\n\n' + ('Unrelated introductory context. '*200) + '\n\n'
                         'Decisive: reconstructed connected skin is missing; reject lash-only relief.\n',encoding='utf-8')
            result=retrieve(root,Store(root/'store'),'connected skin missing',3,
                            sources=[{'path':'lessons.md','role':'experience'}])
            self.assertIn('Decisive',result['matches'][0]['excerpt'])
            self.assertEqual(result['matches'][0]['line_start'],5)
            self.assertEqual(result['matches'][0]['sha256'],__import__('hashlib').sha256(p.read_bytes()).hexdigest())

    def test_authority_and_counterexample_both_survive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'now.md').write_text('Guide default P2 applies now.')
            (root/'old.md').write_text('Guide '+('HD '*100)+' was rejected.')
            result=retrieve(root,Store(root/'store'),'guide HD',1,sources=[
                {'path':'now.md','role':'authority','priority':0}, {'path':'old.md','role':'experience'}])
            self.assertIn('P2',result['authority'][0]['excerpt'])
            self.assertIn('rejected',result['matches'][0]['excerpt'])
            self.assertEqual(applicability({'topology':'old'},{'topology':'new'})['status'],'inapplicable')

    def test_preference_is_separate_from_authorization(self):
        with tempfile.TemporaryDirectory() as d:
            from .init_workspace import initialize
            root=Path(d)/'project'; initialize(root,'Synthetic')
            policy={'tripo':{'mode':'Smart Mesh','model':'P2.0','selection_preferences':{'exact_view_enlarged_local_detail':{'mode':'HD Model','model':'v3.1','status':'test preference'}}}}
            (root/'generation-policy.json').write_text(json.dumps(policy))
            s=ModelingService(workspace=root)
            self.assertEqual(s.select_generation_route()['model'],'P2.0')
            choice=s.select_generation_route('exact_view_enlarged_local_detail')
            self.assertEqual(choice['model'],'v3.1'); self.assertFalse(choice['dispatch_authorized'])
            with self.assertRaisesRegex(ValueError,'Smart Mesh / P2.0'):
                s.references._mesh_policy('mesh','Tripo Studio',{'mode':'HD Model','model':'v3.1',
                    'use_case':'exact_view_enlarged_local_detail'})

    def test_runtime_source_is_observed_and_native_not_contacted(self):
        with tempfile.TemporaryDirectory() as d:
            from .init_workspace import initialize
            root=Path(d)/'project'; initialize(root,'Synthetic')
            policy={'tripo':{'mode':'Smart Mesh','model':'P2.0','selection_preferences':{'exact_view_enlarged_local_detail':{'mode':'HD Model','model':'v3.1','status':'test preference'}}}}
            (root/'generation-policy.json').write_text(json.dumps(policy))
            s=ModelingService(workspace=root)
            r=s.runtime_status()
            self.assertTrue(r['source_matches_loaded'])
            self.assertIn('operation_context',r['signatures'])
            self.assertIn('not contacted',r['native'])


if __name__=='__main__': unittest.main()

class EpisodeTests(unittest.TestCase):
    from .test_workflow import WorkflowTests as _Fixture
    setUp=_Fixture.setUp
    tearDown=_Fixture.tearDown

    def episode(self):
        q=self.s.open_question('Connected form',self.state,'eye','Restore coherent support')['question']
        return self.s.open_episode(q,'native owner','eye only, no paid work',
            {'stage':'guide_qualification','uncertainties':['Unknown hidden depth'],
             'requirements':[{'id':'rim','category':'provisional_constraint','text':'Fixed rim for this trial only',
                              'source':'trial hypothesis','scope':'this trial'}],
             'links':[{'kind':'file','path':str(self.image),'role':'source evidence'}]},'episode')

    def test_public_python_records_inputs_and_survives_index_failure(self):
        from unittest.mock import patch
        from .store import Store
        from .journal import calls
        original=Store.put
        def flaky(store,kind,payload):
            if kind=='operation_fact': raise OSError('index unavailable')
            return original(store,kind,payload)
        with patch.object(Store,'put',flaky):
            result=self.s.open_question('Saved question',self.state,'eye','Intent survives')
        self.assertEqual(result['retention_status'],'needs reconciliation')
        self.assertTrue(Path(result['operation_result_path']).is_file())
        old=self.s.store.current()
        recovered=self.s.reconcile_operation(result['operation_handle'])
        self.assertEqual(recovered['outcome']['result']['question'],old)
        self.assertEqual(self.s.store.current(),old)
        fact=self.s.store.get(recovered['operation_fact'])
        self.assertEqual(fact['intent']['arguments']['title'],'Saved question')
        self.assertEqual(fact['intent']['runtime'],self.s.runtime_status()['loaded']['revision'])

    def test_episode_missing_judgment_recovery_and_cas(self):
        from .ledger import Conflict
        e=self.episode()
        result=self.s.run_episode_operation(e['episode'],'query_geometry',{
            'state':self.state,'object_name':'Face','query':'bounds','parameters':{}})
        self.assertIn('operation_fact',result)
        view=self.s.decision_workspace(e['episode'])
        self.assertEqual(view['judgment_status'],'pending')
        self.assertEqual(view['history']['total_operations'],1)
        self.assertEqual(view['history']['omitted_completed_operations'],1)
        self.assertEqual(self.s.inspect_operations(e['episode'],view='index')['items'][0]['value']['handle'],result['operation_handle'])
        self.assertEqual(view['context']['requirements'][0]['category'],'provisional_constraint')
        r=self.s.revise_episode(e['episode'],e['revision'],{'uncertainties':['Need source judgment']})
        with self.assertRaises(Conflict): self.s.revise_episode(e['episode'],e['revision'],{'stage':'review'})
        self.assertEqual(len(self.s.decision_workspace(e['episode'])['context']['links']),1)
        pending=self.s.reconcile_episode(e['episode'],r['revision'])
        self.assertEqual(pending['judgment_status'],'pending')
        with self.assertRaises(Conflict):self.s.reconcile_episode(e['episode'],pending['revision'],close=True)
        done=self.s.reconcile_episode(e['episode'],pending['revision'],
            character={'status':'unresolved','reason':'Appearance not judged'},
            method={'status':'rejected','reason':'Local metric did not improve connected form'},
            evidence=[{'kind':'file','path':str(self.image),'role':'comparison'}],
            applicability='This fixture only',unresolved=['Appearance'],close=True)
        self.assertEqual(done['status'],'closed')

    def test_read_transports_do_not_write_or_advertise_missing_operation(self):
        self.episode()
        before=sorted(str(p) for p in self.s.store.root.rglob('*'))
        for op,args in [('inspect_situation',{}),('decision_workspace',{}),('retrieve_experience',{'query':'L62 local bend'})]:
            value=self.s.execute(op,args)
            self.assertNotEqual(value['status'],'failed')
        after=sorted(str(p) for p in self.s.store.root.rglob('*'))
        self.assertEqual(before,after)
        self.assertTrue(set(self.s.inspect_situation()['next'])<=set(self.s.operations()))

    def test_summary_replacement_is_typed_refusal_and_add_remove_preserve_exact_links(self):
        from .ledger import PreconditionRefusal, Conflict
        from .episodes import link_id
        e=self.episode();full=self.s.decision_workspace(e['episode'],detail='links')
        original=full['context']['links'];summary=self.s.decision_workspace(e['episode'])
        new={'kind':'record','id':self.state,'role':'new recorded geometry'}
        refused=self.s.execute('revise_episode',{'episode':e['episode'],'expected_revision':e['revision'],
            'patch':{'links':summary['context']['links']+[new]}})
        self.assertEqual(refused['stage'],'revise_episode.link_validation')
        self.assertEqual(refused['effect_status'],'refused before mutation dispatch')
        self.assertFalse(refused['details']['episode_changed'])
        self.assertIn('add_links',refused['recovery'])
        self.assertEqual(self.s.ledger.read(e['episode'])['revision'],e['revision'])
        with self.assertRaises(PreconditionRefusal):
            self.s.revise_episode(e['episode'],e['revision'],{'links':[{'kind':'file','role':'incomplete'}]})
        # Appending and complete round trips use original immutable bytes even
        # when their original source is no longer available at that location.
        self.image.unlink()
        r=self.s.revise_episode(e['episode'],e['revision'],{'add_links':[new]})
        links=self.s.decision_workspace(e['episode'],detail='full')['context']['links']
        self.assertEqual(links[:-1],original)
        with self.assertRaises(Conflict):self.s.revise_episode(e['episode'],e['revision'],{'remove_links':[link_id(original[0])]})
        r=self.s.revise_episode(e['episode'],r['revision'],{'links':links})
        r=self.s.revise_episode(e['episode'],r['revision'],{'remove_links':[link_id(new)]})
        self.assertEqual(self.s.decision_workspace(e['episode'],detail='links')['context']['links'],original)
        with self.assertRaises(PreconditionRefusal):self.s.revise_episode(e['episode'],r['revision'],{'remove_links':['unknown']})

    def test_concurrent_additions_cannot_overwrite_the_winning_revision(self):
        import threading
        from unittest.mock import patch
        from . import episodes
        from .ledger import Conflict
        e=self.episode();before=self.s.decision_workspace(e['episode'],detail='links')['context']['links']
        barrier=threading.Barrier(2);pin=episodes.pin_link;results=[];errors=[]
        def wait_pin(service,link):
            value=pin(service,link);barrier.wait(5);return value
        def add(role):
            try:results.append(self.s.revise_episode(e['episode'],e['revision'],{'add_links':[{'kind':'record','id':self.state,'role':role}]}))
            except Exception as error:errors.append(error)
        with patch.object(episodes,'pin_link',wait_pin):
            workers=[threading.Thread(target=add,args=(str(i),)) for i in range(2)]
            for worker in workers:worker.start()
            for worker in workers:worker.join(6)
        self.assertEqual(len(results),1);self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],Conflict)
        after=self.s.decision_workspace(e['episode'],detail='links')['context']['links']
        self.assertEqual(after[:-1],before);self.assertEqual(len(after),len(before)+1)

    def test_historical_anatomy_preparation_preserves_source_identity_and_refuses_changed_bytes(self):
        from .store import digest
        e=self.episode();source=self.root/'historical-anatomy.md';source.write_text('Reviewed anatomy at qualification')
        expected=digest(source.read_bytes())
        link={'kind':'file','path':str(source),'sha256':expected,'role':'historical anatomy at qualification'}
        r=self.s.revise_episode(e['episode'],e['revision'],{'add_links':[link]})
        pinned=self.s.decision_workspace(e['episode'],detail='links')['context']['links'][-1]
        location=Path(self.s.runtime_status()['store'])/pinned['asset']['path']
        self.assertEqual(pinned['asset']['source'],str(source.resolve()))
        self.assertEqual(digest(location.read_bytes()),expected)
        source.write_text('Later independent experiment entry')
        self.assertEqual(digest(location.read_bytes()),expected)
        with self.assertRaisesRegex(ValueError,'changed|differs'):self.s.revise_episode(e['episode'],r['revision'],{'add_links':[link]})
        self.assertEqual(self.s.ledger.read(e['episode'])['revision'],r['revision'])
        source.unlink()
        location.write_bytes(b'corrupt isolated stored evidence')
        with self.assertRaisesRegex(ValueError,'integrity'):self.s.revise_episode(e['episode'],r['revision'],{'links':[pinned]})

    def test_full_link_round_trip_never_probes_originals_but_other_input_capture_stays_strict(self):
        from unittest.mock import patch
        from .journal import capture_inputs
        from .store import digest
        e=self.episode();view=self.s.decision_workspace(e['episode'],detail='links');links=view['context']['links']
        original=links[0]['path'];probes=[];is_file=Path.is_file
        def guarded(path):
            if str(path)==original:probes.append(str(path));raise PermissionError('Original project access forbidden')
            return is_file(path)
        # Both original-path existence probes and reads must be unnecessary.
        with patch.object(Path,'is_file',guarded):
            r=self.s.revise_episode(e['episode'],e['revision'],{'links':links})
        self.assertEqual(probes,[])
        fact=self.s.store.get(r['operation_fact'])
        self.assertEqual(fact['intent']['arguments']['patch']['links'],links)
        self.assertEqual(fact['intent']['inputs'][0]['asset'],links[0]['asset'])
        forged=json.loads(json.dumps(links));forged[0]['asset']['source']='forged original source identity'
        with self.assertRaisesRegex(ValueError,'canonical retained link'):
            self.s.revise_episode(e['episode'],r['revision'],{'links':forged})
        self.image.write_bytes(b'Later source bytes')
        r=self.s.revise_episode(e['episode'],r['revision'],{'links':links})
        self.assertEqual(self.s.decision_workspace(e['episode'],detail='full')['context']['links'],links)
        with self.assertRaisesRegex(ValueError,'Input changed'):
            capture_inputs(self.s.store,{'path':original,'sha256':links[0]['sha256']},'constraints')
        location=Path(self.s.runtime_status()['store'])/links[0]['asset']['path'];location.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'integrity'):self.s.revise_episode(e['episode'],r['revision'],{'links':links})

    def test_semantic_reverse_and_wrong_binding_not_fixed_by_coordinates(self):
        from .store import canonical,digest
        dep=self.s.record_dependencies(self.state)['dependencies']
        ev={'kind':'file','path':str(self.image),'role':'fixture interpretation'}
        g=self.s.register_semantic_graph([
            {'id':'intent','role':'close lid','interpretation':'instruction'},
            {'id':'control','role':'closure control','interpretation':'construction_evidence'},
            {'id':'surface','role':'upper skin','interpretation':'hypothesis','selector':{'object':'Face','vertex_indices':[0,1]}}],
            [{'from':'intent','to':'control','relation':'implemented by','interpretation':'hypothesis','evidence':[ev]},
             {'from':'control','to':'surface','relation':'deforms','interpretation':'hypothesis','evidence':[ev]}],
            {'included':['fixture lid'],'missing':['lash neighbor unbound']},dep)['graph']
        v=self.s.validate_semantics(g,dep,'invalid','Source cells misidentified',[ev])['validation']
        r=self.s.semantic_impact(g,selector={'object':'Face','vertex_index':0},direction='reverse',dependencies=dep,validation=v)
        self.assertEqual(r['semantic_status'],'invalid');self.assertEqual(r['reached'],3)
        self.assertEqual(r['dependency_status'],'matching recorded dependencies')
        self.assertEqual(r['coverage']['missing'],['lash neighbor unbound'])

    def test_proposal_script_pinned_when_original_changes(self):
        from .store import digest
        script=self.root/'worker.py';script.write_text('original exact script')
        q=self.s.open_question('Test',self.state,'eye','Intent')['question']
        e=self.s.begin_experiment(q,'Test','Native check',[],'zero credits','trial')
        proposal={'kind':'script','label':'test','feature':'eyes','mismatch':'fixture','mechanism':'fixture',
            'constraints':[{'path':str(self.image),'sha256':digest(self.image.read_bytes())}],
            'allowed_objects':['Face'],'script':{'path':str(script),'sha256':digest(script.read_bytes())}}
        r=self.s.propose_change(e['handle'],e['revision'],proposal,[str(self.image)])
        script.write_text('changed later')
        pinned=self.s.ledger.read(e['handle'])['data']['proposal']['script']
        self.assertEqual(Path(pinned['path']).read_text(),'original exact script')
        self.assertEqual(self.s.store.get(r['operation_fact'])['intent']['arguments']['proposal']['script']['path'],str(script))

class PackagedMethodTests(unittest.TestCase):
    def test_packaged_procedures_resolve_bound_method_without_private_history(self):
        from .init_workspace import initialize
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'project'; initialize(root,'Synthetic')
            s=ModelingService(workspace=root)
            result=s.operation_context('guide_qualification',{'observed_failure':'missing connected skin'})
            self.assertTrue(result['procedures'])
            self.assertTrue(all(row['source_passages'] for row in result['procedures']))
            self.assertEqual(result['coverage']['missing_citations'],[])
