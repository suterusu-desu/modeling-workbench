"""Reproduce the five independent recovery review failures with inert native stubs."""
import copy
import json
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from . import test_decisions as fixtures
from .native_bridge import NativeBridgeError
from .ledger import Conflict

class RecoveryRaceTests(unittest.TestCase):
    setUp=fixtures.EpisodeTests.setUp
    tearDown=fixtures.EpisodeTests.tearDown

    def ep(self):return fixtures.EpisodeTests.episode(self)

    def close(self,e):
        return self.s.reconcile_episode(e['episode'],e['revision'],
            character={'status':'unresolved','reason':'no appearance judgment'},
            method={'status':'unresolved','reason':'no causal judgment','integration':{'disposition':'not_generalizable','reason':'Infrastructure fixture makes no modeling-method claim'}},
            evidence=[{'kind':'file','path':str(self.image),'role':'actual fixture evidence'}],
            applicability='fixture only',close=True)

    def test_raw_native_return_survives_internal_ledger_failure(self):
        e=self.ep();sid=self.sid
        class Native:
            def call(self,op,*args,**kwargs):
                if op=='inspect_live':return dict(expected_state='current',geometry_state_id=sid)
                return {'receipt_path':'isolated-stub-receipt','unique_return':'RESULT-123','status':'completed'}
        self.s.native=Native()
        q=self.s.store.get(self.s.ledger.read(e['episode'])['intent']['question'])
        trial=self.s.begin_experiment(self.s.ledger.read(e['episode'])['intent']['question'],'Hypothesis','Actual comparison',[],'zero','trial')
        from .store import digest
        proposal={'kind':'script','label':'test','feature':'eyes','mismatch':'fixture','mechanism':'fixture',
           'constraints':[{'path':str(self.image),'sha256':digest(self.image.read_bytes())}],
           'allowed_objects':['Face'],'script':{'path':str(self.image),'sha256':digest(self.image.read_bytes())}}
        trial=self.s.propose_change(trial['handle'],trial['revision'],proposal,[str(self.image)])
        update=self.s.ledger.update
        def fail(handle,revision,status,*a,**kw):
            if status=='applied':raise OSError('ledger after native return failed')
            return update(handle,revision,status,*a,**kw)
        with patch.object(self.s.ledger,'update',fail):
            result=self.s.run_episode_operation(e['episode'],'apply_trial',dict(experiment=trial['handle'],expected_revision=trial['revision'],expected_state='current',owner='owner'))
        self.assertEqual(result['status'],'failed');self.assertIn('operation_handle',result)
        row=self.s.inspect_operations(e['episode'])['calls'][0]
        returned=[json.loads(Path(x['result_path']).read_bytes()) for x in row['effects']]
        self.assertTrue(any(r.get('result',{}).get('unique_return')=='RESULT-123' for r in returned))
        self.assertEqual(self.s.ledger.read(trial['handle'])['status'],'applying')
        recovered=self.s.reconcile_operation(result['operation_handle'])
        self.assertEqual(recovered['outcome']['status'],'raised')
        with self.assertRaises(Conflict):self.close(e)

    def test_indexed_timeout_is_still_unknown_and_cannot_close(self):
        class Native:
            def call(self,*a,**k):raise NativeBridgeError('timeout','unknown effect','inspect real receipt')
        self.s.native=Native();e=self.ep()
        result=self.s.run_episode_operation(e['episode'],'native_save_checkpoint',{'expected_state':'token','owner':'owner','label':'test'})
        repaired=self.s.reconcile_operation(result['operation_handle'])
        self.assertEqual(repaired['effect_status'],'unknown')
        with self.assertRaises(Conflict):self.close(e)
        resolved=self.s.reconcile_operation(result['operation_handle'],{'effect_status':'confirmed_not_applied','basis':'isolated stub executed no effect'},[str(self.image)])
        self.assertEqual(resolved['effect_status'],'reconciled with evidence')
        self.assertEqual(self.close(e)['status'],'closed')

    def test_baseline_refusal_records_typed_stage_and_no_dispatch(self):
        invoked=[]
        class Native:
            def call(self,operation,*args,**kwargs):
                invoked.append(operation)
                if operation!='inspect_live':raise AssertionError('Mutation must not dispatch')
                return dict(expected_state='new-review-token',geometry_state_id=None)
        self.s.native=Native();e=self.ep()
        trial=self.s.begin_experiment(self.s.ledger.read(e['episode'])['intent']['question'],'H','comparison',[],'zero','refusal')
        r=self.s.run_episode_operation(e['episode'],'apply_trial',dict(experiment=trial['handle'],expected_revision=trial['revision'],expected_state='old-token',owner='owner'))
        self.assertEqual(invoked,['inspect_live'])
        self.assertEqual(r['effect_status'],'refused before mutation dispatch')
        self.assertEqual(r['stage'],'apply_trial.baseline_precondition')
        self.assertFalse(r['details']['mutation_dispatched'])
        self.assertEqual(self.s.reconcile_operation(r['operation_handle'])['effect_status'],'refused before mutation dispatch')
        self.assertEqual(self.close(e)['status'],'closed')

    def test_active_owner_is_not_reconcilable_and_closure_start_race_is_atomic(self):
        from . import episodes
        entered=threading.Event();release=threading.Event();closure_read=threading.Event();continue_close=threading.Event()
        class Native:
            def call(self,*a,**k):entered.set();release.wait(5);return {'status':'completed','value':'native result'}
        self.s.native=Native();e=self.ep();original=episodes.workspace;errors=[];results=[]
        def paused(*a,**k):
            result=original(*a,**k)
            if threading.current_thread().name=='closer':closure_read.set();continue_close.wait(5)
            return result
        def close_thread():
            try:self.close(e)
            except Exception as error:errors.append(error)
        with patch.object(episodes,'workspace',paused):
            closer=threading.Thread(target=close_thread,name='closer');closer.start();self.assertTrue(closure_read.wait(3))
            caller=threading.Thread(target=lambda:results.append(self.s.run_episode_operation(e['episode'],'native_save_checkpoint',{'expected_state':'token','owner':'owner','label':'test'})))
            caller.start();self.assertTrue(entered.wait(3))
            rows=self.s.inspect_operations(e['episode'])['calls']
            active=next(r for r in rows if r['operation']=='native_save_checkpoint')
            self.assertEqual(active['status'],'active/in-flight')
            with self.assertRaisesRegex(ValueError,'still active'):self.s.reconcile_operation(active['handle'])
            continue_close.set();closer.join(3)
            self.assertTrue(errors);self.assertIsInstance(errors[0],Conflict)
            self.assertEqual(self.s.ledger.read(e['episode'])['status'],'active')
            release.set();caller.join(3)
        self.assertEqual(results[0]['status'],'completed')

    def test_partial_judgment_preserves_previously_supplied_component(self):
        e=self.ep()
        first=self.s.reconcile_episode(e['episode'],e['revision'],method={'status':'supported','reason':'recovered result'},
            evidence=[{'kind':'file','path':str(self.image),'role':'comparison'}],applicability='fixture')
        second=self.s.reconcile_episode(e['episode'],first['revision'],character={'status':'unresolved','reason':'appearance pending'})
        judgment=self.s.store.get(second['judgment'])
        self.assertEqual(judgment['method']['status'],'supported');self.assertEqual(second['judgment_status'],'recorded')
        cleared=self.s.reconcile_episode(e['episode'],second['revision'],clear_components=['method'])
        self.assertIsNone(self.s.store.get(cleared['judgment'])['method'])

    def test_original_arguments_and_caller_graph_not_mutated(self):
        dep=self.s.record_dependencies(self.state)['dependencies']
        nodes=[{'id':'n','role':'fixture','interpretation':'fact','evidence':[{'kind':'file','path':str(self.image),'role':'original'}]}]
        before=copy.deepcopy(nodes)
        result=self.s.register_semantic_graph(nodes,[],{'included':['n'],'missing':[]},dep)
        self.assertEqual(nodes,before)
        fact=self.s.store.get(result['operation_fact'])
        intent=json.loads((Path(result['operation_result_path']).parent/'intent.json').read_bytes())
        self.assertEqual(fact['intent'],intent)
        self.assertNotIn('asset',fact['intent']['arguments']['nodes'][0]['evidence'][0])

if __name__=='__main__':unittest.main()

class LeaseCleanupTests(unittest.TestCase):
    setUp=RecoveryRaceTests.setUp
    tearDown=RecoveryRaceTests.tearDown
    ep=RecoveryRaceTests.ep
    def test_failed_final_lease_write_does_not_leak_context_or_lose_result(self):
        from .journal import EPISODE,ACTIVE_CALL
        from .leases import LEASE
        class Native:
            def call(self,*a,**k):return {'status':'completed','unique_return':'FINISHED-456'}
        self.s.native=Native();e=self.ep()
        with patch('modeling_system.service.finish',side_effect=OSError('lease finalization unavailable')):
            result=self.s.run_episode_operation(e['episode'],'native_save_checkpoint',{'expected_state':'token','owner':'owner','label':'test'})
        self.assertEqual(result['status'],'completed');self.assertEqual(result['unique_return'],'FINISHED-456')
        self.assertEqual(result['lease_retention_status'],'needs finalization repair')
        self.assertIsNone(EPISODE.get());self.assertIsNone(LEASE.get());self.assertIsNone(ACTIVE_CALL.get())
        row=self.s.inspect_operations(e['episode'])['calls'][0]
        self.assertNotEqual(row['status'],'active/in-flight')
        self.s.reconcile_operation(result['operation_handle'])
        from .leases import list_leases
        self.assertEqual(list_leases(self.s,e['episode'])[0]['status'],'finished')
        separate=self.s.open_question('Unrelated later call',self.state,'eye','Must not inherit episode')
        fact=self.s.store.get(separate['operation_fact'])
        self.assertIsNone(fact['intent']['episode'])

    def test_exact_final_atomic_write_failure_and_lock_conflict_unwind(self):
        from . import leases
        from .journal import EPISODE,ACTIVE_CALL
        from .leases import LEASE
        for mode in ('write','lock'):
            e=self.ep() if mode=='write' else self.s.open_episode(self.s.store.current(),'owner','fixture scope',{'stage':'review'},'lock-episode')
            original=leases.atomic_write
            def fail(path,data):
                if json.loads(data).get('status')=='finished':
                    if mode=='write':raise OSError('finished lease write failed')
                    raise Conflict('final lease lock busy')
                return original(path,data)
            with patch.object(leases,'atomic_write',fail):
                result=self.s.run_episode_operation(e['episode'],'query_geometry',{'state':self.state,'object_name':'Face','query':'bounds','parameters':{}})
            self.assertEqual(result['status'],'completed');self.assertIn('operation_handle',result)
            self.assertEqual(result['lease_retention_status'],'needs finalization repair')
            self.assertIsNone(EPISODE.get());self.assertIsNone(LEASE.get());self.assertIsNone(ACTIVE_CALL.get())
            self.s.reconcile_operation(result['operation_handle'])
            self.assertTrue(all(r['status']=='finished' for r in leases.list_leases(self.s,e['episode'])))
            later=self.s.open_question('Later direct '+mode,self.state,'eye','No leaked binding')
            self.assertIsNone(self.s.store.get(later['operation_fact'])['intent']['episode'])


class SummaryScalingTests(unittest.TestCase):
    setUp=RecoveryRaceTests.setUp
    tearDown=RecoveryRaceTests.tearDown
    ep=RecoveryRaceTests.ep
    close=RecoveryRaceTests.close

    def test_actual_emitted_expansions_execute_read_only_for_active_and_unindexed_rows(self):
        import anyio
        from .mcp_server import ModelingMCP
        from .store import digest
        entered=threading.Event();release=threading.Event();e=self.ep();results=[]
        class Native:
            def call(self,*args,**kwargs):entered.set();release.wait(8);return {'status':'completed','fixture_only':True}
        self.s.native=Native();put=self.s.store.put
        def fail_index(kind,*args,**kwargs):
            if kind=='operation_fact':raise OSError('Isolated index failure')
            return put(kind,*args,**kwargs)
        with patch.object(self.s.store,'put',fail_index):
            unindexed=self.s.run_episode_operation(e['episode'],'query_geometry',dict(state=self.state,object_name='Face',query='bounds',parameters={}))
        worker=threading.Thread(target=lambda:results.append(self.s.run_episode_operation(e['episode'],'native_save_checkpoint',dict(expected_state='stub',owner='owner',label='pending'))))
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            view=self.s.decision_workspace(e['episode'])
            self.assertEqual(len(view['operations']),2)
            self.assertIn('active/in-flight',[row['retention'] for row in view['operations']])
            schemas={tool.name:tool.input_schema for tool in anyio.run(ModelingMCP(self.s).list_tools)}
            def snapshot():return {str(p):(digest(p.read_bytes()),p.stat().st_mtime_ns) for p in self.s.store.root.rglob('*') if p.is_file()}
            before=snapshot()
            for row in view['operations']:
                descriptor=row['expand'];operation=descriptor['operation'];arguments=descriptor['arguments']
                self.assertIn(operation,self.s.operations())
                self.assertLessEqual(set(arguments),set(schemas[operation]['properties']))
                self.assertLessEqual(set(schemas[operation].get('required',[])),set(arguments))
                expanded=self.s.execute(operation,arguments)
                from .test_bounded_reads import fields
                selected=fields(expanded)
                self.assertEqual(selected['handle'],row['handle'])
                self.assertEqual(selected['effects'],row['effects'])
                if row['handle']==unindexed['operation_handle']:
                    self.assertIsNone(selected['record']);self.assertTrue(selected['result_available'])
            self.assertEqual(snapshot(),before)
        finally:release.set();worker.join(5)
        self.assertEqual(len(results),1)

    def test_completed_history_is_bounded_while_old_unknown_and_guards_survive(self):
        from .store import canonical
        from . import episodes
        class Native:
            def call(self,operation,arguments,**kwargs):
                if arguments.get('label')=='uncertain':raise NativeBridgeError('timeout','after dispatch unknown','inspect receipt')
                return {'status':'completed','fixture_only':True}
        self.s.native=Native();e=self.ep()
        older=self.s.run_episode_operation(e['episode'],'native_save_checkpoint',dict(expected_state='stub',owner='owner',label='uncertain'))
        for index in range(24):
            self.s.run_episode_operation(e['episode'],'native_save_checkpoint',dict(expected_state='stub',owner='owner',label=str(index)))
        before=self.s.decision_workspace(e['episode'])
        for index in range(64):
            self.s.run_episode_operation(e['episode'],'native_save_checkpoint',dict(expected_state='stub',owner='owner',label=str(index+24)))
        after=self.s.decision_workspace(e['episode'])
        self.assertEqual(after['history']['total_operations'],89)
        self.assertEqual(after['sections']['operation_leases']['deferred'],88)
        self.assertEqual(len(after['operations']),1)
        self.assertEqual(len(after['operation_leases']),1)
        unknown=next(r for r in after['operations'] if r['handle']==older['operation_handle'])
        self.assertEqual(unknown['effect_status'],'unknown');self.assertEqual(unknown['owner'],'owner')
        self.assertIn('actual outcome evidence',unknown['next_action'])
        self.assertLess(abs(len(canonical(after))-len(canonical(before))),1000)
        self.assertEqual(after['operations'],sorted(after['operations'],key=episodes.recent_operation_key))
        import inspect
        descriptors=[r['expand'] for r in after['operations']]+[after['sections']['operations']['expand'],after['sections']['operation_leases']['expand'],after['context']['expand_links']]
        descriptors += [w['expand'] for w in after['workflows']]
        for descriptor in descriptors:
            inspect.signature(getattr(self.s,descriptor['operation'])).bind(**descriptor['arguments'])
        expansion=after['operations'][0]['expand']
        expanded=getattr(self.s,expansion['operation'])(**expansion['arguments'])
        self.assertEqual(next(i['value'] for i in expanded['items'] if i.get('key')=='handle'),older['operation_handle'])
        complete=self.s.decision_workspace(e['episode'],detail='links')
        self.assertEqual(len(complete['operations']),89);self.assertEqual(len(complete['operation_leases']),89)
        self.assertEqual(len(self.s.inspect_operations(e['episode'])['calls']),89)
        self.assertEqual(len(self.s.evidence_manifest(e['episode'])['pending_operations']),1)
        self.s.reconcile_operation(older['operation_handle'])
        # Even a deliberately empty UI projection cannot change closure guards.
        with patch('modeling_system.workspace_summary.summary',return_value={}):
            with self.assertRaises(Conflict):self.close(e)
