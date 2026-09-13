"""Focused transport and recovery tests; temporary retained data, no native/provider calls."""
import json
import unittest
from unittest.mock import patch
import anyio
from .bounded_reads import json_chars
from .store import canonical, digest, atomic_write
from .mcp_server import ModelingMCP
from .test_decisions import EpisodeTests


def descriptors(value):
    if isinstance(value,dict):
        if set(('operation','arguments'))<=value.keys(): yield value
        for child in value.values(): yield from descriptors(child)
    elif isinstance(value,list):
        for child in value: yield from descriptors(child)


def fields(page):
    return {item['key']:item['value'] for item in page['items']}


class BoundedReads(unittest.TestCase):
    setUp=EpisodeTests.setUp
    tearDown=EpisodeTests.tearDown
    episode=EpisodeTests.episode

    def snapshot(self):
        return {str(p.relative_to(self.s.store.root)):(digest(p.read_bytes()),p.stat().st_mtime_ns)
                for p in self.s.store.root.rglob('*') if p.is_file()}

    def call(self, descriptor):
        result=self.s.execute(descriptor['operation'],descriptor['arguments'])
        self.assertNotEqual(result['status'],'failed',result)
        self.assertLessEqual(json_chars(result),descriptor['arguments'].get('max_chars',8000))
        return result

    def test_huge_catalog_immutable_fields_and_string_progress_no_writes(self):
        e=self.episode();item=self.s.ledger.read(e['episode'])
        links=[{'kind':'record','id':self.state,'role':'retained '+str(i)} for i in range(1000)]
        r=self.s.revise_episode(e['episode'],e['revision'],{'links':links})
        before=self.snapshot();summary=self.s.decision_workspace(e['episode'])
        self.assertLessEqual(json_chars(summary),24000)
        self.assertEqual(summary['sections']['context.links']['total'],1000)
        self.assertEqual(summary['sections']['context.links']['shown'],0)
        self.assertFalse(summary['incomplete_decision_coverage'])
        read=self.call(summary['context']['expand_links'])
        self.assertLess(read['shown'],1000);self.assertEqual(read['total'],1000)
        self.assertEqual(read['source_revision'],r['revision'])
        self.assertEqual(read['items'][0]['value']['role'],'retained 0')
        self.assertEqual(self.call(read['next'])['offset'],read['shown'])
        self.assertEqual(self.snapshot(),before)
        self.assertEqual(self.s.read_record(r['revision'])['data']['context']['links'],links)
        denied=self.s.execute('revise_episode',{'episode':e['episode'],'expected_revision':r['revision'],
            'patch':{'links':summary['context']['links']}})
        self.assertEqual(denied['stage'],'revise_episode.link_validation')
        self.assertEqual(self.s.ledger.read(e['episode'])['revision'],r['revision'])
        updated=self.s.revise_episode(e['episode'],r['revision'],{'add_links':[{'kind':'record','id':self.state,'role':'new'}]})
        self.assertEqual(len(self.s.read_record(updated['revision'])['data']['context']['links']),1001)
        self.assertEqual(self.call(summary['context']['expand_links'])['total'],1000)
        key=self.s.store.put('fixture',{'large':'\U0001f642"\\\n'*8000,'tail':True})
        root=self.s.read_record(key,path=[],max_chars=3000)
        nested=next(i['value']['expand'] for i in root['items'] if i['key']=='large')
        window=self.call(nested);self.assertGreater(window['shown'],0);self.assertIn('next',window)
        self.assertEqual(self.call(window['next'])['offset'],window['shown'])

    def test_exact_pending_unindexed_calls_and_changed_index(self):
        from .leases import process_identity
        e=self.episode();handle='a'*32;folder=self.s.store.root/'calls'/handle
        intent={'handle':handle,'episode':e['episode'],'operation':'native_apply_proposal','utc':'2026-09-13T00:00:00Z',
                'process':process_identity(),'arguments':{'owner':'sole owner'},'recovery':'Inspect actual receipt; no replay'}
        atomic_write(folder/'intent.json',canonical(intent))
        before=self.snapshot()
        with patch('modeling_system.operation_reads.calls',side_effect=AssertionError('exact read scanned all calls')):
            header=fields(self.s.inspect_operations(handle=handle,view='header'))
            self.assertEqual(header['status'],'active/in-flight');self.assertFalse(header['result_available'])
            result=self.call(header['sections']['result'])
            self.assertFalse(fields(result)['available'])
        self.assertEqual(self.snapshot(),before)
        first=self.s.inspect_operations(episode=e['episode'],view='index',limit=1)
        self.assertEqual(first['counts']['active'],1);self.assertEqual(first['counts']['unresolved'],1)
        atomic_write(folder/'result.json',canonical({'status':'returned','result':{'huge_trial':'x'*50000}}))
        stale=self.s.inspect_operations(episode=e['episode'],view='index',expected_view=first['source_revision'],limit=1)
        self.assertEqual(stale['status'],'conflicting');self.assertNotIn('items',stale)
        self.assertEqual(fields(self.s.inspect_operations(handle=handle,view='header'))['retention'],'needs index reconciliation')
        self.assertEqual(self.call(stale['refresh'])['counts']['active'],0)
        bounded=self.s.inspect_operations(handle=handle,view='result',path=['result'],max_chars=3000)
        self.assertTrue(bounded['items'][0]['value']['deferred_value'])
        self.assertGreater(self.call(bounded['items'][0]['value']['expand'])['shown'],0)

    def test_overflow_preserves_exact_requirements_unknown_counts_and_drilldown(self):
        e=self.episode();required='Indispensable current requirement. '*2000
        self.s.revise_episode(e['episode'],e['revision'],{'requirements':[dict(id='large',category='obligation',text=required,source='user',scope='all')]})
        for n in range(70):
            handle=f'{n:032x}';folder=self.s.store.root/'calls'/handle
            atomic_write(folder/'intent.json',canonical({'handle':handle,'episode':e['episode'],'operation':'fixture_unknown',
                'utc':'2026-09-13T00:00:00Z','arguments':{},'recovery':'Read actual effect'}))
        before=self.snapshot();value=self.s.execute('decision_workspace',{'episode':e['episode']})
        self.assertLessEqual(json_chars(value),24000);self.assertTrue(value['incomplete_decision_coverage'])
        self.assertEqual(value['sections']['operations']['total'],70)
        self.assertEqual(value['sections']['operations']['unknown'],70)
        self.assertEqual(value['sections']['operations']['unresolved'],70)
        self.assertEqual(value['sections']['operations']['shown'],0)
        self.assertEqual(self.s.read_record(value['revision'])['data']['context']['requirements'][0]['text'],required)
        for read in value['reads'].values(): self.call(read)
        self.assertEqual(self.snapshot(),before)

    def test_public_schema_descriptors_dual_transport_and_errors(self):
        e=self.episode();server=ModelingMCP(self.s)
        async def check():
            schemas={t.name:t.input_schema for t in await server.list_tools()}
            response=await server.call_tool('decision_workspace',{'episode':e['episode']})
            structured=response.structured_content
            fallback=json.loads(next(c.text for c in response.content if c.type=='text'))
            self.assertEqual(structured,fallback)
            self.assertLessEqual(json_chars(structured),24000)
            for descriptor in descriptors(structured):
                name=descriptor['operation'];args=descriptor['arguments']
                self.assertTrue(set(args)<=set(schemas[name]['properties']))
                self.assertTrue(set(schemas[name].get('required',[]))<=set(args))
                expanded=await server.call_tool(name,args)
                self.assertFalse(expanded.is_error,expanded)
                self.assertLessEqual(json_chars(expanded.structured_content),args.get('max_chars',8000))
                self.assertEqual(expanded.structured_content,json.loads(next(c.text for c in expanded.content if c.type=='text')))
        before=self.snapshot();anyio.run(check);self.assertEqual(self.snapshot(),before)
        invalid=[({'record':self.state,'path':['missing']},'KeyError'),
                 ({'record':self.state,'path':['objects',999]},'IndexError'),
                 ({'record':self.state,'path':[],'limit':999},'ValueError'),
                 ({'record':'../secret','path':[]},'ValueError'),
                 ({'record':'f'*64,'path':[]},'FileNotFoundError')]
        for args,error in invalid:
            with self.subTest(error=error):
                response=self.s.execute('read_record',args);self.assertEqual(response['error_type'],error)
        for handle in ('../secret','A'*32,'a'*33):
            self.assertEqual(self.s.execute('inspect_operations',{'handle':handle,'view':'intent'})['error_type'],'ValueError')

    def test_episode_index_growth_and_lease_without_call_intent(self):
        from .leases import process_identity
        e=self.episode();index=self.s.decision_workspace()
        self.assertEqual(index['total'],1);self.assertLessEqual(json_chars(index),8000)
        self.s.open_episode(self.s.ledger.read(e['episode'])['intent']['question'],'second owner','separate scope',{},'second')
        stale=self.s.decision_workspace(detail='section',section='episodes',expected_view=index['source_revision'])
        self.assertEqual(stale['error'],'changed_read_view');self.assertEqual(self.call(stale['refresh'])['total'],2)
        lease={'id':'b'*32,'operation_handle':'c'*32,'episode':e['episode'],'episode_revision':e['revision'],
               'operation':'fixture_before_intent','status':'active','process':process_identity()}
        atomic_write(self.s.store.root/'episode-leases'/e['episode']/(lease['id']+'.json'),canonical(lease))
        before=self.snapshot();summary=self.s.decision_workspace(e['episode'])
        self.assertEqual(summary['sections']['operation_leases']['active'],1)
        self.assertEqual(summary['sections']['operations']['total'],0)
        exact=fields(self.call(summary['operation_leases'][0]['expand']))
        self.assertEqual(exact['id'],lease['id']);self.assertEqual(exact['observed_status'],'active')
        self.assertEqual(self.snapshot(),before)


if __name__=='__main__':unittest.main()
