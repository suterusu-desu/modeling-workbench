from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .controller import fingerprint, read_json, write_json
from .judgments import prepare_judgments
from .provider_recovery import reconcile_completed_response, prepare_terminal_retry, terminal_retry_packet
from .typesafe_transport import MODEL, ROUTE, usage_cost


def completed_call(root, packet, *, accounted=True):
    ledger = Path(root) / 'ledger'; ledger.mkdir(exist_ok=True)
    call = ledger / 'call-1'; call.mkdir(exist_ok=True)
    source = Path(root) / 'source-packet.json'; write_json(source, packet)
    write_json(call / 'owner-packet.json', packet)
    wire = {'model': MODEL, 'state': packet['state'], 'questions': packet['questions']}
    answers = {}
    for key, q in packet['questions'].items():
        options = list(q['criteria'])
        answers[key] = {'type': 'choice', 'choice': options[0], 'confidence': 1.,
            'probabilities': {k: float(k == options[0]) for k in options}}
    response = {'model': MODEL, 'answers': answers, 'usage': {'input_tokens': 100, 'output_tokens': 10}}
    cost = usage_cost(response)
    write_json(call / 'decision-1.request.json', {'transport': ROUTE, 'request': wire,
        'local_binding': packet['local_binding'], 'wire_request_digest': fingerprint(wire)})
    write_json(call / 'decision-1.response.json', {'transport': ROUTE, 'http': 200,
        'response': response, 'wire_request_digest': fingerprint(wire), 'elapsed_ms': 12.})
    write_json(call / 'dispatch-count.json', {'transport': ROUTE, 'calls_attempted': 1})
    entry = {'output': str(call), 'packet_digest': fingerprint(packet), 'source_packet': str(source),
        'status': 'failed_or_uncertain' if accounted else 'reserved', 'reserved_usd': .001344,
        'provider_call_index': 1}
    if accounted: entry['estimated_cost_usd'] = cost
    write_json(ledger / 'budget.json', {'attempts': [entry], 'known_cost_usd': cost if accounted else 0.,
        'status': 'needs_reconciliation', 'reserved_usd': 0. if accounted else .001344,
        'provider_slots_reserved_or_attempted': 1})
    return ledger, call


class ProviderRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {'values': {'mesh': 'v1'}, 'active_operations': [],
            'owner': 'owner', 'authority_revision': 'a1'}
        self.packet = prepare_judgments({'goal': 'Repair supported surface'}, [{
            'id': 'operation', 'type': 'choice', 'instructions': 'Choose applicable work',
            'options': [{'id': 'first', 'description': 'Supported repair'}, {'id': 'second', 'description': 'Inspect'}]}],
            {'owner': 'owner', 'authority_revision': 'a1', 'dependencies': {'reads': {'mesh': 'v1'}, 'writes': []}})
        self.ledger, self.call = completed_call(self.root, self.packet)
        self.network = patch('http.client.HTTPSConnection', side_effect=AssertionError('Recovery must not call provider'))
        self.network.start(); self.addCleanup(self.network.stop)

    def recover(self, **kwargs):
        return reconcile_completed_response(self.call, self.ledger, lambda: deepcopy(self.state), **kwargs)

    def test_existing_cost_is_not_added_twice_and_failed_evidence_survives(self):
        original = read_json(self.ledger / 'budget.json')
        for _ in range(2):
            result = self.recover()
            self.assertEqual(result['accounting_delta_usd'], 0)
            self.assertEqual(result['provider_calls_added'], 0)
        after = read_json(self.ledger / 'budget.json')
        self.assertEqual(after['known_cost_usd'], original['known_cost_usd'])
        self.assertEqual(after['provider_slots_reserved_or_attempted'], 1)
        self.assertEqual(len(after['attempts']), 1)
        before = read_json(result['original_ledger_evidence'])
        self.assertEqual(before['ledger']['attempts'][0]['status'], 'failed_or_uncertain')

    def test_crash_before_accounting_is_recovered_once(self):
        completed_call(self.root, self.packet, accounted=False)
        first = self.recover(); second = self.recover()
        self.assertGreater(first['accounting_delta_usd'], 0)
        self.assertEqual(second['accounting_delta_usd'], 0)
        self.assertEqual(read_json(self.ledger / 'budget.json')['reserved_usd'], 0)

    def test_interruption_during_recovery_does_not_double_account(self):
        completed_call(self.root, self.packet, accounted=False)
        original = write_json
        def fail_ledger(path, value):
            if Path(path).name == 'budget.json': raise PermissionError('reader still open')
            original(path, value)
        with patch('modeling_system.provider_recovery.write_json', side_effect=fail_ledger):
            with self.assertRaises(PermissionError): self.recover()
        self.assertTrue((self.call / 'selection.json').exists())
        self.recover(); self.recover()
        self.assertAlmostEqual(read_json(self.ledger / 'budget.json')['known_cost_usd'], .0000042)

    def test_complete_temporary_receipt_can_be_explicitly_recovered(self):
        path = self.call / 'decision-1.response.json'
        temp = self.call / 'decision-1.response.json.tmp-fixture'
        path.rename(temp)
        self.recover(response_path=temp)
        self.assertTrue(temp.exists())
        self.assertEqual(read_json(temp), read_json(path))

    def test_wrong_or_invalid_responses_never_release_or_change_ledger(self):
        path = self.call / 'decision-1.response.json'
        base = read_json(path); ledger_bytes = (self.ledger / 'budget.json').read_bytes()
        changes = [lambda r: r.update(http=503), lambda r: r.update(wire_request_digest='wrong'),
            lambda r: r['response'].update(model='unqualified'),
            lambda r: r['response']['usage'].update(input_tokens=-1),
            lambda r: r['response'].update(answers={}), lambda r: r.update(estimated_cost_usd=9.)]
        for change in changes:
            value = deepcopy(base); change(value); write_json(path, value)
            with self.assertRaises(ValueError): self.recover()
            self.assertEqual((self.ledger / 'budget.json').read_bytes(), ledger_bytes)
            self.assertFalse((self.call / 'selection.json').exists())

    def test_stale_inputs_or_busy_region_do_not_release(self):
        self.state['values']['mesh'] = 'v2'
        with self.assertRaises(ValueError): self.recover()
        self.state['values']['mesh'] = 'v1'
        self.state['active_operations'] = [{'id': 'editing', 'writes': ['mesh']}]
        with self.assertRaises(ValueError): self.recover()

    def test_original_source_and_active_dispatch_lock_are_enforced(self):
        write_json(self.root / 'source-packet.json', {})
        with self.assertRaises(ValueError): self.recover()
        write_json(self.ledger / 'dispatch.lock', {'active': True})
        with self.assertRaises(FileExistsError): self.recover()


class TerminalRetryTests(unittest.TestCase):
    def setUp(self):
        ProviderRecoveryTests.setUp(self)
        path = self.call/'decision-1.response.json'
        response = read_json(path); response.pop('response'); response['http'] = 503
        write_json(path, response)
        ledger = read_json(self.ledger/'budget.json')
        ledger['attempts'][0]['estimated_cost_usd'] = 0.
        ledger['known_cost_usd'] = 0.
        write_json(self.ledger/'budget.json', ledger)

    # This fixture is a terminal error, so only terminal-specific cases apply.
    def recover(self, **kwargs):
        return prepare_terminal_retry(self.call, self.ledger, lambda: deepcopy(self.state),
                                      reason='Owner requested one retry of recorded transient HTTP503')

    def test_terminal_debit_is_once_and_retry_has_distinct_local_identity(self):
        before = read_json(self.ledger/'budget.json')
        first = self.recover(); second = self.recover()
        after = read_json(self.ledger/'budget.json')
        self.assertEqual(after['known_cost_usd'], .001344)
        self.assertEqual(first, second)
        self.assertEqual(len(after['attempts']), len(before['attempts']))
        self.assertEqual(after['provider_slots_reserved_or_attempted'], 1)
        self.assertEqual(after['attempts'][0]['status'], 'terminal_http_error')
        self.assertNotEqual(first['retry_packet_digest'], first['original_packet_digest'])
        self.assertEqual(first['retry_packet']['state'], self.packet['state'])
        self.assertEqual(first['retry_packet']['questions'], self.packet['questions'])
        self.assertEqual(read_json(self.call/'decision-1.response.json')['http'], 503)

    def test_terminal_retry_chain_is_forbidden(self):
        retry = self.recover()['retry_packet']
        request = read_json(self.call/'decision-1.request.json'); request['local_binding'] = retry['local_binding']
        with self.assertRaises(ValueError):
            terminal_retry_packet(retry, request, read_json(self.call/'decision-1.response.json'),
                                  read_json(self.call/'dispatch-count.json'))

    def test_terminal_cost_or_request_limit_never_changes_ledger(self):
        path = self.ledger/'budget.json'; original = read_json(path)
        for update in ({'known_cost_usd': .004}, {'carry_in_requests': 4}):
            value = deepcopy(original); value.update(update); write_json(path, value)
            before = path.read_bytes()
            with self.assertRaises(ValueError): self.recover()
            self.assertEqual(path.read_bytes(), before)

    def test_terminal_requires_exact_503_not_timeout_or_completed_answer(self):
        path = self.call/'decision-1.response.json'; base = read_json(path)
        ledger = (self.ledger/'budget.json').read_bytes()
        for change in ({'http': None}, {'http': 200}, {'http': 401}, {'response': {}}, {'wire_request_digest': 'other'}):
            value = deepcopy(base); value.update(change); write_json(path, value)
            with self.assertRaises(ValueError): self.recover()
            self.assertEqual((self.ledger/'budget.json').read_bytes(), ledger)

    def test_terminal_preserves_unrelated_uncertainty(self):
        path = self.ledger/'budget.json'; value = read_json(path)
        other = deepcopy(value['attempts'][0]); other.update(output=str(self.root/'other'),packet_digest='other')
        value['attempts'].append(other); write_json(path, value)
        with self.assertRaises(ValueError): self.recover()
        self.assertEqual(read_json(path), value)

    def test_terminal_stale_authority_is_not_retried(self):
        self.state['authority_revision'] = 'changed'
        with self.assertRaises(ValueError): self.recover()


class TransientRetryTests(unittest.TestCase):
    def setUp(self):
        ProviderRecoveryTests.setUp(self)
        (self.call/'decision-1.response.json').unlink()
        value=read_json(self.ledger/'budget.json')
        value['policy']={'mode':'normal_use','spending':'existing_account_credits','authority':'explicit-user-authority'}
        value['carry_in_requests']=1000;value['known_cost_usd']=5.
        value['attempts'][0].update(error_type='TimeoutError',estimated_cost_usd=0.)
        write_json(self.ledger/'budget.json',value)

    def recover(self, **kwargs):
        from .provider_recovery import prepare_transient_retry
        return prepare_transient_retry(self.call,self.ledger,lambda:deepcopy(self.state),**kwargs)

    def test_normal_use_has_no_inherited_lifetime_caps_and_debits_timeout_once(self):
        first=self.recover();second=self.recover()
        ledger=read_json(self.ledger/'budget.json')
        self.assertEqual(first,second)
        self.assertAlmostEqual(ledger['known_cost_usd'],5.001344)
        self.assertEqual(ledger['attempts'][0]['provider_outcome'],'unknown')
        self.assertEqual(ledger['attempts'][0]['status'],'response_unavailable_accounted')
        self.assertEqual(len(ledger['attempts']),1)
        self.assertFalse((self.call/'decision-1.response.json').exists())
        self.assertEqual(ledger['policy'],{'mode':'normal_use','spending':'existing_account_credits','authority':'explicit-user-authority'})

    def test_transient_lineage_allows_two_retries_then_stops(self):
        from .provider_recovery import transient_retry_packet
        packet=self.packet
        for ordinal in (1,2):
            request=read_json(self.call/'decision-1.request.json');request['local_binding']=packet['local_binding']
            packet=transient_retry_packet(packet,request,None,read_json(self.call/'dispatch-count.json'),'TimeoutError')
            self.assertEqual(packet['local_binding']['provider_retry']['ordinal'],ordinal)
        request['local_binding']=packet['local_binding']
        with self.assertRaises(ValueError):
            transient_retry_packet(packet,request,None,read_json(self.call/'dispatch-count.json'),'TimeoutError')

    def test_real_credit_or_authorization_failure_is_not_transient(self):
        wire=read_json(self.call/'decision-1.request.json')['wire_request_digest']
        for http in (401,402,403,422):
            write_json(self.call/'decision-1.response.json',{'http':http,'transport':ROUTE,'wire_request_digest':wire})
            with self.assertRaises(ValueError):self.recover()
        self.assertEqual(read_json(self.ledger/'budget.json')['known_cost_usd'],5.)

    def test_explicit_user_limit_is_still_respected(self):
        value=read_json(self.ledger/'budget.json');value['policy']['max_cost_usd']=5.0001
        write_json(self.ledger/'budget.json',value)
        with self.assertRaises(ValueError):self.recover()

    def test_authority_rebinding_does_not_permit_guide_changes(self):
        packet=deepcopy(self.packet);packet['local_binding']['authority_revision']='new-authority'
        self.state['authority_revision']='new-authority'
        binding={'packet':packet,'authority_keys':['authority'],'evidence':['user-authority-receipt']}
        result=self.recover(rebinding=binding)
        self.assertIn('authority_rebound_from',result['retry_packet']['local_binding']['provider_retry'])
        packet['local_binding']['dependencies']['reads']['mesh']='changed'
        with self.assertRaises(ValueError):self.recover(rebinding=binding)

    def test_feedback_rebind_requires_exact_facts_and_bound_hashes(self):
        from .provider_recovery import validate_feedback_rebinding
        before={'findings':[{'findings':['retained measured fact'],'checks':{'analysis':'pass'},
            'qualification':'technical only','applicability':'current inputs'}], 'visual_feedback':[], 'limits':'unchanged'}
        after=deepcopy(before);after['findings'][0]['applicability']='historical; inputs changed'
        proof={'before':before,'after':after}
        old={'operating_feedback':fingerprint(before)};new={'operating_feedback':fingerprint(after)}
        validate_feedback_rebinding(proof,old,new)
        historical=deepcopy(before);historical['visual_feedback']=[{'disposition':'useful','scope':'old authority'}]
        validate_feedback_rebinding({'before':historical,'after':after},
            {'operating_feedback':fingerprint(historical)},new)
        for mutation in ('finding','check','review','hash'):
            altered=deepcopy(proof)
            if mutation=='finding':altered['after']['findings'][0]['findings']=['different fact']
            if mutation=='check':altered['after']['findings'][0]['checks']['analysis']='fail'
            if mutation=='review':altered['after']['visual_feedback']=['new approval']
            changed={'operating_feedback':fingerprint(altered['after']) if mutation!='hash' else 'unbound'}
            with self.assertRaises(ValueError):validate_feedback_rebinding(altered,old,changed)


if __name__ == '__main__': unittest.main()
