from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .controller import fingerprint, read_json, write_json
from .judgments import prepare_judgments
from .provider_recovery import reconcile_completed_response
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


if __name__ == '__main__': unittest.main()
