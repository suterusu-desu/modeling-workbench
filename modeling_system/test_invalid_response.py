from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .controller import fingerprint, read_json, write_json
from .judgments import prepare_judgments, validate_answers
from .invalid_response import prepare_invalid_answer_retry, invalid_answer_retry_packet
from .provider_recovery import transient_retry_packet
from .test_provider_recovery import completed_call


class InvalidAnswerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root = Path(temp.name)
        self.state = {'values': {'mesh': 'v1'}, 'active_operations': [], 'owner': 'owner', 'authority_revision': 'a1'}
        self.packet = prepare_judgments({'goal': 'Resolve actual missing support'}, [{'id': 'method', 'type': 'choice',
            'instructions': 'Choose the supported method', 'options': [{'id': 'fit', 'description': 'Qualified fit'},
            {'id': 'inspect', 'description': 'Inspect relevant support'}]}], {'owner': 'owner', 'authority_revision': 'a1',
            'dependencies': {'reads': {'mesh': 'v1'}, 'writes': []}})
        self.ledger, self.call = completed_call(self.root, self.packet)
        self.path = self.call/'decision-1.response.json'
        self.response = read_json(self.path)
        self.response['response']['answers']['q0']['probabilities'] = {'o0': .49, 'o1': .51}
        write_json(self.path, self.response)
        context = patch('http.client.HTTPSConnection', side_effect=AssertionError('No network in reconciliation'))
        context.start(); self.addCleanup(context.stop)

    def recover(self):
        return prepare_invalid_answer_retry(self.call, self.ledger, lambda: deepcopy(self.state))

    def test_invalid_choice_is_accounted_once_preserving_raw_answer(self):
        raw = self.path.read_bytes(); old = read_json(self.ledger/'budget.json')
        old['known_cost_usd'] += .123456789123
        old['policy'] = {'mode':'normal_use','authority':'user','spending':'existing_account_credits'}
        write_json(self.ledger/'budget.json',old)
        first = self.recover(); second = self.recover()
        self.assertEqual(first, second); self.assertEqual(raw, self.path.read_bytes())
        new = read_json(self.ledger/'budget.json')
        self.assertEqual(new['known_cost_usd'], old['known_cost_usd'])
        self.assertEqual(new['status'], 'ready'); self.assertFalse((self.call/'selection.json').exists())
        self.assertEqual(new['attempts'][0]['status'], 'invalid_answer_accounted')
        self.assertEqual(first['retry_packet']['state'], self.packet['state'])
        self.assertEqual(first['retry_packet']['questions'], self.packet['questions'])

    def test_unbilled_completed_worker_and_interrupted_accounting_resume_once(self):
        ledger = read_json(self.ledger/'budget.json'); ledger['known_cost_usd'] = 0.
        ledger['attempts'][0]['estimated_cost_usd'] = 0.; write_json(self.ledger/'budget.json', ledger)
        original = write_json
        def interrupted(path, value):
            if Path(path).name == 'budget.json': raise PermissionError('synthetic ledger interruption')
            original(path, value)
        with patch('modeling_system.invalid_response.write_json', side_effect=interrupted):
            with self.assertRaises(PermissionError): self.recover()
        self.recover(); self.recover()
        self.assertAlmostEqual(read_json(self.ledger/'budget.json')['known_cost_usd'], .0000042)

    def test_valid_answers_or_wrong_envelope_do_not_become_invalid_answer_retries(self):
        old = (self.ledger/'budget.json').read_bytes()
        changes = [lambda r: r.update(wire_request_digest='wrong'), lambda r: r['response'].update(model='unreviewed'),
            lambda r: r['response']['usage'].update(input_tokens=-1),
            lambda r: r['response']['answers']['q0'].update(choice='o1')]
        for change in changes:
            value = deepcopy(self.response); change(value); write_json(self.path, value)
            with self.assertRaises(ValueError): self.recover()
            self.assertEqual(old, (self.ledger/'budget.json').read_bytes())

    def test_stale_input_is_accounted_but_never_authorizes_retry(self):
        self.state['values']['mesh'] = 'new'
        with self.assertRaisesRegex(ValueError, 'dependency'): self.recover()
        self.assertEqual(read_json(self.ledger/'budget.json')['attempts'][0]['status'], 'invalid_answer_accounted')
        self.assertFalse((self.call/'invalid-answer-retry.json').exists())

    def test_mixed_network_and_invalid_answers_share_two_retry_bound(self):
        packet = self.packet
        request = read_json(self.call/'decision-1.request.json'); dispatch = read_json(self.call/'dispatch-count.json')
        first = transient_retry_packet(packet, request, None, dispatch, 'TimeoutError')
        request['local_binding'] = first['local_binding']
        second = invalid_answer_retry_packet(first, request, self.response, dispatch)
        self.assertEqual(second['local_binding']['provider_retry']['ordinal'], 2)
        request['local_binding'] = second['local_binding']
        with self.assertRaisesRegex(ValueError, 'bound reached'):
            invalid_answer_retry_packet(second, request, self.response, dispatch)
        with self.assertRaisesRegex(ValueError, 'bound reached'):
            transient_retry_packet(second, request, None, dispatch, 'TimeoutError')

    def test_exhausted_request_limit_still_records_known_cost(self):
        ledger = read_json(self.ledger/'budget.json')
        ledger['policy'] = {'mode': 'normal_use', 'authority': 'user', 'max_requests': 1}
        write_json(self.ledger/'budget.json', ledger)
        with self.assertRaisesRegex(ValueError, 'budget'): self.recover()
        self.assertEqual(read_json(self.ledger/'budget.json')['status'], 'ready')
        self.assertFalse((self.call/'invalid-answer-retry.json').exists())

    def test_only_machine_precision_maximum_tie_is_accepted(self):
        questions = {'q': {'type': 'choice', 'criteria': {'a': 'First', 'b': 'Second', 'c': 'Third'}, 'instructions': 'Choose'}}
        answer = {'q': {'type': 'choice', 'choice': 'a', 'confidence': 0.,
            'probabilities': {'a': .45999999999999996, 'b': .08, 'c': .46}}}
        self.assertEqual(validate_answers(questions, answer), {'q': 'a'})
        answer['q']['probabilities'] = {'a': .43, 'b': .13, 'c': .44}
        with self.assertRaisesRegex(ValueError, 'maximum'): validate_answers(questions, answer)


if __name__ == '__main__': unittest.main()
