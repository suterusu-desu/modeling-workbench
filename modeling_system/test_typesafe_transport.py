import json
from pathlib import Path
import tempfile
import unittest
from modeling_system.typesafe_transport import TypeSafeTransport, MODEL


class TypeSafeTransportTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.addCleanup(self.root.cleanup)
        self.records = {}
        self.requests = []
        self.status = 200
        self.headers = {}
        self.response = {'model': MODEL, 'answers': {'action': {
            'type': 'choice', 'choice': 'inspect',
            'probabilities': {'inspect': .9, 'fit': .1}, 'confidence': .8}},
            'usage': {'input_tokens': 100, 'output_tokens': 1000}}
        self.questions = {'action': {'type': 'choice', 'instructions': 'Choose a useful next operation',
            'criteria': {'inspect': 'Resolve missing support', 'fit': 'Apply qualified fitting'}}}
        test = self
        class Connection:
            def __init__(self, host, timeout): test.assertEqual(host, 'api.typesafe.ai')
            def request(self, method, path, **kwargs): test.requests.append((method, path, kwargs))
            def getresponse(self):
                class Response:
                    status = test.status
                    def read(self): return json.dumps(test.response).encode()
                    def getheader(self, name): return test.headers.get(name)
                return Response()
            def close(self): pass
        self.client = TypeSafeTransport(Path(self.root.name),
            lambda path, value: self.records.__setitem__(path.name, value),
            api_key='fixture-secret', connection_factory=Connection)
        self.addCleanup(self.client.close)

    def test_direct_contract_usage_estimate_and_secret_exclusion(self):
        result = self.client.choose({'support': 'incomplete'}, self.questions, {})
        self.assertEqual(result, {'action': 'inspect'})
        self.assertAlmostEqual(self.client.cost, .0000042)  # Output is free.
        self.assertEqual(self.requests[0][:2], ('POST', '/v1/systemone'))
        self.assertNotIn('fixture-secret', json.dumps(self.records))
        self.assertNotIn('provider', self.response)
        self.assertNotIn('id', self.response)
        with self.assertRaises(RuntimeError): self.client.choose({}, self.questions, {})
        self.assertEqual(len(self.requests), 1)

    def test_invalid_choice_keeps_accounting_and_raw_evidence(self):
        self.response['answers']['action']['choice'] = 'unoffered'
        with self.assertRaises(ValueError): self.client.choose({}, self.questions, {})
        self.assertGreater(self.client.cost, 0)
        self.assertIn('response', self.records['decision-1.response.json'])

    def test_changed_model_is_not_silently_adopted(self):
        self.response['model'] = 'future-model'
        with self.assertRaises(ValueError): self.client.choose({}, self.questions, {})
        self.assertGreater(self.client.cost, 0)

    def test_error_body_cannot_leak_a_secret_and_never_retries(self):
        self.status = 401
        self.response = {'error': 'fixture-secret'}
        with self.assertRaises(RuntimeError): self.client.choose({}, self.questions, {})
        self.assertNotIn('fixture-secret', json.dumps(self.records))
        self.assertEqual(len(self.requests), 1)

    def test_missing_usage_does_not_become_zero_cost_success(self):
        del self.response['usage']['input_tokens']
        with self.assertRaises(ValueError): self.client.choose({}, self.questions, {})

    def test_complete_response_is_written_once_with_accounting_already_known(self):
        writes = []
        def write(path, value):
            if path.name.endswith('.response.json'):
                writes.append(value)
                self.assertAlmostEqual(self.client.cost, .0000042)
        self.client.write = write
        self.client.choose({}, self.questions, {})
        self.assertEqual(len(writes), 1)
        self.assertAlmostEqual(writes[0]['estimated_cost_usd'], .0000042)

    def test_failed_response_write_retains_cost_without_another_http_call(self):
        def write(path, value):
            if path.name.endswith('.response.json'): raise PermissionError('reader still open')
        self.client.write = write
        with self.assertRaises(PermissionError): self.client.choose({}, self.questions, {})
        self.assertAlmostEqual(self.client.cost, .0000042)
        with self.assertRaises(RuntimeError): self.client.choose({}, self.questions, {})
        self.assertEqual(len(self.requests), 1)

    def test_larger_budget_reaches_transport_and_is_enforced_before_auth_dispatch(self):
        from .decision_budget import DecisionBudget
        self.questions = {str(i): {'type': 'noul', 'instructions': 'Is this fact relevant?'} for i in range(20)}
        self.response['answers'] = {k: {'type': 'noul', 'noul': .7} for k in self.questions}
        binding = {'decision_budget': DecisionBudget(max_questions=19).record()}
        with self.assertRaises(ValueError): self.client.choose({}, self.questions, binding)
        self.assertEqual(self.requests, [])
        binding['decision_budget']['max_questions'] = 20
        self.client.choose({'reviewed_text': 'x' * 14000}, self.questions, binding)
        self.assertEqual(len(self.requests), 1)

    def test_safe_request_and_retry_metadata_are_retained_on_an_error(self):
        self.status = 429
        self.headers = {'x-typesafe-request-id': 'trace-123', 'Retry-After': '120',
                        'retry-after-ms': '2500', 'Authorization': 'fixture-secret'}
        with self.assertRaises(RuntimeError): self.client.choose({}, self.questions, {})
        receipt = self.records['decision-1.response.json']
        self.assertEqual(receipt['request_id'], 'trace-123')
        self.assertEqual(receipt['retry_after_seconds'], 120)
        self.assertNotIn('fixture-secret', json.dumps(self.records))
        self.assertNotIn('Authorization', receipt)

    def test_invalid_or_credential_bearing_headers_are_excluded(self):
        self.headers = {'x-typesafe-request-id': 'fixture-secret', 'Retry-After': 'NaN', 'retry-after-ms': '-100'}
        self.client.choose({}, self.questions, {})
        receipt = self.records['decision-1.response.json']
        self.assertNotIn('request_id', receipt)
        self.assertNotIn('retry_after_seconds', receipt)

    def test_future_retry_does_not_write_or_dispatch(self):
        import time
        from .provider_recovery import ProviderRetryDeferred
        with self.assertRaises(ProviderRetryDeferred):
            self.client.choose({}, self.questions, {'provider_retry': {'not_before': time.time() + 100}})
        self.assertEqual(self.records, {})
        self.assertEqual(self.requests, [])

    def test_http_date_and_long_hints_remain_delays_not_early_retries(self):
        from .typesafe_transport import response_metadata
        class Reply:
            def getheader(self, name):
                return {'Retry-After': 'Tue, 14 Nov 2023 23:13:20 GMT'}.get(name)
        self.assertEqual(response_metadata(Reply(), '', 1700000000)['retry_after_seconds'], 3600)


if __name__ == '__main__':
    unittest.main()
