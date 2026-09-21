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


if __name__ == '__main__':
    unittest.main()
