"""Direct TypeSafe v1 transport; callers own credentials, budgets and execution.

No import-time network, automatic retry, provider fallback or native effects.
Costs are estimates from reported input tokens and the published model rate,
not provider-returned invoice amounts. Source: https://docs.typesafe.ai/models
"""
import hashlib
import http.client
import json
import time

MODEL = 'jev-1.13.0'
INPUT_USD_PER_MILLION = .042
ROUTE = 'typesafe_direct_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def usage_cost(response):
    usage = response.get('usage', {})
    if any(type(usage.get(k)) is not int or usage[k] < 0
           for k in ('input_tokens', 'output_tokens')):
        raise ValueError('Actual token usage required')
    return usage['input_tokens'] * INPUT_USD_PER_MILLION / 1_000_000


class TypeSafeTransport:
    """One instance per reserved ledger request; never writes or logs credentials."""
    def __init__(self, output, write, *, api_key, connection_factory=None):
        self.output, self.write, self.key = output, write, api_key
        self.connection_factory = connection_factory or http.client.HTTPSConnection
        self.connection = None
        self.calls = 0
        self.cost = 0.
        self.cost_basis = 'reported_input_tokens_at_published_rate'

    def choose(self, state, questions, local_binding):
        from modeling_system.judgments import validate_questions, validate_answers
        if self.calls:
            raise RuntimeError('Request already attempted; reconcile instead of retrying')
        if not self.key or any(c.isspace() for c in self.key):
            raise ValueError('Credential unavailable')
        validate_questions(questions)
        request = {'model': MODEL, 'state': state, 'questions': questions}
        body = json.dumps(request, allow_nan=False).encode()
        if len(body) > 12000:
            raise ValueError('Compact selector payload exceeded 12k bytes')
        prefix = self.output / 'decision-1'
        self.write(prefix.with_suffix('.request.json'), {
            'transport': ROUTE, 'local_binding': local_binding,
            'wire_request_digest': digest(request), 'request': request})
        self.calls = 1
        self.write(self.output / 'dispatch-count.json', {'calls_attempted': 1, 'transport': ROUTE})
        started = time.perf_counter()
        self.connection = self.connection_factory('api.typesafe.ai', timeout=20)
        self.connection.request('POST', '/v1/systemone', body=body,
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        reply = self.connection.getresponse()
        raw = reply.read()
        receipt = {'transport': ROUTE, 'http': reply.status,
                   'wire_request_digest': digest(request),
                   'elapsed_ms': (time.perf_counter() - started) * 1000}
        if reply.status != 200:
            # Error bodies can echo inputs; retain status, not unknown server text.
            self.write(prefix.with_suffix('.response.json'), receipt)
            raise RuntimeError('TypeSafe refused request; reconcile original attempt')
        response = json.loads(raw)
        # Defensive redaction if an upstream service ever echoes the credential.
        if self.key in json.dumps(response):
            self.write(prefix.with_suffix('.response.json'), receipt)
            raise ValueError('Unexpected credential echo in response')
        receipt['response'] = response
        self.write(prefix.with_suffix('.response.json'), receipt)
        self.cost = usage_cost(response)
        receipt.update(estimated_cost_usd=self.cost, cost_basis=self.cost_basis,
                       input_usd_per_million=INPUT_USD_PER_MILLION,
                       pricing_source='https://docs.typesafe.ai/models')
        self.write(prefix.with_suffix('.response.json'), receipt)
        if response.get('model') != MODEL:
            raise ValueError('Unreviewed TypeSafe model revision')
        return validate_answers(questions, response.get('answers'))

    def close(self):
        self.key = ''
        if self.connection is not None:
            self.connection.close()
