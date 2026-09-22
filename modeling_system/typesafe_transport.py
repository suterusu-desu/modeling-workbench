"""Direct TypeSafe v1 transport; callers own credentials, budgets and execution.

No import-time network, automatic retry, provider fallback or native effects.
Costs are estimates from reported input tokens and the published model rate,
not provider-returned invoice amounts. Source: https://docs.typesafe.ai/models
"""
import hashlib
import http.client
import json
import time
import re
from datetime import timezone
from email.utils import parsedate_to_datetime
from .decision_budget import bound_budget

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


def validate_response_envelope(packet, request_receipt, receipt):
    """Verify exact wire identity and reported cost; this does not release answers."""
    expected = {'model': MODEL, 'state': packet['state'], 'questions': packet['questions']}
    if (request_receipt.get('transport') != ROUTE or receipt.get('transport') != ROUTE
            or request_receipt.get('request') != expected
            or request_receipt.get('local_binding') != packet['local_binding']
            or request_receipt.get('wire_request_digest') != digest(expected)
            or receipt.get('wire_request_digest') != digest(expected)
            or receipt.get('http') != 200):
        raise ValueError('Completed response does not match the exact original request')
    response = receipt.get('response', {})
    if response.get('model') != MODEL:
        raise ValueError('Unreviewed TypeSafe model revision')
    cost = usage_cost(response)
    if 'estimated_cost_usd' in receipt and receipt['estimated_cost_usd'] != cost:
        raise ValueError('Response cost disagrees with recorded token usage')
    return {'estimated_cost_usd': cost, 'wire_request_digest': digest(expected)}


def validate_response_receipt(packet, request_receipt, receipt):
    """Verify saved wire evidence and answers without network or ledger writes."""
    from .judgments import validate_answers
    result = validate_response_envelope(packet, request_receipt, receipt)
    result['choices'] = validate_answers(packet['questions'], receipt['response'].get('answers'),
                                        budget=bound_budget(packet['local_binding']))
    return result


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
        budget = bound_budget(local_binding)
        validate_questions(questions, budget=budget)
        budget.check(state, questions, MODEL)
        from .provider_recovery import retry_ready
        retry_ready(local_binding.get('provider_retry', {}).get('not_before'))
        request = {'model': MODEL, 'state': state, 'questions': questions}
        body = json.dumps(request, allow_nan=False).encode()
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
                   'elapsed_ms': (time.perf_counter() - started) * 1000,
                   'received_at': time.time()}
        receipt.update(response_metadata(reply, self.key, receipt['received_at']))
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
        try:
            self.cost = usage_cost(response)
        except ValueError:
            self.write(prefix.with_suffix('.response.json'), receipt)
            raise
        receipt.update(estimated_cost_usd=self.cost, cost_basis=self.cost_basis,
                       input_usd_per_million=INPUT_USD_PER_MILLION,
                       pricing_source='https://docs.typesafe.ai/models')
        self.write(prefix.with_suffix('.response.json'), receipt)
        if response.get('model') != MODEL:
            raise ValueError('Unreviewed TypeSafe model revision')
        return validate_answers(questions, response.get('answers'), budget=budget)

    def close(self):
        self.key = ''
        if self.connection is not None:
            self.connection.close()


def response_metadata(reply, secret, received_at):
    """Retain only bounded validated trace/timing fields, never arbitrary headers."""
    get = getattr(reply, 'getheader', lambda name: None)
    result = {}
    request_id = get('x-typesafe-request-id')
    if (isinstance(request_id, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', request_id)
            and (not secret or secret not in request_id)):
        result['request_id'] = request_id
    delays = []
    for name, scale in (('retry-after-ms', .001), ('Retry-After', 1.)):
        value = get(name)
        if not isinstance(value, str) or len(value) > 100 or secret and secret in value:
            continue
        try:
            if re.fullmatch(r'\d{1,12}(?:\.\d{1,3})?', value.strip()):
                seconds = float(value) * scale
            elif name == 'Retry-After':
                date = parsedate_to_datetime(value)
                seconds = max(0., date.replace(tzinfo=date.tzinfo or timezone.utc).timestamp() - received_at)
            else:
                continue
            # Long valid hints defer; they are not clamped into an early retry.
            if 0 <= seconds <= 253402300799 - received_at:
                delays.append(seconds)
        except (ValueError, TypeError, OverflowError):
            continue
    if delays:
        result['retry_after_seconds'] = max(delays)
    return result
