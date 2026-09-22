"""Account malformed typed answers and retry the same question with bounded lineage.

Original provider bytes and billed usage stay intact. No answer repair, native
effects or provider calls occur here. Wire/model/usage failures remain uncertain.
"""
from copy import deepcopy
from pathlib import Path
import math
import time

from .controller import fingerprint, read_json, write_json
from .decision_budget import bound_budget
from .judgments import validate_answers, validate_questions
from .provider_recovery import SETTLED_PROVIDER, budget_limits, check_current
from .typesafe_transport import ROUTE, validate_response_envelope


def invalid_answer_evidence(packet, request, response, dispatch):
    validated = validate_response_envelope(packet, request, response)
    if dispatch.get('calls_attempted') != 1 or dispatch.get('transport') != ROUTE:
        raise ValueError('Exact single-attempt dispatch evidence required')
    budget = bound_budget(packet['local_binding'])
    validate_questions(packet['questions'], budget=budget)
    try:
        validate_answers(packet['questions'], response['response'].get('answers'), budget=budget)
    except ValueError as error:
        return {**validated, 'validation_error': str(error)}
    raise ValueError('Valid answers require completed-response reconciliation, not a retry')


def invalid_answer_retry_packet(packet, request, response, dispatch):
    invalid_answer_evidence(packet, request, response, dispatch)
    prior = packet['local_binding'].get('provider_retry', {})
    ordinal = prior.get('ordinal', 0) + 1
    if ordinal > 2:
        raise ValueError('Provider retry bound reached; retain invalid answer and stop')
    base = deepcopy(packet); base['local_binding'].pop('provider_retry', None)
    root = fingerprint(base)
    if prior and prior.get('root_packet_digest', prior.get('original_packet_digest')) != root:
        raise ValueError('Retry lineage changed')
    result = deepcopy(base)
    lineage = {'kind': 'invalid_answer', 'ordinal': ordinal, 'root_packet_digest': root,
        'previous_packet_digest': fingerprint(packet), 'failure_digest': fingerprint(response),
        'previous_provider_outcome': 'invalid_answer'}
    if 'received_at' in response:
        received = response['received_at']
        if type(received) not in (int, float) or not math.isfinite(received) or received < 0:
            raise ValueError('Invalid retained provider timing')
        lineage['not_before'] = received + (.5, 2.)[ordinal - 1]
    result['local_binding']['provider_retry'] = lineage
    return result


def prepare_invalid_answer_retry(call_directory, ledger_directory, current_reader):
    """Settle known cost even at retry exhaustion; authorize only fresh successors."""
    call, directory = Path(call_directory).resolve(), Path(ledger_directory).resolve()
    lock = directory/'dispatch.lock'
    with lock.open('x', encoding='utf-8') as stream: stream.write('invalid answer recovery')
    try:
        path = directory/'budget.json'; ledger = read_json(path)
        rows = [a for a in ledger['attempts'] if Path(a['output']).resolve() == call]
        if len(rows) != 1 or rows[0]['status'] not in ('failed_or_uncertain', 'invalid_answer_accounted'):
            raise ValueError('Settled invalid-answer worker required; do not retry an in-flight call')
        entry = rows[0]
        packet, request, response, dispatch = [read_json(call/name) for name in
            ('owner-packet.json', 'decision-1.request.json', 'decision-1.response.json', 'dispatch-count.json')]
        if fingerprint(packet) != entry['packet_digest'] or read_json(entry['source_packet']) != packet:
            raise ValueError('Original packet changed')
        evidence = invalid_answer_evidence(packet, request, response, dispatch)
        cost, total, accounted = evidence['estimated_cost_usd'], ledger['known_cost_usd'], entry.get('estimated_cost_usd', 0.)
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (total, accounted))
                or accounted not in (0., cost) or total + 1e-12 < accounted
                or entry.get('conservative_cost_debit_usd', 0) != 0):
            raise ValueError('Existing accounting conflicts with reported response usage')
        if (call/'selection.json').exists():
            raise ValueError('An invalid answer must not have a released selection')
        before = call/('invalid-answer-before-' + fingerprint(ledger)[:16] + '.json')
        if not before.exists(): write_json(before, ledger)
        original = entry.get('invalid_answer_recovery', {}).get('original_evidence', str(before))
        retained = {'original_evidence': original, 'response_digest': fingerprint(response), **evidence}
        entry.update(status='invalid_answer_accounted', estimated_cost_usd=cost,
            cost_basis='reported_input_tokens_at_published_rate', invalid_answer_recovery=retained)
        ledger['known_cost_usd'] = total + (cost - accounted)
        outstanding = [a for a in ledger['attempts'] if a['status'] not in SETTLED_PROVIDER]
        ledger['status'] = 'needs_reconciliation' if outstanding else 'ready'
        ledger['reserved_usd'] = sum(a.get('reserved_usd', 0.) for a in outstanding if a['status'] == 'reserved')
        write_json(call/'invalid-answer-accounting.json', {**retained, 'accounting_delta_usd': cost-accounted,
            'provider_calls_added': 0, 'native_dispatch': False})
        write_json(path, ledger)
        # Accounting does not depend on whether a new decision is still applicable.
        check_current(packet, current_reader())
        retry = invalid_answer_retry_packet(packet, request, response, dispatch)
        retry_digest = fingerprint(retry)
        existing = [a for a in ledger['attempts'] if a['packet_digest'] == retry_digest]
        if len(existing) > 1 or any(a not in existing for a in outstanding):
            raise ValueError('Unrelated or duplicate unresolved requests cannot be cleared')
        max_requests, max_cost = budget_limits(ledger)
        reserve = entry.get('reserved_usd'); carried = ledger.get('carry_in_requests', 0)
        if (type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0
                or type(carried) is not int or carried < 0):
            raise ValueError('Original reservation and carried request count required')
        used = carried + sum(a['status'] != 'verified_no_dispatch' for a in ledger['attempts'])
        if not existing and (used+1 > max_requests or ledger['known_cost_usd']+reserve > max_cost):
            raise ValueError('Explicit user budget cannot cover the next bounded retry')
        result = {'status': 'invalid_answer_retry_prepared', 'previous_call': str(call),
            'root_packet_digest': retry['local_binding']['provider_retry']['root_packet_digest'],
            'retry_packet': retry, 'retry_packet_digest': retry_digest, 'original_ledger_evidence': original,
            'estimated_cost_usd': cost, 'provider_calls_added': 0, 'native_dispatch': False,
            'not_before': retry['local_binding']['provider_retry'].get('not_before',
                time.time()+(.5, 2.)[retry['local_binding']['provider_retry']['ordinal']-1])}
        receipt_path = call/'invalid-answer-retry.json'
        if receipt_path.exists():
            prior = read_json(receipt_path)
            if prior['retry_packet_digest'] != retry_digest: raise ValueError('Existing retry identity changed')
            return prior
        write_json(receipt_path, result)
        return result
    finally:
        lock.unlink()


def prepare_selection_retry(call, ledger, current_reader, *, rebinding=None):
    """Route known invalid answers separately from transient transport failures."""
    from .provider_recovery import prepare_transient_retry
    response = Path(call)/'decision-1.response.json'
    if response.exists() and read_json(response).get('http') == 200:
        if rebinding is not None: raise ValueError('Invalid-answer retries cannot rebind an existing decision')
        result = prepare_invalid_answer_retry(call, ledger, current_reader)
        return result, Path(call)/'invalid-answer-retry.json'
    return prepare_transient_retry(call, ledger, current_reader, rebinding=rebinding), Path(call)/'transient-retry.json'
