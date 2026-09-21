"""Reconcile a completed TypeSafe request in an existing workspace ledger.

No provider calls, credentials, new budgets or native operations. The legacy
dispatcher and this helper share dispatch.lock. Original failed receipts remain
evidence; accounting is reconciled once under the existing attempt identity.
"""
from copy import deepcopy
from pathlib import Path
import math

from .controller import fingerprint, read_json, write_json
from .typesafe_transport import MODEL, ROUTE, validate_response_receipt

SETTLED_PROVIDER = {'response_validated', 'verified_no_dispatch', 'terminal_http_error'}


def terminal_retry_packet(packet, request, response, dispatch):
    """One explicit retry lineage for a recorded HTTP503; no timeout inference."""
    from .judgments import validate_questions
    validate_questions(packet['questions'])
    wire = {'model': MODEL, 'state': packet['state'], 'questions': packet['questions']}
    if ('provider_retry' in packet['local_binding'] or response.get('http') != 503
            or 'response' in response or response.get('transport') != ROUTE
            or request.get('transport') != ROUTE or request.get('request') != wire
            or request.get('local_binding') != packet['local_binding']
            or request.get('wire_request_digest') != fingerprint(wire)
            or response.get('wire_request_digest') != fingerprint(wire)
            or dispatch.get('calls_attempted') != 1 or dispatch.get('transport') != ROUTE):
        raise ValueError('Exact first-attempt terminal HTTP503 with no answer required; no retry chains')
    result = deepcopy(packet)
    result['local_binding']['provider_retry'] = {'ordinal': 1,
        'original_packet_digest': fingerprint(packet), 'failed_response_digest': fingerprint(response)}
    return result


def prepare_terminal_retry(call_directory, ledger_directory, current_reader, *, reason):
    """Debit the original reservation once and permit one distinct bounded retry.

    This is a conservative accounting allowance, not reported provider billing.
    Original call count and failed receipts stay intact. The existing dispatcher
    must reserve/record the new packet normally; this function sends nothing.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Explicit owner reason for one terminal-error retry required')
    call, directory = Path(call_directory).resolve(), Path(ledger_directory).resolve()
    lock = directory / 'dispatch.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('terminal HTTP response reconciliation')
    try:
        path = directory / 'budget.json'
        ledger = read_json(path)
        matching = [a for a in ledger['attempts'] if Path(a['output']).resolve() == call]
        if len(matching) != 1 or matching[0]['status'] not in ('failed_or_uncertain', 'terminal_http_error'):
            raise ValueError('Exact failed terminal response attempt required')
        entry = matching[0]
        packet = read_json(call / 'owner-packet.json')
        if fingerprint(packet) != entry['packet_digest'] or read_json(entry['source_packet']) != packet:
            raise ValueError('Original owner packet changed')
        request, response, dispatch = [read_json(call / name) for name in
            ('decision-1.request.json', 'decision-1.response.json', 'dispatch-count.json')]
        retry = terminal_retry_packet(packet, request, response, dispatch)
        check_current(packet, current_reader())
        policy = ledger.get('policy')
        max_requests, max_cost = (4, .005) if policy is None else (policy.get('max_requests'), policy.get('max_cost_usd'))
        if (policy is not None and (policy.get('mode') != 'normal_use' or not policy.get('authority'))
                or type(max_requests) is not int or not 1 <= max_requests <= 128
                or type(max_cost) not in (int, float) or not math.isfinite(max_cost) or not 0 < max_cost <= .005):
            raise ValueError('Existing authorized budget limits required')
        reserve = entry.get('reserved_usd')
        total, debited = ledger['known_cost_usd'], entry.get('conservative_cost_debit_usd', 0.)
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (reserve, total, debited))
                or reserve <= 0 or debited not in (0., reserve) or entry.get('estimated_cost_usd', 0.) != 0):
            raise ValueError('Original reservation and unbilled terminal-response accounting required')
        existing = [a for a in ledger['attempts'] if a['packet_digest'] == fingerprint(retry)]
        if len(existing) > 1:
            raise ValueError('Retry identity has multiple attempts')
        unresolved = [a for a in ledger['attempts'] if a is not entry and a not in existing
                      and a['status'] not in SETTLED_PROVIDER]
        if unresolved:
            raise ValueError('Unrelated unresolved provider attempts must remain blocked')
        carried = ledger.get('carry_in_requests', 0)
        if type(carried) is not int or carried < 0:
            raise ValueError('Valid original carried request count required')
        used = carried + sum(a['status'] != 'verified_no_dispatch' for a in ledger['attempts'])
        conservative_total = total + (reserve - debited)
        if not existing and (used + 1 > max_requests or conservative_total + reserve > max_cost):
            raise ValueError('Original conservative debit plus one new reservation exceeds existing budget')
        before = call / ('terminal-before-' + fingerprint(ledger) + '.json')
        if not before.exists(): write_json(before, ledger)
        receipt_path = call / 'terminal-retry.json'
        original_evidence = entry.get('terminal_retry', {}).get('original_ledger_evidence', str(before))
        result = {'status': 'terminal_retry_prepared', 'original_call': str(call),
            'original_packet_digest': fingerprint(packet), 'failed_response_digest': fingerprint(response),
            'retry_packet_digest': fingerprint(retry), 'retry_packet': retry, 'retry_limit': 1,
            'policy_digest': fingerprint(policy), 'conservative_cost_debit_usd': reserve,
            'cost_basis': 'original_reservation_upper_bound_not_reported_billing',
            'original_ledger_evidence': original_evidence, 'reason': reason,
            'provider_calls_added': 0, 'native_dispatch': False}
        if receipt_path.exists():
            original = read_json(receipt_path)
            if any(original.get(k) != result[k] for k in
                   ('original_packet_digest', 'failed_response_digest', 'retry_packet_digest', 'policy_digest')):
                raise ValueError('Original retry lineage or budget policy changed')
            result = original
        else:
            write_json(receipt_path, result)
        entry.update(status='terminal_http_error', conservative_cost_debit_usd=reserve,
            terminal_retry={k: result[k] for k in ('retry_packet_digest', 'original_ledger_evidence')})
        ledger['known_cost_usd'] = conservative_total
        ledger['cost_basis'] = 'historical_cost_and_token_estimates_plus_explicit_conservative_debits'
        # Do not clear or repeat an already attempted retry. Its own effect stays
        # unresolved unless a completed answer can be reconciled normally.
        outstanding = [a for a in ledger['attempts'] if a['status'] not in SETTLED_PROVIDER]
        ledger['reserved_usd'] = sum(a.get('reserved_usd', 0.) for a in outstanding if a['status'] == 'reserved')
        ledger['status'] = 'needs_reconciliation' if outstanding else 'ready'
        write_json(path, ledger)
        return result
    finally:
        lock.unlink()


def check_current(packet, current):
    binding = packet['local_binding']
    if (current.get('owner') != binding['owner'] or
            current.get('authority_revision') != binding['authority_revision']):
        raise ValueError('Current owner and authority must match the original request')
    reads = packet['local_binding']['dependencies']['reads']
    writes = packet['local_binding']['dependencies']['writes']
    if not reads or any(current['values'].get(k) != v for k, v in reads.items()):
        raise ValueError('Relevant dependency changed or missing')
    active = current['active_operations']
    if not isinstance(active, list) or any(not isinstance(a.get('writes'), list) for a in active):
        raise ValueError('Observed active-operation writes required')
    if any(set(a['writes']) & (set(reads) | set(writes)) for a in active):
        raise ValueError('Selection depends on work currently being changed')


def reconcile_completed_response(call_directory, ledger_directory, current_reader, *, response_path=None):
    """Repair exact completed response metadata and accounting, never retry HTTP.

    response_path may name the explicitly inspected complete temporary receipt
    retained after a failed atomic rename. No arbitrary temporary file is chosen.
    Repeating this operation is idempotent; a previously accounted cost is kept.
    """
    call, ledger_directory = Path(call_directory).resolve(), Path(ledger_directory).resolve()
    lock = ledger_directory / 'dispatch.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('completed response reconciliation')
    try:
        ledger_path = ledger_directory / 'budget.json'
        ledger = read_json(ledger_path)
        matches = [a for a in ledger['attempts'] if Path(a['output']).resolve() == call]
        if len(matches) != 1:
            raise ValueError('Exactly one existing ledger attempt must own this response')
        entry = matches[0]
        if entry['status'] not in ('reserved', 'failed_or_uncertain', 'invalidated_after_response', 'response_validated'):
            raise ValueError('Attempt is not a completed-response recovery candidate')
        packet = read_json(call / 'owner-packet.json')
        if (fingerprint(packet) != entry['packet_digest'] or
                read_json(entry['source_packet']) != packet):
            raise ValueError('Original packet or owner source changed')
        source = Path(response_path).resolve() if response_path else call / 'decision-1.response.json'
        if source.parent != call or not source.name.startswith('decision-1.response.json'):
            raise ValueError('Response must belong to the original call directory')
        response = read_json(source)
        request = read_json(call / 'decision-1.request.json')
        validated = validate_response_receipt(packet, request, response)
        dispatch = read_json(call / 'dispatch-count.json')
        if dispatch.get('calls_attempted') != 1 or dispatch.get('transport') != response['transport']:
            raise ValueError('Exact single-attempt dispatch evidence required')
        check_current(packet, current_reader())
        cost = validated['estimated_cost_usd']
        accounted = entry.get('estimated_cost_usd', 0.)
        total = ledger['known_cost_usd']
        if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in (accounted, total))
                or accounted not in (0., cost) or total + 1e-12 < accounted):
            raise ValueError('Existing accounting conflicts with completed-response usage')
        delta = cost - accounted
        selected = {'status': 'validated_selection', 'packet_digest': fingerprint(packet),
            'source_packet': entry['source_packet'], 'local_binding': packet['local_binding'],
            'choices': validated['choices'], 'native_dispatch': False,
            'owner_release_revalidation_required': True, 'elapsed_ms': response.get('elapsed_ms')}
        selected_path = call / 'selection.json'
        if selected_path.exists():
            old = read_json(selected_path)
            if any(old.get(k) != selected[k] for k in ('packet_digest', 'local_binding', 'choices')):
                raise ValueError('Existing selection conflicts with the validated response')
        # Immutable before-images survive a second crash anywhere in recovery.
        before_path = call / ('recovery-before-' + fingerprint(ledger) + '.json')
        if not before_path.exists():
            write_json(before_path, {'ledger': deepcopy(ledger), 'response_source': str(source),
                'response_digest': fingerprint(response)})
        original_evidence = entry.get('completed_response_recovery', {}).get('original_evidence', str(before_path))
        if source != call / 'decision-1.response.json':
            destination = call / 'decision-1.response.json'
            if destination.exists() and read_json(destination) != response:
                raise ValueError('An existing different response must be reconciled explicitly')
            write_json(destination, response)
        write_json(selected_path, selected)
        # Selection is durable before the ledger can advertise it as reusable.
        entry.update(status='response_validated', estimated_cost_usd=cost,
            cost_basis='reported_input_tokens_at_published_rate', transport=response['transport'])
        entry['completed_response_recovery'] = {'original_evidence': original_evidence,
            'response_digest': fingerprint(response)}
        ledger['known_cost_usd'] = total + delta
        unresolved = [a for a in ledger['attempts'] if a['status'] not in SETTLED_PROVIDER]
        ledger['reserved_usd'] = sum(a.get('reserved_usd', 0.) for a in unresolved
                                     if a['status'] == 'reserved')
        ledger['status'] = 'needs_reconciliation' if unresolved else 'ready'
        write_json(ledger_path, ledger)
        result = {'status': 'reconciled_completed_response', 'packet_digest': fingerprint(packet),
            'wire_request_digest': validated['wire_request_digest'], 'response_digest': fingerprint(response),
            'choices': validated['choices'], 'provider_calls_added': 0, 'native_dispatch': False,
            'accounting_delta_usd': delta, 'estimated_cost_usd': cost,
            'original_ledger_evidence': original_evidence, 'ledger_status': ledger['status']}
        write_json(call / 'completed-response-reconciliation.json', result)
        return result
    finally:
        lock.unlink()
