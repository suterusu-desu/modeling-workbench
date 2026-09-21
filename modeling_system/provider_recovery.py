"""Reconcile a completed TypeSafe request in an existing workspace ledger.

No provider calls, credentials, new budgets or native operations. The legacy
dispatcher and this helper share dispatch.lock. Original failed receipts remain
evidence; accounting is reconciled once under the existing attempt identity.
"""
from copy import deepcopy
from pathlib import Path
import math

from .controller import fingerprint, read_json, write_json
from .typesafe_transport import validate_response_receipt


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
        unresolved = [a for a in ledger['attempts'] if a['status'] not in ('response_validated', 'verified_no_dispatch')]
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
