"""One ordinary bounded Jev call for multiple recurring Planner judgments.

No import-time calls, native effects, new budget or background agent.
Transient retries preserve the original request and share the existing bound.
"""
from pathlib import Path
import time
from modeling_system.judgments import prepare_judgments, resolve_judgments
from modeling_system.controller import SelectionNotDispatched
from modeling_system.provider_recovery import reconcile_completed_response

from .provider_dispatch import dispatch_many, release_selection, read, write, digest
from .typesafe_transport import MODEL


def decide(public_state, decisions, *, current_binding=None, active_operations=None,
           current_context=None, ledger_directory, output_directory):
    """Use one fresh binding/lane snapshot per boundary; never cache across calls.

    Legacy separate callbacks remain supported. A session should instead supply
    current_context=lambda: session.judgment_context(binding).
    """
    started = time.perf_counter()
    timings = {'fresh_context_reads': 0, 'fresh_context_ms': 0.}
    if current_context is not None:
        if current_binding is not None or active_operations is not None:
            raise ValueError('Supply a combined context or the legacy pair, not both')
    elif not callable(current_binding) or not callable(active_operations):
        raise ValueError('Actual binding and lane readers are required')

    def snapshot():
        before = time.perf_counter()
        try:
            if current_context is not None:
                return current_context()
            return {'binding': current_binding(), 'active_operations': active_operations()}
        finally:
            timings['fresh_context_reads'] += 1
            timings['fresh_context_ms'] += (time.perf_counter()-before)*1000

    ledger = Path(ledger_directory).resolve()
    if not (ledger/'budget.json').is_file():
        raise ValueError('Existing authorized provider ledger required')
    binding = snapshot()['binding']
    try:
        packet = prepare_judgments(public_state, decisions, binding)
    except ValueError as error:
        raise SelectionNotDispatched(str(error)) from error
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)

    def current():
        observed = snapshot()
        fresh = observed['binding']
        if digest(fresh) != digest(binding):
            raise ValueError('Planner authority, menu context or evidence changed')
        return {'values': fresh['dependencies']['reads'], 'active_operations': observed['active_operations'],
                'owner': fresh['owner'], 'authority_revision': fresh['authority_revision']}

    current()
    before = time.perf_counter()
    call, retry_receipt, reused = resolve_packet(packet, ledger, current, output,
                                               reuse_validated_dispatch=True)
    timings['resolve_packet_ms'] = (time.perf_counter()-before)*1000
    before = time.perf_counter()
    released = release_selection(call/'selection.json', current, require_idle=False)
    timings['release_ms'] = (time.perf_counter()-before)*1000
    response = read(call/'decision-1.response.json')
    expected = {'model': MODEL, 'state': packet['state'], 'questions': packet['questions']}
    if response['wire_request_digest'] != digest(expected):
        raise ValueError('Wrong response binding')
    result = resolve_judgments(packet, response['response']['answers'], snapshot()['binding'])
    timings['total_ms_before_report'] = (time.perf_counter()-started)*1000
    timings['response_http_ms'] = response.get('elapsed_ms')
    result.update(reused=reused, provider_receipt=str(call/'decision-1.response.json'),
                  released_packet_digest=released['packet_digest'],
                  model=response['response']['model'], usage=response['response']['usage'],
                  bridge_timings=timings,
                  timing_limits='Nested wall times overlap. HTTP includes client/network/server; a reused response retains historical HTTP elapsed.')
    write(output/(digest(packet)[:16]+'.judgments.json'), result)
    return result


def resolve_packet(packet, ledger, current, output, *, initial_call=None, rebinding=None, reuse_validated_dispatch=False):
    """Resume one logical choice with up to two transient retries and backoff."""
    import time
    from modeling_system.invalid_response import prepare_selection_retry
    call, retry_receipt, did_dispatch = initial_call, None, False
    while True:
        if call is None:
            matches = [a for a in read(ledger/'budget.json')['attempts'] if a['packet_digest'] == digest(packet)]
            if len(matches) > 1: raise ValueError('Duplicate provider packet identity')
            if matches:
                call = Path(matches[0]['output'])
            else:
                path = output/(digest(packet)[:24]+'.packet.json')
                if path.exists() and read(path) != packet: raise ValueError('Packet identity collision')
                if not path.exists(): write(path,packet)
                did_dispatch = True
                report = dispatch_many([path],ledger,current)
                # Short ancillary basenames; the full digest stays in the receipt.
                write(output/(digest(packet)[:16]+'.dispatch.json'),report)
                call = Path(report['results'][0]['output'])
                # Fresh success is already validated and durably accounted by
                # the dispatcher. Release still observes current state. Saved,
                # interrupted and retried attempts keep completed recovery.
                if (reuse_validated_dispatch and initial_call is None and retry_receipt is None
                        and report['status'] == 'returned' and len(report['results']) == 1
                        and report['results'][0]['status'] == 'response_validated'):
                    return call, None, False
        try:
            reconcile_completed_response(call,ledger,current)
            return call, retry_receipt, not did_dispatch
        except (ValueError, FileNotFoundError):
            authorization, retry_receipt = prepare_selection_retry(call,ledger,current,rebinding=rebinding)
        packet = authorization['retry_packet']
        from modeling_system.provider_recovery import retry_ready, ProviderRetryDeferred
        remaining = max(0., authorization['not_before'] - time.time())
        if remaining > 30:
            raise ProviderRetryDeferred(authorization['not_before'])
        if remaining:
            time.sleep(remaining)
        retry_ready(authorization['not_before'])
        current()  # Re-observe after backoff; never dispatch stale evidence.
        rebinding = None
        call = None
