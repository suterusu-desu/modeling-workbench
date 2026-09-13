"""Direct Python, CLI and MCP share durable input/result capture.

Intent is durable before effects. Result is durable before secondary indexing or
episode linkage. Failure of that linkage never converts success into a retry.
"""
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import inspect
import json
from pathlib import Path
import time
import uuid
from .store import atomic_write, canonical, digest
from .runtime import LOADED
from .leases import process_identity, LEASE

EPISODE = ContextVar('modeling_episode', default=None)
ACTIVE_CALL = ContextVar('modeling_active_call', default=None)


def retained_link_identity(link):
    return digest(canonical({k:link.get(k) for k in ('path','sha256','asset')}))


def capture_inputs(store, value, field='', immutable_episode_links=None):
    result = []
    if isinstance(value, dict):
        if immutable_episode_links is not None and field in ('links','add_links') and value.get('kind')=='file' and value.get('asset'):
            # Complete episode evidence is historical and already content-pinned.
            # Do not probe its original source locator during a relocated round
            # trip. Proposal/native/live dependency capture below stays strict.
            if retained_link_identity(value) not in immutable_episode_links:
                raise ValueError('Pinned episode evidence is not a canonical retained link; supply a new typed file link without an asset override')
            blob=value['asset'];store.resolve_blob(blob)
            if value.get('sha256')!=blob['sha256']:raise ValueError('Pinned episode evidence hash differs from its immutable asset')
            result.append(dict(field=field,asset=blob))
        elif isinstance(value.get('path'), str) and value.get('sha256'):
            p = Path(value['path'])
            if p.is_file():
                blob = store.blob(p)
                if blob['sha256'] != value['sha256']:
                    raise ValueError('Input changed before operation: ' + str(p))
                result.append(dict(field=field, asset=blob))
        else:
            for k, v in value.items():
                result.extend(capture_inputs(store, v, k, immutable_episode_links))
    elif isinstance(value, list):
        for v in value:
            result.extend(capture_inputs(store, v, field, immutable_episode_links))
    elif isinstance(value, str) and (field in ('script','evidence','identity_paths','constraints','anatomy_basis_paths')
                                    or field.endswith(('_path','_paths'))):
        if len(value) < 32760 and '\n' not in value and Path(value).is_file():
            result.append(dict(field=field, asset=store.blob(value)))
    return result


def invoke(service, operation, arguments, function):
    if ACTIVE_CALL.get() is not None:
        return function()
    handle = (LEASE.get() or {}).get('operation_handle') or uuid.uuid4().hex
    folder = service.store.root/'calls'/handle
    arguments=json.loads(canonical(arguments))
    immutable_links=None
    if operation=='revise_episode':
        episode=service.ledger.read(arguments['episode'])
        context=episode['data'].get('context',episode['intent'].get('context',{}))
        immutable_links={retained_link_identity(link) for link in context.get('links',[]) if link.get('kind')=='file' and link.get('asset')}
    intent = dict(handle=handle, operation=operation, arguments=arguments,
                  inputs=capture_inputs(service.store, arguments, immutable_episode_links=immutable_links), episode=EPISODE.get(),
                  runtime=LOADED['revision'], process=process_identity(), lease=LEASE.get(),utc=datetime.now(timezone.utc).isoformat(),
                  recovery='If result is absent, inspect original workflow/native receipt before retrying; no automatic replay')
    atomic_write(folder/'intent.json', canonical(intent))
    token = ACTIVE_CALL.set(handle)
    started = time.perf_counter()
    try:
        result = function()
    except BaseException as error:
        outcome = dict(status='raised', error_type=type(error).__name__, error=str(error),
                       elapsed_seconds=time.perf_counter()-started,
                       effect='Unknown; inspect the original job/trial and native receipt, not this exception alone')
        from .ledger import PreconditionRefusal
        if isinstance(error,PreconditionRefusal):
            outcome.update(effect_status=error.modeling_effect_status,stage=error.modeling_stage,
                           details=error.modeling_details,effect='Mutation not dispatched; typed precondition branch after returned observation')
        try:
            atomic_write(folder/'result.json', canonical(outcome))
            error.modeling_operation_handle=handle
            error.modeling_operation_result_path=str(folder/'result.json')
        finally:
            ACTIVE_CALL.reset(token)
        raise
    ACTIVE_CALL.reset(token)
    outcome = dict(status='returned', result=result, elapsed_seconds=time.perf_counter()-started)
    enriched = dict(result) if isinstance(result, dict) else {'result':result}
    enriched.update(operation_handle=handle)
    try:
        # This standalone durable result is the recovery boundary. No lesson or
        # mutable episode pointer must be written to recover produced work.
        atomic_write(folder/'result.json', canonical(outcome))
        fact = service.store.put('operation_fact', dict(intent=intent, outcome=outcome))
        atomic_write(folder/'record.json', canonical({'record':fact}))
        enriched['operation_fact'] = fact
    except (OSError, ValueError, RuntimeError) as error:
        enriched.update(retention_status='needs reconciliation', retention_error=str(error),
                        recovery='Operation returned; preserve this result and reconcile operation_handle. Do not repeat the operation.')
    enriched['operation_result_path'] = str(folder/'result.json')
    return enriched


def recorded(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        args_bound = inspect.signature(method).bind(self, *args, **kwargs)
        args_bound.apply_defaults()
        arguments = {k:v for k,v in args_bound.arguments.items() if k != 'self'}
        return invoke(self, method.__name__, arguments, lambda: method(self, *args, **kwargs))
    return call


def native_effect(service,operation,arguments,function):
    """Persist the raw returned effect BEFORE any internal service ledger update."""
    parent=ACTIVE_CALL.get()
    if not parent:
        return invoke(service,'native_'+operation,arguments,lambda:native_effect(service,operation,arguments,function))
    folder=service.store.root/'calls'/parent/'effects'/uuid.uuid4().hex
    atomic_write(folder/'intent.json',canonical(dict(operation=operation,arguments=json.loads(canonical(arguments)),process=process_identity())))
    try:result=function()
    except Exception as error:
        atomic_write(folder/'result.json',canonical(dict(status='unknown effect',error_type=type(error).__name__,error=str(error))))
        raise
    atomic_write(folder/'result.json',canonical(dict(status='returned',result=result)))
    return result


def effect_rows(folder):
    rows=[]
    for p in sorted((folder/'effects').glob('*/intent.json')):
        intent=json.loads(p.read_bytes());rp=p.parent/'result.json'
        result=json.loads(rp.read_bytes()) if rp.is_file() else {'status':'unknown effect'}
        rows.append(dict(operation=intent['operation'],status=result['status'],result_path=str(rp)))
    return rows


def call_row(store, p, intent=None):
    """Read one durable call with the same recovery semantics as full history."""
    if intent is None: intent=json.loads(p.read_bytes())
    folder=p.parent
    result=json.loads((folder/'result.json').read_bytes()) if (folder/'result.json').exists() else None
    record=json.loads((folder/'record.json').read_bytes())['record'] if (folder/'record.json').exists() else None
    active=result is None and bool(intent.get('process')) and process_identity(intent['process']['pid'])==intent['process']
    lease=intent.get('lease')
    if lease:
        lease_path=store.root/'episode-leases'/lease['episode']/(lease['id']+'.json')
        if lease_path.is_file():
            from .leases import status as lease_status
            active=lease_status(json.loads(lease_path.read_bytes()),store.root)=='active'
    effects=effect_rows(folder)
    unknown=any(e['status']!='returned' for e in effects) or ((result or {}).get('status')=='raised' and not effects)
    resolved=(folder/'resolution.json').is_file()
    return dict(handle=intent['handle'], operation=intent['operation'], episode=intent.get('episode'),
                     utc=intent['utc'], record=record, status='active/in-flight' if active else (result or {}).get('status','orphaned; effect unknown'),
                     intent_path=str(p), result_path=str(folder/'result.json'),
                     effects=effects,effect_status='reconciled with evidence' if resolved else (result or {}).get('effect_status') or ('unknown' if unknown or result is None else 'returned or failed before native effect'),
                     stage=(result or {}).get('stage'),
                     retention='active/in-flight' if active else 'indexed' if record else 'needs index reconciliation')


def calls(store, episode=None):
    rows = []
    for p in sorted((store.root/'calls').glob('*/intent.json')):
        intent = json.loads(p.read_bytes())
        if episode is not None and intent.get('episode') != episode:
            continue
        rows.append(call_row(store,p,intent))
    return rows


def reconcile(service, handle, observed=None, evidence_paths=None):
    completion_repair=None
    if len(handle)!=32 or any(c not in '0123456789abcdef' for c in handle):
        raise ValueError('Invalid operation handle')
    folder=service.store.root/'calls'/handle
    intent=json.loads((folder/'intent.json').read_bytes())
    row=next(r for r in calls(service.store) if r['handle']==handle)
    if row['status']=='active/in-flight':raise ValueError('Operation owner is still active; do not reconcile running work')
    if observed is not None and (not evidence_paths or observed.get('effect_status') not in ('confirmed_returned','confirmed_not_applied','resolved_failed')):
        raise ValueError('Actual evidence and explicit effect_status required; indexing alone is not effect reconciliation')
    if (folder/'result.json').exists():
        outcome=json.loads((folder/'result.json').read_bytes())
        if observed is not None:
            if not evidence_paths:raise ValueError('Effect reconciliation needs actual native/provider evidence, not a metadata assertion')
            if observed.get('effect_status') not in ('confirmed_returned','confirmed_not_applied','resolved_failed'):
                raise ValueError('Explicit observed effect_status required; indexing is not effect reconciliation')
            resolution=dict(observed=observed,evidence=[service.store.blob(p) for p in evidence_paths],basis='operator observation over actual effect receipts')
            atomic_write(folder/'resolution.json',canonical(resolution))
    else:
        if not observed or not evidence_paths:
            raise ValueError('Missing result requires actual observed outcome and receipt evidence; no replay')
        outcome=dict(status='reconciled observation', observed=observed,
                     evidence=[service.store.blob(p) for p in evidence_paths],
                     automatic_acceptance=False)
        atomic_write(folder/'result.json', canonical(outcome))
    record=service.store.put('operation_fact',dict(intent=intent,outcome=outcome))
    atomic_write(folder/'record.json',canonical({'record':record}))
    if observed is not None:
        if not (folder/'resolution.json').is_file():
            atomic_write(folder/'resolution.json',canonical(dict(observed=observed,evidence=[service.store.blob(p) for p in evidence_paths])))
        lease=intent.get('lease')
        if lease:
            with service.ledger._lock(lease['episode']):
                lease_path=service.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json')
                value=json.loads(lease_path.read_bytes())
                from .leases import status as lease_status
                if lease_status(value,service.store.root)=='active':raise ValueError('Owner operation is active; cannot release its lease')
                value.update(status='finished',effect_resolution=str(folder/'resolution.json'))
                atomic_write(lease_path,canonical(value))
    elif intent.get('lease'):
        lease=intent['lease'];marker=service.store.root/'episode-completions'/lease['episode']/(lease['id']+'.json')
        if marker.is_file():
            completion=json.loads(marker.read_bytes())
            if completion.get('operation_handle')==handle:
                with service.ledger._lock(lease['episode']):
                    lease_path=service.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json')
                    value=json.loads(lease_path.read_bytes());value.update(status=completion['completion_disposition'],operation_handle=handle,result_status=completion.get('result_status'))
                    atomic_write(lease_path,canonical(value))
                completion_repair=dict(path=str(marker),status='existing receipt; lease finalized',disposition=completion['completion_disposition'])
        elif (outcome.get('status')=='returned' or outcome.get('effect_status')=='refused before mutation dispatch') and not any(e['status']!='returned' for e in effect_rows(folder)):
            with service.ledger._lock(lease['episode']):
                lease_path=service.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json')
                value=json.loads(lease_path.read_bytes())
                if (value.get('id')!=lease['id'] or value.get('episode')!=lease['episode']
                        or value.get('operation_handle') not in (None,handle)):
                    raise ValueError('Lease identity differs from the original operation')
                from .leases import mark_return,completion_status
                terminal=dict(outcome.get('result',{}),operation_handle=handle)
                if outcome.get('effect_status')=='refused before mutation dispatch':
                    terminal['effect_status']='refused before mutation dispatch'
                mark_return(service,value,terminal)
                value.update(status=completion_status(terminal),operation_handle=handle,result_status=terminal.get('status'))
                atomic_write(lease_path,canonical(value))
                completion_repair=dict(path=str(marker),status='recovered from original durable result',disposition=value['status'])
    return dict(operation_handle=handle, operation_fact=record, outcome=outcome,
                completion_repair=completion_repair,
                effect_status=next(r['effect_status'] for r in calls(service.store) if r['handle']==handle),
                recovery='Index repaired; effect uncertainty and active leases remain separate. No provider/native operation was repeated')
