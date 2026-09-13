"""Episode operation leases use the same lock as episode closure."""
import json
import os
import sys
import uuid
from contextvars import ContextVar
from .store import atomic_write,canonical,digest
from .ledger import Conflict

LEASE = ContextVar('modeling_episode_lease',default=None)


def completion_status(result):
    return 'finished' if result.get('effect_status')=='refused before mutation dispatch' or result.get('status') not in ('failed','needs attention','conflicting') else 'needs effect reconciliation'


def process_identity(pid=None):
    pid=pid or os.getpid()
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
        api.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        api.GetProcessTimes.argtypes=[wintypes.HANDLE,*([ctypes.POINTER(wintypes.FILETIME)]*4)]
        api.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:return None
        try:
            code=wintypes.DWORD()
            if not api.GetExitCodeProcess(handle,ctypes.byref(code)) or code.value!=259:return None
            times=[wintypes.FILETIME() for _ in range(4)]
            if not api.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return {'pid':pid,'start':'unknown'}
            return {'pid':pid,'start':(times[0].dwHighDateTime<<32)+times[0].dwLowDateTime}
        finally:api.CloseHandle(handle)
    try:
        raw=open('/proc/'+str(pid)+'/stat').read()
        return {'pid':pid,'start':raw[raw.rfind(')')+2:].split()[19]}
    except (OSError,IndexError):
        if pid==os.getpid():return {'pid':pid,'start':'current process'}
        return None


def status(value,root=None):
    if root is not None and value['status']=='active':
        handle=value.get('operation_handle')
        returned=root/'calls'/handle/'result.json' if handle else None
        if returned and returned.is_file():
            outcome=json.loads(returned.read_bytes())
            ok=outcome.get('effect_status')=='refused before mutation dispatch' or (outcome.get('status')=='returned' and completion_status(outcome.get('result',{}))=='finished')
            return 'finished; lease finalization pending' if ok else 'needs effect reconciliation'
        marker=root/'episode-completions'/value['episode']/(value['id']+'.json')
        if marker.is_file():
            completion=json.loads(marker.read_bytes())
            return 'finished; lease finalization pending' if completion['completion_disposition']=='finished' else 'needs effect reconciliation'
    if value['status']=='active' and process_identity(value['process']['pid'])!=value['process']:
        return 'orphaned; effect unknown'
    return value['status']


def list_leases(service,episode):
    rows=[]
    for p in (service.store.root/'episode-leases'/episode).glob('*.json'):
        raw=p.read_bytes();value=json.loads(raw);value['lease_revision']=digest(raw)
        value['observed_status']=status(value,service.store.root);value['path']=str(p);rows.append(value)
    return rows


def acquire(service,episode,operation):
    with service.ledger._lock(episode):
        item=service.ledger.read(episode)
        if item['kind']!='decision_episode' or item['status']!='active':raise Conflict('Episode is not active; reconcile/reopen explicitly')
        value=dict(id=uuid.uuid4().hex,operation_handle=uuid.uuid4().hex,episode=episode,episode_revision=item['revision'],operation=operation,status='active',process=process_identity())
        path=service.store.root/'episode-leases'/episode/(value['id']+'.json')
        atomic_write(path,canonical(value))
        return value


def finish(service,lease,result):
    with service.ledger._lock(lease['episode']):
        value=dict(lease,status=completion_status(result),
                   operation_handle=result.get('operation_handle') or lease.get('operation_handle'),result_status=result.get('status'))
        atomic_write(service.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json'),canonical(value))


def mark_return(service,lease,result):
    """Independent return marker precedes mutable lease finalization."""
    marker=dict(lease_id=lease['id'],episode=lease['episode'],operation_handle=result.get('operation_handle') or lease.get('operation_handle'),
                completion_disposition=completion_status(result),
                result_status=result.get('status'))
    atomic_write(service.store.root/'episode-completions'/lease['episode']/(lease['id']+'.json'),canonical(marker))


def guard_closure(service,episode):
    # Called while holding the episode ledger lock; acquisition cannot race it.
    blocking=[r for r in list_leases(service,episode) if r['observed_status'] not in ('finished','finished; lease finalization pending') and r['id']!=(LEASE.get() or {}).get('id')]
    if blocking:raise Conflict('Episode has active or unresolved operation leases; indexing a timeout does not resolve its effect')


def reconcile_unlinked(service,episode,handle,expected,observed,evidence_paths):
    """Narrow legacy recovery: terminal reconciliation call rejected at Python binding.

    Absence of a handle, a dead process or a later good scene is never proof of
    no effect. Require the immutable original refusal plus the operator's exact
    lease/receipt association. Other ambiguous orphan kinds remain blocked.
    """
    from .operation_reads import identifier
    import inspect
    identifier(episode,64,'episode');identifier(handle,32,'lease handle')
    identifier(expected,64,'expected lease revision')
    if not observed or observed.get('effect_status')!='confirmed_not_applied' or observed.get('lease_id')!=handle or not observed.get('basis') or not evidence_paths:
        raise ValueError('Require confirmed_not_applied, exact lease_id, receipt-association basis and original evidence paths')
    receipt=service.store.get(observed.get('refusal_record',''),'operation')
    if receipt.get('operation')!='reconcile_operation' or receipt.get('status')!='failed' or receipt.get('error_type')!='TypeError' or receipt.get('operation_handle'):
        raise ValueError('Require the original unlinked reconcile_operation argument-refusal record')
    arguments=receipt.get('arguments',{})
    # Check a real binding failure, not any arbitrary TypeError from inside a
    # callable body. Exact old Python refusal text remains in the original record.
    try:inspect.signature(service.reconcile_operation).bind(**arguments)
    except TypeError as error:
        detail=str(error)
    else:raise ValueError('Arguments bind successfully; no proof of pre-invocation refusal')
    if not receipt.get('summary','').endswith(detail):
        raise ValueError('Original refusal does not match the argument-binding failure')
    with service.ledger._lock(episode):
        path=service.store.root/'episode-leases'/episode/(handle+'.json')
        raw=path.read_bytes();value=json.loads(raw)
        if digest(raw)!=expected:raise Conflict('Lease changed; reread before reconciliation')
        if value.get('id')!=handle or value.get('episode')!=episode:raise ValueError('Lease identity mismatch')
        if value.get('operation')!='reconcile_operation' or value.get('operation_handle') or value.get('status')!='needs effect reconciliation' or value.get('result_status')!='failed':
            raise ValueError('Only a terminal unlinked reconciliation refusal is supported; active or ambiguous leases remain blocked')
        marker=service.store.root/'episode-completions'/episode/(handle+'.json')
        completion=json.loads(marker.read_bytes())
        if completion.get('lease_id')!=handle or completion.get('episode')!=episode or completion.get('operation_handle') or completion.get('result_status')!='failed' or completion.get('completion_disposition')!='needs effect reconciliation':
            raise ValueError('Original terminal return marker does not match this unlinked failure')
        for p in (service.store.root/'calls').glob('*/intent.json'):
            intent=json.loads(p.read_bytes())
            if (intent.get('lease') or {}).get('id')==handle:
                raise ValueError('A callable intent exists; reconcile its operation handle instead')
        resolution=dict(lease_id=handle,prior_lease_revision=expected,observed=observed,
            evidence=[service.store.blob(p) for p in evidence_paths],refusal_record=observed['refusal_record'],
            basis='Verified terminal argument refusal; exact lease association attested by operator from original evidence',
            native_or_provider_replayed=False)
        key=service.store.put('lease_resolution',resolution)
        value.update(status='finished',reconciliation=resolution,resolution_record=key)
        atomic_write(path,canonical(value))
    return dict(lease_id=handle,episode=episode,lease_revision=digest(canonical(value)),resolution=key,
        effect_status='refused before mutation dispatch',status='completed',
        recovery='Original refusal and completion marker preserved; only this lease finalized, no operation replayed')
