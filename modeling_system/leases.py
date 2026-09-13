"""Episode operation leases use the same lock as episode closure."""
import json
import os
import sys
import uuid
from contextvars import ContextVar
from .store import atomic_write,canonical
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
        value=json.loads(p.read_bytes());value['observed_status']=status(value,service.store.root);value['path']=str(p);rows.append(value)
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
                   operation_handle=result.get('operation_handle'),result_status=result.get('status'))
        atomic_write(service.store.root/'episode-leases'/lease['episode']/(lease['id']+'.json'),canonical(value))


def mark_return(service,lease,result):
    """Independent return marker precedes mutable lease finalization."""
    marker=dict(lease_id=lease['id'],episode=lease['episode'],operation_handle=result.get('operation_handle'),
                completion_disposition=completion_status(result),
                result_status=result.get('status'))
    atomic_write(service.store.root/'episode-completions'/lease['episode']/(lease['id']+'.json'),canonical(marker))


def guard_closure(service,episode):
    # Called while holding the episode ledger lock; acquisition cannot race it.
    blocking=[r for r in list_leases(service,episode) if r['observed_status'] not in ('finished','finished; lease finalization pending') and r['id']!=(LEASE.get() or {}).get('id')]
    if blocking:raise Conflict('Episode has active or unresolved operation leases; indexing a timeout does not resolve its effect')
