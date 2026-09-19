"""Small, explicitly selected recovery manifest over immutable receipts."""
import json
import os
from .store import atomic_write, canonical, native_path
from .worker_contract import file_sha256
from .ledger import Conflict


def file_status(ref):
    if not isinstance(ref,dict) or not isinstance(ref.get('file'),str) or not isinstance(ref.get('sha256'),str) or len(ref['sha256'])!=64:
        raise ValueError('Checkpoint/artifact reference requires file and sha256; labels do not select content')
    path=native_path(ref['file'])
    try:actual=file_sha256(path)
    except OSError:return dict(ref,status='unavailable')
    return dict(ref,status='verified' if actual==ref['sha256'] else 'changed',observed_sha256=actual)


def select(service, manifest, expected):
    if not isinstance(manifest,dict):raise ValueError('recovery must be an object')
    required=('retained','baseline','active_experiment','native_owner','preview','selected_runtime')
    if any(k not in manifest for k in required):raise ValueError('recovery requires '+', '.join(required))
    for k in ('retained','baseline'):
        if file_status(manifest[k])['status']!='verified':raise ValueError('Recovery '+k+' bytes differ from declared checkpoint')
    if not manifest['native_owner']:raise ValueError('Recovery native_owner must be explicit')
    preview=manifest['preview']
    if preview.get('status') not in ('retained','trial','rejected','unknown'):
        raise ValueError('Preview status must be retained, trial, rejected or unknown')
    if preview['status'] in ('trial','rejected'):
        restore=preview.get('visibility_restore')
        if not restore or file_status(restore)['status']!='verified':raise ValueError('Trial/rejected preview needs hashed visibility_restore receipt for the sole native owner')
    selected=manifest['selected_runtime']
    if not isinstance(selected,dict) or not selected.get('python') or not selected.get('loaded_revision'):
        raise ValueError('Selected runtime needs actual python and loaded_revision')
    key=service.store.put('recovery_manifest',dict(manifest,live_freshness='not checked; owner must reconcile current Blender',
        message_status='No peer delivery inferred from dispatch or a saved manifest'))
    path=service.store.root/'recovery.json';lock=path.with_suffix('.lock')
    try:
        with lock.open('x') as f:f.write(str(os.getpid()))
    except FileExistsError as error:raise Conflict('Recovery selection is being updated; inspect the owner') from error
    try:
        prior=json.loads(path.read_bytes())['record'] if path.exists() else None
        if prior!=expected:raise Conflict('Recovery selection changed; inspect_situation and preserve the newer manifest')
        atomic_write(path,canonical(dict(record=key,previous=prior)))
    finally:lock.unlink(missing_ok=True)
    return key


def inspect(service):
    path=service.store.root/'recovery.json'
    runtime=service.runtime_status()
    current=dict(loaded_revision=runtime['loaded']['revision'],python=runtime['python'],
        source_matches_loaded=runtime['source_matches_loaded'],native_adapter=service.binding.get('native_adapter','unconfigured'),
        native_status='not contacted',other_processes='MCP/cache/operator uptake is not inferred from this process')
    if not path.exists():return dict(status='unselected',runtime=current,base_selection='No filename or latest timestamp selects a retained character')
    key=json.loads(path.read_bytes())['record'];manifest=service.store.get(key,'recovery_manifest')
    checks={k:file_status(manifest[k]) for k in ('retained','baseline')}
    if manifest['preview'].get('visibility_restore'):
        checks['visibility_restore']=file_status(manifest['preview']['visibility_restore'])
    return dict(record=key,status='recorded' if all(r['status']=='verified' for r in checks.values()) else 'needs_reconciliation',
        checkpoints=checks,active_experiment=manifest['active_experiment'],native_owner=manifest['native_owner'],
        preview=manifest['preview'],runtime=current,selected_runtime=manifest['selected_runtime'],
        selected_runtime_matches_process=(manifest['selected_runtime']['loaded_revision']==current['loaded_revision']
            and native_path(manifest['selected_runtime']['python'])==native_path(current['python'])),
        live_freshness=manifest['live_freshness'],message_status=manifest['message_status'])
