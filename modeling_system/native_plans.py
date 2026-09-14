"""Retained public preparation for owner-run isolated native helpers."""
import json
from pathlib import Path
from .mesh_plan import compile_plan
from .library_plan import preflight
from .store import digest
from .bounded_reads import read_descriptor


def _expansions(key):
    descriptor=read_descriptor('read_record',dict(record=key),['plan'],max_chars=8000)
    return dict(reads={'plan':descriptor},detail=descriptor)


def _read(service,path):
    source=service.store.blob(Path(path).resolve())
    value=json.loads(service.store.resolve_blob(source).read_text(encoding='utf-8-sig'))
    return source,value


def mesh(service,case_path):
    source,case=_read(service,case_path)
    if case.get('schema_version')!=1 or not case.get('object') or not case.get('question'):raise ValueError('Version-1 named-object batch case and question required')
    result=compile_plan(case['mesh'],case['steps'],case.get('aliases'))
    key=service.store.put('native_mesh_plan',dict(source=source,case=case,plan=result))
    return dict(plan=key,status='preflighted_offline',native_ready=False,object=case['object'],question=case['question'],
        before_revision=result['before_revision'],after_revision=result['after_revision'],steps=len(result['steps']),
        next_action='Sole native owner must compare the actual raw mesh and expected SaveBoundary, then invoke native_mesh_batch.execute_batch once in an isolated process.',
        **_expansions(key))


def library(service,manifest_path,existing,namespace,limits=None):
    source,manifest=_read(service,manifest_path);plan=preflight(manifest,existing,namespace,limits)
    key=service.store.put('native_library_plan',dict(source=source,manifest=manifest,plan=plan,declared_destination_ids=existing))
    return dict(plan=key,status='preflighted_offline',native_ready=False,counts=plan['counts'],manifest_revision=plan['manifest_revision'],
        next_action='Sole native owner must verify source, adapters and actual destination; native_library.append repeats preflight under its SaveBoundary.',
        **_expansions(key))


def inspect_transaction(receipt_path,expected_sha256):
    path=Path(receipt_path).resolve();raw=path.read_bytes()
    if len(raw)>2000000 or digest(raw)!=expected_sha256:raise ValueError('Native transaction receipt bytes differ from expected bounded receipt')
    receipt=json.loads(raw)
    if receipt.get('schema_version')!=1 or receipt.get('status') not in ('prepared','applying','step_completed','batch_applied','partial_or_uncertain','rollback_opening','rolled_back','rollback_verification_failed','rollback_uncertain'):raise ValueError('Unsupported native transaction receipt')
    artifacts={}
    for name in ('plan','manifest','rollback','candidate','partial','abandoned'):
        ref=receipt.get(name)
        if ref:
            p=Path(ref['path']);matches=p.is_file() and digest(p.read_bytes())==ref['sha256']
            artifacts[name]=dict(reference=ref,bytes_verified=matches)
    complete=receipt['status']=='batch_applied'
    return dict(status=receipt['status'],owner=receipt['owner'],receipt_sha256=expected_sha256,
        applied_steps=[s['id'] for s in receipt.get('applied_steps',[])],pending_step=receipt.get('pending_step'),pending_stage=receipt.get('pending_stage'),
        error=receipt.get('error'),artifacts=artifacts,replay_allowed=False,automatic_rollback_allowed=False,
        next_action='Owner may test the exact current SaveBoundary against the original after token before rollback.' if complete else 'Preserve partial effects; compare the current scene and original checkpoints in a separately identified owner recovery. Do not replay this batch.',
        live_freshness='Not checked; historical artifact inspection does not authorize rollback',appearance_accepted=False)
