"""Offline inspection of one recorded execution result and its durable receipt.

Two wrapper mistakes motivate this inspector: requiring a universal "completed"
status plus an inline candidate when the adapter returned an operation-specific
status and a receipt path, and trusting exit zero when the launcher-required
completion file was missing. Here the effect disposition (a supplied adapter
status mapping), artifact bytes, process exit, required-output completeness,
source preservation, completion-manifest verification and appearance are separate
facts. The pinned case supplies the status vocabulary; this module has no default
statuses. Nothing here contacts Blender, replays, retries or clears a failure.
"""
import json
from pathlib import Path
from .repair_analysis import _pin
from .store import digest
from .worker_contract import ContractError, MANIFEST_KIND, bounded_path, file_sha256, is_sha256, relative_name, verify_manifest

EFFECTS = ('applied', 'completed', 'not_applied', 'failed', 'unknown')
SUCCESS = ('applied', 'completed')
KINDS = ('native_operation', 'worker_run')
DISPOSITIONS = ('verified_consistent', 'consistent_unverified', 'incomplete', 'failed', 'needs_reconciliation')
BASIS = 'supplied adapter contract mapping; not independent native proof'
_HEX = frozenset('0123456789abcdef')


def _path(value, name):
    if (not isinstance(value, list) or not 0 < len(value) <= 24
            or any((not isinstance(p, str) or not p) and type(p) is not int for p in value)):
        raise ValueError(name + ' must be a JSON path list of 1..24 nonempty keys or indices')
    return value


def _get(value, path):
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            return False, None
    return True, value


def _join(path):
    return '/'.join(map(str, path))


def _meta(value, name):
    if (not isinstance(value, dict) or not isinstance(value.get('meaning'), str) or not value['meaning']
            or not isinstance(value.get('evidence'), list) or not value['evidence']):
        raise ValueError(name + ' needs explicit meaning and evidence')
    return value


def _reference(value, name):
    if not isinstance(value, dict) or 'asset' not in value or not isinstance(value.get('path'), str):
        raise ValueError(name + ' must be an exact {path, sha256} file reference')
    return value


def _document(service, reference, name):
    if reference is None:
        return None, None
    _reference(reference, name)
    text = service.store.resolve_blob(reference['asset']).read_text(encoding='utf-8-sig')

    def nonfinite(constant):
        raise ValueError(name + ' contains a non-finite JSON constant: ' + constant)
    value = json.loads(text, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise ValueError(name + ' must be a JSON object')
    return value, Path(reference['asset']['source']).parent


def _adapter(value):
    _meta(value, 'adapter')
    if not isinstance(value.get('id'), str) or not value['id']:
        raise ValueError('adapter needs a nonempty id')
    statuses = value.get('statuses')
    if not isinstance(statuses, dict) or not 0 < len(statuses) <= 64:
        raise ValueError('adapter.statuses must map 1..64 recognized status strings')
    for status, row in statuses.items():
        if (not status or not isinstance(row, dict) or row.get('effect') not in EFFECTS
                or not isinstance(row.get('meaning'), str) or not row['meaning']):
            raise ValueError('Each recognized status needs an effect from ' + '/'.join(EFFECTS) + ' and a meaning: ' + str(status))
    paths = value.get('status_path', {})
    if not isinstance(paths, dict) or set(paths) - {'result', 'receipt'}:
        raise ValueError('status_path may name result and receipt only')
    if 'result' in paths:
        _path(paths['result'], 'status_path.result')
    if paths.get('receipt') is not None:
        _path(paths['receipt'], 'status_path.receipt')
    receipt = value.get('receipt', {})
    if not isinstance(receipt, dict) or set(receipt) - {'required', 'reference'}:
        raise ValueError('adapter.receipt accepts required and reference only')
    if 'required' in receipt and type(receipt['required']) is not bool:
        raise ValueError('adapter.receipt.required must be boolean')
    if receipt.get('reference') is not None:
        _path(receipt['reference'], 'adapter.receipt.reference')
    seen = set()
    checks = value.get('identity_checks', [])
    if not isinstance(checks, list) or len(checks) > 64:
        raise ValueError('identity_checks must be a list of at most 64 checks')
    for row in checks:
        if (not isinstance(row, dict) or set(row) - {'id', 'result', 'receipt', 'equals'}
                or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen or row['id'] == 'receipt_reference'):
            raise ValueError('Identity checks need a distinct id plus result/receipt paths or an equals value')
        seen.add(row['id'])
        for side in ('result', 'receipt'):
            if side in row:
                _path(row[side], 'identity_checks.' + row['id'] + '.' + side)
        if not ('result' in row or 'receipt' in row) or ('equals' not in row and not ('result' in row and 'receipt' in row)):
            raise ValueError('Identity check ' + row['id'] + ' needs both result and receipt paths, or one path and equals')
    seen = set()
    artifacts = value.get('artifacts', [])
    if not isinstance(artifacts, list) or len(artifacts) > 64:
        raise ValueError('artifacts must be a list of at most 64 references')
    for row in artifacts:
        if (not isinstance(row, dict) or set(row) - {'id', 'in', 'path', 'required'} or not isinstance(row.get('id'), str)
                or not row['id'] or row['id'] in seen or row.get('in') not in ('result', 'receipt')):
            raise ValueError('Artifact references need a distinct id, in=result|receipt and a JSON path')
        seen.add(row['id'])
        _path(row.get('path'), 'artifacts.' + row['id'] + '.path')
        if 'required' in row and type(row['required']) is not bool:
            raise ValueError('artifacts.required must be boolean: ' + row['id'])
    for key in ('process', 'required_outputs', 'source_preservation'):
        row = value.get(key)
        if row is None:
            continue
        if not isinstance(row, dict) or row.get('in') not in ('result', 'receipt'):
            raise ValueError(key + ' must name its document with in=result|receipt')
        if key == 'process':
            _path(row.get('returncode'), 'process.returncode')
            codes = row.get('success_codes')
            if not isinstance(codes, list) or not codes or any(type(c) is not int for c in codes):
                raise ValueError('process.success_codes must list integer exit codes')
        elif key == 'required_outputs':
            _path(row.get('directory'), 'required_outputs.directory')
            names = row.get('names')
            if not isinstance(names, list) or not 0 < len(names) <= 64 or len(set(names)) != len(names):
                raise ValueError('required_outputs.names must list 1..64 distinct names')
            for name in names:
                relative_name(name, 'Required output name')
        else:
            for field in ('unchanged', 'before', 'after'):
                _path(row.get(field), 'source_preservation.' + field)
    for key in ('deferred', 'record_reference'):
        if value.get(key) is not None:
            _path(value[key], key)
    return value


def _effect(adapter, result, receipt):
    statuses = adapter['statuses']
    paths = adapter.get('status_path', {})
    result_path = paths.get('result', ['status'])
    receipt_path = paths['receipt'] if 'receipt' in paths else ['status']
    labels = dict(result=None, receipt=None)
    reasons = []
    effect = 'unknown'
    found, status = _get(result, result_path)
    if found:
        labels['result'] = status
    if not found:
        reasons.append('result status missing at ' + _join(result_path))
    elif not isinstance(status, str) or status not in statuses:
        reasons.append('unrecognized result status: ' + json.dumps(status))
    else:
        effect = statuses[status]['effect']
        if effect == 'unknown':
            reasons.append('adapter maps status ' + status + ' to an unknown effect')
    if receipt is None:
        if adapter.get('receipt', {}).get('required', True):
            reasons.append('durable receipt not supplied; the compact result alone cannot establish the effect')
            effect = 'unknown'
    elif receipt_path is not None:
        found, receipt_status = _get(receipt, receipt_path)
        if found:
            labels['receipt'] = receipt_status
        if not found:
            reasons.append('receipt status missing at ' + _join(receipt_path))
            effect = 'unknown'
        elif not isinstance(receipt_status, str) or receipt_status not in statuses:
            reasons.append('unrecognized receipt status: ' + json.dumps(receipt_status))
            effect = 'unknown'
        elif effect != 'unknown' and statuses[receipt_status]['effect'] != effect:
            reasons.append('conflicting result/receipt statuses: ' + json.dumps(labels['result']) + ' vs ' + json.dumps(receipt_status))
            effect = 'unknown'
    return effect, labels, reasons


def _receipt_reference(adapter, case, result, result_folder, receipt):
    reference = adapter.get('receipt', {}).get('reference')
    if reference is None:
        return None
    row = dict(id='receipt_reference', kind='file', outcome='missing', detail='result has no receipt reference at ' + _join(reference))
    found, pointer = _get(result, reference)
    if not found or not isinstance(pointer, str) or not pointer:
        return row
    target = Path(pointer)
    target = target if target.is_absolute() else result_folder / target
    row['referenced'] = str(target)
    if receipt is None:
        row['detail'] = 'result references a receipt but the case supplies none'
    elif not target.is_file():
        row['detail'] = 'referenced receipt file is absent now'
    else:
        pinned = Path(case['receipt']['asset']['source'])
        row['supplied'] = str(pinned)
        if target.resolve() != pinned.resolve():
            row.update(outcome='disagree', detail='result references a different receipt file than the supplied one')
        else:
            current = file_sha256(target)
            same = current == case['receipt']['sha256']
            row.update(outcome='agree' if same else 'disagree', current_sha256=current, pinned_sha256=case['receipt']['sha256'],
                       detail='referenced receipt is the supplied receipt' if same else 'referenced receipt bytes changed since the case pinned it')
    return row


def _identity(adapter, docs):
    rows = []
    for check in adapter.get('identity_checks', []):
        values, missing = {}, []
        for side in ('result', 'receipt'):
            if side not in check:
                continue
            document = docs[side][0]
            if document is None:
                missing.append(side + ' document absent')
                continue
            found, value = _get(document, check[side])
            if found:
                values[side] = value
            else:
                missing.append(side + ' lacks ' + _join(check[side]))
        if missing:
            outcome = 'missing'
        elif 'equals' in check:
            outcome = 'agree' if all(v == check['equals'] for v in values.values()) else 'disagree'
        else:
            outcome = 'agree' if values['result'] == values['receipt'] else 'disagree'
        rows.append(dict(id=check['id'], kind='value', outcome=outcome, values=values, expected=check.get('equals'), missing=missing))
    return rows


def _artifacts(adapter, docs):
    rows = []
    for spec in adapter.get('artifacts', []):
        document, folder = docs[spec['in']]
        row = dict(id=spec['id'], document=spec['in'], path=spec['path'], required=spec.get('required', True), status='unreferenced')
        if document is None:
            row['reason'] = spec['in'] + ' document absent'
        else:
            found, ref = _get(document, spec['path'])
            if not found:
                row['reason'] = 'no reference at ' + _join(spec['path'])
            elif not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not ref['path'] or not is_sha256(ref.get('sha256')):
                row.update(status='invalid_reference', reason='reference is not an exact {path, sha256}')
            else:
                target = Path(ref['path'])
                target = target if target.is_absolute() else folder / target
                row.update(file=str(target), claimed_sha256=ref['sha256'], role=ref.get('role'), claimed_bytes=ref.get('bytes'))
                if not target.is_file():
                    row['status'] = 'missing'
                else:
                    actual = file_sha256(target)
                    row.update(actual_sha256=actual, bytes=target.stat().st_size, status='verified' if actual == ref['sha256'] else 'mismatch')
        rows.append(row)
    return rows


def _process(adapter, docs):
    spec = adapter.get('process')
    if not spec:
        return dict(disposition='not_applicable')
    document = docs[spec['in']][0]
    found, code = _get(document, spec['returncode']) if document is not None else (False, None)
    if not found or type(code) is not int:
        return dict(disposition='unknown', document=spec['in'], returncode=code if found else None, reason='exit code missing or not an integer')
    return dict(disposition='exit_ok' if code in spec['success_codes'] else 'exit_failed', document=spec['in'],
                returncode=code, success_codes=spec['success_codes'])


def _required_outputs(adapter, docs):
    spec = adapter.get('required_outputs')
    if not spec:
        return dict(disposition='not_applicable'), []
    document, folder = docs[spec['in']]
    found, directory = _get(document, spec['directory']) if document is not None else (False, None)
    if not found or not isinstance(directory, str) or not directory:
        return dict(disposition='unknown', reason='output directory missing from the document', names=spec['names']), []
    root = Path(directory)
    root = root if root.is_absolute() else folder / root
    if not root.is_dir():
        return dict(disposition='unknown', reason='output directory absent now', directory=str(root), names=spec['names']), []
    root = root.resolve()
    present, missing, files, candidates = [], [], {}, []
    for name in spec['names']:
        try:
            target = bounded_path(root, name, 'Required output name')
        except ContractError as error:
            files[name] = dict(status='refused', reason=str(error))
            missing.append(name)
            continue
        if target.is_file():
            present.append(name)
            files[name] = dict(status='present', sha256=file_sha256(target), bytes=target.stat().st_size)
            candidates.append(target)
        else:
            missing.append(name)
            files[name] = dict(status='missing')
    return dict(disposition='complete' if not missing else 'incomplete', directory=str(root), present=present,
                missing=missing, files=files), candidates


def _source_preservation(adapter, docs):
    spec = adapter.get('source_preservation')
    if not spec:
        return dict(disposition='not_applicable')
    document = docs[spec['in']][0]
    fields = {k: (_get(document, spec[k]) if document is not None else (False, None)) for k in ('unchanged', 'before', 'after')}
    if not any(found for found, _ in fields.values()):
        return dict(disposition='unknown', reason='no source preservation fields in the document')
    unchanged, before, after = (fields[k][1] if fields[k][0] else None for k in ('unchanged', 'before', 'after'))
    if fields['before'][0] and fields['after'][0] and before is None and after is None:
        return dict(disposition='not_applicable', unchanged=unchanged,
                    reason='the launcher recorded no source input; worker-declared protected sources are verified only through a completion manifest')
    if unchanged is True and is_sha256(before) and before == after:
        return dict(disposition='unchanged', sha256=before)
    if unchanged is False or (isinstance(before, str) and isinstance(after, str) and before != after):
        return dict(disposition='changed', before=before, after=after, unchanged=unchanged)
    return dict(disposition='unknown', reason='inconsistent or partial source preservation fields', before=before, after=after, unchanged=unchanged)


def _manifest(case, receipt, candidates):
    if receipt is not None and receipt.get('kind') == MANIFEST_KIND:
        return dict(verify_manifest(Path(case['receipt']['asset']['source'])), source='receipt')
    for target in candidates:
        probe = verify_manifest(target)
        if probe['disposition'] not in ('absent', 'invalid', 'not_contract_manifest'):
            return dict(probe, source='required_output')
    if candidates:
        return dict(disposition='not_contract_manifest', source='required_output',
                    reason='present required outputs are not worker completion manifests; their content is unverified here')
    return dict(disposition='absent', source=None)


def _record(service, adapter, result):
    if adapter.get('record_reference') is None:
        return None
    found, key = _get(result, adapter['record_reference'])
    if found and isinstance(key, str) and len(key) == 64 and set(key) <= _HEX:
        present = (service.store.root / 'records' / (key + '.json')).is_file()
        return dict(key=key, status='present_in_this_store' if present else 'absent_from_this_store')
    return dict(key=key if found else None, status='invalid_or_missing')


def inspect(service, case_path):
    """Retain one execution result/receipt pair with separate dispositions; never replay, retry or accept."""
    path = Path(case_path).resolve()
    original = service.store.blob(path)
    raw = json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig'))
    try:
        case = _pin(service, raw, path.parent, {})
    except ValueError as error:
        raise ValueError('Execution receipt case reference changed: ' + str(error)) from error
    if case.get('schema_version') != 1 or not isinstance(case.get('question'), str) or not case['question']:
        raise ValueError('Version-1 execution receipt case with a question required')
    operation = _meta(case.get('operation'), 'operation')
    if operation.get('kind') not in KINDS or not isinstance(operation.get('name'), str) or not operation['name']:
        raise ValueError('operation needs kind native_operation|worker_run and a name')
    adapter = _adapter(case.get('adapter'))
    if 'result' not in case:
        raise ValueError('result file reference required')
    result, result_folder = _document(service, case['result'], 'result')
    receipt, receipt_folder = _document(service, case.get('receipt'), 'receipt')
    verification = case.get('independent_verification', [])
    if (not isinstance(verification, list) or len(verification) > 64
            or any(not isinstance(v, dict) or 'asset' not in v or not isinstance(v.get('role'), str) or not v['role'] for v in verification)):
        raise ValueError('independent_verification entries need an exact {path, sha256} reference and a role')
    docs = {'result': (result, result_folder), 'receipt': (receipt, receipt_folder)}

    effect, labels, reasons = _effect(adapter, result, receipt)
    identity = []
    reference = _receipt_reference(adapter, case, result, result_folder, receipt)
    if reference is not None:
        identity.append(reference)
    identity.extend(_identity(adapter, docs))
    for row in identity:
        if row['outcome'] != 'agree' and effect != 'unknown':
            reasons.append('identity check ' + row['id'] + ' ' + row['outcome'])
            effect = 'unknown'
    artifacts = _artifacts(adapter, docs)
    process = _process(adapter, docs)
    outputs, candidates = _required_outputs(adapter, docs)
    source = _source_preservation(adapter, docs)
    manifest = _manifest(case, receipt, candidates)
    record = _record(service, adapter, result)
    deferred = []
    if adapter.get('deferred') is not None:
        found, value = _get(result, adapter['deferred'])
        if found and isinstance(value, dict):
            deferred = sorted(value)
        elif found and isinstance(value, list):
            deferred = [str(v) for v in value]
    verification_rows = [dict(role=v['role'], path=v['path'], sha256=v['sha256'], asset=v['asset'], meaning=v.get('meaning'),
                              bearing='retained separately; does not alter the original result/receipt disposition') for v in verification]

    verified = [a for a in artifacts if a['status'] == 'verified']
    mismatched = [a for a in artifacts if a['status'] == 'mismatch']
    missing_artifacts = [a for a in artifacts if a['required'] and a['status'] in ('missing', 'unreferenced', 'invalid_reference')]
    disagreements = [i for i in identity if i['outcome'] != 'agree']
    contradictions, gaps = [], []
    if effect == 'unknown':
        contradictions.append('effect unknown: ' + '; '.join(reasons))
    if mismatched:
        contradictions.append('artifact bytes differ from the receipt reference: ' + ', '.join(a['id'] for a in mismatched))
    if disagreements:
        contradictions.append('identity checks not agreeing: ' + ', '.join(i['id'] for i in disagreements))
    if source['disposition'] == 'changed':
        (contradictions if effect in SUCCESS else gaps).append('pinned source changed during the run')
    if manifest['disposition'] in ('mismatch', 'invalid'):
        contradictions.append('completion manifest does not verify now')
    if effect in SUCCESS and process['disposition'] == 'exit_failed':
        contradictions.append('process exit failed despite a success status')
    if effect in SUCCESS and process['disposition'] == 'unknown':
        contradictions.append('process exit unknown')
    if outputs['disposition'] == 'incomplete':
        gaps.append('required outputs missing: ' + ', '.join(outputs['missing']))
    if outputs['disposition']=='unknown':gaps.append('required-output completeness is unknown')
    if source['disposition']=='unknown':gaps.append('configured source-preservation evidence is unknown')
    if missing_artifacts:
        gaps.append('required artifacts not verifiable: ' + ', '.join(a['id'] for a in missing_artifacts))
    if effect in SUCCESS and not verified and manifest['disposition'] != 'verified':
        gaps.append('no artifact bytes or completion manifest verified; the success status is a supplied mapping only')
    if contradictions:
        disposition = 'needs_reconciliation'
    elif effect in SUCCESS:
        if outputs['disposition'] in ('incomplete','unknown') or source['disposition']=='unknown' or missing_artifacts:
            disposition = 'incomplete'
        elif verified or manifest['disposition'] == 'verified':
            disposition = 'verified_consistent'
        else:
            disposition = 'consistent_unverified'
    elif process['disposition'] == 'exit_ok' and outputs['disposition'] == 'incomplete':
        disposition = 'incomplete'
    else:
        disposition = 'failed'
    success = disposition == 'verified_consistent'

    actions = []
    if effect == 'unknown':
        actions.append('Reconcile the original operation through its owner and journal with the actual receipts; an unknown or conflicting effect never authorizes replay or retry.')
    if mismatched:
        actions.append('Treat the receipt as stale for the mismatched artifacts; locate the original bytes or reconcile before any use.')
    if disagreements:
        actions.append('Resolve the identity disagreement between the compact result and the receipt before trusting either document.')
    if outputs['disposition'] == 'incomplete':
        actions.append('The launcher-required output is missing although the process exited successfully: keep this receipt as the original classification, verify the produced files independently in a separate linked record, and adopt a completion contract in the worker. Do not fabricate the missing file or rerun native effects merely to clear the classification.')
    if missing_artifacts:
        actions.append('Required artifact references are absent or unreadable; expand the receipt or record that carries them before treating the operation as verified.')
    if source['disposition'] == 'changed':
        actions.append('The pinned source changed during the run; this run is not a preserved-source experiment.')
    if manifest['disposition'] in ('mismatch', 'invalid'):
        actions.append('Completion manifest artifacts, sources or checks no longer verify; reconcile the output directory before reuse.')
    if disposition == 'verified_consistent':
        actions.append('Proceed to independent reopen, comparison and review; verified bytes and a mapped status are not appearance acceptance or native readiness.')
    if disposition == 'consistent_unverified':
        actions.append('Declare artifact references or a completion manifest so success can be verified from bytes rather than a status label.')
    if disposition == 'failed':
        actions.append('The operation is classified failed by its own contract; inspect its log and receipt, then decide any new operation explicitly. Nothing here clears the failure.')
    if deferred:
        actions.append('Fields ' + ', '.join(deferred) + ' are deferred to the receipt/record by design; their absence inline is not a failure.')

    summary = dict(disposition=disposition, effect=effect, effect_basis=BASIS, result_status=labels['result'], receipt_status=labels['receipt'],
                   receipt_supplied=receipt is not None,
                   artifacts=dict(declared=len(artifacts), verified=len(verified), mismatch=len(mismatched),
                                  missing=sum(a['status'] == 'missing' for a in artifacts),
                                  unreferenced=sum(a['status'] in ('unreferenced', 'invalid_reference') for a in artifacts)),
                   identity=dict(agree=sum(i['outcome'] == 'agree' for i in identity), disagree=sum(i['outcome'] == 'disagree' for i in identity),
                                 missing=sum(i['outcome'] == 'missing' for i in identity)),
                   process=process['disposition'], required_outputs=outputs['disposition'], source_preservation=source['disposition'],
                   completion_manifest=manifest['disposition'], independent_verification=len(verification_rows),
                   deferred_fields=deferred, success_established=success)
    limits = ['Effect disposition comes from the supplied adapter status mapping; it is not independent native proof.',
              'Verified artifact bytes establish that referenced files match the receipt now, not that their content is correct or accepted.',
              'Process exit, required-output completeness, source preservation and completion manifest are separate; none substitutes for another.',
              'An unknown or conflicting effect requires reconciliation of the original operation; inspection alone never makes replay safe.',
              'Independent verification records are retained separately and never alter the original receipt or its classification.',
              'Appearance acceptance and native readiness are never granted here.']
    payload = dict(schema_version=1, question=case['question'], operation=operation, case_source=original, case=case,
                   documents=dict(result=result, receipt=receipt),
                   effect=dict(disposition=effect, statuses=labels, reasons=reasons, basis=BASIS, independent_proof=False),
                   identity=identity, artifacts=artifacts, process=process, required_outputs=outputs, source_preservation=source,
                   completion_manifest=manifest, record=record, deferred_fields=deferred, independent_verification=verification_rows,
                   contradictions=contradictions, gaps=gaps, summary=summary, next_actions=actions, limits=limits,
                   disposition=disposition, success_established=success, safe_to_replay=False, native_ready=False, appearance_accepted=False)
    delivery=None
    if case.get('delivery') is not None:
        from .delivery import inspect as inspect_delivery
        delivery=inspect_delivery(service,case['delivery'])
        payload['delivery']=delivery
        summary['delivery']=dict(status=delivery['status'],current_revision=delivery['current_revision'],
            missing_between_anchor_objects=delivery['missing_between_anchor_objects'])
    key = service.store.put('execution_receipt', payload)
    reads = {k: dict(operation='read_record', arguments=dict(record=key, path=[k], limit=10, max_chars=8000))
             for k in ('documents', 'effect', 'identity', 'artifacts', 'process', 'required_outputs', 'source_preservation',
                       'completion_manifest', 'independent_verification', 'case', 'limits')}
    if record is not None and record['status'] == 'present_in_this_store':
        reads['operation_record'] = dict(operation='read_record', arguments=dict(record=record['key'], path=[], limit=10, max_chars=8000))
    if delivery is not None:reads['delivery']=dict(operation='read_record',arguments=dict(record=key,path=['delivery'],max_chars=8000))
    return dict(analysis=key, status='diagnostic_only', disposition=disposition, success_established=success, summary=summary,
                links=dict(result=case['result']['asset'], receipt=case['receipt']['asset'] if receipt is not None else None,
                           independent_verification=[v['asset'] for v in verification]),
                contradictions=contradictions, gaps=gaps, next_actions=actions,
                safe_to_replay=False, native_ready=False, appearance_accepted=False, reads=reads)
