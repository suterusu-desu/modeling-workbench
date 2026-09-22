"""Parameterized controller handlers for the qualified private native lane.

No Blender access occurs on import or construction. Calls belong to an
OperatingSession; the private adapter still enforces owner and expected state.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from .controller import write_json


def checked_file(reference):
    path = Path(reference['path']).resolve(strict=True)
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != reference['sha256']:
        raise ValueError('Pinned native recipe input changed')
    return path


def expanded_result(service, result):
    """Expand the exact returned record locally, without contacting Blender again."""
    if result.get('detail_available'):
        full = service.store.get(result['operation_record'], 'operation')
        return {**full, **{k: v for k, v in result.items() if k not in ('expanded_fields', 'detail_available')}}
    return result


def returned_live(service, result):
    result = expanded_result(service, result)
    live = result.get('live', result)
    if not isinstance(live, dict) or not live.get('expected_state') or not live.get('file'):
        raise ValueError('Native operation did not retain a fresh live-state result')
    return live


def _clean_source(live, owner, source):
    path = checked_file(source)
    if (live.get('owner') != owner or live.get('dirty') is not False
            or Path(live['file']).resolve() != path
            or live.get('saved_file', {}).get('sha256') != source['sha256']):
        raise ValueError('Live owner, clean checkpoint or saved source changed; preserve later user work')


def _bound(item, references):
    if any(ref['sha256'] not in item['reads'].values() for ref in references):
        raise ValueError('Every recipe source and implementation must be a bound task dependency')
    for ref in references:
        checked_file(ref)


def _evidence(path, role):
    return {'kind': 'file', 'path': str(path), 'role': role}


def validate_native_job(item):
    """Read-only runner contract, usable before offering a job to Jev.

    This checks fixed configuration only. Actual file hashes, live owner and
    expected state remain checked at execution; no Blender call or output occurs.
    """
    payload = item.get('payload', {})
    job = payload.get('job')
    if not item.get('workbench', {}).get('native') or not isinstance(job, dict):
        raise ValueError('Native job needs its explicit native contract and job dictionary')
    for name in ('blender', 'input', 'script', 'output_root'):
        if not isinstance(job.get(name), str) or not job[name].strip():
            raise ValueError('Native runner requires a nonempty ' + name + ' path')
    if Path(job['input']).suffix.lower() != '.blend':
        raise ValueError('Native runner input must be a Blender checkpoint')
    if not re.fullmatch(r'[0-9a-f]{64}', str(job.get('source_sha256', ''))):
        raise ValueError('Native runner requires its exact source_sha256')
    dependencies = job.get('dependency_hashes')
    if (not isinstance(dependencies, dict) or job['script'] not in dependencies
            or any(not isinstance(p, str) or not p or not re.fullmatch(r'[0-9a-f]{64}', str(h))
                   for p, h in dependencies.items())):
        raise ValueError('Native script and dependency_hashes must be explicitly pinned')
    for name in ('runner', 'live_source'):
        ref = payload.get(name)
        if (not isinstance(ref, dict) or not isinstance(ref.get('path'), str) or not ref['path']
                or not re.fullmatch(r'[0-9a-f]{64}', str(ref.get('sha256', '')))):
            raise ValueError('Native job requires an exact ' + name + ' file reference')
    timeout = job.get('timeout_seconds', 240)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Native timeout_seconds must be finite and positive')
    threads = job.get('threads', 2)
    if type(threads) is not int or threads < 0:
        raise ValueError('Native threads must be a nonnegative integer')


class NativeJob:
    """Run a pinned isolated job with data parameters, without copied wrapper code.

    Payload: job (existing qualified runner schema), runner reference, live_source
    reference. The runner, script, input and all job dependency_hashes are bound
    task reads. Output classification stays with the runner's actual receipt.
    """
    def __init__(self, service, directory, *, python=None):
        self.service, self.directory = service, Path(directory)
        self.python = str(python or sys.executable)

    preflight = staticmethod(validate_native_job)

    def __call__(self, item, context):
        self.preflight(item)
        payload = item['payload']; job = deepcopy(payload['job'])
        if not item['workbench'].get('native'):
            raise ValueError('Native job requires an explicit native capability')
        refs = [payload['runner'], payload['live_source'],
                {'path': job['input'], 'sha256': job['source_sha256']},
                *({'path': p, 'sha256': sha} for p, sha in job['dependency_hashes'].items())]
        if str(job['script']) not in job['dependency_hashes']:
            raise ValueError('Native script must be pinned in dependency_hashes')
        _bound(item, refs)
        if not item['id'] or Path(item['id']).name != item['id']:
            raise ValueError('A single local task name required')
        folder = self.directory / item['id']
        folder.mkdir(parents=True, exist_ok=False)
        live = self.service.execute('native_inspect_live', {'owner': context['owner'], 'refresh_scene': False})
        write_json(folder/'live.json', live)
        _clean_source(returned_live(self.service, live), context['owner'], payload['live_source'])
        write_json(folder/'job.json', job)
        log = folder/'native.log'
        started = time.perf_counter()
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.Popen([self.python, '-I', str(checked_file(payload['runner'])), str(folder/'job.json')],
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            write_json(folder/'dispatch.json', {'pid': process.pid, 'attempt_key': context['attempt_key'],
                       'status': 'running', 'job': str(folder/'job.json')})
            cancelled = False
            while process.poll() is None:
                if not cancelled:
                    cancelled = context['cancelled'].wait(1)
                else:
                    time.sleep(1)
                # The isolated runner owns its bounded child lifetime. Do not kill
                # only the parent and strand a still-writing Blender process.
                context['progress'](message='Finishing isolated native job' if cancelled else item['description'])
        rows = []
        for line in log.read_text(encoding='utf-8').splitlines():
            if line.startswith('{'):
                try: rows.append(json.loads(line))
                except ValueError: pass
        terminal = rows[-1] if rows else {}
        output = terminal.get('output')
        receipt_path = Path(output)/'receipt.json' if output else None
        receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path and receipt_path.is_file() else {}
        completed = (process.returncode == 0 and terminal.get('status') == 'completed'
                     and receipt.get('status') == 'completed' and receipt.get('source_unchanged') is True
                     and receipt.get('source_sha256_before') == job['source_sha256']
                     and receipt.get('source_sha256_after') == job['source_sha256'])
        # Nonzero exits can leave a saved candidate. Preserve uncertainty until
        # its exact worker receipt is interpreted; never try another variant.
        result = {'status': 'completed' if completed else 'needs_reconciliation', 'output': output,
            'evidence': [str(log)] + ([str(receipt_path)] if receipt_path and receipt_path.is_file() else []),
            'native_stages_ms': {'isolated_job': round((time.perf_counter()-started)*1000, 3)},
            'cancel_requested': cancelled}
        phase_path = Path(output)/'native-stages.json' if output else None
        if phase_path and phase_path.is_file():
            result['evidence'].append(str(phase_path))
            try:
                from .native_bootstrap import phase_summary
                if phase_path.stat().st_size > 65536:
                    raise ValueError('Native timing receipt too large')
                phases = json.loads(phase_path.read_text(encoding='utf-8'))
                result['native_stages_ms'].update({'worker_' + name: value for name, value in phase_summary(phases).items()})
                result['worker_timing_status'] = phases['status']
            except (OSError, ValueError, TypeError, AttributeError):
                result['worker_timing_status'] = 'invalid; raw timing receipt retained'
            # Optional telemetry cannot turn a known effect into a replay or
            # qualify an incomplete native worker. The worker receipt governs.
        write_json(folder/'receipt.json', result)
        return result


class RetainCheckpoint:
    """Fixed reviewed retention using each native operation's fresh returned state.

    Payload: source and candidate file references, reopen reference, target path,
    label, optional pose and display argument dictionaries. Caller binds the real
    independent-reopen verdict; OperatingSession requires actual visual review.
    This handler cannot substitute its own technical or appearance acceptance.
    """
    def __init__(self, service, directory, *, verify_reopen):
        if not callable(verify_reopen):
            raise ValueError('The workspace must interpret its actual independent reopen evidence')
        self.service, self.directory, self.verify_reopen = service, Path(directory), verify_reopen

    def __call__(self, item, context):
        payload = item['payload']; contract = item['workbench']
        if (contract.get('profile') != 'retention' or not contract.get('native')
                or not contract.get('visual_review')):
            raise ValueError('A native retention task with actual appearance review required')
        refs = [payload[key] for key in ('source', 'candidate', 'reopen')]
        _bound(item, refs)
        reopen = json.loads(checked_file(payload['reopen']).read_text(encoding='utf-8'))
        if self.verify_reopen(deepcopy(reopen), deepcopy(payload['candidate'])) is not True:
            raise ValueError('Exact independent reopen evidence did not pass')
        target = Path(payload['target']).resolve()
        if target.exists():
            raise ValueError('Retention needs a new target; reconcile an existing saved result')
        for key in ('pose', 'display'):
            if {'owner', 'expected_state'} & payload.get(key, {}).keys():
                raise ValueError('The handler supplies actual owner and current state')
        if Path(item['id']).name != item['id']:
            raise ValueError('A single local task name required')
        folder = self.directory/item['id']; folder.mkdir(parents=True, exist_ok=False)
        timings, counts, files = {}, {}, []
        def call(name, operation, arguments):
            if context['cancelled'].is_set():
                raise RuntimeError('Retention interrupted; reconcile completed native steps')
            started = time.perf_counter()
            result = self.service.execute(operation, arguments)
            timings[name] = round((time.perf_counter()-started)*1000, 3)
            counts[operation] = counts.get(operation, 0)+1
            path = folder/(name+'.json'); write_json(path, result); files.append(str(path))
            write_json(folder/'progress.json', {'native_stages_ms': timings, 'native_calls': counts, 'evidence': files})
            if result.get('status') in ('failed', 'conflicting', 'needs attention', 'needs_reconciliation', 'unknown') or result.get('retention_status'):
                raise RuntimeError('Native step did not settle; inspect its original retained result')
            return result
        owner = context['owner']
        live = returned_live(self.service, call('preflight', 'native_inspect_live', {'owner': owner, 'refresh_scene': False}))
        already = Path(live['file']).resolve() == checked_file(payload['candidate'])
        _clean_source(live, owner, payload['candidate'] if already else payload['source'])
        if not already:
            opened = call('open', 'native_open_checkpoint', {'owner': owner, 'expected_state': live['expected_state'],
                'source': payload['candidate'], 'load_ui': False})
            live = returned_live(self.service, opened)
            if Path(live['file']).resolve() != checked_file(payload['candidate']):
                raise ValueError('Native open returned a different candidate')
        for key, operation in (('pose', 'native_set_controls'), ('display', 'native_set_display')):
            if payload.get(key):
                result = call(key, operation, {**payload[key], 'owner': owner, 'expected_state': live['expected_state']})
                live = returned_live(self.service, result)
        saved = call('save', 'native_save_checkpoint', {'owner': owner, 'expected_state': live['expected_state'],
            'label': payload['label'], 'path': str(target), 'copy': False})
        live = returned_live(self.service, saved)
        with target.open('rb') as stream:
            target_ref = {'path': str(target), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
        _clean_source(live, owner, target_ref)
        write_json(folder/'visible-live.json', live)
        result = {'status': 'completed', 'file': str(target), 'sha256': target_ref['sha256'],
            'evidence': files+[str(folder/'visible-live.json')], 'native_stages_ms': timings, 'native_calls': counts,
            'checkpoint_open_reused': already, 'user_appearance_accepted': False,
            'workbench': {'checks': {'checkpoint': {'status': 'pass', 'evidence': [_evidence(target, 'Saved clean retained checkpoint'),
                _evidence(folder/'visible-live.json', 'Native save returned current visible state')]}},
                'findings': [{'kind': 'measured', 'scope': 'reviewed checkpoint',
                    'summary': 'Reviewed candidate saved clean. Every mutation retained the native expected-state guard; repeated external inspections were omitted.'}]}}
        write_json(folder/'receipt.json', result)
        return result
