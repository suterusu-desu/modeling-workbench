"""A synchronous, non-replayable retention transaction for qualified adapters.

No Blender access occurs here. Hooks run on the existing sole native lane and
must not yield, invoke modal operators, dispatch jobs or accept arbitrary code.
The initial guard and final full observation are never cached across requests.
Receipts describe historical effects; reading one does not observe current state.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid


def reference(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest}


def checked(ref):
    actual = reference(ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError('Retention dependency changed')
    return Path(actual['path'])


def _write(path, value):
    # Persist intent before native effects. Failure is terminal, never a reason
    # to repeat an effect. The exclusive transaction directory also survives a
    # crash before the first receipt can be written.
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    with pending.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def transaction_directory(root, owner, transaction_id):
    key = json.dumps([owner, transaction_id], ensure_ascii=True).encode()
    return Path(root)/hashlib.sha256(key).hexdigest()


def inspect_receipt(root, owner, transaction_id):
    """Read saved evidence only; never resume, replay, or claim current state."""
    folder = transaction_directory(root, owner, transaction_id)
    path = folder/'receipt.json'
    return {'historical': True, 'replay_allowed': False, 'directory': str(folder),
            'receipt': json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None,
            'claimed': folder.exists()}


def _clean(live, owner, ref):
    if (live.get('owner') != owner or live.get('dirty') is not False
            or not live.get('expected_state') or live.get('geometry_state_error')
            or Path(live['file']).resolve() != checked(ref)
            or Path(live.get('saved_file', {}).get('path', '')).resolve() != checked(ref)
            or live.get('saved_file', {}).get('sha256') != ref['sha256']):
        raise ValueError('Retention requires the exact clean source and actual owner')
    if live.get('geometry_state_id') and live.get('scene_freshness', {}).get('status') != 'current':
        raise ValueError('Stale numerical geometry cannot be presented as current')


def run_retention(arguments, hooks, root):
    """Execute once, using a pinned adapter's fixed synchronous hook methods.

    guard performs fresh owner/expected-state/dependency checks. rollback saves
    the full source and external registry. restore_runtime must compare actual
    native content before/after registration without diagnostic export.
    verify_presentation checks declared controls, guide and display. observe
    returns a fresh complete native state without silently exporting arrays.
    """
    args = deepcopy(arguments)
    owner = args['_owner']; pose = args.get('pose', {}); display = args['display']
    if (not isinstance(owner, str) or not owner.strip()
            or not isinstance(args['transaction_id'], str) or not args['transaction_id'].strip()
            or display.get('mode') != 'GUIDE_WIRE' or pose.get('refresh', False) is not False
            or set(pose)-{'controls', 'guide', 'refresh'}
            or set(display)-{'mode', 'through', 'parts', 'viewport', 'view'}):
        raise ValueError('Only fixed controls and GUIDE_WIRE retention are qualified')
    for value in pose.get('controls', {}).values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite numeric controls required')
    folder = transaction_directory(root, owner, args['transaction_id'])
    folder.parent.mkdir(parents=True, exist_ok=True)
    # Deliberately refuse completed IDs too: their live state is historical.
    folder.mkdir(exist_ok=False)
    receipt_path = folder/'receipt.json'
    receipt = {'schema_version': 1, 'status': 'started', 'request': args,
        'request_sha256': hashlib.sha256(json.dumps(args, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'receipt_path': str(receipt_path), 'phases': [], 'replay_allowed': False,
        'user_appearance_accepted': False}

    def phase(name, action):
        row = {'name': name, 'status': 'started'}
        receipt['phases'].append(row); _write(receipt_path, receipt)
        started = time.perf_counter()
        result = action()
        row.update(status='completed', elapsed_ms=round((time.perf_counter()-started)*1000, 3))
        receipt[name] = result; _write(receipt_path, receipt)
        return result

    refs = [args[key] for key in ('source', 'candidate', 'reopen')]
    target = Path(args['target']).resolve()
    def dependencies():
        for ref in refs: checked(ref)
        hooks.check_dependencies()

    def new_target():
        # Cooperative sole-writer protection, not an OS-level filesystem lock.
        if target.exists() or target.is_symlink():
            raise ValueError('Retention target exists; reconcile it without overwriting')

    def verify_presentation():
        result = hooks.verify_presentation(pose, display)
        if result.get('declared_pose_guide_display_verified') is not True:
            raise ValueError('Declared presentation was not verified')
        return result

    try:
        _write(receipt_path, receipt)
        dependencies(); new_target()
        before = phase('guard', lambda: hooks.guard(args))
        if before['expected_state'] != args['expected_state']:
            raise ValueError('Retention expected state changed')
        already = Path(before['file']).resolve() == checked(args['candidate'])
        _clean(before, owner, args['candidate'] if already else args['source'])
        registry = before['guide_registry']; checked(registry)
        receipt['checkpoint_open_reused'] = already
        phase('rollback', lambda: hooks.rollback(before, folder))
        if not already:
            dependencies(); checked(registry)
            phase('open', lambda: hooks.open_candidate(args['candidate'], args))
        restored = phase('restore', lambda: hooks.restore_runtime(args['candidate'], args, folder))
        if restored.get('character_content_unchanged') is not True:
            raise ValueError('Runtime restoration lacks content preservation evidence')
        if pose: phase('pose', lambda: hooks.set_pose(pose))
        phase('display', lambda: hooks.set_display(display))
        phase('verify_presentation', verify_presentation)
        dependencies(); checked(registry); new_target()
        saved = phase('save', lambda: hooks.save(target))
        if checked(saved) != target:
            raise ValueError('Native save returned a different target')
        live = phase('observe', hooks.observe)
        _clean(live, owner, saved)
        if checked(saved) != target:
            raise ValueError('Saved target changed')
        verify_presentation()
        dependencies(); checked(registry)
        receipt.update(status='completed', saved=saved, live=live)
        _write(receipt_path, receipt)
        return {'status': 'saved', 'saved': saved, 'live': live,
            'receipt_path': str(receipt_path), 'checkpoint_open_reused': already,
            'native_stages_ms': {row['name']: row['elapsed_ms'] for row in receipt['phases']},
            'replay_allowed': False, 'user_appearance_accepted': False}
    except BaseException as error:
        receipt.update(status='needs_reconciliation', error={'type': type(error).__name__, 'message': str(error)})
        try: _write(receipt_path, receipt)
        except BaseException: pass  # Original effect/error and durable claim survive.
        raise
