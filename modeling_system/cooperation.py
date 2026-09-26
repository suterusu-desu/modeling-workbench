"""Supervised Owner/reviewer cooperation over the existing owner and task queue.

Reviews arrive as immutable episode facts. Waiting watches only that mailbox;
it never polls Blender or repeats a request for identical evidence.
"""
from copy import deepcopy
import math
from pathlib import Path
import threading
import time
import uuid

from .controller import fingerprint, read_json, write_json


def compose_catalogs(*catalogs):
    """Share independent pipelines and useful supporting work in one owner catalog.

    Dependencies, profile contracts and task definitions remain unchanged.
    Catalogs must qualify their own inputs and omit redundant work. This grants
    neither parallel native execution nor permission to branch a rejected model.
    """
    if not catalogs or not all(callable(catalog) for catalog in catalogs):
        raise ValueError('Executable catalogs required')

    def catalog(state, outcomes):
        rows = []
        for build in catalogs:
            supplied = build(deepcopy(state), deepcopy(outcomes))
            if not isinstance(supplied, list):
                raise ValueError('Each catalog must return qualified tasks')
            rows.extend(deepcopy(supplied))
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Composed catalogs must use unique task identities')
        return rows
    return catalog


def review_inbox(session, state):
    """Describe exact review dependencies without manufacturing a new gate."""
    requested = {}
    for item in session.items.values():
        finished = session.record['results'].get(item['id'], {})
        if (finished.get('definition') == fingerprint(item) and
                finished.get('status') in ('completed', 'failed', 'no_progress')):
            continue
        task = item['workbench'].get('visual_review')
        if not task:
            continue
        entry = session.record['results'].get(task, {})
        if entry.get('status') != 'completed' or task not in session.items:
            continue
        candidate = session.items[task]
        if fingerprint(candidate) != entry.get('definition'):
            continue
        review = session._review(task, state)
        if review and review['judgment']['disposition'] == 'useful':
            continue
        current = all(state['values'].get(k) == v for k, v in candidate['reads'].items()
                      if k not in candidate['writes'])
        report = entry.get('result', {}).get('workbench', {})
        request = requested.setdefault(task, {
            'task': task, 'basis': session._review_basis(task, state),
            'kind': 'visual_review' if current and review is None else 'reasoning',
            'question': (review['judgment']['next_question'] or review['judgment']['reason']) if review else
                ('Assess overall visible improvement against the guide, earlier useful baseline and affected motion.'
                 if current else 'Source dependencies changed; resolve the evidence basis before dependent use.'),
            'current_inputs': current,
            'judgment': deepcopy(review['judgment']) if review else None,
            'operation_handle': entry.get('operation_handle'),
            'evidence': {key: deepcopy(check.get('evidence', [])) for key, check in report.get('checks', {}).items()},
            'waiting_tasks': []})
        request['waiting_tasks'].append(item['id'])
    return {'requests': list(requested.values()),
        'ready_tasks': [{'task': a['id'], 'lane': a['lane']} for a in state['actions']],
        'blocked_tasks': deepcopy(state['observations']['blocked']),
        'limits': 'Private review index, not acceptance or a new authority. Only dependent work waits; qualified unrelated work can continue.'}


def _feedback_stamp(directory):
    # Only mailbox metadata, not mutable native state or expensive source scans.
    return tuple(sorted((p.name, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                        for p in (Path(directory) / 'reviews').glob('*.json')
                        for s in [p.stat()]))


def request_stop(directory, *, reason):
    """Stop the current supervised interval at its next operation boundary."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('Stop reason required')
    directory = Path(directory)
    run = read_json(directory / 'cooperative-run.json')
    if run['status'] != 'running':
        return {'status': 'already_stopped', 'run_id': run['run_id']}
    result = {'run_id': run['run_id'], 'reason': reason}
    write_json(directory / 'cooperation-stop.json', result)
    return result


def run_cooperatively(session, *, max_steps, feedback_timeout=0, on_status=None,
                      on_handoff=None, cancelled=None):
    """Run useful operations while the owner reviews, optionally awaiting feedback.

    The caller supervises this finite ordinary run and joins/stops it before
    ending its turn. No scheduler, agent, automation or inference is created.
    The optional timeout bounds total idle review waiting, not provider spend.
    Zero preserves the existing immediate-return behavior.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError('Positive bounded step count required')
    if (isinstance(feedback_timeout, bool) or not isinstance(feedback_timeout, (int, float))
            or not math.isfinite(feedback_timeout) or feedback_timeout < 0):
        raise ValueError('Finite nonnegative supervised feedback timeout required')
    stop_event = cancelled if cancelled is not None else threading.Event()
    run = {'run_id': uuid.uuid4().hex, 'status': 'running', 'owner': session.owner,
           'started_at': time.time(), 'max_steps': max_steps, 'feedback_timeout': feedback_timeout}
    path = session.directory / 'cooperative-run.json'
    stop_path = session.directory / 'cooperation-stop.json'
    handoff_revision = None
    waited, completed_during_review = 0.0, 0

    def stopped():
        return stop_event.is_set() or (stop_path.exists() and
               read_json(stop_path).get('run_id') == run['run_id'])

    def status(value):
        nonlocal handoff_revision, completed_during_review
        cooperation = deepcopy(value.get('cooperation', {}))
        requests = cooperation.get('requests', [])
        attention = {k: deepcopy(value[k]) for k in ('reason', 'evidence', 'attention') if k in value}
        handoff = {'run_id': run['run_id'], **cooperation,
                   'selection_attention': attention if value['status'] == 'needs_review' else None}
        revision = fingerprint({'requests': requests, 'attention': handoff['selection_attention']})
        if revision != handoff_revision:
            write_json(session.directory / 'handoff.json', handoff)
            handoff_revision = revision
            if on_handoff is not None and (requests or handoff['selection_attention']):
                on_handoff(deepcopy(handoff))
        if value['status'] == 'completed' and value.get('action') and requests:
            completed_during_review += 1
        if on_status is not None:
            on_status(value)

    # Acquire the original native-owner lock before publishing a running interval.
    with session._controller(status) as controller:
        write_json(path, run)
        result = None
        try:
            for _ in range(max_steps):
                if stopped():
                    result = controller._status('stopped', reason='Supervised owner requested stop')
                    break
                before = _feedback_stamp(session.directory)
                result = controller.tick()
                if result['status'] in ('completed', 'failed', 'no_progress') and result.get('action'):
                    continue
                requests = result.get('cooperation', {}).get('requests', [])
                if (result['status'] != 'needs_review' or waited >= feedback_timeout or
                        not any(row['kind'] == 'visual_review' for row in requests)):
                    break
                remaining = feedback_timeout - waited
                start = time.monotonic()
                controller._status('waiting_feedback', cooperation=result['cooperation'],
                                   reason='Owner review pending; no other applicable work remains')
                while not stopped() and _feedback_stamp(session.directory) == before:
                    left = remaining - (time.monotonic() - start)
                    if left <= 0:
                        break
                    stop_event.wait(min(.1, left))
                waited += time.monotonic() - start
                if stopped():
                    result = controller._status('stopped', reason='Supervised owner requested stop')
                    break
                if _feedback_stamp(session.directory) == before:
                    result = controller._status('needs_review', cooperation=result['cooperation'],
                                               reason='Supervised feedback interval ended')
                    break
                # A changed mailbox only wakes the queue. Exact source, intent,
                # owner, review basis and native checks are rerun before effects.
            else:
                result = controller._status('stopped', reason='Authorized step bound reached')
            return result
        finally:
            run.update(status='closed', finished_at=time.time(),
                       outcome=result['status'] if result else 'interrupted',
                       feedback_wait_seconds=round(waited, 6),
                       operations_completed_while_review_pending=completed_during_review)
            write_json(path, run)
