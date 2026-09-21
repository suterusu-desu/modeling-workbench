"""Persistent owner-run operations with asynchronous, advisory planning.

Only the planner runs on a worker. Native execution, observation, selection and
status callbacks stay on the calling owner's thread. No transport is bundled.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import threading
import time
import uuid


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        # Windows readers can briefly hold a destination without delete sharing.
        # Retry only this metadata rename, never the selected operation itself.
        for attempt in range(6):
            try:
                os.replace(temporary, path)
                break
            except OSError as error:
                if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                    raise
                time.sleep(.02 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class SelectionNotDispatched(ValueError):
    """A selector's deterministic preparation refused before any inference call."""


def applicable(binding, state):
    reads = binding.get('reads')
    return (isinstance(reads, dict) and bool(reads)
            and binding.get('authority_revision') == state['authority_revision']
            and binding.get('stage') == state['stage']
            and all(state['values'].get(k) == v for k, v in reads.items()))


class MailboxPlanner:
    """Exchange plans with an existing agent without adding an inference account.

    The agent reads request.json and atomically writes response.json containing
    request_id and plan. One request is pending at a time. No agent is spawned.
    """
    def __init__(self, directory, timeout=120):
        self.directory = Path(directory)
        self.timeout = timeout
        self.cancelled = threading.Event()

    def __call__(self, request):
        write_json(self.directory / 'request.json', request)
        deadline = time.monotonic() + self.timeout
        response = self.directory / 'response.json'
        while not self.cancelled.wait(.1):
            if response.exists():
                result = read_json(response)
                if result.get('request_id') == request['request_id']:
                    return result['plan']
            if time.monotonic() >= deadline:
                raise TimeoutError('Planner response deadline reached')
        raise InterruptedError('Planner mailbox closed')

    def close(self):
        self.cancelled.set()


class PersistentController:
    def __init__(self, directory, *, owner, observe, execute, select=None,
                 planner=None, initial_plan=None, on_status=None,
                 plan_interval=15, revalidate=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner, self.observe, self.execute = owner, observe, execute
        self.select, self.planner, self.on_status = select, planner, on_status
        self.revalidate = revalidate
        self.timings = {}
        self.plan_interval = plan_interval
        self.thread = threading.get_ident()
        self.cancelled = threading.Event()
        self.closed = False
        self.last_status = {}
        self.pending = None
        self.last_request_key = None
        self.last_request_at = float('-inf')
        self.plan = deepcopy(initial_plan)
        self.lock = self.directory / 'controller.lock'
        with self.lock.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'pid': os.getpid(), 'owner': owner}))
        try:
            state_path = self.directory / 'controller.json'
            self.record = read_json(state_path) if state_path.exists() else {
                'schema_version': 1, 'owner': owner, 'attempts': [],
                'selection': None, 'status': 'ready'}
            if self.record['owner'] != owner:
                raise ValueError('Controller belongs to a different native owner')
            if self.plan is None:
                self.plan = self.record.get('plan')
            self.last_request_key = self.record.get('last_planner_request_key')
        except BaseException:
            self.lock.unlink()
            raise

    def _save(self):
        self.record['plan'] = self.plan
        write_json(self.directory / 'controller.json', self.record)

    def _status(self, status, state=None, action=None, **details):
        current = {
            'status': status, 'owner': self.owner, 'updated_at': time.time(),
            'stage': (state or self.last_status).get('stage'),
            'objective': (self.plan or {}).get('objective'),
            'planner_pending': self.pending is not None,
            'action': None if action is None else action['id'],
            'last_action': action['id'] if action is not None else self.last_status.get('last_action'),
            'completed_operations': sum(a['status'] == 'completed' for a in self.record['attempts']),
            'inference_budget': (state or self.last_status).get('inference_budget'),
            'timings_ms': deepcopy(self.timings),
            **details}
        self.last_status = deepcopy(current)
        write_json(self.directory / 'status.json', current)
        if self.on_status is not None:
            self.on_status(deepcopy(current))
        return current

    def _event(self, kind, **fields):
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': kind, 'time': time.time(), **fields},
                                    allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _observe(self):
        state = deepcopy(self.observe())
        return self._validate_state(state)

    def _validate_state(self, state):
        if state.get('owner') != self.owner or not state.get('authority_revision'):
            raise ValueError('Fresh sole-owner authority is required')
        if not isinstance(state.get('values'), dict) or not state['values']:
            raise ValueError('Measured dependency revisions are required')
        if not isinstance(state.get('stage'), str) or not isinstance(state.get('active_operations'), list):
            raise ValueError('Stage and actual owner operation lane are required')
        actions = state.get('actions', [])
        if len({a['id'] for a in actions}) != len(actions):
            raise ValueError('Action IDs must be unique')
        for action in actions:
            reads, writes = action.get('reads'), action.get('writes')
            if (not action.get('revision') or not isinstance(reads, dict) or not reads
                    or any(not isinstance(v, str) or not v for v in reads.values())
                    or not isinstance(writes, list) or not set(writes) <= set(reads)):
                raise ValueError('Operations require revision, exact reads and covered writes')
        return state

    def _fresh_for_action(self, action, state):
        if self.revalidate is None:
            return self._observe()
        reads = dict(action['reads'])
        reads.update((self.plan or {}).get('reads', {}))
        if self.pending is not None:
            reads.update(self.pending['request']['binding']['reads'])
        expected = {'owner': self.owner, 'authority_revision': state['authority_revision'],
                    'stage': state['stage'], 'reads': reads,
                    'action_fingerprint': fingerprint(action)}
        receipt = deepcopy(self.revalidate(deepcopy(action), deepcopy(expected)))
        # The adapter must read current revisions, not echo the expected argument.
        # Never fill a missing revision from an earlier observation.
        if (not isinstance(receipt, dict) or not isinstance(receipt.get('values'), dict)
                or not set(reads) <= set(receipt['values'])):
            raise ValueError('Selected-action guard omitted required current dependencies')
        fresh = {k: receipt[k] for k in ('owner', 'authority_revision', 'stage', 'values', 'active_operations')}
        fresh['inference_budget'] = receipt.get('inference_budget')
        fresh['actions'] = [action] if receipt.get('action_fingerprint') == expected['action_fingerprint'] else []
        return self._validate_state(fresh)

    def _timed(self, name, callback, *args):
        start = time.perf_counter()
        try:
            return callback(*args)
        finally:
            self.timings[name] = round((time.perf_counter() - start) * 1000, 3)

    def _key(self, action, state):
        return fingerprint({'action': action, 'authority': state['authority_revision'],
                            'stage': state['stage']})

    def _eligible(self, state):
        settled = {a['key'] for a in self.record['attempts']
                   if a['status'] in ('completed', 'failed', 'no_progress')}
        return [a for a in state.get('actions', [])
                if not a.get('completed') and not a.get('blocked')
                and all(state['values'].get(k) == v for k, v in a['reads'].items())
                and self._key(a, state) not in settled]

    def _poll_plan(self, state):
        pending = self.pending
        if pending is None or not pending['done'].is_set():
            return
        self.pending = None
        result = pending.get('result')
        if (pending.get('error') is None and isinstance(result, dict)
                and isinstance(result.get('objective'), str)
                and applicable(pending['request']['binding'], state)
                and applicable(result, state)):
            # Plans prioritize existing owner actions; they never create actions.
            self.plan = deepcopy(result)
            self._save()
            self._event('plan_accepted', request_id=pending['request']['request_id'], plan=result)
        else:
            self._event('plan_discarded', request_id=pending['request']['request_id'],
                        reason=pending.get('error', 'stale or invalid binding'))

    def _refresh_plan(self, state, reason):
        if self.planner is None or self.pending is not None:
            return
        # Identical observations never cause endless periodic inference.
        binding = {'authority_revision': state['authority_revision'],
                   'stage': state['stage'],
                   'reads': state.get('planner_reads', state['values'])}
        if not applicable(binding, state):
            raise ValueError('Planner read scope is stale or empty')
        key = fingerprint({'binding': binding, 'reason': reason,
                           'replan_revision': state.get('replan_revision'),
                           'actions': state.get('actions', [])})
        now = time.monotonic()
        if key == self.last_request_key:
            return
        if reason == 'progress' and now - self.last_request_at < self.plan_interval:
            return
        request = {'request_id': uuid.uuid4().hex, 'binding': deepcopy(binding),
                   'reason': reason, 'snapshot': deepcopy(state), 'plan': deepcopy(self.plan)}
        self.last_request_key, self.last_request_at = key, now
        self.record['last_planner_request_key'] = key
        self._save()
        pending = {'request': request, 'done': threading.Event()}
        self.pending = pending
        self._event('planner_requested', request_id=request['request_id'], binding=binding, reason=reason)

        def plan():
            try:
                pending['result'] = self.planner(deepcopy(request))
            except Exception as exc:
                pending['error'] = type(exc).__name__ + ': ' + str(exc)
            finally:
                pending['done'].set()
        threading.Thread(target=plan, name='workbench-planner', daemon=True).start()

    def tick(self):
        if threading.get_ident() != self.thread or self.closed:
            raise RuntimeError('Only the creating native owner may drive an open controller')
        self.timings = {}
        start = time.perf_counter()
        try:
            result = self._tick()
        except Exception as exc:
            uncertain = (self.record.get('selection') is not None or any(
                a['status'] in ('running', 'needs_reconciliation') for a in self.record['attempts']))
            result = self._status('needs_reconciliation' if uncertain else 'error',
                                  reason=type(exc).__name__ + ': ' + str(exc))
        self.timings['cycle'] = round((time.perf_counter() - start) * 1000, 3)
        result['timings_ms'] = deepcopy(self.timings)
        self.last_status = deepcopy(result)
        write_json(self.directory / 'status.json', result)
        self._event('cycle_timing', timings_ms=self.timings, status=result['status'], action=result['action'])
        return result

    def _tick(self):
        if self.cancelled.is_set():
            return self._status('stopped', reason='Owner cancelled')
        if (self.record.get('selection') is not None
                or any(a['status'] in ('running', 'needs_reconciliation') for a in self.record['attempts'])):
            return self._status('needs_reconciliation', reason='Inspect original receipts; never replay uncertain effects')
        state = self._timed('observe', self._observe)
        self._poll_plan(state)
        if state.get('done'):
            return self._status('completed', state)
        if self.plan is not None and not applicable(self.plan, state):
            self.plan = None
            self._save()
        actions = self._eligible(state)
        reason = 'initial_or_stale' if self.plan is None else ('no_useful_actions' if not actions else 'progress')
        self._refresh_plan(state, reason)
        if state['active_operations']:
            return self._status('waiting_owner', state)
        if not actions:
            if state.get('attention'):
                return self._status('needs_review', state, attention=deepcopy(state['attention']))
            return self._status('idle', state, reason='No new applicable operation; wait for changed evidence or owner plan')
        required = [a for a in actions if a.get('required')]
        selected_plan = None
        if required:
            action = required[0]
        elif len(actions) == 1:
            action = actions[0]
        else:
            if self.plan is None:
                return self._status('waiting_plan', state)
            if self.select is None:
                return self._status('needs_selection', state)
            self.record['selection'] = {'state': state, 'actions': actions, 'plan': self.plan}
            selected_plan = fingerprint(self.plan)
            self._save()  # Interrupted inference is reconciled, never repeated.
            self._status('selecting', state)
            try:
                choice = self._timed('select', self.select, deepcopy(state), deepcopy(actions), deepcopy(self.plan))
                if isinstance(choice, dict) and choice.get('status') == 'needs_review':
                    self.record['selection'] = None
                    self._save()
                    self._event('selection_needs_review', details=choice)
                    return self._status('needs_review', state, reason=choice.get('reason'),
                                        evidence=choice.get('evidence', []))
                action = next(a for a in actions if a['id'] == choice)
            except SelectionNotDispatched as exc:
                self.record['selection'] = None
                self._save()
                self._event('selection_refused_before_dispatch', reason=str(exc))
                return self._status('invalidated', state, reason=str(exc), provider_dispatched=False)
            except Exception as exc:
                self._event('selection_unresolved', error=type(exc).__name__)
                return self._status('needs_reconciliation', state, reason='Selection failed; inspect provider ledger')
            self.record['selection'] = None
            self._save()
            self._event('selected', action=action['id'])
        # A selected-action guard can avoid rebuilding the full observation/menu.
        fresh = self._timed('revalidate', self._fresh_for_action, action, state)
        self._poll_plan(fresh)
        if (fresh['active_operations'] or self.cancelled.is_set()
                or fresh['authority_revision'] != state['authority_revision']
                or fresh['stage'] != state['stage']
                or (selected_plan is not None and (not applicable(self.plan or {}, fresh)
                    or fingerprint(self.plan) != selected_plan))
                or not any(a == action for a in self._eligible(fresh))):
            return self._status('invalidated', fresh, action, reason='State or authorized operation changed before dispatch')
        attempt = {'key': self._key(action, fresh), 'action': deepcopy(action),
                   'status': 'running', 'started_at': time.time()}
        self.record['attempts'].append(attempt)
        self._save()  # Must land before any native effect.
        self._status('running', fresh, action)

        def progress(**details):
            if threading.get_ident() != self.thread:
                raise RuntimeError('Native status callbacks stay on the owner thread')
            return self._status('running', fresh, action, progress=details)

        context = {'owner': self.owner, 'expected_values': deepcopy(action['reads']),
                   'authority_revision': fresh['authority_revision'],
                   'cancelled': self.cancelled, 'progress': progress,
                   'attempt_key': attempt['key']}
        try:
            result = self._timed('execute', self.execute, deepcopy(action), context)
            if not isinstance(result, dict) or result.get('status') not in (
                    'completed', 'failed', 'no_progress', 'needs_reconciliation'):
                raise ValueError('Operation must return a settled result or explicit uncertainty')
            attempt.update(status=result['status'], result=deepcopy(result), finished_at=time.time())
        except Exception as exc:
            attempt.update(status='needs_reconciliation', error=type(exc).__name__ + ': ' + str(exc))
        self._save()
        self._event('operation_result', attempt=attempt)
        return self._status(attempt['status'], fresh, action, result=attempt.get('result'),
                            reason=attempt.get('error'))

    def run(self, max_steps=100):
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError('Positive bounded step count required')
        for _ in range(max_steps):
            result = self.tick()
            if result['status'] not in ('completed', 'failed', 'no_progress') or result['action'] is None:
                return result
        return self._status('stopped', reason='Authorized step bound reached')

    def close(self):
        if not self.closed:
            self.cancelled.set()
            if hasattr(self.planner, 'close'):
                self.planner.close()
            self.closed = True
            if self.last_status:
                current = deepcopy(self.last_status)
                current.update(planner_pending=False, controller_closed=True, updated_at=time.time())
                write_json(self.directory / 'status.json', current)
                if self.on_status is not None:
                    self.on_status(deepcopy(current))
            self.lock.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
