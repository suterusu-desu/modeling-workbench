"""Durable workflow handles backed by immutable, compare-and-swap revisions."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import json
import os
from .store import canonical, digest, atomic_write


class Conflict(RuntimeError):
    pass


class PreconditionRefusal(Conflict):
    """Typed control-flow evidence emitted only before mutation dispatch."""
    def __init__(self,message,stage,details):
        super().__init__(message)
        self.modeling_effect_status='refused before mutation dispatch'
        self.modeling_stage=stage
        self.modeling_details=details


class Ledger:
    def __init__(self, store):
        self.store = store
        self.root = store.root / 'workflow'
        self.root.mkdir(exist_ok=True)

    def _path(self, handle):
        if len(handle) != 64 or any(c not in '0123456789abcdef' for c in handle):
            raise ValueError('Invalid workflow handle')
        return self.root / (handle + '.json')

    @contextmanager
    def _lock(self, handle):
        path = self._path(handle).with_suffix('.lock')
        try:
            with path.open('x') as f:
                f.write(json.dumps({'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}))
        except FileExistsError as e:
            raise Conflict('Workflow write in progress; inspect lock owner before recovery') from e
        try:
            yield
        finally:
            path.unlink()

    def create(self, kind, intent, idempotency_key):
        if not idempotency_key.strip():
            raise ValueError('Stable idempotency key required')
        handle = digest(canonical([kind, idempotency_key]))
        with self._lock(handle):
            if self._path(handle).exists():
                previous = self.read(handle)
                if previous['intent'] != intent:
                    raise Conflict('Idempotency key belongs to a different intent')
                return dict(previous, reused=True)
            value = dict(kind=kind, intent=intent, status='prepared', history=[], data={}, handle=handle)
            return self._write(value, None)

    def _write(self, value, previous):
        value = {k:v for k,v in value.items() if k not in ('revision','reused')}
        value['previous'] = previous
        value['utc'] = datetime.now(timezone.utc).isoformat()
        revision = self.store.put('workflow', value)
        atomic_write(self._path(value['handle']), canonical({'revision': revision}))
        return dict(value, revision=revision, reused=False)

    def read(self, handle):
        revision = json.loads(self._path(handle).read_text())['revision']
        value = self.store.get(revision, 'workflow')
        if value['handle'] != handle:
            raise ValueError('Workflow pointer has incorrect identity')
        return dict(value, revision=revision)

    def update(self, handle, expected_revision, status, data, allowed=None):
        with self._lock(handle):
            value = self.read(handle)
            if value['revision'] != expected_revision:
                raise Conflict('Workflow changed; read its current revision before applying this update')
            if allowed is not None and value['status'] not in allowed:
                raise Conflict('Transition not permitted from ' + value['status'])
            value['history'] = [*value['history'], {'status':value['status'], 'revision':expected_revision}]
            value['status'] = status
            value['data'] = dict(value['data'], **data)
            return self._write(value, expected_revision)

    def list(self, kind=None, limit=20):
        if not 1 <= limit <= 100:
            raise ValueError('Limit must be between one and one hundred')
        rows = [self.read(p.stem) for p in self.root.glob('*.json')]
        return sorted((r for r in rows if kind is None or r['kind']==kind), key=lambda r:r['utc'], reverse=True)[:limit]

    def guarded_update(self, handle, expected_revision, status, data, guard, allowed=None):
        """Check cross-record invariants while holding the same mutation lock."""
        with self._lock(handle):
            value=self.read(handle)
            if value['revision']!=expected_revision:raise Conflict('Workflow changed; reread before update')
            if allowed is not None and value['status'] not in allowed:raise Conflict('Transition not permitted from '+value['status'])
            guard()
            value['history']=[*value['history'],{'status':value['status'],'revision':expected_revision}]
            value['status']=status;value['data']=dict(value['data'],**data)
            return self._write(value,expected_revision)
