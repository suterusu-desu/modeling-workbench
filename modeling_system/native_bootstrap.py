"""Scoped setup and phase receipts inside a owner-selected isolated native job.

Importing this module never imports Blender or opens a scene. Workspace code
supplies the qualified feature API, dependencies and actual scoped checks.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import math
from pathlib import Path
import re
import sys
import time
from .controller import write_json
from .native_recipes import checked_file
from .result_reporting import json_data


SCOPE_ROLES = ('source', 'guide', 'depth', 'sections', 'correspondence', 'preservation', 'adapter')


class StageTimings:
    """Nonoverlapping monotonic phase measurements, retained even on failure."""
    def __init__(self, path, *, clock=time.perf_counter):
        self.path, self.clock = Path(path), clock
        if self.path.exists():
            raise ValueError('Preserve an existing native phase receipt')
        self.started, self.active = clock(), None
        self.record = {'schema_version': 1, 'status': 'running', 'stages': []}
        write_json(self.path, self.record)

    @contextmanager
    def measure(self, name):
        if (self.active is not None or self.record['status'] != 'running'
                or not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name)
                or len(self.record['stages']) >= 100):
            raise ValueError('A bounded, named, nonoverlapping native phase is required')
        self.active = {'name': name, 'status': 'running', 'elapsed_ms': 0.}
        self.record['stages'].append(self.active)
        write_json(self.path, self.record)
        started = self.clock()
        try:
            yield
        except BaseException:
            self.active['status'] = self.record['status'] = 'failed'
            raise
        else:
            self.active['status'] = 'completed'
        finally:
            self.active['elapsed_ms'] = round((self.clock() - started) * 1000, 3)
            self.active = None
            write_json(self.path, self.record)

    def finish(self):
        if self.active is not None:
            raise ValueError('Finish the active native phase first')
        if self.record['status'] == 'running':
            self.record['status'] = 'completed'
        self.record['total_ms'] = round((self.clock() - self.started) * 1000, 3)
        write_json(self.path, self.record)
        return json_data(self.record)


def phase_summary(record):
    """Validate worker timings without letting telemetry redefine native effects."""
    if (record.get('schema_version') != 1 or record.get('status') not in ('running', 'completed', 'failed')
            or not isinstance(record.get('stages'), list) or len(record['stages']) > 100):
        raise ValueError('Invalid native phase receipt')
    result = {}
    for row in record['stages']:
        name, duration = row.get('name'), row.get('elapsed_ms')
        if (not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name)
                or row.get('status') not in ('running', 'completed', 'failed')
                or type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0):
            raise ValueError('Invalid native phase measurement')
        if row['status'] != 'running':
            result[name] = result.get(name, 0.) + duration
    return result


def load_pinned_modules(references):
    """Verify all exact source bytes before executing any qualified module.

    Only the native controller invokes this. Names and file references are
    supplied by the private workspace, never inferred from scene history.
    """
    sources = {}
    for name, ref in references.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z_]\w*', name) or name in sys.modules:
            raise ValueError('A fresh explicit bootstrap module name is required')
        path = Path(ref['path']).resolve(strict=True)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ref['sha256']:
            raise ValueError('Pinned bootstrap module changed')
        sources[name] = (path, raw)
    loaded = {}
    for name, (path, raw) in sources.items():
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile(raw, str(path), 'exec'), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        loaded[name] = module
    return loaded


def begin_scoped_feature(feature_api, feature, mismatch, mechanism, evidence_dir, *,
                         scope_check, protected=(), repair_invalid_bindings=False):
    """Replace broad export only when the operation checks its full dependency closure.

    scope_check(record) must perform actual workspace checks, returning every
    SCOPE_ROLES entry as {status: 'pass', evidence: [{path, sha256}, ...]}.
    Its implementation and files belong in the native job's pinned dependencies.
    This preserves actual feature freshness and binding checks, but does not
    refresh unrelated depth caches or create a whole-scene preparation export.
    """
    if not callable(scope_check) or not mismatch or not mechanism:
        raise ValueError('Actual scoped verification and an explicit modeling intent are required')
    path = Path(evidence_dir) / 'feature-entry.json'
    if path.exists():
        raise ValueError('Preserve the existing feature entry')
    record = feature_api.refresh(feature, refresh_depth=False)
    if not repair_invalid_bindings:
        feature_api.bindings().require_valid()
    checks = json_data(scope_check(record))
    if set(checks) != set(SCOPE_ROLES):
        raise ValueError('Scoped bootstrap must cover every modeling dependency role')
    for check in checks.values():
        if check.get('status') != 'pass' or not isinstance(check.get('evidence'), list) or not check['evidence']:
            raise ValueError('Scoped bootstrap needs actual passing evidence for every role')
        for ref in check['evidence']:
            checked_file(ref)
    record.update(mismatch=mismatch, mechanism=mechanism, protected=list(protected),
                  repair_invalid_bindings=bool(repair_invalid_bindings), scoped_checks=checks,
                  bootstrap_scope='verified feature dependency closure; no broad export')
    feature_api.require_fresh(record)
    write_json(path, json_data(record))
    return record
