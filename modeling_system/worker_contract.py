"""Owner-worker completion contract: verify before the completion file exists.

An isolated launcher classifies a run as completed only when the process exits
successfully, its pinned source is unchanged and a named completion file exists in
the run's output directory. A worker that writes arrays and a report but omits
that file fails the launcher contract despite exit zero; a worker that writes it
first can mark failed work complete. This module makes the completion file the
last step of an explicitly declared contract: required artifact names, named
checks, protected-source hashes and the completion filename are declared before
work, verified after work, and only then written. A failed check, a missing
artifact, a changed source or a different existing manifest never produces a
success manifest.

Standalone by design: standard library plus the package's atomic write, canonical
JSON and digest primitives. No bpy, numpy, service, native transport, process
launching or retry. Import it from the package or load this file alone inside a
worker.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

try:
    from .store import atomic_write, canonical, digest
except ImportError:  # loaded from this file alone, outside the package
    import importlib.util as _util
    _spec = _util.spec_from_file_location('_worker_contract_store', Path(__file__).with_name('store.py'))
    _store = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_store)
    atomic_write, canonical, digest = _store.atomic_write, _store.canonical, _store.digest

__all__ = ['ContractConflict', 'ContractError', 'ContractFailure', 'FAILURE_KIND', 'MANIFEST_KIND',
           'WorkerContract', 'bounded_path', 'file_sha256', 'is_sha256', 'relative_name', 'verify_manifest']

SCHEMA_VERSION = 1
MANIFEST_KIND = 'worker_completion'
FAILURE_KIND = 'worker_contract_failure'
RESERVED = frozenset(('schema_version', 'kind', 'contract_status', 'completion_file', 'completion_written', 'output_dir',
                      'label', 'stage', 'contract', 'artifacts', 'checks', 'sources', 'failures', 'utc'))
MAX_NAME_CHARS = 200
MAX_NAME_SEGMENTS = 8
MAX_CHECK_CHARS = 120
MAX_FAILURE_REPORTS = 99


def _atomic_create(path, data):
    """Publish a complete file without replacing a concurrent or prior writer."""
    path=Path(path);temp=path.with_name(uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        temp.unlink(missing_ok=True)
_HEX = frozenset('0123456789abcdef')


class ContractError(ValueError):
    """Declaration or usage error; nothing was written."""


class ContractFailure(ContractError):
    """Finalization refused: a failure report was written and no completion manifest."""

    def __init__(self, message, report, report_path):
        super().__init__(message)
        self.report = report
        self.report_path = str(report_path)


class ContractConflict(ContractFailure):
    """A different existing completion manifest was preserved untouched."""


def file_sha256(path):
    """Streaming SHA-256 of a file; equals store.digest of its complete bytes."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def is_sha256(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def relative_name(name, what='Artifact name'):
    """A bounded relative POSIX name that cannot leave its output directory."""
    if not isinstance(name, str) or not name or len(name) > MAX_NAME_CHARS:
        raise ContractError(what + ' must be a nonempty string of at most 200 characters')
    if '\\' in name or ':' in name or name.startswith('/') or any(ord(c) < 32 for c in name):
        raise ContractError(what + ' must be a relative POSIX path without drive, backslash or control characters: ' + name)
    parts = name.split('/')
    if len(parts) > MAX_NAME_SEGMENTS or any(p in ('', '.', '..') or p != p.rstrip('. ') for p in parts):
        raise ContractError(what + ' cannot contain empty, dot, parent or trailing dot/space segments: ' + name)
    return name


def bounded_path(root, name, what='Artifact name'):
    """Resolve name under an already resolved root, refusing links, junctions and escapes at every segment."""
    path = root
    for part in relative_name(name, what).split('/'):
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ContractError(what + ' passes through a link or junction: ' + name)
    if not path.resolve().is_relative_to(root):
        raise ContractError(what + ' leaves the output directory: ' + name)
    return path


def _single_name(name, what):
    if name is not None and isinstance(name, str) and '/' in name:
        raise ContractError(what + ' must be a single filename in the output directory')
    return relative_name(name, what)


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _jsonable(value, what):
    try:
        canonical(value)
    except (TypeError, ValueError) as error:
        raise ContractError(what + ' must be finite JSON-serializable data: ' + str(error)) from error
    return value


def _role(value, name):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ContractError('Role must be a nonempty string: ' + name)
    return value


class WorkerContract:
    """Declare, preflight, check, then finalize; the completion file is written last.

    The repository's docs/archive/skill-pages/execution-contracts.md keeps the adoption snippet and manifest layout.
    """

    def __init__(self, output_dir, *, completion_file=None, required_artifacts=None, required_checks=None,
                 protected_sources=None, label=None, failure_file='contract-failure.json', source_root=None):
        if not isinstance(output_dir, (str, Path)) or not str(output_dir):
            raise ContractError('output_dir must be the launcher-assigned output directory')
        self.output_dir = Path(output_dir)
        if completion_file is None:
            raise ContractError('Declare completion_file explicitly: the launcher-required completion name, for example inventory.json')
        self.completion_file = _single_name(completion_file, 'completion_file')
        self.failure_file = _single_name(failure_file, 'failure_file')
        if self.completion_file == self.failure_file:
            raise ContractError('completion_file and failure_file must differ')
        if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > MAX_NAME_CHARS):
            raise ContractError('label must be a nonempty string of at most 200 characters')
        self.label = label
        self.required_artifacts = self._declare_artifacts(required_artifacts)
        self.required_checks = self._declare_checks(required_checks)
        self.protected_sources = self._declare_sources(protected_sources, source_root)
        self._recorded = {}
        self._additional = {}
        self.root = None
        self.preflight_report = None

    # Declarations -----------------------------------------------------------------

    def _declare_artifacts(self, rows):
        if rows is None or not isinstance(rows, list) or not rows:
            raise ContractError('Declare required_artifacts: at least one relative output name that the launcher or reviewers depend on')
        declared = {}
        for row in rows:
            if isinstance(row, str):
                row = dict(name=row)
            if not isinstance(row, dict) or set(row) - {'name', 'sha256', 'role'}:
                raise ContractError('Required artifacts are names or {name, sha256, role} declarations')
            name = relative_name(row.get('name'))
            if name in declared or name in (self.completion_file, self.failure_file):
                raise ContractError('Duplicate artifact name or collision with the completion/failure file: ' + name)
            expected = row.get('sha256')
            if expected is not None and not is_sha256(expected):
                raise ContractError('Expected artifact sha256 must be 64 lowercase hex characters: ' + name)
            declared[name] = dict(sha256=expected, role=_role(row.get('role'), name))
        return declared

    @staticmethod
    def _declare_checks(rows):
        if rows is None or not isinstance(rows, list):
            raise ContractError('Declare required_checks as a list of check names; an explicit empty list is allowed')
        for name in rows:
            if not isinstance(name, str) or not name.strip() or len(name) > MAX_CHECK_CHARS:
                raise ContractError('Check names must be nonempty strings of at most 120 characters')
        if len(set(rows)) != len(rows):
            raise ContractError('Duplicate required check name')
        return list(rows)

    @staticmethod
    def _declare_sources(rows, source_root):
        if rows is None or not isinstance(rows, list):
            raise ContractError('Declare protected_sources as a list of {path, sha256, role}; use an explicit empty list when the worker reads no pinned source')
        declared, seen = [], set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) - {'path', 'sha256', 'role'}
                    or not isinstance(row.get('path'), str) or not row['path']):
                raise ContractError('Protected sources are {path, sha256, role} declarations')
            if not is_sha256(row.get('sha256')):
                raise ContractError('Protected source sha256 must be 64 lowercase hex characters: ' + row['path'])
            path = Path(row['path'])
            if not path.is_absolute():
                if source_root is None:
                    raise ContractError('A relative protected source needs source_root: ' + row['path'])
                path = Path(source_root) / path
            key = str(path.resolve())
            if key in seen:
                raise ContractError('Duplicate protected source: ' + row['path'])
            seen.add(key)
            declared.append(dict(path=str(path), sha256=row['sha256'], role=_role(row.get('role'), row['path'])))
        return declared

    # Environment ------------------------------------------------------------------

    def _resolve_root(self):
        if self.root is None:
            root = self.output_dir.resolve()
            if not root.is_dir():
                raise ContractError('Output directory does not exist: ' + str(self.output_dir))
            self.root = root
        return self.root

    def _verify_sources(self):
        rows = []
        for source in self.protected_sources:
            path = Path(source['path'])
            row = dict(source)
            if not path.is_file():
                row['status'] = 'missing'
            else:
                actual = file_sha256(path)
                row.update(actual_sha256=actual, bytes=path.stat().st_size,
                           status='verified' if actual == source['sha256'] else 'changed')
            rows.append(row)
        return rows

    @staticmethod
    def _source_failures(rows):
        return [dict(kind='changed_source' if s['status'] == 'changed' else 'missing_source', path=s['path'],
                     expected_sha256=s['sha256'], actual_sha256=s.get('actual_sha256'))
                for s in rows if s['status'] != 'verified']

    def preflight(self):
        """Verify the output directory, bounded names and protected sources before any work."""
        root = self._resolve_root()
        completion = bounded_path(root, self.completion_file, 'completion_file')
        bounded_path(root, self.failure_file, 'failure_file')
        for name in self.required_artifacts:
            bounded_path(root, name)
        sources = self._verify_sources()
        failures = self._source_failures(sources)
        if failures:
            report, path = self._failure_report(failures, sources=sources, stage='preflight')
            raise ContractFailure('Protected source verification failed before work; see ' + str(path), report, path)
        self.preflight_report = dict(status='ready', output_dir=str(root), completion_file=self.completion_file,
                                     failure_file=self.failure_file, existing_completion_file=completion.exists(),
                                     required_artifacts=sorted(self.required_artifacts), required_checks=list(self.required_checks),
                                     sources=[{k: v for k, v in s.items() if k != 'actual_sha256'} for s in sources], label=self.label)
        return self.preflight_report

    # Work ---------------------------------------------------------------------------

    def check(self, name, passed, detail=None):
        """Record a named check result. Only a literal True counts; a failure cannot be cleared later."""
        if not isinstance(name, str) or not name.strip() or len(name) > MAX_CHECK_CHARS:
            raise ContractError('Check names must be nonempty strings of at most 120 characters')
        if passed is not True and passed is not False:
            raise ContractError('Check results must be literal booleans: ' + name)
        if detail is not None:
            _jsonable(detail, 'Check detail')
        previous = self._recorded.get(name)
        if previous is not None and previous['passed'] is False and passed:
            raise ContractError('A failed check cannot be cleared by a later pass; record a separately named verification: ' + name)
        row = self._recorded.setdefault(name, dict(passed=passed, details=[]))
        row['passed'] = passed
        if detail is not None:
            row['details'].append(detail)
        return passed

    def artifact(self, name, role=None):
        """Attach a role to a required artifact or declare an additional produced file that must exist at finalize."""
        relative_name(name)
        if name in (self.completion_file, self.failure_file):
            raise ContractError('The completion/failure file is written by the contract, not declared as an artifact: ' + name)
        _role(role, name)
        if name in self.required_artifacts:
            if role is not None:
                self.required_artifacts[name]['role'] = role
        else:
            self._additional[name] = role
        return name

    def fail(self, reason, detail=None):
        """Retain an explicit worker failure report and refuse completion."""
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError('Failure reason required')
        if detail is not None:
            _jsonable(detail, 'Failure detail')
        report, path = self._failure_report([dict(kind='worker_failure', reason=reason, detail=detail)], stage='worker')
        raise ContractFailure('Worker declared failure: ' + reason, report, path)

    # Finalization -------------------------------------------------------------------

    def _extra(self, extra):
        if extra is None:
            return {}
        if not isinstance(extra, dict) or any(not isinstance(k, str) or not k for k in extra):
            raise ContractError('extra must map nonempty string keys to owner fields')
        reserved = sorted(set(extra) & RESERVED)
        if reserved:
            raise ContractError('extra cannot override contract fields: ' + ', '.join(reserved))
        return _jsonable(dict(extra), 'extra')

    def _artifact_row(self, root, name, failures, required):
        declared = self.required_artifacts.get(name, {})
        role = declared.get('role') if required else self._additional.get(name)
        expected = declared.get('sha256')
        try:
            path = bounded_path(root, name)
        except ContractError as error:
            failures.append(dict(kind='invalid_artifact_path', name=name, reason=str(error)))
            return dict(path=name, role=role, required=required, status='invalid')
        if not path.is_file():
            failures.append(dict(kind='missing_artifact', name=name, required=required))
            return dict(path=name, role=role, required=required, status='missing', expected_sha256=expected)
        actual = file_sha256(path)
        row = dict(path=name, role=role, required=required, sha256=actual, bytes=path.stat().st_size, status='verified')
        if expected is not None and expected != actual:
            row.update(status='mismatch', expected_sha256=expected)
            failures.append(dict(kind='artifact_hash_mismatch', name=name, expected_sha256=expected, actual_sha256=actual))
        return row

    def _manifest(self, root, artifacts, checks, sources, extra):
        return dict(extra, schema_version=SCHEMA_VERSION, kind=MANIFEST_KIND, contract_status='satisfied',
                    completion_file=self.completion_file, output_dir=str(root), label=self.label,
                    contract=dict(required_artifacts=[dict(name=n, sha256=self.required_artifacts[n]['sha256']) for n in sorted(self.required_artifacts)],
                                  required_checks=list(self.required_checks),
                                  protected_sources=[dict(path=s['path'], sha256=s['sha256'], role=s['role']) for s in self.protected_sources],
                                  preflight_run=self.preflight_report is not None),
                    artifacts=artifacts, checks=checks,
                    sources=[dict(path=s['path'], sha256=s['sha256'], bytes=s['bytes'], role=s['role'], status='verified') for s in sources])

    @staticmethod
    def _comparable(value):
        return {k: v for k, v in value.items() if k != 'utc'} if isinstance(value, dict) else value

    def _existing(self, completion, manifest, failures):
        """Return an identical existing manifest, or record a conflict that preserves the existing file."""
        if completion.is_symlink() or completion.is_junction():
            failures.append(dict(kind='conflicting_existing_manifest', path=str(completion), reason='completion path is a link or junction'))
            return None
        if completion.is_dir():
            failures.append(dict(kind='conflicting_existing_manifest', path=str(completion), reason='completion path is a directory'))
            return None
        if not completion.exists():
            return None
        raw = completion.read_bytes()
        try:
            existing = json.loads(raw)
        except ValueError:
            existing = None
        if isinstance(existing, dict) and self._comparable(existing) == self._comparable(manifest):
            return existing
        failures.append(dict(kind='conflicting_existing_manifest', path=str(completion), sha256=digest(raw),
                             reason='An existing completion file differs from this finalization; it was preserved unchanged'))
        return None

    def _failure_report(self, failures, artifacts=None, checks=None, sources=None, stage='finalize'):
        root = self._resolve_root()
        report = dict(schema_version=SCHEMA_VERSION, kind=FAILURE_KIND, contract_status='unsatisfied',
                      completion_file=self.completion_file, completion_written=False, output_dir=str(root), label=self.label,
                      stage=stage, failures=failures, artifacts=artifacts or {}, checks=checks or {}, sources=sources or [], utc=_utc())
        stem, dot, suffix = self.failure_file.rpartition('.')
        path = root / self.failure_file
        for n in range(2, MAX_FAILURE_REPORTS + 2):
            if not path.exists() and not path.is_symlink():
                break
            path = root / ((stem + '-' + str(n) + '.' + suffix) if dot else (self.failure_file + '-' + str(n)))
        else:
            raise ContractError('Too many retained failure reports in the output directory')
        _atomic_create(path, canonical(report))
        return report, path

    def finalize(self, extra=None, strict=True):
        """Verify artifacts, checks and sources, then write the completion file atomically as the last step.

        Returns {status: completed, path, repeat, manifest}. On any failure a failure report is
        written and ContractFailure (ContractConflict for a differing existing manifest) is raised,
        or with strict=False {status: failed, path, report} is returned. No success manifest is
        ever written over failed checks, missing artifacts, changed sources or a different existing file.
        """
        extra = self._extra(extra)
        root = self._resolve_root()
        failures, artifacts, checks = [], {}, {}
        if self.preflight_report is None:
            failures.append(dict(kind='missing_preflight',reason='Run the declared preflight before worker effects.'))
        for name in sorted(self.required_artifacts):
            artifacts[name] = self._artifact_row(root, name, failures, required=True)
        for name in sorted(self._additional):
            artifacts[name] = self._artifact_row(root, name, failures, required=False)
        for name in self.required_checks:
            row = self._recorded.get(name)
            if row is None:
                failures.append(dict(kind='unrecorded_check', name=name))
                checks[name] = dict(passed=None, details=[], required=True)
            else:
                checks[name] = dict(row, required=True)
                if row['passed'] is False:
                    failures.append(dict(kind='failed_check', name=name))
        for name, row in self._recorded.items():
            if name not in self.required_checks:
                checks[name] = dict(row, required=False)
                if row['passed'] is False:
                    failures.append(dict(kind='failed_check', name=name))
        sources = self._verify_sources()
        failures.extend(self._source_failures(sources))
        completion = root / self.completion_file
        if not failures:
            manifest = self._manifest(root, artifacts, checks, sources, extra)
            existing = self._existing(completion, manifest, failures)
            if existing is not None:
                return dict(status='completed', path=str(completion), repeat=True, manifest=existing)
        if failures:
            report, path = self._failure_report(failures, artifacts, checks, sources)
            if strict:
                conflict = any(f['kind'] == 'conflicting_existing_manifest' for f in failures)
                raise (ContractConflict if conflict else ContractFailure)('Completion refused; see ' + str(path), report, path)
            return dict(status='failed', path=str(path), repeat=False, report=report)
        manifest['utc'] = _utc()
        try:
            _atomic_create(completion, canonical(manifest))
        except FileExistsError:
            existing=self._existing(completion,manifest,failures)
            if existing is not None:return dict(status='completed',path=str(completion),repeat=True,manifest=existing)
            report,path=self._failure_report(failures,artifacts,checks,sources)
            if strict:raise ContractConflict('Concurrent completion preserved; see '+str(path),report,path)
            return dict(status='failed',path=str(path),repeat=False,report=report)
        return dict(status='completed', path=str(completion), repeat=False, manifest=manifest)


def verify_manifest(path):
    """Re-verify an existing completion manifest now: artifact bytes, protected sources and recorded checks. No writes."""
    path = Path(path)
    if path.is_symlink() or path.is_junction() or not path.is_file():
        return dict(disposition='absent', path=str(path))
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except ValueError:
        return dict(disposition='invalid', path=str(path), reason='not JSON')
    if not isinstance(value, dict) or value.get('kind') != MANIFEST_KIND or value.get('schema_version') != SCHEMA_VERSION:
        return dict(disposition='not_contract_manifest', path=str(path), sha256=digest(raw))
    root = path.parent.resolve()
    problems, artifacts = [], {}
    rows = value.get('artifacts')
    if not isinstance(rows, dict) or not rows:
        rows = {}
        problems.append('no artifacts listed')
    for name, row in sorted(rows.items()):
        if not isinstance(row, dict) or not isinstance(row.get('path'), str) or not is_sha256(row.get('sha256')):
            artifacts[name] = 'invalid_reference'
        else:
            try:
                target = bounded_path(root, row['path'])
            except ContractError:
                artifacts[name] = 'invalid_name'
            else:
                if not target.is_file():
                    artifacts[name] = 'missing'
                else:
                    artifacts[name] = 'verified' if file_sha256(target) == row['sha256'] else 'mismatch'
        if artifacts[name] != 'verified':
            problems.append('artifact ' + name + ' ' + artifacts[name])
    sources = []
    source_rows=value.get('sources')
    if not isinstance(source_rows,list):
        problems.append('invalid source inventory'); source_rows=[]
    for row in source_rows:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str) or not is_sha256(row.get('sha256')):
            status = 'invalid_reference'
        else:
            source = Path(row['path'])
            status = 'missing' if not source.is_file() else ('unchanged' if file_sha256(source) == row['sha256'] else 'changed')
        sources.append(dict(path=row.get('path') if isinstance(row, dict) else None,
                            sha256=row.get('sha256') if isinstance(row, dict) else None, status=status))
        if status != 'unchanged':
            problems.append('protected source ' + str(sources[-1]['path']) + ' ' + status)
    contract = value.get('contract') if isinstance(value.get('contract'), dict) else {}
    try:
        declared=WorkerContract(root,completion_file=value.get('completion_file'),
            required_artifacts=contract.get('required_artifacts'),required_checks=contract.get('required_checks'),
            protected_sources=contract.get('protected_sources'),label=value.get('label'))
        if path.name!=declared.completion_file:problems.append('completion filename differs from declared contract')
        if contract.get('preflight_run') is not True:problems.append('declared preflight was not run')
        for name,spec in declared.required_artifacts.items():
            row=rows.get(name,{})
            if not isinstance(row,dict) or row.get('path')!=name:problems.append('required artifact path differs from declared name: '+name)
            elif spec['sha256'] is not None and row.get('sha256')!=spec['sha256']:
                problems.append('required artifact hash differs from its predeclared hash: '+name)
        declared_sources={str(Path(s['path']).resolve()):s['sha256'] for s in declared.protected_sources}
        listed_sources={str(Path(s['path']).resolve()):s['sha256'] for s in source_rows if isinstance(s,dict) and isinstance(s.get('path'),str)}
        if declared_sources!=listed_sources:problems.append('protected source inventory differs from declared contract')
    except (ContractError,TypeError,KeyError) as error:
        problems.append('invalid completion contract: '+str(error))
    checks = value.get('checks') if isinstance(value.get('checks'), dict) else {}
    required_checks = contract.get('required_checks') if isinstance(contract.get('required_checks'), list) else []
    checks_ok = (all(isinstance(n,str) and isinstance(checks.get(n), dict) and checks[n].get('passed') is True for n in required_checks)
                 and all(isinstance(c, dict) and c.get('passed') is True for c in checks.values()))
    required_rows=contract.get('required_artifacts')
    required_artifacts = [a.get('name') for a in required_rows if isinstance(a, dict)] if isinstance(required_rows,list) else []
    listed = all(isinstance(n,str) and n in rows for n in required_artifacts)
    if not checks_ok:
        problems.append('a required or recorded check is not passed')
    if not listed:
        problems.append('a required artifact is not listed')
    if value.get('contract_status') != 'satisfied':
        problems.append('contract_status is not satisfied')
    return dict(disposition='verified' if not problems else 'mismatch', path=str(path), sha256=digest(raw),
                label=value.get('label'), completion_file=value.get('completion_file'), artifacts=artifacts, sources=sources,
                checks_all_passed=checks_ok, required_artifacts_listed=listed, contract_status=value.get('contract_status'),
                problems=problems,
                limits='Bytes match the manifest now; content correctness, native quality and appearance are not established.')
