"""Observed loaded-source provenance, separate from disk/cache/operator claims."""
from pathlib import Path
import inspect
import os
import sys
from .store import digest, canonical


def source_manifest(root=None):
    root = Path(root or Path(__file__).parent)
    paths = list(root.glob('*.py')) + list(root.glob('*.json'))
    paths += list((root / 'plugin').rglob('*.md'))
    paths += list((root / 'recipe_templates').glob('*.json'))
    rows = {p.relative_to(root).as_posix(): digest(p.read_bytes()) for p in sorted(paths)}
    return dict(files=rows, revision=digest(canonical(rows)))


# Captured during module import, not from a caller-supplied version string.
LOADED = source_manifest()


def runtime_report(service):
    disk = source_manifest()
    methods = {n: str(inspect.signature(getattr(service, n))) for n in service.operations()}
    return dict(schema_version=1, loaded=LOADED, disk_revision=disk['revision'],
                source_matches_loaded=disk['revision'] == LOADED['revision'],
                schema_revision=digest(canonical(methods)), signatures=methods,
                python=sys.executable, pid=os.getpid(), implementation_root=str(Path(__file__).parent.resolve()),
                workspace=str(service.workspace), store=str(service.store.root),
                installed_cache='not inferred; verify a materialized installation manifest separately',
                operator_context='not observable from a version assertion; requires actual use receipt',
                native='not contacted; owner and native adapter version remain separate')
