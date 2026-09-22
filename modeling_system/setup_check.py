"""Offline installation checks, without reading credentials or contacting Blender."""
import argparse
import importlib
import json
from pathlib import Path
import sys

from .bindings import load_binding
from .credentials import credential_status
from .skill_bundle import verify_dependencies
from .store import digest


def inspect_setup(workspace=None, *, plugin=None, ledger=None):
    checks = []
    def check(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
    check('python', sys.version_info >= (3, 12), 'Python 3.12 or newer')
    for name in ('numpy', 'scipy', 'PIL', 'mcp', 'imageio_ffmpeg'):
        try:
            importlib.import_module(name)
            check(name, True, 'Import succeeded in this interpreter')
        except (ImportError, OSError):
            check(name, False, 'Install the package dependencies in this interpreter')
    skills = verify_dependencies()
    check('bundled_skills', skills['status'] == 'verified', skills)
    binding = load_binding(Path(workspace).resolve()) if workspace else None
    native = {'status': 'unconfigured', 'live_verified': False}
    if binding:
        check('workspace', binding['character'] != 'unbound', 'Explicit separate workspace binding')
        config = binding.get('native_configuration', {})
        refs = [config.get('entrypoint'), *config.get('dependencies', [])]
        problems = []
        if binding.get('native_adapter') == 'blender_json_v1':
            for ref in refs:
                if not ref or not isinstance(ref.get('path'), str) or not ref.get('sha256'):
                    problems.append('Missing pinned adapter dependency')
                    continue
                path = (Path(workspace)/ref['path']).resolve()
                if not path.is_file() or digest(path.read_bytes()) != ref['sha256']:
                    problems.append('Missing or changed adapter dependency')
            if not config.get('operations'):
                problems.append('No advertised native operations')
            native.update(status='configured' if not problems else 'invalid', problems=problems)
        elif binding.get('native_adapter') != 'unconfigured':
            native.update(status='unsupported', problems=['Unsupported native adapter kind'])
    if plugin:
        from .installation import verify
        from .service import ModelingService
        if not workspace:
            raise ValueError('Plugin verification requires its explicit workspace')
        try:
            report = verify(ModelingService(workspace=workspace), plugin)
            check('plugin', report['status'] == 'verified fresh launch' and report.get('source_matches_probe')
                  and report['probe']['runtime']['workspace'] == str(Path(workspace).resolve()), report['status'])
        except (OSError, ValueError, KeyError, RuntimeError):
            check('plugin', False, 'Regenerate plugin configuration using this interpreter and workspace')
    provider = {'credentials': credential_status(), 'ledger': 'unconfigured', 'network_verified': False}
    if ledger:
        from .provider_recovery import budget_limits
        path = Path(ledger)/'budget.json'
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('policy', {}).get('mode') != 'normal_use':
                raise ValueError('Explicit normal-use authorization required')
            budget_limits(record)
            provider['ledger'] = record['status']
        except (OSError, ValueError, KeyError):
            provider['ledger'] = 'invalid'
    return {'core_ready': all(row['ok'] for row in checks), 'checks': checks,
            'jev': provider, 'native': native,
            'limits': 'Offline checks do not verify account credits, a running Blender adapter, rig bindings or appearance.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace')
    parser.add_argument('--plugin')
    parser.add_argument('--ledger')
    args = parser.parse_args()
    report = inspect_setup(args.workspace, plugin=args.plugin, ledger=args.ledger)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['core_ready'] else 1)


if __name__ == '__main__':
    main()
