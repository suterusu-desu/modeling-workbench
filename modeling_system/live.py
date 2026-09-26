"""Bind the reference adapter to a character workspace and launch a live Blender that serves it.

    python -m modeling_system.live bind --workspace <dir> --reference reference.json [--owner name] [--port 9876]
    python -m modeling_system.live launch --workspace <dir> --blender <blender executable> [--file x.blend] [--background]

`bind` writes the binding's native adapter section: the installed reference adapter pinned by hash, its operations, the
`reference` configuration (working object, character, guides with roles, controls, optional restore hook) and the
restore hook's file pinned beside it. Rebind after editing the hook or reinstalling the package: stale pins are refused.
`launch` starts Blender with an isolated user profile under the workspace (`runtime/live/profile`), without the user's
startup file or auto-run scripts, and runs `live_bridge` in it. A visible Blender keeps the work in view while the
workbench drives it; `--background` serves without a window.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from . import reference_adapter

ADAPTER = Path(reference_adapter.__file__).resolve()
BRIDGE = Path(__file__).with_name('live_bridge.py').resolve()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_reference_adapter(workspace, reference, *, owner=None, port=None):
    """Write the reference adapter into `modeling-workspace.json`; returns the native configuration written."""
    workspace = Path(workspace).resolve()
    path = workspace / 'modeling-workspace.json'
    binding = json.loads(path.read_text(encoding='utf-8-sig'))
    reference = json.loads(json.dumps(reference))
    if not isinstance(reference, dict) or not reference.get('working'):
        raise ValueError('The reference configuration needs the working object')
    for name, guide in reference.get('guides', {}).items():
        if not isinstance(guide, dict) or not guide.get('object') or guide.get('role') not in reference_adapter.ROLES:
            raise ValueError(f'Guide {name!r} needs an object and a role: {", ".join(reference_adapter.ROLES)}')
    dependencies = []
    if reference.get('restore'):
        hook = (workspace / reference['restore']['path']).resolve(strict=True)
        dependencies.append({'path': str(hook), 'sha256': _sha(hook)})
    owner = owner or binding.get('native_owner')
    if not owner:
        raise ValueError('Name the native owner (--owner) or keep the one in the binding')
    config = {'entrypoint': {'path': str(ADAPTER), 'sha256': _sha(ADAPTER)}, 'dependencies': dependencies,
              'operations': list(reference_adapter.OPERATIONS), 'reference': reference}
    binding.update(native_adapter='blender_json_v1', native_owner=owner,
                   native_bridge_port=int(port or binding.get('native_bridge_port') or 9876),
                   native_configuration=config)
    path.write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')
    return config


def launch(workspace, blender, *, file=None, background=False, port=None, log=None):
    """Start Blender with the live bridge in an isolated profile; returns the process."""
    workspace = Path(workspace).resolve(); live = workspace / 'runtime' / 'live'; profile = live / 'profile'
    env = os.environ.copy(); env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None); env['PYTHONNOUSERSITE'] = '1'
    for key, folder in [('RESOURCES', ''), ('CONFIG', 'config'), ('SCRIPTS', 'scripts'), ('EXTENSIONS', 'extensions'),
                        ('DATAFILES', 'datafiles')]:
        (profile / folder).mkdir(parents=True, exist_ok=True); env['BLENDER_USER_' + key] = str(profile / folder)
    command = [str(blender), '--factory-startup', '--disable-autoexec']
    command += [str(Path(file).resolve(strict=True))] if file else []
    command += ['--background'] if background else []
    command += ['--python', str(BRIDGE), '--', '--workspace', str(workspace)] + (['--port', str(port)] if port else [])
    log = Path(log) if log else live / ('bridge-background.log' if background else 'bridge-window.log')
    flags = 0
    if os.name == 'nt' and not background:
        flags = 0x00000008 | 0x00000200        # a detached window that outlives this process
    with log.open('w', encoding='utf-8') as out:
        return subprocess.Popen(command, cwd=str(live), env=env, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)


def wait_ready(port, timeout=60., process=None):
    """Wait until the bridge accepts connections on 127.0.0.1:port (or the process exits)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'Blender exited with code {process.returncode} before the bridge was ready')
        try:
            socket.create_connection(('127.0.0.1', int(port)), timeout=1.).close()
            return True
        except OSError:
            time.sleep(.25)
    raise TimeoutError(f'The live bridge did not open port {port} within {timeout} s')


def shutdown(port, timeout=10.):
    """Stop a background bridge."""
    with socket.create_connection(('127.0.0.1', int(port)), timeout=timeout) as client:
        client.sendall(json.dumps({'type': 'shutdown'}).encode('utf-8'))
        return json.loads(client.recv(65536).decode('utf-8'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('bind'); b.add_argument('--workspace', required=True); b.add_argument('--reference', required=True)
    b.add_argument('--owner'); b.add_argument('--port', type=int)
    l = sub.add_parser('launch'); l.add_argument('--workspace', required=True); l.add_argument('--blender', required=True)
    l.add_argument('--file'); l.add_argument('--background', action='store_true'); l.add_argument('--port', type=int)
    args = parser.parse_args(argv)
    if args.command == 'bind':
        config = bind_reference_adapter(args.workspace, json.loads(Path(args.reference).read_text(encoding='utf-8')),
                                        owner=args.owner, port=args.port)
        print(json.dumps({'bound': config['entrypoint'], 'operations': config['operations']}, indent=1))
    else:
        process = launch(args.workspace, args.blender, file=args.file, background=args.background, port=args.port)
        print(json.dumps({'pid': process.pid}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
