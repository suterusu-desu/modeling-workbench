"""Materialize a portable plugin with explicit local interpreter and workspace."""
from pathlib import Path
import argparse
import json
import shutil
import sys


def prepare(destination, workspace=None, python=None):
    source=Path(__file__).parent/'plugin'; destination=Path(destination).resolve()
    if workspace is None: raise ValueError('Select an explicit private workspace')
    workspace=Path(workspace).resolve()
    if not (workspace/'modeling-workspace.json').is_file(): raise ValueError('Initialize the selected workspace first')
    # Resolving a POSIX venv symlink selects the base interpreter and loses its
    # installed dependencies. Preserve the chosen executable's environment.
    interpreter=Path(python or sys.executable).expanduser().absolute()
    if not interpreter.is_file(): raise ValueError('Select an installed Python interpreter')
    from .skill_bundle import verify_dependencies
    if verify_dependencies()['status'] != 'verified':
        raise ValueError('Bundled skill dependency failed integrity verification')
    if destination == workspace or destination.is_relative_to(Path(__file__).parent.resolve()):
        raise ValueError('Choose a local plugin directory separate from the package and workspace root')
    files=['.codex-plugin/plugin.json','launch.py']+[p.relative_to(source).as_posix() for p in (source/'skills').rglob('*') if p.is_file()]
    config={'mcpServers':{'modeling-workbench':{'command':str(interpreter),
        'args':[str(destination/'launch.py'),'--workspace',str(workspace)]}}}
    payloads={relative:(source/relative).read_bytes() for relative in files}
    payloads['.mcp.json']=(json.dumps(config,indent=2)+'\n').encode('utf-8')
    for relative,data in payloads.items():
        target=destination/relative
        if target.exists() and (not target.is_file() or target.read_bytes()!=data):
            raise FileExistsError('Existing plugin differs; preserve it and choose a new destination')
    for relative,data in payloads.items():
        target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('destination'); p.add_argument('--workspace',required=True); p.add_argument('--python')
    a=p.parse_args(); print(prepare(a.destination,a.workspace,a.python))
