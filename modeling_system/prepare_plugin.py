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
    interpreter=Path(python or sys.executable).resolve()
    if not interpreter.is_file(): raise ValueError('Select an installed Python interpreter')
    files=['.codex-plugin/plugin.json','launch.py']+[p.relative_to(source).as_posix() for p in (source/'skills').rglob('*') if p.is_file()]
    for relative in files:
        target=destination/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source/relative,target)
    config={'mcpServers':{'modeling-workbench':{'command':str(interpreter),
        'args':[str(destination/'launch.py'),'--workspace',str(workspace)]}}}
    (destination/'.mcp.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    return destination


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('destination'); p.add_argument('--workspace',required=True); p.add_argument('--python')
    a=p.parse_args(); print(prepare(a.destination,a.workspace,a.python))
