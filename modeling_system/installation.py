"""Observed source/cache/process verification; installer remains Codex CLI."""
from pathlib import Path
import json
import os
import subprocess
from .store import digest


def verify(service,cache_root):
    cache=Path(cache_root).resolve();source=Path(__file__).parent/'plugin'
    mismatches=[];checked=[]
    for p in [source/'launch.py',*sorted((source/'skills').rglob('*'))]:
        if not p.is_file():continue
        relative=p.relative_to(source);target=cache/relative
        if not target.is_file() or digest(target.read_bytes())!=digest(p.read_bytes()):mismatches.append(str(relative))
        checked.append(str(relative))
    manifest=json.loads((cache/'.codex-plugin/plugin.json').read_text(encoding='utf-8-sig'))
    config=json.loads((cache/'.mcp.json').read_text(encoding='utf-8-sig'))['mcpServers']['modeling-workbench']
    if mismatches:
        return dict(status='mismatch',cache=str(cache),version=manifest['version'],mismatches=mismatches,
                    recovery='Materialize current source and reinstall with plugin helper/CLI before probing')
    # Exercise the installed arguments verbatim, as Codex does for MCP launch.
    args=[str(config['command']),*config['args'],'--probe']
    process=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',timeout=45,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    if process.returncode:raise RuntimeError('Installed launch probe failed: '+process.stderr[-2000:])
    probe=json.loads(process.stdout)
    return dict(status='verified fresh launch',version=manifest['version'],cache=str(cache),checked_files=checked,
                probe=probe,source_matches_probe=probe['runtime']['loaded']['revision']==service.runtime_status()['disk_revision'],
                installed_manifest_sha256=digest((cache/'.codex-plugin/plugin.json').read_bytes()),
                operator_adoption='Not implied. A fresh operator must read the installed procedure and use this route.')
