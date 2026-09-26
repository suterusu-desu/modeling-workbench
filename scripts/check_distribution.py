"""Check tracked source boundaries; optional private deny terms never enter the repo."""
from pathlib import Path
import json,os,re,subprocess,sys,tomllib

root=Path(__file__).resolve().parents[1]
tracked=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root,capture_output=True)
if tracked.returncode==0 and tracked.stdout:
    files=[root/p for p in tracked.stdout.decode().split('\0') if p]
else:
    files=[p for p in root.rglob('*') if p.is_file() and not any(x in p.parts for x in
        ('.git','.venv','.runtime','.modeling','__pycache__','build','dist')) and not any(x.endswith('.egg-info') for x in p.parts)]
allowed_roots={'modeling_system','scripts','.github','docs'}
allowed_files={'README.md','AGENTS.md','DEVELOPING.md','DESIGN.md','IMPLEMENTATION-QUEUE.md','pyproject.toml','.gitignore','.gitattributes'}
errors=[]
metadata=tomllib.loads((root/'pyproject.toml').read_text(encoding='utf-8'))
distribution=tomllib.loads((root/'modeling_system/distribution-pyproject.toml').read_text(encoding='utf-8'))
plugin=json.loads((root/'modeling_system/plugin/.codex-plugin/plugin.json').read_text(encoding='utf-8'))
if metadata!=distribution or plugin['version']!=metadata['project']['version']:
    errors.append(('package metadata','source, archive and plugin versions/dependencies must agree'))
inventory=json.loads((root/'modeling_system/distribution-files.json').read_text(encoding='utf-8'))['files']
for p in (root/'modeling_system').glob('*.py'):
    if p.name not in inventory: errors.append((p.name,'Python module missing from tools export'))
private_terms=[t.strip().casefold() for t in os.environ.get('MODELING_PRIVATE_TERMS','').split(',') if t.strip()]
for p in files:
    rel=p.relative_to(root)
    if (len(rel.parts)==1 and rel.as_posix() not in allowed_files) or (len(rel.parts)>1 and rel.parts[0] not in allowed_roots):
        errors.append((rel,'outside source inventory'));continue
    if p.suffix not in ('.py','.md','.json','.toml','.txt','.yml','.yaml','') and p.name not in allowed_files:
        errors.append((rel,'unsupported source file type'));continue
    if p.name in ('modeling-workspace.json','.mcp.json') or any(x in rel.parts for x in ('migration','evidence','bound-authority','assets','checkpoints')):
        errors.append((rel,'private workspace material'));continue
    try:text=p.read_text(encoding='utf-8')
    except UnicodeError: errors.append((rel,'binary input'));continue
    if tracked.returncode==0 and tracked.stdout:
        raw=subprocess.run(['git','hash-object','--no-filters',str(p)],cwd=root,capture_output=True)
        filtered=subprocess.run(['git','hash-object','--path='+rel.as_posix(),str(p)],cwd=root,capture_output=True)
        if raw.returncode or filtered.returncode or raw.stdout!=filtered.stdout:
            errors.append((rel,'working bytes differ from Git filters; normalize before building or pinning a release'))
    if re.search(r'[A-Za-z]:[\\/]+Users[\\/]+[^\\/\s]+|/(?:home|Users)/[^/\s]+',text):
        errors.append((rel,'personal absolute location'))
    if any(term in text.casefold() or term in rel.as_posix().casefold() for term in private_terms):
        errors.append((rel,'private marker detected'))
    if re.search(r'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',text):
        errors.append((rel,'credential-shaped material'))
if errors:
    for path,reason in errors:print(str(path)+': '+reason)
    raise SystemExit(1)
print('Source inventory and privacy-pattern checks passed ('+str(len(files))+' files).')
