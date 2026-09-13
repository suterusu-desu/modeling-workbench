"""Explicit project/native bindings; portable core contains no character anatomy."""
from pathlib import Path
import json
from .store import digest, canonical


def load_binding(workspace):
    path = Path(workspace) / 'modeling-workspace.json'
    if not path.exists():
        return {'schema_version': 1, 'character': 'unbound', 'authority': [],
                'native_adapter': 'unconfigured', 'missing': [str(path)]}
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if value.get('schema_version') != 1 or not value.get('character'):
        raise ValueError('Workspace binding requires schema_version=1 and character')
    return dict(value, binding_path=str(path), binding_sha256=digest(path.read_bytes()))


def fingerprint_files(workspace, entries):
    rows = []
    for entry in entries:
        path = (Path(workspace) / (entry['path'] if isinstance(entry, dict) else entry)).resolve()
        rows.append(dict(path=str(path), sha256=digest(path.read_bytes()) if path.is_file() else None,
                         available=path.is_file(),identity=entry.get('id') if isinstance(entry,dict) else None,
                         role=entry.get('role') if isinstance(entry,dict) else None))
    return dict(files=rows, revision=digest(canonical(rows)))


def compare_authority(entered,current):
    """Compare content independently from rebinding; legacy roles match only if unique."""
    changes=[];locations=[]
    for row in current:
        candidates=[e for e in entered if row.get('identity') and e.get('authority_id')==row['identity']]
        if not candidates:candidates=[e for e in entered if e['path']==row['path']]
        if not candidates and row.get('role') and sum(r.get('role')==row['role'] for r in current)==1:
            candidates=[e for e in entered if e.get('role')==row['role']]
        previous=candidates[0] if len(candidates)==1 else None
        if previous is None:
            changes.append(dict(row,comparison='No unambiguous authority identity at episode entry; inspect source'))
            continue
        changed=previous['sha256']!=row['sha256']
        if changed:changes.append(dict(row,previous_sha256=previous['sha256'],comparison='content changed'))
        if previous['path']!=row['path']:
            locations.append(dict(identity=row.get('identity'),role=row.get('role'),before=previous['path'],after=row['path'],content_changed=changed))
    return changes,locations


def authority_path(workspace,binding,role,default):
    entry=next((e for e in binding.get('authority',[]) if e.get('role')==role),None)
    return Path(workspace)/(entry['path'] if entry else default)
