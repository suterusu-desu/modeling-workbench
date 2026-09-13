"""Private workspace transfer and explicit evidence/readiness manifests. Not a shareable source release."""
from pathlib import Path
import json
import sys
import zipfile
import io
from .bindings import load_binding
from .store import digest,canonical,atomic_write
from .runtime import source_manifest


def evidence_manifest(service,episode=None):
    binding=load_binding(service.workspace)
    rows=[]
    for entry in binding.get('evidence',[]):
        p=(service.workspace/entry['path']).resolve()
        rows.append(dict(entry,locator=str(p),available=p.is_file(),sha256=digest(p.read_bytes()) if p.is_file() else None,
                         bytes=p.stat().st_size if p.is_file() else None,embedded=False))
    records={};blobs={};missing=[];workflows={};operation_files=[];pending_operations=[]
    def walk(value):
        if isinstance(value,dict):
            if value.get('kind') in ('record','workflow') and value.get('id'):
                expected=(service.store.root/'records' if value['kind']=='record' else service.ledger.root)/(value['id']+'.json')
                if not expected.is_file():missing.append(dict(link=value,reason='Explicit linked record/workflow missing'))
            if set(('sha256','path'))<=value.keys() and str(value['path']).replace('\\','/').startswith('assets/'):
                if value['sha256'] in blobs:return
                try:
                    p=service.store.resolve_blob(value)
                    blobs[value['sha256']]=dict(value,locator=str(p),available=True,embedded=False,
                                               access='private local evidence store')
                except (ValueError,OSError) as error:
                    missing.append(dict(asset=value,reason=str(error)))
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value):
            if value in records:return
            # Content IDs and workflow handles are distinguished by actual store
            # existence. Unknown hashes remain opaque hashes, never guessed records.
            p=service.store.root/'records'/(value+'.json')
            if p.is_file():
                payload=service.store.get(value);records[value]=dict(locator=str(p),sha256=digest(p.read_bytes()),embedded=False)
                walk(payload)
            elif (service.ledger.root/(value+'.json')).is_file():
                if value in workflows:return
                workflows[value]=service.ledger.read(value)['revision']
                walk(workflows[value])
    if episode:
        service.ledger.read(episode)  # A requested missing episode is not an empty successful export.
        walk(episode)
        for call in service.inspect_operations(episode)['calls']:
            from .journal import ACTIVE_CALL
            if call['handle']==ACTIVE_CALL.get():continue
            if call['record']:walk(call['record'])
            if not call['record'] or call['effect_status']=='unknown' or call['retention']=='active/in-flight':
                pending_operations.append(dict(operation=call['handle'],status=call['status'],effect_status=call['effect_status'],
                    retention=call['retention'],recovery='Original intent/results/effect receipts are included when present; reconcile actual outcome, never replay from this package'))
            for p in sorted((service.store.root/'calls'/call['handle']).rglob('*.json')):
                operation_files.append(dict(locator=str(p),path=p.relative_to(service.store.root).as_posix(),
                                            sha256=digest(p.read_bytes()),bytes=p.stat().st_size,embedded=False))
                walk(json.loads(p.read_bytes()))
        for folder in ('episode-leases','episode-completions'):
            for p in sorted((service.store.root/folder/episode).glob('*.json')):
                operation_files.append(dict(locator=str(p),path=p.relative_to(service.store.root).as_posix(),
                                            sha256=digest(p.read_bytes()),bytes=p.stat().st_size,embedded=False))
    return dict(schema_version=1,character=binding['character'],workspace=str(service.workspace),store=str(service.store.root),episode=episode,
                declared_files=rows,records=records,assets=blobs,workflows=workflows,operation_files=operation_files,missing=missing,pending_operations=pending_operations,
                coverage='Transitive immutable episode records/assets plus explicitly bound external files; no undeclared native state',
                unsupported=binding.get('limits',[]))


def readiness(service,episode=None,manifest_path=None):
    binding=load_binding(service.workspace)
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf-8-sig')) if manifest_path else evidence_manifest(service,episode)
    missing=[];changed=[]
    for e in manifest.get('declared_files',[])+list(manifest.get('assets',{}).values())+list(manifest.get('records',{}).values())+manifest.get('operation_files',[]):
        p=Path(manifest_path).resolve().parent/e['package_path'] if manifest_path and e.get('embedded') and e.get('package_path') else Path(e['locator'])
        if not p.is_file():missing.append(e)
        elif e.get('sha256') and digest(p.read_bytes())!=e['sha256']:changed.append(e)
    return dict(character=binding['character'],binding_status='bound' if binding['character']!='unbound' else 'needs explicit character binding',
                missing=missing,changed=changed,manifest_missing=manifest.get('missing',[]),
                pending_recovery=manifest.get('pending_operations',[]),
                runtime=service.runtime_status(),native_adapter=binding.get('native_adapter','unconfigured'),
                native_readiness='not contacted; installation/binding do not establish live ownership or fresh geometry',
                provider_readiness='Read bound policy/transport and actual account; package supplies no credits or dispatch authority',
                usable_for_recorded_analysis=not missing and not changed and not manifest.get('missing'),
                transfer='Core and explicit adapters available; other-character modeling quality is unproven')


def export(service,output_path,episode=None,include_evidence=False):
    path=Path(output_path).resolve()
    if path.exists():raise FileExistsError('Preserve prior package; choose a new output path')
    manifest=evidence_manifest(service,episode)
    root=Path(__file__).parent
    files={p.relative_to(root.parent).as_posix():p.read_bytes() for p in sorted(root.glob('*.py'))}
    for p in [*root.glob('*.json'),*root.glob('*.md'),*root.glob('*.toml'),root/'requirements-lock.txt']:
        if p.is_file():files['modeling_system/'+p.name]=p.read_bytes()
    for p in (root/'plugin').rglob('*'):
        if p.is_file() and p.suffix in ('.py','.md','.json','.yaml'):files[p.relative_to(root.parent).as_posix()]=p.read_bytes()
    for name in ('pyproject.toml','README.md'):
        p=root.parent/name
        if p.is_file():files[name]=p.read_bytes()
    binding=dict(load_binding(service.workspace))
    for key in ('binding_path','binding_sha256'):binding.pop(key,None)
    binding['authority']=[dict(e,path='bound-authority/'+e['path']) for e in binding.get('authority',[])]
    binding['store']='evidence'
    if episode:binding['selected_episode']=episode
    binding['native_adapter']='unconfigured'
    binding['native_configuration_required']='Declare and verify the actual native adapter/owner in the destination; never inherit a live connection from an evidence package'
    files['modeling-workspace.json']=json.dumps(binding,indent=2).encode()
    # Authority is part of the declared context; preserve exact current text,
    # leaving linked native/private assets in the explicit manifest by default.
    for entry in load_binding(service.workspace).get('authority',[]):
        p=(service.workspace/entry['path']).resolve()
        if p.is_file():files['bound-authority/'+entry['path']]=p.read_bytes()
    if include_evidence:
        if manifest['missing']:raise ValueError('Resolve missing immutable evidence before embedding an episode')
        for key,row in manifest['records'].items():
            row['package_path']='evidence/records/'+key+'.json'
            files[row['package_path']]=Path(row['locator']).read_bytes();row['embedded']=True
        for row in manifest['assets'].values():
            row['package_path']='evidence/'+row['path'].replace('\\','/')
            files[row['package_path']]=Path(row['locator']).read_bytes();row['embedded']=True
        for row in manifest['declared_files']:
            if row['available']:
                declared=Path(row['path'])
                row['package_path']=('external-evidence/'+row['sha256']+declared.suffix) if declared.is_absolute() else declared.as_posix()
                files[row['package_path']]=Path(row['locator']).read_bytes();row['embedded']=True
        for handle,revision in manifest['workflows'].items():
            files['evidence/workflow/'+handle+'.json']=canonical({'revision':revision})
        for row in manifest['operation_files']:
            row['package_path']='evidence/'+row['path']
            files[row['package_path']]=Path(row['locator']).read_bytes();row['embedded']=True
    files['evidence-manifest.json']=json.dumps(manifest,indent=2).encode()
    if include_evidence:
        binding['evidence']=[dict(e,path=row.get('package_path',e['path'])) for e,row in zip(binding.get('evidence',[]),manifest['declared_files'])]
        files['modeling-workspace.json']=json.dumps(binding,indent=2).encode()
    files['PACKAGE-README.md']=b'''# Recoverable modeling package\n\nInstall with Python 3.12+: python -m pip install .\nConfigure an explicit modeling-workspace.json for your workspace. Then run\npython -B -m modeling_system --workspace WORKSPACE decision_workspace\nRead modeling_system/plugin/skills/modeling-workbench/SKILL.md and references.\nEvidence manifest declares private locators, hashes, coverage and access.\nMissing assets must be supplied or explicitly left unsupported. No credits,\nprovider authorization, live state or other-character quality are transferred.\nNative geometry operations require an explicitly configured workspace adapter.\nThe core works without native control using an unconfigured native adapter.\n'''
    inventory={name:dict(sha256=digest(data),bytes=len(data)) for name,data in sorted(files.items())}
    package=dict(schema_version=1,source_revision=source_manifest()['revision'],files=inventory,
                 evidence_embedded=include_evidence,episode=episode,external_manifest='evidence-manifest.json')
    files['package-manifest.json']=json.dumps(package,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,data)
    atomic_write(path,buffer.getvalue())
    # Independently verify the actual archive, not the inputs passed to ZipFile.
    with zipfile.ZipFile(path) as z:
        for name,item in inventory.items():
            if digest(z.read(name))!=item['sha256']:raise ValueError('Package verification failed: '+name)
    return dict(path=str(path),sha256=digest(path.read_bytes()),bytes=path.stat().st_size,files=len(files),
                source_revision=package['source_revision'],manifest=manifest,
                transfer='Archive integrity verified; fresh-context operator and character quality remain separate evidence')
