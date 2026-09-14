"""Owner-run append with native dependency closure, content checks and checkpoints.

Requires an explicitly pinned content adapter, an isolated SaveBoundary and a
source manifest measured in a separate source-file process. No driver ban is
substituted for actual dependency/content verification.
"""
import inspect,json
from pathlib import Path
from .library_plan import COLLECTIONS,preflight
from .native_mesh_batch import file_ref,save_copy
from .store import canonical,digest,atomic_write

def id_key(item):return item.bl_rna.identifier+':'+item.name

def _adapter(function,reference):
    path=Path(inspect.getsourcefile(function)).resolve()
    if path!=Path(reference['path']).resolve() or file_ref(path)['sha256']!=reference['sha256']:raise ValueError('Library content adapter source pin differs')

def _graph(bpy,roots):
    users=bpy.data.user_map();deps={}
    for dependency,owners in users.items():
        for owner in owners:deps.setdefault(owner,set()).add(dependency)
    seen=set();todo=list(roots)
    while todo:
        item=todo.pop()
        if item in seen:continue
        if len(seen)>=2048:raise ValueError('Native library dependency budget exceeded')
        seen.add(item);todo.extend(deps.get(item,set())-seen)
    return seen,deps

def describe(bpy,roots,library_reference,fingerprint_id,adapter_reference):
    """Measure native closure in an isolated source process; external files are pinned."""
    if not bpy.app.background or bpy.data.is_dirty:raise ValueError('Use an unchanged saved library in a separate isolated source process')
    if file_ref(bpy.data.filepath)['sha256']!=library_reference['sha256'] or file_ref(library_reference['path'])['sha256']!=library_reference['sha256']:raise ValueError('Loaded library source differs from pinned bytes')
    _adapter(fingerprint_id,adapter_reference);closure,deps=_graph(bpy,roots)
    keys={x:id_key(x) for x in closure}
    if len(set(keys.values()))!=len(keys):raise ValueError('Ambiguous native ID names require an explicit owner identity adapter')
    rows=[]
    for item in sorted(closure,key=id_key):
        if item.library:raise ValueError('Linked secondary library needs its own measured closure; no implicit recursive append')
        counts={}
        if item.bl_rna.identifier=='Mesh':counts=dict(vertices=len(item.vertices),edges=len(item.edges),faces=len(item.polygons),loops=len(item.loops))
        rows.append(dict(id=keys[item],type=item.bl_rna.identifier,name=item.name,content_sha256=fingerprint_id(item,lambda x:keys[x]),counts=counts))
    external={}
    for item,paths in bpy.data.file_path_map(subset=list(closure),include_libraries=False).items():
        for path in paths:
            full=Path(bpy.path.abspath(path,start=str(Path(library_reference['path']).parent),library=item.library)).resolve()
            external[str(full)]=file_ref(full)
    manifest=dict(schema_version=1,coverage='complete_declared_roots',library=library_reference,adapter=adapter_reference,
        roots=sorted(keys[x] for x in roots),ids=rows,dependencies=sorted([[keys[x],keys[d]] for x in closure for d in deps.get(x,set()) if d in closure]),
        external_files=list(external.values()),limits='Native ID links and file paths; the pinned content adapter must cover relevant geometry, attributes, materials, animation and driver semantics. Arbitrary Python namespace dependencies are not inferred.')
    preflight(manifest,[],namespace='check_');return manifest

def append(boundary,manifest,namespace,output_dir,expected,fingerprint_id,fingerprint_host,host_adapter_reference,limits=None):
    """Append exact measured closure; rename every ID explicitly and verify native references."""
    boundary.require(expected);bpy=boundary.bpy;_adapter(fingerprint_id,manifest['adapter'])
    _adapter(fingerprint_host,host_adapter_reference)
    source=manifest['library']
    for ref in [source,*manifest['external_files']]:
        if file_ref(ref['path'])['sha256']!=ref['sha256']:raise ValueError('Library or external dependency bytes changed')
    original=set(bpy.data.user_map());host={(x.bl_rna.identifier,x.name) for x in original}
    original_names={x:id_key(x) for x in original}
    # Adapter hashes the original ID set and relationships between those IDs.
    # The only expected host addition is linking new roots into a scene collection.
    host_before=fingerprint_host(original)
    if not isinstance(host_before,str) or len(host_before)!=64 or any(c not in '0123456789abcdef' for c in host_before):raise ValueError('Original host content SHA256 required')
    plan=preflight(manifest,host,namespace,limits);out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_write(out/'plan.json',canonical(plan));atomic_write(out/'manifest.json',canonical(manifest))
    receipt=dict(schema_version=1,status='prepared',pending_stage='rollback_checkpoint',owner=boundary.owner,before=expected,plan=file_ref(out/'plan.json'),manifest=file_ref(out/'manifest.json'),receipt_path=str(out/'receipt.json'),appearance_accepted=False,host_adapter=host_adapter_reference,original_host_sha256=host_before)
    atomic_write(out/'receipt.json',canonical(receipt))
    grouped={}
    for row in manifest['ids']:grouped.setdefault(COLLECTIONS[row['type']],[]).append(row)
    try:
        receipt['rollback']=save_copy(bpy,out/'rollback.blend');guard=boundary.stamp()
        if guard['native']!=expected['native'] or guard['file']!=expected['file']:raise RuntimeError('Import checkpoint changed host content/source')
        receipt.pop('pending_stage',None);atomic_write(out/'receipt.json',canonical(receipt))
        boundary.require(guard)
        with bpy.data.libraries.load(source['path'],link=False,reuse_local_id=False) as (available,loaded):
            # Check every category/name before assigning any output request.
            for collection,rows in grouped.items():
                if any(row['name'] not in getattr(available,collection) for row in rows):raise ValueError('Measured library ID is absent from source')
            for collection,rows in grouped.items():setattr(loaded,collection,[row['name'] for row in rows])
        mapping={}
        for collection,rows in grouped.items():
            for row,item in zip(rows,getattr(loaded,collection)):
                if item is None or item in original:raise RuntimeError('Library append missing an ID or silently reused a host ID')
                mapping[item]=row['id']
        if len(mapping)!=len(manifest['ids']):raise RuntimeError('Native appended ID identity collapsed')
        for item,key in mapping.items():item.name=plan['renames'][key]['name']
        if any(item.name!=plan['renames'][key]['name'] for item,key in mapping.items()):raise RuntimeError('Native ID rename collided or truncated')
        root_items=[x for x,key in mapping.items() if key in manifest['roots']]
        for item in root_items:
            if item.bl_rna.identifier=='Object':bpy.context.scene.collection.objects.link(item)
            else:bpy.context.scene.collection.children.link(item)
        bpy.context.view_layer.update();closure,deps=_graph(bpy,root_items)
        if closure!=set(mapping):raise RuntimeError('Actual native append closure differs from measured source closure')
        actual_edges=sorted([[mapping[x],mapping[d]] for x in closure for d in deps.get(x,set()) if d in closure])
        if actual_edges!=manifest['dependencies']:raise RuntimeError('Shared/parent/driver dependency relationships changed')
        source_rows={r['id']:r for r in manifest['ids']}
        for item,key in mapping.items():
            if fingerprint_id(item,lambda x:mapping[x])!=source_rows[key]['content_sha256']:raise RuntimeError('Native content differs from source under explicit ID renaming: '+key)
        # Blender retains one transient source-library provenance ID during
        # append. It has no linked payload users and is not saved as payload.
        extra=set(bpy.data.user_map())-original-set(mapping)
        provenance=[];user_map=bpy.data.user_map()
        for item in extra:
            if (item.bl_rna.identifier!='Library' or len(extra)>1 or item.parent is not None or user_map.get(item,set())
                    or not item.use_extra_user or item.use_fake_user or item.users!=1
                    or Path(bpy.path.abspath(item.filepath)).resolve()!=Path(source['path']).resolve()
                    or any(x.library is item for x in mapping)):
                raise RuntimeError('Append created undeclared native IDs: '+str(sorted(id_key(x) for x in extra)))
            provenance.append(dict(id=id_key(item),source=source,users=item.users,use_extra_user=item.use_extra_user,
                parent=None,linked_payload_users=[],saved_payload=False,
                meaning='Measured transient source provenance; saved native content uses pinned weak references'))
        weak=[]
        for item,key in mapping.items():
            ref=item.library_weak_reference
            if ref:
                if Path(bpy.path.abspath(ref.filepath)).resolve()!=Path(source['path']).resolve():raise RuntimeError('Appended weak source reference differs from pinned library')
                weak.append(dict(source=key,filepath=ref.filepath,id_name=ref.id_name))
        receipt.update(runtime_library_provenance=provenance,durable_weak_references=weak)
        if not original<=set(bpy.data.user_map()) or any(id_key(x)!=original_names[x] for x in original) or fingerprint_host(original)!=host_before:raise RuntimeError('Append changed original host IDs, content or relationships')
        receipt['imported']=[dict(source=key,target=id_key(item)) for item,key in mapping.items()]
        before_save=boundary.stamp();receipt['candidate']=save_copy(bpy,out/'candidate.blend');after=boundary.stamp()
        if after['native']!=before_save['native'] or after['file']!=before_save['file'] or fingerprint_host(original)!=host_before:raise RuntimeError('Final import checkpoint changed native/source content')
        receipt.update(status='batch_applied',after=after,content_and_dependency_closure_verified=True)
        atomic_write(out/'receipt.json',canonical(receipt));return receipt
    except Exception as error:
        receipt.update(status='partial_or_uncertain',error=str(error),automatic_rollback_allowed=False,recovery='Preserve appended IDs and original rollback copy; owner reconciles this invocation without replay')
        try:receipt['partial']=save_copy(bpy,out/'partial.blend');receipt['after']=boundary.stamp()
        except Exception as problem:receipt['preservation_error']=str(problem)
        atomic_write(out/'receipt.json',canonical(receipt));raise
