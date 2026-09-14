"""Owner-run raw-mesh batches in isolated Blender, with checkpoints and exact lineage.

No transport or live-owner takeover. The caller supplies its verified full native
content fingerprint; partial effects remain recoverable and are never replayed.
"""
import json,uuid,hashlib
from pathlib import Path
from .mesh_plan import compile_plan,mesh_revision
from .store import canonical,digest,atomic_write

def file_ref(path):
    p=Path(path);return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)

def read_mesh(ob,identity_attribute='workbench_vertex_id',face_identity_attribute=None):
    if ob.type!='MESH' or ob.mode!='OBJECT' or ob.data.users!=1:raise ValueError('One independently owned raw mesh in object mode required')
    if face_identity_attribute==identity_attribute:raise ValueError('Vertex and face identity attributes must differ')
    m=ob.data
    if m.shape_keys or len(ob.modifiers) or len(ob.vertex_groups) or len(m.uv_layers) or m.animation_data or ob.animation_data:raise ValueError('Rig, modifier, weight, UV and mesh animation data require a separately qualified writer')
    builtins={'position','.edge_verts','.corner_vert','.corner_edge','.select_vert','.select_edge','.select_poly','.hide_vert','.hide_edge','.hide_poly','material_index','sharp_face','sharp_edge'}
    unsupported=[a.name for a in m.attributes if a.name not in builtins|{identity_attribute,face_identity_attribute}]
    if unsupported:raise ValueError('Unsupported mesh attributes: '+str(unsupported))
    if any(e.use_edge_sharp or e.use_seam for e in m.edges):raise ValueError('Explicit sharp-edge/seam preservation needs a separate qualified writer')
    if any(v.hide for v in m.vertices) or any(e.hide for e in m.edges) or any(p.hide for p in m.polygons):raise ValueError('Hidden geometry is outside this explicitly visible raw mesh contract')
    used={tuple(sorted((p.vertices[i-1],p.vertices[i]))) for p in m.polygons for i in range(len(p.vertices))}
    if any(tuple(sorted(e.vertices)) not in used for e in m.edges):raise ValueError('Loose-edge preservation needs an explicit edge-domain writer')
    attr=m.attributes.get(identity_attribute)
    if attr and (attr.domain!='POINT' or attr.data_type!='INT'):raise ValueError('Identity attribute must be POINT INT')
    ids=[str(x.value) for x in attr.data] if attr else [str(v.index) for v in m.vertices]
    value=dict(ids=ids,co=[list(v.co) for v in m.vertices],faces=[[ids[i] for i in p.vertices] for p in m.polygons],
        face_materials=[p.material_index for p in m.polygons],face_smooth=[p.use_smooth for p in m.polygons])
    if face_identity_attribute:
        fa=m.attributes.get(face_identity_attribute)
        if fa is None or fa.domain!='FACE' or fa.data_type!='INT':raise ValueError('Declared face identity attribute must exist as FACE INT')
        value['face_ids']=[str(x.value) for x in fa.data]
    mesh_revision(value);return value

def write_mesh(ob,value,identity_attribute,face_identity_attribute=None):
    m=ob.data;indices={v:i for i,v in enumerate(value['ids'])}
    materials=list(m.materials)
    m.clear_geometry();m.from_pydata(value['co'],[],[[indices[v] for v in f] for f in value['faces']]);m.update()
    for p,index,smooth in zip(m.polygons,value['face_materials'],value['face_smooth']):p.material_index=index;p.use_smooth=smooth
    attr=m.attributes.get(identity_attribute)
    if attr is None:attr=m.attributes.new(identity_attribute,'INT','POINT')
    attr.data.foreach_set('value',[int(v) for v in value['ids']]);ob.update_tag(refresh={'DATA'})
    if face_identity_attribute:
        fa=m.attributes.get(face_identity_attribute)
        if fa is None:fa=m.attributes.new(face_identity_attribute,'INT','FACE')
        fa.data.foreach_set('value',[int(v) for v in value['face_ids']])
    if list(m.materials)!=materials or read_mesh(ob,identity_attribute,face_identity_attribute)!=value:raise RuntimeError('Native raw mesh roundtrip differs from prepared result')

class SaveBoundary:
    """A later native edit, file save, file replacement or session restart invalidates rollback."""
    KEY='Modeling workbench isolated transaction boundary'
    def __init__(self,bpy,owner,fingerprint):
        if not bpy.app.background:raise ValueError('This bounded raw-mesh adapter runs in an isolated background Blender only')
        if not isinstance(owner,str) or not owner:raise ValueError('Explicit stable owner required')
        self.bpy=bpy;self.owner=owner;self.fingerprint=fingerprint;self.epoch=0;self.session=uuid.uuid4().hex
        existing=bpy.app.driver_namespace.get(self.KEY)
        if existing:raise ValueError('Another transaction boundary already owns this isolated process')
        bpy.app.driver_namespace[self.KEY]=dict(owner=owner,session=self.session)
        def saved(*args):self.epoch+=1
        self.handler=saved;bpy.app.handlers.save_post.append(saved)
    def stamp(self):
        if self.bpy.app.driver_namespace.get(self.KEY)!=dict(owner=self.owner,session=self.session):raise ValueError('Native owner/session changed')
        path=self.bpy.data.filepath;native=self.fingerprint()
        if not isinstance(native,str) or len(native)!=64 or any(c not in '0123456789abcdef' for c in native):raise ValueError('Adapter must supply an exact full-scope content SHA256')
        return dict(owner=self.owner,session=self.session,save_epoch=self.epoch,native=native,
            file=file_ref(path) if path and Path(path).is_file() else dict(path=path,sha256=None),dirty=bool(self.bpy.data.is_dirty))
    def require(self,expected):
        current=self.stamp()
        if current!=expected:raise ValueError('Rollback or operation conflicts with later native edits/save/session: preserve later work')
        return current
    def close(self):
        if self.handler in self.bpy.app.handlers.save_post:self.bpy.app.handlers.save_post.remove(self.handler)
        if self.bpy.app.driver_namespace.get(self.KEY)==dict(owner=self.owner,session=self.session):self.bpy.app.driver_namespace.pop(self.KEY)

def save_copy(bpy,path):
    p=Path(path)
    if p.exists():raise ValueError('Checkpoint destination exists; preserve it')
    source=bpy.data.filepath
    if 'FINISHED' not in bpy.ops.wm.save_as_mainfile(filepath=str(p),copy=True,compress=True) or bpy.data.filepath!=source:raise RuntimeError('Checkpoint copy failed')
    return file_ref(p)

def execute_batch(boundary,object_name,steps,output_dir,expected,aliases=None,identity_attribute='workbench_vertex_id',face_identity_attribute=None):
    """One synchronous dispatch; all typed steps compile before the first native write."""
    boundary.require(expected);bpy=boundary.bpy;ob=bpy.data.objects[object_name]
    before=read_mesh(ob,identity_attribute,face_identity_attribute);plan=compile_plan(before,steps,aliases)
    out=Path(output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    atomic_write(out/'plan.json',canonical(plan))
    receipt=dict(schema_version=1,status='prepared',pending_stage='rollback_checkpoint',owner=boundary.owner,object=object_name,identity_attribute=identity_attribute,face_identity_attribute=face_identity_attribute,
        before=expected,plan=file_ref(out/'plan.json'),applied_steps=[],receipt_path=str(out/'receipt.json'),appearance_accepted=False)
    atomic_write(out/'receipt.json',canonical(receipt))
    try:
        receipt['rollback']=save_copy(bpy,out/'rollback.blend');guard=boundary.stamp()
        if guard['native']!=expected['native'] or guard['file']!=expected['file']:raise RuntimeError('Checkpoint saving changed native content/source')
        receipt.pop('pending_stage',None);atomic_write(out/'receipt.json',canonical(receipt))
        for step in plan['steps']:
            boundary.require(guard)
            if read_mesh(ob,identity_attribute,face_identity_attribute)!=step['before']:raise ValueError('Native mesh differs from the preflighted step')
            receipt.update(status='applying',pending_step=step['id']);atomic_write(out/'receipt.json',canonical(receipt))
            write_mesh(ob,step['after'],identity_attribute,face_identity_attribute);bpy.context.view_layer.update()
            guard=boundary.stamp()
            folder=out/('step-'+str(len(receipt['applied_steps'])));folder.mkdir()
            domain=lambda mesh,revision:dict(state=revision,object=object_name,domain='vertex',topology_hash=digest(canonical([mesh['ids'],mesh['faces']])),scope='Complete declared raw mesh vertex domain',ids=mesh['ids'])
            table=dict(schema_version=1,operation_id=boundary.session+':'+step['id'],before=domain(step['before'],step['before_revision']),after=domain(step['after'],step['after_revision']),relations=step['relations'])
            atomic_write(folder/'lineage.json',canonical(table))
            lineage_receipt=dict(operation_id=table['operation_id'],status='topology_change_recorded',lineage_sha256=file_ref(folder/'lineage.json')['sha256'],
                before_revision=digest(canonical(table['before'])),after_revision=digest(canonical(table['after'])),native_roundtrip_verified=True)
            atomic_write(folder/'receipt.json',canonical(lineage_receipt))
            atomic_write(folder/'case.json',canonical(dict(schema_version=1,lineage=file_ref(folder/'lineage.json'),operation_receipt=file_ref(folder/'receipt.json'))))
            receipt['applied_steps'].append(dict(id=step['id'],native=guard,lineage_case=file_ref(folder/'case.json'),aliases=step['aliases']))
            receipt.pop('pending_step',None);receipt['status']='step_completed';atomic_write(out/'receipt.json',canonical(receipt))
        boundary.require(guard)
        receipt['candidate']=save_copy(bpy,out/'candidate.blend');after_save=boundary.stamp()
        if after_save['native']!=guard['native'] or after_save['file']!=guard['file']:raise RuntimeError('Candidate save changed native content/source; saved candidate is not the current verified state')
        receipt['after']=after_save;receipt['status']='batch_applied'
        atomic_write(out/'receipt.json',canonical(receipt));return receipt
    except Exception as error:
        receipt.update(status='partial_or_uncertain',automatic_rollback_allowed=False,error=str(error),recovery='Inspect current state and original steps; never replay the batch automatically')
        try:receipt['after']=boundary.stamp();receipt['partial']=save_copy(bpy,out/'partial.blend');receipt['after']=boundary.stamp()
        except Exception as preservation_error:receipt['preservation_error']=str(preservation_error)
        atomic_write(out/'receipt.json',canonical(receipt));raise

def rollback_batch(boundary,receipt_path):
    """Restore a saved isolated transaction only if no later edit or save occurred."""
    path=Path(receipt_path);r=json.loads(path.read_bytes())
    if r['owner']!=boundary.owner or r['status']!='batch_applied':raise ValueError('No owned active transaction to roll back')
    expected=r['after'];boundary.require(expected)
    if file_ref(r['rollback']['path'])!=r['rollback']:raise ValueError('Rollback checkpoint bytes changed')
    abandoned=save_copy(boundary.bpy,path.parent/'abandoned.blend');after_copy=boundary.stamp()
    if after_copy['native']!=expected['native'] or after_copy['file']!=expected['file']:raise ValueError('Later native/source change while preserving abandoned state')
    bpy=boundary.bpy;before_native=r['before']['native'];boundary.close()
    result=dict(r,status='rollback_opening',abandoned=abandoned,original_file_not_overwritten=True)
    atomic_write(path.parent/'rollback-result.json',canonical(result))
    try:
        if 'FINISHED' not in bpy.ops.wm.open_mainfile(filepath=r['rollback']['path'],load_ui=False,use_scripts=False):raise RuntimeError('Native rollback reopen failed')
        actual=boundary.fingerprint()
        result.update(status='rolled_back' if actual==before_native else 'rollback_verification_failed',restored_native=actual)
    except Exception as error:
        result.update(status='rollback_uncertain',error=str(error),replay_allowed=False)
        atomic_write(path.parent/'rollback-result.json',canonical(result));raise
    atomic_write(path.parent/'rollback-result.json',canonical(result));return result
