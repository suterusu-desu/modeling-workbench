"""Closed raw-mesh operation plans with exact vertex ancestry and scoped aliases.

This compiler is pure. Native application must verify the same snapshot and preserve
unsupported mesh/rig data before accepting a compiled plan.
"""
import copy,math,struct
from .store import canonical,digest

OPS={'translate','split_vertex','merge_vertices','delete_vertices'}
def f32(value):return struct.unpack('<f',struct.pack('<f',float(value)))[0]
def validate_mesh(mesh):
    if not isinstance(mesh,dict) or set(mesh)-{'face_ids'}!={'ids','co','faces','face_materials','face_smooth'}:raise ValueError('Exact raw mesh fields required')
    ids=mesh['ids'];co=mesh['co'];faces=mesh['faces']
    if not isinstance(ids,list) or len(ids)>250000 or len(set(ids))!=len(ids) or any(not isinstance(x,str) or not x.isdigit() or str(int(x))!=x or int(x)>2147483647 for x in ids):raise ValueError('Unique bounded numeric string vertex identities required')
    known=set(ids)
    if len(co)!=len(ids) or any(len(v)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) or f32(x)!=x for x in v) for v in co):raise ValueError('Exact finite native float32 coordinates required')
    if len(faces)>250000 or sum(map(len,faces))>1500000:raise ValueError('Mesh exceeds declared bounds')
    if any(len(f)<3 or len(set(f))!=len(f) or any(x not in known for x in f) for f in faces):raise ValueError('Faces require distinct known vertex identities')
    if len(mesh['face_materials'])!=len(faces) or any(type(i) is not int or i<0 for i in mesh['face_materials']):raise ValueError('Face material indices required')
    if len(mesh['face_smooth'])!=len(faces) or any(type(b) is not bool for b in mesh['face_smooth']):raise ValueError('Face smoothing flags required')
    if 'face_ids' in mesh:
        fi=mesh['face_ids']
        if not isinstance(fi,list) or len(fi)!=len(faces) or len(set(fi))!=len(fi) or any(not isinstance(i,str) or not i.isdigit() or str(int(i))!=i or int(i)>2147483647 for i in fi):raise ValueError('Canonical distinct face identities required')
    return mesh

def mesh_revision(mesh):return digest(canonical(validate_mesh(mesh)))

def compile_plan(mesh,steps,aliases=None):
    """Preflight the entire typed sequence before producing any native effect."""
    current=copy.deepcopy(validate_mesh(mesh));aliases=copy.deepcopy(aliases or {})
    if not isinstance(steps,list) or not 1<=len(steps)<=64:raise ValueError('One to 64 steps required')
    if not isinstance(aliases,dict) or len(aliases)>128:raise ValueError('Bounded aliases required')
    for key,ids in aliases.items():
        if not isinstance(key,str) or not key or not isinstance(ids,list) or len(set(ids))!=len(ids) or any(x not in current['ids'] for x in ids):raise ValueError('Alias must name an exact current selection')
    records=[];names=set()
    for step in steps:
        if not isinstance(step,dict) or step.get('op') not in OPS:raise ValueError('Unsupported closed mesh operation')
        name=step.get('id')
        if not isinstance(name,str) or not name or name in names:raise ValueError('Unique step identity required')
        names.add(name);op=step['op'];allowed={'id','op','selection','alias_policy','output_alias'}
        allowed|={'delta'} if op=='translate' else {'faces','new_id'} if op=='split_vertex' else {'survivor'} if op=='merge_vertices' else set()
        if set(step)-allowed:raise ValueError('Unknown step arguments')
        selection=step.get('selection')
        selected=aliases.get(selection) if isinstance(selection,str) else selection
        if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected) or any(i not in current['ids'] for i in selected):raise ValueError('Exact nonempty selection required; no stale aliases')
        selected=list(selected);before=copy.deepcopy(current);src=before['ids'];coords=dict(zip(src,copy.deepcopy(before['co'])))
        ancestry={i:[i] for i in src};relations=[]
        if op=='translate':
            delta=step.get('delta')
            if not isinstance(delta,list) or len(delta)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in delta):raise ValueError('Finite XYZ translation required')
            for i in selected:coords[i]=[f32(a+b) for a,b in zip(coords[i],delta)]
        elif op=='split_vertex':
            if len(selected)!=1:raise ValueError('Split needs one vertex')
            old=selected[0];new=step.get('new_id');faces=step.get('faces')
            if not isinstance(new,str) or not new.isdigit() or str(int(new))!=new or int(new)>2147483647 or new in src:raise ValueError('Distinct numeric output identity required')
            incident=[i for i,f in enumerate(current['faces']) if old in f]
            if not isinstance(faces,list) or not faces or len(set(faces))!=len(faces) or any(type(i) is not int or i not in incident for i in faces) or set(faces)==set(incident):raise ValueError('Split must explicitly partition incident faces into two nonempty sets')
            for i in faces:current['faces'][i]=[new if v==old else v for v in current['faces'][i]]
            current['ids'].append(new);coords[new]=list(coords[old]);ancestry[new]=[old]
        elif op=='merge_vertices':
            survivor=step.get('survivor')
            if len(selected)<2 or survivor not in selected:raise ValueError('Merge requires at least two vertices and explicit selected survivor')
            coords[survivor]=[f32(sum(coords[i][axis] for i in selected)/len(selected)) for axis in range(3)]
            ancestry[survivor]=list(selected)
            current['ids']=[i for i in src if i not in selected or i==survivor]
            current['faces']=[[survivor if i in selected else i for i in f] for f in current['faces']]
        elif op=='delete_vertices':
            current['ids']=[i for i in src if i not in selected]
        # Drop only explicitly affected invalid faces; preserve material and smoothing ownership.
        faces=[];materials=[];smooth=[];removed=[]
        kept=set(current['ids'])
        for i,face in enumerate(current['faces']):
            cleaned=[v for j,v in enumerate(face) if v!=face[j-1]]
            if any(v not in kept for v in cleaned) or len(set(cleaned))<3:
                removed.append(i);continue
            if len(set(cleaned))!=len(cleaned):raise ValueError('Merge would create a self-touching polygon; choose a smaller explicit operation')
            faces.append(cleaned);materials.append(before['face_materials'][i]);smooth.append(before['face_smooth'][i])
        if 'face_ids' in current:current['face_ids']=[v for i,v in enumerate(before['face_ids']) if i not in removed]
        current.update(co=[coords[i] for i in current['ids']],faces=faces,face_materials=materials,face_smooth=smooth)
        validate_mesh(current)
        descendants={i:[] for i in src}
        for j in current['ids']:
            for i in ancestry[j]:descendants[i].append(j)
        grouped=set()
        for target in current['ids']:
            parents=ancestry[target]
            if len(parents)>1:
                relations.append(dict(kind='merged',sources=parents,targets=[target],weights={target:{i:1/len(parents) for i in parents}},weight_semantics='Explicit equal coordinate average before native float32 rounding'));grouped.update(parents)
        for source in src:
            if source in grouped:continue
            targets=descendants[source]
            relations.append(dict(kind='deleted' if not targets else 'preserved' if len(targets)==1 else 'split',sources=[source],targets=targets))
        policy=step.get('alias_policy','strict')
        if policy not in ('strict','any','all'):raise ValueError('Explicit strict/any/all alias policy required')
        remapped={}
        for alias,values in aliases.items():
            partial=[j for j in current['ids'] if set(ancestry[j])&set(values) and not set(ancestry[j])<=set(values)]
            if partial and policy=='strict':raise ValueError('Alias partially crosses a merge; explicit any/all policy required before native effects')
            remapped[alias]=[j for j in current['ids'] if (bool(set(ancestry[j])&set(values)) if policy=='any' else set(ancestry[j])<=set(values))]
        if step.get('output_alias'):
            alias=step['output_alias']
            if not isinstance(alias,str) or alias in remapped:raise ValueError('Output alias must be a new name')
            remapped[alias]=[j for j in current['ids'] if set(ancestry[j])&set(selected)]
        aliases=remapped
        if sum(len(r['after']['ids']) for r in records)+len(current['ids'])>1000000:raise ValueError('Aggregate batch vertex budget exceeded')
        records.append(dict(id=name,operation=op,before=before,after=copy.deepcopy(current),before_revision=mesh_revision(before),after_revision=mesh_revision(current),
            relations=relations,removed_face_indices=removed,aliases=copy.deepcopy(aliases)))
    return dict(schema_version=1,before=copy.deepcopy(mesh),before_revision=mesh_revision(mesh),steps=records,after=current,after_revision=mesh_revision(current),aliases=aliases,
        scope='Closed raw mesh vertex operations; no rig, UV, arbitrary attribute or anatomical preservation inferred')
