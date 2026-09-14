"""Portable content adapter for static mesh libraries with object drivers.

Explicit supported content, not a claim to serialize every Blender feature.
Unsupported payloads fail in source inspection before destination append. Wider
libraries supply a separately pinned adapter with their own native qualification.
"""
import math
from .store import canonical,digest


def _plain(value,resolve):
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Nonfinite library content')
        return value
    if hasattr(value,'id_type') and hasattr(value,'name_full'):return {'id':resolve(value)}
    if hasattr(value,'items'):return {str(k):_plain(v,resolve) for k,v in value.items()}
    if isinstance(value,set):return sorted(value)
    try:return [_plain(x,resolve) for x in value]
    except TypeError:raise ValueError('Unsupported library property type: '+type(value).__name__)


def _drivers(item,resolve):
    animation=getattr(item,'animation_data',None)
    if animation is None:return None
    if animation.action or len(animation.nla_tracks):raise ValueError('Actions/NLA require a qualified animation content adapter')
    if len(animation.drivers)>128 or sum(len(f.keyframe_points)+len(f.sampled_points) for f in animation.drivers)>65536:raise ValueError('Driver curve content exceeds bundled adapter budget')
    curves=[]
    for f in animation.drivers:
        if f.driver.type=='SCRIPTED' and not f.driver.is_simple_expression:raise ValueError('External Python driver namespace requires separately pinned execution dependencies')
        if any(v.type not in ('SINGLE_PROP','TRANSFORMS','ROTATION_DIFF','LOC_DIFF') or any(t.id is None for t in v.targets) for v in f.driver.variables):raise ValueError('Driver variables require explicit native ID targets within the measured closure')
        modifiers=[]
        for modifier in f.modifiers:
            props={}
            for p in modifier.bl_rna.properties:
                if p.identifier=='rna_type' or p.is_readonly:continue
                if p.type in ('COLLECTION','POINTER'):raise ValueError('Unsupported driver modifier nested content')
                props[p.identifier]=_plain(getattr(modifier,p.identifier),resolve)
            modifiers.append(props)
        point_fields=('co','handle_left','handle_right','handle_left_type','handle_right_type','interpolation','type','easing','back','amplitude','period')
        curves.append(dict(path=f.data_path,index=f.array_index,mute=f.mute,lock=f.lock,extrapolation=f.extrapolation,auto_smoothing=f.auto_smoothing,
            keyframes=[{k:_plain(getattr(p,k),resolve) for k in point_fields} for p in f.keyframe_points],samples=[list(p.co) for p in f.sampled_points],
            type=f.driver.type,expression=f.driver.expression,use_self=f.driver.use_self,modifiers=modifiers,
            variables=[dict(name=v.name,type=v.type,targets=[dict(id=_plain(t.id,resolve),data_path=t.data_path,
                bone_target=t.bone_target,transform_type=t.transform_type,transform_space=t.transform_space,
                rotation_mode=t.rotation_mode,context_property=t.context_property,use_fallback_value=t.use_fallback_value,fallback_value=t.fallback_value) for t in v.targets]) for v in f.driver.variables]))
    return curves


def _material_tree(tree,resolve):
    if tree is None:return None
    if not tree.is_embedded_data:raise ValueError('A shared shader group needs separately qualified library closure')
    if len(tree.nodes)>64 or len(tree.links)>256:raise ValueError('Material tree exceeds bundled adapter budget')
    nodes=[]
    for node in tree.nodes:
        if node.bl_idname not in ('ShaderNodeBsdfPrincipled','ShaderNodeOutputMaterial'):raise ValueError('Bundled material adapter needs qualification for node '+node.bl_idname)
        props={}
        for prop in node.bl_rna.properties:
            key=prop.identifier
            if key=='rna_type' or prop.is_readonly or prop.type=='COLLECTION':continue
            value=getattr(node,key)
            if key=='parent':props[key]=value.name if value else None
            else:props[key]=_plain(value,resolve)
        nodes.append(dict(name=node.name,type=node.bl_idname,properties=props,
            inputs=[dict(id=s.identifier,type=s.bl_idname,default=_plain(s.default_value,resolve) if hasattr(s,'default_value') else None,hide=s.hide) for s in node.inputs],
            outputs=[dict(id=s.identifier,type=s.bl_idname,default=_plain(s.default_value,resolve) if hasattr(s,'default_value') else None,hide=s.hide) for s in node.outputs]))
    return dict(nodes=sorted(nodes,key=lambda n:n['name']),links=sorted([[l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier] for l in tree.links]),
        custom={k:_plain(tree[k],resolve) for k in tree.keys()},drivers=_drivers(tree,resolve))


def fingerprint_id(item,resolve):
    """Hash supported saved content with ID references normalized through resolve."""
    kind=item.bl_rna.identifier
    if kind not in ('Object','Mesh','Material','Collection'):raise ValueError('Bundled library content adapter does not qualify '+kind)
    record=dict(type=kind,custom={k:_plain(item[k],resolve) for k in item.keys()},drivers=_drivers(item,resolve))
    if kind=='Object':
        if item.type not in ('MESH','EMPTY') or len(item.modifiers) or len(item.constraints) or len(item.vertex_groups) or item.instance_type!='NONE':raise ValueError('Rig/modifier/constraint/instance content needs a separately qualified adapter')
        fields=('location','rotation_mode','rotation_euler','rotation_quaternion','rotation_axis_angle','scale',
            'delta_location','delta_rotation_euler','delta_rotation_quaternion','delta_scale','matrix_parent_inverse',
            'parent_type','parent_bone','parent_vertices','hide_viewport','hide_render','display_type','color',
            'lock_location','lock_rotation','lock_scale','show_in_front')
        record.update(data=_plain(item.data,resolve),parent=_plain(item.parent,resolve),fields={k:_plain(getattr(item,k),resolve) for k in fields},
            material_slots=[dict(link=s.link,material=_plain(s.material,resolve)) for s in item.material_slots])
    elif kind=='Mesh':
        if item.shape_keys or getattr(item,'has_custom_normals',False):raise ValueError('Shape keys/custom normals need a separately qualified adapter')
        fields={'FLOAT':'value','INT':'value','BOOLEAN':'value','FLOAT_VECTOR':'vector','FLOAT_COLOR':'color',
            'BYTE_COLOR':'color','FLOAT2':'vector','INT8':'value','INT32_2D':'value','QUATERNION':'value','FLOAT4X4':'value'}
        attributes={}
        for attr in item.attributes:
            if attr.name.startswith('.select'):continue
            if attr.data_type not in fields:raise ValueError('Unsupported library attribute '+attr.data_type)
            attributes[attr.name]=dict(domain=attr.domain,type=attr.data_type,values=[_plain(getattr(x,fields[attr.data_type]),resolve) for x in attr.data])
        record.update(co=[list(v.co) for v in item.vertices],edges=[dict(vertices=list(e.vertices),sharp=e.use_edge_sharp,seam=e.use_seam,hide=e.hide) for e in item.edges],
            faces=[dict(vertices=list(p.vertices),material=p.material_index,smooth=p.use_smooth,hide=p.hide) for p in item.polygons],
            vertex_hidden=[v.hide for v in item.vertices],attributes=attributes,materials=[_plain(m,resolve) for m in item.materials])
    elif kind=='Material':
        record['node_tree']=_material_tree(item.node_tree,resolve)
        record['fields']={}
        for prop in item.bl_rna.properties:
            if prop.identifier in ('rna_type','name','name_full','node_tree') or prop.is_readonly:continue
            if prop.type=='COLLECTION':
                if len(getattr(item,prop.identifier)):raise ValueError('Unsupported material collection '+prop.identifier)
                continue
            if prop.type=='POINTER':
                if getattr(item,prop.identifier) is not None:raise ValueError('Unsupported material nested property '+prop.identifier)
                continue
            record['fields'][prop.identifier]=_plain(getattr(item,prop.identifier),resolve)
    else:
        record.update(objects=sorted(resolve(o) for o in item.objects),children=sorted(resolve(c) for c in item.children),
            hide_viewport=item.hide_viewport,hide_render=item.hide_render,instance_offset=list(item.instance_offset))
    return digest(canonical(record))
