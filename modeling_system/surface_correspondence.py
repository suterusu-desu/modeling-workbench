"""Ordered surface alternatives and exact section connectivity; no admission."""
import json
from pathlib import Path
import numpy as np
from .repair_analysis import DOMAINS, _pin, _load


def _meta(value, name):
    if not isinstance(value, dict) or not value.get('meaning') or not value.get('evidence'):
        raise ValueError(name+' needs explicit meaning and evidence')


def _vector(value, name, size=3):
    a=np.asarray(value)
    if a.shape!=(size,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError(name+' requires a finite vector')
    return a.astype(float)


def _coverage(value):
    if (not isinstance(value,dict) or value.get('status') not in ('unknown','declared_subset','complete_for_declared_query')
            or not isinstance(value.get('missing'),list)):
        raise ValueError('Explicit query coverage and missing evidence required')
    if value['status']=='complete_for_declared_query' and value['missing']:
        raise ValueError('Complete declared query cannot have missing evidence')


def _node(value):
    # Exact source edge/vertex identity, never coordinate proximity.
    if not isinstance(value,str) or not value:
        raise ValueError('Nonempty exact node identity required')
    return value


def _target(value):
    if not isinstance(value,dict) or any(k not in value for k in (*DOMAINS,'target_id')):
        raise ValueError('Exact target identity, state, pose and coordinate domain required')


def _plane(point, plane):
    if not isinstance(plane,dict) or plane.get('axis') not in ('X','Y','Z'):
        raise ValueError('Exact axis-aligned section plane required')
    value,tolerance=plane.get('value'),plane.get('tolerance')
    if (type(value) not in (int,float) or type(tolerance) not in (int,float)
            or not np.isfinite([value,tolerance]).all() or tolerance<0):
        raise ValueError('Finite plane value and explicit nonnegative tolerance required')
    if abs(point['XYZ'.index(plane['axis'])]-value)>tolerance:
        raise ValueError('Station or hit lies outside the declared section plane')


def _graphs(rows):
    if not isinstance(rows,list) or len(rows)>128:raise ValueError('At most 128 section graphs')
    graphs={};reports=[];total=0
    for g in rows:
        name=_node(g.get('id'));_meta(g,'Section graph');_coverage(g.get('coverage'))
        if name in graphs:raise ValueError('Duplicate section graph')
        _target(g.get('context'))
        plane=g['context'].get('plane')
        if not isinstance(plane,dict):raise ValueError('Section graph needs exact plane context')
        segments=g.get('segments');parent={};by_triangle={}
        if not isinstance(segments,list):raise ValueError('Section segments required')
        total+=len(segments)
        if total>250000:raise ValueError('Section segment budget exceeded')
        def find(a):
            parent.setdefault(a,a)
            while parent[a]!=a:
                parent[a]=parent[parent[a]];a=parent[a]
            return a
        for seg in segments:
            tri=seg.get('triangle')
            if type(tri) is not int or tri<0 or tri in by_triangle:raise ValueError('Unique nonnegative section triangle required')
            nodes=seg.get('nodes',[])
            if len(nodes)!=2 or _node(nodes[0])==_node(nodes[1]):raise ValueError('Section needs two distinct exact nodes')
            if type(seg.get('selected')) is not bool:raise ValueError('Explicit section selection required')
            by_triangle[tri]=seg
            if seg['selected']:
                a,b=map(find,nodes);parent[b]=a
        groups={}
        for tri,seg in by_triangle.items():
            if seg['selected']:groups.setdefault(find(seg['nodes'][0]),[]).append(tri)
        branches={tri:min(group) for group in groups.values() for tri in group}
        graphs[name]=(by_triangle,branches,g['context'])
        reports.append(dict(id=name,coverage=g['coverage'],context=g['context'],
            selected_components=[dict(id=min(group),triangles=sorted(group)) for group in groups.values()],
            meaning='Exact selected section connectivity only; component IDs carry no anatomy.'))
    return graphs,reports


def inspect(service, case_path, expected_state):
    path=Path(case_path).resolve();original=service.store.blob(path)
    raw=json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig'))
    case=_pin(service,raw,path.parent,{})
    if case.get('schema_version')!=1 or not case.get('question'):raise ValueError('Version-1 correspondence question required')
    state=case.get('state',{})
    if any(k not in state or k not in expected_state or state[k]!=expected_state[k] for k in DOMAINS):
        raise ValueError('Correspondence and expected recorded state/pose/domain differ')
    for name in ('source_topology','roles','selection','orientation'):_meta(case.get(name),name)
    topology=case['source_topology']
    if topology.get('kind') not in ('native_polygon_edges','triangulation_edges'):raise ValueError('Explicit source edge semantics required')
    z=_load(service,case['arrays']);edge=z['source_edges']
    if edge.ndim!=2 or edge.shape[1]!=2 or edge.dtype.kind not in 'iu' or len(edge)>1500000 or np.any(edge<0):
        raise ValueError('Bounded nonnegative integer source edge pairs required')
    edges={tuple(sorted(map(int,p))) for p in edge}
    if len(edges)!=len(edge) or any(a==b for a,b in edges):raise ValueError('Duplicate or self source edges')
    graphs,graph_reports=_graphs(case.get('section_graphs',[]))
    paths=case.get('paths')
    if not isinstance(paths,list) or not 0<len(paths)<=128:raise ValueError('1..128 ordered paths required')
    ids=set();station_rows=[];transitions=[];path_rows=[];hit_count=0
    for p in paths:
        name=_node(p.get('id'))
        if name in ids:raise ValueError('Duplicate path identity')
        ids.add(name);_meta(p,'Ordered path');_coverage(p.get('coverage'))
        if p.get('kind') not in ('material_edges','spatial_probes'):raise ValueError('Explicit path kind required')
        if type(p.get('all_alternatives_retained')) is not bool:raise ValueError('Alternative retention declaration required')
        _target(p.get('target_context'))
        if not p['target_context'].get('query'):raise ValueError('Explicit query domain required')
        graph_id=p.get('section_graph')
        if graph_id is not None and graph_id not in graphs:raise ValueError('Unknown section graph')
        graph=graphs.get(graph_id)
        if graph and any(graph[2][k]!=p['target_context'][k] for k in (*DOMAINS,'target_id')):
            raise ValueError('Section graph and path target state/pose/domain differ')
        stations=p.get('stations')
        if not isinstance(stations,list) or len(stations)<2:raise ValueError('At least two ordered stations required')
        if len(station_rows)+len(stations)>10000:raise ValueError('Station budget exceeded')
        local=[]
        for i,s in enumerate(stations):
            point=_vector(s.get('point'),'Source point')
            if graph:_plane(point,graph[2]['plane'])
            if p['kind']=='material_edges' and (type(s.get('source_vertex')) is not int or s['source_vertex']<0):
                raise ValueError('Material station needs exact nonnegative source vertex')
            hits=s.get('alternatives')
            if not isinstance(hits,list):raise ValueError('Every station retains its alternatives, including empty lists')
            hit_count+=len(hits)
            if hit_count>100000:raise ValueError('Alternative budget exceeded')
            selected=[];branches=set();unmapped=False
            for rank,h in enumerate(hits):
                if type(h.get('full_order')) is not int or h['full_order']!=rank:raise ValueError('Full alternative order must be contiguous and retained')
                if type(h.get('triangle')) is not int or h['triangle']<0 or type(h.get('selected')) is not bool:
                    raise ValueError('Exact triangle and explicit selection required')
                hit_point=_vector(h.get('point'),'Hit point');bary=_vector(h.get('barycentric'),'Barycentric')
                if graph:_plane(hit_point,graph[2]['plane'])
                normal=_vector(h.get('normal'),'Winding-oriented normal')
                if abs(bary.sum()-1)>1e-6 or np.any(bary < -1e-6) or np.linalg.norm(normal)<1e-12:
                    raise ValueError('Invalid barycentric or normal evidence')
                if graph and h['triangle'] in graph[0] and graph[0][h['triangle']]['selected']!=h['selected']:
                    raise ValueError('Hit selection disagrees with section selection')
                if h['selected']:
                    selected.append(rank)
                    if graph and h['triangle'] in graph[1]:branches.add(graph[1][h['triangle']])
                    else:unmapped=True
            row=dict(path=name,station=i,source_vertex=s.get('source_vertex'),point=point.tolist(),
                alternatives=hits,selected_orders=selected,first_full_order=0 if hits else None,
                first_selected_order=selected[0] if selected else None,selected_components=sorted(branches),
                branch_evidence='unknown' if unmapped or not graph else 'exact_section_graph',
                query_coverage=p['coverage'],all_alternatives_retained=p['all_alternatives_retained'])
            local.append(row);station_rows.append(row)
        for i,(a,b) in enumerate(zip(local,local[1:])):
            source_connected=(tuple(sorted((a['source_vertex'],b['source_vertex']))) in edges) if p['kind']=='material_edges' else None
            complete=p['all_alternatives_retained'] and p['coverage']['status']=='complete_for_declared_query'
            if not complete:relation='unknown_selection_coverage'
            elif not a['selected_orders'] or not b['selected_orders']:relation='selection_gap'
            elif a['branch_evidence']=='unknown' or b['branch_evidence']=='unknown':relation='unknown_branch_evidence'
            elif len(a['selected_components'])!=1 or len(b['selected_components'])!=1:relation='ambiguous_section_branches'
            elif a['selected_components']==b['selected_components']:relation='same_selected_section_component'
            else:relation='different_selected_section_components'
            transitions.append(dict(path=name,stations=[i,i+1],source_edge_connected=source_connected,
                section_relation=relation,section_graph=graph_id,continuous_correspondence_proven=False))
        path_rows.append(dict(id=name,kind=p['kind'],stations=len(local),target_context=p['target_context'],
            coverage=p['coverage'],meaning=p['meaning'],evidence=p['evidence']))
    summary=dict(paths=len(paths),stations=len(station_rows),alternatives=hit_count,
        stations_without_selected_hit=sum(not r['selected_orders'] for r in station_rows),
        stations_with_multiple_selected_alternatives=sum(len(r['selected_orders'])>1 for r in station_rows),
        selected_first_behind_full_first=sum(r['first_selected_order'] is not None and r['first_selected_order']>0 for r in station_rows),
        missing_source_edges=sum(t['source_edge_connected'] is False for t in transitions),
        transition_relations={k:sum(t['section_relation']==k for t in transitions) for k in sorted({t['section_relation'] for t in transitions})})
    actions=[]
    if summary['missing_source_edges']:actions.append('Inspect missing source edges; the authored vertex order does not establish a connected material path.')
    if summary['stations_without_selected_hit']:actions.append('Preserve unsupported stations and query limits; do not bridge a selection gap or infer global absence.')
    if summary['selected_first_behind_full_first']:actions.append('Review foreground alternatives and exclusions before interpreting a selected hit as surface ownership.')
    if any(t['section_relation']!='same_selected_section_component' for t in transitions):
        actions.append('Inspect exact branch alternatives, missing connectivity and coverage; component numbers and normals cannot supply anatomical correspondence.')
    limits=['Historical supplied geometry/order evidence only; no live query or target admission.',
        'Source order, section connectivity, selected membership and owner-supplied roles are distinct relations.',
        'A shared selected section component proves neither continuous correspondence nor pose/appearance acceptance.',
        'Different components mean disconnected in this supplied selected section graph, not global surface disconnection.',
        'Full-order ranks are supplied query evidence; no ray geometry is recomputed or alternative completeness inferred.',
        'All alternatives and original evidence remain retained; orientation never classifies anatomy.']
    payload=dict(schema_version=1,question=case['question'],state=state,case_source=original,case=case,
        summary=summary,paths=path_rows,stations=station_rows,transitions=transitions,sections=graph_reports,
        next_actions=actions,limits=limits,native_ready=False,target_admission=False)
    record=service.store.put('surface_correspondence',payload)
    return dict(analysis=record,status='diagnostic_only',summary=summary,next_actions=actions,native_ready=False,target_admission=False,
        reads={k:dict(operation='read_record',arguments=dict(record=record,path=[k],limit=10,max_chars=8000))
            for k in ('paths','stations','transitions','sections','case','limits')})
