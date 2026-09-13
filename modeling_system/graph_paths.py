"""Bounded shortest-path evidence on an explicitly scoped recorded graph."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .repair_analysis import DOMAINS, _pin, _load
from .store import canonical


def inspect(service,case_path,expected_state,max_nodes=100000,max_edges=500000):
    for value,limit in ((max_nodes,250000),(max_edges,3000000)):
        if type(value) is not int or not 1<=value<=limit:raise ValueError('Positive bounded node/edge search budgets required')
    p=Path(case_path).resolve();original=service.store.blob(p)
    case=_pin(service,json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig')),p.parent,{})
    if case.get('schema_version')!=1 or not case.get('question'):raise ValueError('Version-1 graph-path question required')
    state=case.get('state',{})
    if any(k not in state or k not in expected_state or state[k]!=expected_state[k] for k in DOMAINS):
        raise ValueError('Recorded graph and expected state/pose/domain differ')
    for name in ('graph','selection'):
        if not case.get(name,{}).get('meaning') or not case[name].get('evidence'):raise ValueError('Graph and selection need explicit meanings and evidence')
    coverage=case['graph'].get('coverage',{})
    if coverage.get('status') not in ('unknown','declared_subset','complete_for_declared_graph') or not isinstance(coverage.get('missing'),list):
        raise ValueError('Explicit graph coverage required')
    if coverage['status']=='complete_for_declared_graph' and coverage['missing']:raise ValueError('Complete graph cannot declare missing coverage')
    z=_load(service,case['arrays']);ids=z['node_ids'];edges=z['edges']
    if ids.ndim!=1 or ids.dtype.kind not in 'iuU' or not 0<len(ids)<=250000:raise ValueError('Bounded integer/string node identities required')
    names=ids.astype(str).tolist();lookup={n:i for i,n in enumerate(names)}
    if len(lookup)!=len(names) or any(not n for n in names):raise ValueError('Distinct nonempty graph identities required')
    if edges.ndim!=2 or edges.shape[1]!=2 or edges.dtype.kind not in 'iu' or len(edges)>1500000:
        raise ValueError('Bounded integer graph-edge pairs required')
    if np.any(edges<0) or np.any(edges>=len(ids)) or np.any(edges[:,0]==edges[:,1]):raise ValueError('Invalid graph-edge endpoints')
    if len({tuple(sorted(map(int,e))) for e in edges})!=len(edges):raise ValueError('Duplicate undirected graph edges')
    masks={}
    for name,size in (('allowed_nodes',len(ids)),('allowed_edges',len(edges))):
        a=z[name]
        if a.shape!=(size,) or a.dtype.kind not in 'biu' or not np.isin(a,[0,1]).all():raise ValueError('Explicit boolean node/edge selection required')
        masks[name]=a.astype(bool)
    start,goal=case.get('start'),case.get('goal')
    if not isinstance(start,str) or start not in lookup or not isinstance(goal,str) or goal not in lookup:
        raise ValueError('Exact start and goal node identities required')
    source,target=lookup[start],lookup[goal];allowed=masks['allowed_nodes']
    adjacency=[[] for _ in names]
    for enabled,(a,b) in zip(masks['allowed_edges'],edges):
        if enabled and allowed[a] and allowed[b]:adjacency[a].append(int(b));adjacency[b].append(int(a))
    for values in adjacency:values.sort()
    distances={source:0};ways={source:1};parents={source:[]};queue=deque([source]);examined=0;exhausted=False
    if not allowed[source] or not allowed[target]:status='endpoint_outside_selection';queue.clear()
    else:
        status='no_path_in_declared_selection'
        while queue:
            node=queue.popleft()
            if target in distances and distances[node]>=distances[target]:break
            for neighbor in adjacency[node]:
                if examined>=max_edges:exhausted=True;break
                examined+=1;distance=distances[node]+1
                if neighbor not in distances:
                    if len(distances)>=max_nodes:exhausted=True;break
                    distances[neighbor]=distance;ways[neighbor]=ways[node];parents[neighbor]=[node];queue.append(neighbor)
                elif distances[neighbor]==distance:
                    ways[neighbor]=min(2,ways[neighbor]+ways[node]);parents[neighbor].append(node)
            if exhausted:break
        if exhausted:status='incomplete_budget'
        elif target in distances:status='path_found'
    path=[]
    if target in distances and allowed[source] and allowed[target]:
        cursor=target;path=[names[cursor]]
        while cursor!=source:cursor=parents[cursor][0];path.append(names[cursor])
        path.reverse()
    unique=('ambiguous_shortest_paths' if ways.get(target,0)>1 else 'unique_shortest_path') if status=='path_found' else 'unknown'
    summary=dict(status=status,shortest_path_status=unique,hops=len(path)-1 if path else None,
        discovered_nodes=len(distances),examined_directed_edges=examined,declared_nodes=len(ids),declared_edges=len(edges),
        excluded_nodes=int((~allowed).sum()),excluded_edges=int((~masks['allowed_edges']).sum()),coverage=coverage,
        budget=dict(max_nodes=max_nodes,max_edges=max_edges))
    predecessors=[dict(node=names[n],distance=distances[n],parents=[names[v] for v in parents[n]]) for n in sorted(distances,key=lambda v:(distances[v],v))]
    limits=['Shortest paths in the declared selected graph only; no inferred anatomy, surface correspondence or native permission.',
        'Unique shortest path does not mean unique route: longer alternatives are not enumerated.',
        'An incomplete search is unknown even if it already found a candidate path; it never proves absence.',
        'No path in this graph does not prove global unreachability, hidden-surface absence or absence outside the declared selection.',
        'Graph edge semantics and selection are supplied evidence; coordinates do not infer adjacency.']
    payload=dict(case_source=original,case=case,state=state,summary=summary,path=path,predecessors=predecessors,limits=limits,native_ready=False,target_admission=False)
    record=service.store.put('graph_path',payload)
    return dict(analysis=record,status=status,summary=summary,native_ready=False,target_admission=False,
        reads={k:dict(operation='read_record',arguments=dict(record=record,path=[k],limit=10,max_chars=8000)) for k in ('path','predecessors','case','limits')})
