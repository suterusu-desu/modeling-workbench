"""Explicit, bidirectional semantic relationships with independent validity."""
import json
import numpy as np
from .episodes import pin_link, INTERPRETATIONS
from .store import canonical, digest


def dependency_state(service,state,binding_paths=None,target_paths=None,observation=None):
    recorded=service.store.get(state,'state')
    native=json.loads(service.store.resolve_blob(recorded['source_record']).read_text(encoding='utf-8-sig'))
    objects={}
    source_objects={o['name']:o for o in native['state']['objects']}
    for obj in recorded['objects']:
        data=dict(geometry=obj.get('geometry_hash'),source=digest(canonical(source_objects[obj['name']].get('source',{}))))
        if obj.get('asset'):
            arrays=service.wb.arrays(state,obj['name'])
            data['topology']=digest(canonical({k:digest(arrays[k].tobytes()) for k in
                ('tri','polygon_vertices','polygon_loop_start','polygon_loop_total','loop_vertex','poly_loop_start','poly_loop_total') if k in arrays}))
            data['attributes']=digest(canonical({k:digest(v.tobytes()) for k,v in arrays.items() if k.startswith(('POINT__','FACE__'))}))
        objects[obj['name']]=data
    binding=[pin_link(service,dict(kind='file',path=p,role='semantic binding dependency')) for p in binding_paths or []]
    target=[pin_link(service,dict(kind='file',path=p,role='target/registration dependency')) for p in target_paths or []]
    ob=service.store.get(observation,'observation') if observation else None
    if ob and ob['state']!=state: raise ValueError('Observation must identify this exact recorded state')
    payload=dict(state=state,source_state_id=recorded['source_state_id'],objects=objects,pose=recorded['controls'],
                 guide=recorded['guide'],bindings=binding,targets=target,
                 semantic_status='unvalidated; file integrity and fresh geometry do not establish anatomical meaning',
                 view=digest(canonical(ob['metadata'])) if ob else None,
                 mode='recorded dependencies, not live freshness')
    return dict(dependencies=service.store.put('dependency_state',payload),**payload)


def dependency_diff(before,after,domains=None):
    domains=domains or ['objects','pose','guide','bindings','targets']
    allowed={'objects','pose','guide','bindings','targets','view','semantic_validation'}
    if set(domains)-allowed: raise ValueError('Unknown dependency domain')
    return [dict(domain=d,before=before.get(d),after=after.get(d)) for d in domains if before.get(d)!=after.get(d)]


def register_graph(service,nodes,edges,coverage,dependencies):
    nodes=json.loads(canonical(nodes));edges=json.loads(canonical(edges));coverage=json.loads(canonical(coverage))
    service.store.get(dependencies,'dependency_state')
    ids=[n.get('id') for n in nodes]
    if not all(ids) or len(ids)!=len(set(ids)): raise ValueError('Unique semantic node IDs required')
    if not coverage.get('included') or 'missing' not in coverage: raise ValueError('Explicit included and missing semantic coverage required')
    for node in nodes:
        if node.get('interpretation') not in INTERPRETATIONS or not node.get('role'):
            raise ValueError('Node needs role and labeled interpretation')
        node['evidence']=[pin_link(service,l) for l in node.get('evidence',[])]
    for edge in edges:
        if edge.get('from') not in ids or edge.get('to') not in ids or not edge.get('relation'):
            raise ValueError('Edge must join known node IDs with a named relationship')
        if edge.get('interpretation') not in INTERPRETATIONS or not edge.get('evidence'):
            raise ValueError('Relationship needs interpretation and actual provenance')
        edge['evidence']=[pin_link(service,l) for l in edge['evidence']]
    payload=dict(nodes=nodes,edges=edges,coverage=coverage,dependencies=dependencies,
                 limits='Known authored/measured relations only; proximity and indices do not prove anatomical correspondence')
    return dict(graph=service.store.put('semantic_graph',payload),nodes=len(nodes),edges=len(edges),coverage=coverage)


def validate_semantics(service,graph,dependencies,disposition,reason,evidence):
    g=service.store.get(graph,'semantic_graph'); current=service.store.get(dependencies,'dependency_state')
    prior=service.store.get(g['dependencies'],'dependency_state')
    if disposition not in ('valid','invalid','unknown') or not reason.strip() or not evidence:
        raise ValueError('Explicit semantic disposition, reason and evidence required')
    changed=dependency_diff(prior,current,['objects','bindings'])
    # An agent may author a new binding after native validation, but cannot stamp
    # a graph bound to changed topology/source as valid merely by refreshing.
    if disposition=='valid' and changed:
        raise ValueError('Binding dependencies changed; register a repaired graph with actual semantics evidence')
    payload=dict(graph=graph,dependencies=dependencies,disposition=disposition,reason=reason,
                 evidence=[pin_link(service,l) for l in evidence],changed=changed,
                 basis='Operator semantic judgment over exact dependencies, not automatically inferred from coordinates')
    return dict(validation=service.store.put('semantic_validation',payload),**payload)


def impact(service,graph,node=None,selector=None,direction='both',depth=3,dependencies=None,validation=None):
    g=service.store.get(graph,'semantic_graph')
    if direction not in ('forward','reverse','both') or not 0<=depth<=8: raise ValueError('Invalid traversal scope')
    by_id={n['id']:n for n in g['nodes']}
    seeds={node} if node else set()
    if selector:
        for n in g['nodes']:
            sel=n.get('selector',{})
            if all((value in sel.get('vertex_indices',[]) if key=='vertex_index' else sel.get(key)==value)
                   for key,value in selector.items()): seeds.add(n['id'])
    if not seeds or not seeds<=by_id.keys(): raise ValueError('No explicit semantic binding for query; proximity is not a fallback')
    seen=set(seeds); found=[];front=set(seeds)
    for _ in range(depth):
        following=set()
        for i,e in enumerate(g['edges']):
            if (direction in ('forward','both') and e['from'] in front) or (direction in ('reverse','both') and e['to'] in front):
                if i not in found: found.append(i)
                following.update((e['from'],e['to']))
        following-=seen;seen|=following;front=following
    changed=[]
    if dependencies:
        changed=dependency_diff(service.store.get(g['dependencies'],'dependency_state'),service.store.get(dependencies,'dependency_state'))
    semantic='unvalidated'
    if validation:
        v=service.store.get(validation,'semantic_validation')
        if v['graph']!=graph or (dependencies and v['dependencies']!=dependencies):
            raise ValueError('Semantic validation belongs to another graph/dependency state')
        semantic=v['disposition']
    return dict(graph=graph,nodes=[by_id[i] for i in sorted(seen)],edges=[g['edges'][i] for i in found],
                direction=direction,depth=depth,coverage=g['coverage'],changed_dependencies=changed,
                semantic_status=semantic,dependency_status='stale' if changed else 'matching recorded dependencies' if dependencies else 'not checked',
                missing_links='Declared graph only; traversal depth can omit further nodes',total_nodes=len(g['nodes']),reached=len(seen))
