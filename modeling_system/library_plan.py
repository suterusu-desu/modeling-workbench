"""Explicit, bounded library dependency closure and collision planning."""
from .store import canonical,digest

COLLECTIONS={'Object':'objects','Mesh':'meshes','Material':'materials','Collection':'collections','Armature':'armatures',
    'Action':'actions','Key':'shape_keys','ShaderNodeTree':'node_groups','GeometryNodeTree':'node_groups','CompositorNodeTree':'node_groups',
    'Image':'images','Curve':'curves','Light':'lights','Camera':'cameras','Texture':'textures'}
DEFAULT_LIMITS=dict(ids=2048,objects=256,vertices=1000000,edges=2000000,faces=1000000,loops=4000000)

def preflight(manifest,existing,namespace,limits=None):
    """No native effects; a namespace maps every declared ID, including dependencies."""
    if manifest.get('schema_version')!=1 or manifest.get('coverage')!='complete_declared_roots':raise ValueError('Complete version-1 declared root closure required')
    if not isinstance(namespace,str) or not namespace or len(namespace)>24 or any(not (c.isascii() and (c.isalnum() or c in '._-')) for c in namespace):raise ValueError('Explicit short ASCII namespace required')
    limits={**DEFAULT_LIMITS,**(limits or {})}
    if set(limits)!=set(DEFAULT_LIMITS) or any(type(v) is not int or not 0<v<=DEFAULT_LIMITS[k] for k,v in limits.items()):raise ValueError('Resource limits may only tighten the supported ceilings')
    rows=manifest.get('ids');roots=manifest.get('roots');edges=manifest.get('dependencies')
    if not isinstance(rows,list) or not rows or len(rows)>limits['ids']:raise ValueError('Library ID budget exceeded or empty closure')
    by_id={};renames={};totals={k:0 for k in limits};totals['ids']=len(rows)
    existing={(COLLECTIONS.get(kind,kind),name) for kind,name in existing};planned_names=set()
    for row in rows:
        kind=row.get('type');name=row.get('name');key=row.get('id')
        if kind not in COLLECTIONS:raise ValueError('Unsupported declared library ID type: '+str(kind))
        if not isinstance(name,str) or not name or key!=kind+':'+name or key in by_id:raise ValueError('Distinct exact typed library identities required')
        if not isinstance(row.get('content_sha256'),str) or len(row['content_sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['content_sha256']):raise ValueError('Per-ID adapter content SHA256 required')
        output=namespace+name
        if len(output.encode('utf-8'))>63:raise ValueError('Namespaced Blender ID exceeds conservative name bound')
        destination=(COLLECTIONS[kind],output)
        if destination in existing or destination in planned_names:raise ValueError('Destination ID collision: '+kind+':'+output)
        planned_names.add(destination)
        renames[key]=dict(type=kind,name=output);by_id[key]=row
        totals['objects']+=kind=='Object'
        counts=row.get('counts',{})
        if set(counts)-{'vertices','edges','faces','loops'}:raise ValueError('Unknown library resource counts')
        for count,n in counts.items():
            if type(n) is not int or n<0:raise ValueError('Nonnegative resource counts required')
            totals[count]+=n
    if any(totals[k]>limits[k] for k in limits):raise ValueError('Library aggregate resource budget exceeded')
    if not isinstance(roots,list) or not roots or len(set(roots))!=len(roots) or any(i not in by_id or by_id[i]['type'] not in ('Object','Collection') for i in roots):raise ValueError('Known object/collection roots required')
    if not isinstance(edges,list) or len(edges)>100000:raise ValueError('Bounded explicit dependency edges required')
    deps={i:set() for i in by_id}
    for pair in edges:
        if not isinstance(pair,list) or len(pair)!=2 or any(i not in by_id for i in pair):raise ValueError('Dependency escapes declared closure')
        deps[pair[0]].add(pair[1])
    if len({tuple(pair) for pair in edges})!=len(edges):raise ValueError('Duplicate dependency edges are not canonical')
    seen=set();pending=list(roots)
    while pending:
        key=pending.pop()
        if key in seen:continue
        seen.add(key);pending.extend(deps[key]-seen)
    if seen!=set(by_id):raise ValueError('Unreachable IDs are not part of the declared root closure')
    return dict(schema_version=1,manifest_revision=digest(canonical(manifest)),roots=roots,renames=renames,counts=totals,limits=limits,
        collision_policy='explicit namespace; no overwrite or silent reuse',native_ready=False)
