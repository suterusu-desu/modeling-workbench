"""Exact numerical queries on declared triangle surfaces; no Blender dependency."""
import numpy as np


def validate(arrays):
    co, tri = arrays['co'], arrays['tri']
    if co.ndim != 2 or co.shape[1] != 3 or not np.isfinite(co).all():
        raise ValueError('Coordinates must be finite XYZ rows')
    if tri.ndim != 2 or tri.shape[1] != 3 or tri.dtype.kind not in 'iu':
        raise ValueError('Topology must be integer triangle rows')
    if tri.size and (tri.min() < 0 or tri.max() >= len(co)):
        raise ValueError('Triangle index outside this geometry')


def select_triangles(arrays, selection=None):
    validate(arrays)
    ids = np.arange(len(arrays['tri']))
    if selection is None:
        return ids
    if 'material' in selection:
        if 'triangle_material' not in arrays:
            raise ValueError('This surface has no recorded material ownership')
        return ids[arrays['triangle_material'] == selection['material']]
    if 'component' in selection:
        if 'triangle_component' not in arrays:
            raise ValueError('This surface has no recorded component semantics')
        return ids[arrays['triangle_component'] == selection['component']]
    attribute = selection.get('attribute')
    if attribute not in arrays:
        raise ValueError('Semantic attribute unavailable: ' + str(attribute))
    values = arrays[attribute]
    threshold = float(selection.get('minimum', 0.99))
    if attribute.startswith('FACE__'):
        if 'triangle_polygon' not in arrays:
            raise ValueError('Polygon ownership unavailable')
        mask = values[arrays['triangle_polygon']] >= threshold
    elif attribute.startswith('POINT__'):
        mask = np.all(values[arrays['tri']] >= threshold, axis=1)
    else:
        raise ValueError('Explicit POINT__ or FACE__ semantic domain required')
    return ids[mask]


def nearest_surface(arrays, point, selection=None):
    p = np.asarray(point, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Query point must be one finite XYZ coordinate')
    ids = select_triangles(arrays, selection)
    if not len(ids):
        return {'status': 'unsupported', 'reason': 'No triangles satisfy the declared selection', 'eligible_triangles': 0}
    triangles = arrays['co'][arrays['tri'][ids]]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    normal = np.cross(ab, ac)
    n2 = np.einsum('ij,ij->i', normal, normal)
    projection = p - normal * (np.einsum('ij,ij->i', p-a, normal) / np.maximum(n2, 1e-300))[:, None]
    ap = projection-a
    d00 = np.einsum('ij,ij->i', ab, ab)
    d01 = np.einsum('ij,ij->i', ab, ac)
    d11 = np.einsum('ij,ij->i', ac, ac)
    d20 = np.einsum('ij,ij->i', ap, ab)
    d21 = np.einsum('ij,ij->i', ap, ac)
    denom = d00*d11-d01*d01
    safe = np.where(np.abs(denom) > 1e-30, denom, 1)
    v, w = (d11*d20-d01*d21)/safe, (d00*d21-d01*d20)/safe
    inside = (n2 > 1e-30) & (v >= -1e-12) & (w >= -1e-12) & (v+w <= 1+1e-12)
    candidates = [projection]
    distances = [np.where(inside, np.sum((projection-p)**2, axis=1), np.inf)]
    for start, end in ((a,b), (b,c), (c,a)):
        edge = end-start
        t = np.clip(np.einsum('ij,ij->i', p-start, edge) / np.maximum(np.sum(edge*edge,axis=1), 1e-300), 0, 1)
        closest = start+t[:,None]*edge
        candidates.append(closest)
        distances.append(np.sum((closest-p)**2, axis=1))
    ds = np.stack(distances)
    region, row = np.unravel_index(np.argmin(ds), ds.shape)
    index = int(ids[row]); hit = candidates[region][row]
    result = {'status': 'completed', 'point': hit.tolist(), 'distance': float(np.sqrt(ds[region,row])),
              'triangle_index': index, 'vertex_indices': arrays['tri'][index].tolist(),
              'eligible_triangles': int(len(ids)), 'total_triangles': len(arrays['tri']),
              'normal': (normal[row]/np.sqrt(n2[row])).tolist() if n2[row] > 1e-30 else None,
              'interpretation': 'Nearest point on the selected recorded surface; anatomical correspondence is not implied'}
    for key in ('triangle_polygon', 'triangle_component', 'triangle_material'):
        if key in arrays: result[key] = int(arrays[key][index])
    return result


def plane_sections(arrays, axis, value, selection=None):
    if axis not in (0,1,2) or not np.isfinite(value):
        raise ValueError('A finite plane and XYZ axis index are required')
    ids = select_triangles(arrays, selection)
    from .guide_fitting import triangle_sections
    cut = triangle_sections(arrays['co'], arrays['tri'][ids], value, axis=axis)
    segments = cut['segments'].tolist()
    return {'segments': segments, 'segment_count': len(segments),
            'triangle_indices': ids[cut['triangles']].tolist(),
            'coplanar_triangles_excluded': len(cut['coplanar_triangles']),
            'eligible_triangles': len(ids), 'definition': 'Fixed plane intersections with recorded native triangles'}


def compare(before, after, correspondence='triangles'):
    validate(before); validate(after)
    same_triangles=np.array_equal(before['tri'],after['tri'])
    if correspondence=='triangles':
        matched=same_triangles
    elif correspondence=='recorded_polygon_loops':
        fields=('loops','polygon_starts','polygon_lengths')
        matched=all(k in before and k in after and np.array_equal(before[k],after[k]) for k in fields)
    else:
        raise ValueError('Correspondence must be triangles or recorded_polygon_loops')
    if before['co'].shape != after['co'].shape or not matched:
        raise ValueError('Topology differs; a validated correspondence is required')
    delta = after['co']-before['co']
    distance = np.linalg.norm(delta,axis=1)
    worst = np.argsort(distance)[-5:][::-1]
    return {'changed_vertices': int(np.any(delta != 0,axis=1).sum()),
            'measurement':'Endpoint displacement; cumulative travel is not measured without ordered intermediate states',
            'maximum_distance': float(distance.max(initial=0)),
            'axis_minimum': delta.min(axis=0).tolist() if len(delta) else [0,0,0],
            'axis_maximum': delta.max(axis=0).tolist() if len(delta) else [0,0,0],
            'worst_vertices': [{'index':int(i),'distance':float(distance[i]),'delta':delta[i].tolist()} for i in worst],
            'triangle_layout_equal':same_triangles,
            'correspondence':correspondence,
            'correspondence_limit':'Matching recorded connectivity supports vertex comparison. Polygon-loop mode permits retessellated diagonals; triangle-index transfer and semantic identity are not implied.'}


def ray_hits(arrays, origin, direction, selection=None, maximum_distance=None):
    """All forward triangle intersections, sorted; no nearest-sheet substitution."""
    origin,direction=np.asarray(origin,float),np.asarray(direction,float)
    if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite([origin,direction]).all() or np.linalg.norm(direction)<1e-12:
        raise ValueError('Finite origin and nonzero direction required')
    direction=direction/np.linalg.norm(direction);ids=select_triangles(arrays,selection)
    tri=arrays['co'][arrays['tri'][ids]];e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h)
    usable=np.abs(det)>1e-12;inv=np.where(usable,1/np.where(usable,det,1),0)
    s=origin-tri[:,0];u=inv*np.einsum('ij,ij->i',s,h);q=np.cross(s,e1)
    v=inv*(q@direction);t=inv*np.einsum('ij,ij->i',e2,q)
    mask=usable&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>=0)
    if maximum_distance is not None:
        if maximum_distance<0:raise ValueError('Nonnegative ray range required')
        mask &= t<=maximum_distance
    rows=sorted(np.where(mask)[0],key=lambda i:t[i])
    return {'hits':[{'triangle_index':int(ids[i]),'distance':float(t[i]),'point':(origin+t[i]*direction).tolist(),
                     'barycentric':[float(1-u[i]-v[i]),float(u[i]),float(v[i])]} for i in rows],
            'eligible_triangles':len(ids),'interpretation':'Forward geometric intersections; coincident edge hits retain both triangle owners'}


def material_path(arrays, vertex_indices):
    validate(arrays);indices=np.asarray(vertex_indices)
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or len(indices)<2 or indices.min()<0 or indices.max()>=len(arrays['co']):
        raise ValueError('At least two recorded vertex indices required')
    co=arrays['co'][indices];length=np.linalg.norm(np.diff(co,axis=0),axis=1)
    return {'points':co.tolist(),'vertex_indices':indices.tolist(),'length':float(length.sum()),'segment_lengths':length.tolist(),
            'interpretation':'Authored vertex correspondence in this topology; not a fixed spatial cross-section'}
