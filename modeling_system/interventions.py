"""Evaluated displacement, qualified guide support and realized native comparison.

The native adapter stays unchanged. This evidence accompanies its existing
proposal; numerical support is not proof of solver causality or appearance.
"""
import numpy as np
from . import geometry
from .store import canonical, digest


def attachment_frame(points):
    x=points[1]-points[0];normal=np.cross(x,points[2]-points[0])
    if np.linalg.norm(x)<1e-12 or np.linalg.norm(normal)<1e-12:
        raise ValueError('Attachment support frame is degenerate')
    x=x/np.linalg.norm(x);normal=normal/np.linalg.norm(normal)
    return np.stack([x,np.cross(normal,x),normal])


def support_populations(changed, direct, interpolated=(), attachment=()):
    """Keep measured/declared membership and both denominators explicit.

    This arithmetic is not admission: qualification and actual constraints are
    checked separately by assess(). Historical masks can use the same reporting.
    """
    changed,direct,interpolated,attachment=map(set,(changed,direct,interpolated,attachment))
    if direct&interpolated or direct&attachment or interpolated&attachment:
        raise ValueError('Direct, interpolated and attachment populations must be disjoint')
    surveyed=direct|interpolated|attachment
    return dict(changed_count=len(changed),direct_changed=len(changed&direct),
        interpolated_changed=len(changed&interpolated),attachment_changed=len(changed&attachment),
        direct_fraction=len(changed&direct)/len(changed) if changed else None,
        denominator='actually changed evaluated vertices in this layer',
        surveyed_count=len(surveyed),surveyed_direct=len(direct),
        surveyed_direct_fraction=len(direct)/len(surveyed) if surveyed else None)


def assess(service, case, baseline, proposal):
    if not isinstance(case,dict) or case.get('mode') not in ('guide_fit','construction_repair'):
        raise ValueError('intervention.mode must be guide_fit or construction_repair')
    if case.get('state')!=baseline:
        raise ValueError('Intervention must bind the exact experiment state')
    state=service.store.get(baseline,'state')
    predicted=service.store.get(case.get('predicted_state',''),'state')
    if case.get('pose')!=state['controls'] or predicted['controls']!=state['controls']:
        raise ValueError('Intervention baseline, prediction and declared pose must match')
    if not case.get('mechanism') or not case.get('construction_evidence'):
        raise ValueError('Intervention needs mechanism and exact construction_evidence records')
    for key in case['construction_evidence']:service.store.get(key)
    if case['mode']=='construction_repair':
        repair=case.get('repair',{})
        if any(not repair.get(k) for k in ('reason','recovery_checkpoint','revised_constraints','validation_plan')):
            raise ValueError('Construction repair needs reason, recovery_checkpoint, revised_constraints and validation_plan; preserve old evidence')
        checkpoint=repair['recovery_checkpoint'];asset=service.store.blob(checkpoint['path'])
        if asset['sha256']!=checkpoint.get('sha256'):raise ValueError('Repair recovery checkpoint changed')
    rows=case.get('layers',[])
    if not rows or len({r.get('object') for r in rows})!=len(rows):raise ValueError('Distinct affected/dependent layers required')
    names={r['object'] for r in rows}
    # Require coverage of every captured evaluated surface, including unchanged
    # dependents. An omitted layer is a coverage hole, not evidence of no motion.
    captured={o['name'] for o in state['objects'] if o.get('asset')}
    if names!=captured or names!={o['name'] for o in predicted['objects'] if o.get('asset')}:
        raise ValueError('Intervention layers must cover every recorded surface in baseline and prediction; record scoped inclusion/exclusion at capture')
    results=[]
    for row in rows:
        name=row['object']
        if not row.get('semantic_component'):raise ValueError('Each layer needs its semantic_component')
        a,b=service.wb.arrays(baseline,name),service.wb.arrays(case['predicted_state'],name)
        geometry.compare(a,b,case.get('correspondence','triangles'))
        for obj in (state,predicted):
            entry=next(o for o in obj['objects'] if o['name']==name)
            if 'evaluated' not in str(entry.get('evaluation','')).lower():
                raise ValueError('Intervention requires explicitly evaluated geometry: '+name)
        displacement=b['co']-a['co'];changed=np.flatnonzero(np.any(displacement!=0,axis=1))
        if len(changed) and name not in proposal['allowed_objects']:
            raise ValueError('Predicted changed layer is outside proposal allowed_objects: '+name)
        support={};direct=set();violations=[]
        for group in row.get('support',[]):
            ids=group.get('indices',[])
            if not ids or any(type(i) is not int or not 0<=i<len(a['co']) or i in support for i in ids):
                raise ValueError('Support indices must be distinct valid vertices per layer')
            if group.get('kind') not in ('direct','interpolated','attachment'):raise ValueError('Support kind must be direct, interpolated or attachment')
            tolerance=group.get('tolerance')
            if type(tolerance) not in (int,float) or not np.isfinite(tolerance) or tolerance<0 or not group.get('basis'):
                raise ValueError('Each support needs finite nonnegative tolerance and evidence-based basis')
            if group['kind']=='direct':
                q=service.store.get(group['qualification'],'guide_qualification')
                service.references.require_usable_review(q['review'])
                if q['native_state']!=baseline or q['pose']!=case.get('guide_pose'):
                    raise ValueError('Guide support is not qualified for this baseline and pose')
                with np.load(service.store.resolve_blob(q['registered']),allow_pickle=False) as z:guide={k:z[k] for k in z.files}
                selected=geometry.select_triangles(guide,group.get('selection'))
                admitted=set()
                for region in q['support']:
                    admitted.update(map(int,geometry.select_triangles(guide,region.get('selection'))))
                if not len(selected) or not set(map(int,selected))<=admitted:
                    raise ValueError('Direct support leaves qualified guide triangles')
                for i in ids:
                    measured=geometry.nearest_surface(guide,b['co'][i],group.get('selection'))
                    distance=measured.get('distance')
                    if distance is None:raise ValueError('Guide query returned no supported surface')
                    support[i]=dict(kind='direct',qualification=group['qualification'],selection=group.get('selection'),
                                    distance=distance,tolerance=tolerance,basis=group['basis'])
                    direct.add(i)
                    if distance>tolerance:violations.append(i)
            elif group['kind']=='interpolated':
                neighbors=group.get('neighbors',[]);weights=np.asarray(group.get('weights',[]),float)
                if (not neighbors or weights.shape!=(len(neighbors),) or not np.isfinite(weights).all()
                        or np.any(weights<0) or not np.isclose(weights.sum(),1)
                        or any(type(i) is not int or not 0<=i<len(a['co']) for i in neighbors)):
                    raise ValueError('Interpolated support needs explicit neighboring vertices and normalized nonnegative weights')
                expected=weights@displacement[neighbors]
                for i in ids:
                    error=float(np.linalg.norm(displacement[i]-expected))
                    support[i]=dict(kind='interpolated',neighbors=neighbors,weights=weights.tolist(),residual=error,tolerance=tolerance,basis=group['basis'])
                    if error>tolerance:violations.append(i)
            else:
                driver=group.get('driver_object');neighbors=group.get('driver_indices',[])
                if driver not in names or driver==name or len(neighbors)!=3 or len(set(neighbors))!=3:
                    raise ValueError('Attachment needs another captured driver_object and three distinct driver_indices')
                da=service.wb.arrays(baseline,driver)['co'];db=service.wb.arrays(case['predicted_state'],driver)['co']
                if any(type(i) is not int or not 0<=i<len(da) for i in neighbors):raise ValueError('Attachment driver index outside captured surface')
                offsets=np.asarray(group.get('rest_offsets'),float);weights=np.asarray(group.get('weights'),float)
                if offsets.shape!=(len(ids),3) or weights.shape!=(3,) or not np.isfinite(offsets).all() or not np.isfinite(weights).all() or np.any(weights<0) or not np.isclose(weights.sum(),1):
                    raise ValueError('Attachment needs measured Nx3 rest_offsets and three normalized weights')
                before=weights@da[neighbors]+offsets@attachment_frame(da[neighbors])
                baseline_error=np.linalg.norm(before-a['co'][ids],axis=1)
                if np.any(baseline_error>tolerance):raise ValueError('Attachment predictor does not reproduce the independent baseline')
                after=weights@db[neighbors]+offsets@attachment_frame(db[neighbors])
                for pos,i in enumerate(ids):
                    error=float(np.linalg.norm(after[pos]-b['co'][i]))
                    support[i]=dict(kind='attachment',driver_object=driver,driver_indices=neighbors,rest_offset=offsets[pos].tolist(),
                        weights=weights.tolist(),baseline_residual=float(baseline_error[pos]),residual=error,tolerance=tolerance,basis=group['basis'])
                    if error>tolerance:violations.append(i)
        for value in support.values():
            if value['kind']=='interpolated' and not set(value['neighbors'])<=direct:
                raise ValueError('Interpolation must trace to directly qualified neighbors in this layer')
        unsupported=[int(i) for i in changed if i not in support]
        results.append(dict(object=name,semantic_component=row['semantic_component'],changed_vertices=changed.tolist(),
            support={str(k):v for k,v in support.items()},unsupported=unsupported,violations=sorted(set(violations)),
            **support_populations(changed,direct,[i for i,s in support.items() if s['kind']=='interpolated'],
                                 [i for i,s in support.items() if s['kind']=='attachment'])))
    by_name={r['object']:r for r in results}
    def grounded(name,index,visiting):
        pair=(name,index)
        if pair in visiting:raise ValueError('Attachment support dependency cycle')
        layer=by_name[name]
        if index not in layer['changed_vertices']:return True
        if index in layer['violations']:return False
        support=layer['support'].get(str(index))
        if not support:return False
        if support['kind']=='direct':return True
        if support['kind']=='interpolated':return all(grounded(name,i,visiting|{pair}) for i in support['neighbors'])
        return all(grounded(support['driver_object'],i,visiting|{pair}) for i in support['driver_indices'])
    for layer in results:
        layer['unqualified_dependencies']=[i for i in layer['changed_vertices'] if not grounded(layer['object'],i,set())]
    qualified=all(not r['unsupported'] and not r['violations'] and not r['unqualified_dependencies'] for r in results)
    result=dict(case=case,layers=results,numerically_supported=qualified,allowed_objects=proposal['allowed_objects'],
        disposition='construction_repair' if case['mode']=='construction_repair' else 'supported_prediction' if qualified else 'unsupported_prediction',
        proposal_sha256=digest(canonical(proposal)),realized_comparison='pending',
        limits='Numerical support for the pinned prediction; declared construction evidence must establish how constraints were used. No screenshot supplies solver causality or appearance acceptance.')
    return dict(record=service.store.put('intervention',result),**result)


def compare(service, intervention, realized_state):
    item=service.store.get(intervention,'intervention');case=item['case']
    actual=service.store.get(realized_state,'state')
    if actual['controls']!=case['pose']:raise ValueError('Realized comparison requires the same control pose')
    if {o['name'] for o in actual['objects'] if o.get('asset')}!={r['object'] for r in item['layers']}:
        raise ValueError('Realized capture must include all declared layers')
    tolerance=case.get('realization_tolerance')
    if type(tolerance) not in (int,float) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('Intervention needs finite nonnegative realization_tolerance')
    rows=[]
    for layer in item['layers']:
        name=layer['object'];a=service.wb.arrays(case['state'],name);p=service.wb.arrays(case['predicted_state'],name);b=service.wb.arrays(realized_state,name)
        geometry.compare(a,b,case.get('correspondence','triangles'))
        error=np.linalg.norm(p['co']-b['co'],axis=1)
        moved=np.flatnonzero(np.any(b['co']!=a['co'],axis=1))
        unsupported=[int(i) for i in moved if str(i) not in layer['support']]
        rows.append(dict(object=name,max_residual=float(error.max(initial=0)),exceeds_tolerance=np.flatnonzero(error>tolerance).tolist(),
                         newly_unsupported=unsupported,changed_count=len(moved)))
    actual_support=assess(service,dict(case,predicted_state=realized_state),case['state'],{'allowed_objects':item['allowed_objects']})
    result=dict(intervention=intervention,realized_state=realized_state,layers=rows,tolerance=tolerance,
                realized_support=actual_support['record'],realized_numerically_supported=actual_support['numerically_supported'],
                agrees=all(not r['exceeds_tolerance'] and not r['newly_unsupported'] for r in rows) and actual_support['numerically_supported'],
                appearance_accepted=False,basis='Complete recorded evaluated layers at one matched pose; finite coverage only')
    return dict(record=service.store.put('intervention_comparison',result),**result)
