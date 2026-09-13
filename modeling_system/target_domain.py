"""Offline candidate/selection/semantic/transition inventories with separate meanings."""
import json
from pathlib import Path
import numpy as np
from .control_coverage import _ids
from .repair_analysis import DOMAINS, _pin, _load


def _boolean(z, key, n):
    value=z[key]
    if value.shape != (n,) or value.dtype.kind not in 'biu' or not np.isin(value,[0,1]).all():
        raise ValueError(key+' must contain one explicit boolean per point')
    return value.astype(bool)


def _edges(value, n, name):
    if value.ndim!=2 or value.shape[1]!=2 or value.dtype.kind not in 'iu' or len(value)>1500000:
        raise ValueError(name+' requires bounded integer endpoint pairs')
    if np.any(value<0) or np.any(value>=n) or np.any(value[:,0]==value[:,1]):
        raise ValueError(name+' contains invalid endpoint indices')
    pairs=[tuple(map(int,p)) for p in np.sort(value,axis=1)]
    if len(set(pairs))!=len(pairs):raise ValueError(name+' contains duplicate undirected edges')
    return set(pairs)


def inspect(service, case_path, expected_state):
    path=Path(case_path).resolve();original=service.store.blob(path)
    raw=json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig'))
    case=_pin(service,raw,path.parent,{})
    if case.get('schema_version')!=1 or not case.get('question'):
        raise ValueError('Version-1 target domain case with a question required')
    state=case.get('state',{})
    if any(d not in state or d not in expected_state or state[d]!=expected_state[d] for d in DOMAINS):
        raise ValueError('Target inventory and expected recorded state/pose/domain differ')
    inventory=case.get('inventory',{});scope=inventory.get('coverage',{})
    if (type(inventory.get('independent_of_support_mask')) is not bool or not inventory.get('basis')
            or not inventory.get('evidence') or not inventory.get('point_identity')
            or scope.get('status') not in ('unknown','declared_subset','complete_for_declared_screen')
            or not isinstance(scope.get('missing'),list)):
        raise ValueError('Inventory needs explicit identity, screen/mask independence, evidence and coverage')
    if scope['status']=='complete_for_declared_screen' and (scope['missing'] or not inventory['independent_of_support_mask']):
        raise ValueError('Complete independent screen cannot inherit support-mask filtering or missing coverage')
    for name in ('selection','response','topology','semantics','support_mask'):
        if not case.get(name,{}).get('meaning') or not case[name].get('evidence'):
            raise ValueError(name+' needs its own meaning and evidence')
    threshold=case['support_mask'].get('threshold')
    if type(threshold) not in (int,float) or not np.isfinite(threshold):
        raise ValueError('Finite explicit support-mask threshold required')
    fields=case['semantics'].get('fields',[]);_ids(fields,'semantic fields')
    if len(fields)>32:raise ValueError('At most 32 semantic fields are supported')
    z=_load(service,case['arrays']);native=z['point_ids']
    if native.ndim!=1 or native.dtype.kind not in 'iuU' or not 0<len(native)<=250000:
        raise ValueError('Ordered point_ids must be integer or string identities, 1..250000')
    ids=native.astype(str).tolist();universe=_ids(ids,'point_ids');n=len(ids)
    candidate=_boolean(z,'candidate',n);selected=_boolean(z,'selected',n);response=_boolean(z,'response_covered',n)
    mask=z['mask_values'];labels=z['semantic_values']
    if mask.shape!=(n,) or mask.dtype.kind not in 'fiu' or not np.isfinite(mask).all():
        raise ValueError('Mask values must be finite and aligned to ordered points')
    if labels.shape!=(n,len(fields)) or labels.dtype.kind not in 'fiu':
        raise ValueError('Semantic values must align to ordered points and named fields')
    # NaN/Inf are permitted only in original numeric semantic evidence. The
    # retained diagnostic uses explicit null/unknown, never fabricated labels.
    known=np.isfinite(labels);inside=mask>threshold
    edges=_edges(z['edges'],n,'edges')
    by_category={name:set() for name in ('fit_fit','fit_context','context_context')}
    for a,b in edges:
        kind='fit_fit' if selected[a] and selected[b] else 'fit_context' if selected[a] or selected[b] else 'context_context'
        by_category[kind].add((a,b))
    exclusions=case.get('exclusions',[]);reasons={}
    for index,row in enumerate(exclusions):
        members=_ids(row.get('points'),'exclusion points',universe)
        if (not row.get('reason') or not row.get('evidence')
                or row.get('interpretation') not in ('protected','trial_restriction','context','unresolved')):
            raise ValueError('Exclusions need reason, evidence and explicit interpretation')
        for point in members:reasons.setdefault(point,[]).append(index)
    correspondence={}
    for row in case.get('correspondence_reviews',[]):
        members=_ids(row.get('points'),'correspondence review points',universe)
        if row.get('status') not in ('supported','unresolved','rejected') or not row.get('reason') or not row.get('evidence'):
            raise ValueError('Correspondence review needs status, reason and evidence')
        if members & correspondence.keys():raise ValueError('Conflicting current correspondence reviews')
        for point in members:correspondence[point]=row
    checks=case.get('checks',[])
    _ids([c['id'] for c in checks],'check families')
    if len(checks)>16:raise ValueError('At most 16 separately named check families are supported')
    reports=[];transitions_checked=set();check_sets={}
    for check in checks:
        if not check.get('meaning') or not check.get('evidence'):
            raise ValueError('Each check family needs its own measurement meaning and evidence')
        required=_ids(check.get('required_categories'),'required categories',set(by_category))
        checked=_edges(z[check['edge_array']],n,check['id'])
        check_sets[check['id']]=checked
        if not checked<=edges:raise ValueError('Check endpoints must belong to the declared topology edge universe')
        transitions_checked|=checked & by_category['fit_context']
        reports.append(dict(id=check['id'],meaning=check['meaning'],evidence=check['evidence'],
            required_categories=sorted(required),categories={kind:dict(total=len(values),checked=len(values&checked),
                unchecked=len(values-checked),required=kind in required) for kind,values in by_category.items()},
            missing_required_edges={kind:[[ids[a],ids[b]] for a,b in sorted(by_category[kind]-checked)] for kind in required},
            limits='Coverage of this declared edge check only; other measurement families are not interchangeable.'))
    focus=np.flatnonzero(candidate|selected)
    points=[dict(id=ids[i],geometric_candidate=bool(candidate[i]),selected=bool(selected[i]),
        response_covered=bool(response[i]),mask_value=float(mask[i]),inside_authored_mask=bool(inside[i]),
        semantics={name:(float(labels[i,j]) if known[i,j] else None) for j,name in enumerate(fields)},
        unknown_semantics=[name for j,name in enumerate(fields) if not known[i,j]],
        numeric_label_status='known' if fields and known[i].all() else 'unknown',
        correspondence_status=correspondence.get(ids[i],{}).get('status','unreviewed'),
        exclusion_records=reasons.get(ids[i],[])) for i in focus]
    transitions=[]
    # Store exact transition identities without repeatedly expanding the mesh.
    for a,b in sorted(by_category['fit_context']):
        transitions.append(dict(points=[ids[a],ids[b]],response_covered=bool(response[a] and response[b]),
            checked_by=[name for name,values in check_sets.items() if (a,b) in values]))
    missing_semantics=~known.all(axis=1) if fields else np.ones(n,dtype=bool)
    summary=dict(declared_points=n,geometric_candidates=int(candidate.sum()),selected_points=int(selected.sum()),
        candidates_outside_authored_mask=int((candidate&~inside).sum()),unselected_candidates=int((candidate&~selected).sum()),
        selected_outside_geometric_screen=int((selected&~candidate).sum()),
        candidates_without_response=int((candidate&~response).sum()),
        candidates_with_unknown_numeric_labels=int((candidate&missing_semantics).sum()),
        selected_with_unknown_numeric_labels=int((selected&missing_semantics).sum()),
        candidates_without_supported_correspondence=sum(bool(candidate[i]) and correspondence.get(ids[i],{}).get('status')!='supported' for i in range(n)),
        unselected_candidates_without_exclusion_provenance=sum(candidate[i] and not selected[i] and ids[i] not in reasons for i in range(n)),
        declared_edges=len(edges),fit_context_edges=len(transitions),
        fit_context_edges_checked_by_any_family=len(transitions_checked),
        fit_context_edges_without_response=sum(not r['response_covered'] for r in transitions),
        independent_geometric_screen=inventory['independent_of_support_mask'],screen_coverage=scope['status'])
    summary['unselected_candidates_without_exclusion_provenance']=int(summary['unselected_candidates_without_exclusion_provenance'])
    actions=[]
    if not inventory['independent_of_support_mask'] or scope['status']!='complete_for_declared_screen':
        actions.append('Inspect missing geometric-screen coverage independently of the authored deformation mask.')
    if any(candidate[i] and not inside[i] and correspondence.get(ids[i],{}).get('status') in (None,'unresolved') for i in range(n)):
        actions.append('Review mask-excluded geometric candidates whose region correspondence is still unresolved; mask disagreement alone does not admit targets or relax protection.')
    if summary['candidates_with_unknown_numeric_labels']:
        actions.append('Preserve unknown numeric labels explicitly. Independently reviewed region correspondence can be supported without inventing missing label values.')
    if any(candidate[i] and correspondence.get(ids[i],{}).get('status') in (None,'unresolved') for i in range(n)):
        actions.append('Review candidate region correspondence independently of geometric hits, numeric labels and support-mask membership.')
    if summary['unselected_candidates_without_exclusion_provenance']:
        actions.append('Record why candidate points remain outside the fit; distinguish protection, context, trial restrictions and unresolved qualification.')
    if summary['fit_context_edges_without_response']:
        actions.append('Recover or justify missing transition response coverage before making finite deformation claims.')
    if any(v['unchecked'] for r in reports for v in r['categories'].values() if v['required']):
        actions.append('Inspect unchecked required transition/neighborhood edges within each named measurement family; fit-only gradients do not cover fit-to-context joins.')
    payload=dict(schema_version=1,question=case['question'],state=state,case_source=original,case=case,
        summary=summary,points=points,transitions=transitions,checks=reports,next_actions=actions,
        native_ready=False,target_admission=False,limits=[
            'Recorded supplied inventory only; no live geometry, visibility or anatomical inference.',
            'Geometric candidacy, selected fit, authored support and qualified response have different meanings.',
            'Null semantic labels preserve unknown original numeric evidence; they are not permission to fit.',
            'Coverage counts do not prove smoothness, correct measurement values or appearance.',
            'Check families retain separate meanings; a union count cannot substitute one metric for another.'])
    record=service.store.put('target_domain',payload)
    return dict(analysis=record,status='diagnostic_only',summary=summary,next_actions=actions,native_ready=False,target_admission=False,
        reads={key:dict(operation='read_record',arguments=dict(record=record,path=[key],limit=10,max_chars=8000))
            for key in ('points','transitions','checks','case','limits')})
