"""Dependency-bound sparse response computation and measured native comparison.

No bpy, universal anatomy, or extrapolated motion proof. The sparse pointwise
adapter operates one explicitly named control/output coordinate domain.
"""
from pathlib import Path
import time
import numpy as np
from .episodes import pin_link
from .semantics import dependency_diff

KINDS={'exact_linear','exact_piecewise','local_derivative','approximate'}


def arrays(service,asset):
    with np.load(service.store.resolve_blob(asset),allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def validate_arrays(z):
    required={'data','indices','indptr','shape','baseline_input','baseline_output','output_ids'}
    if required-set(z): raise ValueError('Missing response arrays: '+str(sorted(required-set(z))))
    shape=z['shape']
    if shape.shape!=(2,) or shape.dtype.kind not in 'iu' or min(shape)<1: raise ValueError('Positive two-dimensional CSR shape required')
    m,n=map(int,shape)
    if (z['indptr'].shape!=(m+1,) or z['indptr'].dtype.kind not in 'iu' or z['indptr'][0]!=0
        or z['indptr'][-1]!=len(z['data']) or np.any(np.diff(z['indptr'])<0)
        or z['indices'].dtype.kind not in 'iu' or len(z['indices'])!=len(z['data'])
        or np.any(z['indices']<0) or np.any(z['indices']>=n)):
        raise ValueError('Invalid sparse CSR matrix')
    for key,size in [('baseline_input',n),('baseline_output',m),('output_ids',m)]:
        if z[key].shape!=(size,): raise ValueError('Invalid '+key+' shape')
    if z['output_ids'].dtype.kind not in 'iu' or np.any(z['output_ids']<0) or len(set(z['output_ids']))!=m:
        raise ValueError('Output IDs must be distinct nonnegative native indices')
    if any(not np.isfinite(z[k]).all() for k in ('data','baseline_input','baseline_output')):
        raise ValueError('Finite response coefficients and coordinates required')
    if 'limit' in z:
        if any(z[k].shape!=(m,) for k in ('limit','selected')) or not np.isfinite(z['limit']).all(): raise ValueError('Invalid support arrays')
        if 'epsilon' not in z or z['epsilon'].size!=1 or not float(z['epsilon'].item())>0: raise ValueError('Positive support epsilon required')
    return m,n


def multiply(z,x):
    row=np.repeat(np.arange(int(z['shape'][0])),np.diff(z['indptr']))
    return np.bincount(row,weights=z['data']*x[z['indices']],minlength=int(z['shape'][0]))


def pointwise(z,x):
    raw=multiply(z,x)
    if 'limit' not in z: return raw
    limit=z['limit'];epsilon=float(z['epsilon'].item())
    h=np.maximum(1-np.abs(raw-limit)/epsilon,0)
    supported=np.minimum(raw,limit)-epsilon/4*h*h
    return np.where(z['selected'].astype(bool),supported,raw)


def register(service,dependencies,semantic_validation,spec,arrays_path,evidence):
    dep=service.store.get(dependencies,'dependency_state')
    validation=service.store.get(semantic_validation,'semantic_validation')
    if validation['dependencies']!=dependencies or validation['disposition']!='valid':
        raise ValueError('Response requires actual valid semantic binding evidence for its dependencies')
    if spec.get('kind') not in KINDS or spec.get('adapter')!='sparse_pointwise_v1':
        raise ValueError('Supported adapter sparse_pointwise_v1; explicit exact_linear/exact_piecewise/local_derivative/approximate kind required')
    if not all(spec.get(k) for k in ('input_domain','output_domain','units','validity','influence')) or not evidence:
        raise ValueError('Response needs explicit coordinate domains, units, validity, influence coverage and evidence')
    if not spec['validity'].get('dependency_domains') or 'max_abs_delta' not in spec['validity']:
        raise ValueError('Validity needs dependency domains and finite maximum change; no silent extrapolation')
    if not {'objects','pose','guide','bindings','targets'}<=set(spec['validity']['dependency_domains']):
        raise ValueError('Response validity must include geometry/construction, pose, guide, semantic bindings and targets')
    maximum=float(spec['validity']['max_abs_delta'])
    if not np.isfinite(maximum) or maximum<0: raise ValueError('Invalid maximum response step')
    influence=spec['influence']
    if not all(k in influence for k in ('included','missing','boundary','complete_for_controls')):
        raise ValueError('Influence needs included, missing, boundary and explicitly qualified complete_for_controls')
    asset=service.store.blob(arrays_path); z=arrays(service,asset);m,n=validate_arrays(z)
    if 'bound_source_state' in z and str(z['bound_source_state'].item())!=dep['source_state_id']:
        raise ValueError('Adapter export belongs to another native source state; do not treat historical ancestry as a current response')
    if not all(all(k in spec[domain] for k in ('object','axis','frame')) for domain in ('input_domain','output_domain')):
        raise ValueError('Input/output domains need actual object, coordinate axis and frame')
    if any(spec[domain]['axis'] not in (0,1,2) for domain in ('input_domain','output_domain')):
        raise ValueError('Coordinate axes must be 0, 1 or 2')
    free=spec['validity'].get('free_controls',list(range(n)))
    if len(free)!=len(set(free)) or any(type(i)!=int or i<0 or i>=n for i in free): raise ValueError('Invalid free control indices')
    tolerance=float(spec['validity'].get('baseline_tolerance',1e-7))
    if not np.isfinite(tolerance) or tolerance<0: raise ValueError('Invalid baseline tolerance')
    if spec['kind']=='exact_linear' and 'limit' in z: raise ValueError('Pointwise support is nonlinear; declare exact_piecewise')
    error=float(np.max(np.abs(pointwise(z,z['baseline_input'])-z['baseline_output'])))
    if spec['kind'].startswith('exact') and error>tolerance:
        raise ValueError('Exact response does not reconstruct recorded native baseline within declared tolerance')
    domain=spec['output_domain']
    if domain['frame']=='world':
        native=service.wb.arrays(dep['state'],domain['object'])['co']
        if z['output_ids'].max()>=len(native):raise ValueError('Response output indices exceed recorded native geometry')
        native_error=float(np.max(np.abs(native[z['output_ids'],domain['axis']]-z['baseline_output'])))
    elif domain['frame']=='native local' and spec.get('adapter_evidence'):
        adapter=service.store.get(spec['adapter_evidence'],'response_adapter')
        if (adapter['converted']['sha256']!=asset['sha256'] or adapter['source_state_id']!=dep['source_state_id']
            or adapter['source_geometry_hash']!=dep['objects'][domain['object']]['geometry'] or domain['axis']!=1):
            raise ValueError('Local ancestry response must match its actual adapter, native object, axis and source state')
        native_error=adapter['current_maximum_error']
    else:
        raise ValueError('Native baseline validation needs recorded world coordinates or the explicitly bound native-local ancestry adapter')
    if spec['kind'].startswith('exact') and native_error>tolerance:
        raise ValueError('Response baseline differs from actual recorded native coordinates')
    payload=dict(dependencies=dependencies,semantic_validation=semantic_validation,spec=spec,arrays=asset,
                 evidence=[pin_link(service,l) for l in evidence],baseline_error=error,
                 native_baseline_error=native_error,
                 coverage=dict(outputs=m,controls=n,nonzeros=len(z['data'])),
                 qualification='Numerical baseline checked. Influence completeness remains scoped to supplied evidence.')
    return dict(response=service.store.put('response_model',payload),**payload)


def predict(service,response,dependencies,control_delta,evidence):
    model=service.store.get(response,'response_model');spec=model['spec'];start=time.perf_counter()
    before=service.store.get(model['dependencies'],'dependency_state');current=service.store.get(dependencies,'dependency_state')
    changed=dependency_diff(before,current,spec['validity']['dependency_domains'])
    if changed: raise ValueError('Response dependencies changed: '+', '.join(c['domain'] for c in changed))
    z=arrays(service,model['arrays']);m,n=validate_arrays(z)
    delta=np.zeros(n,dtype=float)
    if not control_delta: raise ValueError('Explicit control deltas required')
    allowed=set(spec['validity'].get('free_controls',range(n)))
    for key,value in control_delta.items():
        i=int(key)
        if str(i)!=str(key) or i not in allowed or not np.isfinite(value): raise ValueError('Control is outside qualified finite-change scope')
        delta[i]=value
    if np.max(np.abs(delta))>spec['validity']['max_abs_delta']: raise ValueError('Step exceeds response validity; no extrapolation')
    if not evidence: raise ValueError('Prediction needs current guide/depth/section constraints, not a post-check alone')
    constraints=[pin_link(service,l) for l in evidence]
    if spec['kind'] in ('exact_linear','exact_piecewise'):
        candidate=pointwise(z,z['baseline_input']+delta)
    else:
        candidate=z['baseline_output']+multiply(z,delta)
    influence_rows=np.unique(np.repeat(np.arange(m),np.diff(z['indptr']))[np.isin(z['indices'],np.flatnonzero(delta))])
    complete=set(np.flatnonzero(delta))<=set(spec['influence']['complete_for_controls']) and not spec['influence']['missing']
    payload=dict(response=response,dependencies=dependencies,control_delta=control_delta,
                 predicted=candidate.tolist(),output_ids=z['output_ids'].tolist(),affected_output_ids=z['output_ids'][influence_rows].tolist(),
                 kind=spec['kind'],constraints=constraints,elapsed_seconds=time.perf_counter()-start,
                 influence=dict(declared=spec['influence'],complete_for_this_change=complete,
                                meaning='Completeness is bound to supplied ancestry/boundary evidence, not inferred from sparse zeros'),
                 readiness='native comparison required' if complete else 'partial influence; expand boundary evidence before native use',
                 user_appearance_acceptance='not implied')
    return dict(prediction=service.store.put('prediction',payload),**payload)


def compare_native(service,prediction,native_state,object_name,axis,tolerance,inspection):
    pred=service.store.get(prediction,'prediction');model=service.store.get(pred['response'],'response_model')
    if axis not in (0,1,2) or tolerance<0 or not np.isfinite(tolerance): raise ValueError('Finite coordinate tolerance required')
    if object_name!=model['spec']['output_domain'].get('object') or axis!=model['spec']['output_domain'].get('axis'):
        raise ValueError('Native comparison must use the response object and coordinate axis')
    if not inspection.get('evidence') or not all(k in inspection for k in ('whole','close','angles','guide_depth_sections','motion','appearance')):
        raise ValueError('Report actual inspected whole/close/angles/depth/motion and appearance coverage, including pending items')
    actual=service.wb.arrays(native_state,object_name);ids=np.asarray(pred['output_ids'],dtype=int)
    if len(ids) and ids.max()>=len(actual['co']): raise ValueError('Native output indices no longer valid')
    # Matching topology is independently checked; coordinates alone never repair semantics.
    base=service.store.get(model['dependencies'],'dependency_state')
    baseline=service.wb.arrays(base['state'],object_name)
    if not np.array_equal(baseline['tri'],actual['tri']): raise ValueError('Native tessellation changed; explicit correspondence repair required')
    if model['spec']['output_domain']['frame']!='world':
        raise ValueError('Native comparison currently accepts recorded world coordinates only; explicitly transform/rebind local response first')
    difference=actual['co'][ids,axis]-np.asarray(pred['predicted'])
    order=np.argsort(np.abs(difference))[::-1]
    error=float(np.max(np.abs(difference))) if len(ids) else 0
    payload=dict(prediction=prediction,native_state=native_state,object=object_name,axis=axis,tolerance=tolerance,
                 maximum_error=error,p95_error=float(np.percentile(np.abs(difference),95)),
                 worst=[dict(native_vertex=int(ids[i]),error=float(difference[i])) for i in order[:8]],
                 numerical_status='matches' if error<=tolerance else 'regression',
                 inspection=dict(inspection,evidence=[pin_link(service,l) for l in inspection['evidence']]),
                 user_appearance_acceptance='not implied; numerical agreement is not appearance approval',
                 influence=pred['influence'])
    return dict(comparison=service.store.put('prediction_comparison',payload),**payload)


def reuse(service,record,before,after,domains):
    service.store.get(record)
    changes=dependency_diff(service.store.get(before,'dependency_state'),service.store.get(after,'dependency_state'),domains)
    return dict(record=record,reusable=not changes,changed=changes,domains=domains,
                mode='Recorded dependency equivalence only; current live state and semantic validity not inferred',
                limits='Reusing geometry does not reuse a changed view; a response must enforce its own registered domains')
