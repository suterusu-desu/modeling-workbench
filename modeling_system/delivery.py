"""Revision-bound exported representation checks for execution receipts."""
import numpy as np
from .recovery import file_status


def inspect(service, case):
    if not isinstance(case,dict):raise ValueError('delivery must be an object')
    for k in ('source','current_source','native_mechanism','scope','deferred','objects','materials','exports','samples','tolerance','units'):
        if k not in case:raise ValueError('delivery requires '+k)
    if not case['native_mechanism'] or not case['scope'] or not case['objects'] or not case['exports'] or not case['units']:
        raise ValueError('Delivery mechanism, scope, objects, exports and units must be explicit')
    if not isinstance(case['native_mechanism'],dict) or 'asset' not in case['native_mechanism']:
        raise ValueError('delivery.native_mechanism needs a pinned {path, sha256} native mechanism/control capture')
    source=file_status(case['source']);current=file_status(case['current_source'])
    exports=[file_status(r) for r in case['exports']]
    current_revision=(source['status']=='verified' and current['status']=='verified' and source['sha256']==current['sha256'])
    tolerance=case['tolerance']
    if type(tolerance) not in (float,int) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('delivery.tolerance must be finite and nonnegative in declared units')
    rows=[];between=False
    for sample in case['samples']:
        if not isinstance(sample.get('controls'),dict) or sample.get('kind') not in ('anchor','between_anchor'):
            raise ValueError('Delivery samples need explicit controls and kind anchor|between_anchor')
        if sample.get('source_sha256')!=source['sha256'] or sample.get('export_sha256') not in {r['sha256'] for r in exports}:
            raise ValueError('Delivery sample must bind exact source/export hashes')
        if sample.get('object') not in case['objects']:raise ValueError('Delivery sample object is outside declared coverage')
        # Measured arrays are pinned by the existing execution-case importer.
        ref=sample.get('comparison')
        if not isinstance(ref,dict) or 'asset' not in ref:raise ValueError('Delivery sample comparison needs exact NPZ {path, sha256} with source/output arrays')
        with np.load(service.store.resolve_blob(ref['asset']),allow_pickle=False) as z:
            a,b=np.asarray(z['source'],float),np.asarray(z['output'],float)
        if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or not len(a) or not np.isfinite([a,b]).all():
            raise ValueError('Delivery comparison needs corresponding finite nonempty Nx3 source/output samples')
        distance=np.linalg.norm(a-b,axis=1)
        rows.append(dict(object=sample['object'],controls=sample['controls'],kind=sample['kind'],
            samples=len(a),maximum=float(distance.max()),rms=float(np.sqrt(np.mean(distance**2))),
            exceeds_tolerance=int(np.count_nonzero(distance>tolerance)),comparison=ref))
        between|=sample['kind']=='between_anchor'
    missing=[name for name in case['objects'] if not any(r['object']==name and r['kind']=='between_anchor' for r in rows)]
    worst=sorted(rows,key=lambda r:r['maximum'],reverse=True)
    return dict(source=source,current_source=current,exports=exports,current_revision=current_revision,
        status='stale' if not current_revision or any(r['status']!='verified' for r in exports) else
               'needs_samples' if not between or missing else 'residual_exceeds_tolerance' if any(r['exceeds_tolerance'] for r in rows) else 'sampled_agreement',
        samples=rows,missing_between_anchor_objects=missing,scope=case['scope'],deferred=case['deferred'],
        native_mechanism=case['native_mechanism'],objects=case['objects'],materials=case['materials'],units=case['units'],
        next_sample_basis=[dict(object=r['object'],controls=r['controls'],maximum=r['maximum']) for r in worst[:5]],
        limits='Verify native correspondence and refine sampling around measured error; finite samples do not prove continuous motion. More samples cannot repair a discontinuous source. Appearance acceptance is separate.')
