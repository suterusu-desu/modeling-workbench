"""Explicit adapter for the retained subdivision/ocular export, not universal anatomy."""
from pathlib import Path
import json
import io
import numpy as np
from .store import atomic_write, digest
from .responses import validate_arrays, pointwise


def adapt(service,npz_path,schema_path,verification_path):
    source=service.store.blob(npz_path);schema=service.store.blob(schema_path);verification=service.store.blob(verification_path)
    provenance=json.loads(service.store.resolve_blob(verification).read_text(encoding='utf-8-sig'))
    source_schema=json.loads(service.store.resolve_blob(schema).read_text(encoding='utf-8-sig'))
    if not all(k in provenance for k in ('source_state_id','source_face_geometry_hash','raw_topology','source_modifiers','support_contract')):
        raise ValueError('Actual ancestry verification provenance required')
    if 'downstream_support' not in source_schema: raise ValueError('Missing declared pointwise support schema')
    with np.load(service.store.resolve_blob(source),allow_pickle=False) as loaded:
        z={k:loaded[k] for k in loaded.files}
    converted=dict(data=z['S_data'],indices=z['S_indices'],indptr=z['S_indptr'],shape=z['S_shape'],
                   baseline_input=z['raw_current_xyz'][:,1],baseline_output=z['native_current_xyz'][:,1],
                   output_ids=z['native_vertex_ids'],limit=np.where(z['support_selected'],z['support_limit_Y'],0),
                   selected=z['support_selected'],epsilon=z['support_epsilon'],
                   bound_source_state=np.asarray(provenance['source_state_id']))
    validate_arrays(converted)
    current_error=float(np.max(np.abs(pointwise(converted,converted['baseline_input'])-converted['baseline_output'])))
    quarter_error=float(np.max(np.abs(pointwise(converted,z['raw_quarter_xyz'][:,1])-z['native_quarter_xyz'][:,1])))
    if max(current_error,quarter_error)>3e-7: raise ValueError('Adapted pointwise formula does not reproduce retained independent native arrays')
    buffer=io.BytesIO();np.savez_compressed(buffer,**converted);data=buffer.getvalue()
    path=service.store.root/'derived'/(digest(data)+'.npz');atomic_write(path,data)
    payload=dict(adapter='ancestry_ocular_v1',converted=service.store.blob(path),source=source,schema=schema,verification=verification,
                 kind='exact_piecewise',source_state_id=provenance['source_state_id'],
                 source_geometry_hash=provenance['source_face_geometry_hash'],
                 topology=provenance['raw_topology'],modifiers=provenance['source_modifiers'],support=provenance['support_contract'],
                 frame=source_schema['coordinates'],current_maximum_error=current_error,quarter_maximum_error=quarter_error,
                 measured_outputs=len(converted['output_ids']),raw_controls=len(converted['baseline_input']),
                 limits=['Historical scoped reconstruction, not current T13 validation or final appearance',
                         'Y-only at fixed XZ/pose/topology/modifiers/ocular surfaces',
                         '950 captured outputs are not a complete character influence boundary',
                         'Do not use current derivative as a finite-change rule across support branches'],
                 next='Bind only matching historical dependencies and valid semantics; expand complete influence before native use')
    return dict(adapter_evidence=service.store.put('response_adapter',payload),**payload)
