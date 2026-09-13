"""Exact operation-produced topology relations; no proximity or native writer."""
import json
import math
from pathlib import Path
from .repair_analysis import _pin
from .store import canonical,digest


def _ids(values):
    if not isinstance(values,list) or len(values)>250000 or any(not isinstance(v,str) or not v for v in values):
        raise ValueError('Bounded explicit string identities required')
    if len(set(values))!=len(values):raise ValueError('Duplicate topology identities')
    return set(values)


def _domain(value):
    if not isinstance(value,dict) or any(not value.get(k) for k in ('state','object','domain','topology_hash','scope')):
        raise ValueError('Exact state/object/domain/topology and scope required')
    if value['domain'] not in ('vertex','edge','polygon'):raise ValueError('Explicit topology domain required')
    return _ids(value.get('ids'))


def _read(service,ref):
    return json.loads(service.store.resolve_blob(ref['asset']).read_text(encoding='utf-8-sig'))


def _result(service,payload):
    key=service.store.put('topology_lineage',payload)
    return dict(lineage=key,status='recorded_topology_relations',before_revision=digest(canonical(payload['before'])),
        after_revision=digest(canonical(payload['after'])),summary=payload['summary'],native_ready=False,
        reads={k:dict(operation='read_record',arguments=dict(record=key,path=[k],limit=10,max_chars=8000))
            for k in ('before','after','ancestry','evidence','limits')})


LIMITS=['Operation-supplied topology relations only; no inferred proximity mapping or native writer.',
    'Declared identity scope does not establish coverage of a larger asset.',
    'Topology ancestry does not establish deformation response, anatomy or artistic acceptance.',
    'Weights are retained operation evidence; no automatic interpolation or driver/rig transfer.']


def record(service,case_path):
    p=Path(case_path).resolve();original=service.store.blob(p)
    case=_pin(service,json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig')),p.parent,{})
    if case.get('schema_version')!=1:raise ValueError('Version-1 lineage case required')
    table=_read(service,case['lineage']);receipt=_read(service,case['operation_receipt'])
    if table.get('schema_version')!=1 or not table.get('operation_id'):raise ValueError('Operation-produced version-1 lineage required')
    if (receipt.get('operation_id')!=table['operation_id'] or receipt.get('lineage_sha256')!=case['lineage']['sha256']
            or receipt.get('status')!='topology_change_recorded'):
        raise ValueError('Operation receipt does not bind this exact lineage artifact')
    before,after=table.get('before'),table.get('after');src,dst=_domain(before),_domain(after)
    if before['domain']!=after['domain']:raise ValueError('Cross-domain relations need a separate explicit contract')
    if receipt.get('before_revision')!=digest(canonical(before)) or receipt.get('after_revision')!=digest(canonical(after)):
        raise ValueError('Receipt before/after revision mismatch')
    rows=table.get('relations')
    if not isinstance(rows,list) or len(rows)>250000:raise ValueError('Bounded topology relations required')
    seen_src=set();seen_dst=set();deleted_src=set();mapped_src=set();ancestry=[];counts={};relation_entries=0
    for r in rows:
        a,b=_ids(r.get('sources')),_ids(r.get('targets'));kind=r.get('kind')
        valid={'preserved':len(a)==1 and len(b)==1,'split':len(a)==1 and len(b)>1,
            'merged':len(a)>1 and len(b)==1,'deleted':len(a)==1 and not b,
            'created':not a and len(b)==1,'derived':bool(a and b)}
        if kind not in valid or not valid[kind]:raise ValueError('Relation cardinality does not match declared kind')
        if (not a<=src or not b<=dst or b&seen_dst or
                (kind=='deleted' and a&(deleted_src|mapped_src)) or (kind!='deleted' and a&deleted_src)):
            raise ValueError('Unknown or multiply disposed topology identity')
        relation_entries+=max(1,len(a))*len(b)
        if relation_entries>1500000:raise ValueError('Lineage ancestry budget exceeded')
        seen_src|=a;seen_dst|=b;counts[kind]=counts.get(kind,0)+1
        if kind=='deleted':deleted_src|=a
        else:mapped_src|=a
        weights=r.get('weights')
        if weights is not None:
            if not r.get('weight_semantics') or not isinstance(weights,dict) or set(weights)!=b:
                raise ValueError('Weights need explicit semantics and every target')
            for values in weights.values():
                if not isinstance(values,dict) or set(values)!=a or any(type(v) not in (int,float) or not math.isfinite(v) for v in values.values()):
                    raise ValueError('Weights must explicitly cover every source with finite values')
        for target in sorted(b):
            ancestry.append(dict(target=target,sources=sorted(a),introduced_origins=[] if a else [dict(operation=table['operation_id'],target=target)],
                relation_kind=kind,weights=None if weights is None else weights[target],weight_semantics=r.get('weight_semantics')))
    if seen_src!=src or seen_dst!=dst:raise ValueError('Every declared before/after identity needs an explicit disposition')
    payload=dict(lineage_type='operation',before=before,after=after,ancestry=ancestry,
        summary=dict(before_identities=len(src),after_identities=len(dst),relations=counts),
        evidence=dict(case_source=original,case=case,operation_id=table['operation_id']),limits=LIMITS)
    return _result(service,payload)


def compose(service,lineages):
    if not isinstance(lineages,list) or not 2<=len(lineages)<=64:raise ValueError('Compose 2..64 exact lineage records')
    records=[service.store.get(k,'topology_lineage') for k in lineages]
    for a,b in zip(records,records[1:]):
        if a['after']!=b['before']:raise ValueError('Lineage endpoints differ; stale or unrelated revisions cannot compose')
    ancestry={r['target']:dict(r) for r in records[0]['ancestry']}
    for record in records[1:]:
        following={};entries=0
        for row in record['ancestry']:
            sources=set();novel=list(row['introduced_origins'])
            for source in row['sources']:
                previous=ancestry[source];sources.update(previous['sources']);novel.extend(previous['introduced_origins'])
            unique={canonical(n):n for n in novel}
            entries+=len(sources)+len(unique)
            if entries>1500000:raise ValueError('Composed ancestry budget exceeded; retain a shorter scoped chain')
            following[row['target']]=dict(target=row['target'],sources=sorted(sources),
                introduced_origins=[unique[k] for k in sorted(unique)],relation_kind='composed',weights=None,
                weight_semantics='No composed weight rule inferred; original weights remain in linked operation records.')
        ancestry=following
    rows=list(ancestry.values())
    payload=dict(lineage_type='composition',before=records[0]['before'],after=records[-1]['after'],ancestry=rows,
        summary=dict(operations=len(lineages),before_identities=len(records[0]['before']['ids']),after_identities=len(rows),
            targets_with_introduced_ancestry=sum(bool(r['introduced_origins']) for r in rows)),
        evidence=dict(lineages=lineages),limits=LIMITS+['Composition preserves introduced ancestry and does not invent a weight-combination rule.'])
    return _result(service,payload)


def remap(service,lineage,selection,expected_source,direction='forward',policy='strict'):
    if direction not in ('forward','reverse') or policy not in ('strict','any','all'):
        raise ValueError('Explicit forward/reverse direction and strict/any/all policy required')
    record=service.store.get(lineage,'topology_lineage')
    source=record['before' if direction=='forward' else 'after'];target=record['after' if direction=='forward' else 'before']
    if expected_source!=digest(canonical(source)):raise ValueError('Selection belongs to a stale or different topology revision')
    chosen=_ids(selection)
    if not chosen<=set(source['ids']):raise ValueError('Unknown selected source identity')
    if direction=='forward':rows=record['ancestry']
    else:
        inverse={v:[] for v in target['ids']}
        for row in record['ancestry']:
            for v in row['sources']:inverse[v].append(row['target'])
        rows=[dict(target=v,sources=parents,introduced_origins=[]) for v,parents in inverse.items()]
    candidates=[];unambiguous=[];ambiguous=[];reached=set()
    for row in rows:
        ancestors=set(row['sources']);hit=ancestors&chosen
        if not hit:continue
        candidates.append(row['target']);reached|=hit
        if ancestors<=chosen and not row['introduced_origins']:unambiguous.append(row['target'])
        else:ambiguous.append(dict(target=row['target'],selected_sources=sorted(hit),unselected_sources=sorted(ancestors-chosen),introduced_origins=row['introduced_origins']))
    selected=None if policy=='strict' and ambiguous else sorted(candidates if policy=='any' else unambiguous)
    payload=dict(lineage=lineage,direction=direction,policy=policy,source_revision=expected_source,
        target_revision=digest(canonical(target)),input_selection=selection,selection=selected,candidates=sorted(candidates),
        ambiguous=ambiguous,without_counterpart=sorted(chosen-reached),native_ready=False,
        introduced_ancestry=[dict(target=r['target'],origins=r['introduced_origins']) for r in record['ancestry']
            if direction=='reverse' and r['target'] in chosen and r['introduced_origins']],
        limits=['Exact ancestry remap only. Strict mode withholds a selector for partial merge/derived ancestry.',
            'Any/all policies are explicit selection semantics, not permission to modify geometry.'])
    key=service.store.put('topology_selection',payload)
    return dict(selection_record=key,status='ambiguous' if selected is None else 'mapped',selected_count=None if selected is None else len(selected),
        ambiguous_targets=len(ambiguous),without_counterpart=len(chosen-reached),target_revision=payload['target_revision'],native_ready=False,
        reads={k:dict(operation='read_record',arguments=dict(record=key,path=[k],limit=10,max_chars=8000))
            for k in ('selection','candidates','ambiguous','without_counterpart','introduced_ancestry','limits')})
