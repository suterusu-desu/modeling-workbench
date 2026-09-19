"""Decision-first display projection; canonical records and lifecycle guards stay separate."""
from collections import Counter
from .bounded_reads import json_chars, page, read_descriptor
from .store import canonical, digest
from .operation_reads import counts, header, unresolved

SUMMARY_CHARS=24000
SECTIONS={'context','question','baseline','workflows','authority','authority_changed_since_entry',
          'authority_location_rebindings','judgment_summary','next_operations'}


def record_read(record, path=None):
    return read_descriptor('read_record',{'record':record},path)


def section_read(episode, section):
    return read_descriptor('decision_workspace',{'episode':episode,'detail':'section','section':section})


def metadata(total, shown, revision, basis, coverage):
    return dict(total=total,shown=shown,deferred=total-shown,source_revision=revision,
                selection_basis=basis,coverage=coverage)


def section(raw, name, path, offset, limit, max_chars, expected_view):
    if name not in SECTIONS: raise ValueError('Unsupported workspace section; choose '+', '.join(sorted(SECTIONS)))
    if name not in raw: raise KeyError(name)
    return page(raw[name],'decision_workspace',{'episode':raw['episode'],'detail':'section','section':name},
                path=path,offset=offset,limit=limit,max_chars=max_chars,expected_view=expected_view,
                selection_basis='Exact composed section at observation fingerprint; no native freshness claim')


def summary(service, raw, item):
    result={k:v for k,v in raw.items() if k not in ('workflows','operations','operation_leases','context','authority','next_operations')}
    episode=raw['episode'];revision=raw['revision'];context=dict(raw['context']);links=context.pop('links',[])
    context_path=['data','context'] if 'context' in item['data'] else ['intent','context']
    context['links']={'projection':'deferred_evidence_catalog'}
    # Tiny indexes remain useful, but are still marked as noncanonical projections.
    if len(links)<=4:
        from .episodes import link_id
        context['links']=[dict(projection='evidence_link_summary',link_index=i,link_id=link_id(l),
                               kind=l['kind'],role=l['role']) for i,l in enumerate(links)]
    context['expand_links']=record_read(revision,context_path+['links'])
    result['context']=context
    sections={'context.links':metadata(len(links),len(links) if len(links)<=4 else 0,revision,'All if <=4; otherwise deferred',
                    'Exact canonical links at immutable episode ID')}
    sections['context.links']['by_kind']=dict(Counter(l['kind'] for l in links))
    for field,value in context.items():
        if isinstance(value,list) and field!='links':
            sections['context.'+field]=metadata(len(value),len(value),revision,'All authored entries','Exact; may predate later effects')
    workflows=[];reviews={}
    for w in raw['workflows']:
        # Do not infer relevance from a filename or a most-recent generation.
        fields=('kind','handle','status','recovery','review','qualified_support','target_disposition')
        value={k:v for k,v in w.items() if k in fields and v is not None and v!=[]}
        disposition=dict(value.get('target_disposition',{}))
        rejection=disposition.pop('evidence',None)
        if rejection:
            disposition.update(record=w['guide_rejection'],reason=rejection.get('reason'),expand=record_read(w['guide_rejection']))
        if disposition:value['target_disposition']=disposition
        review=w.get('source_review')
        if review:
            reviews[w['review']]={k:v for k,v in review.items() if k in ('disposition','region','guard_band_pixels','conflicts','judgment_source')}
            reviews[w['review']]['expand']=record_read(w['review'])
        value['expand']=record_read(w['revision'])
        if w['kind']=='diagnostic_recipe':
            value.update({k:w[k] for k in ('steps','next_read','reason') if k in w})
        workflows.append(value)
    result['workflows']=workflows;result['source_reviews']=reviews
    result['question']={'role':raw['question']['role'],
                        'expand':record_read(raw['question']['record'])}
    baseline=raw['baseline']
    result['baseline']={k:baseline[k] for k in ('label','state','source_state_id','guide')}
    result['baseline']['coverage']={k:v for k,v in baseline['coverage'].items() if k in ('inventoried','queryable','excluded','declaration')}
    result['baseline']['expand']=record_read(baseline['state'])
    sections['workflows']=metadata(len(workflows),len(workflows),digest(canonical(raw['workflows'])),
        'All associated workflows',
        'Exact dispositions/support/exclusions; expand prompts, receipts and trial data')
    sections['source_reviews']=metadata(len(reviews),len(reviews),digest(canonical(reviews)),
        'All linked reviews','Exact disposition/region/conflicts; agent review, not user acceptance')
    rows=raw['operations'];leases=raw['operation_leases'];totals=counts(rows)
    pending=[r for r in rows if unresolved(r)]
    result['operations']=[header(service,r) for r in pending]
    for operation in result['operations']: operation.pop('episode',None)
    pending_leases=[r for r in leases if r['observed_status']!='finished']
    result['operation_leases']=[]
    for row in pending_leases:
        value={k:v for k,v in row.items() if k not in ('path','episode')}
        value['next_action']=('Wait for the actual owner; no reconciliation of running work' if row['observed_status']=='active' else
            'Preserve the returned result; reconcile the original operation and actual effects, never replay automatically')
        value['expand']=read_descriptor('inspect_operations',{'episode':episode,'handle':row['id'],'view':'lease'})
        result['operation_leases'].append(value)
    sections['operations']=metadata(len(rows),len(pending),digest(canonical(sorted(rows,key=lambda r:r['handle']))),
        'All unresolved; completed deferred','Exact effects/recovery; display only')
    sections['operations'].update(totals)
    sections['operations']['expand']=read_descriptor('inspect_operations',{'episode':episode,'view':'index'})
    sections['operation_leases']=metadata(len(leases),len(pending_leases),digest(canonical(sorted(leases,key=lambda r:r['id']))),
        'All nonfinished leases, even without call intent','Observed status; no native freshness')
    sections['operation_leases'].update(active=sum(r['observed_status']=='active' for r in leases),unresolved=len(pending_leases))
    sections['operation_leases']['expand']=read_descriptor('inspect_operations',{'episode':episode,'view':'leases'})
    result['history']={'total_operations':totals['total'],'unresolved_operations':totals['unresolved'],
                       'active_operations':totals['active'],'unknown_effect_operations':totals['unknown'],
                       'completed_operations':len(rows)-len(pending),'omitted_completed_operations':len(rows)-len(pending),
                       'total_leases':len(leases),'unresolved_leases':len(pending_leases)}
    result['authority']={'expand':section_read(episode,'authority'),
                         'coverage':'Inventory deferred; all changes inline'}
    for name in ('authority_changed_since_entry','authority_location_rebindings'):
        sections[name]=metadata(len(raw[name]),len(raw[name]),digest(canonical(raw[name])),
            'All changes/rebindings','Current files vs pinned entry')
    result['next_operations']={'expand':section_read(episode,'next_operations')}
    sections['next_operations']=metadata(len(raw['next_operations']),0,revision,
        'All deferred','Recovery stays inline; plans grant no authorization')
    result['sections']=sections
    result.update(projection='decision_workspace_summary',incomplete_decision_coverage=False,
        summary_limit_json_chars=SUMMARY_CHARS,
        coverage='Current decision, dispositions and unresolved effects. Historical catalogs/payloads deferred. Display projection, not canonical editable data.',
        interpretation_sources='Authored context may predate effects. Current workflow dispositions and scoped judgment stay separate; applied/saved is not appearance acceptance.')
    # Preserve the exact retained checkpoint return without a live-state claim.
    checkpoint=next((r for r in rows if r['operation']=='native_save_checkpoint' and r['status']=='returned' and r['record']),None)
    if checkpoint:
        saved=service.store.get(checkpoint['record'],'operation_fact')['outcome']['result']
        result['latest_saved_checkpoint']=dict(file=saved.get('file'),status=saved.get('status'),utc=checkpoint['utc'],
            meaning='Recorded return only; no live freshness or appearance acceptance',expand=record_read(checkpoint['record'],['outcome','result']))
    if json_chars(result)>SUMMARY_CHARS-512:
        result=overflow(raw,sections)
    return result


def overflow(raw, sections):
    # Even an arbitrarily large question/owner/requirement must not bypass the cap.
    keep=('episode','revision','judgment','runtime','full_episode_record')
    result={k:raw[k] for k in keep if k in raw}
    result.update(projection='decision_workspace_summary',overflow=True,incomplete_decision_coverage=True,
        summary_limit_json_chars=SUMMARY_CHARS,
        coverage='Indispensable decision/safety content exceeds the summary budget. This is incomplete orientation, not sufficient context for mutation. Expand current context, judgment, authority, workflows and every unresolved operation/lease before deciding.',
        freshness=raw['freshness'],user_appearance_acceptance='Never inferred',
        sections={k:{**v,'shown':0,'deferred':v['total'],'coverage':'Deferred by overflow; read exact section before deciding'} for k,v in sections.items()},
        reads={name:section_read(raw['episode'],name) for name in sorted(SECTIONS) if name in raw})
    result['reads']['episode']=record_read(raw['revision'])
    result['reads']['operations']=read_descriptor('inspect_operations',{'episode':raw['episode'],'view':'index','selection':'unresolved'})
    result['reads']['operation_leases']=read_descriptor('inspect_operations',{'episode':raw['episode'],'view':'leases','selection':'unresolved'})
    return result
