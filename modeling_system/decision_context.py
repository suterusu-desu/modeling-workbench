"""Applicable procedures and actual source passages at the operation decision point."""
from pathlib import Path
import json
from .bindings import load_binding, fingerprint_files
from .experience import retrieve, applicability, passages


def operation_context(service, stage, context=None, job=None, retain=False):
    binding = load_binding(service.workspace)
    context = context or {}
    if job:
        intent=service.ledger.read(job)['intent'];settings=intent.get('settings',{})
        context={**context,**{k:settings[k] for k in ('use_case','intent_class','comparison_id','reconstruction_id') if k in settings},
                 'provider':intent.get('provider'),'kind':intent.get('kind')}
    policy_path = service.policy_path
    policy = json.loads(policy_path.read_text(encoding='utf-8-sig')) if policy_path.is_file() else None
    index = service.workspace/binding.get('procedures', 'modeling_system/procedures.json')
    rules = json.loads(index.read_text(encoding='utf-8-sig')).get('procedures', []) if index.is_file() else []
    selected = []
    for rule in rules:
        if stage not in rule['stages']:
            continue
        row = dict(rule, applicability=applicability(rule.get('conditions', {}), context))
        row['source_passages'] = []
        # Each procedural requirement retrieves its own decisive passage. A large
        # authority heading or unrelated more frequent terms cannot crowd it out.
        seen=set()
        for citation in rule.get('citations', []):
            candidates=[]
            for source in binding.get('authority',[]):
                if source.get('role')!=citation['role']: continue
                path=service.workspace/source['path']
                if path.suffix.lower()!='.md' or not path.is_file(): continue
                found=passages(path,citation.get('heading',citation.get('contains','')))
                candidates.extend(r for r in found if
                    (citation.get('heading','').casefold() in r['title'].casefold() if 'heading' in citation else
                     citation['contains'].casefold() in r['excerpt'].casefold()))
            candidates.sort(key=lambda r:-r['score'])
            for found in candidates[:1]:
                key=(found['sha256'],found['line_start'],found['line_end'])
                if key in seen: continue
                seen.add(key)
                row['source_passages'].append({k:found[k] for k in ('source','sha256','line_start','line_end','title','excerpt','coverage')})
        row['available_operations'] = [n for n in row['operations'] if n in service.operations()]
        selected.append(row)
    result = dict(stage=stage, context=context, character=binding['character'], procedures=selected,
                  authority=fingerprint_files(service.workspace, binding.get('authority', [])),
                  provider_policy=policy if stage in ('generation','guide_qualification','recovery') else None,
                  route_selection=service.select_generation_route(context.get('use_case','general')) if stage == 'generation' else None,
                  resource_effect='Read-only local files/records; no store, provider or native writes',
                  limits=['Applicability and source passages guide agent judgment, not automatic artistic approval',
                          'Policy is checked again at actual claim; this is not a live panel preflight'])
    if job:
        from .episodes import summarize_workflow
        item = service.ledger.read(job)
        result['job'] = summarize_workflow(service,job)
        result['recovery'] = ('Reconcile this same job before any dispatch' if item['status'] in
                              ('dispatching','submitted','unknown') else 'Inspect actual delivered outputs and reviews; do not infer viewer identity')
    result['coverage']={'procedure_count':len(selected),'source_passages':sum(len(p['source_passages']) for p in selected),
                        'missing_citations':[p['id'] for p in selected if not p['source_passages']]}
    if not retain:
        result['retention']={'status':'not retained','operation':'capture_operation_context','arguments':dict(stage=stage,context=context,job=job)}
        return result
    pinned=[]
    for entry in result['authority']['files']:
        if not entry['available']:continue
        asset=service.store.blob(entry['path'])
        if asset['sha256']!=entry['sha256']:raise ValueError('Authority changed during explicit context capture; reread before retaining')
        pinned.append(dict(source=entry['path'],asset=asset))
    result.update(pinned_authority=pinned,resource_effect='Explicit local context/assets and operation journal retention; no native/provider call')
    record=service.store.put('decision_context',result)
    return dict(result,context_record=record,retention={'status':'retained','record':record})
