"""Evidence-backed procedure promotion without invented causal/appearance success."""
from .episodes import pin_link


def integration_disposition(service,method,close=False):
    """Keep an actionable method close-out inside the existing judgment API."""
    if method is None:return None
    value=dict(method);integration=value.get('integration')
    if integration is None:
        if close:raise ValueError('Method close-out needs integration: implemented, experimental, or not_generalizable')
        return value
    if not isinstance(integration,dict) or integration.get('disposition') not in ('implemented','experimental','not_generalizable') or not integration.get('reason'):
        raise ValueError('Integration needs an explicit disposition and reason')
    integration=dict(integration)
    disposition=integration['disposition']
    if disposition in ('implemented','experimental'):
        if not all(integration.get(k) for k in ('mechanism','scope','artifacts','limits')):
            raise ValueError('Implemented/experimental integration needs mechanism, scope, artifact links and limits')
        if disposition=='implemented' and value.get('status')!='supported':
            raise ValueError('Implemented benefit requires a narrowly supported method judgment; unfinished benefits remain experimental')
    integration['artifacts']=[pin_link(service,l) for l in integration.get('artifacts',[])]
    adoption=integration.get('adoption',{})
    if disposition=='implemented' and set(adoption)!={'source','installed','runtime','operator'}:
        raise ValueError('Implemented integration needs separate source/installed/runtime/operator adoption dispositions')
    normalized={}
    for stage,entry in adoption.items():
        if stage not in ('source','installed','runtime','operator') or not isinstance(entry,dict) or entry.get('status') not in ('evidenced','pending','not_applicable') or not entry.get('reason'):
            raise ValueError('Adoption requires stage, evidenced/pending/not_applicable status and reason')
        if entry['status']=='evidenced' and not entry.get('evidence'):
            raise ValueError('Evidenced adoption requires actual receipt links')
        normalized[stage]=dict(entry,evidence=[pin_link(service,l) for l in entry.get('evidence',[])])
    integration['adoption']=normalized
    integration['adoption_complete']=bool(normalized) and all(x['status']!='pending' for x in normalized.values()) and set(normalized)=={'source','installed','runtime','operator'}
    value['integration']=integration
    return value


def promote(service,procedure_id,judgments,instruction,stages,conditions,limits,counterexamples,executable_paths,level):
    if level not in ('local','reusable') or not instruction.strip() or not stages or not limits:
        raise ValueError('Procedure needs instruction, stages, explicit limits and local/reusable level')
    outcomes=[service.store.get(k,'episode_judgment') for k in judgments]
    if not outcomes or any((o.get('method') or {}).get('status')!='supported' for o in outcomes):
        raise ValueError('Promotion requires explicitly supported method judgments; character status does not establish method success')
    if level=='reusable' and len({o['episode'] for o in outcomes})<2:
        raise ValueError('Reusable promotion requires demonstrated reuse in distinct episodes')
    payload=dict(procedure_id=procedure_id,judgments=judgments,instruction=instruction,stages=stages,
                 conditions=conditions,limits=limits,counterexamples=[pin_link(service,l) for l in counterexamples],
                 executable=[pin_link(service,dict(kind='file',path=p,role='exact procedure executable')) for p in executable_paths],
                 level=level,adoption='local record; installed/operator use must be observed separately',
                 appearance_acceptance='not implied')
    return dict(procedure=service.store.put('procedure',payload),**payload)
