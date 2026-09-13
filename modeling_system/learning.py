"""Evidence-backed procedure promotion without invented causal/appearance success."""
from .episodes import pin_link


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
