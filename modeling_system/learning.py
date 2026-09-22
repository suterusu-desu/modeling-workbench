"""Workspace learning from exact review receipts and explicitly authored lessons.

Only already-public visual judgments and deliberate lesson text enter selection.
Private evidence and scope identity remain in the content-addressed store. Review
recall does not interpret images, operate Blender or imply formal method promotion.
"""
from copy import deepcopy
from pathlib import Path

from .controller import fingerprint, read_json
from .episodes import pin_link
from .store import canonical


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


def validate_lesson(lesson):
    fields = {'mechanism', 'observation', 'next_use', 'limits', 'conditions'}
    if (not isinstance(lesson, dict) or set(lesson) != fields
            or any(not isinstance(lesson[key], str) or not lesson[key].strip()
                   for key in fields - {'conditions'})
            or not isinstance(lesson['conditions'], dict)):
        raise ValueError('A lesson needs public mechanism, observation, next_use, limits and conditions')
    for key, value in lesson['conditions'].items():
        values = value if isinstance(value, list) else [value]
        if (not isinstance(key, str) or not key or not values
                or any(type(item) not in (str, bool, int, float) for item in values)):
            raise ValueError('Lesson conditions use explicit named scalar values or alternatives')
    if len(canonical(lesson)) > 1600:
        raise ValueError('Keep the decision lesson within 1600 bytes; pin detail as evidence')
    return deepcopy(lesson)


def record_lesson(service, *, lesson, evidence):
    """Retain a scoped, deliberately public interpretation with exact private proof.

    Conditions describe where the observation was made, not universal necessary
    or sufficient rules. Use this for a cross-case lesson; ordinary record_review
    submissions are retained automatically and may include the same lesson shape.
    """
    lesson = validate_lesson(lesson)
    if not evidence:
        raise ValueError('A retained lesson requires actual supporting evidence')
    return service.store.put('modeling_experience', {
        'public': {'lesson': lesson},
        'evidence': [pin_link(service, link) for link in evidence],
        'origin': 'authored lesson', 'claims': 'Scoped observation; no automatic method promotion'})


def record_method_experience(service, *, lesson, evidence, knowledge_status, candidate_disposition,
                             interpretation, protected_outcomes=(), supersedes=(), authority='reviewer',
                             classification=None, review_fact=None):
    """Idempotent reviewed import/close-out, with knowledge and candidate separate.

    Demonstrated knowledge may concern a rejected candidate. Interpretations are
    explicit source-backed reviewer input, never automatic aesthetic judgments.
    Model classification is advisory until supplied by that reviewer. A known
    fact can be recorded directly without another provider call.
    """
    lesson = validate_lesson(lesson)
    if (knowledge_status not in ('demonstrated', 'conditional', 'unverified', 'superseded')
            or candidate_disposition not in ('scoped_success', 'failed', 'rejected', 'unresolved')
            or authority not in ('reviewer', 'user', 'historical_curator')
            or not interpretation or not evidence
            or classification not in (None, 'recurring_failure', 'supported_lesson', 'unresolved_hypothesis')):
        raise ValueError('Explicit knowledge status, candidate disposition and source-backed interpretation required')
    if classification == 'supported_lesson' and knowledge_status != 'demonstrated':
        raise ValueError('A supported classification requires demonstrated scoped evidence')
    if review_fact is not None:
        fact = service.store.get(review_fact, 'operation_fact')
        if fact['intent']['operation'] != 'review_modeling_task':
            raise ValueError('Post-review experience needs an actual completed visual review fact')
        actual = fact['outcome']['result']['judgment']['disposition']
        if actual == 'rejected' and candidate_disposition == 'scoped_success':
            raise ValueError('Lesson classification cannot reverse the actual rejected review')
    pinned = [pin_link(service, link) for link in evidence]
    protected = [pin_link(service, link) for link in protected_outcomes]
    identity = fingerprint({'mechanism': lesson['mechanism'].strip().casefold(),
        'evidence': sorted(pinned, key=canonical), 'protected_outcomes': sorted(protected, key=canonical)})
    public = {'lesson': lesson, 'knowledge_status': knowledge_status,
        'candidate_disposition': candidate_disposition, 'classification': classification,
        'interpretation': interpretation}
    prior = dict(service.store.records(('modeling_experience',)))
    for key in supersedes:
        old = prior.get(key)
        if old is None: raise ValueError('Explicit correction must identify existing experience')
        if old.get('authority') == 'user' and old['public'].get('candidate_disposition') == 'rejected' and authority != 'user':
            raise ValueError('Only new user authority can supersede an explicit user rejection')
    for key, old in prior.items():
        if old.get('continuity_identity') == identity:
            if old['public'] == public and old.get('authority') == authority and old.get('review_fact') == review_fact:
                return key
            if key not in supersedes and key not in {v for r in prior.values() for v in r.get('supersedes', [])}:
                raise ValueError('Changed interpretation requires explicit correction provenance')
    return service.store.put('modeling_experience', {'public': public, 'evidence': pinned,
        'protected_outcomes': protected, 'continuity_identity': identity, 'authority': authority,
        'supersedes': list(supersedes), 'review_fact': review_fact,
        'origin': 'reviewed source-backed experience',
        'claims': 'Knowledge status and candidate outcome are separate; no automatic appearance or method promotion'})


def import_method_experiences(service, records):
    """Import reviewed records once; raw historical prose is never auto-promoted."""
    return {'records': [record_method_experience(service, **row) for row in records],
            'native_calls': 0, 'provider_calls': 0}


def retain_review(service, directory, review):
    """Index an existing immutable review, without reopening its scene or queue."""
    directory = Path(directory)
    binding = read_json(directory / 'operating-session.json')
    if Path(binding['workspace']).resolve() != Path(service.workspace).resolve():
        raise ValueError('Review belongs to a different workspace')
    fact = service.store.get(review['operation_fact'], 'operation_fact')
    intent = fact['intent']
    if (intent['operation'] != 'review_modeling_task' or intent['episode'] != binding['episode']
            or intent['handle'] != review['operation_handle']):
        raise ValueError('Review receipt identity or episode differs from its binding')
    actual = fact['outcome']['result']
    if any(review.get(key) != value for key, value in actual.items()):
        raise ValueError('Review index differs from its immutable episode fact')
    judgment = actual['judgment']
    if (set(judgment) != {'disposition', 'scope', 'reason', 'next_question'}
            or judgment['disposition'] not in ('useful', 'rejected', 'unresolved')):
        raise ValueError('A scoped visual review receipt is required')
    public = {'review': deepcopy(judgment)}
    if actual.get('lesson') is not None:
        public['lesson'] = validate_lesson(actual['lesson'])
    # Task IDs, paths, hashes, timestamps and native bindings are never projected.
    identity = {'directory': str(directory.resolve()), 'episode': binding['episode'],
                'task': actual['task'], 'basis': actual['basis']}
    return service.store.put('modeling_experience', {
        'public': public, 'evidence': deepcopy(actual['evidence']),
        'origin': 'actual scoped review', 'identity': fingerprint(identity),
        'source': {**identity, 'review_fact': review['operation_fact'],
                   'operation_handle': review['operation_handle']},
        'submitted_at': actual['submitted_at'],
        'claims': 'Historical visual judgment; not current state, retention or user acceptance'})


def import_review_history(service, directories):
    """One-time migration of exact receipts; no replay or edits to old scopes.

    Also repairs an interrupted learning index after the review fact was saved.
    Repeating this operation returns the same content IDs.
    """
    records = []
    for directory in directories:
        for path in sorted((Path(directory) / 'reviews').glob('*.json')):
            records.append(retain_review(service, directory, read_json(path)))
    return {'records': sorted(set(records)), 'native_calls': 0, 'provider_calls': 0}


def experience_rows(store):
    """Latest interpretation of each exact review basis, plus separate lessons.

    Corrected judgments supersede their earlier interpretation in retrieval only.
    Old immutable records stay available; a changed candidate is a different case.
    """
    latest, lessons = {}, []
    records = list(store.records(('modeling_experience',)))
    superseded = {key for _, value in records for key in value.get('supersedes', [])}
    for key, value in records:
        if key in superseded: continue  # Original immutable evidence remains readable.
        if 'identity' not in value:
            lessons.append((key, value))
            continue
        identity = value['identity']
        old = latest.get(identity)
        order = lambda row: (row[1]['submitted_at'], row[1]['source']['operation_handle'])
        if old is None or order((key, value)) > order(old):
            latest[identity] = (key, value)
    return sorted(lessons + list(latest.values()))


def public_experience(row):
    """Project only the dedicated, deliberately public part of learning records."""
    if row.get('record_kind') != 'modeling_experience':
        return None
    content = deepcopy(row['excerpt']['public'])
    fit = row['applicability']
    if 'lesson' in content:
        # Context values can be private. Expose only the condition names and
        # whether they match; the authored observation remains a historical fact.
        content['condition_comparison'] = {
            'status': fit['status'], 'unknown': fit['missing'],
            'different': [conflict['field'] for conflict in fit['conflicts']]}
    return content
