"""Bind the generic episode-backed session to the existing direct TypeSafe ledger."""
from copy import deepcopy
from pathlib import Path
from modeling_system.controller import fingerprint
from modeling_system.operating_session import OperatingSession
from .planning_batch import decide


def create_session(directory, *, service, episode, owner, goal, observe_context,
                   catalog, handlers, public_projection, ledger_directory, report=None, budget=None, experience=None, preservation=None):
    directory = Path(directory)
    session = None

    def judge(public_state, decisions, binding):
        return decide(public_state, decisions,
            current_context=lambda: session.judgment_context(binding),
            ledger_directory=ledger_directory, output_directory=directory/'judgments')

    session = OperatingSession(directory, service=service, episode=episode,
        owner=owner, goal=goal, observe_context=observe_context, catalog=catalog,
        handlers=handlers, judge=judge, public_projection=public_projection, report=report, budget=budget, experience=experience, preservation=preservation, require_preservation=True)
    _apply_authority_overrides(session)
    return session


def recover_completed_selection(session, ledger_directory, *, response_path=None):
    """Recover the current exact provider answer without a request or native call."""
    from modeling_system.controller import read_json
    from modeling_system.provider_recovery import reconcile_completed_response
    from modeling_system.judgments import prepare_judgments
    ledger = Path(ledger_directory)
    controller = read_json(session.directory / 'controller' / 'controller.json')
    pending = controller.get('selection')
    if pending is None:
        raise ValueError('No pending selection to recover')
    expected = fingerprint(pending)
    batch = read_json(session.selector.directory / 'batches' / (expected + '.json'))
    attempts = [a for a in read_json(ledger / 'budget.json')['attempts']
                if a['packet_digest'] == fingerprint(batch['packet'])]
    if len(attempts) != 1:
        raise ValueError('Exactly one matching original provider attempt required')
    call = Path(attempts[0]['output'])
    reconcile_completed_response(call, ledger, session.observe, response_path=response_path)
    return session.recover_completed_selection(expected_selection=expected,
        packet_path=call/'owner-packet.json', request_path=call/'decision-1.request.json',
        response_path=call/'decision-1.response.json',
        reconciliation_path=call/'completed-response-reconciliation.json')


def retry_terminal_selection(session, ledger_directory, *, reason):
    """Owner-only explicit one-HTTP503 retry; resume native work separately."""
    from modeling_system.controller import applicable, read_json, write_json
    from modeling_system.provider_recovery import prepare_terminal_retry, reconcile_completed_response
    from .provider_dispatch import dispatch_many
    ledger = Path(ledger_directory)
    controller_path = session.directory/'controller/controller.json'
    if (controller_path.parent/'controller.lock').exists():
        raise ValueError('Stop the controller before terminal-response recovery')
    record = read_json(controller_path); pending = record.get('selection')
    if pending is None or any(a['status'] in ('running','needs_reconciliation') for a in record['attempts']):
        raise ValueError('Exact pending choice and no uncertain native effect required')
    expected = fingerprint(pending)
    batch = read_json(session.selector.directory/'batches'/(expected+'.json'))
    def current():
        state = session.observe()
        if (state['active_operations'] or state['owner'] != pending['state']['owner']
                or state['authority_revision'] != pending['state']['authority_revision']
                or state['stage'] != pending['state']['stage'] or state['actions'] != pending['actions']
                or not applicable(pending['plan'], state)
                or session.selector.project(deepcopy(state),deepcopy(state['actions']),deepcopy(pending['plan'])) != batch['public']
                or fingerprint(read_json(controller_path).get('selection')) != expected):
            raise ValueError('Original pending selection context changed; no retry or release')
        return state
    current()
    matches = [a for a in read_json(ledger/'budget.json')['attempts'] if a['packet_digest'] == fingerprint(batch['packet'])]
    if len(matches) != 1: raise ValueError('Exactly one original failed provider attempt required')
    original = Path(matches[0]['output'])
    authorization = prepare_terminal_retry(original, ledger, current, reason=reason)
    packet = authorization['retry_packet']
    packet_path = session.directory/'judgments'/(fingerprint(packet)+'-terminal-retry-packet.json')
    if packet_path.exists():
        if read_json(packet_path) != packet: raise ValueError('Retry packet changed')
    else:
        write_json(packet_path,packet)
    attempted = [a for a in read_json(ledger/'budget.json')['attempts'] if a['packet_digest'] == fingerprint(packet)]
    if len(attempted) > 1: raise ValueError('Duplicate retry attempts')
    if attempted:
        call = Path(attempted[0]['output'])
    else:
        dispatched = dispatch_many([packet_path],ledger,current)
        write_json(packet_path.with_name(packet_path.stem+'-dispatch.json'),dispatched)
        call = Path(dispatched['results'][0]['output'])
    # Only a validated answer can restore selection. A second HTTP error or an
    # unknown effect stays stopped under this new attempt; no retry chain.
    reconcile_completed_response(call,ledger,current)
    return session.recover_completed_selection(expected_selection=expected,
        packet_path=call/'owner-packet.json',request_path=call/'decision-1.request.json',
        response_path=call/'decision-1.response.json',reconciliation_path=call/'completed-response-reconciliation.json',
        terminal_retry_path=original/'terminal-retry.json')


def _apply_authority_overrides(session):
    from modeling_system.controller import read_json
    original = session.user_catalog
    def catalog(state, results):
        items = deepcopy(original(state,results))
        path = session.directory/'authority-overrides.json'
        if path.exists():
            record = read_json(path)
            if record['owner'] != session.owner: raise ValueError('Authority override owner changed')
            for item in items:
                row = record['tasks'].get(item['id'])
                if row:
                    if fingerprint(item) != row['original_definition']:
                        raise ValueError('Original task changed; authority override is not applicable')
                    item['reads'].update(row['reads'])
        return items
    session.user_catalog = catalog


def rebind_pending_authority(session, *, authority_keys, evidence):
    """Refresh only explicit authority dependencies; archive the old pending choice."""
    from modeling_system.controller import read_json, write_json
    from modeling_system.episodes import pin_link
    path = session.directory/'controller/controller.json'
    if (path.parent/'controller.lock').exists(): raise ValueError('Controller must be stopped for authority rebind')
    controller = read_json(path); pending = controller.get('selection')
    if pending is None or any(a['status'] in ('running','needs_reconciliation') for a in controller['attempts']):
        raise ValueError('Pending choice without uncertain native effects required')
    expected = fingerprint(pending)
    original_batch = read_json(session.selector.directory/'batches'/(expected+'.json'))
    state = session._context()
    if not authority_keys or not evidence or state['active_operations']: raise ValueError('Explicit authority evidence and idle owner lane required')
    allowed = set(authority_keys)|{'work_queue_scope'}
    # Initialize the private catalog as in normal operation, then validate every
    # existing native/guide/input revision before changing the metadata binding.
    items = session.user_catalog(deepcopy(state),deepcopy(session.record['results']))
    state = session._context()
    for action in pending['actions']:
        for key,value in action['reads'].items():
            if key not in allowed and not key.startswith('operating_') and state['values'].get(key) != value:
                raise ValueError('Non-authority input changed: '+key)
    overrides_path = session.directory/'authority-overrides.json'
    overrides = read_json(overrides_path) if overrides_path.exists() else {'owner':session.owner,'tasks':{}}
    original_catalog = session.user_catalog
    ids = {a['id'] for a in pending['actions']}
    new_rows = {}
    for item in items:
        if item['id'] in ids:
            new_rows[item['id']] = {'original_definition':fingerprint(item),
                'reads':{k:state['values'][k] for k in authority_keys if k in item['reads']}}
    if set(new_rows) != ids: raise ValueError('Pending task definitions unavailable')
    def rebound_catalog(state, results):
        rows = deepcopy(original_catalog(state,results))
        for item in rows:
            if item['id'] in new_rows:
                if fingerprint(item) != new_rows[item['id']]['original_definition']:
                    raise ValueError('Pending task definition changed')
                item['reads'].update(new_rows[item['id']]['reads'])
        return rows
    session.user_catalog = rebound_catalog
    fresh = session.observe()
    feedback_rebinding = session.authority_feedback_rebinding(pending['state'], fresh, authority_keys)
    allowed.add('operating_feedback')
    old_actions = {a['id']:a for a in pending['actions']}
    if {a['id'] for a in fresh['actions']} != ids: raise ValueError('Qualified operation menu changed')
    for action in fresh['actions']:
        old = old_actions[action['id']]
        if ({k:v for k,v in action.items() if k not in ('revision','reads')} !=
                {k:v for k,v in old.items() if k not in ('revision','reads')}
                or set(action['reads']) != set(old['reads'])
                or any(action['reads'][k] != old['reads'][k] for k in old['reads'] if k not in allowed)):
            raise ValueError('Authority refresh changed a non-authority action input')
    plan = deepcopy(pending['plan']);plan['authority_revision']=fresh['authority_revision']
    plan['reads']={k:fresh['values'][k] for k in plan['reads']}
    batch = session.selector(fresh,fresh['actions'],plan,prepare_only=True)
    updated = {'state':fresh,'actions':fresh['actions'],'plan':plan}
    pinned = [pin_link(session.service, {'kind':'file','path':str(p),'role':'User authority correction'}) for p in evidence]
    result = session.service._run_episode_callback(session.episode,'rebind_selection_authority',
        {'original_selection':pending,'updated_selection':updated,'authority_keys':authority_keys,'evidence':pinned},
        lambda:{'status':'completed','original_packet_digest':fingerprint(original_batch['packet']),
            'updated_packet_digest':fingerprint(batch['packet']),'authority_keys':list(authority_keys),
            'feedback_rebinding':feedback_rebinding,
            'evidence':pinned,'provider_calls_added':0,'native_effects':False})
    rebind_path=session.directory/'authority-rebind.json'
    write_json(rebind_path,result)
    # Store the original private definitions plus narrowly replaced authority reads.
    # Use a fresh wrapper next time; never modify the frozen source/catalog files.
    overrides['tasks'].update(new_rows);write_json(overrides_path,overrides)
    session.user_catalog=original_catalog
    current=read_json(path)
    if fingerprint(current.get('selection')) != expected: raise ValueError('Pending selection changed during rebind')
    current['selection']=updated;current['plan']=plan
    current.setdefault('authority_rebindings',[]).append(result);write_json(path,current)
    return result


def recover_transient_selection(session, ledger_directory):
    """Resume an interrupted Jev choice; native execution remains a separate run."""
    from modeling_system.controller import applicable, read_json
    from .planning_batch import resolve_packet
    ledger=Path(ledger_directory);path=session.directory/'controller/controller.json'
    if (path.parent/'controller.lock').exists():raise ValueError('Stop the controller before recovery')
    record=read_json(path);pending=record.get('selection')
    if pending is None or any(a['status'] in ('running','needs_reconciliation') for a in record['attempts']):
        raise ValueError('Pending selection with no uncertain native effects required')
    expected=fingerprint(pending);batch=read_json(session.selector.directory/'batches'/(expected+'.json'))
    def current():
        state=session.observe()
        if (state['active_operations'] or state['owner']!=pending['state']['owner']
                or state['authority_revision']!=pending['state']['authority_revision']
                or state['stage']!=pending['state']['stage'] or state['actions']!=pending['actions']
                or not applicable(pending['plan'],state)
                or session.selector.project(deepcopy(state),deepcopy(state['actions']),deepcopy(pending['plan']))!=batch['public']
                or fingerprint(read_json(path).get('selection'))!=expected):
            raise ValueError('Pending selection or actual dependencies changed')
        return state
    current(); rebinding=None;original_digest=fingerprint(batch['packet'])
    rebind_path=session.directory/'authority-rebind.json'
    if rebind_path.exists():
        row=read_json(rebind_path)
        if row['updated_packet_digest']==original_digest:
            original_digest=row['original_packet_digest']
            rebinding={'packet':batch['packet'],'authority_keys':row['authority_keys'],'evidence':[str(rebind_path)]}
            if 'feedback_rebinding' in row:
                rebinding['feedback_rebinding']=row['feedback_rebinding']
    matches=[a for a in read_json(ledger/'budget.json')['attempts'] if a['packet_digest']==original_digest]
    if len(matches)!=1:raise ValueError('Exactly one original interrupted attempt required')
    call,retry_receipt,_=resolve_packet(batch['packet'],ledger,current,session.directory/'judgments',
        initial_call=Path(matches[0]['output']),rebinding=rebinding)
    return session.recover_completed_selection(expected_selection=expected,
        packet_path=call/'owner-packet.json',request_path=call/'decision-1.request.json',
        response_path=call/'decision-1.response.json',reconciliation_path=call/'completed-response-reconciliation.json',
        transient_retry_path=retry_receipt)
