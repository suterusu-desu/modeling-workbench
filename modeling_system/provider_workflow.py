"""Persisted provider-workflow composition and aggregate cost preview.

A version-1 bounded acyclic graph binds existing exact reference jobs, current
image reviews and diagnostic recipe outputs at named nodes. References keeps
generation source, settings, review validity and rejection authoritative; the
ledger keeps each job's own lifecycle. This module never dispatches, fetches a
price, spends credits or replaces claim_job/reconcile_job.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
import json
import math
import re
from .episodes import pin_link
from .ledger import Conflict
from .store import canonical, digest

KIND = 'provider_workflow'
NODE_KINDS = ('job', 'review', 'diagnostic')
JOB_KINDS = ('image', 'video', 'mesh')
BUSY = ('dispatching', 'submitted', 'unknown')
TERMINAL = ('failed', 'cancelled')
SATISFIED_STEPS = ('reusable', 'accepted')
INVALID = ('invalidated', 'stale', 'rejected', 'superseded', 'unusable', 'unreadable')
MAX_QUOTE_VALIDITY = timedelta(days=30)
UNIT = re.compile(r'^[A-Za-z][A-Za-z0-9_.:-]{0,63}$')
SCOPE = ('Composition over existing jobs, reviews and recipe outputs. claim_job and reconcile_job remain the only '
         'dispatch and result authorities; References keeps review validity; nothing here spends or prices.')


def _now():
    return datetime.now(timezone.utc)


def _time(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be an ISO 8601 timestamp with an explicit timezone')
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError(name + ' is not an ISO 8601 timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(name + ' requires an explicit timezone offset')
    return parsed.astimezone(timezone.utc)


def _amount(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('Quote amount must be an integer, finite number or decimal string')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Quote amount must be finite; NaN or infinity is not a price')
        value = repr(value)
    elif isinstance(value, int):
        value = str(value)
    if not isinstance(value, str):
        raise ValueError('Quote amount must be an integer, finite number or decimal string')
    if len(value)>128:raise ValueError('Quote decimal text exceeds the supported bound')
    try:
        amount = Decimal(value.strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError('Quote amount is not a decimal number') from error
    if not amount.is_finite():
        raise ValueError('Quote amount must be finite; NaN or infinity is not a price')
    if amount < 0:
        raise ValueError('Negative quote amounts are refused')
    if amount==0:return Decimal(0)
    if amount.adjusted() > 17 or -amount.as_tuple().exponent > 12:
        raise ValueError('Quote amount exceeds the supported precision (18 integer and 12 fractional digits)')
    return amount


def _text(amount):
    # normalize() applies the caller's Decimal context and can silently round.
    value=format(amount,'f')
    return value.rstrip('0').rstrip('.') if '.' in value else value


def _sum(left,right):
    # At most 24 nodes, each <=19 integer and 12 fractional digits. Keep the
    # complete sum independent of a caller's default Decimal precision.
    with localcontext() as context:
        context.prec=64
        return _text(Decimal(left)+Decimal(right))


def validate_graph(graph):
    """Version-1 bounded acyclic node graph; returns nodes by ID, dependencies and a topological order."""
    if (not isinstance(graph, dict) or graph.get('schema_version') != 1
            or not isinstance(graph.get('name'), str) or not graph['name'].strip()):
        raise ValueError('Version-1 named provider graph required')
    if set(graph) - {'schema_version', 'name', 'nodes', 'purpose'}:
        raise ValueError('Unknown provider graph fields')
    if len(canonical(graph)) > 32000:
        raise ValueError('Provider graph must fit within 32000 bytes')
    nodes = graph.get('nodes')
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 24:
        raise ValueError('Provider graph supports 1..24 nodes')
    by_id = {}
    for node in nodes:
        if (not isinstance(node, dict) or not isinstance(node.get('id'), str) or not node['id'].strip()
                or node['id'] in by_id):
            raise ValueError('Distinct nonempty node IDs required')
        if node.get('kind') not in NODE_KINDS:
            raise ValueError('Node kind must be job, review or diagnostic: ' + node['id'])
        if set(node) - {'id', 'kind', 'depends_on', 'source', 'expects', 'purpose'}:
            raise ValueError('Unknown node fields: ' + node['id'])
        if 'purpose' in node and not isinstance(node['purpose'], str):
            raise ValueError('Node purpose must be text: ' + node['id'])
        by_id[node['id']] = node
    dependencies = {}
    for name, node in by_id.items():
        declared = node.get('depends_on', [])
        if not isinstance(declared, list) or any(not isinstance(d, str) or d not in by_id or d == name for d in declared):
            raise ValueError('depends_on must name other declared nodes: ' + name)
        deps = set(declared)
        source = node.get('source')
        expects = node.get('expects', {})
        if node['kind'] == 'diagnostic':
            if 'source' in node or 'expects' in node:
                raise ValueError('Diagnostic nodes bind a recipe step; they take no source or expectations: ' + name)
        elif node['kind'] == 'review':
            if 'expects' in node:
                raise ValueError('Review nodes take no expectations: ' + name)
            if not isinstance(source, str) or by_id.get(source, {}).get('kind') != 'job':
                raise ValueError('Review node requires source naming a job node: ' + name)
            deps.add(source)
        else:
            if (not isinstance(expects, dict) or set(expects) - {'kind', 'provider', 'intent_class'}
                    or any(not isinstance(v, str) or not v.strip() for v in expects.values())):
                raise ValueError('Job expectations may name kind, provider and intent_class as text: ' + name)
            if 'kind' in expects and expects['kind'] not in JOB_KINDS:
                raise ValueError('Expected job kind must be image, video or mesh: ' + name)
            if source is not None:
                if not isinstance(source, str) or source == name or by_id.get(source, {}).get('kind') not in ('job', 'review'):
                    raise ValueError('Job source must name an upstream job or review node: ' + name)
                if expects.get('kind') == 'image':
                    raise ValueError('An image job has no reviewed source: ' + name)
                deps.add(source)
        dependencies[name] = sorted(deps)
    order = []
    visiting = set()

    def visit(name):
        if name in visiting:
            raise ValueError('Cycle in provider graph dependencies')
        if name in order:
            return
        visiting.add(name)
        for parent in dependencies[name]:
            visit(parent)
        visiting.remove(name)
        order.append(name)
    for name in by_id:
        visit(name)
    return by_id, dependencies, order


def _read(service, workflow, expected=None):
    item = service.ledger.read(workflow)
    if item['kind'] != KIND:
        raise ValueError('Expected provider workflow')
    if item['intent']['workspace'] != str(service.workspace):
        raise ValueError('Provider workflow belongs to a different workspace')
    if expected is not None and item['revision'] != expected:
        raise Conflict('Provider workflow changed; inspect its current revision before this update')
    return item


def _job(service, handle):
    try:
        job = service.ledger.read(handle)
    except FileNotFoundError as error:
        raise ValueError('Unknown job handle') from error
    if job['kind'] != 'reference_job':
        raise ValueError('Only an existing reference job can be bound at a job node')
    return job


def _record(service, key, kind):
    try:
        return service.store.get(key, kind)
    except FileNotFoundError as error:
        raise ValueError('Unknown ' + kind + ' record') from error


def _identity(binding):
    if binding is None:
        return None
    if binding['kind'] == 'job':
        return {'job': binding['job']}
    if binding['kind'] == 'review':
        return {'review': binding['review']}
    return {'recipe': binding['recipe'], 'step': binding['step'], 'result': binding['result']}


def _expectation_mismatch(intent, expects):
    reasons = []
    if 'kind' in expects and intent.get('kind') != expects['kind']:
        reasons.append('job kind is ' + str(intent.get('kind')) + ', node expects ' + expects['kind'])
    if 'provider' in expects and str(intent.get('provider', '')).strip().casefold() != expects['provider'].strip().casefold():
        reasons.append('provider differs from expected ' + expects['provider'])
    if 'intent_class' in expects and intent.get('intent_class') != expects['intent_class']:
        reasons.append('intent_class differs from expected ' + expects['intent_class'])
    return reasons


def _lineage(service, node, intent, bindings, nodes):
    """Reasons the bound job does not descend from the current source binding, plus its reviewed source."""
    source = node.get('source')
    reviewed = intent.get('reviewed_source')
    if not source:
        return [], reviewed
    if not reviewed:
        return ['bound job has no reviewed source'], None
    upstream = bindings.get(source)
    if upstream is None:
        return ['source node ' + source + ' is unbound'], reviewed
    if nodes[source]['kind'] == 'review':
        if reviewed != upstream['review']:
            return ['job was prepared from a different review than the bound gate ' + source], reviewed
        root = bindings.get(nodes[source]['source'])
        if root is None or root['job'] != upstream['job']:
            return ['gate ' + source + ' no longer reviews the current source job'], reviewed
    else:
        review = _record(service, reviewed, 'reference_review')
        if review['job'] != upstream['job']:
            return ['reviewed source belongs to a different job than source node ' + source], reviewed
    return [], reviewed


def _review_status(service, review):
    try:
        service.references.require_usable_review(review)
        return 'usable', None
    except (ValueError, KeyError, OSError) as error:
        text = str(error)
        return ('rejected' if 'rejected' in text else 'superseded' if 'superseded' in text else 'unusable'), text


def _planned_next(node, blocked, plan, bindings, nodes):
    if blocked:
        return dict(action='wait', reason='upstream ' + ', '.join(d + ' is ' + plan[d]['status'] for d in blocked))
    if node['kind'] == 'job':
        source = node.get('source')
        if source and nodes[source]['kind'] == 'review':
            return dict(operation='request_reference', arguments=dict(kind=node.get('expects', {}).get('kind', 'video or mesh'),
                        reviewed_source=bindings[source]['review']),
                        reason='Prepare a job from the bound review, then bind it here. Preparation is not submission.')
        if source:
            return dict(operation='review_reference', arguments=dict(job=bindings[source]['job']),
                        reason='Review an output of the upstream job, prepare a job from that usable review, then bind it here.')
        return dict(operation='request_reference', reason='Prepare a job matching this node, then bind it here. Preparation is not submission.')
    if node['kind'] == 'review':
        return dict(operation='review_reference', arguments=dict(job=bindings[node['source']]['job']),
                    reason='Review the upstream output explicitly, then bind the review record here.')
    return dict(operation='inspect_recipe', reason='Run or review the recipe step to a reusable/accepted result, then bind recipe and step here.')


def _job_next(row, blocked):
    status = row['status']
    job = row['job']
    if status in BUSY:
        return dict(operation='reconcile_job', arguments=dict(job=job, expected_revision=row['job_revision']),
                    reason='Job is ' + status + '; attach the actual provider outcome to this same job. No re-claim and no new key.')
    if status in TERMINAL:
        return dict(action='bind a distinct newly prepared job at this node with a reason',
                    reason='Job is ' + status + '; its receipt stays in history. The same key cannot be retried.')
    if status == 'completed':
        kind = row.get('job_kind')
        operation = 'review_reference' if kind == 'image' else 'qualify_guide' if kind == 'mesh' else 'analyze_video'
        return dict(operation=operation, arguments=dict(job=job),
                    reason='Completed; no further dispatch from this node. Review or qualify the actual outputs.')
    if status == 'prepared':
        if row['binding_validity'] != 'current' or row.get('dependency_changes'):
            return dict(action='rebind with a reason, or bind a distinct new job', reason='; '.join(row['reasons']))
        if blocked:
            return dict(action='wait', reason='blocked by ' + ', '.join(blocked))
        if row.get('source_review', {}).get('status') not in (None, 'usable'):
            return dict(action='prepare and bind a new job from a current usable review',
                        reason='source review is ' + row['source_review']['status'])
        return dict(operation='claim_job', arguments=dict(job=job, expected_revision=row['job_revision']),
                    reason='Prepared only. claim_job records dispatch intent after its own policy, review and live preflight checks; this composition grants no authority.')
    return dict(operation='inspect_workflow', arguments=dict(handle=job), reason='Unrecognized job status ' + str(status))


def _job_row(service, row, node, binding, bindings, nodes, blocked, changed):
    job = _job(service, binding['job'])
    intent = job['intent']
    data = job['data']
    row.update(job=binding['job'], job_revision=job['revision'], status=job['status'], job_kind=intent.get('kind'),
               provider=intent.get('provider'), outputs=len(data.get('outputs', [])),
               read_job=dict(operation='inspect_workflow', arguments=dict(handle=binding['job'])))
    if intent.get('intent_class'):
        row['intent_class'] = intent['intent_class']
    if data.get('latest_reviews'):
        row['latest_reviews'] = data['latest_reviews']
    if data.get('rejected_outputs'):
        row['rejected_outputs'] = data['rejected_outputs']
    validity = _expectation_mismatch(intent, node.get('expects', {}))
    lineage, reviewed = _lineage(service, node, intent, bindings, nodes)
    validity += lineage
    review_ok = True
    if reviewed:
        status, reason = _review_status(service, reviewed)
        row['source_review'] = dict(review=reviewed, status=status)
        if reason:
            row['source_review']['reason'] = reason
        review_ok = status == 'usable'
        if not review_ok:
            row['reasons'].append('source review is ' + status)
    row['binding_validity'] = 'invalidated' if validity else 'current'
    row['reasons'] += validity
    if changed:
        row['reasons'].append('dependencies changed since binding: ' + ', '.join(changed) + '; rebind with a reason to acknowledge')
    if data.get('guide_rejection'):
        row['guide_rejection'] = data['guide_rejection']
        row['reasons'].append('reconstructed guide was rejected')
    row['satisfied'] = job['status'] == 'completed' and not blocked and review_ok and not validity and not changed and not data.get('guide_rejection')
    row['ready_for_dispatch'] = job['status'] == 'prepared' and not blocked and not validity and not changed and review_ok
    row['next'].append(_job_next(row, blocked))


def _review_row(service, row, node, binding, bindings, changed):
    row.update(review=binding['review'], job=binding['job'], output_index=binding['output_index'],
               read_review=dict(operation='read_record', arguments=dict(record=binding['review'], path=[])))
    upstream = bindings.get(node['source'])
    if upstream is None or upstream['job'] != binding['job']:
        row['status'] = 'invalidated'
        row['reasons'].append('review belongs to a different job than the current source binding')
    else:
        row['status'], reason = _review_status(service, binding['review'])
        if reason:
            row['reasons'].append(reason)
    if changed:
        row['reasons'].append('dependencies changed since binding: ' + ', '.join(changed) + '; rebind with a reason to acknowledge')
    row['satisfied'] = row['status'] == 'usable' and not changed
    if row['status'] == 'invalidated':
        row['next'].append(dict(action='bind a review of the current source job', reason=row['reasons'][0]))
    elif row['status'] == 'rejected':
        row['next'].append(dict(action='prepare a new upstream proposal with its own lineage',
                                reason='a rejected image stays blocked; an older positive review cannot revive it'))
    elif row['status'] == 'superseded':
        row['next'].append(dict(action='bind the current review of this output', reason='the bound review was superseded'))
    elif not row['satisfied']:
        row['next'].append(dict(action='rebind with a reason' if changed else 'inspect the review and its job',
                                reason='; '.join(row['reasons'])))


def _diagnostic_row(service, row, binding, changed):
    from .recipes import inspect_recipe
    row.update(recipe=binding['recipe'], step=binding['step'], result=binding['result'],
               read_recipe=dict(operation='inspect_recipe', arguments=dict(recipe=binding['recipe'])))
    try:
        recipe = inspect_recipe(service, binding['recipe'])
        step = next((s for s in recipe['steps'] if s['id'] == binding['step']), None)
        if step is None:
            row['status'] = 'invalidated'
            row['reasons'].append('recipe step no longer exists')
        else:
            row['step_status'] = step['status']
            if step['status'] in SATISFIED_STEPS and step.get('result') == binding['result']:
                row['status'] = 'satisfied'
            else:
                row['status'] = 'stale'
                moved = step.get('result') not in (None, binding['result'])
                row['reasons'].append('recipe step is ' + step['status'] + ('; its result changed' if moved else ''))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        row['status'] = 'unreadable'
        row['reasons'].append(str(error))
    if changed:
        row['reasons'].append('dependencies changed since binding: ' + ', '.join(changed) + '; rebind with a reason to acknowledge')
    row['satisfied'] = row['status'] == 'satisfied' and not changed
    if not row['satisfied']:
        row['next'].append(dict(operation='inspect_recipe', arguments=dict(recipe=binding['recipe']),
                                reason='Recover, rerun or re-review the step to a current reusable/accepted result, then rebind with a reason'))


def _plan(service, item):
    nodes, deps, order = validate_graph(item['intent']['graph'])
    bindings = item['data'].get('bindings', {})
    history = item['data'].get('history', [])
    quotes = item['data'].get('quotes', {})
    plan = {}
    for name in order:
        node = nodes[name]
        binding = bindings.get(name)
        row = dict(id=name, kind=node['kind'], dependencies=deps[name], satisfied=False, reasons=[], next=[],
                   history=sum(h['node'] == name for h in history))
        if node.get('purpose'):
            row['purpose'] = node['purpose']
        if node.get('source'):
            row['source'] = node['source']
        if quotes.get(name):
            row['quotes'] = len(quotes[name])
        blocked = [d for d in deps[name] if not plan[d]['satisfied']]
        if blocked:
            row['blocked_by'] = blocked
        if binding is None:
            row['status'] = 'planned'
            row['reasons'].append('nothing bound at this node')
            row['next'].append(_planned_next(node, blocked, plan, bindings, nodes))
        else:
            changed = [d for d in deps[name] if binding.get('dependencies', {}).get(d) != _identity(bindings.get(d))]
            if changed:
                row['dependency_changes'] = changed
            if node['kind'] == 'job':
                _job_row(service, row, node, binding, bindings, nodes, blocked, changed)
            elif node['kind'] == 'review':
                _review_row(service, row, node, binding, bindings, changed)
            else:
                _diagnostic_row(service, row, binding, changed)
        if blocked:
            row['satisfied']=False
            row['reasons'].append('upstream applicability is not satisfied: '+', '.join(blocked))
            if not row['next']:row['next'].append(dict(action='wait',reason='blocked by '+', '.join(blocked)))
        plan[name] = row
    return plan, nodes, deps, order


def _invalidated(row):
    return (row.get('binding_validity') == 'invalidated' or bool(row.get('dependency_changes'))
            or row.get('status') in INVALID)


def _next_steps(rows):
    priority = {'busy': 0, 'branch': 1, 'invalid': 2, 'ready': 3, 'planned': 4, 'other': 5, 'wait': 6}
    steps = []
    for row in rows:
        if not row['next']:
            continue
        status = row.get('status')
        if row.get('blocked_by') and status == 'planned':
            group = 'wait'
        elif status in BUSY:
            group = 'busy'
        elif status in TERMINAL:
            group = 'branch'
        elif _invalidated(row):
            group = 'invalid'
        elif row.get('ready_for_dispatch'):
            group = 'ready'
        elif status == 'planned':
            group = 'planned'
        else:
            group = 'other'
        steps.append((priority[group], len(steps), dict(node=row['id'], **row['next'][0])))
    return [step for _, _, step in sorted(steps, key=lambda s: (s[0], s[1]))]


def inspect(service, workflow):
    """Read-only node statuses, binding validity, dependency invalidation and compact next steps."""
    item = _read(service, workflow)
    plan, nodes, deps, order = _plan(service, item)
    rows = [plan[name] for name in order]
    summary = dict(ready_for_dispatch=[r['id'] for r in rows if r.get('ready_for_dispatch')],
                   busy=[r['id'] for r in rows if r.get('status') in BUSY],
                   needs_new_branch=[r['id'] for r in rows if r.get('status') in TERMINAL],
                   invalidated=[r['id'] for r in rows if _invalidated(r)],
                   blocked=[r['id'] for r in rows if r.get('blocked_by')],
                   planned=[r['id'] for r in rows if r['status'] == 'planned'],
                   satisfied=[r['id'] for r in rows if r['satisfied']])
    return dict(workflow=workflow, revision=item['revision'], episode=item['intent']['episode'],
                graph=dict(name=item['intent']['graph']['name'], revision=item['intent']['graph_revision'], nodes=len(rows)),
                nodes=rows, summary=summary, next_steps=_next_steps(rows), history=len(item['data'].get('history', [])),
                dispatch_authorized=False, live_provider_checked=False, scope=SCOPE,
                detail=dict(operation='read_record', arguments=dict(record=item['revision'], path=[], limit=10, max_chars=8000)))


def create(service, episode, graph, idempotency_key):
    """Instantiate one graph in one workspace/episode; identical retries recover the same instance."""
    validate_graph(graph)
    ep = service.ledger.read(episode)
    if ep['kind'] != 'decision_episode' or ep['status'] != 'active':
        raise ValueError('Provider workflow needs an active existing episode')
    if not isinstance(idempotency_key, str) or not idempotency_key.strip():
        raise ValueError('Stable workflow instance key required')
    graph = json.loads(canonical(graph))
    intent = dict(schema_version=1, graph=graph, graph_revision=digest(canonical(graph)), episode=episode,
                  workspace=str(service.workspace))
    item = service.ledger.create(KIND, intent, digest(canonical([str(service.workspace), episode, idempotency_key])))
    reused = item.get('reused', False)
    if 'bindings' not in item['data']:
        item = service.ledger.update(item['handle'], item['revision'], 'active', dict(bindings={}, history=[], quotes={}))
    result = dict(inspect(service, item['handle']), reused=reused)
    try:
        ep = service.ledger.read(episode)
        context = ep['data'].get('context', ep['intent'].get('context', {}))
        if not any(l.get('kind') == 'workflow' and l.get('id') == item['handle'] for l in context.get('links', [])):
            service.revise_episode(episode, ep['revision'], {'add_links': [dict(kind='workflow', id=item['handle'],
                                                                                role='Provider workflow composition')]})
    except (ValueError, RuntimeError, OSError) as error:
        result['episode_link_pending'] = str(error)
    return result


def _validate_binding(service, node, spec, current, nodes, node_deps, now):
    if not isinstance(spec, dict):
        raise ValueError('Binding must be an object: ' + node['id'])
    dependencies = {d: _identity(current.get(d)) for d in node_deps}
    if node['kind'] == 'job':
        if set(spec) != {'job'} or not isinstance(spec['job'], str):
            raise ValueError('Job node binding is {"job": handle}: ' + node['id'])
        job = _job(service, spec['job'])
        intent = job['intent']
        mismatch = _expectation_mismatch(intent, node.get('expects', {}))
        if mismatch:
            raise ValueError('Bound job does not satisfy node ' + node['id'] + ': ' + '; '.join(mismatch))
        if node.get('source'):
            if node['source'] not in current:
                raise ValueError('Bind source node ' + node['source'] + ' before its dependent job ' + node['id'])
            lineage, reviewed = _lineage(service, node, intent, current, nodes)
            if lineage:
                raise ValueError('Job lineage does not match node ' + node['id'] + ': ' + '; '.join(lineage))
            service.references.require_usable_review(reviewed)
        return dict(kind='job', job=spec['job'], bound_revision=job['revision'], bound_status=job['status'], utc=now,
                    dependencies=dependencies)
    if node['kind'] == 'review':
        if set(spec) != {'review'} or not isinstance(spec['review'], str):
            raise ValueError('Review node binding is {"review": record}: ' + node['id'])
        review = _record(service, spec['review'], 'reference_review')
        upstream = current.get(node['source'])
        if upstream is None:
            raise ValueError('Bind source job node ' + node['source'] + ' before its review gate ' + node['id'])
        if review['job'] != upstream['job']:
            raise ValueError('Review belongs to a different job than source node ' + node['source'])
        service.references.require_usable_review(spec['review'])
        return dict(kind='review', review=spec['review'], job=review['job'], output_index=str(review['output_index']),
                    utc=now, dependencies=dependencies)
    if set(spec) != {'recipe', 'step'} or not all(isinstance(spec[k], str) for k in spec):
        raise ValueError('Diagnostic node binding is {"recipe": handle, "step": id}: ' + node['id'])
    from .recipes import inspect_recipe
    recipe = inspect_recipe(service, spec['recipe'])
    step = next((s for s in recipe['steps'] if s['id'] == spec['step']), None)
    if step is None:
        raise ValueError('Unknown recipe step ' + spec['step'])
    if step['status'] not in SATISFIED_STEPS or not step.get('result'):
        raise ValueError('Recipe step ' + spec['step'] + ' is ' + step['status'] + '; bind only a current reusable or accepted result')
    return dict(kind='diagnostic', recipe=spec['recipe'], step=spec['step'], result=step['result'],
                recipe_revision=recipe['revision'], utc=now, dependencies=dependencies)


def bind(service, workflow, expected_revision, bindings, reason=''):
    """Bind existing jobs, reviews or recipe steps at named nodes in one compare-and-swap write."""
    item = _read(service, workflow, expected_revision)
    plan, nodes, deps, order = _plan(service, item)
    if not isinstance(bindings, dict) or not bindings or set(bindings) - set(nodes):
        raise ValueError('Bind known named nodes only')
    if not isinstance(reason, str):
        raise ValueError('reason must be text')
    current = dict(item['data'].get('bindings', {}))
    history = list(item['data'].get('history', []))
    now = _now().isoformat()
    for name in order:
        if name not in bindings:
            continue
        new = _validate_binding(service, nodes[name], bindings[name], current, nodes, deps[name], now)
        if name in current:
            if not reason.strip():
                raise ValueError('Replacing the binding at ' + name + ' requires an explicit reason; the prior binding stays in history')
            history.append(dict(node=name, previous=current[name], observed_status=plan[name]['status'], reason=reason,
                                utc=now, replaced_by=_identity(new)))
        current[name] = new
    jobs = [b['job'] for b in current.values() if b['kind'] == 'job']
    if len(jobs) != len(set(jobs)):
        raise ValueError('A job can be bound at only one node of this workflow; it must not be charged twice')
    service.ledger.update(workflow, item['revision'], 'active', dict(bindings=current, history=history))
    return inspect(service, workflow)


def _ancestors(deps, node):
    found = set()
    pending = list(deps[node])
    while pending:
        name = pending.pop()
        if name not in found:
            found.add(name)
            pending.extend(deps[name])
    return sorted(found)


def _basis(service, item, plan, deps, node):
    """Exact identities a quote is bound to: job intent, current review state and every upstream binding."""
    job = _job(service, plan[node]['job'])
    intent = job['intent']
    review = None
    if intent.get('reviewed_source'):
        reviewed = _record(service, intent['reviewed_source'], 'reference_review')
        source = _job(service, reviewed['job'])
        index = str(reviewed['output_index'])
        review = dict(review=intent['reviewed_source'], job=reviewed['job'], output_index=index, disposition=reviewed['disposition'],
                      latest=source['data'].get('latest_reviews', {}).get(index),
                      rejected=index in source['data'].get('rejected_outputs', []))
    bindings = item['data'].get('bindings', {})
    return dict(node=node, job=plan[node]['job'], intent=digest(canonical(intent)), review=review,
                dependencies={d: _identity(bindings.get(d)) for d in _ancestors(deps, node)},
                applicability={d:dict(satisfied=plan[d]['satisfied'],binding_validity=plan[d].get('binding_validity'),
                    dependency_changes=plan[d].get('dependency_changes',[]),source_review=plan[d].get('source_review',{}).get('status')) for d in _ancestors(deps,node)},
                graph=item['intent']['graph_revision'])


def _job_evidence(job):
    settings = job['intent'].get('settings') or {}
    preflight = job['data'].get('provider_preflight') or {}
    return dict(displayed_generation_credits_at_preparation=settings.get('displayed_generation_credits'),
                displayed_generation_credits_at_claim=preflight.get('displayed_generation_credits'),
                receipt_recorded=bool(job['data'].get('receipt')),
                meaning='Provider-displayed values retained by References at preparation/claim; evidence only, never a quote or a sum')


def _current_quote(service, item, plan, deps, node, now):
    keys = item['data'].get('quotes', {}).get(node, [])
    if not keys:
        return dict(status='missing', reasons=['no quote evidence recorded for this node'], history=0)
    quote = _record(service, keys[-1], 'provider_quote')
    result = dict(record=keys[-1], amount=quote['amount'], unit=quote['unit'], observed_at=quote['observed_at'],
                  expires_at=quote['expires_at'], source=quote['source'], history=len(keys), reasons=[])
    current = _basis(service, item, plan, deps, node)
    changed = [k for k in current if current[k] != quote['basis'].get(k)]
    expired = _time(quote['expires_at'], 'expires_at') <= now
    if changed:
        result['reasons'].append('quote basis changed: ' + ', '.join(changed) + '; record a new quote for the current binding')
    if expired:
        result['reasons'].append('quote expired at ' + quote['expires_at'])
    result['status'] = 'stale' if changed else 'expired' if expired else 'known'
    return result


def preview(service, workflow):
    """Read-only aggregate cost preview recomputed from current bindings, statuses, reviews and quote freshness."""
    item = _read(service, workflow)
    plan, nodes, deps, order = _plan(service, item)
    now = _now()
    groups = {k: dict(nodes=[], subtotals={}, unquoted=[]) for k in ('remaining', 'busy', 'completed', 'terminal')}
    unknown = []
    rows = []
    for name in order:
        row = plan[name]
        entry = dict(id=name, kind=row['kind'], status=row['status'], satisfied=row['satisfied'], reasons=list(row['reasons']), next=row['next'])
        if row['kind'] != 'job':
            entry['cost_role'] = 'none'
            rows.append(entry)
            continue
        role = ('remaining' if row['status'] in ('planned', 'prepared') else 'busy' if row['status'] in BUSY
                else 'completed' if row['status'] == 'completed' else 'terminal')
        entry['cost_role'] = role
        groups[role]['nodes'].append(name)
        if row.get('job'):
            entry.update(job=row['job'], ready_for_dispatch=row.get('ready_for_dispatch', False),
                         job_evidence=_job_evidence(_job(service, row['job'])))
            quote = _current_quote(service, item, plan, deps, name, now)
        else:
            quote = dict(status='missing', reasons=['unbound node; its future input price is unknown'], history=0)
        entry['quote'] = quote
        if quote['status'] == 'known' and role != 'terminal':
            subtotals = groups[role]['subtotals']
            subtotals[quote['unit']] = _sum(subtotals.get(quote['unit'], '0'),quote['amount'])
        elif role == 'remaining':
            unknown.append(name)
            entry['reasons'].append('remaining spend unknown: ' + '; '.join(quote['reasons']))
        else:
            groups[role]['unquoted'].append(name)
        rows.append(entry)
    aggregate = dict(
        remaining=dict(known_subtotals=groups['remaining']['subtotals'], unknown_nodes=unknown, complete=not unknown,
                       nodes=groups['remaining']['nodes'],
                       meaning='Known current quotes for prepared or planned nodes, per denomination; incomplete while any such node is unknown'),
        busy=dict(nodes=groups['busy']['nodes'], quoted_subtotals=groups['busy']['subtotals'], unquoted_nodes=groups['busy']['unquoted'],
                  meaning='Claimed, submitted or unknown jobs; excluded from remaining spend and never a dispatch suggestion'),
        completed=dict(nodes=groups['completed']['nodes'], quoted_subtotals=groups['completed']['subtotals'],
                       unquoted_nodes=groups['completed']['unquoted'], meaning='Completed jobs; historical, excluded from remaining spend'),
        terminal=dict(nodes=groups['terminal']['nodes'], meaning='Failed or cancelled jobs; excluded from every subtotal and needing a distinct new branch'))
    return dict(workflow=workflow, revision=item['revision'], computed_at=now.isoformat(), nodes=rows, aggregate=aggregate,
                next_steps=_next_steps([plan[name] for name in order]),
                denominations='Distinct unit strings are distinct denominations; nothing is converted or summed across units',
                dispatch_authorized=False, prices_fetched=False, spend_effect='none', scope=SCOPE)


def quote(service, workflow, expected_revision, node, amount, unit, observed_at, expires_at, source, evidence):
    """Retain observed displayed-price evidence for a bound job node; no purchase, reservation or dispatch."""
    item = _read(service, workflow, expected_revision)
    plan, nodes, deps, order = _plan(service, item)
    if node not in nodes or nodes[node]['kind'] != 'job':
        raise ValueError('Quotes attach to job nodes')
    row = plan[node]
    if not row.get('job'):
        raise ValueError('Bind a job before quoting; an unbound future input price stays unknown')
    if row['status'] in TERMINAL:
        raise ValueError('Failed/cancelled job needs a distinct new branch; do not quote it')
    if row['binding_validity'] != 'current' or row.get('dependency_changes'):
        raise ValueError('Binding is invalidated; rebind with a reason before quoting')
    if row.get('blocked_by') or row.get('source_review',{}).get('status') not in (None,'usable'):
        raise ValueError('Current dependencies or source review are not usable; no applicable quote can be recorded')
    value = _amount(amount)
    if not isinstance(unit, str) or not UNIT.match(unit):
        raise ValueError('Unit must be a short denomination label such as USD or tripo:credits')
    now = _now()
    observed = _time(observed_at, 'observed_at')
    expires = _time(expires_at, 'expires_at')
    if observed > now:
        raise ValueError('observed_at is in the future')
    if expires <= observed or expires <= now:
        raise ValueError('expires_at must follow observed_at and the current time')
    if expires - observed > MAX_QUOTE_VALIDITY:
        raise ValueError('Quote validity is bounded to 30 days; observe the price again later')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Name where the price was observed')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('Quote evidence links required')
    pinned = [pin_link(service, link) for link in evidence]
    basis = _basis(service, item, plan, deps, node)
    payload = dict(workflow=workflow, node=node, job=row['job'], amount=_text(value), unit=unit,
                   observed_at=observed.isoformat(), expires_at=expires.isoformat(), source=source, evidence=pinned,
                   basis=basis, basis_digest=digest(canonical(basis)), recorded_at=now.isoformat(),
                   meaning='Observed displayed price bound to the current job intent, review and dependency basis; not a purchase, reservation or dispatch authorization')
    key = service.store.put('provider_quote', payload)
    quotes = {k: list(v) for k, v in item['data'].get('quotes', {}).items()}
    quotes[node] = [*quotes.get(node, []), key]
    service.ledger.update(workflow, item['revision'], 'active', dict(quotes=quotes))
    return dict(preview(service, workflow), quote=key)
