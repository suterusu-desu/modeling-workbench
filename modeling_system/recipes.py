"""Bounded offline recipe composition over the existing episode/service journal."""
import inspect
import json
from pathlib import Path
import uuid
from .episodes import pin_link
from .journal import ACTIVE_CALL
from .leases import LEASE, process_identity
from .ledger import Conflict
from .repair_analysis import _pin
from .runtime import LOADED
from .store import atomic_write, canonical, digest

OUTPUTS = {
    'inspect_target_domain': {'analysis','status','summary','next_actions','native_ready','target_admission','reads'},
    'inspect_control_coverage': {'analysis','status','summary','next_actions','native_ready','controllability','reads'},
    'analyze_repair': {'analysis','disposition','question','numerical_status','diagnostics','diagnostic_count',
        'diagnostic_categories','nominations','prediction','native_ready','user_appearance_acceptance','coverage','reads'},
}


def bundled_template(name):
    if name != 'diagnostic-review': raise ValueError('Unknown bundled recipe; available: diagnostic-review')
    template=json.loads((Path(__file__).parent/'recipe_templates'/(name+'.json')).read_bytes())
    return dict(name=name,content_revision=digest(canonical(template)),template=template)


def _references(value):
    if isinstance(value, dict):
        if '$input' in value or '$step' in value:
            kind = '$input' if '$input' in value else '$step'
            if set(value) - {kind, 'path'} or not isinstance(value[kind], str):
                raise ValueError('Recipe references need one input/step name and optional path')
            path = value.get('path', [])
            if not isinstance(path, list) or any(not isinstance(p, str) and type(p) is not int for p in path):
                raise ValueError('Reference path must be a list of field names or indices')
            yield kind, value[kind], path
        else:
            for v in value.values(): yield from _references(v)
    elif isinstance(value, list):
        for v in value: yield from _references(v)


def _validate(service, template):
    if len(canonical(template)) > 64000 or template.get('schema_version') != 1 or not template.get('name'):
        raise ValueError('Version-1 named recipe must fit within 64000 bytes')
    inputs = template.get('inputs', {})
    if not isinstance(inputs, dict) or not inputs or any(not name or kind not in ('case','json') for name,kind in inputs.items()):
        raise ValueError('Recipe inputs must declare case or json types')
    nodes = template.get('steps', [])
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 24:
        raise ValueError('Recipe supports 1..24 steps')
    by_id = {n['id']: n for n in nodes}
    if len(by_id) != len(nodes) or any(not isinstance(k, str) or not k for k in by_id):
        raise ValueError('Distinct nonempty step IDs required')
    dependencies = {}
    for node in nodes:
        input_dependencies=node.get('input_dependencies',[])
        if not isinstance(input_dependencies,list) or any(not isinstance(n,str) or n not in inputs for n in input_dependencies):
            raise ValueError('Input dependencies must name declared inputs')
        if node.get('kind') not in ('operation','review'):
            raise ValueError('Step must be an operation or review')
        if node['kind'] == 'operation':
            operation = node.get('operation')
            if operation not in OUTPUTS:
                raise ValueError('Recipe operation is not an allowed offline diagnostic')
            inspect.signature(getattr(service, operation)).bind(**node.get('arguments', {}))
            source=node['arguments'].get('case_path')
            if not isinstance(source,dict) or set(source)!={'$input'} or inputs.get(source['$input'])!='case':
                raise ValueError('Analysis case_path must reference a declared pinned case input')
        else:
            evidence=node.get('evidence_from')
            if (not isinstance(node.get('question'),str) or not node['question'].strip()
                    or not isinstance(evidence,list) or not evidence
                    or any(not isinstance(e,dict) or '$step' not in e or set(e)-{'$step','path'} for e in evidence)):
                raise ValueError('Review needs a question and a nonempty list of upstream step references')
        deps = set(node.get('depends_on', []))
        if not deps <= by_id.keys(): raise ValueError('Unknown dependency step')
        for kind, name, path in _references(node):
            if kind == '$input':
                if name not in inputs: raise ValueError('Unknown input reference')
                if inputs[name] == 'case' and path: raise ValueError('Case input resolves to its retained JSON file, not a field path')
            else:
                if name not in by_id: raise ValueError('Unknown step output reference')
                parent = by_id[name]
                allowed = OUTPUTS.get(parent.get('operation'),set()) if parent['kind'] == 'operation' else {'decision','reason','evidence','basis'}
                if path and path[0] not in allowed: raise ValueError('Unknown output field')
                deps.add(name)
        for check in node.get('checks', []):
            if set(check) != {'value','equals'}: raise ValueError('Checks require value and equals')
        dependencies[node['id']] = sorted(deps)
    order=[]; visiting=set()
    def visit(name):
        if name in visiting: raise ValueError('Cycle in recipe dependencies')
        if name in order: return
        visiting.add(name)
        for parent in dependencies[name]: visit(parent)
        visiting.remove(name); order.append(name)
    for name in by_id: visit(name)
    return by_id, dependencies, order


def _bind(service, types, bindings):
    if set(bindings) != set(types): raise ValueError('Supply every declared input, without unknown inputs')
    bound={}
    for name, kind in types.items():
        value=bindings[name]
        if kind == 'json':
            if len(canonical(value)) > 8000000: raise ValueError('JSON input exceeds bounded recipe scope')
            bound[name]=dict(kind=kind, value=json.loads(canonical(value)))
        else:
            if not isinstance(value,dict) or set(value) != {'path','sha256'}:
                raise ValueError('Case input requires exact path and sha256')
            p=Path(value['path']).resolve(); asset=service.store.blob(p)
            if asset['sha256'] != value['sha256']: raise ValueError('Case input changed')
            raw=json.loads(service.store.resolve_blob(asset).read_text(encoding='utf-8-sig'))
            bound[name]=dict(kind=kind,original=asset,pinned=_pin(service,raw,p.parent,{}))
    return bound


def _materialize(service, bound):
    if bound['kind'] == 'json': return bound['value']
    def restore(value):
        if isinstance(value,dict):
            if value.get('asset') and value.get('sha256') and value.get('path'):
                return {**{k:v for k,v in value.items() if k != 'asset'},
                        'path':str(service.store.resolve_blob(value['asset']))}
            return {k:restore(v) for k,v in value.items()}
        if isinstance(value,list): return [restore(v) for v in value]
        return value
    content=canonical(restore(bound['pinned']))
    path=service.store.root/'derived'/'recipe-inputs'/(digest(content)+'.json')
    if path.exists():
        if path.read_bytes() != content: raise ValueError('Materialized recipe input was modified')
    else: atomic_write(path,content)
    return str(path)


def _read(service, recipe, expected=None):
    item=service.ledger.read(recipe)
    if item['kind'] != 'diagnostic_recipe': raise ValueError('Expected diagnostic recipe')
    if item['intent']['workspace'] != str(service.workspace): raise ValueError('Recipe belongs to a different workspace')
    if expected is not None and item['revision'] != expected: raise Conflict('Recipe changed; inspect current revision')
    return item


def _resolve(value, inputs, results):
    if isinstance(value,dict):
        if '$input' in value or '$step' in value:
            v=inputs[value['$input']] if '$input' in value else results[value['$step']]
            for key in value.get('path',[]): v=v[key]
            return v
        return {k:_resolve(v,inputs,results) for k,v in value.items()}
    if isinstance(value,list): return [_resolve(v,inputs,results) for v in value]
    return value


def _plan(service, item, materialize=None):
    template=service.store.get(item['intent']['template'],'recipe_template')
    nodes,deps,order=_validate(service,template)
    bound=item['data']['bindings']
    tokens={n:v['value'] if v['kind']=='json' else 'retained-case:'+digest(canonical(v)) for n,v in bound.items()}
    results={}; plan={}
    for name in order:
        node=nodes[name]; attempts=item['data'].get('attempts',{}).get(name,[])
        unresolved=next((a for a in attempts if a['status']=='reserved'),None)
        blocked=[p for p in deps[name] if plan[p]['status'] not in ('reusable','accepted')]
        row=dict(id=name,kind=node['kind'],operation=node.get('operation'),dependencies=deps[name],status='blocked',blocked_by=blocked)
        if node['kind']=='review':
            row['question']=node['question']
            row['evidence_reads']=[dict(operation='read_record',arguments=dict(record=plan[e['$step']]['result'],
                path=e.get('path',[]),limit=10,max_chars=8000)) for e in node['evidence_from'] if plan[e['$step']].get('result')]
        if unresolved:
            row.update(status='needs_recovery',attempt=unresolved['id'],operation_handle=unresolved.get('handle'))
            handle=unresolved.get('handle')
            returned=handle and (service.store.root/'calls'/handle/'result.json').exists()
            completed=unresolved.get('lease') and (service.store.root/'episode-completions'/item['intent']['episode']/(unresolved['lease']+'.json')).exists()
            if not returned and not completed and process_identity(unresolved['process']['pid'])==unresolved['process']:
                row['status']='running'
        elif not blocked:
            resolved=_resolve(node.get('arguments',node.get('evidence_from')),tokens,results)
            checks=_resolve(node.get('checks',[]),tokens,results)
            key=digest(canonical(dict(template=item['intent']['template'],step=name,args=resolved,checks=checks,
                input_dependencies={n:tokens[n] for n in node.get('input_dependencies',[])},
                dependencies={p:dict(key=plan[p]['key'],result=plan[p].get('result')) for p in deps[name]},runtime=LOADED['revision'],
                signature=str(inspect.signature(getattr(service,node['operation']))) if node['kind']=='operation' else 'review-v1')))
            row['key']=key
            hit=next((a for a in reversed(attempts) if a['key']==key),None)
            row['checks_pass']=all(c['value']==c['equals'] for c in checks)
            if not row['checks_pass']: row.update(status='blocked',reason='Explicit result checks failed')
            elif hit:
                row.update(status=hit['status'],attempt=hit['id'],operation_handle=hit.get('handle'))
                if hit.get('result'):
                    results[name]=service.store.get(hit['result'],'recipe_step_result'); row['result']=hit['result']
            else: row['status']='stale' if attempts else ('awaiting_review' if node['kind']=='review' else 'ready')
            if name == materialize:
                inputs=dict(tokens)
                for kind,input_name,_ in _references(node['arguments']):
                    if kind=='$input': inputs[input_name]=_materialize(service,bound[input_name])
                row['arguments']=_resolve(node['arguments'],inputs,results)
        row['historical_attempts']=len(attempts)
        if row.get('result'):
            row['read_result']=dict(operation='read_record',arguments=dict(record=row['result'],path=[],limit=10,max_chars=8000))
        plan[name]=row
    return plan, nodes


def inspect_recipe(service, recipe):
    item=_read(service,recipe); plan,_=_plan(service,item)
    return dict(recipe=recipe,revision=item['revision'],episode=item['intent']['episode'],steps=list(plan.values()),
        bindings={k:dict(kind=v['kind'],content_revision=digest(canonical(v))) for k,v in item['data']['bindings'].items()},
        live_source_checked=False,
        scope='Offline diagnostic recipe; review acceptance is not native or appearance authorization',
        detail=dict(operation='read_record',arguments=dict(record=item['revision'],path=[],limit=10,max_chars=8000)))


def create(service, episode, template, bindings, idempotency_key):
    _validate(service,template)
    ep=service.ledger.read(episode)
    if ep['kind'] != 'decision_episode' or ep['status'] != 'active': raise ValueError('Recipe needs an active existing episode')
    if not idempotency_key.strip(): raise ValueError('Stable recipe instance key required')
    bound=_bind(service,template['inputs'],bindings); spec=service.store.put('recipe_template',template)
    item=service.ledger.create('diagnostic_recipe',dict(template=spec,episode=episode,workspace=str(service.workspace),initial_bindings=bound),
        digest(canonical([str(service.workspace),episode,idempotency_key])))
    if 'bindings' not in item['data']:
        item=service.ledger.update(item['handle'],item['revision'],'active',dict(bindings=bound,attempts={}))
    result=inspect_recipe(service,item['handle'])
    try:
        ep=service.ledger.read(episode); context=ep['data'].get('context',ep['intent'].get('context',{}))
        if not any(l.get('kind')=='workflow' and l.get('id')==item['handle'] for l in context.get('links',[])):
            service.revise_episode(episode,ep['revision'],{'add_links':[dict(kind='workflow',id=item['handle'],role='Reusable offline diagnostic recipe')]})
    except (ValueError,RuntimeError,OSError) as error: result['episode_link_pending']=str(error)
    return result


def _idle(item):
    if any(a['status']=='reserved' for rows in item['data'].get('attempts',{}).values() for a in rows):
        raise Conflict('Recipe has an in-flight or uncertain operation; recover it before changing inputs or reviews')


def revise_inputs(service, recipe, expected_revision, bindings):
    item=_read(service,recipe,expected_revision);_idle(item)
    template=service.store.get(item['intent']['template'],'recipe_template')
    if not bindings or set(bindings)-template['inputs'].keys(): raise ValueError('Revise known named inputs only')
    updates=_bind(service,{k:template['inputs'][k] for k in bindings},bindings)
    service.ledger.update(recipe,item['revision'],'active',dict(bindings={**item['data']['bindings'],**updates}))
    return inspect_recipe(service,recipe)


def _append(service,item,step,attempt):
    attempts={k:list(v) for k,v in item['data']['attempts'].items()}
    attempts[step]=[*attempts.get(step,[]),attempt]
    return service.ledger.update(item['handle'],item['revision'],'active',dict(attempts=attempts))


def _finish(service,recipe,step,attempt_id,result,status):
    item=_read(service,recipe)
    attempts={k:[dict(a) for a in v] for k,v in item['data']['attempts'].items()}
    attempt=next(a for a in attempts[step] if a['id']==attempt_id)
    if attempt['status'] != 'reserved': raise Conflict('Attempt already finalized')
    attempt.update(status=status,result=service.store.put('recipe_step_result',result))
    service.ledger.update(recipe,item['revision'],'active',dict(attempts=attempts))


def run_step(service,recipe,step,expected_revision):
    if ACTIVE_CALL.get() or LEASE.get(): raise ValueError('Call run_recipe_step directly; it is already an episode execution facade')
    item=_read(service,recipe,expected_revision);_idle(item);plan,nodes=_plan(service,item);row=plan[step]
    if row['status']=='reusable': return inspect_recipe(service,recipe)
    if row['kind'] != 'operation' or row['status'] not in ('ready','stale'):
        raise ValueError('Step is not ready for offline execution')
    execution,_=_plan(service,item,materialize=step)
    args=execution[step]['arguments'];operation=nodes[step]['operation']
    inspect.signature(getattr(service,operation)).bind(**args)
    attempt=dict(id=uuid.uuid4().hex,key=row['key'],status='reserved',process=process_identity(),operation=operation,arguments=args)
    _append(service,item,step,attempt)
    def prepared(lease):
        current=_read(service,recipe)
        attempts={k:[dict(a) for a in v] for k,v in current['data']['attempts'].items()}
        last=attempts[step][-1]
        if last['id']!=attempt['id'] or last['status']!='reserved': raise Conflict('Dispatch reservation changed')
        last.update(handle=lease['operation_handle'],lease=lease['id'])
        service.ledger.update(recipe,current['revision'],'active',dict(attempts=attempts))
    try:
        result=service._run_episode_operation(item['intent']['episode'],operation,args,prepared=prepared)
    except Exception as error:
        latest=_read(service,recipe)['data']['attempts'][step][-1]
        if not latest.get('handle'):
            _finish(service,recipe,step,attempt['id'],dict(status='failed',reason=str(error),
                effect_status='refused before mutation dispatch'),'failed')
        raise
    _finish(service,recipe,step,attempt['id'],result,'failed' if result.get('status') in ('failed','conflicting','needs attention') else 'reusable')
    return inspect_recipe(service,recipe)


def review_step(service,recipe,step,expected_revision,decision,reason,evidence):
    item=_read(service,recipe,expected_revision);_idle(item);plan,_=_plan(service,item);row=plan[step]
    if row['kind']!='review' or row['status'] not in ('awaiting_review','stale','accepted','rejected'):
        raise ValueError('Review is not ready or already decided for these inputs')
    if decision not in ('accepted','rejected') or not reason.strip() or not evidence:
        raise ValueError('Explicit diagnostic review decision, reason and evidence required')
    result=dict(decision=decision,reason=reason,evidence=[pin_link(service,l) for l in evidence],
        basis=dict(input_key=row['key'],upstream_evidence=row['evidence_reads']))
    _append(service,item,step,dict(id=uuid.uuid4().hex,key=row['key'],status=decision,result=service.store.put('recipe_step_result',result)))
    return inspect_recipe(service,recipe)


def recover_step(service,recipe,step,expected_revision):
    item=_read(service,recipe,expected_revision)
    attempt=next((a for a in item['data']['attempts'].get(step,[]) if a['status']=='reserved'),None)
    if not attempt: raise ValueError('No reserved step to recover')
    handle=attempt.get('handle')
    if not handle: return dict(**inspect_recipe(service,recipe),recovery='Reservation has no known operation handle; inspect the owner and episode leases. No replay or automatic clearing.')
    folder=service.store.root/'calls'/handle
    if not (folder/'result.json').exists(): return dict(**inspect_recipe(service,recipe),recovery='Operation result is absent; reconcile the same handle using actual evidence. Never replay uncertain work.')
    intent=json.loads((folder/'intent.json').read_bytes());outcome=json.loads((folder/'result.json').read_bytes())
    if intent['episode']!=item['intent']['episode'] or intent['operation']!=attempt['operation'] or intent['arguments']!=attempt['arguments']:
        raise Conflict('Operation receipt does not match reserved step')
    if outcome['status']!='returned': return dict(**inspect_recipe(service,recipe),recovery='Original operation did not return normally; use its existing reconciliation path and retain the failure. No automatic retry.')
    result=outcome['result']
    _finish(service,recipe,step,attempt['id'],result,'failed' if result.get('status') in ('failed','conflicting','needs attention') else 'reusable')
    return inspect_recipe(service,recipe)
