"""Decision episodes compose existing immutable state and durable workflows.

The episode owns interpretation and links, not another copy of native state,
provider lifecycle or current human authority.
"""
from pathlib import Path
from datetime import datetime
import json
from .bindings import load_binding, fingerprint_files, compare_authority
from .store import canonical, digest
from .journal import calls, ACTIVE_CALL
from .ledger import Conflict, PreconditionRefusal

CATEGORIES={'obligation','soft_goal','provisional_constraint','unresolved_choice'}
INTERPRETATIONS={'fact','observation','construction_evidence','hypothesis','prediction','artistic_judgment','instruction'}
PENDING={'dispatching','submitted','unknown','applying','retaining','rejecting','needs_reconciliation'}

def recent_operation_key(row):
    try:timestamp=datetime.fromisoformat(row['utc'].replace('Z','+00:00')).timestamp()
    except (KeyError,ValueError):timestamp=float('-inf')
    return (-timestamp,row['handle'])


def link_refusal(message):
    error=PreconditionRefusal(message,'revise_episode.link_validation',
        {'mutation_dispatched':False,'native_or_provider_called':False,'episode_changed':False,
         'local_record_status':'Operation intent and supplied input evidence may already be retained; episode update was refused'})
    error.modeling_recovery="Read the latest episode revision and use add_links/remove_links, or supply complete links from detail='links'/'full'. The episode was not updated; no native inspection or replay is required."
    raise error


def validate_link_shape(link):
    if not isinstance(link,dict):link_refusal('Evidence link must be a typed object')
    if any(k in link for k in ('projection','link_index','source_name','study_fields_available')):
        link_refusal("Summary evidence entries are display projections, not replacement links. Use patch={'add_links':[new typed link]} with expected_revision, or obtain complete links with detail='links'/'full'.")
    if not link.get('role') or link.get('kind') not in ('record','workflow','file'):
        link_refusal('Evidence link requires kind (record/workflow/file) and role')
    field='path' if link['kind']=='file' else 'id'
    if not isinstance(link.get(field),str) or not link[field]:
        link_refusal('Complete '+link['kind']+' evidence link requires '+field+"; use add_links to append without copying a summary")


def link_id(link):
    return digest(canonical(link))


def pin_link(service, link):
    validate_link_shape(link)
    link=dict(link)
    kind=link.get('kind')
    if not link.get('role'):
        raise ValueError('Every evidence link needs its decision role')
    if kind=='record':
        service.store.get(link['id'])
    elif kind=='workflow':
        item=service.ledger.read(link['id'])
        link['pinned_revision']=item['revision']
    elif kind=='file':
        if link.get('asset'):
            service.store.resolve_blob(link['asset'])
            if link.get('sha256')!=link['asset']['sha256']:link_refusal('Pinned evidence hash differs from its immutable asset')
            return link
        path=(service.workspace/link['path']).resolve()
        asset=service.store.blob(path)
        if link.get('sha256') and asset['sha256']!=link['sha256']:
            raise ValueError('Evidence source differs from expected hash')
        link.update(asset=asset,sha256=asset['sha256'],path=str(path))
    else:
        raise ValueError('Evidence link kind must be record, workflow or file')
    return link


def validate_context(service, context):
    if not isinstance(context,dict): raise ValueError('Episode context must be an object')
    allowed={'stage','feature','mechanism','requirements','hypotheses','uncertainties','judgments','links','semantic_graph','next_question','resource_bound','context'}
    if set(context)-allowed: raise ValueError('Unknown episode context fields: '+str(sorted(set(context)-allowed)))
    value=dict(context)
    for req in value.get('requirements',[]):
        if req.get('category') not in CATEGORIES or not all(req.get(k) for k in ('id','text','source','scope')):
            raise ValueError('Requirement needs id, text, category, source and scope; provisional is not an obligation')
    ids=[x['id'] for x in value.get('requirements',[])]
    if len(ids)!=len(set(ids)): raise ValueError('Requirement IDs must be unique')
    for hypothesis in value.get('hypotheses',[]):
        if not all(hypothesis.get(k) for k in ('id','explanation','prediction','discriminating_observation')):
            raise ValueError('Hypothesis needs explanation, prediction and discriminating observation')
    if not isinstance(value.get('links',[]),list):link_refusal('links must be a list of complete typed evidence links')
    for link in value.get('links',[]):validate_link_shape(link)
    value['links']=[pin_link(service,l) for l in value.get('links',[])]
    if value.get('semantic_graph'): service.store.get(value['semantic_graph'],'semantic_graph')
    return value


def open_episode(service, question, owner, scope, context, idempotency_key):
    service.store.get(question,'question')
    if not owner.strip() or not scope.strip(): raise ValueError('Owner and scope required')
    validated=validate_context(service,context)
    binding=load_binding(service.workspace)
    # Stable idempotency uses declared intent, not volatile file timestamps.
    item=service.ledger.create('decision_episode',dict(question=question,owner=owner,scope=scope,
                 character=binding['character'],context=validated),idempotency_key)
    if item['status']=='prepared':
        authority=[pin_link(service,{'kind':'file','path':e['path'],'role':e.get('role','authority'),'authority_id':e.get('id')})
                   for e in binding.get('authority',[]) if (service.workspace/e['path']).is_file()]
        item=service.ledger.update(item['handle'],item['revision'],'active',
                                  dict(context=validated,authority_at_entry=authority,judgment_status='pending'))
    return dict(episode=item['handle'],revision=item['revision'],status=item['status'],reused=item.get('reused',False))


def revise_episode(service,episode,expected_revision,patch):
    item=service.ledger.read(episode)
    if item['kind']!='decision_episode': raise ValueError('Expected decision episode')
    if item['revision']!=expected_revision:raise Conflict('Episode revision changed; read latest revision before patching evidence')
    if not isinstance(patch,dict):link_refusal('Episode patch must be an object')
    patch=dict(patch)
    additions=patch.pop('add_links',[]);removals=patch.pop('remove_links',[])
    if not isinstance(additions,list) or not isinstance(removals,list):link_refusal('add_links and remove_links must be lists')
    if 'links' in patch and (additions or removals):link_refusal('Use either complete links replacement or add_links/remove_links in one revision')
    context=dict(item['data'].get('context',item['intent']['context']))
    existing=context.get('links',[])
    if any(not isinstance(key,str) or key not in {link_id(l) for l in existing} for key in removals):
        link_refusal('remove_links requires link_id values present in the expected episode revision')
    for link in additions:validate_link_shape(link)
    # A full requirement replacement remains in immutable previous revisions.
    context.update(validate_context(service,patch))
    # Omitted links must not drop the existing evidence index.
    if 'links' not in patch:
        context['links']=[l for l in existing if link_id(l) not in removals]+[pin_link(service,l) for l in additions]
    updated=service.ledger.update(episode,expected_revision,'active',{'context':context},allowed={'active','closed'})
    return dict(episode=episode,revision=updated['revision'],status=updated['status'])


def summarize_workflow(service,handle):
    item=service.ledger.read(handle);intent=item['intent'];data=item['data']
    result={k:item[k] for k in ('kind','handle','revision','status')}
    result.update(purpose=intent.get('requested_change',intent.get('hypothesis',intent.get('title'))),
                  recovery='Reconcile original handle before further effects' if item['status'] in PENDING else None)
    if item['kind']=='reference_job':
        result.update(intent_class=intent.get('intent_class'),provider=intent.get('provider'),requested=intent.get('settings'),
                      source=intent.get('generation_input'),review=intent.get('reviewed_source'),
                      provider_id=data.get('provider_id'),outputs=data.get('outputs',[]),
                      actual_receipt=data.get('receipt'),qualification=data.get('qualification'),
                      guide_rejection=data.get('guide_rejection'),reviews=data.get('reviews',[]))
        qualifications=[key for key,value in service.store.records(('guide_qualification',)) if value.get('job')==handle]
        result['qualification_records']=qualifications
        result['qualified_support']=[dict(record=key,**{k:v for k,v in service.store.get(key).items() if k in
            ('native_state','source_state_id','pose','region','role','source_pose','support','limitations','anchor_residuals')}) for key in qualifications]
        if data.get('guide_rejection'):
            result['target_disposition']={'status':'rejected','evidence':service.store.get(data['guide_rejection'])}
        elif qualifications:
            result['target_disposition']={'status':'qualified regions only','records':qualifications}
        else:
            result['target_disposition']={'status':'not qualified','basis':'Provider completion alone is not fitting support'}
        review=intent.get('reviewed_source')
        if review:
            result['source_review']=service.store.get(review,'reference_review')
    elif item['kind']=='diagnostic_recipe':
        try:
            recipe=service.inspect_recipe(handle)
            result.update(purpose='Reusable offline diagnostic recipe',
                steps=[{k:v for k,v in step.items() if k in ('id','kind','status','blocked_by')} for step in recipe['steps']],
                next_read=dict(operation='inspect_recipe',arguments=dict(recipe=handle)))
        except (ValueError,KeyError,RuntimeError,OSError) as error:
            result.update(status='needs inspection',reason=str(error))
    elif item['kind']=='experiment':
        result.update(hypothesis=intent.get('hypothesis'),state=intent.get('state'),
                      checkpoint=data.get('checkpoint'),native_trial=data.get('native_trial'),
                      judgment='Local native retention is separate from appearance and method judgment')
        if data.get('native_trial'):
            trial=data['native_trial'];live=trial.get('live',{})
            result['native_trial']={k:v for k,v in trial.items() if k!='live'}
            result['native_trial']['live_at_return']={k:v for k,v in live.items() if k in
                ('expected_state','geometry_state_id','native_content_hash','file','saved_file','dirty')}
            result['native_trial']['full_native_state']='read_record(workflow revision).data.native_trial.live'
    return result


def coverage(state,full=False):
    objects=state.get('objects',[])
    groups={};roles={}
    for obj in objects:
        label=(obj.get('evaluation') or 'queryable, evaluation unspecified') if obj.get('asset') else 'non-surface dependency'
        if not isinstance(label,str): label=json.dumps(label,sort_keys=True)
        groups.setdefault(label,[]).append(obj['name'])
        roles.setdefault(obj.get('geometry_role') or 'unspecified',[]).append(obj['name'])
    result=dict(inventoried=len(objects),queryable=sum(bool(o.get('asset')) for o in objects),
                classes={k:dict(count=len(v),**({'objects':v} if full else {})) for k,v in groups.items()},
                roles={k:len(v) for k,v in roles.items()},
                excluded=[r for r in state.get('coverage',[]) if not r.get('included')],source_limits=state.get('source_limits'),
                expansion='read_record(state) contains each object, evaluation, hidden/view-layer status, scenes and exact arrays')
    if full:result['declared']=state.get('coverage')
    return result


def evidence_summaries(service,links,episode_revision=None):
    summaries=[];states=[]
    for index,link in enumerate(links):
        if link['kind']=='record':
            data=service.store.get(link['id'])
            if 'source_state_id' in data and 'objects' in data:
                states.append(dict(record=link['id'],role=link['role'],source_state_id=data['source_state_id'],
                                   controls=data.get('controls'),guide=data.get('guide'),coverage=coverage(data)))
        if link['kind']!='file' or not link['path'].lower().endswith('.json') or not link.get('asset'):continue
        if link['asset'].get('bytes',0)>1000000:
            summaries.append(dict(role=link['role'],asset=link['asset'],coverage='Large source; expand exact asset instead of loading arrays into context'));continue
        data=json.loads(service.store.resolve_blob(link['asset']).read_text(encoding='utf-8-sig'))
        if not isinstance(data,dict):continue
        # Preserve every scalar/list-of-scalars finding and explicit nested-field
        # coverage. No invented paraphrase, acceptance or fixed character schema.
        fields={k:v for k,v in data.items() if isinstance(v,(str,int,float,bool,type(None))) or
                (isinstance(v,list) and all(isinstance(x,(str,int,float,bool,type(None))) for x in v))}
        summaries.append(dict(role=link['role'],source=link['path'],sha256=link['sha256'],fields=fields,
                              nested_fields=[k for k in data if k not in fields],expand=link['asset']))
        if episode_revision:
            summary=summaries[-1]
            # Exact source path/hash already appears in context.links[index].
            # Keep all findings, support and exclusions; deduplicate locators only.
            summary.pop('source');summary.pop('sha256')
            summary['context_link_index']=index
            summary['expand']=dict(operation='read_record',arguments={'record':episode_revision},select_field='data.context.links['+str(index)+'].asset')
    return summaries,states


def workspace(service,episode=None,detail='summary',since=None,section=None,path=None,offset=0,limit=20,max_chars=8000,expected_view=None):
    if detail not in ('summary','links','full','section'): raise ValueError('detail must be summary, links, full or section')
    if detail!='section' and (section is not None or path is not None or offset!=0 or limit!=20 or max_chars!=8000 or expected_view is not None):
        raise ValueError('Window options require detail=section')
    if detail=='section' and (since or (not episode and section!='episodes')): raise ValueError('Section reads require episode (except episodes index) and do not accept since')
    binding=load_binding(service.workspace)
    if not episode and detail in ('summary','section'):
        from .bounded_reads import page
        rows=[service.ledger.read(p.stem) for p in sorted(service.ledger.root.glob('*.json'))]
        entries=[dict(episode=x['handle'],revision=x['revision'],status=x['status'],scope=x['intent']['scope'],
                     owner=x['intent']['owner'],expand=dict(operation='decision_workspace',arguments={'episode':x['handle']}))
                 for x in rows if x['kind']=='decision_episode']
        return page(entries,'decision_workspace',{'detail':'section','section':'episodes'},path=path,offset=offset,limit=limit,
                    max_chars=max_chars,expected_view=expected_view,selection_basis='All retained episodes; ascending handle; no native access')
    if not episode:
        episodes=[x for x in service.ledger.list(kind='decision_episode',limit=100)]
        return dict(episodes=[dict(episode=x['handle'],revision=x['revision'],status=x['status'],
                    scope=x['intent']['scope'],owner=x['intent']['owner']) for x in episodes],
                    binding=binding,mode='historical; no native access',next_operations=['open_episode','inspect_situation'])
    item=service.ledger.read(episode)
    if item['kind']!='decision_episode': raise ValueError('Expected decision episode')
    context=item['data'].get('context',item['intent']['context'])
    q=service.store.get(item['intent']['question'],'question');state=service.store.get(q['state'],'state')
    judgment=service.store.get(item['data']['judgment'],'episode_judgment') if item['data'].get('judgment') else None
    links=context.get('links',[])
    workflows=[summarize_workflow(service,l['id']) for l in links if l['kind']=='workflow']
    recorded_calls=sorted([c for c in calls(service.store,episode) if c['handle']!=ACTIVE_CALL.get()],key=recent_operation_key)
    # Operation facts are joined by episode, without a second mutable pointer.
    # Workflow outcomes remain authoritative even when episode closure is pending.
    for call in recorded_calls:
        if call['record']:
            fact=service.store.get(call['record'],'operation_fact')
            result=fact['outcome'].get('result',{})
            handle=result.get('handle') or result.get('job') or result.get('experiment')
            if handle and not any(w['handle']==handle for w in workflows):
                try: workflows.append(summarize_workflow(service,handle))
                except (FileNotFoundError,ValueError): pass
    pending=[w for w in workflows if w['status'] in PENDING]
    next_ops=[]
    if pending: next_ops.append(dict(operation='inspect_workflow',reason='Resolve uncertain job/trial first',cost='local read',handles=[w['handle'] for w in pending]))
    if any(c['retention']!='indexed' for c in recorded_calls):
        if any(c['retention'] not in ('indexed','active/in-flight') for c in recorded_calls):
            next_ops.append(dict(operation='reconcile_operation',reason='Recover non-running operation linkage; actual unknown effects require evidence separately',cost='local store write'))
    for h in context.get('hypotheses',[]):
        next_ops.append(dict(operation='query_geometry',reason=h['discriminating_observation'],cost='recorded geometry computation; no provider credits',
            executable=False,kind='planning recommendation requiring observation design',
            required_arguments=['state','object_name','query','parameters'],
            candidate_baseline=dict(state=q['state'],role='Historical question baseline; does not select current native state',
                                    coverage='read_record(state).objects and coverage identify queryable surfaces and exclusions'),
            native_owner=item['intent']['owner'],
            readiness='Choose matching recorded state/object/selection and numerical query. Appearance or full-motion inspection needs its own owner-controlled capture/review; query_geometry alone cannot perform it.'))
    next_ops.append(dict(operation='operation_context',arguments={'stage':context.get('stage','review')},reason='Read applicable method for the authored stage; inspect effective workflow/judgment dispositions before choosing the next action',cost='read-only local evidence lookup',executable=True))
    if item['data'].get('judgment_status','pending')=='pending':
        next_ops.append(dict(operation='reconcile_episode',reason='Save/recovery is available now; supply only missing causal/artistic judgment when ready',cost='local records'))
    next_ops=[x for x in next_ops if x['operation'] in service.operations()]
    authority=fingerprint_files(service.workspace,binding.get('authority',[]))
    changed,rebound=compare_authority(item['data'].get('authority_at_entry',[]),authority['files'])
    result=dict(episode=episode,revision=item['revision'],status=item['status'],character=item['intent']['character'],
                scope=item['intent']['scope'],owner=item['intent']['owner'],
                current_question=(judgment or {}).get('next_question') or context.get('next_question') or q['title'],
                question=dict(record=item['intent']['question'],title=q['title'],intent=q['intent'],region=q['region'],role='Historical episode entry question'),
                context=context, authority=authority,authority_changed_since_entry=changed,
                authority_location_rebindings=rebound,
                baseline=dict(label='historical question baseline',state=q['state'],source_state_id=state['source_state_id'],controls=state.get('controls'),
                              guide=state.get('guide'),checkpoint=state.get('native_checkpoint_assertion'),coverage=coverage(state)),
                freshness='Historical pinned baseline. Current authority may name a newer save; neither is a live-state check.',
                workflows=workflows,operations=recorded_calls,judgment=item['data'].get('judgment'),
                judgment_status=item['data'].get('judgment_status','pending'),next_operations=next_ops,
                runtime=service.runtime_status()['loaded']['revision'],
                full_episode_record=item['revision'],user_appearance_acceptance='Never inferred')
    from .leases import list_leases
    result['operation_leases']=list_leases(service,episode)
    if result['judgment']:
        result['judgment_summary']={k:judgment[k] for k in ('character','method','applicability','unresolved','next_question')}
        result['judgment_summary']['full_record']=result['judgment']
    if context.get('semantic_graph'):
        result['semantic_graph']=dict(record=context['semantic_graph'],expand='semantic_impact',
                                      validity='Authored/measured relations require separate current dependency and semantic validation')
    if detail=='full': result['full_question']=q;result['full_state']=state;result['full_episode']=item
    if detail=='section':
        from .workspace_summary import section as read_section
        return read_section(result,section,path,offset,limit,max_chars,expected_view)
    raw_result=result
    if detail=='summary':
        from .workspace_summary import summary
        result=summary(service,result,item)
    else:
        result['evidence_summaries'],result['related_states']=evidence_summaries(service,links)
    result['workspace_revision']=digest(canonical(result))
    if since:
        previous=service.store.get(since,'workspace_snapshot')
        result['changed_sections']=[k for k,v in result.items() if k!='workspace_revision' and previous.get(k)!=v]
    if detail=='summary':
        from .workspace_summary import overflow, SUMMARY_CHARS
        from .bounded_reads import json_chars
        if json_chars(result)>SUMMARY_CHARS-128:
            result=overflow(raw_result,result['sections'])
            if since: result['prior_snapshot']=dict(operation='read_record',arguments={'record':since,'path':[]})
            result['workspace_revision']=digest(canonical(result))
    return result


def reconcile_episode(service,episode,expected_revision,character,method,evidence,applicability,
                      unresolved,next_question,close=False,clear_components=()):
    item=service.ledger.read(episode)
    if item['revision']!=expected_revision: raise Conflict('Episode changed; preserve newer decisions and reread')
    if set(clear_components)-{'character','method'}:raise ValueError('Only character/method components can be explicitly cleared')
    prior=service.store.get(item['data']['judgment'],'episode_judgment') if item['data'].get('judgment') else {}
    if character is None and 'character' not in clear_components:character=prior.get('character')
    if method is None and 'method' not in clear_components:method=prior.get('method')
    if not applicability:applicability=prior.get('applicability','')
    if unresolved is None:unresolved=prior.get('unresolved',[])
    if not next_question:next_question=prior.get('next_question','')
    for label,value in [('character',character),('method',method)]:
        if value is not None and (value.get('status') not in ('supported','rejected','unresolved') or not value.get('reason')):
            raise ValueError(label+' needs separate status and reason')
    pinned=[pin_link(service,l) for l in evidence] if evidence else prior.get('evidence',[])
    if (character or method) and (not pinned or not applicability):
        raise ValueError('Judgment requires evidence and applicability')
    pending=character is None or method is None
    if close:
        # Complete data is mandatory here; default summary intentionally bounds
        # completed presentation history and must never decide closure safety.
        view=workspace(service,episode,detail='links')
        if pending or any(w['status'] in PENDING for w in view['workflows']) or any(c['retention']!='indexed' or c['effect_status']=='unknown' for c in view['operations']):
            raise Conflict('Close requires reconciled operations/jobs and explicit judgments, which may be unresolved')
    from .learning import integration_disposition
    method=integration_disposition(service,method,close)
    payload=dict(episode=episode,character=character,method=method,evidence=pinned,applicability=applicability,
                 unresolved=unresolved,next_question=next_question,user_appearance_acceptance='not implied')
    key=service.store.put('episode_judgment',payload)
    from .leases import guard_closure
    updated=service.ledger.guarded_update(episode,expected_revision,'closed' if close else 'active',
                    dict(judgment=key,judgment_status='pending' if pending else 'recorded'),
                    lambda:guard_closure(service,episode) if close else None,allowed={'active','closed'})
    return dict(episode=episode,revision=updated['revision'],judgment=key,judgment_status=updated['data']['judgment_status'],
                status=updated['status'],method_integration=(method or {}).get('integration') or {'disposition':'pending'},
                save_dependency='None; native save/recovery is available even with pending judgment')
