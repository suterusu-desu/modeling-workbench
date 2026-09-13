"""Bounded retained operation views; exact handles do not scan call history."""
import json
from .bounded_reads import page, read_descriptor, validate_window
from .journal import calls, call_row
from .leases import list_leases, status as lease_status


def identifier(value, length, name):
    if not isinstance(value, str) or len(value) != length or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid ' + name)


def unresolved(row):
    return row['retention'] != 'indexed' or row['effect_status'] == 'unknown'


def counts(rows):
    return dict(total=len(rows), active=sum(r['status'] == 'active/in-flight' for r in rows),
                unresolved=sum(unresolved(r) for r in rows), unknown=sum(r['effect_status'] == 'unknown' for r in rows))


def header(service, row, intent=None):
    value={k:v for k,v in row.items() if k not in ('intent_path','result_path','effects')}
    intent=intent if intent is not None else json.loads((service.store.root/'calls'/row['handle']/'intent.json').read_bytes())
    value['effects']=[{k:e[k] for k in ('operation','status')} for e in row['effects']]
    if unresolved(row):
        value['owner']=intent.get('arguments',{}).get('owner') or {'process':intent.get('process')}
        value['recovery']=intent.get('recovery')
        value['next_action']=('Wait for the actual owner; do not reconcile in-flight work' if row['status']=='active/in-flight' else
            'Inspect original effect receipts and reconcile_operation with actual outcome evidence; do not replay' if row['effect_status']=='unknown' else
            'reconcile_operation(handle) repairs retained result/index; do not repeat the operation')
    value['expand']=read_descriptor('inspect_operations',{'handle':row['handle'],'view':'header'})
    return value


def read(service, episode, handle, view, selection, path, offset, limit, max_chars, expected_view):
    if episode is not None: identifier(episode,64,'episode')
    if selection not in ('all','unresolved','active'): raise ValueError('selection must be all, unresolved or active')
    if view=='full' and handle is None:
        if path is not None or offset!=0 or limit!=20 or max_chars!=8000 or expected_view is not None or selection!='all':
            raise ValueError('Full history does not accept window options; use view=index')
        return {'calls':calls(service.store,episode)}
    validate_window([] if path is None else path,offset,limit,max_chars)
    args={'view':view}
    if episode is not None: args['episode']=episode
    if handle is not None:
        identifier(handle,32,'operation or lease handle');args['handle']=handle
        if selection!='all': raise ValueError('Selection applies only to indexes')
        if view=='lease':
            if episode is None: raise ValueError('Exact lease read requires episode')
            p=service.store.root/'episode-leases'/episode/(handle+'.json')
            raw=p.read_bytes();value=json.loads(raw)
            from .store import digest
            value['lease_revision']=digest(raw)
            if value['id']!=handle or value['episode']!=episode: raise ValueError('Lease identity mismatch')
            value['observed_status']=lease_status(value,service.store.root)
        else:
            if view not in ('header','intent','result','effects','resolution'): raise ValueError('Exact operation view must be header, intent, result, effects or resolution')
            folder=service.store.root/'calls'/handle
            intent=json.loads((folder/'intent.json').read_bytes())
            if intent['handle']!=handle: raise ValueError('Operation identity mismatch')
            if episode is not None and intent.get('episode')!=episode: raise ValueError('Operation belongs to another episode')
            row=call_row(service.store,folder/'intent.json',intent)
            if view=='header':
                value=header(service,row,intent)
                value['sections']={name:read_descriptor('inspect_operations',{'handle':handle,'view':name})
                                   for name in ('intent','result','effects','resolution')}
                value['result_available']=(folder/'result.json').is_file()
            elif view=='effects':
                value=[]
                for p in sorted((folder/'effects').glob('*/intent.json')):
                    identifier(p.parent.name,32,'effect handle')
                    rp=p.parent/'result.json'
                    value.append({'handle':p.parent.name,'intent':json.loads(p.read_bytes()),
                                  'result':json.loads(rp.read_bytes()) if rp.is_file() else None,
                                  'result_available':rp.is_file()})
            elif view=='intent': value=intent
            else:
                p=folder/(view+'.json')
                value=json.loads(p.read_bytes()) if p.is_file() else {'available':False,'section':view,
                    'operation_status':row['status'],'effect_status':row['effect_status'],
                    'recovery':'Inspect the original owner and effect receipts; absence is not permission to replay'}
    elif view in ('index','leases'):
        args['selection']=selection
        if view=='index':
            rows=sorted(calls(service.store,episode),key=lambda r:r['handle']); totals=counts(rows)
            selected=[r for r in rows if selection=='all' or (unresolved(r) if selection=='unresolved' else r['status']=='active/in-flight')]
            value=[header(service,r) for r in selected]
        else:
            if episode is None: raise ValueError('Lease index requires episode')
            rows=sorted(list_leases(service,episode),key=lambda r:r['id'])
            totals={'total':len(rows),'active':sum(r['observed_status']=='active' for r in rows),
                    'unresolved':sum(r['observed_status']!='finished' for r in rows)}
            value=[{**{k:v for k,v in r.items() if k!='path'},'expand':read_descriptor('inspect_operations',
                {'episode':episode,'handle':r['id'],'view':'lease'})} for r in rows
                if selection=='all' or (r['observed_status']!='finished' if selection=='unresolved' else r['observed_status']=='active')]
        # Bind the full observed index, including rows outside the selected status.
        from .store import digest, canonical
        fingerprint=digest(canonical(rows))
        return page(value,'inspect_operations',args,path=path,offset=offset,limit=limit,max_chars=max_chars,
                    expected_view=expected_view,view_revision=fingerprint,counts=totals,
                    selection_basis='Observed retained index; ascending handle/id; status filter='+selection)
    else: raise ValueError('Use a handle with an exact view, or view=index/leases')
    return page(value,'inspect_operations',args,path=path,offset=offset,limit=limit,max_chars=max_chars,
                expected_view=expected_view,selection_basis='Exact retained handle and fixed section; observation fingerprint, no native contact')
