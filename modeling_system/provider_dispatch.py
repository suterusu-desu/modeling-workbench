"""At most two independent selections; one persisted budget, no native dispatch."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

from .typesafe_transport import TypeSafeTransport as Selector, digest
from .credentials import read_key
from modeling_system.judgments import validate_questions

RESERVE_USD = 64000*.042/1000000  # Published combined input-token ceiling; original reservations stay unchanged.


def budget_limits(ledger):
    from modeling_system.provider_recovery import budget_limits as current_limits
    return current_limits(ledger)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    # Same durable retry/retained-temp behavior as controller and queue receipts.
    from modeling_system.controller import write_json
    write_json(path, value)


def initialize_ledger(directory, *, authority, max_requests=None, max_cost_usd=None):
    """Create accounting only with explicit authority; never reset an existing ledger."""
    if not isinstance(authority, str) or not authority.strip():
        raise ValueError('Record the user authorization and scope for provider use')
    policy = {'mode': 'normal_use', 'authority': authority,
              'max_requests': max_requests, 'max_cost_usd': max_cost_usd}
    budget_limits({'policy': policy})
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError('Use the existing provider ledger; never reset accounting')
    record = {'schema_version': 1, 'policy': policy, 'status': 'ready',
              'attempts': [], 'known_cost_usd': 0., 'reserved_usd': 0.}
    # Exclusive creation also protects two simultaneous initializers.
    with (directory/'budget.json').open('x', encoding='utf-8') as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write('\n')
    return record


def dependencies(packet):
    if not {'state','questions','local_binding'} <= set(packet) or set(packet)-{'state','questions','local_binding','dispatch_mapping'}:
        raise ValueError('Unexpected packet schema')
    if not isinstance(packet['state'],dict):
        raise ValueError('Reviewed public state object required')
    from modeling_system.decision_budget import bound_budget
    budget = bound_budget(packet['local_binding'])
    validate_questions(packet['questions'], budget=budget)
    budget.check(packet['state'], packet['questions'])
    deps = packet['local_binding']['dependencies']
    reads, writes = deps['reads'],deps['writes']
    if not isinstance(reads,dict) or not reads or any(not isinstance(k,str) or not isinstance(v,str) or not v for k,v in reads.items()):
        raise ValueError('Explicit dependency key/revision reads required')
    if not isinstance(writes,list) or len(writes)!=len(set(writes)) or not set(writes)<=set(reads):
        raise ValueError('Potential writes need distinct keys with baseline read revisions')
    return reads,set(writes)


def current_check(packet,current,release=False):
    reads,writes = dependencies(packet)
    if any(current['values'].get(k)!=v for k,v in reads.items()):
        raise ValueError('Relevant dependency changed or missing')
    active = current['active_operations']
    if not isinstance(active,list): raise ValueError('Owner operation-lane state required')
    active_writes = set()
    for operation in active:
        if not operation.get('id') or not isinstance(operation.get('writes'),list):
            raise ValueError('Explicit active-operation write coverage required')
        active_writes.update(operation['writes'])
    if active_writes & (set(reads)|writes):
        raise ValueError('Selection depends on a region being changed')
    if release and active:
        raise ValueError('Owner native lane is busy; retain selection without execution')


def independent(packets):
    if not 1<=len(packets)<=2: raise ValueError('One or two packets per bounded wave')
    if len({digest(p) for p in packets})!=len(packets): raise ValueError('Duplicate packet')
    footprints = [dependencies(p) for p in packets]
    if len(footprints)==2:
        (ra,wa),(rb,wb) = footprints
        if wa & (set(rb)|wb) or wb & (set(ra)|wa):
            raise ValueError('Read/write conflict between candidate selections')


def checked_choices(packet,choices):
    if not isinstance(choices,dict) or set(choices)!=set(packet['questions']):
        raise ValueError('Incomplete selected question set')
    import math
    for key, value in choices.items():
        q=packet['questions'][key]
        if q['type']=='choice':
            if not isinstance(value,str) or value not in q['criteria']:
                raise ValueError('Selected option outside original menu')
        else:
            upper=1 if q['type']=='noul' else len(q['criteria'])-1
            if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=upper:
                raise ValueError('Selected typed value outside original range')



def release_selection(path,current_reader,*,require_idle=True):
    """Owner calls immediately before its serial dispatcher; never executes itself."""
    selected=read(path)
    if selected.get('status')!='validated_selection': raise ValueError('Selection not valid')
    packet=read(Path(path).parent/'owner-packet.json')
    if digest(packet)!=selected['packet_digest']: raise ValueError('Packet/menu changed')
    if digest(read(selected['source_packet']))!=selected['packet_digest']:
        raise ValueError('Owner source packet/menu changed after selection')
    checked_choices(packet,selected['choices'])
    current_check(packet,current_reader(),release=require_idle)
    return {'choices':selected['choices'],'local_binding':packet['local_binding'],
            'packet_digest':selected['packet_digest'],'dependencies_revalidated':True,
            'native_dispatch':False,'scope':'Owner-supplied dependency snapshot; actual native dispatch remains owner-only'}


def dispatch_many(packet_paths,trial,current_reader,*,_selector_factory=Selector,_key_reader=read_key):
    """Callbacks provide current owner dependency state; underscored hooks are offline tests only."""
    trial=Path(trial).resolve()
    packets=[read(p) for p in packet_paths]
    independent(packets)
    from modeling_system.provider_recovery import retry_ready
    for packet in packets:
        retry_ready(packet['local_binding'].get('provider_retry', {}).get('not_before'))
    initial=current_reader()
    for packet in packets: current_check(packet,initial)
    trial.mkdir(parents=True,exist_ok=True)
    # Missing local credentials are a known pre-dispatch refusal, not a request
    # requiring uncertain-effect recovery or a consumed reservation.
    key = _key_reader()
    lock=trial/'dispatch.lock'  # Same exclusive lock used by the serial dispatcher.
    with lock.open('x',encoding='utf-8') as f: f.write(digest(packets))
    ledger=None; reserved=[]
    try:
        ledger_path=trial/'budget.json'
        if not ledger_path.is_file(): raise ValueError('Existing authorized ledger required; no automatic trial creation')
        ledger=read(ledger_path)
        max_requests, max_cost = budget_limits(ledger)
        used=ledger.get('carry_in_requests', 0)+sum(a['status']!='verified_no_dispatch' for a in ledger['attempts'])
        if ledger['status']!='ready' or used+len(packets)>max_requests:
            raise ValueError('Completed, uncertain, or exhausted trial')
        if ledger['known_cost_usd']+RESERVE_USD*len(packets)>max_cost:
            raise ValueError('Both requests must reserve remaining shared cost before dispatch')
        for packet in packets:
            if any(a['packet_digest']==digest(packet) and a['status']!='verified_no_dispatch' for a in ledger['attempts']):
                raise ValueError('Previously attempted packet; no repeat')
        # All reservations land durably before auth/network/submitting either worker.
        for i,packet in enumerate(packets):
            number=len(ledger['attempts'])+1
            output=trial/f'call-{number}'
            output.mkdir(exist_ok=False)
            write(output/'owner-packet.json',packet)
            entry={'packet_digest':digest(packet),'status':'reserved','output':str(output),
                   'source_packet':str(Path(packet_paths[i]).resolve()),
                   'reserved_usd':RESERVE_USD,'provider_call_index':used+i+1}
            ledger['attempts'].append(entry)
            reserved.append((entry,packet,output))
        ledger.update(status='in_flight_or_uncertain',reserved_usd=RESERVE_USD*len(packets),
                      provider_slots_reserved_or_attempted=used+len(packets))
        write(ledger_path,ledger)
        # Refuse stale/conflicting work before launching after credential retrieval.
        before=current_reader()
        for entry,packet,_ in reserved:
            current_check(packet,before)
            if digest(read(entry['source_packet']))!=entry['packet_digest']:
                raise ValueError('Owner packet/menu changed before dispatch')
        write(trial/f'wave-{used+1}.json',{'packets':[digest(p) for p in packets],
              'max_in_flight':2,'owner_active_operations_at_start':before['active_operations'],
              'scope':'Selection can overlap an owner-declared disjoint native operation; no native operation is started here'})

        def select(item):
            entry,packet,output=item
            model=_selector_factory(output,write,api_key=key)
            start=time.perf_counter()
            try:
                choices=model.choose(packet['state'],packet['questions'],packet['local_binding'])
                return {'choices':choices,'cost':model.cost,'elapsed_ms':(time.perf_counter()-start)*1000,'error':None}
            except Exception as exc:
                return {'cost':model.cost,'error':type(exc).__name__}
            finally:
                model.close()

        results=[]
        # Only HTTP/choice work runs concurrently. Ledger writes remain in this thread.
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(select,item):item for item in reserved}
            for future in as_completed(futures):
                entry,packet,output=futures[future]
                result=future.result()
                ledger['known_cost_usd']+=result['cost']
                entry['estimated_cost_usd']=result['cost']
                entry['cost_basis']='reported_input_tokens_at_published_rate'
                entry['transport']='typesafe_direct_v1'
                ledger['cost_basis']='historical_provider_cost_plus_direct_token_rate_estimates'
                if result['error']:
                    entry.update(status='failed_or_uncertain',error_type=result['error'])
                else:
                    try:
                        current_check(packet,current_reader())
                        if digest(read(entry['source_packet']))!=entry['packet_digest']:
                            raise ValueError('Owner packet/menu changed during selection')
                        checked_choices(packet,result['choices'])
                        if ledger['known_cost_usd']>max_cost: raise ValueError('Actual budget exceeded')
                        entry['status']='response_validated'
                        selected={'status':'validated_selection','packet_digest':digest(packet),
                                  'source_packet':entry['source_packet'],
                                  'local_binding':packet['local_binding'],'choices':result['choices'],
                                  'native_dispatch':False,'owner_release_revalidation_required':True,
                                  'elapsed_ms':result['elapsed_ms']}
                        write(output/'selection.json',selected)
                    except Exception as exc:
                        entry.update(status='invalidated_after_response',error_type=type(exc).__name__)
                ledger['reserved_usd']-=entry['reserved_usd']
                results.append({'output':str(output),'status':entry['status']})
                write(ledger_path,ledger)
        ok=all(e['status']=='response_validated' for e,_,_ in reserved)
        ledger.update(status='ready' if ok else 'needs_reconciliation',reserved_usd=0.)
        write(ledger_path,ledger)
        report={'status':'returned' if ok else 'stopped','requests_in_wave':len(packets),
                'estimated_total_trial_cost_usd':ledger['known_cost_usd'] if ok else None,
                'known_total_trial_cost_usd':ledger['known_cost_usd'],'results':results,
                'native_calls':0,'concurrent_recipe_writes':0}
        write(trial/f'wave-{used+1}-result.json',report)
        return report
    finally:
        key=''
        # An interrupted wave keeps its persisted reservations and blocks repeats.
        lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-ledger',type=Path)
    parser.add_argument('--authority')
    parser.add_argument('--max-requests',type=int)
    parser.add_argument('--max-cost-usd',type=float)
    parser.add_argument('--packet',type=Path,action='append')
    parser.add_argument('--trial',type=Path)
    parser.add_argument('--dependency-state',type=Path)
    parser.add_argument('--release',type=Path)
    args=parser.parse_args()
    if args.init_ledger:
        initialize_ledger(args.init_ledger, authority=args.authority,
                          max_requests=args.max_requests, max_cost_usd=args.max_cost_usd)
        print(json.dumps({'status':'initialized','provider_calls':0}))
        raise SystemExit(0)
    if not args.dependency_state:
        parser.error('--dependency-state is required for dispatch/release')
    reader=lambda:read(args.dependency_state)
    try:
        result=release_selection(args.release,reader) if args.release else dispatch_many(args.packet or [],args.trial,reader)
        print(json.dumps(result))
        raise SystemExit(0 if result.get('status')!='stopped' else 1)
    except Exception as exc:
        print(json.dumps({'status':'stopped','failure_type':type(exc).__name__,'retry':False}))
        raise SystemExit(1)
