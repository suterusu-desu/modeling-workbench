"""Exact source-time scheduling, preserving independent captured sample identities."""
from bisect import bisect_right
from fractions import Fraction
import json
from .store import canonical,atomic_write,digest


def plan(service,plan_path,fps=60,normal_cycles=3,slow_cycles=1,slow_speed=.25,lead=.25,rest=.6):
    if not 1<=fps<=120 or not 0<=normal_cycles<=10 or not 0<=slow_cycles<=10 or normal_cycles+slow_cycles==0:
        raise ValueError('Bounded fps and cycle counts required')
    if not .05<=slow_speed<=1 or min(lead,rest)<0:raise ValueError('Invalid playback timing')
    source=service.store.blob(plan_path)
    data=json.loads(service.store.resolve_blob(source).read_text(encoding='utf-8-sig'))
    document=data.get('plan',data);rows=document['frames']
    if len(rows)<2 or any('pose' not in r or not r.get('before_state') or not r.get('after_state') for r in rows):
        raise ValueError('Exact recorded before/after states, ordered poses and source times required')
    times=[Fraction(str(r['pose']['time'])) for r in rows]
    if times[0]!=0 or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Strictly increasing source times starting at zero required')
    cycle=Fraction(str(lead))+times[-1]+Fraction(str(rest));schedule=[];coverage=[]
    for speed,count,label in ((Fraction(1),normal_cycles,'Recorded timing'),(Fraction(str(slow_speed)),slow_cycles,'Quarter speed' if slow_speed==.25 else 'Slow')):
        for _ in range(count):
            cycle_index=len(coverage);seen=set()
            for frame in range(round(cycle/speed*fps)):
                t=Fraction(frame,fps)*speed-Fraction(str(lead))
                index=max(0,min(len(rows)-1,bisect_right(times,t)-1));seen.add(index)
                schedule.append(dict(output_frame=len(schedule),source_index=index,source_time=float(times[index]),
                    action_time=float(t),speed=float(speed),label=label,cycle_index=cycle_index,cycle_frame=frame,
                    phase=rows[index]['pose'].get('label','unlabeled; no inferred phase'),
                    before_state=rows[index]['before_state'],after_state=rows[index]['after_state']))
            coverage.append(dict(cycle_index=cycle_index,speed=float(speed),source_indices=sorted(seen),
                                 missing_indices=sorted(set(range(len(rows)))-seen)))
    path=service.store.root/'derived'/(digest(canonical(schedule))+'-frame-map.json')
    atomic_write(path,canonical(schedule))
    payload=dict(source=source,schedule=service.store.blob(path),frames=len(schedule),duration_seconds=len(schedule)/fps,
                 fps=fps,cycles=coverage,all_samples_each_cycle=all(not c['missing_indices'] for c in coverage),
                 method='Exact rational per-cycle times and ordered sample-and-hold; no modulo or nearest-control substitution',
                 geometry='unchanged; only supplied recorded sample identities',
                 limitations=['Finite playback scheduling, not continuous geometry or naturalness',
                             'Intervals shorter than available output sampling can remain unseen; coverage is reported per cycle',
                             'Existing videos are not rewritten; this operation returns a new exact frame map'])
    return dict(schedule_record=service.store.put('ordered_motion_schedule',payload),**payload)
