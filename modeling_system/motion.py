"""Finite motion evidence, matched view comparison and deterministic video replay."""
from pathlib import Path
import json
import math
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import imageio_ffmpeg
from .store import canonical, digest, atomic_write


class Motion:
    def __init__(self, workbench):
        self.wb, self.store = workbench, workbench.store

    def import_player(self, manifest_path):
        path=Path(manifest_path).resolve()
        m=json.loads(path.read_text(encoding='utf-8'))
        if not m.get('frames'):
            raise ValueError('Comparison manifest has no frames')
        seen=set();views=set()
        for frame in m['frames']:
            if not math.isfinite(frame['blink']):raise ValueError('Finite pose controls required')
            for side in ('before','after'):
                for view,asset in frame[side].items():
                    if not isinstance(asset,dict) or 'path' not in asset:continue
                    views.add(view)
                    b=self.store.blob(asset['path'])
                    if b['sha256']!=asset['sha256']:raise ValueError('Comparison source changed')
                    asset['stored']=b;seen.add(b['sha256'])
        payload={'source':self.store.blob(path),'manifest':m,'views':sorted(views),'distinct_images':len(seen),
                 'coverage':{'poses':len(m['frames']),'reopening':'reversed stored geometry; no independent reopening capture',
                             'view_matching':'declared by source manifest; not upgraded to calibrated geometry'},
                 'limits':m.get('source_limits',m.get('limits'))}
        key=self.store.put('motion',payload)
        return {'motion':key,'views':payload['views'],'coverage':payload['coverage'],'distinct_images':len(seen)}

    def compare(self, baseline_observations, candidate_observations, phases, object_name=None, reference_mapping=None, correspondence='triangles'):
        if not baseline_observations or len(baseline_observations)!=len(candidate_observations) or len(phases)!=len(candidate_observations):
            raise ValueError('Matched baseline, candidate and phase rows required')
        rows=[]
        for before,after,phase in zip(baseline_observations,candidate_observations,phases):
            b,a=self.store.get(before,'observation'),self.store.get(after,'observation')
            fields=('view_projection_matrix','resolution','crop','renderer','shading','projection_type')
            disagreements=[k for k in fields if b['metadata'].get(k)!=a['metadata'].get(k)]
            calibration=all(b['metadata'].get(k) is not None and a['metadata'].get(k) is not None for k in ('view_projection_matrix','resolution'))
            measured=self.wb.compare_geometry(b['state'],a['state'],object_name,correspondence) if object_name else None
            rows.append(dict(baseline=before,candidate=after,phase=phase,view_disagreements=disagreements,
                             view_match='different' if disagreements else 'calibrated conditions match' if calibration else 'insufficient calibration',geometry=measured))
        payload=dict(rows=rows,reference_mapping=reference_mapping,
                     coverage={'paired_poses':len(rows),'visual_inspection':'not implied by comparison or replay',
                               'continuous_motion':'finite samples only','reference_phase':'caller correspondence; distinct from native controls'},
                     limits='Equal timestamps or eye aperture do not establish mechanical correspondence. Missing measurements remain unresolved.')
        key=self.store.put('motion_comparison',payload)
        return dict(comparison=key,**payload)

    def extract_video(self, video_path, crop=None, maximum_frames=240):
        if not 1<=maximum_frames<=1000:raise ValueError('Frame limit must be one to one thousand')
        asset=self.store.blob(video_path);source=self.store.resolve_blob(asset)
        reader=imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24')
        metadata=next(reader);w,h=metadata['size'];fps=metadata['fps']
        if fps<=0:raise ValueError('Video does not expose a usable nominal frame rate')
        directory=self.store.root/'video-frames'/asset['sha256'];directory.mkdir(parents=True,exist_ok=True)
        frames=[];previous=None;truncated=False
        try:
            for i,raw in enumerate(reader):
                if i>=maximum_frames:truncated=True;break
                im=Image.frombytes('RGB',(w,h),raw)
                if crop:
                    x,y,cw,ch=crop
                    if min(x,y)<0 or min(cw,ch)<=0 or x+cw>w or y+ch>h:raise ValueError('Crop outside video')
                    im=im.crop((x,y,x+cw,y+ch))
                gray=np.asarray(im.convert('L').resize((128,128)),dtype=float)
                delta=float(np.mean(np.abs(gray-previous))) if previous is not None else 0.
                previous=gray
                target=directory/(digest(canonical(crop))+f'-{i:05d}.png')
                if not target.exists():im.save(target)
                frames.append(dict(index=i,time_seconds=i/fps,image=self.store.blob(target),mean_pixel_change=delta))
        finally:
            reader.close()
        # Changes nominate inspection frames, never automatically authorize a guide.
        informative=sorted({0,len(frames)-1,*[f['index'] for f in sorted(frames,key=lambda f:f['mean_pixel_change'],reverse=True)[:8]]})
        payload=dict(video=asset,metadata=metadata,crop=crop,frames=frames,informative_frames=informative,truncated=truncated,
                     timestamp_basis='decoded ordinal divided by nominal fps; variable-rate presentation timestamps not verified',
                     selection_basis='image change nominates inspection; not anatomy, pose truth or automatic guide selection')
        key=self.store.put('video_analysis',payload)
        return {'analysis':key,'frames':len(frames),'informative_frames':informative,'truncated':truncated,
                'timestamp_basis':payload['timestamp_basis'],'detail_record':str(self.store.root/'records'/(key+'.json'))}

    def assess_continuity(self, motion, video_analysis, mapping, action_control='blink', baseline_motion=None, object_name=None, correspondence='triangles'):
        """Link actual native capture order to reviewed video phases, exposing incomplete coverage without approving motion."""
        native=self.store.get(motion,'native_motion');video=self.store.get(video_analysis,'video_analysis')
        baseline=self.store.get(baseline_motion,'native_motion') if baseline_motion else None
        if not mapping:raise ValueError('Explicit native index, reference index, phase and correspondence basis required')
        phase_order={'open':0,'closing':1,'contact':2,'reopening':3,'open_rest':4}
        fields=('view_projection_matrix','resolution','crop','renderer','shading','projection_type')
        seen_receipts=set();view_conditions={};rows=[];missing=[];previous=(-1,-1,-1)
        for entry in mapping:
            ni,vi,phase=entry['native_index'],entry['reference_index'],entry['phase']
            if type(ni)!=int or type(vi)!=int or not 0<=ni<len(native['frames']) or not 0<=vi<len(video['frames']):
                raise ValueError('Mapping index is outside the captured native or decoded video sequence')
            if phase not in phase_order or not str(entry.get('basis','')).strip():raise ValueError('Declared motion phase and reviewed correspondence basis required')
            if ni<=previous[0] or vi<previous[1] or phase_order[phase]<previous[2]:
                raise ValueError('Mapping must follow actual native capture, video and closing/contact/reopening order')
            previous=(ni,vi,phase_order[phase]);frame=native['frames'][ni];reference=video['frames'][vi]
            receipt=frame['native_receipt'];self.store.resolve_blob(receipt)
            if receipt['sha256'] in seen_receipts:raise ValueError('A repeated capture receipt cannot stand in for independently captured reopening')
            seen_receipts.add(receipt['sha256']);self.store.resolve_blob(reference['image'])
            if action_control not in frame['controls']:raise ValueError('Named action control is absent from actual captured controls')
            control=float(frame['controls'][action_control])
            if not math.isfinite(control):raise ValueError('Finite actual control values required')
            observations=frame['observations'];view_rows=[]
            if len(observations)<2:missing.append('close-up and context observations at native index '+str(ni))
            for slot,item in enumerate(observations):
                ob=self.store.get(item['observation'],'observation');self.store.resolve_blob(ob['image'])
                if ob['state']!=frame['state']:raise ValueError('Motion observation and numerical geometry identify different states')
                conditions={k:ob['metadata'].get(k) for k in fields}
                if any(conditions[k] is None for k in ('view_projection_matrix','resolution')):
                    missing.append('calibrated view at native index '+str(ni))
                if slot in view_conditions and conditions!=view_conditions[slot]:raise ValueError('Native viewing conditions changed during the motion')
                view_conditions[slot]=conditions
                view_rows.append({'slot':slot,'observation':item['observation'],'role':ob['role']})
            # A full viewport capture can have the same pixel crop at two different
            # zoom levels. Framing belongs to the projection as well as the crop.
            framing_fields=('view_projection_matrix','crop')
            if len({canonical({k:self.store.get(x['observation'],'observation')['metadata'].get(k)
                               for k in framing_fields}) for x in observations})<2:
                missing.append('distinct close-up and context framing at native index '+str(ni))
            comparison=None;measured=None
            if baseline:
                if ni>=len(baseline['frames']):raise ValueError('Baseline lacks the corresponding capture index')
                before=baseline['frames'][ni]
                if before['pose'].get('controls')!=frame['pose'].get('controls') or before['controls'][action_control]!=frame['controls'][action_control]:
                    raise ValueError('Baseline and candidate were captured at different requested action poses')
                if len(before['observations'])!=len(observations):raise ValueError('Baseline and candidate view coverage differs')
                comparison=self.compare([x['observation'] for x in before['observations']],
                    [x['observation'] for x in observations],[entry]*len(observations),None,[entry],correspondence)
                if any(x['view_match']!='calibrated conditions match' for x in comparison['rows']):
                    raise ValueError('Baseline and candidate viewing conditions do not match')
                if object_name:measured=self.wb.compare_geometry(before['state'],frame['state'],object_name,correspondence)
            rows.append(dict(native_index=ni,reference_index=vi,phase=phase,basis=entry['basis'],actual_control=control,
                reference_time_seconds=reference['time_seconds'],state=frame['state'],views=view_rows,
                comparison=comparison['comparison'] if comparison else None,geometry=measured))
        for phase,direction in (('closing',1),('reopening',-1)):
            selected=[r for r in rows if r['phase']==phase]
            values=[r['actual_control'] for r in selected]
            if len(set(values))<2 or len({r['reference_index'] for r in selected})<2:missing.append('distinct native and video samples during '+phase)
            if any(direction*(b-a)<0 for a,b in zip(values,values[1:])):
                raise ValueError('Actual action controls disagree with the declared motion direction')
        for phase in ('open','contact','open_rest'):
            if not any(r['phase']==phase for r in rows):missing.append(phase+' phase')
        for label,sequence in (('candidate',native),('baseline',baseline)):
            if sequence and not sequence['coverage'].get('complete_requested_capture'):missing.append(label+' complete requested capture')
            if sequence and not all(sequence['native_preservation'].get(k) is True for k in ('keys_unchanged','pose_restored')):
                missing.append(label+' preserved native keys and restored pose')
        result=dict(motion=motion,baseline_motion=baseline_motion,video_analysis=video_analysis,action_control=action_control,
                    rows=rows,coverage_status='partial' if missing else 'complete requested phase evidence',missing=sorted(set(missing)),
                    appearance_acceptance='not established; inspect actual motion, identity and local skin behavior',
                    limits=['Finite samples do not prove every intermediate.',
                            'Phase correspondence is reviewed evidence, not inferred from equal timestamps or aperture.',
                            'Video pixels do not determine unseen depth; native guides, sections and geometry still constrain fitting.',
                            'Repeated poses are independently captured; this alone does not prove dynamic mechanics.'])
        key=self.store.put('motion_continuity',result)
        return dict(continuity=key,**result)

    def encode(self, motion, output_dir, fps=60, normal_cycles=3, slow_cycles=1, slow_speed=.25, panel_size=None):
        item=self.store.get(motion,'motion');m=item['manifest']
        if not 1<=fps<=120 or not 0<=normal_cycles<=10 or not 0<=slow_cycles<=10 or not .05<=slow_speed<=1 or normal_cycles+slow_cycles==0:
            raise ValueError('Replay settings outside bounded supported range')
        root=Path(output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
        settings=dict(motion=motion,fps=fps,normal_cycles=normal_cycles,slow_cycles=slow_cycles,slow_speed=slow_speed,panel_size=panel_size)
        receipt_path=root/'receipt.json'
        if receipt_path.exists():
            old=json.loads(receipt_path.read_text())
            if old['settings']!=settings:raise ValueError('Output directory belongs to another replay; choose a new directory')
            for asset in old['outputs']:self.store.resolve_blob(asset['stored'])
            return dict(old,reused=True)
        timing=m['timing'];close,hold,reopen=(float(timing[k]) for k in ('close','hold','reopen'))
        if min(close,reopen)<=0 or hold<0 or m['cycle_s']<close+hold+reopen:raise ValueError('Invalid action timing')
        schedule=[]
        for speed,cycles,label in ((1.,normal_cycles,'Normal'),(slow_speed,slow_cycles,'Slow')):
            for local in range(round(m['cycle_s']*cycles/speed*fps)):
                t=(local/fps*speed)%m['cycle_s'];smooth=lambda u:u*u*(3-2*u)
                if t<close:b,phase=smooth(t/close),'closing'
                elif t<close+hold:b,phase=1.,'closed hold'
                elif t<close+hold+reopen:b,phase=1-smooth((t-close-hold)/reopen),'reopening'
                else:b,phase=0.,'open rest'
                index=min(range(len(m['frames'])),key=lambda k:abs(m['frames'][k]['blink']-b))
                schedule.append(dict(time_seconds=len(schedule)/fps,source_pose=index,phase=phase,speed=speed,label=label,
                                     requested_control=b,stored_control=m['frames'][index]['blink']))
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();outputs=[]
        for view in item['views']:
            first=m['frames'][0]['before'][view]['stored']
            with Image.open(self.store.resolve_blob(first)) as im:original=im.size
            pw,ph=panel_size or [960,round(960*original[1]/original[0])]
            pw,ph=int(pw)//2*2,int(ph)//2*2
            if min(pw,ph)<32 or max(pw,ph)>2048:raise ValueError('Panel dimensions outside supported range')
            size=(pw*2,ph+80);tiles=[]
            for frame in m['frames']:
                tile=Image.new('RGB',(pw*2,ph),(20,25,32))
                for column,side in enumerate(('before','after')):
                    with Image.open(self.store.resolve_blob(frame[side][view]['stored'])) as im:
                        fitted=ImageOps.contain(im.convert('RGB'),(pw,ph),Image.Resampling.LANCZOS)
                        tile.paste(fitted,(column*pw+(pw-fitted.width)//2,(ph-fitted.height)//2))
                tiles.append(tile)
            target=root/(view+'-comparison.mp4')
            if target.exists():raise ValueError('Unreceipted output exists; inspect partial run and use a new directory')
            command=[ffmpeg,'-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r',str(fps),
                     '-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            try:
                for row in schedule:
                    im=Image.new('RGB',size,(20,25,32));im.paste(tiles[row['source_pose']],(0,36));draw=ImageDraw.Draw(im)
                    draw.text((12,10),m['labels']['before'],fill='white');draw.text((pw+12,10),m['labels']['after'],fill='white')
                    draw.text((12,ph+48),f"{row['label']} | {row['phase']} | stored {row['stored_control']:.0%} | {len(m['frames'])} captured poses",fill='white')
                    process.stdin.write(im.tobytes())
                process.stdin.close();error=process.stderr.read().decode(errors='replace');code=process.wait()
                if code:raise RuntimeError(error)
            finally:
                if process.poll() is None:process.kill();process.wait()
            check=subprocess.run([ffmpeg,'-v','error','-i',str(target),'-f','null','-'],capture_output=True,text=True)
            if check.returncode or check.stderr.strip():raise RuntimeError('Replay decode failed: '+check.stderr)
            outputs.append(dict(view=view,path=str(target),stored=self.store.blob(target),decoded_without_errors=True))
        receipt=dict(settings=settings,outputs=outputs,frames=len(schedule),duration_seconds=len(schedule)/fps,
                     audio=False,coverage=item['coverage'],limits='Nearest captured pose only; no generated in-betweens or continuous perception claim.')
        atomic_write(root/'frame-map.json',canonical(schedule));atomic_write(receipt_path,canonical(receipt))
        key=self.store.put('replay',receipt)
        return dict(receipt,replay=key,reused=False)
