"""The agent entry point: question, immutable state, observations and decisions."""
from pathlib import Path
from datetime import datetime, timezone
import json
import time
import numpy as np
from .store import Store, canonical, digest
from . import geometry


class Workbench:
    def __init__(self, store):
        self.store = Store(store)

    def import_scene(self, record_path):
        started=time.perf_counter()
        record_path=Path(record_path).resolve()
        record=json.loads(record_path.read_text(encoding='utf-8'))
        if digest(canonical(record['state'])) != record['state_id']:
            raise ValueError('Source scene state fingerprint does not match its content')
        objects=[]
        for item in record['state']['objects']:
            entry={'name':item['name'],'type':item['type'],'source':item.get('source',{})}
            if 'arrays' in item:
                source=Path(item['arrays'])
                with np.load(source, allow_pickle=False) as loaded:
                    arrays={k:loaded[k] for k in loaded.files}
                geometry.validate(arrays)
                ah=digest(canonical({k:digest(v.tobytes()) for k,v in arrays.items()}))
                if ah != item['geometry_hash']:
                    raise ValueError('Source array fingerprint mismatch: '+item['name'])
                co=arrays['co']
                entry.update(geometry_hash=ah,asset=self.store.blob(source),vertices=len(co),triangles=len(arrays['tri']),
                             bounds=[co.min(axis=0).tolist(),co.max(axis=0).tolist()] if len(co) else None,
                             semantics=sorted(k for k in arrays if k.startswith(('POINT__','FACE__')) or k=='triangle_component'),
                             evaluation=item.get('evaluation'),geometry_role=item.get('role'))
            objects.append(entry)
        payload={'source_state_id':record['state_id'],'source_record':self.store.blob(record_path),
                 'coordinate_frame':'world','units':'original Blender coordinate units; no physical scale assumed',
                 'geometry_revision':digest(canonical([{k:o.get(k) for k in ('name','geometry_hash')} for o in objects if 'asset' in o])),
                 'objects':objects,'coverage':record['coverage'],'controls':record['state']['controls'],
                 'guide':record['state']['selected_guide'],'references':record['state']['references'],
                 'source_limits':record.get('limits'),'native_checkpoint_assertion':record.get('native_checkpoint'),
                 'historical_query_supported':True}
        key=self.store.put('state',payload)
        return {'status':'completed','state':key,'geometry_revision':payload['geometry_revision'],
                'objects':len(objects),'mesh_objects':sum('asset' in o for o in objects),
                'seconds':time.perf_counter()-started,'source_fingerprints_verified':True,
                'mode':'immutable snapshot; no current Blender-state claim'}

    def open_question(self, title, state, region, intent, protected=(), references=()):
        if not title.strip() or not region.strip() or not intent.strip():
            raise ValueError('Question, region and intent are required')
        self.store.get(state,'state')
        previous=self.store.current()
        record={'title':title,'state':state,'region':region,'intent':intent,'protected':list(protected),
                'references':list(references),'observations':[],'outcomes':[],
                'previous':previous,'utc':datetime.now(timezone.utc).isoformat()}
        key=self.store.put('question',record)
        self.store.set_current(key,expected=previous)
        return {'status':'completed','question':key,'summary':title}

    def _revise(self, question, field, reference):
        record=self.store.get(question,'question')
        record=dict(record,previous=question)
        record[field]=[*record[field],reference]
        key=self.store.put('question',record)
        self.store.set_current(key,expected=question)
        return key

    def arrays(self, state, object_name):
        state=self.store.get(state,'state')
        ob=next((o for o in state['objects'] if o['name']==object_name),None)
        if ob is None or 'asset' not in ob:
            raise ValueError('Recorded mesh unavailable: '+object_name)
        path=self.store.resolve_blob(ob['asset'])
        with np.load(path,allow_pickle=False) as data:
            return {k:data[k] for k in data.files}

    def query_geometry(self, state, object_name, query, selection=None, **parameters):
        started=time.perf_counter()
        arrays=self.arrays(state,object_name)
        if query=='nearest': result=geometry.nearest_surface(arrays,parameters['point'],selection)
        elif query=='section': result=geometry.plane_sections(arrays,int(parameters['axis']),float(parameters['value']),selection)
        elif query=='bounds': result={'bounds':[arrays['co'].min(axis=0).tolist(),arrays['co'].max(axis=0).tolist()],
                                      'vertices':len(arrays['co']),'triangles':len(arrays['tri'])}
        elif query=='ray': result=geometry.ray_hits(arrays,parameters['origin'],parameters['direction'],selection,parameters.get('maximum_distance'))
        elif query=='material_path': result=geometry.material_path(arrays,parameters['vertex_indices'])
        else: raise ValueError('Unknown query; supported: nearest, section, bounds, ray, material_path')
        receipt={'state':state,'object':object_name,'query':query,'selection':selection,'parameters':parameters,
                 'result':result,'mode':'immutable snapshot','seconds':time.perf_counter()-started}
        key=self.store.put('query',receipt)
        return {'status':result.get('status','completed'),'record':key,'state':state,'mode':'immutable snapshot',
                'summary':{k:v for k,v in result.items() if k not in ('segments','triangle_indices','points','vertex_indices','segment_lengths','hits')},
                'detail_record':str(self.store.root/'records'/(key+'.json'))}

    def compare_geometry(self,before,after,object_name,correspondence='triangles'):
        result=geometry.compare(self.arrays(before,object_name),self.arrays(after,object_name),correspondence)
        payload={'before':before,'after':after,'object':object_name,'result':result}
        key=self.store.put('comparison',payload)
        return {'status':'completed','record':key,**payload}

    def attach_observation(self,question,image_path,metadata,role='actual mesh preview'):
        q=self.store.get(question,'question');state=self.store.get(q['state'],'state')
        declared=metadata.get('scene_state_id')
        if declared is not None and declared!=state['source_state_id']:
            raise ValueError('Image state does not match the question state')
        asset=self.store.blob(image_path)
        if metadata.get('image_sha256') and metadata['image_sha256']!=asset['sha256']:
            raise ValueError('Image differs from the capture receipt')
        calibrated=all(k in metadata for k in ('evaluated_camera_world','view_projection_matrix','resolution','scene_state_id'))
        if calibrated:
            for key in ('evaluated_camera_world','view_projection_matrix'):
                matrix=np.asarray(metadata[key],dtype=float)
                if matrix.shape!=(4,4) or not np.isfinite(matrix).all():raise ValueError('Invalid '+key)
            if len(metadata['resolution'])!=2 or min(metadata['resolution'])<=0:raise ValueError('Invalid resolution')
            if not metadata.get('evaluated_owning_scene'): calibrated=False
        observation={'state':q['state'],'geometry_revision':state['geometry_revision'],'image':asset,
                     'metadata':metadata,'role':role,'projection_status':'evaluated capture recorded' if calibrated else 'unverified; visual reference only',
                     'state_association':'matching recorded capture state' if declared else 'caller association; unverified capture state'}
        key=self.store.put('observation',observation)
        next_question=self._revise(question,'observations',key)
        return {'status':'completed' if calibrated else 'partial','question':next_question,'observation':key,
                'projection_status':observation['projection_status']}

    def project(self,observation,points):
        ob=self.store.get(observation,'observation')
        if ob['projection_status']!='evaluated capture recorded':
            raise ValueError('Projection is unverified; obtain an evaluated owning-scene capture')
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise ValueError('Finite XYZ rows required')
        matrix=np.asarray(ob['metadata']['view_projection_matrix'])
        clip=np.column_stack([points,np.ones(len(points))])@matrix.T
        valid=np.abs(clip[:,3])>1e-12
        ndc=np.full((len(points),3),np.nan);ndc[valid]=clip[valid,:3]/clip[valid,3,None]
        width,height=ob['metadata'].get('native_resolution',ob['metadata']['resolution'])
        crop=ob['metadata'].get('crop',[0,0,width,height])
        if crop is None:crop=[0,0,width,height]
        rows=[]
        for i in range(len(points)):
            pixel=[float((ndc[i,0]+1)*width/2-crop[0]),float((1-ndc[i,1])*height/2-crop[1])] if valid[i] else None
            rows.append({'pixel':pixel,
                         'inside_clip':bool(valid[i] and clip[i,3]>0 and np.all(np.abs(ndc[i])<=1)),
                         'inside_crop':bool(pixel is not None and 0<=pixel[0]<crop[2] and 0<=pixel[1]<crop[3]),
                         'visibility':'not tested; projection alone does not determine occlusion'})
        return {'state':ob['state'],'observation':observation,'points':rows}

    def record_outcome(self,question,character,method,evidence,applicability):
        if not evidence or not applicability.strip():raise ValueError('Evidence and applicability are required')
        allowed={'supported','rejected','unresolved'}
        if character.get('status') not in allowed or method.get('status') not in allowed:raise ValueError('Explicit supported/rejected/unresolved dispositions required')
        payload={'question':question,'character':character,'method':method,'evidence':list(evidence),'applicability':applicability,
                 'user_appearance_acceptance':'not implied','human_ledger':'The bound workspace lesson and decision records remain authoritative'}
        key=self.store.put('outcome',payload)
        return {'status':'completed','outcome':key,'question':self._revise(question,'outcomes',key)}

    def inspect_situation(self,question=None):
        question=question or self.store.current()
        if not question:return {'status':'needs evidence','summary':'No current modeling question is selected','next':['import_scene','open_question']}
        q=self.store.get(question,'question');s=self.store.get(q['state'],'state')
        observations=[self.store.get(o,'observation') for o in q['observations']]
        return {'status':'completed','question':question,'title':q['title'],'intent':q['intent'],'region':q['region'],
                'state':q['state'],'mode':'immutable snapshot; live mutations remain with the Blender owner',
                'guide':s['guide'],'controls':s['controls'],'protected':q['protected'],
                'reference_roles':list(s['references']) if isinstance(s['references'],dict) else [],
                'state_detail':str(self.store.root/'records'/(q['state']+'.json')),
                'capabilities':str(Path(__file__).with_name('capabilities.json')),
                'coverage':{'mesh_objects':sum(o['type']=='MESH' for o in s['objects']),
                            'excluded':[x for x in s['coverage'] if not x.get('included')]},
                'observations':[{'record':key,'role':o['role'],'projection':o['projection_status']} for key,o in zip(q['observations'],observations)],
                'outcomes':q['outcomes'],'next':['query_geometry','attach_observation','compare_geometry','record_outcome'],
                'detail_record':str(self.store.root/'records'/(question+'.json'))}
