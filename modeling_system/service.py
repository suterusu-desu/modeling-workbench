"""One domain service behind Python, CLI and MCP transports."""
from pathlib import Path
import inspect
import json
import time
from .workbench import Workbench
from .ledger import Ledger, Conflict, PreconditionRefusal
from .references import References
from .motion import Motion
from .experience import retrieve
from .native_bridge import NativeBridge, NativeBridgeError, operation_schemas
from .store import canonical, digest
from .bindings import load_binding, authority_path
from .runtime import runtime_report
from .journal import recorded, invoke, EPISODE, native_effect
from .leases import LEASE, acquire, finish, mark_return


class UnconfiguredNative:
    def call(self,*args,**kwargs):
        raise ValueError('Native adapter is unconfigured for this character/workspace. Bind and verify the native owner before native operations; recorded analysis remains available.')

READ_OPERATIONS={'capabilities','runtime_status','inspect_situation','inspect_workflow','read_record',
                 'retrieve_experience','decision_workspace','semantic_impact','inspect_operations','check_reuse',
                 'package_readiness','evidence_manifest','operation_context','select_generation_route'}


class ModelingService:
    def __init__(self, workspace=None, store=None, native=None):
        self.workspace=Path(workspace or Path.cwd()).resolve()
        self.binding=load_binding(self.workspace)
        self.wb=Workbench(store or self.workspace/self.binding.get('store','.modeling/state'))
        self.store=self.wb.store;self.ledger=Ledger(self.store)
        self.policy_path=authority_path(self.workspace,self.binding,'provider policy','MESH-GENERATION.json')
        self.references=References(self.wb,self.ledger,self.policy_path);self.motion=Motion(self.wb)
        self.native=native or (NativeBridge(self.workspace) if self.binding.get('native_adapter')=='blender_json_v1' else UnconfiguredNative())

    def execute(self, operation, arguments):
        started=time.perf_counter()
        try:
            if operation.startswith('native_'):
                args=dict(arguments);owner=args.pop('owner',None)
                result=invoke(self,operation,arguments,lambda:self._native_call(operation[7:],args,owner=owner))
            else:
                if operation not in self.operations():raise ValueError('Unknown domain operation')
                result=getattr(self,operation)(**arguments)
            result=dict(result) if isinstance(result,dict) else {'result':result}
            result.setdefault('status','completed')
        except NativeBridgeError as error:
            result={'status':'conflicting' if error.code=='stale_state' else 'needs attention','summary':error.message,
                    'error':error.as_dict(),'recovery':error.recovery}
            if hasattr(error,'modeling_operation_handle'):
                result.update(operation_handle=error.modeling_operation_handle,operation_result_path=error.modeling_operation_result_path)
        except (ValueError,KeyError,IndexError,TypeError,OSError,RuntimeError) as error:
            result={'status':'conflicting' if isinstance(error,Conflict) else 'failed','summary':str(error),
                    'recovery':'Inspect the referenced state or workflow record, correct the precondition, and resume; no automatic retry.',
                    'error_type':type(error).__name__}
            if hasattr(error,'modeling_operation_handle'):
                result.update(operation_handle=error.modeling_operation_handle,operation_result_path=error.modeling_operation_result_path)
            if isinstance(error,PreconditionRefusal):
                result.update(effect_status=error.modeling_effect_status,stage=error.modeling_stage,details=error.modeling_details,
                              recovery=getattr(error,'modeling_recovery','Proposal was not dispatched. Inspect the recorded baseline/live mismatch, capture and explicitly rebase under the sole owner; preserve the original proposal and do not blindly retry.'))
        result['operation']=operation;result['elapsed_seconds']=time.perf_counter()-started
        if operation in READ_OPERATIONS:
            return result
        try:
            receipt=self.store.put('operation',dict(result,arguments=arguments))
        except (OSError,RuntimeError,ValueError) as error:
            return dict(result,retention_status='needs reconciliation',retention_error=str(error),
                        recovery='Operation may have returned successfully. Recover operation_handle/result path; do not repeat effects because receipt indexing failed.')
        # Heavy state/camera/sequence detail is stored once. No truncation without a detail link.
        if operation not in ('read_record','capabilities') and len(canonical(result))>14000:
            compact={k:v for k,v in result.items() if not isinstance(v,(dict,list)) or k in ('error','expected_state','coverage','influence','constraints','cycles')}
            compact['expanded_fields']={k:'read_record(operation_record).'+k for k in result if k not in compact}
            compact['detail_available']=True;result=compact
        result.update(operation_record=receipt,detail_record=str(self.store.root/'records'/(receipt+'.json')))
        return result

    def _native_call(self, operation, arguments, owner=None):
        return native_effect(self,operation,dict(arguments,owner=owner),lambda:self.native.call(operation,arguments,**({'owner':owner} if owner is not None else {})))

    @classmethod
    def operations(cls):
        return [n for n,fn in inspect.getmembers(cls,inspect.isfunction) if not n.startswith('_') and n not in ('execute','operations')]

    def capabilities(self) -> dict:
        """Discover installed operations, real native schemas, capability parity and transport boundaries."""
        return {'inventory':json.loads(Path(__file__).with_name('capabilities.json').read_text(encoding='utf-8')),
                'operations':self.operations(),'native_operations':operation_schemas(),
                'native_adapter':self.binding.get('native_adapter','unconfigured'),
                'generation_transport':'Prepared linked requests use installed image/provider tools; preparation never submits or spends credits.'}

    def runtime_status(self) -> dict:
        """Report observed loaded source/schema and on-disk mismatch; never infer operator adoption or native freshness."""
        return runtime_report(self)

    def operation_context(self, stage: str, context: dict | None = None, job: str | None = None) -> dict:
        """Read current policy, applicable procedures and exact passages without any store, provider or native writes."""
        from .decision_context import operation_context
        return operation_context(self, stage, context, job)

    def capture_operation_context(self, stage: str, context: dict | None = None, job: str | None = None) -> dict:
        """Explicitly retain a decision's exact current policy/procedure evidence with automatic operation facts; no provider/native call."""
        from .decision_context import operation_context
        return operation_context(self, stage, context, job, retain=True)

    def select_generation_route(self, use_case: str = 'general') -> dict:
        """Read user model preference for a use case; this never grants dispatch authorization or recycles a job."""
        path=self.policy_path
        if not path.is_file():
            return {'status':'needs evidence','reason':'No bound mesh-generation policy','dispatch_authorized':False}
        policy=json.loads(path.read_text(encoding='utf-8-sig'))['tripo']
        preference=policy.get('selection_preferences',{}).get(use_case)
        return dict(use_case=use_case, mode=(preference or policy)['mode'],model=(preference or policy)['model'],
                    basis=preference or {'source':policy.get('decision_source'),'scope':policy.get('scope')},
                    policy=dict(path=str(path),sha256=digest(path.read_bytes()),access='current bound file; lookup does not copy or retain it'),dispatch_authorized=False,
                    next='Use completed evidence when suitable. Any new HD dispatch requires a distinct explicit scoped policy authorization; a use_case label cannot bypass claim guards.')

    def inspect_situation(self, question: str | None = None, live: bool = False) -> dict:
        """Read a question record ID, or focus current context with ordinary question text; text never changes the selected question. live=False never contacts Blender."""
        focus=None
        if question is not None and not (len(question)==64 and all(c in '0123456789abcdef' for c in question)):
            if not question.strip():raise ValueError('Supply a question record ID or nonempty focus text')
            focus=question;question=None
        selection=None
        if question is None and self.store.current() is None and self.binding.get('selected_episode'):
            episode=self.ledger.read(self.binding['selected_episode'])
            if episode['kind']!='decision_episode':raise ValueError('Bound selected episode is not a decision episode')
            question=episode['intent']['question']
            selection=dict(episode=episode['handle'],revision=episode['revision'],meaning='Read-only selected package episode; historical question, not live state or a production selector update')
        result=self.wb.inspect_situation(question)
        if focus is not None:
            result['requested_focus']=focus
            result['experience']=self.retrieve_experience(focus)
        if selection:result['selection_basis']=selection
        result['authority']=[str(self.workspace/e['path']) for e in self.binding.get('authority',[])]
        if self.policy_path.exists():
            result['mesh_generation_settings']=json.loads(self.policy_path.read_text(encoding='utf-8-sig'))
        if live:result['live']=self._native_call('inspect_live',{})
        from .episodes import summarize_workflow,coverage
        result['active_workflows']=[summarize_workflow(self,x['handle']) for x in self.ledger.list(limit=6)]
        if result.get('state'): result['coverage']=coverage(self.store.get(result['state'],'state'))
        result['next']=[n for n in ['decision_workspace','record_observation','query_geometry','operation_context'] if n in self.operations()]
        return result

    def open_episode(self, question: str, owner: str, scope: str, context: dict, idempotency_key: str) -> dict:
        """Open a recoverable decision over existing question/evidence. Requirements distinguish obligations, soft goals, provisional constraints and choices."""
        from .episodes import open_episode
        return open_episode(self,question,owner,scope,context,idempotency_key)

    def revise_episode(self, episode: str, expected_revision: str, patch: dict) -> dict:
        """Revise with compare-and-swap. patch add_links appends typed links; remove_links removes exact link_id values. links requires a complete replacement, never summary rows."""
        from .episodes import revise_episode
        return revise_episode(self,episode,expected_revision,patch)

    def decision_workspace(self, episode: str | None = None, detail: str = 'summary', since: str | None = None,
                           section: str | None = None, path: list[str | int] | None = None, offset: int = 0,
                           limit: int = 20, max_chars: int = 8000, expected_view: str | None = None) -> dict:
        """Read compact progressively expandable episode, actual job dispositions, coverage and pending judgments; no native contact or receipt write."""
        from .episodes import workspace
        return workspace(self,episode,detail,since,section,path,offset,limit,max_chars,expected_view)

    def snapshot_workspace(self, episode: str) -> dict:
        """Pin the current composed workspace for later section-diff resumption; never claims live state."""
        result=self.decision_workspace(episode)
        return {'snapshot':self.store.put('workspace_snapshot',result),'episode':episode}

    def run_episode_operation(self, episode: str, operation: str, arguments: dict) -> dict:
        """Run an existing operation with automatic episode linkage; exact inputs/result survive failure of later judgment/indexing."""
        if operation in ('run_episode_operation','execute'): raise ValueError('Recursive episode execution is not allowed')
        # Bind before acquiring a lease: a typo must not manufacture an unknown
        # native effect with no callable operation to reconcile.
        try:
            if operation.startswith('native_'):
                from .native_bridge import validate_arguments
                args=dict(arguments);owner=args.pop('owner',None)
                validate_arguments(operation[7:],args)
                if operation!='native_inspect_live' and (not isinstance(owner,str) or not owner.strip()):
                    raise ValueError('Native changes require a stable owner')
            else:
                if operation not in self.operations():raise ValueError('Unknown domain operation')
                inspect.signature(getattr(self,operation)).bind(**arguments)
            canonical(arguments)
        except (TypeError,ValueError,NativeBridgeError) as error:
            raise PreconditionRefusal(str(error),'run_episode_operation.argument_binding',
                dict(mutation_dispatched=False,lease_acquired=False,operation=operation)) from error
        lease=acquire(self,episode,operation)
        token=EPISODE.set(episode);lease_token=LEASE.set(lease)
        result={'status':'failed','reason':'Invocation did not return; effect requires reconciliation'}
        try:
            result=self.execute(operation,arguments)
            return result
        finally:
            try:
                mark_return(self,lease,result)
                finish(self,lease,result)
            except (OSError,RuntimeError,ValueError) as error:
                result.update(lease_retention_status='needs finalization repair',lease_id=lease['id'],lease_retention_error=str(error),
                              recovery='Preserve the returned operation result. Repair lease/index metadata via reconcile_operation; do not repeat the native/provider effect.')
            finally:
                LEASE.reset(lease_token);EPISODE.reset(token)

    def inspect_operations(self, episode: str | None = None, handle: str | None = None, view: str = 'full',
                           selection: str = 'all', path: list[str | int] | None = None, offset: int = 0,
                           limit: int = 20, max_chars: int = 8000, expected_view: str | None = None) -> dict:
        """Read retained calls without replay. Use view=index/leases for bounded stable indexes; handle plus header/intent/result/effects/resolution for exact calls, or lease with episode. Full is the legacy complete read."""
        from .operation_reads import read
        return read(self,episode,handle,view,selection,path,offset,limit,max_chars,expected_view)

    def reconcile_operation(self, handle: str, observed: dict | None = None, evidence_paths: list[str] | None = None,
                            episode: str | None = None, expected_lease: str | None = None) -> dict:
        """Recover a call without replay. For a legacy unlinked argument-refusal lease, supply episode, lease handle/hash from inspect_operations, and exact refusal evidence; active/ambiguous effects remain blocked."""
        if episode is not None:
            from .leases import reconcile_unlinked
            return reconcile_unlinked(self,episode,handle,expected_lease,observed,evidence_paths)
        if expected_lease is not None:raise ValueError('expected_lease requires episode')
        from .journal import reconcile
        return reconcile(self,handle,observed,evidence_paths)

    def reconcile_episode(self, episode: str, expected_revision: str, character: dict | None = None,
                          method: dict | None = None, evidence: list[dict] | None = None, applicability: str = '',
                          unresolved: list[str] | None = None, next_question: str = '', close: bool = False,
                          clear_components: list[str] | None = None) -> dict:
        """Join separate character/method judgments to recoverable facts; pending judgments never block save/recovery."""
        from .episodes import reconcile_episode
        return reconcile_episode(self,episode,expected_revision,character,method,evidence or [],applicability,unresolved,next_question,close,clear_components or [])

    def record_dependencies(self, state: str, binding_paths: list[str] | None = None, target_paths: list[str] | None = None,
                            observation: str | None = None) -> dict:
        """Fingerprint exact recorded geometry, topology, construction, pose, bindings, targets and view separately."""
        from .semantics import dependency_state
        return dependency_state(self,state,binding_paths,target_paths,observation)

    def promote_procedure(self, procedure_id: str, judgments: list[str], instruction: str, stages: list[str],
                          conditions: dict, limits: list[str], counterexamples: list[dict] | None = None,
                          executable_paths: list[str] | None = None, level: str = 'local') -> dict:
        """Promote explicit supported method judgments with limits/executables; reusable level requires distinct episode reuse."""
        from .learning import promote
        return promote(self,procedure_id,judgments,instruction,stages,conditions,limits,counterexamples or [],executable_paths or [],level)

    def inspect_control_coverage(self, case_path: str, expected_state: dict) -> dict:
        """Retain offline reverse target ancestry, omitted controls and explicit restrictions; no native effects or controllability claim."""
        from .control_coverage import inspect
        return inspect(self, case_path, expected_state)

    def analyze_repair(self, case_path: str, proposed_delta: list[float] | None = None, max_seconds: float = 20) -> dict:
        """Retain bounded offline coupled XYZ fit/preservation diagnostics and evidence-acquisition nominations; never apply or dispatch."""
        from .repair_analysis import analyze
        return analyze(self, case_path, proposed_delta, max_seconds)

    def register_response(self, dependencies: str, semantic_validation: str, spec: dict, arrays_path: str, evidence: list[dict]) -> dict:
        """Validate a sparse exact/piecewise/derivative/approximate response against bound dependencies, semantics, native baseline and influence scope."""
        from .responses import register
        return register(self,dependencies,semantic_validation,spec,arrays_path,evidence)

    def predict_response(self, response: str, dependencies: str, control_delta: dict, evidence: list[dict]) -> dict:
        """Compute a scoped counterfactual with actual guide/depth constraints; stale dependencies and out-of-range changes are refused."""
        from .responses import predict
        return predict(self,response,dependencies,control_delta,evidence)

    def compare_prediction(self, prediction: str, native_state: str, object_name: str, axis: int,
                           tolerance: float, inspection: dict) -> dict:
        """Compare predicted/native owned coordinates and worst outliers; retain whole/close/depth/motion/appearance judgments independently."""
        from .responses import compare_native
        return compare_native(self,prediction,native_state,object_name,axis,tolerance,inspection)

    def check_reuse(self, record: str, before: str, after: str, domains: list[str]) -> dict:
        """Read whether specified recorded dependency domains still match; view, geometry and semantic claims remain distinct."""
        from .responses import reuse
        return reuse(self,record,before,after,domains)

    def adapt_ancestry_response(self, npz_path: str, schema_path: str, verification_path: str) -> dict:
        """Reuse the retained scoped subdivision/ocular export through an explicit adapter, measuring historical reconstruction and declaring current/influence gaps."""
        from .ancestry_adapter import adapt
        return adapt(self,npz_path,schema_path,verification_path)

    def schedule_ordered_motion(self, plan_path: str, fps: int = 60, normal_cycles: int = 3, slow_cycles: int = 1,
                                slow_speed: float = .25, lead: float = .25, rest: float = .6) -> dict:
        """Schedule exact recorded source identities independently per cycle, preserving closing/reopening and reporting actual omitted samples."""
        from .ordered_motion import plan
        return plan(self,plan_path,fps,normal_cycles,slow_cycles,slow_speed,lead,rest)

    def evidence_manifest(self, episode: str | None = None) -> dict:
        """Declare complete reachable episode evidence and external private file locators/hashes/access/coverage without embedding native assets."""
        from .portable import evidence_manifest
        return evidence_manifest(self,episode)

    def package_readiness(self, episode: str | None = None, manifest_path: str | None = None) -> dict:
        """Verify declared evidence availability/integrity and actual runtime separately from native control and unproven character transfer."""
        from .portable import readiness
        return readiness(self,episode,manifest_path)

    def export_package(self, output_path: str, episode: str | None = None, include_evidence: bool = False) -> dict:
        """Export tools only by default. An explicit episode or include_evidence=True exports a PRIVATE workspace transfer, including authority and locators."""
        from .portable import export
        from .distribution import export_tools
        if episode is None and not include_evidence: return export_tools(output_path)
        return export(self,output_path,episode,include_evidence)

    def verify_installation(self, cache_root: str) -> dict:
        """Compare installed launch/skill bytes and probe the actual fresh launch MCP schema; no native/provider effects or inferred operator adoption."""
        from .installation import verify
        return verify(self,cache_root)

    def register_semantic_graph(self, nodes: list[dict], edges: list[dict], coverage: dict, dependencies: str) -> dict:
        """Bind authored/measured feature-control-surface-neighbor relationships with source evidence and explicit missing coverage."""
        from .semantics import register_graph
        return register_graph(self,nodes,edges,coverage,dependencies)

    def validate_semantics(self, graph: str, dependencies: str, disposition: str, reason: str, evidence: list[dict]) -> dict:
        """Record separate anatomical binding judgment; fresh coordinates cannot approve stale semantic mappings."""
        from .semantics import validate_semantics
        return validate_semantics(self,graph,dependencies,disposition,reason,evidence)

    def semantic_impact(self, graph: str, node: str | None = None, selector: dict | None = None, direction: str = 'both',
                        depth: int = 3, dependencies: str | None = None, validation: str | None = None) -> dict:
        """Traverse explicit relationships in both directions, including surface-to-control/intent; expose stale and missing coverage."""
        from .semantics import impact
        return impact(self,graph,node,selector,direction,depth,dependencies,validation)

    def import_scene(self, record_path: str) -> dict:
        """Import and fingerprint the complete recorded geometry; explicitly historical until a live check."""
        return self.wb.import_scene(record_path)

    def synchronize_scene(self) -> dict:
        """Refresh dirty native geometry on demand and import a verified current scene; keep historical questions intact."""
        live=self._native_call('inspect_live',{'refresh_scene':True})
        freshness=live.get('scene_freshness',{})
        if freshness.get('status')!='current' or not freshness.get('record_path') or not live.get('geometry_state_id'):
            raise Conflict('Native numerical scene is not current; inspect freshness and synchronization failure before reasoning from it')
        imported=self.wb.import_scene(freshness['record_path'])
        state=self.store.get(imported['state'],'state')
        checked=self._native_call('inspect_live',{})
        if checked.get('scene_freshness',{}).get('status')!='current' or checked.get('geometry_state_id')!=state['source_state_id']:
            raise Conflict('Blender changed while importing its scene record; imported evidence remains historical and must be refreshed')
        return dict(state=imported['state'],source_state_id=state['source_state_id'],geometry_revision=imported['geometry_revision'],
            objects=imported['objects'],mesh_objects=imported['mesh_objects'],freshness=checked['scene_freshness'],
            expected_state=checked['expected_state'],mode='current at final native verification; later changes invalidate this claim',
            coverage=state['coverage'],historical_question_unchanged=True)

    def query_live_geometry(self, object_name: str, query: str, parameters: dict, selection: dict | None = None) -> dict:
        """Synchronize then query the actual current scene, refusing results if Blender changes during measurement."""
        synchronized=self.synchronize_scene()
        result=self.wb.query_geometry(synchronized['state'],object_name,query,selection,**parameters)
        checked=self._native_call('inspect_live',{})
        if checked.get('scene_freshness',{}).get('status')!='current' or checked.get('geometry_state_id')!=synchronized['source_state_id']:
            raise Conflict('Blender changed during the live query; do not use the result as current geometry')
        return dict(result,mode='live source verified after measurement',native_state=synchronized['source_state_id'],
                    freshness=checked['scene_freshness'],expected_state=checked['expected_state'])

    def open_question(self, title: str, state: str, region: str, intent: str, protected: list[str] | None = None, references: list[str] | None = None) -> dict:
        """Select a question over a verified snapshot, with protected features and source references."""
        return self.wb.open_question(title,state,region,intent,protected or [],references or [])

    def query_geometry(self, state: str, object_name: str, query: str, parameters: dict, selection: dict | None = None) -> dict:
        """Query exact historical surfaces: nearest, ray, section, bounds or material_path. Semantic selection is explicit."""
        return self.wb.query_geometry(state,object_name,query,selection,**parameters)

    def compare_geometry(self, before: str, after: str, object_name: str, correspondence: str = 'triangles') -> dict:
        """Measure corresponding geometry; changed tessellation requires explicit matching polygon loops."""
        return self.wb.compare_geometry(before,after,object_name,correspondence)

    def record_observation(self, state: str, image_path: str, metadata: dict, role: str = 'actual mesh preview') -> dict:
        """Link a capture to an immutable scene without changing another agent's current question."""
        import numpy as np
        s=self.store.get(state,'state');asset=self.store.blob(image_path);metadata=dict(metadata)
        if metadata.get('scene_state_id')!=s['source_state_id']:raise ValueError('Capture does not identify this exact geometry state')
        if metadata.get('image_sha256')!=asset['sha256']:raise ValueError('Image does not match capture metadata')
        if metadata.get('viewport_depth'):
            depth=dict(metadata['viewport_depth']);pinned=self.store.blob(depth['path'])
            if pinned['sha256']!=depth['sha256']:raise ValueError('Viewport depth differs from the matching image receipt')
            metadata['viewport_depth']=dict(depth,stored=pinned)
        if metadata.get('view_matrix') is not None and metadata.get('scene'):
            view=np.asarray(metadata['view_matrix'],float)
            if view.shape!=(4,4) or not np.isfinite(view).all():raise ValueError('Invalid viewport matrix')
            metadata.setdefault('evaluated_camera_world',np.linalg.inv(view).tolist())
            metadata.setdefault('evaluated_owning_scene',metadata['scene'])
        calibrated=all(metadata.get(k) is not None for k in ('evaluated_camera_world','view_projection_matrix','resolution','evaluated_owning_scene'))
        if calibrated:
            import numpy as np
            for k in ('evaluated_camera_world','view_projection_matrix'):
                a=np.asarray(metadata[k],float)
                if a.shape!=(4,4) or not np.isfinite(a).all():raise ValueError('Invalid capture matrix')
        value={'state':state,'geometry_revision':s['geometry_revision'],'image':asset,'metadata':metadata,'role':role,
               'projection_status':'evaluated capture recorded' if calibrated else 'unverified; visual reference only',
               'state_association':'matching recorded capture state'}
        return {'observation':self.store.put('observation',value),'state':state,'projection_status':value['projection_status']}

    def _pin_diagnostics(self,bundle):
        result={}
        for name,entry in bundle['diagnostics'].items():
            if entry.get('state_id')!=bundle['state_id']:raise ValueError('Capture diagnostic belongs to a different geometry state')
            artifacts={k:self.store.blob(entry[k]) for k in ('path','metadata') if entry.get(k)}
            result[name]=dict(entry,stored=artifacts)
        return result

    def capture_view_bundle(self, expected_state: str, owner: str, feature: str = 'eyes', viewport: dict | None = None,
                            crop: list[int] | None = None, channels: list[str] | None = None) -> dict:
        """Capture the actual inspected viewport and context, then import their exact geometry and observations into the linked store."""
        args={'expected_state':expected_state,'feature':feature,'context':True}
        for k,v in [('viewport',viewport),('crop',crop),('channels',channels)]:
            if v is not None:args[k]=v
        bundle=self._native_call('capture_view',args,owner=owner)
        state=self.wb.import_scene(bundle['record_path'])['state']
        observations=[self.record_observation(state,x['image'],x,x['role']) for x in bundle['observations']]
        payload=dict(native_receipt=self.store.blob(bundle['receipt_path']),state=state,observations=observations,
                     diagnostics=self._pin_diagnostics(bundle),view=bundle['view'],live=bundle.get('live'),
                     unavailable_channels=bundle.get('unavailable_channels',[]))
        return dict(bundle=self.store.put('view_bundle',payload),**payload)

    def ingest_capture(self, receipt_path: str) -> dict:
        """Resume an existing native capture without recapturing or changing the active Blender view."""
        bundle=json.loads(Path(receipt_path).read_text(encoding='utf-8'))
        state=self.wb.import_scene(bundle['record_path'])['state']
        rows=[self.record_observation(state,x['image'],x,x['role']) for x in bundle['observations']]
        value={'state':state,'observations':rows,'native_receipt':self.store.blob(receipt_path),'diagnostics':self._pin_diagnostics(bundle),'view':bundle['view']}
        return dict(bundle=self.store.put('view_bundle',value),**value)

    def project_points(self, observation: str, points: list[list[float]]) -> dict:
        """Project world points through the stored camera; projection alone does not establish visibility."""
        return self.wb.project(observation,points)

    def query_pixel(self, observation: str, pixel: list[float], object_names: list[str], selection: dict | None = None, maximum_hits: int = 8) -> dict:
        """Trace a cropped-image pixel through declared recorded surfaces, retaining occlusion alternatives and triangle owners."""
        import numpy as np
        from .geometry import ray_hits
        ob=self.store.get(observation,'observation');meta=ob['metadata']
        if ob['projection_status']!='evaluated capture recorded':raise ValueError('Pixel queries require a calibrated observation')
        if len(pixel)!=2 or not np.isfinite(pixel).all() or not object_names or not 1<=maximum_hits<=100:raise ValueError('Finite image pixel, explicit objects and bounded hit count required')
        rw,rh=meta['resolution'];width,height=meta.get('native_resolution',[rw,rh]);crop=meta.get('crop') or [0,0,width,height]
        if not 0<=pixel[0]<rw or not 0<=pixel[1]<rh:raise ValueError('Pixel is outside this captured crop')
        x=2*(pixel[0]+crop[0])/width-1;y=1-2*(pixel[1]+crop[1])/height
        inverse=np.linalg.inv(np.asarray(meta['view_projection_matrix'],float));endpoints=[]
        for z in (-1,1):
            p=inverse@np.array([x,y,z,1.])
            if abs(p[3])<1e-12:raise ValueError('Projection has an infinite clip endpoint')
            endpoints.append(p[:3]/p[3])
        direction=endpoints[1]-endpoints[0];length=float(np.linalg.norm(direction));hits=[]
        for name in object_names:
            result=ray_hits(self.wb.arrays(ob['state'],name),endpoints[0],direction,selection,maximum_distance=length)
            hits.extend(dict(hit,object=name) for hit in result['hits'])
        hits.sort(key=lambda hit:hit['distance'])
        return {'observation':observation,'state':ob['state'],'pixel':pixel,'hits':hits[:maximum_hits],'total_hits':len(hits),
                'visibility':'Front-to-back geometric candidates among declared objects. Material transparency, guide visibility and overlay rendering are not inferred.',
                'units':'recorded world-coordinate units','ray_origin':endpoints[0].tolist(),'ray_direction':(direction/length).tolist()}

    def request_reference(self, observation: str, identity_paths: list[str], requested_change: str, kind: str, provider: str,
                          settings: dict, authorization: dict, idempotency_key: str, reviewed_source: str | None = None) -> dict:
        """Prepare an image/video/mesh job tied to the exact mesh view. Video uses a reviewed image and explicit audio-off setting."""
        decision=self.capture_operation_context('generation',{'provider':provider,'kind':kind,
            **{k:settings[k] for k in ('use_case','intent_class','comparison_id','reconstruction_id') if k in settings}})
        result=self.references.request(observation,identity_paths,requested_change,kind,provider,settings,authorization,idempotency_key,reviewed_source)
        return dict(result,decision_context=decision['context_record'])

    def claim_job(self, job: str, expected_revision: str, provider_preflight: dict | None = None) -> dict:
        """Persist one dispatch intent. Tripo checks current policy and fresh mode/model/cost/balance/observed_at/source evidence; a configured comparison also requires source_sha256."""
        decision=self.capture_operation_context('generation',job=job)
        return dict(self.references.claim(job,expected_revision,provider_preflight),decision_context=decision['context_record'])

    def prepare_guide(self, review: str, provider: str, settings: dict, authorization: dict, idempotency_key: str) -> dict:
        """Prepare one mesh-generation job from a reviewed exact-view image; no multi-pose image bundle or automatic spending."""
        reviewed=self.store.get(review,'reference_review');source=self.ledger.read(reviewed['job'])
        decision=self.capture_operation_context('generation',{'provider':provider,'kind':'mesh',
            **{k:settings[k] for k in ('use_case','intent_class','comparison_id','reconstruction_id') if k in settings}})
        result=self.references.request(reviewed['observation'],[str(self.store.resolve_blob(a)) for a in source['intent']['identities']],
            'Reconstruct the reviewed pose and supported '+reviewed['region']+' region, preserving surrounding attachment context.',
            'mesh',provider,settings,authorization,idempotency_key,review)
        return dict(result,decision_context=decision['context_record'])

    def reconcile_job(self, job: str, expected_revision: str, status: str, provider_id: str | None = None,
                      output_paths: list[str] | None = None, receipt: dict | None = None) -> dict:
        """Attach actual provider state and outputs to the same job after success, failure or an uncertain response."""
        return self.references.reconcile(job,expected_revision,status,provider_id,output_paths or [],receipt)

    def inspect_workflow(self, handle: str) -> dict:
        """Read the latest durable reference job or trial, including recovery state and prior revisions."""
        return self.ledger.read(handle)

    def review_reference(self, job: str, output_index: int, disposition: str, region: str, guard_band: float, observed: str,
                         identity_agreement: str, view_agreement: str, pose_agreement: str, conflicts: list[str], landmarks: dict | None = None) -> dict:
        """Record visual target judgment and optional measured landmark drift, with supported region and guard band."""
        return self.references.review(job,output_index,disposition,region,guard_band,observed,identity_agreement,view_agreement,pose_agreement,conflicts,landmarks)

    def qualify_guide(self, job: str, output_index: int, native_state: str, source_anchors: list[list[float]], native_anchors: list[list[float]],
                      pose: str, region: str, source_pose: str, role: str, maximum_anchor_error: float, support: list[dict], expected_registry: dict, review: str) -> dict:
        """Register a reviewed generated NPZ mesh using stable anchors and explicit pose/region support; preserve source geometry."""
        return self.references.qualify(job,output_index,native_state,source_anchors,native_anchors,pose,region,source_pose,role,maximum_anchor_error,support,expected_registry,review)

    def reject_guide(self, job: str, reason: str, evidence_paths: list[str]) -> dict:
        """Retain an unusable reconstructed guide and block its future qualification or installation."""
        item=self.ledger.read(job)
        if item['status']!='completed' or item['intent']['kind']!='mesh' or not reason.strip() or not evidence_paths:
            raise ValueError('A completed mesh job, explicit rejection reason and actual evidence are required')
        payload={'job':job,'reason':reason,'evidence':[self.store.blob(p) for p in evidence_paths]}
        key=self.store.put('guide_rejection',payload)
        updated=self.ledger.update(job,item['revision'],'completed',{'guide_rejection':key},allowed={'completed'})
        return {'rejection':key,'workflow_revision':updated['revision'],**payload}

    def begin_experiment(self, question: str, hypothesis: str, comparison_plan: str, protected: list[str], resource_bound: str, idempotency_key: str) -> dict:
        """Open one recoverable intervention with the modeling question, discriminating comparison and resource bound."""
        q=self.store.get(question,'question')
        if not hypothesis.strip() or not comparison_plan.strip() or not resource_bound.strip():raise ValueError('Hypothesis, comparison and resource bound required')
        return self.ledger.create('experiment',dict(question=question,state=q['state'],hypothesis=hypothesis,comparison_plan=comparison_plan,
                                                  protected=protected,resource_bound=resource_bound),idempotency_key)

    def integrate_guide(self, qualification: str, expected_state: str, owner: str, target_id: str, feature: str,
                        controls: dict, anatomy_basis_paths: list[str]) -> dict:
        """Install the qualified target through the native transaction, reusing its immutable source/review/registration lineage."""
        q=self.store.get(qualification,'guide_qualification');review=self.references.require_usable_review(q['review'])
        if self.ledger.read(q['job'])['data'].get('guide_rejection'):
            raise ValueError('Reconstructed guide was rejected after qualification; no native installation is allowed')
        source=self.ledger.read(review['job']);image=source['data']['outputs'][review['output_index']]
        def ref(asset,role=None):
            result={'path':str(self.store.resolve_blob(asset)),'sha256':asset['sha256']}
            if role:result['role']=role
            return result
        def record_ref(key):
            path=self.store.root/'records'/(key+'.json');self.store.get(key)
            return {'path':str(path),'sha256':digest(path.read_bytes())}
        provenance={'raw_mesh':ref(q['source']),'artwork':[ref(image)],'registration_record':record_ref(qualification),
                    'derivation_record':record_ref(self.ledger.read(q['job'])['revision']),
                    'anatomy_basis':[ref(self.store.blob(p)) for p in anatomy_basis_paths]}
        native={'target_id':target_id,'registered_npz':ref(q['registered']),'source_state_id':q['source_state_id'],
                'feature':feature,'pose':controls,'region_support':{'region':q['region'],'supported':q['support'],'limits':q['limitations']},
                'target_role':q['role'],'reviewed':True,'review_record':record_ref(q['review']),'provenance':provenance}
        return self._native_call('integrate_guide',{'expected_state':expected_state,'qualification':native,'expected_registry':q['expected_registry']},owner=owner)

    def propose_change(self, experiment: str, expected_revision: str, proposal: dict, evidence: list[str], response_model: dict | None = None) -> dict:
        """Record a native shape-key or advanced script proposal with guide constraints and explicit response-model limits."""
        from .native_bridge import validate_arguments
        item=self.ledger.read(experiment)
        if item['kind']!='experiment' or not evidence:raise ValueError('Experiment and existing evidence required')
        validate_arguments('apply_proposal',{'expected_state':'schema-validation-only','proposal':proposal})
        evidence_assets=[self.store.blob(path) for path in evidence]
        proposal=json.loads(json.dumps(proposal))
        for ref in proposal.get('constraints',[])+([proposal['script']] if proposal.get('script') else []):
            blob=self.store.blob(ref['path'])
            if blob['sha256']!=ref['sha256']:raise ValueError('Proposal input changed before recording')
            ref['path']=str(self.store.resolve_blob(blob))
        if response_model:
            if response_model.get('prediction'):
                predicted=self.store.get(response_model['prediction'],'prediction')
                dependencies=self.store.get(predicted['dependencies'],'dependency_state')
                if dependencies['state']!=item['intent']['state']:
                    raise ValueError('Prediction and experiment must share exact recorded baseline')
                if not predicted['influence']['complete_for_this_change']:
                    raise ValueError('Prediction influence is incomplete; qualify the affected neighborhood/boundary before native proposal')
                response_model=dict(response_model,validation='registered numerical response; native/appearance comparison pending')
            else:
                required=('kind','source_state','topology','modifier_order','pose','support_dependencies','coverage','validation')
                if any(k not in response_model for k in required):raise ValueError('Response model needs state, topology, modifier, pose, support and influenced-boundary validation')
                response_model=dict(response_model,validation_status='legacy declared assumptions, not a validated response_model record')
        return self.ledger.update(experiment,expected_revision,'proposed',dict(proposal=proposal,evidence=evidence,evidence_assets=evidence_assets,response_model=response_model),allowed={'prepared','proposed'})

    def apply_trial(self, experiment: str, expected_revision: str, expected_state: str, owner: str) -> dict:
        """Checkpoint and apply the recorded proposal through the sole Blender owner; uncertain outcomes are never retried."""
        item=self.ledger.read(experiment)
        baseline=self.store.get(item['intent']['state'],'state')
        live=self._native_call('inspect_live',{})
        if live['expected_state']!=expected_state or live.get('geometry_state_id')!=baseline['source_state_id']:
            raise PreconditionRefusal('The experiment baseline differs from the live scene; capture and rebase the proposal before applying it',
                'apply_trial.baseline_precondition',dict(expected_state=expected_state,observed_expected_state=live['expected_state'],
                    baseline_source_state=baseline['source_state_id'],observed_geometry_state=live.get('geometry_state_id'),
                    mutation_dispatched=False,returned_observation='inspect_live',scene_freshness=live.get('scene_freshness')))
        claimed=self.ledger.update(experiment,expected_revision,'applying',{},allowed={'proposed'})
        try:
            result=self._native_call('apply_proposal',{'expected_state':expected_state,'proposal':item['data']['proposal']},owner=owner)
        except Exception as error:
            self.ledger.update(experiment,claimed['revision'],'needs_reconciliation',{'error':str(error)},allowed={'applying'})
            raise
        return self.ledger.update(experiment,claimed['revision'],'applied',{'native_trial':result},allowed={'applying'})

    def resolve_trial(self, experiment: str, expected_revision: str, native_receipt_path: str, observed_status: str, evidence: str) -> dict:
        """Reconcile an interrupted native trial using its persisted receipt before any further mutation."""
        if observed_status not in ('applied','rejected','failed') or not evidence.strip():raise ValueError('Observed disposition and evidence required')
        receipt=self.store.blob(native_receipt_path)
        return self.ledger.update(experiment,expected_revision,observed_status,dict(reconciliation=dict(receipt=receipt,evidence=evidence)),allowed={'applying','needs_reconciliation'})

    def reject_trial(self, experiment: str, expected_revision: str, expected_state: str, owner: str, receipt_path: str, reason: str) -> dict:
        """Restore the recorded trial baseline only if no later live edit would be overwritten; retain rejected evidence."""
        if not reason.strip():raise ValueError('Rejection reason required')
        claimed=self.ledger.update(experiment,expected_revision,'rejecting',{'reason':reason},allowed={'applied'})
        try:result=self._native_call('rollback_trial',{'expected_state':expected_state,'receipt_path':receipt_path},owner=owner)
        except Exception as error:
            self.ledger.update(experiment,claimed['revision'],'needs_reconciliation',{'error':str(error)},allowed={'rejecting'});raise
        return self.ledger.update(experiment,claimed['revision'],'rejected',{'rollback':result},allowed={'rejecting'})

    def retain_candidate(self, experiment: str, expected_revision: str, expected_state: str, owner: str, label: str,
                         comparison_record: str, reopen_receipt_path: str, path: str | None = None) -> dict:
        """Save a reviewed local candidate with comparison and independent reopen evidence; user appearance acceptance stays separate."""
        comparison=self.store.get(comparison_record)
        reopened=self.store.blob(reopen_receipt_path);verification=json.loads(self.store.resolve_blob(reopened).read_text(encoding='utf-8'))
        live=self._native_call('inspect_live',{})
        if live['expected_state']!=expected_state:raise Conflict('Live state changed before candidate retention')
        if verification.get('status')!='pass' or not verification.get('source_unchanged') or verification.get('native_content_hash')!=live['native_content_hash']:
            raise ValueError('Independent reopen does not verify this candidate native content')
        source=Path(verification['source'])
        if digest(source.read_bytes())!=verification.get('source_sha256'):raise ValueError('Independently reopened source changed')
        claimed=self.ledger.update(experiment,expected_revision,'retaining',{'comparison':comparison_record,'reopen':reopened},allowed={'applied'})
        try:
            args={'expected_state':expected_state,'label':label}
            if path:args['path']=path
            if (not path or Path(path).resolve()==Path(live['file']).resolve()) and live.get('saved_file'):
                args['expected_file_sha256']=live['saved_file']['sha256']
            result=self._native_call('save_checkpoint',args,owner=owner)
        except Exception as error:
            self.ledger.update(experiment,claimed['revision'],'needs_reconciliation',{'error':str(error)},allowed={'retaining'});raise
        return self.ledger.update(experiment,claimed['revision'],'retained',{'checkpoint':result,'user_appearance_acceptance':'not implied'},allowed={'retaining'})

    def import_motion(self, manifest_path: str) -> dict:
        """Import a matched finite-pose comparison, verifying every captured image and preserving its coverage limits."""
        return self.motion.import_player(manifest_path)

    def ingest_motion_capture(self, receipt_path: str) -> dict:
        """Import native motion with the exact close/context observation pair and full geometry for every captured pose."""
        receipt=json.loads(Path(receipt_path).read_text(encoding='utf-8'));frames=[]
        for row in receipt['frames']:
            captured=self.ingest_capture(row['bundle']['receipt_path'])
            frames.append({'index':row['index'],'pose':row['requested_pose'],'controls':row['controls'],**captured})
        value={'source':self.store.blob(receipt_path),'frames':frames,'view':receipt['view'],
               'coverage':{'poses':len(frames),'complete_requested_capture':receipt['status']=='completed',
                           'limits':receipt['sampling_limit'],'visual_inspection':'not implied'},
               'native_preservation':{'keys_unchanged':receipt.get('keys_unchanged'),'pose_restored':receipt.get('pose_restored')}}
        return {'motion':self.store.put('native_motion',value),'frames':len(frames),'coverage':value['coverage']}

    def compare_motion(self, baseline_observations: list[str], candidate_observations: list[str], phases: list[dict],
                       object_name: str | None = None, reference_mapping: list[dict] | None = None, correspondence: str = 'triangles') -> dict:
        """Compare matched native poses/views; reference timestamps and phase correspondence remain explicit separate evidence."""
        return self.motion.compare(baseline_observations,candidate_observations,phases,object_name,reference_mapping,correspondence)

    def analyze_video(self, video_path: str, crop: list[int] | None = None, maximum_frames: int = 240) -> dict:
        """Extract generated motion at full or selected close-up framing and nominate changing frames for inspection."""
        return self.motion.extract_video(video_path,crop,maximum_frames)

    def assess_motion_continuity(self, motion: str, video_analysis: str, mapping: list[dict], action_control: str = 'blink',
                                 baseline_motion: str | None = None, object_name: str | None = None, correspondence: str = 'triangles') -> dict:
        """Validate ordered closing/contact/reopening evidence against decoded video, with matched close/context views and optional baseline geometry."""
        return self.motion.assess_continuity(motion,video_analysis,mapping,action_control,baseline_motion,object_name,correspondence)

    def export_replay(self, motion: str, output_dir: str, fps: int = 60, normal_cycles: int = 3,
                      slow_cycles: int = 1, slow_speed: float = .25, panel_size: list[int] | None = None) -> dict:
        """Encode normal/slow, whole/close-up comparisons from pinned source frames, with decode verification and frame mapping."""
        return self.motion.encode(motion,output_dir,fps,normal_cycles,slow_cycles,slow_speed,panel_size)

    def record_outcome(self, question: str, character: dict, method: dict, evidence: list[str], applicability: str) -> dict:
        """Retain separate character and method dispositions with evidence and applicability; preserve earlier records."""
        return self.wb.record_outcome(question,character,method,evidence,applicability)

    def retrieve_experience(self, query: str, limit: int = 6, context: dict | None = None) -> dict:
        """Retrieve concise relevant decisions, successes, failures and limits with exact source links."""
        return retrieve(self.workspace,self.store,query,limit,context)

    def read_record(self, record: str, path: list[str | int] | None = None, offset: int = 0,
                    limit: int = 20, max_chars: int = 8000) -> dict:
        """Read immutable retained JSON. Omit path for full legacy data; [] selects a bounded root. Keys/indices select fields; integer object selectors use sorted keys. Windows preserve content identity."""
        if path is None:
            if offset!=0 or limit!=20 or max_chars!=8000: raise ValueError('Window options require path; use [] for a bounded root')
            return self.store.get(record)
        from .bounded_reads import record as read
        return read(self,record,path,offset,limit,max_chars)

# Every public effectful/fact-producing Python method uses the same journal as
# MCP/CLI. Reads are intentionally not recorded. Facade calls establish scope;
# nested public operations belong to the enclosing operation's complete result.
for _name in ModelingService.operations():
    if _name not in READ_OPERATIONS | {'run_episode_operation', 'reconcile_operation'}:
        setattr(ModelingService, _name, recorded(getattr(ModelingService, _name)))
