"""Exact-view reference lineage, durable external jobs, and numerical guide registration.

Generators remain explicit transports. Preparing a request never claims submission.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import numpy as np
from PIL import Image
from . import geometry
from .store import canonical, digest, atomic_write


class References:
    def __init__(self, workbench, ledger, generation_policy=None):
        self.wb, self.store, self.ledger = workbench, workbench.store, ledger
        self.generation_policy = generation_policy

    @staticmethod
    def _route(mode, model):
        mode = str(mode).strip().replace('_', ' ').casefold()
        model = str(model).strip().casefold()
        # Only known UI labels are aliases; arbitrary model versions stay distinct.
        if model == 'p2.0 - preview':
            model = 'p2.0'
        return mode, model

    def _local_reconstruction_policy(self, required, provider, settings, generation_input, reviewed_source):
        identifier=settings['reconstruction_id']
        configured=required.get('authorized_local_reconstructions',{}).get(identifier)
        if not configured or configured.get('status')!='authorized':
            raise ValueError('Local reconstruction is not currently authorized in MESH-GENERATION.json')
        if (configured.get('intent_class')!='local_guide_reconstruction'
                or settings.get('intent_class')!='local_guide_reconstruction'
                or configured.get('use_case')!='exact_view_enlarged_local_detail'
                or settings.get('use_case')!=configured['use_case']):
            raise ValueError('Local reconstruction requires its explicit local-guide intent class and use case')
        if (provider.strip().casefold() not in ('tripo studio','tripo studio website')
                or configured.get('provider_id')!='tripo' or settings.get('provider_id','tripo')!='tripo'
                or configured.get('transport')!='Tripo Studio website' or settings.get('transport')!=configured['transport']
                or required.get('transport',{}).get('selected')!=configured['transport']
                or self._route(configured.get('mode'),configured.get('model'))!=('hd model','v3.1')
                or self._route(settings.get('mode'),settings.get('model'))!=('hd model','v3.1')):
            raise ValueError('Local reconstruction requires the authorized HD Model / v3.1 Studio website route')
        authority=configured.get('authorization',{});key=configured.get('idempotency_key')
        if not authority.get('scope') or not authority.get('source') or not isinstance(key,str) or not key.strip():
            raise ValueError('Local reconstruction requires existing scoped authorization provenance and a fixed new job key')
        if any(c.get('idempotency_key')==key for c in required.get('authorized_comparisons',{}).values()):
            raise ValueError('Local reconstruction cannot reuse a comparison job key')
        outputs=configured.get('output_settings',{})
        if not isinstance(outputs,dict) or not outputs.get('topology') or not outputs.get('polycount'):
            raise ValueError('Local reconstruction requires explicit topology and polycount choices')
        if any(settings.get(name)!=value or type(settings.get(name)) is not type(value) for name,value in outputs.items()):
            raise ValueError('Prepared output settings differ from the authorized local reconstruction')
        cap=configured.get('max_existing_credit_cost');cost=settings.get('displayed_generation_credits')
        if any(type(value) not in (int,float) or not math.isfinite(value) or value<0 for value in (cap,cost)) or cost>cap:
            raise ValueError('Local reconstruction exceeds its explicit existing-credit cost ceiling')
        if not generation_input or not reviewed_source:raise ValueError('Local reconstruction requires the selected reviewed input')
        self.store.resolve_blob(generation_input);review=self.require_usable_review(reviewed_source)
        if (reviewed_source!=configured.get('review_id') or review['job']!=configured.get('source_image_job')
                or generation_input['sha256']!=configured.get('source_sha256') or review['source_sha256']!=generation_input['sha256']):
            raise ValueError('Local reconstruction source/review identity differs from the selected authorization')
        return dict(mode='HD Model',model='v3.1',reconstruction_authorization={'reconstruction_id':identifier,'policy':configured})

    def _mesh_policy(self, kind, provider, settings, generation_input=None, reviewed_source=None):
        comparison_id = settings.get('comparison_id')
        reconstruction_id=settings.get('reconstruction_id');intent_class=settings.get('intent_class')
        if intent_class is not None and kind!='mesh':raise ValueError('Mesh intent_class is only valid for mesh generation')
        if reconstruction_id is not None and (kind!='mesh' or not isinstance(reconstruction_id,str) or not reconstruction_id):
            raise ValueError('reconstruction_id must identify a configured local mesh reconstruction')
        if comparison_id is not None and reconstruction_id is not None:
            raise ValueError('Comparison and ordinary local reconstruction authorizations are distinct')
        expected_class='local_guide_reconstruction' if reconstruction_id is not None else 'model_comparison' if comparison_id is not None else 'general_mesh'
        if intent_class is not None and intent_class!=expected_class:
            raise ValueError('Generation intent_class does not match its configured authorization type')
        if comparison_id is not None and (kind != 'mesh' or not isinstance(comparison_id, str) or not comparison_id):
            raise ValueError('A comparison_id must identify an authorized mesh comparison')
        if not self.generation_policy or not Path(self.generation_policy).exists():
            if comparison_id is not None or reconstruction_id is not None:
                raise ValueError('Local reconstruction authorization is unavailable' if reconstruction_id is not None else 'Mesh comparison authorization is unavailable')
            return None
        if kind != 'mesh':
            return None
        policy = json.loads(Path(self.generation_policy).read_text(encoding='utf-8'))
        if 'tripo' not in provider.casefold() and str(settings.get('provider_id', '')).casefold() != 'tripo':
            if comparison_id is not None or reconstruction_id is not None:
                raise ValueError('Local reconstruction is authorized only for Tripo Studio' if reconstruction_id is not None else 'Mesh comparison is authorized only for Tripo')
            return None
        required = policy['tripo']
        route = self._route(settings.get('mode', ''), settings.get('model', ''))
        if reconstruction_id is not None:
            return self._local_reconstruction_policy(required,provider,settings,generation_input,reviewed_source)
        if comparison_id is not None:
            comparison = required.get('authorized_comparisons', {}).get(comparison_id)
            if not comparison or comparison.get('status') != 'authorized':
                raise ValueError('Mesh comparison is not currently authorized in MESH-GENERATION.json')
            if (comparison.get('provider_id') != 'tripo'
                    or provider.strip().casefold() not in ('tripo', 'tripo studio', 'tripo hd model v3.1')
                    or str(settings.get('provider_id', 'tripo')).casefold() != 'tripo'
                    or self._route(comparison.get('mode'), comparison.get('model')) != ('hd model', 'v3.1')
                    or route != ('hd model', 'v3.1')):
                raise ValueError('Authorized comparison route is Tripo HD Model / v3.1 only')
            provenance = comparison.get('authorization', {})
            if (not provenance.get('scope') or not provenance.get('source')
                    or not str(comparison.get('idempotency_key', '')).strip()):
                raise ValueError('Configured comparison requires authorization provenance and a fixed job key')
            # Read the actual prepared bytes; a caller-supplied matching hash is not proof.
            if not generation_input:
                raise ValueError('Comparison requires the reviewed generation input')
            self.store.resolve_blob(generation_input)
            if generation_input['sha256'] != comparison.get('source_sha256'):
                raise ValueError('Generation input does not match the authorized comparison source')
            return dict(mode='HD Model', model='v3.1',
                        comparison_authorization={'comparison_id': comparison_id, 'policy': comparison})
        if (route != self._route(required['mode'], required['model'])
                or 'hd model' in provider.casefold() or 'v3.1' in provider.casefold()):
            raise ValueError('Tripo requires Smart Mesh / P2.0 for this workspace. '
                             'Do not reuse HD v3.1 defaults; read MESH-GENERATION.json.')
        return required

    VIEW_SLOTS = ('front', 'left', 'right', 'back')

    def _multi_view_rule(self):
        """The workspace's multi-view requirement for Tripo meshes, or None when the policy has none."""
        if not self.generation_policy or not Path(self.generation_policy).exists():
            return None
        rule = json.loads(Path(self.generation_policy).read_text(encoding='utf-8-sig')).get('tripo', {}).get('multi_view_requirement')
        return rule if isinstance(rule, dict) and rule.get('required') else None

    def _view_inputs(self, views, reviewed_source):
        """Bind each multi-view slot to the exact bytes of a currently usable image review."""
        if (not isinstance(views, dict) or not views or set(views) - set(self.VIEW_SLOTS)
                or not all(isinstance(v, str) and v for v in views.values())):
            raise ValueError('Multi-view inputs map front/left/right/back slots to usable image review IDs')
        if views.get('front') != reviewed_source:
            raise ValueError('The front slot must be the reviewed source of the mesh request')
        bound = {}
        for slot in self.VIEW_SLOTS:
            if slot not in views:
                continue
            review = self.require_usable_review(views[slot])
            source = self.ledger.read(review['job'])['data']['outputs'][review['output_index']]
            if review['source_sha256'] != source['sha256']:
                raise ValueError('Reviewed ' + slot + ' view changed')
            self.store.resolve_blob(source)
            bound[slot] = {'review': views[slot], 'source': source}
        hashes = [entry['source']['sha256'] for entry in bound.values()]
        if len(set(hashes)) != len(hashes):
            raise ValueError('Each view slot needs its own image; one image in several slots is still a single-view mesh')
        return bound

    def _require_views(self, kind, required, view_inputs):
        rule = self._multi_view_rule() if kind == 'mesh' and required is not None else None
        if rule:
            minimum = rule.get('minimum_distinct_views', 2)
            if type(minimum) is not int or minimum < 2:
                raise ValueError('multi_view_requirement.minimum_distinct_views must be an integer of at least 2')
            if not view_inputs or len(view_inputs) < minimum:
                raise ValueError('This workspace requires at least %d distinct reviewed views in multi-view mode '
                                 '(settings.views); a mesh is never generated from a single image' % minimum)

    def _provider_preflight(self, required, settings, preflight, view_inputs=None):
        if required is None:
            return
        if not isinstance(preflight, dict):
            raise ValueError('Tripo dispatch requires a fresh recorded provider_preflight from the generation panel')
        if (self._route(preflight.get('mode'), preflight.get('model')) != self._route(required['mode'], required['model'])
                or not isinstance(preflight.get('source'), str) or not preflight['source'].strip()):
            raise ValueError('Live Tripo panel must confirm ' + required['mode'] + ' / ' + required['model'] + ' before dispatch')
        if view_inputs and preflight.get('slot_sha256') != {slot: entry['source']['sha256'] for slot, entry in view_inputs.items()}:
            raise ValueError('Live Tripo panel must show exactly the prepared image in every view slot (slot_sha256)')
        outputs = settings.get('output_settings')
        if outputs is not None and (not isinstance(outputs, dict) or any(
                preflight.get(k) != v or type(preflight.get(k)) is not type(v) for k, v in outputs.items())):
            raise ValueError('Live Tripo panel output settings differ from the prepared request')
        comparison = required.get('comparison_authorization')
        if comparison and preflight.get('source_sha256') != comparison['policy']['source_sha256']:
            raise ValueError('Live Tripo panel must identify the authorized comparison source_sha256')
        reconstruction=required.get('reconstruction_authorization')
        if reconstruction:
            configured=reconstruction['policy']
            if preflight.get('source_sha256')!=configured['source_sha256'] or any(preflight.get(k)!=v or type(preflight.get(k)) is not type(v) for k,v in configured['output_settings'].items()):
                raise ValueError('Live Tripo panel source and output settings must match the authorized local reconstruction')
        try:
            if not isinstance(preflight['observed_at'], str):
                raise ValueError('observed_at must be an ISO timestamp')
            observed = datetime.fromisoformat(preflight['observed_at'].replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - observed).total_seconds()
            credits = preflight['displayed_generation_credits']
            balance = preflight['observed_balance']
            valid = (0 <= age <= 600 and type(credits) in (int, float) and type(balance) in (int, float)
                     and math.isfinite(credits) and math.isfinite(balance)
                     and 0 <= credits <= balance and credits == settings.get('displayed_generation_credits'))
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError('Tripo preflight is stale or its displayed cost/balance differs from the prepared request')

    def require_usable_review(self, review_id, _seen=None):
        seen=set() if _seen is None else set(_seen)
        if review_id in seen:raise ValueError('Reference lineage contains a cycle; inspect the original jobs')
        seen.add(review_id)
        review = self.store.get(review_id, 'reference_review')
        job = self.ledger.read(review['job'])
        index = str(review['output_index'])
        if job['status'] != 'completed' or review['disposition'] not in ('useful', 'partial'):
            raise ValueError('Source image does not have a usable completed review')
        if index in job['data'].get('rejected_outputs', []):
            raise ValueError('Source image was rejected; use a new reviewed version, not an earlier positive review')
        latest = job['data'].get('latest_reviews', {}).get(index)
        if latest is not None and latest != review_id:
            raise ValueError('Reference review is superseded; inspect the current image review')
        if not all(str(review.get(k, '')).strip() for k in ('identity_agreement', 'view_agreement', 'pose_agreement')):
            raise ValueError('Explicit likeness, view and pose review is required before downstream generation')
        if job['intent'].get('reviewed_source'):
            self.require_usable_review(job['intent']['reviewed_source'], seen)
        return review

    def lineage_status(self, job):
        item=self.ledger.read(job)
        try:
            if item['intent'].get('reviewed_source'):
                self.require_usable_review(item['intent']['reviewed_source'])
            return dict(status='current', basis='current recorded source reviews; not live geometry or appearance acceptance')
        except ValueError as error:
            return dict(status='stale', reason=str(error), recovery='Preserve this job and outputs. Review the original source; no automatic generation or retry.')

    def request(self, observation, identity_paths, requested_change, kind, provider, settings, authorization, idempotency_key, reviewed_source=None):
        if kind not in ('image', 'video', 'mesh'):
            raise ValueError('Reference kind must be image, video or mesh')
        ob = self.store.get(observation, 'observation')
        if ob['state_association'] != 'matching recorded capture state':
            raise ValueError('An exact-state mesh observation is required')
        self.store.resolve_blob(ob['image'])
        if not identity_paths or not requested_change.strip() or not provider.strip():
            raise ValueError('Character references, requested change and explicit provider required')
        identities = [self.store.blob(p) for p in identity_paths]
        source = None
        if kind == 'image' and reviewed_source:
            review=self.require_usable_review(reviewed_source)
            if review['observation']!=observation:raise ValueError('Reviewed source must use this exact observation')
            source=self.ledger.read(review['job'])['data']['outputs'][review['output_index']]
        if kind in ('video', 'mesh'):
            if not reviewed_source:
                raise ValueError('Video and mesh generation require a reviewed image derived from this view')
            review = self.require_usable_review(reviewed_source)
            if review['observation'] != observation or review['disposition'] not in ('useful', 'partial'):
                raise ValueError('Reviewed source does not support this exact-view question')
            output = self.ledger.read(review['job'])
            source = output['data']['outputs'][review['output_index']]
            if review['source_sha256'] != source['sha256']:
                raise ValueError('Reviewed source changed')
        if not authorization.get('scope') or not authorization.get('source'):
            raise ValueError('Record existing user authorization scope and its source')
        settings = dict(settings)
        if 'views' in settings and kind != 'mesh':
            raise ValueError('View slots belong to mesh generation')
        view_inputs = self._view_inputs(settings['views'], reviewed_source) if 'views' in settings else None
        generation_input = source or ob['image']
        input_path = str(self.store.resolve_blob(generation_input))
        required = self._mesh_policy(kind, provider, settings, generation_input, reviewed_source)
        self._require_views(kind, required, view_inputs)
        comparison = (required or {}).get('comparison_authorization')
        reconstruction=(required or {}).get('reconstruction_authorization')
        if comparison and idempotency_key != comparison['policy']['idempotency_key']:
            raise ValueError('Comparison requires its configured idempotency_key; recover the existing job instead of creating another')
        if reconstruction:
            configured=reconstruction['policy']
            if idempotency_key!=configured['idempotency_key']:
                raise ValueError('Local reconstruction requires its fixed job key; reconcile the existing job instead of retrying')
            if any(authorization.get(k)!=configured['authorization'][k] for k in ('scope','source')):
                raise ValueError('Local reconstruction must retain its configured existing authorization scope/source')
        if kind == 'video':
            settings['generate_audio'] = False
        intent = dict(observation=observation, state=ob['state'], preview=ob['image'], identities=identities,
                      requested_change=requested_change, kind=kind, provider=provider, settings=settings,
                      authorization=authorization, reviewed_source=reviewed_source, generation_input=generation_input)
        intent['reference_roles']=dict(view_pose_crop=ob['image'],character_identity=identities,
            motion_phase=settings.get('motion_phase'),qualified_depth=None,
            provisional_native_anatomy=ob['state'],disputed_assumptions=settings.get('disputed_assumptions',[]),
            independence='Derived targets do not independently validate anatomy copied from their source')
        if comparison:
            intent['comparison_authorization'] = comparison
        if reconstruction:
            intent.update(intent_class='local_guide_reconstruction',reconstruction_authorization=reconstruction)
        if view_inputs:
            intent['view_inputs'] = view_inputs
        # Older prepared jobs predate explicit roles. Preserve their immutable
        # intent during an exact retry instead of manufacturing another job.
        handle=digest(canonical(['reference_job',idempotency_key]))
        try:previous=self.ledger.read(handle)
        except FileNotFoundError:previous=None
        roles=intent['reference_roles']
        if previous and 'reference_roles' not in previous['intent'] and previous['intent']=={k:v for k,v in intent.items() if k!='reference_roles'}:
            intent=previous['intent']
        job = self.ledger.create('reference_job', intent, idempotency_key)
        inputs = [input_path]
        if kind == 'image':
            inputs += [str(self.store.resolve_blob(x)) for x in identities]
        slots = {slot: str(self.store.resolve_blob(entry['source'])) for slot, entry in intent.get('view_inputs', {}).items()}
        if slots:
            inputs = list(slots.values())
        prompt = ('Depict the character defined by the supplied authoritative identity artwork at the viewing conditions of the first image. '
                  'Use that mesh preview for camera perspective, crop, zoom, spatial framing and pose; its faulty anatomy is not identity authority. '
                  'Character references govern feature design, proportions, likeness and intended style. '
                  'Review the resulting likeness before using the output for mesh generation. ' + requested_change)
        if kind == 'video':
            prompt += ' Fixed camera and framing. Silent video, no speech, music or audio.'
        return dict(job=job['handle'], revision=job['revision'], status=job['status'], reused=job['reused'], transport={'provider':provider, 'kind':kind, 'input_paths':inputs, 'prompt':prompt,
                                       'settings':settings, 'reference_roles':roles, 'submission':'not submitted by preparation',
                                       **({'view_slots':slots} if slots else {}),
                                       'next':'claim_job before dispatch; reconcile the same job after an uncertain response'},
                    limits='Generated view, likeness and motion require review; source depth is not inferred from pixels.')

    def claim(self, handle, revision, provider_preflight=None):
        job = self.ledger.read(handle)
        if job['kind']!='reference_job':
            raise ValueError('Only a reference job can be claimed for generator dispatch')
        if job['intent'].get('reviewed_source'):
            review = self.require_usable_review(job['intent']['reviewed_source'])
            source = self.ledger.read(review['job'])['data']['outputs'][review['output_index']]
            if (job['intent']['generation_input'] != source or review['source_sha256'] != source['sha256']
                    or review['observation'] != job['intent']['observation']):
                raise ValueError('Prepared generation input no longer matches its reviewed source')
        intent = job['intent']
        self.store.resolve_blob(intent['generation_input'])
        required = self._mesh_policy(intent['kind'], intent['provider'], intent['settings'], intent['generation_input'], intent.get('reviewed_source'))
        comparison = (required or {}).get('comparison_authorization')
        if comparison != intent.get('comparison_authorization'):
            raise ValueError('Comparison authorization changed or was not bound at preparation; inspect the existing job')
        if comparison and handle != digest(canonical(['reference_job', comparison['policy']['idempotency_key']])):
            raise ValueError('Comparison job does not match the configured fixed job identity')
        reconstruction=(required or {}).get('reconstruction_authorization')
        if reconstruction!=intent.get('reconstruction_authorization'):
            raise ValueError('Local reconstruction authorization changed or was not bound at preparation')
        if reconstruction and (intent.get('intent_class')!='local_guide_reconstruction'
                or handle!=digest(canonical(['reference_job',reconstruction['policy']['idempotency_key']]))):
            raise ValueError('Local reconstruction job identity or intent class differs from its authorization')
        views = intent.get('view_inputs')
        for slot, entry in (views or {}).items():
            review = self.require_usable_review(entry['review'])
            source = self.ledger.read(review['job'])['data']['outputs'][review['output_index']]
            if source != entry['source'] or review['source_sha256'] != source['sha256']:
                raise ValueError('Prepared ' + slot + ' view no longer matches its reviewed source')
            self.store.resolve_blob(source)
        self._require_views(intent['kind'], required, views)
        self._provider_preflight(required, intent['settings'], provider_preflight, views)
        data = {'submission_intent':True}
        if provider_preflight is not None:
            data['provider_preflight'] = provider_preflight
        return self.ledger.update(handle, revision, 'dispatching', data, allowed={'prepared'})

    def reconcile(self, handle, revision, status, provider_id=None, output_paths=(), receipt=None):
        job = self.ledger.read(handle)
        if job['kind'] != 'reference_job':
            raise ValueError('Expected reference job')
        if status not in ('submitted','unknown','completed','failed','cancelled'):
            raise ValueError('Unsupported job result state')
        if status == 'completed' and not output_paths:
            raise ValueError('Completed jobs require actual output artifacts')
        assets = [self.store.blob(p) for p in output_paths]
        receipt = receipt or {}
        # Credentials are never part of a job receipt.
        if any(x in canonical(receipt).decode().lower() for x in ('authorization:', 'bearer ', 'api_key', 'access_token')):
            raise ValueError('Remove credentials from the provider receipt')
        allowed = {'dispatching','submitted','unknown'}
        if status == 'cancelled' and not provider_id and not output_paths:
            allowed.add('prepared')
        return self.ledger.update(handle, revision, status, dict(provider_id=provider_id,outputs=assets,receipt=receipt),
                                  allowed=allowed)

    def review(self, job, output_index, disposition, region, guard_band, observed, identity_agreement, view_agreement, pose_agreement, conflicts, landmarks=None):
        measured=None
        if landmarks is not None:
            if not isinstance(landmarks,dict) or not {'source','output'} <= landmarks.keys():
                raise ValueError('landmarks requires source and output: paired nonempty Nx2 pixel arrays. Correct the review input; reuse the completed job and image.')
            try:before,after=np.asarray(landmarks['source'],float),np.asarray(landmarks['output'],float)
            except (TypeError,ValueError) as error:raise ValueError('landmarks.source/output must be finite numeric Nx2 arrays') from error
            if before.shape!=after.shape or before.ndim!=2 or before.shape[1]!=2 or not len(before) or not np.isfinite([before,after]).all():
                raise ValueError('landmarks.source/output must be paired nonempty finite Nx2 pixel arrays')
            delta=np.linalg.norm(after-before,axis=1)
            measured=dict(samples=len(delta),maximum_pixel_drift=float(delta.max()),rms_pixel_drift=float(np.sqrt(np.mean(delta**2))),
                          definition='Registration only; no automatic identity judgment')
        item = self.ledger.read(job)
        if item['status'] != 'completed' or item['intent']['kind'] != 'image':
            raise ValueError('A completed image job is required')
        if disposition not in ('useful','partial','rejected') or not region or not observed:
            raise ValueError('Explicit region, visual observations and disposition required')
        if guard_band < 0:
            raise ValueError('Guard band must be nonnegative pixels')
        if disposition != 'rejected':
            if str(output_index) in item['data'].get('rejected_outputs', []):
                raise ValueError('Rejected image remains ineligible; create a corrected version for a new review')
            if not all(str(x).strip() for x in (identity_agreement, view_agreement, pose_agreement)):
                raise ValueError('A usable review must explicitly judge likeness, view and pose')
        asset = item['data']['outputs'][output_index]
        path = self.store.resolve_blob(asset)
        with Image.open(path) as im:
            resolution = list(im.size)
        payload = dict(job=job, output_index=output_index, observation=item['intent']['observation'], source_sha256=asset['sha256'],
                       disposition=disposition, region=region, guard_band_pixels=guard_band, resolution=resolution,
                       visual_observations=observed, identity_agreement=identity_agreement, view_agreement=view_agreement,
                       pose_agreement=pose_agreement, conflicts=conflicts, measured=measured, judgment_source='agent review; user acceptance not implied')
        key = self.store.put('reference_review',payload)
        latest = dict(item['data'].get('latest_reviews', {}), **{str(output_index):key})
        rejected = sorted(set(item['data'].get('rejected_outputs', []) + ([str(output_index)] if disposition == 'rejected' else [])))
        updated = self.ledger.update(job, item['revision'], 'completed', {'latest_reviews':latest, 'rejected_outputs':rejected}, allowed={'completed'})
        return {'review':key, 'workflow_revision':updated['revision'], **payload}

    def qualify(self, job, output_index, native_state, source_anchors, native_anchors, pose, region, source_pose, role,
                maximum_anchor_error, support, expected_registry, review):
        item = self.ledger.read(job)
        if item['status']!='completed' or item['intent']['kind']!='mesh':
            raise ValueError('Completed mesh job required')
        if item['data'].get('guide_rejection'):
            raise ValueError('Reconstructed guide was rejected; preserve it as evidence and use a new reconstruction')
        reviewed = self.require_usable_review(review)
        if item['intent']['reviewed_source'] != review or reviewed['disposition'] not in ('useful','partial'):
            raise ValueError('Mesh does not descend from the reviewed image')
        self.store.get(native_state,'state')
        if source_pose != pose:
            raise ValueError('Source pose does not support requested pose; Neutral cannot stand in for a posed target')
        if not region or not support or not expected_registry or role not in ('pose_target','construction','partial_target'):
            raise ValueError('Region support, registry precondition and explicit target role required')
        if maximum_anchor_error <= 0:
            raise ValueError('Positive registration error bound required')
        asset = item['data']['outputs'][output_index]
        path = self.store.resolve_blob(asset)
        with np.load(path,allow_pickle=False) as f:
            arrays = {k:f[k] for k in f.files}
        geometry.validate(arrays)
        a,b = np.asarray(source_anchors,float),np.asarray(native_anchors,float)
        if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=3 or len(a)<3 or not np.isfinite([a,b]).all():
            raise ValueError('At least three finite paired XYZ anchors required')
        ac,bc=a-a.mean(0),b-b.mean(0)
        if min(np.linalg.matrix_rank(ac),np.linalg.matrix_rank(bc))<2:
            raise ValueError('Registration anchors must not be collinear')
        u,s,vt=np.linalg.svd(ac.T@bc)
        signs=np.ones(3);signs[-1]=np.linalg.det(u@vt)
        rotation=u@np.diag(signs)@vt
        scale=float(np.sum(s*signs)/np.sum(ac*ac))
        if scale<=0:raise ValueError('Invalid registration scale')
        translation=b.mean(0)-scale*a.mean(0)@rotation
        residual=np.linalg.norm(scale*a@rotation+translation-b,axis=1)
        if residual.max()>maximum_anchor_error:raise ValueError('Registration exceeds declared anchor tolerance')
        transform=np.eye(4);transform[:3,:3]=scale*rotation.T;transform[:3,3]=translation
        registered=dict(arrays,co=scale*arrays['co']@rotation+translation)
        coverage=[]
        for srow in support:
            if not srow.get('meaning') or not srow.get('correspondence_evidence'):
                raise ValueError('Each region requires semantic meaning and correspondence evidence')
            ids=geometry.select_triangles(registered,srow.get('selection'))
            if not len(ids):raise ValueError('Declared guide region has no supporting triangles')
            coverage.append(dict(srow, triangles=len(ids)))
        directory=self.store.root/'registered';directory.mkdir(exist_ok=True)
        dest=directory/(digest(canonical([asset['sha256'],transform.tolist()]))+'.npz')
        if not dest.exists():
            import io
            buf=io.BytesIO();np.savez_compressed(buf,**registered);atomic_write(dest,buf.getvalue())
        payload=dict(job=job, review=review, source=asset, registered=self.store.blob(dest), native_state=native_state,
                     source_state_id=self.store.get(native_state,'state')['source_state_id'], pose=pose, source_pose=source_pose,
                     region=region, role=role, support=coverage, expected_registry=expected_registry,
                     source_to_world=transform.tolist(), anchor_residuals=residual.tolist(),
                     limitations='Anchor registration and declared surface support are measured; hidden anatomy and generated identity still require review.')
        key=self.store.put('guide_qualification',payload)
        return dict(qualification=key, **payload)
