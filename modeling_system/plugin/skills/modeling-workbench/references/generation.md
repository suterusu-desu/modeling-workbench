# Generation decision and recovery

Read the bound workspace policy and `operation_context` before selecting a route. The default is general Smart Mesh/P2.0. The user preference for `exact_view_enlarged_local_detail` selects HD/v3.1 as a starting candidate; it grants no new job, spend or scope. Do not pass a use-case label as an authorization.

The completed source-bound comparison is evidence. Read its actual job/source/output identities and actual topology separately from requested settings. The selected browser asset can be older than the submitted job. Recover the same uncertain job; do not create another key to replace a charged or completed slot.

For another HD dispatch the policy must contain an applicable separately authorized source/experiment configuration. Existing guards re-read it at preparation and claim. A source must have a current usable likeness/view/pose review before reconstruction. Unknown hidden depth is a legitimate reconstruction question, not a reason to demand completed depth qualification before reconstruction.

Ordinary local guide reconstruction is a distinct intent class, not a comparison. Select it with `settings.intent_class='local_guide_reconstruction'` and `settings.reconstruction_id` referring to an authorized entry in `tripo.authorized_local_reconstructions`. The entry binds `use_case='exact_view_enlarged_local_detail'`, Studio website transport, HD/v3.1, source image job, current usable `review_id`, actual source SHA, a new fixed `idempotency_key`, explicit `output_settings` including topology/polycount, `max_existing_credit_cost`, and existing authorization scope/source. Retain actual selection evidence and exclusions with the entry. A pending entry or a local-HD preference alone does not authorize preparation/claim.

Pass the same use case, `transport='Tripo Studio website'`, route, output settings and observed generation cost in preparation, with the configured authorization scope/source and fixed key. The live claim preflight must identify the source SHA and every configured output setting as well as fresh mode/model/time/cost/balance. Preparation and claim compare actual stored image bytes and the selected review/job; claim also rechecks the full authorization snapshot. Changes require explicit reconciliation of the same prepared job, not a new retry key. Neither a comparison ID nor a generic P2 request can be relabeled to consume this ordinary local authorization. No API transport, funding, purchase or expanded character scope is supplied by this contract.

## Multi-view mesh jobs

A mesh reconstructed from one image invents the unseen sides; a provider's multi-view mode takes separate images of
the same pose in named slots. Bind them with `prepare_guide(front_review, provider, settings, ...)` where
`settings['views'] = {'front': front_review, 'left': ..., 'right': ..., 'back': ...}`. Each slot is a currently usable
image review (an accepted existing drawing enters as an image job whose outputs are those files, reviewed like any
other source); `front` must be the request's reviewed source; every slot needs its own image, so the same picture in
two slots is refused. A workspace policy with

```json
"tripo": {"multi_view_requirement": {"required": true, "minimum_distinct_views": 2}}
```

refuses a mesh request with fewer distinct reviewed views, at preparation and again at claim, so a job prepared before
the rule existed cannot be dispatched after it (cancel it). The prepared transport lists `view_slots`, the exact files
to load. `claim_job` re-checks every slot's review and bytes, and the live preflight must report `slot_sha256` for
exactly the prepared slots, read from what the panel actually holds; `settings['output_settings']` (for example
`{"topology": "Quad", "polycount": 25000}`) must match the panel's values and types. A later rejection of any view's
review blocks the claim.

After delivery inspect actual connected skin, registration, depth/sections, ocular relationships and exclusions. Missing foreground skin is the applicability of the complete-skin preparation lesson, not a universal crop prohibition. Useful interior support may coexist with invented backs or brows. Keep rejected and unresolved regions explicit.

`runtime_status` reports observed process source and method signatures. A current disk hash does not update an already loaded process. New plugin materialization/cache, a fresh service process/schema, and the operator actually reading/using this procedure are distinct adoption evidence.

`operation_context` and `select_generation_route` perform no local store writes. For a concrete decision that needs retained provenance, call `capture_operation_context` with the same stage/context/job; it pins exact authority bytes and automatically journals the explicit retention action.
