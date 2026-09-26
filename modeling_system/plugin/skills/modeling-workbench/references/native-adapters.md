# Native adapter boundary

The native bridge imports the selected installation under an isolated package namespace. An older `modeling_system` already loaded in Blender cannot redirect dispatch through its cached package path, and its modules are not reloaded by the new bridge. The bound private adapter must still preserve and verify actual ownership, native state and its own dependencies; namespace separation does not establish native capability or permission to switch a running study.

The reusable package owns recorded geometry, numerical queries, source lineage, decisions, response analysis, operation journals, episode leases and transport schemas. A private workspace owns its native scene/rig integration. No specific scene names, controls, vertex identities, coordinate scale, guide registry or external helper paths ship in the tools repository.

The package ships a generic reference adapter and a live bridge ([guides and display](guide-and-show.md)): a workspace
that describes its working mesh, guides with roles, controls and an optional restore hook can bind them with
`python -m modeling_system.live bind` instead of writing an adapter. A private adapter remains the route for richer
native operations.

Set `native_adapter` to `blender_json_v1` only after the private adapter is ready. `native_configuration` contains `entrypoint: {path, sha256}`, `dependencies: [{path, sha256}, ...]`, and `operations: [...]`. Paths resolve relative to the workspace. All executable dependencies must be pinned. The default is `unconfigured`; unsupported operations fail explicitly.

The entrypoint exports `execute(operation, arguments)` and returns the standard `{ok: true, result: ...}` or `{ok: false, error: {code, message, recovery, details}}` envelope. It runs inside Blender through the existing local JSON bridge. It receives `_workspace` and `_owner`; the public arguments must conform to `operation_schemas()` in `native_bridge.py`. Adapters are trusted local code, not a security sandbox. The dispatcher verifies pinned input bytes and advertised operations; it does not establish anatomical correctness or implement an adapter's native invariants on its behalf.

An adapter must implement and independently verify these invariants before advertising mutation operations:

- `inspect_live` reports source content and expected-state identity, current saved file, dirty state, actual coverage and dependencies without silently making historical geometry fresh.
- A single persistent editing owner guards native changes, including pose, view and state-changing inspection. Refuse stale expected-state tokens and conflicting owners.
- `prepare_state` exports complete declared geometry with topology/material/component/semantic identities and inclusion/exclusion coverage. Native snapshots follow the scene importer contract in `workbench.py` and synthetic fixtures.
- `capture_view` binds images, exact camera matrices/crop, native geometry, guide/depth/sections, context and pose to one state. Projected coordinates alone do not establish visibility.
- `apply_proposal` checks scope and protected objects/datablocks, pins exact scripts and constraints, saves recovery state before mutation, and independently checks resulting geometry. An uncertain response must be reconciled, never blindly retried.
- Guide integration requires a distinct qualified identity, supported regions/exclusions, current source state, explicit pose/control mapping, reviewed source/derivation/registration/art evidence, an expected registry hash, recoverable registry update and exact installed geometry/binding verification.
- Rollback refuses to erase later edits; saved checkpoints preserve previous files and verify actual bytes. Reopen uses an isolated Blender profile through an explicitly bound runner. The package supplies [isolated_blender](isolated-blender.md); scene/runtime qualification remains separate.
- Motion capture independently records actual ordered poses and dependencies. Do not substitute reversed frames or equal timestamps for measured reopening or reviewed video correspondence.

A private adapter may compose existing verified native code behind a small entrypoint. Keep the parent project and its evidence local. The source package deliberately does not advertise successful native behavior for a merely configured adapter. Use contract tests, a disposable synthetic Blender scene, then a bounded owner-controlled real-workspace trial to establish adoption. Never switch an active project's runtime solely because the generic package was updated.

## Native observations and export cost

Keep source verification separate from numerical scene export. Check actual
native content, external dependencies, controls, views and saved-file identity
on each guarded operation. Opening, posing, displaying and saving a reviewed
checkpoint need not rebuild a whole-scene array record. An adapter may defer
that export if its expected-state token still covers fresh native dependencies;
report the numerical cache as dirty/unavailable and omit a current geometry ID.
Numerical queries and explicit synchronization must rebuild and verify that cache
before use. Keep requested depth/section refreshes, binding checks and recovery.

Within one synchronous guarded call, rollback preparation may reuse the guard's
content manifest only when no intervening native effect or asynchronous yield
occurred. Consume that snapshot before effects; never reuse it for the returned
post-effect state, a later request or a different operation. A dirty bit or an
unchanged file hash alone does not establish live content freshness.

`native_timing.NativeTimings` records nested adapter phases without importing or
calling Blender. Wrap the guard, rollback save, file open, bootstrap and returned
observation separately; keep parent and child durations distinct. An optional
`sink` persists progress through the adapter's existing receipt writer. Failed
telemetry does not change the operation outcome or authorize replay. Measure the
next useful operation; do not rerun a retained edit solely for a timing sample.

## Parts set aside from view

`set_display` and a retention's `display` accept `hide`: the names of parts a
review sets aside (for example lashes while the lids are worked on). It is
presentation only. The adapter hides exactly the listed parts in the viewport,
shows again the parts an earlier call hid that are no longer listed, refuses a
name it does not know instead of ignoring it, and reports the current list in its
live state so the expected-state token covers what the viewer sees. Hidden parts
stay in the file with their geometry and bindings and keep following the rig;
trial renders set parts aside through their own arguments. A retention that
declares `hide` saves the checkpoint with those parts hidden, so whoever opens it
sees the reviewed view. In real use a live session kept showing the lashes after
every retention while the reviews hid them, so the visible work did not match the
reviewed one. A part set aside is not judged and not removed; name it in every
review that relied on the view.

## Operator wrappers and worker completion

Read the selected adapter's actual result contract before asserting success. Transport success, operation disposition, saved-artifact verification and review acceptance are separate facts. Do not require a universal `status == "completed"`: an adapter may return an operation-specific status such as `applied_trial` or `saved`. Compact results can point to a durable receipt instead of embedding candidate details. Expand the returned receipt and verify the linked file bytes and operation identity before continuing. An unexpected wrapper assertion after dispatch does not establish that the native action failed; reconcile the original operation before considering any retry.

A custom reopen worker must satisfy the explicitly selected runner's output contract as well as its own assertions. Confirm required output names and contents before dispatch. When the runner requires `inventory.json`, writing only arrays or a separate report is insufficient even if Blender exits with code zero. Keep process exit, required-output completeness, source preservation and geometry verification visible separately. Missing metadata does not prove that the underlying saved geometry is wrong, and a successful numerical check does not make an incomplete launcher receipt successful.

Preserve an original failed launcher receipt. Any independent verification must be a new record linking the original receipt and exact output hashes, with the remaining limitation stated. Do not fabricate a historical inventory or rerun native effects merely to clear a wrapper error. `inspect_execution_receipt` and `worker_contract.WorkerContract` implement explicit adapter-specific receipt inspection and required-output completion; read [execution contracts](execution-contracts.md). The helper enforces its declared contract, not an arbitrary private runner's undocumented schema.

For bounded topology batches, native lineage, later-save-aware rollback and measured library append, read [isolated native transactions](native-transactions.md). These owner-run helpers complement existing qualified workspace operations; they do not take over live Blender or silently widen content support.

Test private adapters in a clean Python namespace after a full Blender restart. A warm process may hide a canonical-package import in a nested recorder or helper. Audit the entire imported dependency closure, including secondary callbacks, and inject explicitly pinned functions through a qualified binding. Verify source paths, hashes and callback module identity before use. Do not fix an isolated adapter by aliasing a different implementation to the canonical package name. Preserve failed attempts and prove unchanged native content after recovery.

Current limitation: independent locally repaired guide lineage/review is retained via explicit episode evidence and a qualified advanced native transaction. There is not yet a typed core derivative-admission lifecycle. Preserve the original raw rejection and qualify the new asset independently.

## Mirror source inventory

`mirror_inventory.mirror_structure(modifier, object_reference=...)` records the
native Mirror axes, bisect/flip settings, clipping/merge thresholds, UV settings,
vertex-group behavior and exact mirror-object identity. The resolver must identify
the Blender Object and its linked library; `None` means the modifier uses its own
object origin. The adapter must also retain the source and mirror object's actual
transforms, parents, constraints and animation in its scene dependency inventory.
Fields are checked against native Mirror RNA; missing or unsupported new settings
fail explicitly. See the [Blender Mirror API](https://docs.blender.org/api/5.3/bpy.types.MirrorModifier.html).

This is source description, not a topology qualification. Mirror can duplicate,
bisect and merge geometry; do not add it to a connectivity-preserving suffix list
or silently reenroll attachment indices. Preserve modifier order and separately
qualify evaluated correspondence. A live adapter upgrade may refresh its pinned
Python inventory cache under the owner check; that does not authorize reopening,
rebuilding or changing the scene.
