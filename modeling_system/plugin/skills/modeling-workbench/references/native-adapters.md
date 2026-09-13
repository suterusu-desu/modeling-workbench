# Native adapter boundary

The reusable package owns recorded geometry, numerical queries, source lineage, decisions, response analysis, operation journals, episode leases and transport schemas. A private workspace owns its native scene/rig integration. No specific scene names, controls, vertex identities, coordinate scale, guide registry or external helper paths ship in the tools repository.

Set `native_adapter` to `blender_json_v1` only after the private adapter is ready. `native_configuration` contains `entrypoint: {path, sha256}`, `dependencies: [{path, sha256}, ...]`, and `operations: [...]`. Paths resolve relative to the workspace. All executable dependencies must be pinned. The default is `unconfigured`; unsupported operations fail explicitly.

The entrypoint exports `execute(operation, arguments)` and returns the standard `{ok: true, result: ...}` or `{ok: false, error: {code, message, recovery, details}}` envelope. It runs inside Blender through the existing local JSON bridge. It receives `_workspace` and `_owner`; the public arguments must conform to `operation_schemas()` in `native_bridge.py`. Adapters are trusted local code, not a security sandbox. The dispatcher verifies pinned input bytes and advertised operations; it does not establish anatomical correctness or implement an adapter's native invariants on its behalf.

An adapter must implement and independently verify these invariants before advertising mutation operations:

- `inspect_live` reports source content and expected-state identity, current saved file, dirty state, actual coverage and dependencies without silently making historical geometry fresh.
- A single persistent editing owner guards native changes, including pose, view and state-changing inspection. Refuse stale expected-state tokens and conflicting owners.
- `prepare_state` exports complete declared geometry with topology/material/component/semantic identities and inclusion/exclusion coverage. Native snapshots follow the scene importer contract in `workbench.py` and synthetic fixtures.
- `capture_view` binds images, exact camera matrices/crop, native geometry, guide/depth/sections, context and pose to one state. Projected coordinates alone do not establish visibility.
- `apply_proposal` checks scope and protected objects/datablocks, pins exact scripts and constraints, saves recovery state before mutation, and independently checks resulting geometry. An uncertain response must be reconciled, never blindly retried.
- Guide integration requires a distinct qualified identity, supported regions/exclusions, current source state, explicit pose/control mapping, reviewed source/derivation/registration/art evidence, an expected registry hash, recoverable registry update and exact installed geometry/binding verification.
- Rollback refuses to erase later edits; saved checkpoints preserve previous files and verify actual bytes. Reopen uses an isolated Blender profile through an explicitly supplied local runner.
- Motion capture independently records actual ordered poses and dependencies. Do not substitute reversed frames or equal timestamps for measured reopening or reviewed video correspondence.

A private adapter may compose existing verified native code behind a small entrypoint. Keep the parent project and its evidence local. The source package deliberately does not advertise successful native behavior for a merely configured adapter. Use contract tests, a disposable synthetic Blender scene, then a bounded owner-controlled real-workspace trial to establish adoption. Never switch an active project's runtime solely because the generic package was updated.

Current limitation: independent locally repaired guide lineage/review is retained via explicit episode evidence and a qualified advanced native transaction. There is not yet a typed core derivative-admission lifecycle. Preserve the original raw rejection and qualify the new asset independently.
