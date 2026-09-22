# Portable installation

The wheel and tools-only archive contain the core, controller, direct TypeSafe
transport, shared provider accounting/recovery, both agent skills and their
references. No private project or previous conversation is needed to load them.
External inputs are Python, Blender and its verified local bridge, your accounts,
and the character's references, scene bindings and qualified adapter.
Tool portability does not establish native qualification on a new rig.

## Install and check

Use Python 3.12 or newer on Windows, macOS or Linux. Create and activate a virtual
environment, then run `python -m pip install .` from the source checkout or
extracted tools archive. Alternatively install the built wheel. Recreate virtual
environments and generated MCP configuration on each machine; do not copy their
absolute paths. Dependencies resolve platform-appropriate packages.

```sh
python -m modeling_system.init_workspace /path/to/character --character "My character"
python -m modeling_system.setup_check --workspace /path/to/character
python -m modeling_system.prepare_plugin /path/to/modeling-workbench --workspace /path/to/character
python -m modeling_system.setup_check --workspace /path/to/character --plugin /path/to/modeling-workbench
```

Quote paths containing spaces. The initializer preserves existing files and
creates a private workspace with `AGENTS.md` and complete skills under
`.agents/skills/`. The generated plugin also contains both skills. Clients that
do not discover project-local skills should read those two `SKILL.md` files
explicitly. Use one discovery route per client to avoid duplicate registration.
No second npm or Claude plugin installation is needed.

The official TypeSafe skill is bundled unmodified with its MIT license and
commit/file hashes in `skill-dependencies.json`. Its targeted live-docs guidance
still applies. The doctor checks those bytes without downloading anything.
Core readiness, provider configuration and native readiness are separate:
the doctor never reads credential files or contacts Blender or TypeSafe.

## Jev credentials and accounting

Set `TYPESAFE_API_KEY_FILE` in the launch environment to your external credential
file, or set `TYPESAFE_API_KEY` directly. File configuration takes precedence.
The file accepts a single key or `TYPESAFE_API_KEY=...`. There is no default user
path. Restart the client/launch process as needed to inherit the variable. Never
put a key in a prompt, command argument, workspace binding or archive.

Reuse existing provider accounting. For a **new** workspace with actual user
authorization, initialize its ledger once:

```sh
python -m modeling_system.provider_dispatch --init-ledger /path/to/character/runtime/jev-ledger --authority "User-authorized modeling with existing account credits"
python -m modeling_system.setup_check --workspace /path/to/character --ledger /path/to/character/runtime/jev-ledger
```

The text records an existing instruction; it does not obtain permission. Normal
use adds no invented lifetime request or monetary cap. Optional `--max-requests`
and `--max-cost-usd` record actual user limits. No purchase/topup occurs. Original
responses, reservations, reported usage estimates, bounded retries and uncertain
attempts stay in the same ledger. Initialization refuses to overwrite it.

Use the packaged integration instead of a private transport script:

```python
from modeling_system.jev_session import create_session

session = create_session(
    queue_directory,
    service=service, episode=active_episode, owner=sole_owner,
    goal={"objective": authorized_goal},
    observe_context=read_actual_revisions,
    catalog=qualified_catalog, handlers=qualified_handlers,
    public_projection=reviewed_public_projection,
    ledger_directory=existing_ledger,
)
session.run(max_steps=authorized_step_bound)
```

Callbacks follow the [operating-session contract](operating-session.md). They
describe this character and qualified capabilities. `planning_batch.decide`
uses the same transport for recurring judgments outside the action loop.
`jev_session` includes completed-response recovery, transient recovery and
explicit authority rebinding. Keep original receipts; never replay uncertain
native effects. Session creation does not start a goal or automation.

For composed saved-array scopes, `python -m modeling_system.cooperative_scopes MANIFEST` uses the same session and ledger. Its manifest binds `runtime_revision` from `runtime.source_manifest()`, workspace, queue directory, owner, episode, objective, exact source-scope hashes, ledger directory and the authorized step bound. Workspace is relative to the manifest; other data paths are relative to the workspace. Optional `authority_file`/`state_file` default to `AGENTS.md`/`PROJECT.md`. Completed tasks stay frozen, input changes block stale work, and this runner advertises no native handlers.

## Native and optional integrations

[Native adapter qualification](native-adapters.md) defines the scene entrypoint
and pinned dependencies. The current socket transport expects a local Blender
bridge accepting an `execute_code` JSON envelope on loopback (default port 9876).
Its successful response contains captured stdout in `result.result`; the workbench
extracts its framed operation result. A connection using another wire protocol
is insufficient. Supply and qualify that bridge and the scene-specific adapter. The packaged
[isolated Blender runner](isolated-blender.md) handles copied inputs, isolated
profiles and job receipts; qualify its use on the selected Blender runtime.
No universal rig adapter is bundled.

Blender's Python stays separate from the offline environment. Keep SciPy fitting
offline and pass pinned arrays to native operations. Do not insert an offline
environment's `site-packages` into Blender. Bind actual executable/runner paths
on each machine. Astra develops capabilities; Jev selects their native use.

Image, mesh and video generation are optional external transports. Existing art
needs none of those accounts. Each route retains its actual authorization, source
review and recovery semantics; no credentials or credits are bundled.

## Transfer and upgrades

Share tools-only exports. Character workspaces and explicit evidence transfers
remain private. Recreate the environment/plugin, restore private evidence through
the existing transfer mechanism, rebind external files, verify hashes, and qualify
the local native adapter. Saved absolute paths are not portable configuration.

Completed evidence remains historical. In-flight native/provider effects need
reconciliation on the original host; moving files does not grant a new owner or
permit replay. An existing live project keeps its pinned runtime until its owner
deliberately verifies and adopts a replacement. Upgrading tools does not overwrite
local skill customizations or silently switch a running character session.
