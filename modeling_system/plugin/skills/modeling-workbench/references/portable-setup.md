# Portable setup

The wheel and tools archive contain the core, direct owner session, complete
workbench skill and references, numerical operations and isolated Blender runner.
No external inference service, credentials or second skill installation is needed.

## Install and check

Use Python 3.12 or newer on Windows, macOS or Linux. Create and activate a virtual
environment and run `python -m pip install .` from the source checkout or extracted
tools archive, or install the built wheel. Recreate virtual environments on each
machine; do not transfer their absolute executable paths.

```sh
python -m modeling_system.init_workspace /path/to/character --character "My character"
python -m modeling_system.prepare_plugin /path/to/modeling-workbench --workspace /path/to/character
python -m modeling_system.setup_check --workspace /path/to/character --plugin /path/to/modeling-workbench
```

Quote paths containing spaces. The initializer creates private material folders,
authority records and complete `.agents/skills/modeling-workbench` guidance. The
generated plugin includes the same skill. If a client does not discover local
skills, read that `SKILL.md` explicitly. Use one discovery route to avoid duplicate
registration. Existing files and customized skills are never overwritten.

The doctor checks core imports, bundled guidance, plugin launch and pinned native
configuration. It never contacts Blender or an inference service and does not
read credential files. Configured native dependencies are not live verification.

## Operate

Change native files with [trial, reopen and retain](trial-reopen-retain.md). Use
[direct sessions](operating-session.md), `ModelingService`, or the installed CLI/MCP
service for the rest. `direct_session.create_session` requires preservation policies
for affected appearance edits and retention; the owner supplies explicit qualified
operations and choices. No provider ledger or public prompt projection is needed.
Session creation does not start a goal or automation.

## Native and optional integrations

Supply Blender and a [qualified scene adapter](native-adapters.md). The socket
transport expects a local bridge accepting `execute_code` JSON on loopback
(default port 9876), returning captured stdout in `result.result`. The workbench
extracts its framed operation result. Another bridge protocol is insufficient.
Pin the entrypoint and dependencies and qualify the actual local rig. The bundled
reference adapter and live bridge cover the common loop (inspect, open, pose by
custom properties or shape keys, a guide overlapped as a blue wire, save), bound
with `python -m modeling_system.live bind` and started with `... live launch`;
a rig that needs setup after loading supplies a restore hook. See
[guides and display](guide-and-show.md).

The [isolated runner](isolated-blender.md) copies inputs, isolates profiles and
records jobs. Blender's embedded Python is separate from offline preparation:
never add the offline environment's entire `site-packages` to a native worker.
Keep heavy fitting offline and pass pinned arrays to native operations.

Image, mesh and video generation are optional external integrations with separate
authorization, sources and recoverable job accounting. Credentials and credits
are never bundled. Existing art requires none of those accounts.

## One environment per workspace

Keep one virtual environment per character workspace with an editable install of its own clone of this repository:
`python -m venv .venv`, then `.venv` Python `-m pip install -e <clone>`. Upgrading is `git pull` in that clone (reinstall
only when dependencies change); there is no new environment per release. Use a clone the workspace owns, not a checkout
someone is editing: the preservation checks pin the installed files by hash, so an edit landing in the middle of a trial
refuses its measurement. Pull between trials.

## Transfer and upgrade

Share tools-only exports. Restore character assets/evidence separately, recreate
the environment and plugin, rebind local external files, verify hashes and qualify
the native adapter. Moving files does not establish current live state or transfer
a running native lane. Reconcile uncertain effects on the original host.

Old evidence, provider ledgers and checkpoints remain readable history. They are
not executable plans. Start direct work in a new session directory and explicitly
bind current operations, arguments, conditions and preservation; old inferred task
definitions are rejected instead of silently executed. Keep installed runtime,
native adapter and actual operator adoption receipts distinct.
