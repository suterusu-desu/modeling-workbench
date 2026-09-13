# Modeling Workbench

Reusable numerical modeling tools and an evidence-based character workflow. The package links geometry, exact-view observations, target qualification, recoverable interventions and retained methods. It contains no character assets or project history.

## Install and start a private workspace

Use Python 3.12 or newer in a virtual environment:

```sh
python -m pip install .
python -m modeling_system.init_workspace /path/to/private-project --character "Example character"
python -m modeling_system --workspace /path/to/private-project inspect
python -m modeling_system --workspace /path/to/private-project decision_workspace
```

Keep the private project outside the tools checkout. The initializer creates current-state, decision, method and lesson records, a workspace binding and an unauthorized provider policy. Populate actual identity references, assets, object/control/semantic mappings, units and scopes there. No named character or native adapter is inherited.

Read [the complete retained method](method.md), then [episodes and recovery](episodes.md). Native modeling requires the [adapter contract](native-adapters.md). The generic core performs historical inspection, numerical analysis, retention and recovery without Blender; it does not claim turnkey live control or modeling quality on every rig. A usable live adapter and a qualified character binding are separate requirements.

## Agent interfaces

Python, the JSON-file CLI and MCP use the same service. Instantiate `ModelingService(workspace=...)` explicitly, or launch `modeling-workbench-mcp --workspace /path/to/private-project`. Keep one native owner. Begin with `inspect_situation`, `decision_workspace` and the bound method; read the capability and runtime reports before assuming an operation is available natively.

For a Codex plugin, install the package in the chosen interpreter first, then materialize local configuration outside this checkout:

```sh
python -m modeling_system.prepare_plugin /path/to/local-plugin --workspace /path/to/private-project
```

Use the normal Codex plugin installation flow for that materialized directory. The tracked plugin source has no machine paths; the generated local configuration binds the interpreter and private workspace. Installation does not migrate an existing Blender runtime. Verify the installed launcher, then actual operator use, before changing an active project.

## Sharing and transfer

`export_package(output_path)` with no episode and no evidence produces tools only. It never reads a private workspace to construct its archive. The source tree and its synthetic tests can also be distributed directly.

An explicit episode or `include_evidence=True` selects a **private workspace transfer**. It can include identity, authority, original paths, operation history and assets. Such archives belong with the private project and are not shareable source releases. Do not upload them to the tools repository.

Run the synthetic suite with `python -m unittest discover -s modeling_system -t .`. It verifies contracts and recorded workflows; it is not an artistic evaluation on a new character.
