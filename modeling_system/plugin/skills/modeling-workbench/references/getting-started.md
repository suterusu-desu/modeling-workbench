# Modeling Workbench

Reusable numerical modeling tools and an evidence-based character workflow. The package links geometry, exact-view observations, target qualification, recoverable interventions and retained methods. It contains no character assets or project history.

## Install and start a private workspace

Clone or download this repository first. In its root directory, use Python 3.12 or newer in a virtual environment. On Windows, `py -3.12` (or a newer installed version) can replace `python` in the first command:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` on macOS/Linux. Then:

```sh
python -m pip install .
python -m modeling_system.init_workspace /path/to/private-project --character "Example character"
python -m modeling_system --workspace /path/to/private-project inspect
python -m modeling_system --workspace /path/to/private-project decision_workspace
```

Replace `/path/to/private-project` with a new empty directory outside the repository; quote paths containing spaces. Initialization reports `native_adapter: unconfigured`. The first inspection reports `needs evidence` with no selected modeling question; the decision workspace has `total: 0`. These are expected first-run results. These commands do not launch Blender or submit generation jobs.

For a small numerical check with no character assets, accounts or Blender, run:

```sh
python -c "import numpy as np; from modeling_system.geometry import nearest_surface; a={'co':np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),'tri':np.array([[0,1,2]],dtype=np.int32)}; print(nearest_surface(a,[0.25,0.25,1.])['point'])"
```

The result should be `[0.25, 0.25, 0.0]`: the closest point on a synthetic triangle. This verifies a basic numerical operation, not native Blender integration.

Keep the private project outside the tools checkout. The initializer creates current-state, decision, method and lesson records, a workspace binding and an unauthorized provider policy. Populate actual identity references, assets, object/control/semantic mappings, units and scopes there. No named character or native adapter is inherited.

Read [the complete retained method](method.md), then [episodes and recovery](episodes.md). Native modeling requires the [adapter contract](native-adapters.md). The generic core performs historical inspection, numerical analysis, retention and recovery without Blender; it does not claim turnkey live control or modeling quality on every rig. A usable live adapter and a qualified character binding are separate requirements.

## Prepare your first character

1. Put your chosen identity artwork in `references/` and untouched source assets in `originals/`. Record which reference governs identity in `PROJECT.md`; put scope choices and rejected directions in `DECISIONS.md`.
2. Choose one modest first goal, such as checking a neutral head's proportions. Record what should improve, what must stay intact and which views can reveal a regression.
3. Create or import a base mesh in Blender through your existing modeling workflow. Save the original and a working checkpoint. The workspace initializer does not generate a base mesh or rig.
4. Before asking the workbench to control Blender, implement or supply the [native adapter](native-adapters.md), pin its dependencies, establish scene units and object/control/region bindings, and verify it on a disposable scene. A general Blender MCP connection alone does not establish this package's adapter contract.
5. Record the actual baseline mesh and matching views. Review existing references or prepare a new reference through an external image service. If a 3D target is needed, reconstruct it through your configured mesh-generation route and qualify its actual coverage before fitting.
6. Run one recoverable, guide-constrained trial. Compare the affected region, wider character and relevant motion. Save the result and the supported, rejected or unresolved lesson.

If you do not yet have an adapter, you can use the method manually in Blender and explore the offline core, but automated native character creation remains an integration task. Do not change `native_adapter` merely to bypass an unconfigured error.

## Generation services and costs

Image generation supplies proposed identity/pose/detail references; mesh generation supplies candidate 3D guides; video generation can supply motion-reference clips. Each result has a different review stage. Attractive pixels do not qualify reconstructed depth, and a generated video is not proof that a rig deforms correctly.

Bring existing references or choose external services and transports appropriate to those stages. The workbench provides preparation, review, job reconciliation and workflow records, not bundled image/video models or provider dispatch adapters. A new workspace authorizes no generation. Configure and authorize the actual provider route privately before any dispatch. Tripo's documented route uses Studio; image and video providers are not mandatory fixed dependencies. No universal end-to-end provider compatibility is claimed.

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
