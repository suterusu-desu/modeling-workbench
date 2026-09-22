# Modeling Workbench

Astra and Jev operate one evidence-based workbench: Astra maintains character
intent and reviews actual images; Jev chooses useful modeling work; qualified
capabilities execute it and feed their results into the next decision.
The [operating session](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md)
links the existing queue, tools and native adapter through one recoverable episode.
See the [system design](DESIGN.md) for responsibilities and retained workflows.

Actual visual reviews and conditional lessons carry into later Jev decisions
automatically. Their exact evidence stays in the private workspace; compact
context distinguishes observed failures, changed prerequisites and remaining
uncertainties. See [retained learning](modeling_system/plugin/skills/modeling-workbench/references/retained-learning.md).

Reusable tools and an evidence-based workflow for character modeling: exact recorded geometry, matched observations, guide/depth qualification, coupled response analysis, recoverable trials and retained learning.

[Start here](modeling_system/plugin/skills/modeling-workbench/references/getting-started.md) | [Retained method](modeling_system/plugin/skills/modeling-workbench/references/method.md) | [Native adapters](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md) | [Agent skill](modeling_system/plugin/skills/modeling-workbench/SKILL.md)

The tools repository contains source, generic methods and synthetic tests. Characters have separate private workspaces containing assets, identity references, rig/object/semantic bindings, current decisions and full experiment history. `python -m modeling_system.init_workspace PATH --character LABEL` creates that boundary explicitly.

Python, CLI and MCP expose the same core. Native Blender operations require a separately configured and verified workspace adapter. This package preserves the modeling process and its contracts; it does not claim identical results or validated native control on every character. Broader quality requires actual trials across different characters.

Tools-only export is the default. Explicit workspace/evidence transfers remain private. See [development and contribution boundaries](DEVELOPING.md).

## How Astra and Jev work together

- **Astra** maintains character intent, interprets images, resolves difficult modeling questions and develops new capabilities.
- **Jev** selects actions and recurring planning choices from qualified capabilities: priorities, methods, evidence collection, repair and recovery.
- **The workbench** executes selected capabilities through one native Blender lane and retains dependencies, results and review feedback in a recoverable operating session.

The [persistent controller](modeling_system/plugin/skills/modeling-workbench/references/controller.md) supports sustained work with bounded budgets, asynchronous planning, compatible action batches and per-stage timings. The private adapter supplies native operations and the Blender status display. Saved findings and Astra's review feedback inform subsequent choices; visual acceptance remains separate from numerical checks.

The [TypeSafe integration](modeling_system/plugin/skills/modeling-workbench/references/judgments.md) provides typed Jev judgments through the direct API. Your private workspace supplies credentials, usage accounting and the qualified action catalog. See the [operating-session contract](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md) to connect the controller, queue, adapter and evidence records.

The [candidate pipeline](modeling_system/plugin/skills/modeling-workbench/references/candidate-pipeline.md)
reuses measured guide sections and material paths, preserves explicit interpolation
provenance, and continues selected candidates through verification and reviewed
retention. Session accounting separates inference, native execution and reported
Astra work so optimization follows observed bottlenecks.

[Executable methods](modeling_system/plugin/skills/modeling-workbench/references/method-control.md)
give Jev reusable alternatives for preparation, diagnosis and recovery, with
automatic retrieval of relevant failures. Saved-array recipes expose prepared
effect size, active geometry coverage and pose correspondence. Timed Astra work
and actual Jev choices remain visible in session metrics.

[Parameterized operations](modeling_system/plugin/skills/modeling-workbench/references/operation-recipes.md)
reuse guide-landmark fitting, material motion and native job/retention handlers.
Jev chooses among applicable data-bound methods; standalone native operations
retain an action/defer choice. Retention uses returned fresh state to avoid
duplicate scene inspections and skips reopening an already loaded exact candidate,
while preserving native guards, independent reopen evidence and actual visual review.

[Preparation and scoped setup](modeling_system/plugin/skills/modeling-workbench/references/preparation-and-bootstrap.md)
check declared dependencies and array shapes, preserve numerical results, and
keep long reports recoverable without repeating completed work. Complete projected
triangle contact uses contained interpolation and exact fallback for thin
fragments, catching intersections missed by sparse samples. Qualified native
jobs can replace broad scene exports with explicit feature checks and record
setup, execution and evidence timings; actual savings depend on the workload.
Native job configuration is checked before Jev selection, so missing runner
fields can be corrected without spending a decision or opening Blender.
Adapters can separate fresh native-state checks from costly numerical exports
and use nested call timings to locate slow checkpoint handovers. Immutable
preparation results stay reusable when task inputs exclude their own output logs.

[Typed capability control](modeling_system/plugin/skills/modeling-workbench/references/typed-control.md)
lets Jev select qualified operation arguments alongside the operation itself.
When relevant experience exceeds the initial context, it selects useful sources
before the action decision. Named uncertainty, optional grounded claim checks,
and outcome-linked decision records keep choices connected to actual modeling
evidence. Explicit decision budgets and provider retry timing use the existing
workspace accounting and recovery path.

The response reader validates provider answers while preserving their original
values. See the [reader compatibility limits](modeling_system/plugin/skills/modeling-workbench/references/judgments.md)
for its narrow handling of rounded Choice probabilities.

## What can I use today?

This is an experimental workbench for technical artists and developers building characters with an AI assistant. You can create a private character workspace, inspect recorded geometry, compare surfaces, qualify reference support, organize recoverable experiments, and reuse supported diagnostic recipes.

**It is not yet a turnkey image-to-rigged-character application.** Live Blender work needs a verified adapter for your scene and rig; a universally usable starter adapter is not supplied. A fresh workspace starts with native operations unconfigured. General tools and synthetic checks do not establish finished character quality on your assets.

The modeling loop is:

1. Choose the character's identity references and a bounded modeling question.
2. Record the working mesh, pose and matching views.
3. Review or create a target; register a 3D guide and qualify its supported regions.
4. Fit the working mesh under guide/depth constraints in a recoverable trial.
5. Check multiple views, relevant motion and neighboring surfaces; retain the result and what was learned.

An assistant can use the same core through Python, CLI or MCP. The method still requires artistic judgment, meaningful references and verified native integration.

## Dependencies

| Dependency | When needed | Included here? |
| --- | --- | --- |
| Python 3.12+ | Install and run the core | Install separately; use a virtual environment. |
| NumPy, SciPy, Pillow, MCP SDK, imageio-ffmpeg | Package runtime | Installed by `pip install .`; exact allowed ranges are in [pyproject.toml](pyproject.toml). |
| Blender and a qualified workspace adapter | Inspect or edit a live character | Blender, the bridge setup and your rig/scene bindings are separate. See [native adapters](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md). |
| TypeSafe API access | Jev-driven action selection and planning | Supply your own credentials and account-credit authorization in the private workspace; offline core use does not require it. |
| An AI assistant / MCP client | Agent-operated workflow | Bring your own client, model access and applicable subscription. CLI/Python use does not require an AI account. |
| Image generation or editing service | Create new character/pose/detail reference images | Optional external service. Bring a supported transport and account; existing artwork can also supply references. No image model or credits are bundled. |
| Tripo Studio | Reconstruct generated 3D guides in the documented Tripo route | Optional external service with your own account and authorized credits. The package records and checks jobs; it does not operate the website by itself. |
| Video generation service | Create new motion-reference clips | Optional external service and transport. Existing reference video can be analyzed without generating another clip. No video model or credits are bundled. |

Tripo, image generation and video generation are **workflow integrations**, not prerequisites for installing or trying the offline core. The package does not automatically connect accounts, submit provider jobs or purchase credits. Provider transport, source review, permitted settings and spending scope belong in the private workspace. Tripo Studio and Tripo API use separate credit systems; a Studio subscription does not fund API calls. See [Tripo's explanation](https://www.tripo3d.ai/help/api-plugins/tripo-studiotripo-api) and the [generation contract](modeling_system/plugin/skills/modeling-workbench/references/generation.md).

## Start a character

Follow [the setup and first-character guide](modeling_system/plugin/skills/modeling-workbench/references/getting-started.md). It includes a no-generation first run and explains what must be supplied before native modeling can begin. Keep the character workspace outside this source checkout.

## Acknowledgments and design references

We studied these adjacent projects while developing the workflow and reusable tools:

- [Blender Research MCP](https://github.com/Haiyang-Bian/blender-research-mcp), reviewed at [`d623c4d`](https://github.com/Haiyang-Bian/blender-research-mcp/tree/d623c4d116716f68b78fed77be93d42548247303): revision-bound scene evidence, explicit component lineage, bounded native operations, boundary diagnostics and recovery-aware workflows informed our design study.
- [Blender Terracotta](https://github.com/ShamanAndrey/blender-terracotta), reviewed at [`8fda74e`](https://github.com/ShamanAndrey/blender-terracotta/tree/8fda74ec41ca43cfb6b34e86f9d11f43127391ac): graph-based workflow composition, generation cost previews and completed-result reuse informed our reusable recipe work. Applicability and evidence review remain explicit in our implementation.

These are learning acknowledgments, not claims of endorsement or equivalent capabilities. They are design references rather than required runtime dependencies. Follow each upstream project's own licensing terms when using its code or assets; these acknowledgments do not replace any required notices for incorporated material.

## Contributing and sharing

This repository contains reusable source and synthetic examples, not the original character, private adapter, provider accounts or production evidence. Share only reviewed tools-only exports; workspace/evidence exports can contain personal paths and private assets.

Issues and pull requests are welcome. See [DEVELOPING.md](DEVELOPING.md) for development checks and the boundary between reusable tools and private character material.

No license is currently included. Licensing is pending; do not describe the repository as an open-source release until a license is selected and added.
