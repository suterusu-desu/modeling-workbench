# Modeling Workbench

A portable workbench for guide-constrained character modeling in Blender. The
modeling owner chooses and executes qualified operations directly, reviews actual
images, and retains useful geometry and methods. The tools preserve dependencies,
evidence, recovery and established shape across the whole workflow.

No inference service, model credential or external agent skill is required.
The package includes its operator skill, numerical preparation, native job runner,
queues, candidate pipelines, review handoffs and experience store. Character
references, scenes, rig bindings and local native adapters belong in separate
private workspaces.

## Install

Use Python 3.12 or newer in a virtual environment on Windows, macOS or Linux.
Install from this checkout, an extracted tools archive or a built wheel:

```sh
python -m pip install .
python -m modeling_system.init_workspace /path/to/character --character "My character"
python -m modeling_system.setup_check --workspace /path/to/character
python -m modeling_system.prepare_plugin /path/to/modeling-workbench --workspace /path/to/character
```

Quote paths containing spaces. Initialization creates the character's own
authority, decisions, methods, lessons and material directories, together with a
complete project-local workbench skill. It preserves existing files and grants no
modeling, generation or unattended-work authorization. The generated plugin uses
the same installed Python service as the CLI. Other clients can read the bundled
`SKILL.md` directly.

See [portable setup](modeling_system/plugin/skills/modeling-workbench/references/portable-setup.md)
and [native adapter qualification](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md).
Blender and a qualified local bridge/scene adapter are external prerequisites for
native work. There is no universal rig adapter or bundled character.

## Use the modeling loop

Start with current workspace authority and `inspect_situation`. Identify the
dominant visible defect and a question that changes the next edit. Reuse unchanged
evidence. Registered guides, depth and corresponding sections must constrain the
actual correction of static shape, including transitions and attachments.

Build a moving feature as a mechanism first: study how the reference avatars
construct it, build that mechanism (for example a lid turning as one piece on a
hinge with one timing) with the rest pose on the accepted neutral, then let pose
guides fit its few parameters and check it by overlap. See
[build the mechanism first](modeling_system/plugin/skills/modeling-workbench/references/build-the-mechanism-first.md).

Make one recoverable candidate, compare its appearance early against the guide
and an earlier useful baseline, then verify affected motion and independently
reopen consequential results. Retain an overall visible improvement with its
limitations. Technical success, operator retention and user acceptance are
different facts.

The [operator skill](modeling_system/plugin/skills/modeling-workbench/SKILL.md)
and [retained method](modeling_system/plugin/skills/modeling-workbench/references/method.md)
carry this process to another agent without requiring conversation history.

## Direct operation and batching

Use `ModelingService` for owner-controlled operations, or the packaged
`direct_session.create_session` for continuing work:

```python
from modeling_system.direct_session import create_session

session = create_session(
    queue_directory, service=service, episode=active_episode, owner=sole_owner,
    goal={"objective": authorized_objective},
    observe_context=read_actual_revisions,
    catalog=qualified_catalog, handlers=qualified_handlers,
    task_order=["inspect", "prepare", "candidate"],
    preservation=qualified_preservation_policy,
)
session.run(max_steps=authorized_step_bound)
```

These callbacks bind the current character's real capabilities and dependencies.
An eligible singleton or fixed prerequisite runs directly. When multiple actions
are eligible, an explicit task order or local owner selector resolves the choice;
the default requests a scoped decision. The queue never invents priorities or
asks a model to choose. Native effects remain serial under one owner.

[Operating sessions](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md)
link the episode, task contracts, current observations, immutable receipts,
preservation and reviews. [Candidate pipelines](modeling_system/plugin/skills/modeling-workbench/references/candidate-pipeline.md)
continue fixed preparation, correction, verification and retention without a
controller restart per operation. Independent qualified work can continue while
an exact review dependency waits.

## Capabilities

- [Mechanism-first construction](modeling_system/plugin/skills/modeling-workbench/references/build-the-mechanism-first.md): a lid landed on the opposing lid (still, or lifted in the middle on the same hinge) by a turn on a hinge, rolled with one pace, attached parts such as lashes carried on the turning edge, and a [closing-edge diagnostic](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md#how-an-edge-closes) that reports opposing-lid travel, closing rate along the edge and distance from one roll.
- [Guides and display](modeling_system/plugin/skills/modeling-workbench/references/guide-and-show.md): registration of a generated guide on stationary anatomy, guide roles (identity, pose check, detail), a reference adapter and live bridge that show any registered guide overlapped on a visible Blender without a private adapter, overlap renders, motion videos with several columns and landmark-aligned overlays on drawings.
- [Standard construction checks](modeling_system/plugin/skills/modeling-workbench/references/construction-checks.md): one short declaration (region, accepted rest, protected objects, clearance obstacles, symmetry, closing edges, blend-shape tolerance) measures every candidate's saved poses and generates the guard for native trials and retention, without a per-trial adapter.
- [Parameterized methods](modeling_system/plugin/skills/modeling-workbench/references/operation-recipes.md): reusable fitting and preparation operations, qualified native jobs and synchronous checkpoint retention, with [parts set aside from view](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md#parts-set-aside-from-view) in the live display and the retained file.
- [Preservation and connected repair](modeling_system/plugin/skills/modeling-workbench/references/preservation-and-lessons.md): source-bound constraints consumed by fitters, final evaluated measurements, a pre-launch check of pinned sources, an offline preview of the declared cells and scoped tolerance when connected repair is necessary.
- [Preparation and native setup](modeling_system/plugin/skills/modeling-workbench/references/preparation-and-bootstrap.md): array contracts, correspondence, [planar and 3D-surface finite-element metrics, crowded-material relaxation, shape-preserving deformation, optionally inside per-vertex guide intervals, and planar re-layout of collapsed material, harmonic or keeping a reference layout](modeling_system/plugin/skills/modeling-workbench/references/metric-fitting.md), [motion paths](modeling_system/plugin/skills/modeling-workbench/references/motion-paths.md) (hinged parts, rolled in-betweens, obstacle clearance, re-timing and smooth joins for an inherited motion, material section bends), [pose guides from generated variants joined onto an accepted neutral](modeling_system/plugin/skills/modeling-workbench/references/guide-synthesis.md) (front depth maps, thin-relief removal, variant change and spread, smooth pinned depth fields, tolerance bands, alignment on still skin with trimmed refits, clearance in front of the character's own anatomy, retreat between poses, forward-only volume fits, smooth fits into a tolerance band, guide surfaces), projected triangle contact, support, [material-crowding, local-fold and bend-map](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md) diagnostics and measured setup reuse, including repeated bootstrap in a long-lived visible session.
- [Evidence and experience](modeling_system/plugin/skills/modeling-workbench/references/retained-learning.md): immutable actual observations, scoped visual reviews with [matched review sheets](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md), relevance-ranked retrieval, conditional techniques and failure evidence available in later scopes.
- [Recovery](modeling_system/plugin/skills/modeling-workbench/references/intervention-and-recovery.md): exact original receipts, effect reconciliation, [session settlement](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md) of a task whose capability raised after its effect, and protection from duplicate native effects or overwriting later edits.
- [Optional reference generation](modeling_system/plugin/skills/modeling-workbench/references/provider-workflows.md): source-bound image, mesh and video jobs with separate authorization and accounting, including [multi-view mesh jobs](modeling_system/plugin/skills/modeling-workbench/references/generation.md) that bind each provider slot to a reviewed image and can refuse single-image meshes. Existing art needs no generation account.

The [capability inventory](modeling_system/capabilities.json) maps the complete
surface. `operating_protocol()` reports the installed profiles and routes.
Session metrics report actual execution, native stages, review time and missing
workload coverage; they do not claim unmeasured speedups.

## Portability and limits

Tools-only exports exclude character material and identifying records. Recreate
the virtual environment and generated plugin on each machine. Restore private
evidence separately, rebind external paths and qualify the actual Blender runtime
and scene adapter. Old receipts stay readable; interrupted effects require
reconciliation on their original host before any retry.

CI runs synthetic tests and clean wheel/tools-archive installs on Windows, macOS
and Linux. That establishes package behavior, not artistic quality on an unseen
character or compatibility with every Blender rig. Native adapter qualification,
source/guide authority and actual visual review remain explicit requirements.

Read [development boundaries](DEVELOPING.md) before submitting a pull request.
This repository is public for reading, forking and contributions. Licensing for
the workbench is pending; a license has not yet been selected.
