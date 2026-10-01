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
python -m pip install .          # or: python -m pip install -e /path/to/your/clone  (upgrade with git pull)
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

Start with [one useful correction cycle](modeling_system/plugin/skills/modeling-workbench/references/useful-workflow.md):
clarify the visible target, choose supported shape fitting or coherent motion
construction, diagnose only uncertainty that changes the next edit, build one
recoverable comparison, and keep the supported gain with its limits. This route
curates observed local successes and known failures; it does not claim every
experiment or package feature improved appearance. Tools and diagnostics serve
that cycle. Cross-character transfer still needs evidence in the new workspace.

Inspect the problem area and reuse qualified spatial support. When a named shape
or likeness constraint is missing, generate an image using all of the character's
references, reconstruct reviewed views into a mesh, register its useful support
onto the character and fit to it. Show the guide overlapped on the live model
(the character solid orange, the guide a blue wire cage in the same coordinates).
Start from the workspace's current page (`PROJECT.md`): the mechanisms, guides
and their roles, the last owner-approved state and the next action. Registered
guides, depth and corresponding sections constrain the actual correction of
static shape, including transitions and attachments.

Build a moving feature as a mechanism first: study how the reference avatars
construct it (extract their shapes and measure them), build that mechanism (for
example a lid turning as one piece on a hinge with one timing) with the rest pose
on the accepted neutral, then let pose guides fit its few parameters and check it
by overlap. See
[build the mechanism first](modeling_system/plugin/skills/modeling-workbench/references/build-the-mechanism-first.md)
and [study the avatars](modeling_system/plugin/skills/modeling-workbench/references/study-avatars.md).

Make one recoverable candidate, compare its appearance early against the guide
and an earlier useful baseline, then verify affected motion and independently
reopen consequential results. Retain an overall visible improvement with its
limitations. Technical success, operator retention and user acceptance are
different facts.

The [operator skill](modeling_system/plugin/skills/modeling-workbench/SKILL.md) is
a short router: the loop, eight verbs (guide, show, construct, check, study,
trial / reopen / retain, log, notes) and the page to read for each task. It
carries this process to another agent without requiring conversation history.

## Keep a visual correction target across sessions

Start the optional local feedback UI manually, pointing at the private character
workspace (and, optionally, its existing evidence store):

```sh
python -m modeling_system visual-feedback-ui --workspace /path/to/character --port 8765
# equivalent: python -m modeling_system.visual_feedback_server --workspace /path/to/character
```

Open `http://127.0.0.1:8765/`. The visible workflow explains the correction cycle
and offers evidence-informed method drafts with their applicability and limits.
Create a board, import a source-bound inspection
image, drag to mark a region, and save the exact dated user wording. Record the
interpretation, features to preserve, proposed method and expected appearance.
Qualify reference images by local role, including supported, excluded and unknown
portions; a guide's overall rejection remains visible. Compare a labelled baseline
on the left with an isolated trial on the right, then save feedback on those exact
region/target/interpretation/source/result revisions. A correction creates a new
interpretation and preserves the earlier one. Reopen the saved board or download
its Markdown target summary. Stop the server with Ctrl+C when finished.

The working view highlights the selected region's inspection, current comparison
and useful reference images. Other saved images and exact provenance stay in
expandable views; correction history has its own tab. Archive incidental images
or finished boards to reduce clutter, then restore them from the archive when
needed. Archiving only changes presentation: pinned bytes, references, comparisons
and prior decisions remain recoverable. An archived image still appears wherever
an existing target or review needs it.

Images and records use the same content-addressed Workbench Store and revisioned
Ledger as the service. The `visual_feedback_*` operations are also available through
Python, CLI JSON input and MCP. No additional UI framework, relay, account or native
adapter is needed. This is a standalone loopback UI; it is not embedded in Blender
and does not edit a scene or adopt a new runtime. Keep its private store outside
this package checkout.

Matching compares recorded camera/display-state/pose/size/region descriptors,
including the marked inspection's framing against both comparison images;
missing information stays unknown and differences stay unmatched. Recorded source
assertions do not authenticate native capture or live state. User agreement is an
explicit dated fact for a target and interpretation; baseline approval and result
acceptance require their own exact user facts. Proposed/running/built, technical
verification, visual usefulness, retention and owner acceptance remain independent.
The flow adds no mandatory approval gate to modeling.

See [visual feedback](modeling_system/plugin/skills/modeling-workbench/references/visual-feedback.md)
for the service record shapes and correction/reopen workflow.

## Direct operation and batching

Change the native file with three plain calls, `trial`, `reopen` and `retain`
([trial, reopen, retain](modeling_system/plugin/skills/modeling-workbench/references/trial-reopen-retain.md)):

```python
from modeling_system.trials import Trials
lane = Trials(service, owner, trials_dir, blender=blender, reference=reference,
              objects=['Face'], poses=poses, declaration='eye-declaration.json')
lane.trial('lift03', source=live_checkpoint, construction='build_lift.py')
lane.reopen('lift03')
lane.retain('lift03', target=new_checkpoint, label=label, review=review)
```

The construction-first verbs are also service operations, so the CLI and the MCP server expose them
(`check_candidate`, `register_guide`, `construct`, `study_blink`, `bake_poses`, `audit_face`, `overlay_on_drawing`,
`review_sheet`, `review_variants`):

```sh
python -m modeling_system check_candidate --workspace /path/to/character --input check.json
```

Use `ModelingService` for owner-controlled operations, or the packaged
`direct_session.create_session` for continuing work in an existing session setup:

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
preservation and reviews. Independent qualified work can continue while
an exact review dependency waits.

## Capabilities

- [Native polygon shape diagnostics](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md#compare-native-polygon-shapes): compare matching recorded polygon loops while measuring each capture's actual tessellation. Per-polygon native normal angles, degeneracy and projected crossings remain separate; inherited crossings are distinguished from newly observed projections. No 3D self-intersection or acceptance verdict is implied.
- [Source-surface coverage diagnostic](modeling_system/plugin/skills/modeling-workbench/references/source-coverage.md): explicit source-triangle/barycentric correspondence, UV reversals, conservative gap/overlap area intervals, boundary drift and corresponding interior 3D samples. Missing or unqualified correspondence remains unknown; this is an offline measurement, not a solver or retention gate.
- [Assigned support fitting](modeling_system/plugin/skills/modeling-workbench/references/metric-fitting.md#fit-explicitly-assigned-support-domains-together): one coupled endpoint fit with caller-assigned support ownership and independent domain normals. Nearby sheets cannot substitute for the assigned support; boundary escape remains unknown even with zero normal-band residual.
- [Ordered source fitting](modeling_system/plugin/skills/modeling-workbench/references/metric-fitting.md#preserve-declared-source-cell-order-during-the-fit): constrained redistribution within declared source cells, or opt-in UV motion across actual source-chart triangles with exact piecewise lifts. Connected UV face area bounds and the fixed footprint boundary remain enforced. Invalid starts and blocked steps are reported; chart order and finite triangle-facing checks do not certify 3D coverage, collisions or appearance.
- [Mechanism-first construction](modeling_system/plugin/skills/modeling-workbench/references/build-the-mechanism-first.md): a lid landed on the opposing lid (still, or lifted in the middle on the same hinge) by a turn on a hinge, rolled with one pace, attached parts such as lashes carried on the turning edge, and a [closing-edge diagnostic](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md#how-an-edge-closes) that reports opposing-lid travel, closing rate along the edge and distance from one roll.
- [Study the avatars](modeling_system/plugin/skills/modeling-workbench/references/study-avatars.md): a read-only extraction worker for a reference avatar's mesh and shape keys, and measures of how it builds a moving feature: which simple motion explains it, how far the moving band reaches, travel ratios, corner travel, the closed line's depth and whether the loops around the opening are closed quad loops.
- [Guides and display](modeling_system/plugin/skills/modeling-workbench/references/guide-and-show.md): registration of a generated guide on stationary anatomy, guide roles (identity, pose check, detail), a reference adapter and live bridge that show any registered guide overlapped on a visible Blender without a private adapter, overlap renders, motion videos with several columns, landmark-aligned overlays on drawings, and built options of an uncertain decision shown together in one call for the owner to choose between.
- [Bake to blend shapes](modeling_system/plugin/skills/modeling-workbench/references/bake.md): the fewest shapes that carry a built motion (the end pose plus correctives driven by bumps of the same weight), per-phase error and clearance, the clip curve, left and right shapes for winks, a Unity .anim keying every shape on every renderer, and an FBX exported (with the source objects' materials and UV maps) and round-tripped through Blender.
- [Face construction audit](modeling_system/plugin/skills/modeling-workbench/references/face-checks.md): every face shape, control and combination checked from one declaration over extracted shapes: regions and a still lid margin, one straight path per control with no phase-gated keys, a rigid jaw hinge the skin follows by a weight, bounded lip falloff, visemes as base-shape mixes in the right slots, left/right splits, lips and teeth in combinations, single-frame shapes and full-state clips.
- [Standard construction checks](modeling_system/plugin/skills/modeling-workbench/references/construction-checks.md): one short declaration (region, accepted rest, protected objects, selected regions that must match the accepted baseline at declared phases, clearance obstacles, symmetry, closing edges and corners, moving band, gaze limits, a carried lash strip, combined closed poses, blend-shape tolerance) measures every candidate's saved poses and generates the guard for native trials and retention, without a per-trial adapter.
  Native captures include ordered polygon connectivity so regional preservation can handle changing quad/ngon diagonals;
  actual triangles remain available at every pose for collision queries. Legacy captures without sufficient correspondence stay unknown.
- [Native jobs and retention](modeling_system/plugin/skills/modeling-workbench/references/operation-recipes.md): material fields, qualified native jobs and synchronous checkpoint retention, with [parts set aside from view](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md#parts-set-aside-from-view) in the live display and the retained file.
- [Preservation and connected repair](modeling_system/plugin/skills/modeling-workbench/references/preservation-and-lessons.md): source-bound constraints consumed by fitters, final evaluated measurements, a pre-launch check of pinned sources, an offline preview of the declared cells and scoped tolerance when connected repair is necessary.
- [Preparation and native setup](modeling_system/plugin/skills/modeling-workbench/references/preparation-and-bootstrap.md): array contracts, correspondence, [planar and 3D-surface finite-element metrics, crowded-material relaxation, shape-preserving deformation, optionally inside per-vertex guide intervals or on an arbitrary support surface within an offset band along its normal, and planar re-layout of collapsed material, harmonic or keeping a reference layout](modeling_system/plugin/skills/modeling-workbench/references/metric-fitting.md) with a [report of its projected held boundary](modeling_system/plugin/skills/modeling-workbench/references/metric-fitting.md#check-the-projected-boundary-before-the-layout) (loops, crossings, touches, overlaps, reflex corners), [motion paths](modeling_system/plugin/skills/modeling-workbench/references/motion-paths.md) (hinged parts, rolled in-betweens, obstacle clearance, re-timing and smooth joins for an inherited motion, material section bends), [pose guides from generated variants joined onto an accepted neutral](modeling_system/plugin/skills/modeling-workbench/references/guide-synthesis.md) (front depth maps, thin-relief removal, variant change and spread, smooth pinned depth fields, tolerance bands, alignment on still skin with trimmed refits, clearance in front of the character's own anatomy, retreat between poses, forward-only volume fits, smooth fits into a tolerance band, guide surfaces), projected triangle contact, support, [material-crowding, local-fold and bend-map](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md) diagnostics, [pixel traces of a visible defect to the recorded surfaces that could produce it](modeling_system/plugin/skills/modeling-workbench/references/construction-diagnostics.md#find-the-geometry-a-visible-defect-belongs-to) and measured setup reuse, including repeated bootstrap in a long-lived visible session.
- [Construction and failure notes](modeling_system/plugin/skills/modeling-workbench/references/notes.md): what real modeling taught, as symptom, cause and remedy, read before an edit.
- [Evidence and experience](modeling_system/plugin/skills/modeling-workbench/references/retained-learning.md): immutable actual observations, scoped visual reviews with [matched review sheets](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md), relevance-ranked retrieval, conditional techniques and failure evidence available in later scopes.
- [Recovery](modeling_system/plugin/skills/modeling-workbench/references/intervention-and-recovery.md): exact original receipts, effect reconciliation, [session settlement](modeling_system/plugin/skills/modeling-workbench/references/operating-session.md) of a task whose capability raised after its effect, and protection from duplicate native effects or overwriting later edits.
- [Optional reference generation](modeling_system/plugin/skills/modeling-workbench/references/generation.md): source-bound image, mesh and video jobs with separate authorization and accounting, including [multi-view mesh jobs](modeling_system/plugin/skills/modeling-workbench/references/generation.md) that bind each provider slot to a reviewed image and can refuse single-image meshes. Existing art needs no generation account.

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
