---
name: modeling-workbench
description: Operate the local modeling workbench for a character in Blender - study how reference avatars build a moving feature and build that mechanism, register generated guide meshes and show them overlapped on the live model, check candidates against a short declaration, and run guarded trials, reopen and retention. Use with an explicitly bound private character workspace; assets, identity, scope and generation authority remain there.
---

## The loop

The owner's loop: discover the problem area; inspect it and feed the inspection into image generation; pair it with
all of the character's references (everything, including manga and anime art: "the soul of the character"); the new
image reveals the inspected area with the character's likeness; generate a mesh to fit that likeness; register it onto
the character and fit to it, shown overlapped: the editable character solid orange, the guide a blue wire cage in the
same coordinates, never side by side. Where no guide does the job, solve the geometry so it truly represents the drawn
image, within a tolerance of the character's references.

Moving features come first as mechanisms: before changing how anything moves (a blink, a jaw, a mouth, an expression),
read [build the mechanism first](references/build-the-mechanism-first.md). Study how the reference avatars construct
the feature and build that mechanism, for example a lid turning as one piece on a hinge with one timing. Keep the rest
pose exactly on the accepted neutral. Pose guides fit the mechanism's few parameters and check it by overlap and
volume; they are not per-point targets at every pose. Fix a motion defect by changing the mechanism, never by stacking
per-phase fields or patches. Static shape (the rest pose, smoothing, joins) is fitted to registered guides, depth and
corresponding sections during the edit itself.

## Start

Read the workspace's current page (`PROJECT.md`, from the [current-page template](references/current-page.md)): the
mechanism, the accepted rest, the last owner-approved state, open owner decisions and the next action. Read the pages
below for the task at hand, not the whole skill. [Portable setup](references/portable-setup.md) installs and binds a
workspace; [getting started](references/getting-started.md) walks a first session. The verbs are Python calls and
also service operations on the CLI (`python -m modeling_system <operation> --input args.json`) and MCP:
`check_candidate`, `register_guide`, `construct`, `study_blink`, `bake_poses`, `audit_face`, `overlay_on_drawing`,
`review_sheet`, the generation
ledger and the `native_*` operations.

## The verbs

**guide** - acquire and register guides. Request images with `request_reference`, claim, dispatch and reconcile mesh
jobs without ever submitting a duplicate of an uncertain job ([generation](references/generation.md); never a mesh from
a single image). Register a generated mesh onto the accepted neutral on stationary anatomy, and give every guide a role
(identity, pose check, detail): [guides and display](references/guide-and-show.md). Join several generated variants of
a pose onto the accepted neutral with a tolerance band: [guide synthesis](references/guide-synthesis.md).

**show** - keep the work visible: any registered guide overlapped live in Blender through the reference adapter and
live bridge, overlap renders, motion videos (normal speed, then slow; several columns for variants), landmark-aligned
overlays on drawings ([guides and display](references/guide-and-show.md)) and matched review sheets. Numbers can reject
a candidate; approval comes from looking at the whole motion.

**construct** - build the mechanism: hinged lid landing on the opposing lid (still or lifted in the middle), one-pace
roll, clearance, attached parts carried on the edge ([motion paths](references/motion-paths.md)); bake it to the fewest
blend shapes plus a clip curve and round-trip the FBX ([bake](references/bake.md)). Deformers and guide
volume fits are bounded clean-up of a surface, not the way to create motion ([metric fitting](references/metric-fitting.md)).

**check** - measure every candidate the same way from one declaration: rest identity, motion outside the region,
protected objects, clearance, folds, reversals, symmetry, how an edge closes, the closed line's depth and how many blend
shapes carry the motion ([construction checks](references/construction-checks.md)); for the face, every shape, control
and combination through the [face audit](references/face-checks.md). Diagnose persistent creases,
crowding and section bends with [construction diagnostics](references/construction-diagnostics.md).

**study** - learn the construction from the reference avatars: extract their shapes and measure the moving band, the
simple motion that explains it, travel ratios, corners, the closed line, loop topology, the jaw hinge and viseme mixes
([study the avatars](references/study-avatars.md)).
A study ends in building what it found.

**trial / reopen / retain** - change the native file only through three plain calls
([trial, reopen, retain](references/trial-reopen-retain.md)): an isolated trial of the clean live source measured by the
declaration's standard checks, an independent reopen, and retention through the live owner, which refuses without a
passed reopen, passing checks and a recorded review of the watched motion. An unseen checkpoint is an experiment.
[Isolated Blender](references/isolated-blender.md) and [native adapters](references/native-adapters.md) describe the
runner and the adapter boundary; [operating sessions](references/operating-session.md) remain for workspaces that use
them, guarded by `checks.standard_policy` ([preservation](references/preservation-and-lessons.md)).

**log** - operations, reviews and job accounting are journaled in the workspace store ([episodes](references/episodes.md));
recover an interrupted effect from its receipts, never by replaying it
([intervention and recovery](references/intervention-and-recovery.md)).

**notes** - [construction and failure notes](references/notes.md): what real modeling taught, as symptom, cause and
remedy. Read the ones for the feature you are changing; add a note when a general finding changes what gets built.
The workspace store keeps scoped reviews and lessons as an optional log ([retained learning](references/retained-learning.md),
[method integration](references/method-integration.md)); it is not where every agent looks.

## Rules that hold everywhere

- Current workspace authority, then the owner's latest words, govern. Package defaults and old receipts authorize nothing.
- One native writer owns the live file; every change passes the owner and expected-state guards. Do not bypass them.
- The accepted neutral is the likeness authority. A generated guide supports a pose or a detail; it does not redefine
  the character. A user rejection stands over any earlier review.
- Compare by overlap in the same coordinates, at matched views and poses, against the guide and the last approved state.
- A new capability that the next edit needs belongs in the workbench, with tests, not in a private script.
- Credentials, character material and native bindings stay in the private workspace.

## Other references

[Retained method](references/method.md) (orient, observe, acquire, qualify, intervene, retain),
[control coverage](references/control-coverage.md) and [repair analysis](references/repair-analysis.md) (offline
diagnoses of control reach and coupled responses). Use `runtime_status()` to confirm which installation is loaded, and
`operation_context` for the procedures that apply to a generation or qualification decision.
