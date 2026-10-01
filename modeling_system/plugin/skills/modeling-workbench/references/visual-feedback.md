# Durable visual correction feedback

Use this flow when repeated trials drift from what the user meant or a candidate
code alone fails to explain what changed. It retains communication evidence; it
does not substitute for native geometry constraints or whole-result visual review.

Start `python -m modeling_system visual-feedback-ui --workspace /path/to/private-character`
manually. Open the printed `http://127.0.0.1:8765/` URL and stop with Ctrl+C after
use. `--store` can select the workspace's existing Store; `--port` selects another
loopback port. No native adapter or new dependency is needed. Installation into a
modeler's runtime is a separate guarded action.

1. Create a board. Import an inspection image with a plain-language label, exact
   source/version, capture date and camera/light/display-state/scale/pose descriptors. Unknown
   descriptors stay unknown. Source bytes are content-pinned without modification.
2. Drag to mark the region or enter normalized `[x,y,width,height]`. Save exact user
   wording, its date and source. Each region has its own identity; revising a region
   creates a new immutable target revision linked to its predecessor.
3. Record the concise interpretation, preserved features, proposed operation and
   expected visible appearance. Link qualified reference roles: image provenance,
   original overall guide disposition/rejection reason, supported portions and
   reason, excluded portions and reason, unknown portions and reason, and local
   qualification evidence. A rejected guide can supply qualified local appearance
   evidence. This never clears its rejection or adopts its geometry.
4. Save a fixed LEFT baseline / RIGHT isolated trial review with plain-language
   captions, changed and unchanged/unresolved areas, exact annotations and source
   metadata. Both panels retain their date/camera/light/state/scale/pose. Add an
   **eye and surrounding face** pair and a **whole-face** pair of these same
   baseline/trial versions. Declare each image's visible extent when importing it.
   These broader views appear first, fitted to the entire source image; optional
   closeups stay in a secondary expandable view. Fit, zoom and drag-to-pan controls
   act together on LEFT and RIGHT. Target outlines are hidden during skin review
   and can be shown without a filled overlay. No target coordinates are inferred
   on a different camera. Missing broader captures remain visibly unavailable;
   fitting an existing tight crop does not restore absent facial context.
   Exact descriptor
   equality yields **matched declared inputs**, not authenticated native matching.
   Different camera/light/state/scale/pose/size/region is **unmatched**; missing descriptors or
   indistinguishable source/result versions are **unknown**. Do not imply a surface
   repair merely because a silhouette or contour changed.
5. Save feedback on the displayed region, target/proposal revision, inspection,
   comparison, trial and source/result versions. A correction creates a new plan,
   visibly superseding the earlier interpretation. Old feedback stays on its old
   result. Reopen on a conflict; the browser preserves unsaved text. An explicit
   historical submission can archive old feedback but cannot activate a correction.
   A draft stays bound to the comparison on which typing began. Switching results
   cannot retarget it: return to its original comparison or explicitly clear it.
6. Record user agreement only from explicit wording/date/source, and only for its
   exact target/interpretation revision. Agreement is not result acceptance.
   Execution (`proposed`, `running`, `built`), technical verification, visually
   useful, retained and owner acceptance are separate reported facets for the exact
   comparison/result. Owner acceptance/rejection needs its own explicit user fact.
7. Reopen the board and review the correction history. Download the Markdown target
   summary for a human-readable handoff; all prior records remain retrievable by
   content address. Keep exports, private captions and assets outside public source.

## Keep the working view useful

The selected region's pinned inspection annotation, current interpretation,
comparison and useful reference roles are foregrounded. Other source images and
exact provenance stay in closed expandable views. History marks current versus
historical revisions and keeps the original result binding visible.

Archive an incidental image or finished board through the UI, then restore it from
Archived images or Show archived boards. This is a recoverable presentation choice;
it deletes no source bytes, rejected candidates, target revisions or history. An
archived image remains visible wherever a saved target/reference/review needs it.
No inference model automatically decides which evidence to discard. Archiving
outside this optional feedback store is not provided.

## Service and transport shapes

All mutations except board creation require `board` and `expected_revision` from
the last `visual_feedback_read`. Ledger compare-and-swap refuses concurrent/stale
updates. Python calls, CLI (`--input args.json`) and MCP expose the same operations:

- `visual_feedback_create(title, idempotency_key)` returns `board`/`revision`.
- `visual_feedback_image(..., image_path, metadata)` pins bytes. Metadata requires
  `label`, `source`, `source_version`, `captured_at`; optional `camera`,
  `display_state`, `pose`, `lighting`, `framing` (scale/crop descriptor), `view_role`
  (`eye_context`, `whole_face`, `detail`), existing Store `capture_record`, and `image_sha256`.
  A receipt link is retrievable evidence, not inferred native authentication.
- `visual_feedback_target(..., image, label, wording, date, source, region)` creates
  a region. Revision also requires `target_id` and current `supersedes` target record.
- `visual_feedback_plan(..., target, interpretation, preserved_features, method,
  expected_appearance, references, supersedes)` records an interpretation/proposal.
  References are `visual_feedback_reference` record IDs, not unchecked guide geometry.
- `visual_feedback_compare(..., target, plan, baseline, trial, baseline_caption,
  trial_caption, changed, unchanged, baseline_region, trial_region, baseline_approval, context_views)`
  pins a review. Optional approval is `{actor:"user",wording,date,source}`; generic
  review metadata is not accepted as a user fact.
  Optional `context_views` contains at most one `{role:"eye_context",baseline,trial}`
  and one `{role:"whole_face",baseline,trial}`. Each pair retains its own descriptor
  matching verdict and must name the primary baseline/trial versions and pose.
  Different versions/light/framing remain unmatched; missing fields remain unknown.
  Old immutable comparisons retain their original stored verdict; the browser also
  displays missing light/framing qualification without rewriting their evidence.
- `visual_feedback_submit(..., comparison, target, plan, inspection, trial,
  source_version, result_version, wording, date, source, correction, historical)`
  records exact-scoped feedback. Correction supplies the complete new
  `{interpretation,preserved_features,method,expected_appearance,references}`.
- `visual_feedback_agreement(..., target, plan, fact)` stores the explicit user fact.
- `visual_feedback_state(..., comparison, facet, value, evidence, fact)` stores an
  independent reported outcome. `fact` is only for explicit owner acceptance/rejection.
- `visual_feedback_presentation(..., collection, record, archived, reason)` appends
  an archive/restore event for a `board` or `images` record. Reopening returns
  derived `presentation` state alongside all unchanged evidence and history.
- `visual_feedback_read(board)` reopens the current target/interpretation pointers
  and full history; `visual_feedback_summary(board)` returns Markdown.
  `visual_feedback_export(board, output)` writes a new summary without overwriting.

These are optional communication records. They execute no native work, enforce no
new native approval prerequisite and cannot authenticate the identity of the person
who entered an asserted user fact. Native capture, authority, retention and artistic
acceptance must still be qualified in the bound workspace.
