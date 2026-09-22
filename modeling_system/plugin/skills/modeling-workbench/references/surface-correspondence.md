# Ordered surface correspondence evidence

Use `inspect_surface_correspondence(case_path, expected_state)` to retain an offline ordered path, its complete supplied alternatives, missing source edges and exact selected-section connectivity. It uses the existing service journal and pinned-case diagnostic recipe route. It does not query Blender or qualify a target. Source code, a separate installed runtime and actual operator adoption must be verified independently.

The question is whether the proposed correspondence is supported over the declared sampled order. A selected-first hit can hide excluded foreground geometry; a shared normal or component label cannot resolve anatomy. A gap is retained as a station, never skipped when comparing adjacent samples. Disconnected planar cuts do not establish a disconnected full 3D sheet or rule out an out-of-plane route. Even a shared selected section component does not prove continuous correspondence between samples.

## When corresponding poses follow different branches

Connected-component membership and original triangle ancestry identify geometry,
not corresponding anatomical roles. A trace can reach a tagged feature in one
pose while continuing into a return or neighboring surface in another. That
disqualifies this proposed pairing; it does not establish that either guide is
unusable. Reuse the existing section and image evidence to distinguish the roles.

For an ambiguous proposed pairing, retain:

- The intended region and role on each exact source/pose, with corresponding
  start, end and any intermediate anatomical landmarks. A radial column derived
  from the current model supplies a sampling direction, not target authority.
- Each candidate branch's ordered points, original triangle/edge identities,
  orientation and query limits. Keep selected and rejected alternatives with
  reasons such as wrong surface role, crossing a return, missing landmark support
  or unresolved ambiguity. A branch is not rejected merely for being farther away.
- The evidence for matching those roles and landmarks across poses, including
  exclusions and unsupported spans. Whole-head nearest points, equal curve
  fractions, a first tagged parent or a coordinate extremum do not establish that
  match. Arclength remapping operates only within an already qualified pairing.

These are meanings and evidence retained by the existing `roles`, `selection`,
path and alternative records; they are not an automatic anatomical classifier
or an additional review service. Reuse a sufficient recorded qualification.

Keep crop-created endpoints distinct from actual mesh boundaries. Likewise,
epsilon-welded section points are proximity evidence, not exact edge identities;
they can bridge separate sheets or conceal branch alternatives. An open or
ambiguous trace means the proposed route is unresolved within its declared
selection and query. Preserve crop bounds, excluded/coplanar segments and the
producer's endpoint convention before interpreting why it stopped.

## Version-1 pinned case

Supply `schema_version: 1`, `question`, and `state` matching all `expected_state` fields: `source_state_id`, `pose`, `frame`, `units`, `dependency_fingerprint`. This is a historical identity check. `source_topology`, `roles`, `selection` and `orientation` each need `meaning` and `evidence`. Owner-authored roles and support limits remain separate from measurements. Declare the winding and coordinate convention for normal evidence. Each evidence item may be an exact `{path, sha256}` file reference, which the existing store pins and retains.

`source_topology.kind` is `native_polygon_edges` or `triangulation_edges`; the latter does not establish native polygon adjacency. `arrays: {path, sha256}` identifies an NPZ with `source_edges`, unique nonnegative integer vertex-ID pairs, shape `(E,2)`, without self edges. These are actual identities, not indices into another list. The supplied edge universe and its coverage belong in source-topology evidence. Missing consecutive edges are diagnostic facts, not silently repaired paths.

`paths` contains 1–128 objects with distinct `id`, `meaning`, `evidence`, `kind` (`material_edges` or `spatial_probes`), `coverage`, `all_alternatives_retained` (boolean), `target_context`, optional `section_graph`, and ordered `stations`. Coverage is `{status, missing}` with `unknown`, `declared_subset` or `complete_for_declared_query`; completeness requires an empty missing list. Explicitly limit the geometry, ray range, selection, view and poses. Retaining every returned hit does not prove the query includes every relevant surface.

Each target context carries the same five state/domain fields as above plus `target_id` and `query`. A referenced graph must match that target identity, pose and domain exactly. Each station has `point`, optional `source_vertex` (required for material paths), and `alternatives`, including an empty list when no hit was returned. Extra evidence fields remain in the pinned case. Each alternative retains `triangle`, `point`, `barycentric`, winding-oriented `normal`, explicit boolean `selected`, and contiguous zero-based `full_order`. Preserve coincident triangle owners and all unselected alternatives. Ranks are supplied evidence: this diagnostic does not recompute rays or establish their completeness.

Optional `section_graphs` contain distinct `id`, `meaning`, `evidence`, `coverage`, `context`, and `segments`. Context has the target identity/domain fields plus the exact `plane`. Each segment has a unique nonnegative `triangle`, two distinct string `nodes` encoding exact original mesh-edge/vertex identities, and boolean `selected`. Namespace node identities within their graph. The diagnostic derives selected components from shared identities; supplied component labels are ignored. Do not create nodes through epsilon proximity or bridge missing/coplanar segments. Retain the actual section points, endpoint/junction information, exclusions and producer files in the pinned evidence. A graph is a declared section, not proof of full-surface topology.

The implemented plane is axis-aligned: `{axis: "X" | "Y" | "Z", value, tolerance}` with finite value and explicit nonnegative tolerance in the recorded units. Every associated source station and hit must lie within that plane tolerance. Other section types require a future explicit contract; reusing triangle IDs from another plane is refused. Keep oblique sections as their original pinned evidence rather than relabeling them as axis-aligned. A deliberately transformed coordinate frame must transform all associated points and normals consistently and retain its exact transform and source identities; changing the plane label alone is insufficient.

There are limits of 10,000 stations, 100,000 alternatives, 128 section graphs, 250,000 section segments and 1,500,000 source edges. Larger inputs require an explicitly scoped study; never silently truncate controlling alternatives.

## Read the result

`summary` reports unsupported stations, multiple selected alternatives, selected-first behind full-first, missing source edges and adjacent-station relation counts. `reads` links to bounded `paths`, `stations`, `transitions`, `sections`, `case` and `limits`. All original alternatives remain retained.

Each transition separates `source_edge_connected` (null for spatial probes) from `section_relation`: selection gap, unknown selection coverage, unknown branch evidence, ambiguous section branches, same selected section component, or different selected section components. Missing comparable graphs remain unknown; matching guide labels across different planes do not substitute for connectivity evidence. Incomplete query coverage remains unknown even when the returned subset appears to have one branch. Review station alternatives and graph coverage before interpreting a summary.

`continuous_correspondence_proven`, `native_ready` and `target_admission` remain false. The report organizes evidence for the existing owner qualification and independent review process. It cannot change a mask, selection, rejection, pose mapping, native geometry or appearance judgment.
