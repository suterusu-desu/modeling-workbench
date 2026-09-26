# Lazy component catalogs on stored geometry

Use a component catalog when a local question on a large disconnected mesh would otherwise require transferring the whole triangle list. The catalog is built once on one exact stored state and object, keeps its complete member mapping in a content-addressed NPZ inside the store, and answers with compact counts, per-component metrics, bounded reads and lazy exact selection of one component. It never contacts Blender, writes native data, admits a target or infers anatomy.

The stored state ID and the object's recorded geometry hash are the only authority. A catalog is bound to that revision. It is not refreshed, remapped or matched by proximity when the geometry changes; a changed revision needs a new catalog.

## Interface

The public service exposes `build_component_catalog`, `read_component_catalog`, `locate_component` and `select_component`. Build retains a catalog through the existing operation journal. Read, locate and select are read-only, including through CLI/MCP; select returns recorded IDs and does not change Blender selection. Their module helpers are:

| Helper | Arguments | Result |
| --- | --- | --- |
| `build(service, state, object_name, connectivity="shared_edge", max_triangles=None)` | Stored state record ID and object name. | Compact catalog result with `catalog` record ID, identity, `summary`, `preview`, `members` reference, `reads`, `select`, `locate`, `processing`, `limits`. |
| `read(service, catalog, offset=0, limit=20, max_chars=8000, order="largest")` | Catalog ID and window. | Bounded window of per-component metric rows with `next`; no member identities. |
| `locate(service, catalog, triangle)` | One exact recorded triangle index. | The containing component's metrics and a ready `select` descriptor. |
| `select(service, catalog, component, expected_state, expected_geometry_hash=None, members="triangles", offset=0, limit=256, max_chars=8000)` | Component identity plus the expected state (required) and geometry hash (optional). | One component's exact members, paged, after identity verification. |

Returned descriptors name the wrapper operations `build_component_catalog`, `read_component_catalog`, `locate_component` and `select_component` (module constants `BUILD_OPERATION`, `READ_OPERATION`, `LOCATE_OPERATION`, `SELECT_OPERATION`). Descriptors under `reads` other than `components` use the existing `read_record` operation on the catalog record. The pure function `analyze(arrays, connectivity, max_triangles)` computes the same result from an in-memory array dictionary without storage.

## Connectivity semantics

An empty catalog returns null selection/location suggestions. A nonempty metric page's `select` descriptor selects its first displayed component; it is executable without filling an unspecified component. Choose another displayed ID explicitly when needed. Member continuation windows include their own byte-budgeted next descriptor.

`shared_edge` (default): two triangles connect when they share one undirected edge, meaning the same two vertex indices. `shared_vertex`: two triangles connect when they share one vertex index, a coarser relation that merges shells touching at a single vertex. Both use recorded indices only. Coincident coordinates never connect distinct indices; duplicated vertices remain separate components, and `vertices_in_multiple_components` reports vertex-touch contacts left disconnected by the edge relation.

A nonmanifold edge (three or more incident triangles) joins every incident face into one component. That is an explicit ambiguity, not a resolved sheet ownership; the catalog reports nonmanifold and boundary edge counts, the maximum edge degree and a degree histogram. Repeated-index triangles, zero-area triangles with distinct indices, and duplicate triangles remain members and are counted explicitly in disjoint categories. Invalid indices and nonfinite coordinates are refused.

Connectivity is triangle-based. When `triangle_polygon` is recorded, polygon membership is preserved: component rows report distinct polygons, and `polygons.split_across_components` counts recorded polygons whose triangles fall into different components. Native polygon-edge adjacency is not recomputed from that mapping. Without recorded polygon topology, a triangulation diagonal cannot be distinguished from a native edge, and the catalog says so in `connectivity.basis`. A recorded `triangle_component` semantic is reported separately as `recorded_component_semantics`; it never defines catalog connectivity.

## Identities and metrics

Each component is identified as `c<minimum triangle index>` within the exact revision, with `label` as its zero-based position in ascending order. Identities are deterministic: rebuilding on the same stored state produces the same catalog record and asset, and identical geometry under another state ID produces identical identities in a separate catalog. Identities carry no anatomical meaning.

Rows report `triangles`, `vertices`, `bounds`, `surface_area`, `boundary_edges`, `nonmanifold_edges`, `max_edge_degree`, `degenerate_triangles`, `duplicate_triangles` and, when recorded, `polygons` and `recorded_components`. The summary reports totals, referenced and unreferenced vertices, largest and single-triangle components, edge statistics, degenerate and duplicate counts and total surface area. `processing` states the processed triangle count, budget and algorithm; `truncated` is always false because oversized inputs are refused rather than cut.

## Retained mapping and bounded reads

`members.asset` is the content hash and store path of the NPZ holding `component_label` (component per triangle), `member_order` with `member_start` (triangles grouped by component), all per-component metric columns, the largest-first ordering and the catalog identity strings. The compact result never inlines it. `read` pages metric rows largest first or by label, at most 64 rows per window and within `max_chars`. Each read verifies the asset hash and its consistency with the record.

## Selection

`select` verifies, in order: the catalog record kind and content hash; that `expected_state` equals the catalog state; that an optional `expected_geometry_hash` equals the catalog geometry hash; that the stored arrays still verify and hash to the catalog geometry; and that the member asset verifies. A catalog built on another state is rejected as stale, with a hint saying whether the requested state records identical geometry (so a rebuild there yields identical identities) or different geometry. Then only the requested component is materialized: `triangles` (original recorded indices), `vertices` (distinct referenced vertex indices) or `polygons` (distinct recorded polygon indices, requiring `triangle_polygon`). Windows shrink to fit `max_chars` and continue through `members.next`. The selection result carries the component row, identity and verification facts; the catalog record holds the limits.

## Limits

The default budget is 2,000,000 triangles, raised explicitly through `max_triangles` up to 16,000,000. Larger inputs need a scoped study; nothing is truncated silently. The catalog describes recorded triangles: closure, volume, visibility, pose applicability and appearance are not evaluated. Section, ray and nearest queries remain the existing geometry operations; a catalog only narrows which exact triangles a local question concerns.

After installation, a catalog can be exercised on any imported state with a recorded mesh object. Resource comparison against whole-mesh reads and actual operator use are separate acceptance evidence.
