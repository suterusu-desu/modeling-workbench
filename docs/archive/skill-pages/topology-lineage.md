# Exact topology lineage and selection remapping

Use `record_topology_lineage(case_path)` when a topology-producing operation provides an exact lineage artifact and matching receipt. The offline tools validate and retain declared relations; they neither implement a native topology writer nor infer identities by proximity. Same-topology response ancestry is a separate relation.

## Producer contract

A version-1 case has `lineage: {path, sha256}` and `operation_receipt: {path, sha256}`. Both original files are pinned. The lineage JSON contains `schema_version: 1`, `operation_id`, `before`, `after`, and `relations`. Each domain descriptor contains `state`, `object`, `domain` (`vertex`, `edge` or `polygon`), `topology_hash`, explicit `scope` and distinct string `ids`. The domain must remain the same across one relation record. Descriptors are exact revisions, including their declared identity order and scope; identical coordinates do not substitute for matching descriptors.

The producer receipt contains the same `operation_id`, `status: "topology_change_recorded"`, `lineage_sha256`, and `before_revision`/`after_revision`: SHA-256 of each descriptor serialized with the workbench's canonical JSON. This verifies mutual binding of the supplied files. A native adapter must independently establish that the recorded operation actually produced this mapping; matching declarations alone are not that verification. Do not fabricate a receipt from an old mesh merely to satisfy the contract.

Each relation supplies `sources`, `targets` and a kind: `preserved` (one to one), `split` (one to many), `merged` (many to one), `deleted` (one to none), `created` (none to one), or `derived` (nonempty source and target groups). Every declared output has exactly one origin relation; every input has at least one explicit disposition. An input may be preserved and also contribute to a derived output. A deleted input cannot simultaneously have descendants. This preserves exact overlapping ancestry without conflating an original vertex with a new interpolated vertex. Missing dispositions stay visible; a scoped inventory does not silently claim completeness over the whole asset.

Optional `weights` maps every target to an explicit finite value for every source, with mandatory `weight_semantics`. The tool preserves the operation's values without assuming normalization, interpolation, rig transfer or weight propagation through composed operations. A derived group's ancestry is the stated group; unknown finer pairing is not invented.

Limits are 250,000 identities per domain, 250,000 relation groups and 1,500,000 expanded ancestry entries. Large studies need a smaller explicit scope instead of silent truncation.

## Composition and selectors

`compose_topology_lineage(lineages)` accepts 2–64 retained records whose adjacent after/before descriptors match exactly. It preserves the original records and distinguishes original ancestors from elements introduced midway through the chain. A descendant partly derived from newly created geometry does not become purely ancestral to the original mesh. Composition does not invent a weight-combination rule.

`remap_topology_selection(lineage, selection, expected_source, direction="forward", policy="strict")` requires the returned before revision for forward mapping or after revision for reverse mapping. Unknown IDs and stale revisions are refused. Strict mode retains candidates and ambiguity but withholds a selector if a merged/derived target has unselected or introduced ancestry. Reverse mapping similarly reports partial split selection. Deletions and elements with no original counterpart remain explicit; reverse reads retain introduced ancestry separately.

Explicit `any` includes descendants touched by any selected ancestor; `all` includes only descendants whose complete original ancestry is selected and which have no introduced ancestry. Those are selection semantics, not permission to edit geometry. The result reports counts and executable bounded reads for the retained selector, candidates, ambiguity and losses. It does not eagerly send every ID to the operator.

These operations share the existing service and journal. They preserve historical input/receipt bytes and never contact Blender. `native_ready` remains false. Actual producer adoption requires a bounded native-owner test against real before/after topology and preservation evidence, separate from the synthetic split/merge/composition regressions.
