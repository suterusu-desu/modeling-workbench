# Inspect target-domain and transition coverage

Use `inspect_target_domain(case_path, expected_state)` when a target list inherits a deformation mask or a fit-only edge screen, or saved numeric labels contain unknown values. This offline diagnostic retains the supplied inventory and its exact source files through the existing service/journal. It also runs as a pinned case operation in [diagnostic recipes](recipes.md). It does not discover visibility in Blender, qualify a target, relax a mask or apply a fit.

Read the reported distinctions separately: geometric candidates; selected fit points; authored support-mask membership; recorded response rows; numeric label availability; owner-reviewed region correspondence; and each named edge-check family. A point can have unknown numeric material coordinates while its region correspondence is independently supported. Numeric values alone also do not prove that correspondence. Fit-fit guide gradients, fit-to-context displacement terms and native quad-normal preservation are different measurements.

## Case contract

The version-1 JSON case contains `question`, `state`, `inventory`, `selection`, `response`, `support_mask`, `semantics`, `topology`, `arrays`, `checks`, and optional `exclusions` and `correspondence_reviews`. `expected_state` must match the recorded `source_state_id`, `pose`, `frame`, `units` and `dependency_fingerprint`. Preserve exact saved owner state and geometry hashes; an older diagnostic case does not automatically identify a newer baseline. This match is historical, not a live-state check.

`inventory` supplies `point_identity`, `basis`, `evidence`, `independent_of_support_mask` (boolean) and `coverage: {status, missing}`. Status is `unknown`, `declared_subset` or `complete_for_declared_screen`. The last requires an independent screen with no declared missing coverage; it does not establish anatomical completeness. Retain a declared subset when triangle interiors, other poses or views are unmeasured.

`selection`, `response`, `support_mask`, `semantics` and `topology` each need their own `meaning` and `evidence`. Evidence may contain exact hash-pinned file references. Support-mask membership uses the explicit finite `threshold` with strict `value > threshold`. `semantics.fields` names the ordered numeric label columns. Unknown labels are not zero, absent points or automatically rejected regions.

`arrays: {path, sha256}` points to an NPZ with no pickle data:

| Array | Meaning |
| --- | --- |
| `point_ids` | Distinct ordered integer or string identities, at most 250,000; reported as strings. |
| `candidate`, `selected`, `response_covered` | Independent explicit boolean vectors aligned to those points. No field is inferred from another. |
| `mask_values` | Finite authored-mask values for every point. |
| `semantic_values` | Numeric matrix `(points, fields)`. Only these values may be nonfinite: original bytes are retained; report values become explicit null with unknown field names. |
| `edges` | Unique undirected topology edge pairs using zero-based indices into ordered points, at most 1,500,000. |
| Named check arrays | Unique endpoint pairs in the same topology edge universe. Empty `(0,2)` arrays explicitly record no checks. |

Each check is `{id, meaning, evidence, edge_array, required_categories}`. Categories are `fit_fit`, `fit_context` and `context_context`, determined by selected fit membership. `required_categories` states which edge categories the diagnostic question requires this particular family to cover. Use an empty list for a narrower descriptive check whose required domain differs, and explain that domain in its meaning/evidence. Do not demand all triangulation diagonals from a native-quad metric or treat its central edge as interchangeable with the entire two-face support. The diagnostic checks endpoint membership, not measurement implementation or values.

An exclusion is `{points, reason, interpretation, evidence}` with point IDs as strings; interpretations are `protected`, `trial_restriction`, `context` or `unresolved`. Missing exclusion provenance stays visible rather than silently becoming protection. A correspondence review is `{points, status, reason, evidence}`, with `supported`, `unresolved` or `rejected` status. Current review groups must not overlap. These are supplied owner judgments for the stated region and recorded state; the diagnostic does not validate their artistic truth.

## Interpret and retain

The compact summary identifies mask-excluded candidates, unselected candidates, uncovered response rows, unknown numeric labels, absent supported correspondence and unchecked transitions. Exact `points`, `transitions`, `checks`, `case` and `limits` are available via bounded `read_record` descriptors. A union count only identifies edges with at least one named check; it never asserts equivalence between those checks.

Since 0.2.7, mask/region review suggestions target unreviewed or unresolved correspondence. A supported or rejected current correspondence is not automatically reopened because its mask membership differs or numeric labels remain unknown. Preserve the discrepancy as a fact and revisit the judgment when its actual evidence or applicability changes. Remaining geometric-screen coverage is a separate question even when all listed candidates have supported correspondence.

Use the report to prepare a justified qualification or measurement question. Preserve an original mask-filtered case and a separately reviewed expanded case as different trials. A newly supported correspondence can justify a changed selection through the existing owner workflow; this diagnostic never changes it automatically. `native_ready` and `target_admission` remain false. Geometry, guide-depth evidence, protected relationships, finite native response, connected form and appearance still require their existing independent checks.
