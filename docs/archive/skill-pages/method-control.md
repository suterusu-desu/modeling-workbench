# Executable methods

`MethodCatalog` registers qualified preparation, diagnosis and recovery mechanisms
once, then binds them to the current scope. Each method has an ID, description,
`build(state, previous)` factory and optional observed `conditions`.
`ParameterizedCatalog` binds registered implementations to data without copying
factories; duplicate mechanism/parameter aliases are rejected.

The owner explicitly chooses among eligible methods through the session's order
or selector. Missing or contradictory conditions exclude a method. Use actual
task prerequisites for named observations and recovery, with evidence from
completed outcomes; inference-only `method_checks` and `remedy_methods` are not
supported. This prevents an unknown applicability claim from becoming an edit.

`catalog.followup(stage)` calls the chosen method's declared next factory with its
actual task/result, preserving original parameters. It does not choose a new
method or retry a failed variant. CandidatePipeline freezes the materialized
stages and enforces evidence/review dependencies.

Use `ArrayPreparation` for saved-array fitting, correspondence, coverage and
effect measurement. See [operation recipes](operation-recipes.md) and
[preparation contracts](preparation-and-bootstrap.md). Numerical success cannot
approve likeness. Retain useful candidates only after actual matched review and
independent reopen as required by the scope.

`RetainedContext` retrieves exact source passages and conditional local lessons
into bounded owner observations. Keep missing coverage, contradictory evidence
and current-vs-historical status visible. Ranking by retrieval relevance is not
model inference or evidence qualification. Read the original linked source when
the compact context cannot settle a consequential question.

Use intervention and review timers to measure actual workload. Session metrics
count explicit owner choices separately from singleton/fixed execution, report
native stages and identify missing intervals. They do not infer saved reasoning
turns or a speedup from incomplete observations.
