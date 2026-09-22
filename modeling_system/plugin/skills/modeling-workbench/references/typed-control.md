# Explicit typed capability arguments

`capability_calls.bind_call(item, selected)` binds owner-selected closed-set values
into an existing handler payload before the task enters the queue. It validates
exact option dependencies, path separation, value identity, set membership and
declared incompatibilities. It does not generate prompts or infer defaults.

```python
from modeling_system.capability_calls import bind_call

item["parameters"] = {"arguments": {
    "support": {"description": "Qualified guide support", "path": ["support"],
        "options": [{"id": "local", "description": "Source-bound local samples",
            "value": local_sample_ids, "reads": {"guide": actual_guide_revision}}]}
}}
bound_task = bind_call(item, {"support": "local"})
```

The `parameters.arguments` mapping names the payload destination and qualified
values. An argument can use `type: set` with an explicit list of option IDs;
empty sets require `allow_empty: true`. A declared default is selected only by
explicit `__default__`. Missing, incompatible or unknown choices raise before
execution. `workbench.invocation` retains the original specification and exact
choices; the queue hashes the fully bound task.

Meaningful method/observation choices remain owner decisions based on current
facts and named uncertainties. Known eligibility checks belong in code, task
prerequisites and preservation policies. Unsupported evidence stays unresolved.
Historical decision traces remain readable but never become new authorizations.
