# Development and contribution boundaries

[System design and workflow composition](DESIGN.md) and the [finite integration queue](IMPLEMENTATION-QUEUE.md) distinguish implemented interfaces, installed verification, actual operator use and unsupported content.

Install in an isolated Python environment with `python -m pip install .`. Run `python -m unittest discover -s modeling_system -t .` and `python scripts/check_distribution.py` before publishing source changes. Native, provider and artistic acceptance are separate from this synthetic suite.

Normalize working text to repository line endings before building and pinning an installation. The distribution check compares working bytes with Git filters so a committed snapshot cannot silently differ from the tested wheel merely because a metadata writer emitted CRLF. Keep earlier materializations immutable; compare semantic metadata and exact executable bytes when a final packaging correction changes only text encoding or prose.

Keep character workspaces outside the source checkout. Do not commit project bindings, assets, generated images, models, recordings, job/episode IDs, personal paths, private history, local adapter configuration or credentials. Synthetic examples must be authored independently of private geometry. Source distribution uses an explicit package inventory and never exports the selected workspace by default.

Retain exact experiments privately. Promote the general mechanism, supported conditions, limitations and counterexamples into the method; add a synthetic regression where it tests meaningful behavior. Local improvement is not universal validation. This is the feedback loop through which the shared tools become more capable without publishing private character history.

Use the packaged [method integration procedure](modeling_system/plugin/skills/modeling-workbench/references/method-integration.md) for consequential method close-out. Evidence retention, procedure promotion, mechanism-based retrieval and a fresh use check form one linked workflow. Narrow diagnostic benefits can be supported while the character remains unresolved. Keep source, installation, runtime and operator adoption receipts separate, including pending or inapplicable stages. The synthetic integration test demonstrates this route without native or provider work.

The initial portability boundary preserves the generic core and a native adapter protocol. Further native adapter implementations, first-class local guide derivative review, and real cross-character quality studies remain explicit work. New implementations should retain sole ownership, freshness, complete influenced regions, guide/depth constraints, recovery and separate appearance judgment.

Existing private projects must switch deliberately: pin the exact package/interpreter, bind the private adapter and dependencies, verify schemas and source, run an isolated contract check, then use a bounded actual episode through its native owner. Do not hot-swap a running modeling session during an unrelated guide trial.
