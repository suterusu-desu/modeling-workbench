# Development and contribution boundaries

[System design and workflow composition](DESIGN.md) and the [finite integration queue](IMPLEMENTATION-QUEUE.md) distinguish implemented interfaces, installed verification, actual operator use and unsupported content.

Install in an isolated Python environment with `python -m pip install .`, or `python -m pip install -e .` for an
editable install that follows the checkout. Run `python -m unittest discover -s modeling_system -t .` and `python scripts/check_distribution.py` and `python scripts/check_clean_install.py` before publishing source changes. The clean check installs the wheel and tools archive into new environments outside the checkout, strips inherited credentials/workspace configuration, exercises the shipped direct integration without inference services, and probes the generated plugin. CI runs on Windows, macOS and Linux. Native, provider and artistic acceptance are separate from this synthetic suite; set
`MODELING_BLENDER` to a Blender executable to also run the tests that drive a background Blender (the live bridge and
reference adapter, the isolated workers, trial / reopen / retain).

A release is one line and the checks: set `__version__` in `modeling_system/__init__.py` (packaging, the prepared
plugin manifest and the MCP server read it; no other file carries the version), list any new package file in
`modeling_system/distribution-files.json`, run the suite and both checks, commit and push. Ship a capability when the
next real edit uses it, and batch small changes into one release rather than one release per function.

The workbench environment and Blender's embedded Python are separate dependency domains. Do not add the workbench environment's entire `site-packages` directory to a native worker's search path. Compiled extensions must support the worker's Python version, ABI and platform; importing them successfully in offline preparation does not verify native compatibility. See [Blender's bundled-Python guidance](https://github.com/blender/blender/blob/main/doc/python_api/rst/info_tips_and_tricks.rst#bundled-python--extensions) and [Python wheel compatibility tags](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/).

Keep SciPy-heavy fitting and analysis in recorded-array preparation and pass immutable results to the native capability. For small native geometry queries, prefer Blender-provided facilities such as `mathutils.kdtree`. If a new native dependency is necessary, qualify its imports in the actual worker runtime through the existing controller before scene effects, and retain that runtime identity with the capability. An import failure is a failed operation; source and effect receipts determine whether a corrected attempt can proceed. A selected task alone does not establish native execution.

Normalize working text to repository line endings before building and pinning an installation. The distribution check compares working bytes with Git filters so a committed snapshot cannot silently differ from the tested wheel merely because a metadata writer emitted CRLF. Keep earlier materializations immutable; compare semantic metadata and exact executable bytes when a final packaging correction changes only text encoding or prose.

Keep character workspaces outside the source checkout. Do not commit project bindings, assets, generated images, models, recordings, job/episode IDs, personal paths, private history, local adapter configuration or credentials. Synthetic examples must be authored independently of private geometry. Source distribution uses an explicit package inventory and never exports the selected workspace by default.

Retain exact experiments privately. Promote the general mechanism, supported conditions, limitations and counterexamples into the method; add a synthetic regression where it tests meaningful behavior. Local improvement is not universal validation. This is the feedback loop through which the shared tools become more capable without publishing private character history.

Use the packaged [method integration procedure](modeling_system/plugin/skills/modeling-workbench/references/method-integration.md) for consequential method close-out. Evidence retention, procedure promotion, mechanism-based retrieval and a fresh use check form one linked workflow. Narrow diagnostic benefits can be supported while the character remains unresolved. Keep source, installation, runtime and operator adoption receipts separate, including pending or inapplicable stages. The synthetic integration test demonstrates this route without native or provider work.

The initial portability boundary preserves the generic core and a native adapter protocol. Further native adapter implementations, first-class local guide derivative review, and real cross-character quality studies remain explicit work. New implementations should retain sole ownership, freshness, complete influenced regions, guide/depth constraints, recovery and separate appearance judgment.

Existing private projects must switch deliberately: pin the exact package/interpreter, bind the private adapter and dependencies, verify schemas and source, run an isolated contract check, then use a bounded actual episode through its native owner. Do not hot-swap a running modeling session during an unrelated guide trial.
