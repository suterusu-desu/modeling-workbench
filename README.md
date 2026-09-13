# Modeling Workbench

Reusable tools and an evidence-based workflow for character modeling: exact recorded geometry, matched observations, guide/depth qualification, coupled response analysis, recoverable trials and retained learning.

[Start here](modeling_system/plugin/skills/modeling-workbench/references/getting-started.md) ? [Retained method](modeling_system/plugin/skills/modeling-workbench/references/method.md) ? [Native adapters](modeling_system/plugin/skills/modeling-workbench/references/native-adapters.md) ? [Agent skill](modeling_system/plugin/skills/modeling-workbench/SKILL.md)

The tools repository contains source, generic methods and synthetic tests. Characters have separate private workspaces containing assets, identity references, rig/object/semantic bindings, current decisions and full experiment history. `python -m modeling_system.init_workspace PATH --character LABEL` creates that boundary explicitly.

Python, CLI and MCP expose the same core. Native Blender operations require a separately configured and verified workspace adapter. This package preserves the modeling process and its contracts; it does not claim identical results or validated native control on every character. Broader quality requires actual trials across different characters.

Tools-only export is the default. Explicit workspace/evidence transfers remain private. See [development and contribution boundaries](DEVELOPING.md).
