# Worker completion and execution receipt contracts

Two verified wrapper mistakes motivate this pair of tools. A wrapper required a universal `status == "completed"` plus an inline candidate although the adapter had succeeded and returned an operation-specific status with a durable receipt path. A custom reopen worker exited zero and wrote correct arrays and a report but omitted the completion file its launcher requires, so the launcher classified the run as failed. The wrapper failures do not by themselves establish native failure; native correctness needs its own evidence. `worker_contract.WorkerContract` makes the launcher-required completion file the last verified step of an owner-authored worker. `inspect_execution_receipt(case_path)` reads a pinned result/receipt pair against an explicit adapter contract and keeps each disposition separate. Neither contacts Blender, replays, retries or clears a failure.

## Worker completion contract

Declare the contract before work, run `preflight()`, record named checks while working, and call `finalize()` once. The completion file is written atomically only after every required artifact exists inside the output directory, every required check was recorded as a literal `True`, no recorded check failed, and every protected source still hashes to its declaration. If finalization detects an unsatisfied contract, a failure report is written and `ContractFailure` is raised; with `--python-exit-code` the process exits nonzero, no completion file exists, and the launcher classifies the run as failed for the right reason.

```python
import sys, json, os
from pathlib import Path
sys.path.insert(0, JOB['workbench_source'])                 # pinned checkout or installed package parent
from modeling_system.worker_contract import WorkerContract  # or load worker_contract.py as a file

contract = WorkerContract(OUT_DIR, completion_file='inventory.json',
    required_artifacts=['preflight.json', 'source-inventory.json', 'head-arrays.npz', 'candidate.blend',
                        {'name': 'source.fbx', 'sha256': JOB['asset_sha256'], 'role': 'copied pinned source'}],
    required_checks=['autoexec_disabled', 'source_hash_verified', 'head_mesh_present'],
    protected_sources=[{'path': JOB['source_asset'], 'sha256': JOB['asset_sha256'], 'role': 'pinned source asset'}],
    label=JOB['case'])
preflight = contract.preflight()                            # refuses before native work
with (Path(OUT_DIR) / 'preflight.json').open('x', encoding='utf-8') as f:
    json.dump(preflight, f, allow_nan=False)
    f.flush(); os.fsync(f.fileno())                          # retain before a native crash can erase context
contract.check('autoexec_disabled', not bpy.context.preferences.filepaths.use_scripts_auto_execute)
...                                                         # import, inventory, save; use contract.fail(reason) for an explicit refusal
contract.check('source_hash_verified', sha(source) == JOB['asset_sha256'], detail={'sha256': JOB['asset_sha256']})
contract.check('head_mesh_present', head is not None)
contract.finalize(extra={'status': 'source_setup_inventoried', 'case': JOB['case']})
raise SystemExit(0)
```

### Save geometry before rendering

A successful preflight return and in-memory check records do not survive a hard process crash by themselves. Retain the preflight result as a separately named required artifact before native effects. Keep each invocation in a new isolated output directory; an existing preflight file is evidence to reconcile, not a reason to truncate it. A partial preflight file is not completion proof.

Save and verify the candidate and numerical roundtrip outputs before entering rendering or another failure-prone stage. Persist the exact checkpoint path/hash and completed numerical checks as their own intermediate record. Finalize the full worker only after its declared renders and other outputs exist and checks pass. A saved checkpoint or preflight record must never be called the final inventory for an unfinished run. A hard native crash may bypass Python exception handling and produce neither a contract failure report nor a completion manifest; inspect the original launcher receipt and independently verify whatever durable artifacts survived.

If rendering fails, preserve that invocation and its failed receipt. The native owner may use the saved candidate for a separately identified render/reopen continuation under existing scope, with input hashes and a contract matching that continuation. Do not reconstruct successful completion for the original run, silently reuse its output directory, or replay geometry just to regenerate a completion marker. Rendering route selection remains environment-specific and must be verified by the owner.

Declarations are validated in the constructor: `completion_file` (single filename), a nonempty `required_artifacts` list, an explicit `required_checks` list and an explicit `protected_sources` list are all required, so a missing declaration fails before work rather than after it. Artifact names are bounded relative POSIX names: no drive, backslash, absolute prefix, empty, `.` or `..` segment, more than eight segments or 200 characters. At finalize each name is resolved under the output directory and refused if any segment is a symbolic link or junction or the resolved path leaves the directory. Names may carry an expected `sha256`, which binds a copied import to its pinned source.

`protected_sources` are independent of the launcher's optional `.blend` input: a launcher that received no input still leaves the worker responsible for the asset it copied and imported. Paths are absolute, or relative to an explicit `source_root`. Preflight hashes every source and refuses to start on a mismatch; finalize hashes them again before writing. A worker check that failed cannot be cleared by a later pass under the same name. `artifact(name, role)` attaches a role or declares an additional produced file that must also exist. `finalize(extra=...)` merges owner fields such as an operation-specific `status`; reserved contract keys are refused.

The success manifest has `schema_version: 1`, `kind: worker_completion`, `contract_status: satisfied`, the declared `contract`, per-artifact `path`, `sha256`, `bytes`, `role`, `required` and `status`, recorded `checks` with details, verified `sources`, the resolved `output_dir`, `label` and `utc`. A repeated finalization with an identical outcome returns the existing manifest without rewriting it. A differing existing completion file, link or directory is preserved untouched, a failure report is written and `ContractConflict` is raised. Failure reports (`kind: worker_contract_failure`) never reuse a name; they list `missing_preflight`, `missing_artifact`, `artifact_hash_mismatch`, `invalid_artifact_path`, `unrecorded_check`, `failed_check`, `changed_source`, `missing_source`, `worker_failure` or `conflicting_existing_manifest`. `verify_manifest(path)` re-verifies an existing manifest's artifact bytes, sources and checks without writing.

## Execution receipt inspection

`inspect_execution_receipt(case_path)` pins a version-1 case through the existing store and returns an `execution_receipt` record with bounded `read_record` descriptors. The public service exposes this operation; the module has no default statuses, so every recognized status and its effect is supplied evidence from the owner's verified adapter contract, never independent native proof.

The case contains `schema_version: 1`, `question`, `operation` (`kind` of `native_operation` or `worker_run`, `name`, `meaning`, `evidence`), `result` as an exact `{path, sha256}` reference to the document the caller received, an optional `receipt` reference to the durable expanded receipt, an optional `independent_verification` list of `{path, sha256, role}` references, and `adapter`:

| Field | Meaning |
| --- | --- |
| `id`, `meaning`, `evidence` | Which verified adapter/launcher contract this mapping was read from. |
| `statuses` | Recognized status strings mapped to `{effect, meaning}`; effects are `applied`, `completed`, `not_applied`, `failed` or `unknown`. |
| `status_path` | Optional JSON paths for the result and receipt status (default `["status"]`); `receipt: null` skips a receipt that carries no operation status. |
| `receipt` | `{required, reference}`: whether a durable receipt is required and where the result names its path. |
| `identity_checks` | `{id, result, receipt}` value agreement or `{id, result|receipt, equals}` constants. |
| `artifacts` | `{id, in, path, required}` JSON paths to `{path, sha256}` references whose bytes are hashed now. |
| `process` | `{in, returncode, success_codes}` for launcher receipts. |
| `required_outputs` | `{in, directory, names}`: launcher-required files in the recorded output directory; names are bounded like artifact names. |
| `source_preservation` | `{in, unchanged, before, after}` launcher hash fields. |
| `deferred`, `record_reference` | Fields a compact result defers by design, and a store record key to link. |

Dispositions stay separate in the record: `effect` (mapped status plus reasons), `identity`, `artifacts` (`verified`, `mismatch`, `missing`, `unreferenced`, `invalid_reference`), `process` (`exit_ok`, `exit_failed`, `unknown`, `not_applicable`), `required_outputs` (`complete`, `incomplete`, `unknown`, `not_applicable`), `source_preservation` (`unchanged`, `changed`, `unknown`, `not_applicable`) and `completion_manifest` (a present worker manifest re-verified now). The compact `disposition` is `verified_consistent` only when a success effect is uncontradicted and at least one artifact or the completion manifest verified from bytes; `consistent_unverified` when only a status label supports success; `incomplete` when a launcher-required output or required artifact is absent, including exit zero without the completion file; `failed` when the contract itself classified failure; `needs_reconciliation` for an unrecognized, conflicting, unknown-mapped or missing-receipt effect, a receipt reference disagreement, mismatched bytes, a changed source under a success claim, a failed or unknown exit under a success claim, or a manifest that no longer verifies.

`success_established` is true only for `verified_consistent`. `safe_to_replay`, `native_ready` and `appearance_accepted` are always false: an unexpected post-dispatch shape routes to reconciliation of the original operation, never to a retry. Independent verification records are linked and listed with their bearing but never change the original receipt's disposition, and the original files are only read. Use the result to expand a compact result through its receipt, to see that a missing launcher file is a contract gap rather than a native failure, and to decide the next explicit operation with the owner.

Completion publication uses an exclusive atomic filesystem link from a flushed temporary file. A concurrent completion file is preserved; unsupported filesystem link semantics fail explicitly. This is a local trusted-worker contract, not cryptographic attestation. Verification checks declarations against the manifest as well as current bytes. A declaration cannot prove the time at which an untrusted worker actually ran preflight. Configured but unknown output or source fields keep success incomplete. `success_established` means consistency with the declared adapter contract and verified evidence, not independently proven native correctness.
