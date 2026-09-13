"""Offline reverse structural coverage; never infer controllability from ancestry."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from .repair_analysis import DOMAINS, _pin, _load


def _ids(values, name, universe=None):
    if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
            or len(values) != len(set(values))):
        raise ValueError(name + ' must contain distinct nonempty string IDs')
    result = set(values)
    if universe is not None and not result <= universe:
        raise ValueError(name + ' contains unknown IDs')
    return result


def inspect(service, case_path, expected_state):
    path = Path(case_path).resolve()
    original = service.store.blob(path)
    raw = json.loads(service.store.resolve_blob(original).read_text(encoding='utf-8-sig'))
    case = _pin(service, raw, path.parent, {})
    if case.get('schema_version') != 1 or not case.get('question'):
        raise ValueError('Version-1 control coverage case with a question required')
    state = case.get('state', {})
    if any(d not in state or d not in expected_state or state[d] != expected_state[d] for d in DOMAINS):
        raise ValueError('Control coverage and expected recorded state/pose/domain differ')
    ancestry = case.get('ancestry', {})
    if not ancestry.get('evidence') or ancestry.get('kind') != 'structural_dependency':
        raise ValueError('Ancestry needs structural_dependency semantics and evidence; a zero derivative is not absence of ancestry')
    if any(ancestry.get(d) != state[d] for d in DOMAINS):
        raise ValueError('Ancestry must bind the same recorded state/pose/domain')
    outputs = case.get('outputs', [])
    controls = case.get('controls', [])
    if not 0 < len(outputs) <= 250000 or not 0 < len(controls) <= 100000:
        raise ValueError('Coverage supports 1..250000 outputs and 1..100000 controls')
    out_ids = _ids([o['id'] for o in outputs], 'outputs')
    control_ids = _ids([c['id'] for c in controls], 'controls')
    if any(not o.get('meaning') for o in outputs) or any(not c.get('meaning') for c in controls):
        raise ValueError('Explicit native/measurement and control meanings required')
    targets = _ids(case.get('targets'), 'targets', out_ids)
    selected = _ids(case.get('selected_controls'), 'selected_controls', control_ids)
    allowed = _ids(case.get('allowed_controls'), 'allowed_controls', selected)
    if not targets:
        raise ValueError('At least one exact target output required')
    target_scope = case.get('target_scope', dict(status='unknown', missing=['Target-domain coverage was not supplied']))
    if target_scope.get('status') not in ('unknown', 'declared_subset', 'complete_for_question'):
        raise ValueError('Target scope must be unknown, declared_subset or complete_for_question')
    if target_scope['status'] == 'complete_for_question' and (
            not target_scope.get('basis') or not target_scope.get('evidence') or target_scope.get('missing') != []):
        raise ValueError('Complete target scope needs a basis, evidence and explicit empty missing coverage')
    exclusions = case.get('exclusions', [])
    excluded = _ids([e['control'] for e in exclusions], 'exclusions', control_ids)
    if excluded & allowed or any(not e.get('reason') or not e.get('evidence') or
            e.get('interpretation') not in ('protected', 'trial_restriction', 'unresolved') for e in exclusions):
        raise ValueError('Excluded controls need reasons, evidence and protected/trial_restriction/unresolved interpretation')
    if not selected - allowed <= excluded:
        raise ValueError('Every selected but disallowed control needs an explicit exclusion')
    coverage = ancestry.get('coverage', {})
    complete_rows = _ids(coverage.get('complete_for_outputs', []), 'complete_for_outputs', out_ids)
    if type(coverage.get('universe_complete')) is not bool or not isinstance(coverage.get('missing'), list):
        raise ValueError('Declare control-universe completeness and missing coverage')
    if not all(ancestry.get('arrays', {}).get(key) for key in ('path', 'sha256', 'asset')):
        raise ValueError('Hash-pinned ancestry arrays required')
    z = _load(service, ancestry['arrays'])
    n, k = len(outputs), len(controls)
    if np.asarray(z['shape']).tolist() != [n, k]:
        raise ValueError('Ancestry CSR shape must match ordered outputs and controls')
    indptr, indices, data = z['indptr'], z['indices'], z['data']
    if (indices.dtype.kind not in 'iu' or indptr.dtype.kind not in 'iu'
            or indptr.shape != (n+1,) or indices.ndim != 1 or data.shape != indices.shape
            or len(indices) > 10000000 or indptr[0] != 0 or indptr[-1] != len(indices)
            or np.any(np.diff(indptr.astype(np.int64)) < 0) or np.any(indices >= k)
            or np.any(indices < 0) or not np.all(data == 1)):
        raise ValueError('Finite bounded structural CSR requires integer indices and one per known dependency, no numeric response weights')
    matrix = sparse.csr_matrix((np.ones(len(indices), dtype=bool), indices, indptr), shape=(n, k))
    matrix.sum_duplicates()
    matrix.sort_indices()
    names = [c['id'] for c in controls]
    omitted_union, potential_union = set(), set()
    rows = []
    for i, output in enumerate(outputs):
        if output['id'] not in targets:
            continue
        parents = {names[j] for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]}
        omitted = parents - selected
        potential = omitted - excluded
        omitted_union |= omitted
        potential_union |= potential
        complete = (coverage['universe_complete'] and output['id'] in complete_rows
                    and not coverage['missing'])
        rows.append(dict(output=output, complete=complete, ancestors=sorted(parents),
                         selected_ancestors=sorted(parents & selected),
                         allowed_ancestors=sorted(parents & allowed),
                         omitted_ancestors=sorted(omitted),
                         excluded_ancestors=sorted(parents & excluded),
                         unclassified_omitted_ancestors=sorted(potential)))
    reverse_complete = all(r['complete'] for r in rows)
    summary = dict(targets=len(rows), declared_controls=k, selected_controls=len(selected),
                   allowed_controls=len(allowed), known_omitted_ancestors=len(omitted_union),
                   unclassified_omitted_ancestors=len(potential_union),
                   targets_with_omitted_ancestors=sum(bool(r['omitted_ancestors']) for r in rows),
                   targets_without_allowed_ancestors=sum(not r['allowed_ancestors'] for r in rows),
                   incomplete_targets=sum(not r['complete'] for r in rows),
                   reverse_coverage_complete=reverse_complete,
                   target_scope_status=target_scope['status'])
    status = ('incomplete_ancestry' if not reverse_complete else
              'omitted_ancestors' if omitted_union else 'selected_covers_declared_target_ancestry')
    actions = []
    if target_scope['status'] != 'complete_for_question':
        actions.append('Inventory the guide-supported target domain; this reverse report covers only the listed targets, not every point that should be fitted.')
    if not reverse_complete:
        actions.append('Recover full target-to-control ancestry; a matrix restricted to inherited controls cannot prove reverse completeness.')
    if omitted_union:
        actions.append('Classify omitted controls against actual protected relationships and provisional trial restrictions before changing the family.')
        actions.append('Expand forward influence for omitted controls beyond the declared output rows before judging their preservation or admissibility.')
    if summary['targets_without_allowed_ancestors']:
        actions.append('Inspect targets with no allowed ancestor; current restrictions remove their known structural paths. Do not relax protection automatically.')
    actions.append('Measure the qualified coupled response and fit/preservation feasibility for any revised family, then review connected form in native geometry.')
    payload = dict(schema_version=1, question=case['question'], state=state, case=case,
                   case_source=original, status=status, summary=summary, targets=rows, target_scope=target_scope,
                   omitted_ancestors=sorted(omitted_union),
                   unclassified_omitted_ancestors=sorted(potential_union),
                   next_actions=actions, native_ready=False, controllability='not_established',
                   limits=['Recorded structural ancestry only; no live freshness or native mutation.',
                           'An ancestor can have zero current derivative or be blocked by downstream support.',
                           'Complete ancestry does not prove rank, useful finite control freedom, feasibility, smoothness or appearance.',
                           'Omission does not grant permission to edit; exclusions retain their stated interpretation.',
                           'Consequences of omitted controls outside the declared output rows have not been measured by this report.',
                           'Forward influence and actual native prediction agreement are independent of reverse coverage.',
                           'No new guide or generation is justified by this report alone.'])
    record = service.store.put('control_coverage', payload)
    return dict(analysis=record, status=status, summary=summary, next_actions=actions,
                native_ready=False, controllability='not_established',
                reads={key: dict(operation='read_record', arguments=dict(record=record, path=[key], limit=10, max_chars=8000))
                       for key in ('targets', 'target_scope', 'omitted_ancestors', 'unclassified_omitted_ancestors', 'case', 'limits')})
