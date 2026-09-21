"""Join retained choices to actual results and reviews; reuse raw scores offline."""
from copy import deepcopy
from pathlib import Path
import math

from .controller import read_json, fingerprint


def composite_scores(judgments, weights, *, violations=None):
    """Combine compensating ordinal preferences, preserving any material violation.

    Weights and violation booleans are explicit policy, not inferred thresholds.
    Missing dimensions remain unknown. No inference, calibration or native IO.
    """
    if (not weights or any(type(w) not in (int, float) or not math.isfinite(w) or w < 0
                           for w in weights.values()) or sum(weights.values()) <= 0):
        raise ValueError('Finite nonnegative weights with positive total required')
    rows = []
    for candidate, dimensions in judgments.items():
        values, missing = {}, []
        for name, weight in weights.items():
            if weight == 0:
                continue
            answer = dimensions.get(name)
            if not answer:
                missing.append(name)
                continue
            legend = answer.get('legend', {})
            from .judgments import validate_answers
            if set(legend) != {str(i) for i in range(len(legend))}:
                raise ValueError('Score rubric must have ordered ordinal levels')
            validate_answers({'score': {'type': 'score', 'instructions': 'Retained score',
                'criteria': [legend[str(i)] for i in range(len(legend))]}}, {'score': answer})
            values[name] = answer['score'] / (len(legend) - 1)
        violation = (violations or {}).get(candidate, False)
        if type(violation) is not bool:
            raise ValueError('Material violations require an explicit policy boolean')
        total = None if missing else sum(values[n] * weights[n] for n in values) / sum(weights.values())
        rows.append({'candidate': candidate, 'score': total, 'missing': missing, 'material_violation': violation,
                     'normalized': values, 'raw_judgments': deepcopy(dimensions)})
    return sorted(rows, key=lambda r: (r['material_violation'], r['score'] is None,
                                      -(r['score'] or 0), r['candidate']))


def decision_outcomes(directory, reviews=None):
    directory = Path(directory)
    queue = read_json(directory / 'work-queue.json')
    reviews = reviews or {}
    rows = []
    for task, entry in queue['results'].items():
        decision = entry.get('decision')
        trace = None
        if decision:
            trace_path = directory / 'advice' / 'decisions' / (decision['selection_key'] + '.json')
            trace = read_json(trace_path)
            if (trace.get('choice') != task or trace.get('selection_key') != decision['selection_key']
                    or fingerprint(trace) != decision['trace_revision']):
                raise ValueError('Decision trace does not match the actual executed task')
        review = reviews.get(task)
        rows.append({'task': task, 'definition': entry['definition'], 'decision': trace,
            'execution_status': entry['status'], 'operation_handle': entry.get('operation_handle'),
            'actual_report': deepcopy(entry.get('result', {}).get('workbench')),
            'review': deepcopy(review), 'review_state': 'recorded' if review else 'unreviewed',
            'failure_class': 'execution' if entry['status'] == 'failed' else
                'uncertain_effect' if entry['status'] == 'needs_reconciliation' else
                'appearance_rejected' if review and review['judgment']['disposition'] == 'rejected' else None,
            'astra_effort': 'See measured session intervals; unrecorded work remains unknown'})
    return {'cases': rows, 'case_count': len(rows),
            'with_decision': sum(row['decision'] is not None for row in rows),
            'reviewed': sum(row['review'] is not None for row in rows),
            'calibration': 'Not established; traces and scoped reviews are observations, not trained weights'}
