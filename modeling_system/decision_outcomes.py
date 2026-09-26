"""Join retained choices to actual results and reviews; preserve historical provenance."""
from copy import deepcopy
from pathlib import Path

from .controller import read_json, fingerprint




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
            'owner_effort': 'See measured session intervals; unrecorded work remains unknown'})
    return {'cases': rows, 'case_count': len(rows),
            'with_decision': sum(row['decision'] is not None for row in rows),
            'reviewed': sum(row['review'] is not None for row in rows),
            'calibration': 'Not established; traces and scoped reviews are observations, not trained weights'}
