"""Observed workload accounting; no invented counterfactual speedup."""
from collections import Counter
from pathlib import Path
import json
from .controller import read_json

INTERVENTION_KINDS = {'correspondence_preparation', 'capability_development',
                      'catalog_preparation', 'recovery', 'evidence_interpretation'}


def session_metrics(directory):
    directory = Path(directory)
    path = directory / 'work-queue.json'
    queue = read_json(path) if path.exists() else {'results': {}}
    totals, actions, incomplete = Counter(), Counter(), 0
    events = directory / 'controller/events.jsonl'
    if events.exists():
        for line in events.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                incomplete += 1
                continue
            if row.get('event') == 'cycle_timing':
                totals.update(row['timings_ms'])
                if row.get('action'):
                    actions[row['action']] += row['timings_ms'].get('execute', 0)
    interventions = [read_json(p) for p in (directory / 'interventions').glob('*.json')]
    reviews = [read_json(p) for p in (directory / 'reviews').glob('*.json')]
    results = queue['results']
    native_ms = sum(row.get('result', {}).get('handler_elapsed_ms', 0)
                    for row in results.values() if row.get('task', {}).get('workbench', {}).get('native'))
    missing_native_times = sum('handler_elapsed_ms' not in row.get('result', {})
        for row in results.values() if row.get('task', {}).get('workbench', {}).get('native'))
    retained = [key for key, row in results.items() if row['status'] == 'completed'
        and row.get('task', {}).get('workbench', {}).get('profile') == 'retention'
        and row.get('result', {}).get('workbench', {}).get('qualification') == 'complete']
    return {'cycle_totals_ms': dict(totals), 'execution_by_task_ms': dict(actions),
        'native_handler_ms': round(native_ms, 3), 'native_tasks_without_timing': missing_native_times,
        'retained_tasks': retained,
        'outcomes': dict(Counter(row['status'] for row in results.values())),
        'astra_interventions': dict(Counter(row['kind'] for row in interventions)),
        'reported_astra_seconds': sum(row.get('seconds') or 0 for row in interventions),
        'interventions_without_duration': sum(row.get('seconds') is None for row in interventions),
        'visual_review_submissions': len(reviews), 'incomplete_event_lines': incomplete,
        'avoided_astra_turns': None,
        'coverage': 'Controller times and review submissions are automatic. Other Astra work is explicitly reported; unreported time is unknown. Native handler time includes its prerequisites. Cycle times overlap component times; do not add them together. Retention is not user acceptance. No baseline speedup is inferred.'}
