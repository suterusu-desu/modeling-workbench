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
    totals, actions, routes, lanes, incomplete = Counter(), Counter(), Counter(), Counter(), 0
    choices = []
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
            elif row.get('event') == 'action_route':
                routes[row['route']] += 1
                if row['route'] == 'owner_selection':
                    lanes[row.get('lane', 'unspecified')] += 1
                    choices.append({k: row.get(k) for k in ('action', 'lane', 'offered')})
    interventions = [read_json(p) for p in (directory / 'interventions').glob('*.json')]
    reviews = [read_json(p) for p in (directory / 'reviews').glob('*.json')]
    timers = [read_json(p) for p in (directory / 'intervention-timers').glob('*.json')]
    results = queue['results']
    review_work = [read_json(p) for p in (directory / 'review-work').glob('*.json')]
    timed_reviews = {row.get('review') for row in review_work if row.get('completed')}
    native_stages, native_calls, implementations = Counter(), Counter(), Counter()
    for row in results.values():
        native_stages.update(row.get('result', {}).get('native_stages_ms', {}))
        native_calls.update(row.get('result', {}).get('native_calls', {}))
        recipe = row.get('task', {}).get('workbench', {}).get('recipe')
        if recipe:
            implementations[recipe['implementation']] += 1
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
        'native_stages_ms': dict(native_stages), 'native_calls': dict(native_calls),
        'checkpoint_opens_reused': sum(row.get('result', {}).get('checkpoint_open_reused') is True for row in results.values()),
        'parameterized_implementations': dict(implementations),
        'tasks_without_recipe_identity': sum(not row.get('task', {}).get('workbench', {}).get('recipe') for row in results.values()),
        'outcomes': dict(Counter(row['status'] for row in results.values())),
        'owner_interventions': dict(Counter(row['kind'] for row in interventions)),
        'reported_owner_seconds': sum(row.get('seconds') or 0 for row in interventions),
        'interventions_without_duration': sum(row.get('seconds') is None for row in interventions),
        'visual_review_submissions': len(reviews), 'incomplete_event_lines': incomplete,
        'visual_reviews_without_timing': sum(row.get('operation_handle') not in timed_reviews for row in reviews),
        'action_routes': dict(routes), 'owner_choices_by_lane': dict(lanes), 'owner_choices': choices,
        'open_intervention_timers': [row['token'] for row in timers if row['status'] == 'running'],
        'timed_interventions': sum(row.get('seconds') is not None for row in interventions),
        'avoided_owner_turns': None,
        'coverage': 'Controller times, routes, parameterized implementation use, native recipe stages and review submissions are automatic. Routes record explicit owner selections and fixed continuations. Historical route names remain unchanged in action_routes. begin_review measures evidence opening through submission; missing timers are counted explicitly. Other explicit timers measure elapsed work, not model tokens. Unreported effort remains unknown. Native and cycle totals overlap components. Retention is not user acceptance; no speedup is inferred.'}


def compare_sessions(baseline, current, *, comparison_scope):
    """Observed difference for an explicitly declared comparison, not causality."""
    if not isinstance(comparison_scope, str) or not comparison_scope.strip():
        raise ValueError('Describe why these scopes are comparable')
    before, after = session_metrics(baseline), session_metrics(current)
    known = all(not row['interventions_without_duration'] and not row['open_intervention_timers']
                and not row['visual_reviews_without_timing']
                and row['timed_interventions'] for row in (before, after))
    return {'comparison_scope': comparison_scope, 'baseline': before, 'current': after,
            'reported_owner_seconds_difference': after['reported_owner_seconds'] - before['reported_owner_seconds'] if known else None,
            'limits': 'Only recorded intervals and outcomes are compared. Different model quality or task difficulty can explain differences. Unreported work and avoided turns remain unknown.'}
