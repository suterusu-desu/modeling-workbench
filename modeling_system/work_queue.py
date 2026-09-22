"""Dependency-aware work across modeling lanes, executed by one native owner.

The owner supplies qualified operations, priorities and observed revisions;
this module handles durable completion, prerequisites and exact invalidation.
A catalog is authorized work, not a source of new modeling authority.
"""
from copy import deepcopy
from pathlib import Path
from .controller import PersistentController, fingerprint, read_json, write_json
from .work_limits import work_limits

LANES = ('diagnosis', 'repair', 'verification', 'review', 'recovery',
         'experience', 'preparation')
class OwnerSelection:
    """Choose only an explicitly ordered eligible task; never infer a priority."""
    def __init__(self, task_order=None):
        if task_order is not None and (not isinstance(task_order, (list, tuple))
                or any(not isinstance(key, str) or not key for key in task_order)
                or len(set(task_order)) != len(task_order)):
            raise ValueError('Task order must contain distinct task IDs')
        self.task_order = tuple(task_order or ())

    def __call__(self, state, actions, plan):
        eligible = {action['id'] for action in actions}
        for key in self.task_order:
            if key in eligible:
                return key
        return {'status': 'needs_review',
                'reason': 'Choose an eligible task explicitly or supply an authorized task order',
                'evidence': [{'task': row['id'], 'description': row['description']} for row in actions]}


class WorkQueue:
    def __init__(self, directory, *, owner, goal, observe_context, catalog, handlers, select=None, task_order=None, budget=None):
        self.limits = work_limits(budget)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner, self.goal = owner, deepcopy(goal)
        self.observe_context, self.catalog = observe_context, catalog
        if select is not None and task_order is not None:
            raise ValueError('Supply a selector or task order, not both')
        self.handlers, self.select = handlers, select if select is not None else OwnerSelection(task_order)
        self.path = self.directory / 'work-queue.json'
        self.record = read_json(self.path) if self.path.exists() else {
            'owner': owner, 'goal': fingerprint(goal), 'results': {}}
        if self.record['owner'] != owner or self.record['goal'] != fingerprint(goal):
            raise ValueError('Queue belongs to a different owner or goal')
        self.items = {}

    def _context(self):
        state = deepcopy(self.observe_context())
        if state.get('owner') != self.owner or not state.get('authority_revision'):
            raise ValueError('Current owner and authority required')
        if not isinstance(state.get('values'), dict) or not state['values']:
            raise ValueError('Fresh observed revisions required')
        if not isinstance(state.get('active_operations'), list):
            raise ValueError('Actual execution lane state required')
        scope = fingerprint({'goal': self.goal, 'authority': state['authority_revision']})
        state['values']['work_queue_scope'] = scope
        state['stage'] = 'shared modeling work queue'
        return state

    def observe(self):
        state = self._context()
        if any(r.get('status') in ('running', 'needs_reconciliation') for r in self.record['results'].values()):
            raise ValueError('Uncertain queue effect; reconcile the original receipt')
        supplied = self.catalog(deepcopy(state), deepcopy(self.record['results']))
        if not isinstance(supplied, list) or len(supplied) > self.limits.max_tasks:
            raise ValueError('Catalog must be a bounded list of useful work')
        items = {}
        for item in supplied:
            item = deepcopy(item)
            if (not isinstance(item.get('id'), str) or not item['id'] or item['id'] in items
                    or item.get('lane') not in LANES or not item.get('revision')
                    or item.get('handler') not in self.handlers
                    or not item.get('description') or not item.get('completion_condition')):
                raise ValueError('Unique qualified work with executable handler and completion condition required')
            if (not isinstance(item.get('reads'), dict) or not item['reads']
                    or not isinstance(item.get('writes'), list)
                    or not set(item['writes']) <= set(item['reads'])):
                raise ValueError('Work requires relevant reads and covered writes')
            item.setdefault('requires', {})
            if (not isinstance(item['requires'], dict) or any(
                    not isinstance(statuses, list) or not statuses or
                    not set(statuses) <= {'completed', 'failed', 'no_progress'}
                    for statuses in item['requires'].values())):
                raise ValueError('Prerequisites require explicit settled outcomes')
            items[item['id']] = item
        self.items = items
        definitions = {key: fingerprint(item) for key, item in items.items()}
        outcomes = self.record['results']
        # Detect cycles and missing prerequisites rather than calling them done.
        visited = set()
        def visit(key, parents):
            if key in parents:
                raise ValueError('Cyclic work prerequisites')
            if key in visited:
                return
            for dep in items[key]['requires']:
                if dep not in items:
                    raise ValueError('Missing prerequisite work item')
                visit(dep, parents | {key})
            visited.add(key)
        for key in items:
            visit(key, set())
        inherited = {}
        def dependency_reads(key):
            if key not in inherited:
                values = dict(items[key]['reads'])
                for dep in items[key]['requires']:
                    for name, revision in dependency_reads(dep).items():
                        # A prerequisite's writes have post-effect revisions that
                        # the dependent operation must bind from actual receipts.
                        if name in items[dep]['writes']:
                            continue
                        if name in values and values[name] != revision:
                            raise ValueError(f'Conflicting prerequisite input revisions for {name!r}: '
                                f'{key!r} disagrees with {dep!r}. Generated outputs must be declared '
                                'as prerequisite writes and bound from actual completed evidence.')
                        values[name] = revision
                inherited[key] = values
            return inherited[key]
        actions, blocked = [], {}
        all_settled = bool(items)
        for key, item in items.items():
            result = outcomes.get(key, {})
            matching = result.get('definition') == definitions[key]
            settled = matching and result.get('status') in ('completed', 'failed', 'no_progress')
            if matching and result.get('status') in ('running', 'needs_reconciliation'):
                raise ValueError('Uncertain queue effect; reconcile the original receipt')
            if settled:
                continue
            all_settled = False
            reads = deepcopy(dependency_reads(key))
            reads['work_queue_scope'] = state['values']['work_queue_scope']
            ready = not item.get('blocked') and all(state['values'].get(k) == v for k, v in reads.items())
            for dep, allowed in item['requires'].items():
                prior = outcomes.get(dep, {})
                matches = prior.get('definition') == definitions[dep] and prior.get('status') in allowed
                token = fingerprint(prior)
                state['values']['work_result:' + dep] = token
                reads['work_result:' + dep] = token
                ready = ready and matches
            if not ready:
                blocked[key] = 'Blocked, stale dependency or unmet prerequisite'
                continue
            actions.append({'id': key, 'revision': definitions[key], 'lane': item['lane'],
                'description': deepcopy(item['description']),
                'completion_condition': item['completion_condition'],
                'reads': reads, 'writes': item['writes'], 'required': bool(item.get('required', False))})
            if item.get('decision') or item.get('select_with_jev'):
                raise ValueError('Legacy inferred tasks must be explicitly rebound by the owner in a new catalog')
            if item.get('workbench', {}).get('method_choice'):
                actions[-1]['method_choice'] = item['workbench']['method_choice']
        state.update(actions=actions, done=all_settled and all(
            outcomes.get(key, {}).get('status') == 'completed' for key in items),
            observations={'public': deepcopy(state.get('public_state', {})),
                          'blocked': blocked,
                          'settled': {k: v['status'] for k, v in outcomes.items()}},
            planner_reads={'work_queue_scope': state['values']['work_queue_scope']})
        return state

    def execute(self, action, context):
        item = deepcopy(self.items[action['id']])
        if fingerprint(item) != action['revision']:
            raise ValueError('Work definition changed before execution')
        entry = {'definition': action['revision'], 'status': 'running',
                 'lane': item['lane'], 'attempt_key': context['attempt_key']}
        self.record['results'][item['id']] = entry
        write_json(self.path, self.record)
        try:
            result = self.handlers[item['handler']](item, context)
            if (not isinstance(result, dict) or result.get('status') not in
                    ('completed', 'failed', 'no_progress', 'needs_reconciliation')):
                raise ValueError('Handler must return an explicit outcome and evidence')
            entry.update(status=result['status'], result=deepcopy(result))
        except BaseException:
            entry['status'] = 'needs_reconciliation'
            write_json(self.path, self.record)
            raise
        write_json(self.path, self.record)
        return result

    def _controller(self, on_status=None):
        state = self._context()
        plan = {'objective': self.goal['objective'], 'authority_revision': state['authority_revision'],
                'stage': state['stage'], 'reads': {'work_queue_scope': state['values']['work_queue_scope']}}
        return PersistentController(self.directory / 'controller', owner=self.owner,
                observe=self.observe, execute=self.execute, select=self.select,
                initial_plan=plan, on_status=on_status)

    def run(self, *, max_steps, on_status=None):
        with self._controller(on_status) as controller:
            return controller.run(max_steps=max_steps)
