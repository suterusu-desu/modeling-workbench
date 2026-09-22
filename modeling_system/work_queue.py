"""Dependency-aware work across modeling lanes, executed by one native owner.

Adapters supply qualified operations and observed revisions. Jev selects work;
this module handles durable completion, prerequisites and exact invalidation.
Neither metadata nor an inference result supplies modeling authority.
"""
from copy import deepcopy
from pathlib import Path
from .controller import PersistentController, SelectionNotDispatched, fingerprint, read_json, write_json
from .decision_budget import decision_budget

LANES = ('diagnosis', 'repair', 'verification', 'review', 'recovery',
         'experience', 'preparation')
DEFER = '__needs_review__'
# Advance when selection question meanings or their composition change.
SELECTION_POLICY = 'qualified-method-prerequisites-v9'
LANE_QUESTIONS = {
    'diagnosis': 'Which offered observation best distinguishes the remaining plausible causes and changes the next edit?',
    'repair': 'Which offered qualified method best addresses the observed failure mechanism while preserving retained gains?',
    'verification': 'Which offered verification resolves a concrete structural or realization concern using the least new work?',
    'review': 'Which offered evidence preparation makes the unresolved appearance question assessable with matched views?',
    'recovery': 'Which offered recovery matches the recorded failure and known effects without repeating an uncertain operation?',
    'experience': 'Which offered experience retrieval best supplies an applicable method or warns against repeating the observed failure?',
    'preparation': 'Which offered preparation resolves the missing correspondence, support or input needed for a useful edit?',
}


class WorkQueue:
    def __init__(self, directory, *, owner, goal, observe_context, catalog, handlers, select, budget=None):
        self.decision_budget = decision_budget(budget)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.owner, self.goal = owner, deepcopy(goal)
        self.observe_context, self.catalog = observe_context, catalog
        self.handlers, self.select = handlers, select
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
        if not isinstance(supplied, list) or len(supplied) > self.decision_budget.max_tasks:
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
            if item.get('select_with_jev'):
                actions[-1]['select_with_jev'] = True
            if item.get('workbench', {}).get('method_choice'):
                actions[-1]['method_choice'] = item['workbench']['method_choice']
            if item.get('decision'):
                from .capability_calls import validate_call
                validate_call(item)
                if item.get('required'):
                    raise ValueError('Contextual decisions cannot be required housekeeping')
                actions[-1]['decision'] = deepcopy(item['decision'])
                actions[-1]['select_with_jev'] = True
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

    def run(self, *, max_steps, on_status=None):
        state = self._context()
        plan = {'objective': self.goal['objective'], 'authority_revision': state['authority_revision'],
                'stage': state['stage'], 'reads': {'work_queue_scope': state['values']['work_queue_scope']}}
        with PersistentController(self.directory / 'controller', owner=self.owner,
                observe=self.observe, execute=self.execute, select=self.select,
                initial_plan=plan, on_status=on_status) as controller:
            return controller.run(max_steps=max_steps)


class LaneSelector:
    """Batch next-lane and conditional lane choices; retain exact applicable advice.

judge(public_state, decisions, binding) returns resolve_judgments output.
The workspace callback owns transport, credential, ledger and fresh release.
project(snapshot, actions, plan) must provide reviewed public state, lane-specific
facts and descriptions keyed by action ID; private IDs/payloads never go on wire.
"""
    def __init__(self, directory, *, judge, project, budget=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.judge, self.project = judge, project
        self.budget = decision_budget(budget)
        self.path = self.directory / 'lane-advice.json'
        self.cache = read_json(self.path) if self.path.exists() else {}

    def __call__(self, snapshot, actions, plan, *, prepare_only=False):
        public = self.project(deepcopy(snapshot), deepcopy(actions), deepcopy(plan))
        if (not isinstance(public, dict) or not isinstance(public.get('state'), dict)
                or set(public.get('descriptions', {})) != {a['id'] for a in actions}):
            raise SelectionNotDispatched('Complete reviewed public projection required; inference was not called')
        original_public = deepcopy(public)
        grouped = {lane: [a for a in actions if a['lane'] == lane] for lane in LANES}
        grouped = {k: v for k, v in grouped.items() if v}
        reads = {}
        for action in actions:
            for k, v in action['reads'].items():
                if k in reads and reads[k] != v:
                    raise SelectionNotDispatched('Conflicting evidence revisions; inference was not called')
                reads[k] = v
        binding = {'owner': snapshot['owner'], 'authority_revision': snapshot['authority_revision'],
            'dependencies': {'reads': reads, 'writes': sorted({k for a in actions for k in a['writes']})},
            'menu': fingerprint(actions), 'plan': fingerprint(plan), 'decision_budget': self.budget.record()}
        selection_key = fingerprint({'binding': binding, 'public': public, 'policy': SELECTION_POLICY})
        last_path = self.directory / 'last-batch.json'
        if last_path.exists():
            prior = read_json(last_path)
            if not prepare_only and prior.get('selection_key') == selection_key and 'choice' in prior:
                return deepcopy(prior['choice'])
        selection = {'state': snapshot, 'actions': actions, 'plan': plan}
        candidates = public.get('experience_candidates', [])
        retrieval = None
        retrieval_dispatched = False
        if len(candidates) > len(public['state'].get('retained_experience', {}).get('passages', [])):
            from .decision_evidence import retrieval_questions
            retrieval_key = fingerprint({'binding': binding, 'public': original_public, 'policy': SELECTION_POLICY})
            path = self.directory / 'retrieval' / (retrieval_key + '.json')
            if path.exists():
                retrieval = read_json(path)
            else:
                stage_state = {k: deepcopy(v) for k, v in public['state'].items() if k != 'retained_experience'}
                stage_state['candidates'] = candidates
                decisions = retrieval_questions(candidates)
                from .judgments import prepare_judgments
                try:
                    packet = prepare_judgments(stage_state, decisions, binding)
                except ValueError as error:
                    raise SelectionNotDispatched(str(error)) from error
                batch = {'phase': 'experience', 'public': original_public, 'packet': packet,
                    'binding': binding, 'decisions': decisions, 'candidates': candidates,
                    'retrieval_key': retrieval_key, 'selection_key': selection_key}
                write_json(self.directory / 'batches' / (fingerprint(selection) + '.json'), batch)
                if prepare_only:
                    return deepcopy(batch)
                result = self.judge(stage_state, decisions, binding)
                retrieval = self._complete_retrieval(batch, result)
                retrieval_dispatched = True
            public['state']['retained_experience']['passages'] = deepcopy(retrieval['passages'])
            public['state']['retained_experience']['semantic_coverage'] = {
                k: deepcopy(retrieval[k]) for k in ('excluded', 'unreturned_conflicts', 'candidate_count',
                                                  'coverage_by_relationship', 'selection_policy')}
            public['rank_experience'] = []  # These exact sources were just judged.
        decisions, lane_keys, resolved = [], {}, {}
        # Choices for different lanes share state but never assume other answers.
        for lane, offered in grouped.items():
            options = [{'id': a['id'], 'description': {
                'operation': public['descriptions'][a['id']],
                **({'observation': a['decision']['observation']} if a.get('decision', {}).get('observation') else {})}}
                for a in offered]
            lane_key = fingerprint({'lane': lane, 'options': options, 'policy': SELECTION_POLICY,
                'actions': offered, 'owner': snapshot['owner'],
                'reads': {k: v for a in offered for k, v in a['reads'].items()},
                'state': public['state'], 'facts': public.get('lane_facts', {}).get(lane),
                'authority': binding['authority_revision'], 'plan': binding['plan']})
            lane_keys[lane] = lane_key
            cached = self.cache.get(lane_key)
            if cached and cached['choice'] in {a['id'] for a in offered} | {DEFER}:
                resolved[lane] = cached
            else:
                decisions.append({'id': lane, 'type': 'choice', 'instructions': {
                    'question': 'If this lane is next: ' + LANE_QUESTIONS[lane],
                    'lane': lane, 'facts': public.get('lane_facts', {}).get(lane, {}),
                    'limits': 'Use supplied evidence, applicability and failures. Avoid redundant work and repeated failed mechanisms. Defer when no offered operation applies; do not infer appearance acceptance or authorize unsupported geometry.'},
                    'options': options + [{'id': DEFER, 'description': 'No applicable operation: return missing evidence or capability to the reasoning owner.'}]})
        fixed_lane = next(iter(grouped)) if len(grouped) == 1 else None
        if fixed_lane is None:
            decisions.append({'id': 'next_lane', 'type': 'choice',
            'instructions': 'Which available work lane best advances the goal now? Prioritize dominant observed defects and resolving uncertainty that changes an edit; avoid unnecessary review or bookkeeping. Lane answers are independent; choose defer if the available work cannot usefully proceed.',
            'options': [{'id': lane, 'description': {'lane': lane,
                'facts': public.get('lane_facts', {}).get(lane, {}),
                'available_operations': [public['descriptions'][a['id']] for a in offered]}}
                for lane, offered in grouped.items()] + [{'id': DEFER, 'description': 'Need new capability, conflicting-evidence resolution or actual visual interpretation.'}]})
        # Relevance is independently useful for subsequent context reuse. All
        # passages are already visible to action questions in this same batch;
        # no question assumes another question's answer.
        call_mappings, claim_mappings, method_mappings = {}, {}, {}
        from .capability_calls import argument_questions
        from .decision_evidence import claim_questions
        from .method_reasoning import method_questions
        for ordinal, action in enumerate(actions):
            described = {**action, 'public_description': public['descriptions'][action['id']]}
            try:
                arguments, argument_map = argument_questions(described, ordinal)
                claims, claim_map = claim_questions(described, ordinal)
                method, method_map = method_questions(described, ordinal, actions, public['descriptions'])
            except (ValueError, KeyError, TypeError) as error:
                raise SelectionNotDispatched(str(error), completed_evidence=retrieval_dispatched) from error
            decisions.extend(arguments + claims + method)
            call_mappings[action['id']] = argument_map
            claim_mappings[action['id']] = claim_map
            method_mappings[action['id']] = method_map
        ranked_passages = public.get('rank_experience', [])[:max(0, self.budget.max_questions - len(decisions))]
        for position, passage in enumerate(ranked_passages):
            decisions.append({'id': 'experience_relevance_' + str(passage['index']), 'type': 'score',
                'instructions': {'question': 'How directly does the specified retained passage help select the next offered operation for the current defect, including warnings against repeating a failed mechanism?',
                                 'passage': 'retained_experience.passages[' + str(position) + ']'},
                'criteria': ['Unrelated to the current mechanism or scope',
                             'Related context but does not distinguish offered methods',
                             'Directly supports a method or identifies an applicable failure to avoid']})
        from .judgments import prepare_judgments
        try:
            packet = prepare_judgments(public['state'], decisions, binding) if decisions else None
        except ValueError as error:
            raise SelectionNotDispatched(str(error), completed_evidence=retrieval_dispatched) from error
        batch = {'phase': 'action', 'selection_key': selection_key, 'public': original_public, 'packet': packet,
            'binding': binding, 'grouped': grouped, 'resolved': resolved,
            'lane_keys': lane_keys, 'decisions': decisions, 'fixed_lane': fixed_lane,
            'call_mappings': call_mappings, 'claim_mappings': claim_mappings,
            'method_mappings': method_mappings,
            'retrieval': retrieval, 'decision_state': public['state']}
        # Retain the exact fan-out and cached branches before any provider call.
        # Recovery never has to reconstruct questions from a later cache state.
        try:
            write_json(self.directory / 'batches' / (fingerprint(selection) + '.json'), batch)
        except OSError as error:
            raise SelectionNotDispatched('Could not retain selection preparation; inference was not called') from error
        if prepare_only:
            return deepcopy(batch)
        result = self.judge(public['state'], decisions, binding) if decisions else {
            'binding': binding, 'judgments': {}, 'provider_receipt': None}
        return self._complete(batch, result)

    def recover_completed(self, selection, packet, response, provider_receipt):
        from .judgments import resolve_judgments
        batch = read_json(self.directory / 'batches' / (fingerprint(selection) + '.json'))
        if batch['packet'] != packet:
            raise ValueError('Completed response belongs to different selection questions')
        result = resolve_judgments(packet, response['answers'], batch['binding'])
        result['provider_receipt'] = provider_receipt
        result['model'] = response.get('model')
        result['usage'] = deepcopy(response.get('usage'))
        if batch.get('phase') == 'experience':
            self._complete_retrieval(batch, result)
            return {'status': 'needs_review', 'reason': 'Evidence choice recovered; resume the ordinary controller to select the operation'}
        return self._complete(batch, result)

    def _complete_retrieval(self, batch, result):
        from .decision_evidence import select_passages
        if (result.get('binding') != batch['binding']
                or set(result.get('judgments', {})) != {d['id'] for d in batch['decisions']}):
            raise ValueError('Retrieval answer does not match its exact binding and question set')
        retained = select_passages(batch['candidates'], result['judgments'], self.budget)
        retained.update(packet=deepcopy(batch['packet']), provider_receipt=result.get('provider_receipt'),
                        model=result.get('model'), usage=deepcopy(result.get('usage')))
        write_json(self.directory / 'retrieval' / (batch['retrieval_key'] + '.json'), retained)
        return retained

    def _complete(self, batch, result):
        binding, grouped = batch['binding'], batch['grouped']
        resolved, lane_keys = deepcopy(batch['resolved']), batch['lane_keys']
        if (result.get('binding') != binding or
                set(result.get('judgments', {})) != {d['id'] for d in batch['decisions']}):
            raise ValueError('Judgment release binding changed')
        answers = result['judgments']
        for lane in grouped:
            if lane not in resolved:
                answer = answers[lane]
                if answer.get('choice') not in {a['id'] for a in grouped[lane]} | {DEFER}:
                    raise ValueError('Lane choice outside menu')
                resolved[lane] = answer
                self.cache[lane_keys[lane]] = deepcopy(answer)
        lane = batch.get('fixed_lane') or answers['next_lane']['choice']
        if lane == DEFER or lane in resolved and resolved[lane]['choice'] == DEFER:
            choice = {'status': 'needs_review', 'reason': 'Jev found no applicable next operation',
                    'evidence': [str(self.directory / 'last-batch.json')]}
        elif lane not in resolved:
            raise ValueError('Priority selected an unavailable lane')
        else:
            choice = resolved[lane]['choice']
        call, checks, method_checks, method_route = None, {}, {}, None
        if isinstance(choice, str):
            from .method_reasoning import resolve_method, allowed_remedy
            initial = choice
            evaluated = resolve_method(batch.get('method_mappings', {}).get(choice, {}), answers)
            method_checks[choice] = evaluated
            if evaluated['status'] == 'unmet':
                remedy = evaluated['remedy']
                offered = {a['id']: a for rows in grouped.values() for a in rows}
                if remedy in offered and remedy != initial and allowed_remedy(offered[initial], offered[remedy]):
                    remedial = resolve_method(batch.get('method_mappings', {}).get(remedy, {}), answers)
                    method_checks[remedy] = remedial
                    if remedial['status'] == 'ready':
                        choice = remedy; lane = offered[remedy]['lane']
                        method_route = {'proposed': initial, 'selected': remedy, 'unmet': evaluated['unmet']}
                    else:
                        choice = {'status': 'needs_review', 'reason': 'The scoped remedy also lacks a required condition', 'method_checks': method_checks}
                else:
                    choice = {'status': 'needs_review', 'reason': 'Resolve only the selected method prerequisite', 'method_checks': method_checks}
        if isinstance(choice, str):
            action = next(a for a in grouped[lane] if a['id'] == choice)
            from .capability_calls import resolve_call
            if action.get('decision', {}).get('arguments'):
                call = resolve_call(action, batch['call_mappings'][choice], answers)
                if call['status'] != 'bound':
                    choice = {'status': 'needs_review', 'reason': call['reason']}
            for key, name in batch.get('claim_mappings', {}).get(action['id'], {}).items():
                checks[name] = deepcopy(answers[key])
            if any(c['choice'] != 'supported' for c in checks.values()):
                choice = {'status': 'needs_review', 'reason': 'A consequential claim lacks source support', 'claims': checks}
        trace = {'binding': binding, 'judgments': answers,
            'actions': [deepcopy(a) for offered in grouped.values() for a in offered],
            'applicable_lane_choices': resolved, 'provider_receipt': result.get('provider_receipt'),
            'question_count': len(batch['decisions']), 'selection_key': batch['selection_key'], 'choice': choice,
            'policy': SELECTION_POLICY, 'packet': batch['packet'], 'decision_state': batch.get('decision_state'),
            'question_revision': fingerprint(batch['decisions']), 'model': result.get('model'),
            'usage': deepcopy(result.get('usage')),
            'call': call, 'claim_checks': checks, 'method_checks': method_checks,
            'method_route': method_route, 'retrieval': batch.get('retrieval'),
            'method_rankings': {lane: sorted(answer.get('probabilities', {}).items(), key=lambda row: (-row[1], row[0]))
                                for lane, answer in resolved.items()},
            'experience_rankings': sorted([{'index': int(key.rsplit('_', 1)[1]), 'judgment': value}
                for key, value in answers.items() if key.startswith('experience_relevance_')],
                key=lambda row: (-row['judgment']['score'], row['index']))}
        write_json(self.directory / 'decisions' / (batch['selection_key'] + '.json'), trace)
        write_json(self.directory / 'last-batch.json', trace)
        # Complete decisions (including deferrals) now survive restarts without
        # another priority call. Different context still invalidates this result.
        self.cache = dict(list(self.cache.items())[-128:])
        write_json(self.path, self.cache)
        return choice
