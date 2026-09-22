"""Owner-controlled operation composed over existing episodes, services and queues.

The session is an execution index, not a new character or evidence authority.
Private adapters qualify native capabilities; this layer makes their contracts,
actual findings and visual feedback survive routing, execution and recovery.
"""
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
import time
import uuid

from .controller import applicable, fingerprint, read_json, write_json
from .result_reporting import json_data
from .episodes import pin_link
from .journal import EPISODE, ACTIVE_CALL, invoke
from .leases import LEASE
from .store import canonical, digest
from .work_queue import WorkQueue
from .session_metrics import session_metrics


# Input names denote evidence relationships, not invented anatomical bindings.
# Output checks describe verified artifacts; "pass" never approves appearance.
PROFILES = {
    'analysis': {'inputs': [], 'outputs': ['analysis']},
    'inspection': {'inputs': ['subject'], 'outputs': ['observation']},
    'appearance_edit': {'inputs': ['identity', 'baseline', 'guide', 'depth',
        'sections', 'correspondence', 'preservation', 'adapter'],
        'outputs': ['candidate', 'guide_fit', 'realization', 'preservation', 'comparison']},
    'capture': {'inputs': ['subject', 'view', 'pose'], 'outputs': ['observation']},
    'verification': {'inputs': ['candidate'], 'outputs': ['verification']},
    'display': {'inputs': ['subject'], 'outputs': ['display']},
    'checkpoint': {'inputs': ['candidate'], 'outputs': ['checkpoint']},
    'retention': {'inputs': ['candidate', 'reopen'], 'outputs': ['checkpoint']},
    'generation': {'inputs': ['source', 'policy', 'job'], 'outputs': ['job']},
    'recovery': {'inputs': ['receipt'], 'outputs': ['reconciliation']},
    'learning': {'inputs': ['evidence'], 'outputs': ['method']},
}
FINDING_KINDS = {'measured', 'predicted', 'hypothesis', 'scope_noop',
                 'unsupported', 'appearance', 'failure', 'method'}
SETTLED = {'completed', 'failed', 'no_progress'}


def protocol():
    inventory = json.loads(Path(__file__).with_name('capabilities.json').read_bytes())
    return {'schema_version': 1,
        'roles': {
            'owner': 'Choose and execute qualified operations directly, preserve intent and likeness criteria, and review actual images.',
            'workbench': 'Compile dependencies and contracts, execute chosen capabilities, retain exact facts, reconcile effects and expose the next useful work.',
            'native_adapter': 'One owner; atomic expected-state checks, real guide-constrained construction, protection, save/reopen and visible state.',
        },
        'profiles': deepcopy(PROFILES),
        'lanes': ['diagnosis', 'repair', 'verification', 'review', 'recovery', 'experience', 'preparation'],
        'capabilities': [{k: row[k] for k in ('name', 'route', 'control')}
                         for row in inventory['capabilities']],
        'claims': 'Execution, measured realization, saved/reopened state, appearance judgment, user acceptance and method benefit stay separate.',
        'parallelism': 'Explicitly ordered useful work shares one queue; immutable preparation may overlap through a qualified adapter. Native effects remain serial. Persistence grants no new budget or unattended work.'}


def _binding_path(directory):
    return Path(directory) / 'operating-session.json'


def _reviews(service, directory):
    """Immutable submissions let Astra review while an unrelated task runs.

    The queue owner alone writes work-queue.json. Feedback is a separate index of
    existing episode operation receipts, reread at each decision boundary.
    """
    rows = {}
    for path in (Path(directory) / 'reviews').glob('*.json'):
        row = read_json(path)
        fact = service.store.get(row['operation_fact'], 'operation_fact')
        retained = fact['outcome']['result']
        if any(row.get(key) != value for key, value in retained.items()):
            raise ValueError('Review index differs from its immutable episode fact')
        prior = rows.get(row['task'])
        if not prior or (row['submitted_at'], row['operation_handle']) > (prior['submitted_at'], prior['operation_handle']):
            rows[row['task']] = row
    return rows


def inspect_session(service, directory):
    from .decision_outcomes import decision_outcomes
    binding = read_json(_binding_path(directory))
    if binding['workspace'] != str(service.workspace):
        raise ValueError('Operating session belongs to a different workspace')
    episode = service.ledger.read(binding['episode'])
    queue = read_json(Path(directory) / 'work-queue.json')
    reviews = _reviews(service, directory)
    return {'binding': binding, 'episode_revision': episode['revision'],
        'mode': 'Retained execution index; live freshness is checked by the owner at use',
        'tasks': {key: {'status': row['status'], 'operation_handle': row.get('operation_handle'),
            'report': row.get('result', {}).get('workbench'),
            'recovery': 'Reconcile original operation handle; never replay uncertain effects'}
            for key, row in queue['results'].items()},
        'reviews': reviews,
        'cooperation': read_json(Path(directory) / 'handoff.json') if (Path(directory) / 'handoff.json').exists() else None,
        'metrics': session_metrics(directory),
        'decision_outcomes': decision_outcomes(directory, reviews),
        'detail': {'episode': binding['episode'], 'operation': 'decision_workspace'}}


class OperatingSession(WorkQueue):
    """Episode-backed WorkQueue with mandatory result-to-decision feedback.

    Handlers return status plus a workbench report. report(item, result) can adapt
    an existing qualified handler without rewriting its native implementation.
    Reports contain checks with exact evidence and deliberately public findings.
    The owner supplies explicit priorities; reports and historical evidence remain local.
    """
    def __init__(self, directory, *, service, episode, owner, goal, observe_context,
                 catalog, handlers, select=None, task_order=None, report=None, experience=None, budget=None,
                 preservation=None, require_preservation=False):
        self.service, self.episode = service, episode
        self.report_adapter = report
        self.private_handlers = dict(handlers)
        self.private_handlers.setdefault('service', self._service)
        self.preservation = preservation
        self.require_preservation = require_preservation
        from .work_limits import work_limits
        self.limits = work_limits(budget)
        from .retained_context import RetainedContext
        self.experience = experience if experience is not None else RetainedContext(service, sources=[], budget=self.limits)
        self.user_catalog = catalog
        self._validate_episode(owner)
        super().__init__(directory, owner=owner, goal=goal, observe_context=observe_context,
            catalog=self._catalog, handlers={key: self._handle for key in self.private_handlers},
            select=select, task_order=task_order, budget=self.limits)
        binding = {'schema_version': 1, 'workspace': str(service.workspace),
            'episode': episode, 'owner': owner, 'goal': fingerprint(goal)}
        if preservation or require_preservation:
            binding['preservation'] = {'required': bool(require_preservation),
                'policy': preservation.revision if preservation else None}
        path = _binding_path(directory)
        if path.exists():
            retained = read_json(path)
            if any(retained.get(k) != v for k, v in binding.items()):
                raise ValueError('Session binding changed; do not reinterpret a previous queue')
        else:
            # Adoption records old receipts as historical; never retroactively
            # asserts contracts, native verification or this runtime's execution.
            historical = deepcopy(self.record['results'])
            token = EPISODE.set(episode)
            try:
                retained = invoke(service, 'attach_operating_session', binding,
                    lambda: {'binding': binding, 'historical_results': historical,
                             'qualification': 'Prior results retain their original provenance'})
            finally:
                EPISODE.reset(token)
            write_json(path, {**binding, 'attachment_operation': retained['operation_handle']})
        from .learning import import_review_history
        import_review_history(service, [self.directory])

    def _validate_episode(self, owner):
        episode = self.service.ledger.read(self.episode)
        if (episode['kind'] != 'decision_episode' or episode['status'] != 'active'
                or episode['intent']['owner'] != owner):
            raise ValueError('An active decision episode owned by the native owner is required')
        return episode

    def run(self, *, max_steps, on_status=None, feedback_timeout=0, on_handoff=None, cancelled=None):
        from .cooperation import run_cooperatively
        return run_cooperatively(self, max_steps=max_steps, on_status=on_status,
            feedback_timeout=feedback_timeout, on_handoff=on_handoff, cancelled=cancelled)

    def _context(self):
        state = super()._context()
        episode = self._validate_episode(self.owner)
        context = episode['data'].get('context', episode['intent']['context'])
        # Journaling new results must not invalidate the goal. Actual intent,
        # requirements and feedback do. The current character remains external.
        state['values']['operating_intent'] = fingerprint({
            'episode': self.episode, 'scope': episode['intent']['scope'],
            'requirements': context.get('requirements', []),
            'next_question': context.get('next_question'), 'profiles': PROFILES})
        if self.preservation:
            state['values'].update(self.preservation.context_values())
        return state

    def _catalog(self, state, outcomes):
        items = self.user_catalog(state, outcomes)
        for item in items:
            prior = outcomes.get(item.get('id'), {})
            if prior.get('definition') == fingerprint(item) and prior.get('status') in SETTLED:
                continue
            self._contract(item)
        return items

    def _contract(self, item):
        contract = item.get('workbench', {})
        profile = PROFILES.get(contract.get('profile'))
        if not profile or not contract.get('capability') or not contract.get('method'):
            raise ValueError('New work needs a workbench profile, capability and retained method identity')
        bindings = contract.get('bindings', {})
        required = set(profile['inputs'])
        if contract.get('native'):
            required.add('adapter')
        if item.get('required') and contract.get('profile') == 'appearance_edit':
            raise ValueError('A substantive modeling pass cannot be required housekeeping')
        if not required <= bindings.keys():
            raise ValueError('Missing modeling input relationships: ' + ', '.join(sorted(required - bindings.keys())))
        for role, keys in bindings.items():
            if not isinstance(keys, list) or not keys or not set(keys) <= item['reads'].keys():
                raise ValueError('Each modeling input relationship must bind exact observed reads: ' + role)
        if contract['profile'] == 'retention' and not contract.get('visual_review'):
            raise ValueError('Retention requires exact candidate appearance feedback, separate from technical checks')
        if contract.get('visual_review') and contract['visual_review'] not in item.get('requires', {}):
            raise ValueError('Appearance feedback must name a declared candidate prerequisite')
        for dependency, checks in contract.get('consumes', {}).items():
            if dependency not in item.get('requires', {}) or not checks:
                raise ValueError('Consumed evidence must name a declared prerequisite and its checks')
        if item.get('decision') or item.get('select_with_jev'):
            raise ValueError('Legacy inferred tasks require explicit owner-bound arguments and conditions')
        if item.get('parameters', {}).get('arguments'):
            from .capability_calls import bind_call
            invocation = contract.get('invocation', {})
            if (invocation.get('status') != 'bound' or invocation.get('task') != item['id']
                    or invocation.get('specification') != fingerprint(item['parameters'])):
                raise ValueError('Typed arguments require an explicit owner-bound invocation')
            expected = bind_call(item, invocation.get('selected'))
            if (expected['payload'] != item.get('payload')
                    or expected['workbench']['invocation']['arguments'] != invocation.get('arguments')):
                raise ValueError('Task payload differs from its explicit qualified argument choices')
        if item['handler'] == 'service':
            payload = item['payload']
            operation, arguments = payload['operation'], payload.get('arguments', {})
            if operation.startswith('native_'):
                from .native_bridge import validate_arguments
                args = dict(arguments); args.pop('owner', None)
                validate_arguments(operation[7:], args)
                if arguments.get('owner') != self.owner:
                    raise ValueError('Native service capability must name the same sole owner')
            else:
                if operation not in self.service.operations() or operation == 'run_episode_operation':
                    raise ValueError('Service capability must call one existing operation without nesting episode dispatch')
                inspect.signature(getattr(self.service, operation)).bind(**arguments)
                if operation == 'run_recipe_step':
                    recipe = self.service.ledger.read(arguments['recipe'])
                    if recipe['intent']['episode'] != self.episode:
                        raise ValueError('Recipe and operating session must share an episode')
        return contract

    def _review_basis(self, task, state):
        result = self.record['results'][task]
        item = self.items.get(task) or result['task']
        if fingerprint(item) != result['definition']:
            raise ValueError('Task definition changed since execution')
        return fingerprint({'definition': result['definition'], 'result': result.get('result'),
            'intent': state['values']['operating_intent'],
            'values': {key: state['values'].get(key) for key in item['reads']}})

    def _review(self, task, state):
        review = self.reviews.get(task)
        if not review or task not in self.items or task not in self.record['results']:
            return None
        if fingerprint(self.items[task]) != self.record['results'][task]['definition']:
            return None
        return review if review['basis'] == self._review_basis(task, state) else None


    def observe(self):
        state = super().observe()
        self.reviews = _reviews(self.service, self.directory)
        ready = []
        blocked = state['observations']['blocked']
        for action in state['actions']:
            item = self.items[action['id']]
            contract = item['workbench']
            if (self.require_preservation and self.preservation is None
                    and (contract['profile'] in ('appearance_edit', 'retention') or contract.get('preservation'))):
                blocked[action['id']] = {'preservation': 'Bind the required established outcomes for this affected operation'}
                continue
            if self.preservation:
                try:
                    self.preservation.preflight(item, self.record['results'])
                except (ValueError, KeyError, OSError) as error:
                    blocked[action['id']] = {'preservation': str(error)}
                    continue
            preflight = getattr(self.private_handlers[item['handler']], 'preflight', None)
            if preflight is not None:
                try:
                    preflight(deepcopy(item))
                except ValueError as error:
                    blocked[action['id']] = {'preflight': str(error)}
                    continue
            missing = []
            for task, checks in contract.get('consumes', {}).items():
                prior = self.record['results'].get(task, {}).get('result', {}).get('workbench', {})
                missing.extend(task + ':' + check for check in checks
                    if prior.get('checks', {}).get(check, {}).get('status') != 'pass')
            task = contract.get('visual_review')
            if task:
                review = self._review(task, state)
                if not review or review['judgment']['disposition'] != 'useful':
                    missing.append('current scoped appearance review:' + task)
            if missing:
                blocked[action['id']] = {'missing_evidence': missing}
            else:
                ready.append(action)
        state['actions'] = ready
        feedback = self._feedback(state)
        state['observations']['workbench'] = feedback
        if self.preservation:
            state['observations']['required_preservation'] = self.preservation.public()
            state['values']['operating_preservation'] = fingerprint(state['observations']['required_preservation'])
        state['values']['operating_feedback'] = fingerprint(feedback)
        state['values']['operating_observation'] = fingerprint(state.get('public_state', {}))
        for action in ready:
            action['reads'].update({k: state['values'][k]
                for k in ('operating_intent', 'operating_feedback', 'operating_observation')})
            if self.preservation:
                action['reads']['operating_preservation'] = state['values']['operating_preservation']
        if self.experience and ready and not any(action.get('required') for action in ready):
            from .retained_context import RetainedContext
            record = self.experience(deepcopy(state),
                [deepcopy(self.items[action['id']]) for action in ready], deepcopy(self.record['results']))
            RetainedContext.retain(self.directory, record)
            self._experience_record = deepcopy(record)
            state['observations']['experience'] = record['public']
            state['values']['operating_experience'] = record['revision']
            for action in ready:
                action['reads']['operating_experience'] = record['revision']
        if not ready and blocked:
            state['attention'] = {'reason': 'Resolve the scoped input, evidence or appearance question; saved work remains available',
                                  'tasks': deepcopy(blocked)}
        from .cooperation import review_inbox
        state['observations']['cooperation'] = review_inbox(self, state)
        return state

    def _feedback(self, state):
        # Facts are compiled after every actual operation, never frozen in a
        # startup prompt. IDs/locators remain private; only authored public text
        # and explicit coverage enter the compact owner view.
        facts = []
        for task, entry in self.record['results'].items():
            report = entry.get('result', {}).get('workbench')
            if report and task in self.items:
                applicable = all(state['values'].get(k) == v for k, v in
                    report.get('basis', {}).items() if k not in self.items[task]['writes'])
                facts.append({'findings': report['findings'], 'checks': {
                    k: v['status'] for k, v in report['checks'].items()},
                    'applicability': 'current inputs' if applicable else 'historical; inputs changed',
                    'qualification': report['qualification']})
        reviews = [row['judgment'] for task in self.reviews
                   if (row := self._review(task, state))]
        feedback = {'findings': facts, 'visual_feedback': reviews,
            'limits': 'Technical execution and metrics do not approve appearance. Scope_noop excludes only its stated scope; retain useful gains elsewhere.'}
        if len(canonical(feedback)) > self.limits.context_bytes:
            raise ValueError('Modeling feedback exceeds the context limit; narrow the active catalog, retaining old facts in the episode')
        return feedback



    def _service(self, item, context):
        payload = item['payload']
        if payload['operation'] == 'run_recipe_step':
            # A recipe already owns its exact step journal and lease. Preserve
            # that boundary as a linked child, rather than flattening or forking
            # the recipe engine. The parent task lease remains active on disk.
            call_token, lease_token = ACTIVE_CALL.set(None), LEASE.set(None)
            try:
                result = self.service.execute(payload['operation'], payload.get('arguments', {}))
            finally:
                LEASE.reset(lease_token); ACTIVE_CALL.reset(call_token)
        else:
            result = self.service.execute(payload['operation'], payload.get('arguments', {}))
        # A returned prepared/submitted/retained job is a completed service call,
        # not a finished modeling workflow. Keep its domain status and handle.
        status = 'completed'
        if result.get('status') in ('failed', 'conflicting', 'needs attention', 'needs_reconciliation', 'unknown'):
            status = 'failed' if result.get('effect_status') == 'refused before mutation dispatch' else 'needs_reconciliation'
        linked = {'handle': result['recipe']} if result.get('recipe') else {}
        return {'status': status, 'service_result': result, **linked,
                **{key: result[key] for key in ('handle', 'job', 'experiment') if key in result}}

    def _report(self, item, result, context):
        raw = json_data(self.report_adapter(item, result) if self.report_adapter else result.get('workbench', {}))
        findings = raw.get('findings', [])
        if not isinstance(findings, list) or any(not isinstance(row, dict)
                or set(row) != {'kind', 'scope', 'summary'} or row['kind'] not in FINDING_KINDS
                or not row['scope'] or not row['summary'] for row in findings):
            raise ValueError('Findings require kind, public scope and public summary; keep private locators in evidence')
        overflow = len(canonical(findings)) > 2000
        checks = {}
        for name, check in raw.get('checks', {}).items():
            if check.get('status') not in ('pass', 'fail', 'unknown') or not check.get('evidence'):
                raise ValueError('Checks require pass/fail/unknown and exact evidence')
            checks[name] = {**deepcopy(check),
                'evidence': [pin_link(self.service, link) for link in check['evidence']]}
        if overflow:
            # The effect is known. Retain all qualifications, withhold dependent
            # use and repair only its public projection, without native replay.
            handle = self.record['results'][item['id']]['operation_handle']
            path = self.service.store.root / 'calls' / handle / ('report-' + fingerprint(raw) + '.json')
            write_json(path, raw)
            detail = pin_link(self.service, {'kind': 'file', 'path': str(path), 'role': 'Complete report requiring compact projection'})
            for check in checks.values():
                if check['status'] == 'pass':
                    check['status'] = 'unknown'
                check['evidence'].append(detail)
            findings = [{'kind': 'failure', 'scope': 'report projection',
                'summary': 'The operation returned and its complete report is retained. The report exceeds the public context budget; review its full limitations and repair the compact projection before dependent use. Do not repeat the operation.'}]
        required = list(PROFILES[item['workbench']['profile']]['outputs'])
        if self.preservation and self.preservation.relevant(item):
            assessment = result.get('preservation')
            evidence = assessment.get('evidence', []) if assessment else []
            checks['preservation'] = {'status': assessment['status'] if assessment else 'unknown',
                'evidence': [pin_link(self.service, {'kind': 'file', 'path': ref['path'],
                    'role': 'Source-bound evaluated preservation measurement'}) for ref in evidence]}
            if 'preservation' not in required: required.append('preservation')
        missing = [key for key in required if checks.get(key, {}).get('status') != 'pass']
        return {'checks': checks, 'findings': deepcopy(findings),
            'report_needs_repair': overflow,
            'qualification': 'complete' if not missing else 'needs evidence',
            'missing': missing, 'basis': {key: context['expected_values'][key] for key in item['reads']},
            'context_record': result.get('operation_context_record'),
            'capability': item['workbench']['capability'], 'method': item['workbench']['method'],
            'appearance_acceptance': 'not implied', 'method_benefit': 'not implied'}

    def _handle(self, item, context):
        original_item = deepcopy(item)
        contract = self._contract(item)
        arguments = {'task': original_item, 'compiled_task': item, 'expected_values': context['expected_values'],
                     'attempt_key': context['attempt_key']}

        def reserve(lease):
            entry = self.record['results'][item['id']]
            entry.update(operation_handle=lease['operation_handle'], episode=self.episode, task=original_item)
            write_json(self.path, self.record)

        def run():
            if (self.require_preservation and self.preservation is None
                    and (contract['profile'] in ('appearance_edit', 'retention') or contract.get('preservation'))):
                raise ValueError('Affected operation has no bound required preservation policy')
            # Existing procedure and policy lookup stays in the common service.
            decision_context = self.service.capture_operation_context(
                contract.get('stage', item['lane']), {'mechanism': contract['method']})
            started = time.perf_counter()
            prepared, consumption = (self.preservation.prepare(item, self.record['results'])
                if self.preservation else (item, None))
            result = self.private_handlers[item['handler']](prepared, context)
            if not isinstance(result, dict) or result.get('status') not in SETTLED | {'needs_reconciliation'}:
                raise ValueError('Capability must return an explicit execution disposition')
            # Persist the returned effect before report validation. A report/index
            # failure cannot erase successful work or make it safe to repeat.
            handle = self.record['results'][item['id']]['operation_handle']
            result = json_data(result)
            result['handler_elapsed_ms'] = round((time.perf_counter() - started) * 1000, 3)
            result['operation_context_record'] = decision_context['context_record']
            write_json(self.service.store.root / 'calls' / handle / 'capability-result.json', result)
            if consumption is not None:
                try:
                    result['preservation'] = self.preservation.assess(item, result, consumption)
                except (ValueError, KeyError, OSError, TypeError) as error:
                    # The original effect was saved. Missing or invalid evidence
                    # blocks promotion, never repeats the native operation.
                    result['preservation'] = {'policy': self.preservation.revision, 'status': 'unknown',
                        'evidence': [], 'reason': str(error)}
                write_json(self.service.store.root / 'calls' / handle / 'preservation-result.json', result['preservation'])
            result['workbench'] = self._report(item, result, context)
            result['workbench']['context_record'] = decision_context['context_record']
            result['workbench']['capability'] = contract['capability']
            result['workbench']['method'] = contract['method']
            return result

        return self.service._run_episode_callback(self.episode, 'execute_modeling_task',
            arguments, run, prepared=reserve)

    def review_basis(self, task):
        """Read the exact review basis after looking at linked actual images."""
        self.record = read_json(self.path)
        state = self._context()
        if self.record['results'][task]['status'] != 'completed':
            raise ValueError('Review requires an actual completed result')
        return self._review_basis(task, state)

    def record_review(self, task, *, expected_basis, judgment, evidence, lesson=None, experience=None):
        """Record actual scoped visual interpretation and optional conditional lesson.

        Judgment and lesson text are deliberately public, as in current feedback.
        Evidence, task identity and locators stay private. Only actual review supplies a judgment.
        """
        from .learning import validate_lesson, retain_review
        if lesson is not None:
            lesson = validate_lesson(lesson)
        if expected_basis != self.review_basis(task):
            raise ValueError('Candidate, inputs or intent changed since visual review')
        if (set(judgment) != {'disposition', 'scope', 'reason', 'next_question'}
                or judgment['disposition'] not in ('useful', 'rejected', 'unresolved')
                or not judgment['scope'] or not judgment['reason'] or not evidence):
            raise ValueError('Scoped visual judgment needs disposition, reason, next question and actual evidence')
        pinned = [pin_link(self.service, link) for link in evidence]
        result = self.service._run_episode_callback(self.episode, 'review_modeling_task',
            {'task': task, 'basis': expected_basis, 'judgment': judgment, 'evidence': pinned, 'lesson': lesson},
            lambda: {'status': 'completed', 'basis': expected_basis, 'task': task,
                'submitted_at': datetime.now(timezone.utc).isoformat(),
                'judgment': deepcopy(judgment), 'evidence': pinned, 'lesson': lesson,
                'user_acceptance': 'not implied'})
        write_json(self.directory / 'reviews' / (result['operation_handle'] + '.json'), result)
        retain_review(self.service, self.directory, result)
        self._account_review(result)
        if experience is not None:
            self.record_review_experience(result['operation_fact'], **experience)
        return result

    def record_review_experience(self, review_fact, **experience):
        """Source-linked close-out after the actual saved review; safe to resume."""
        from .learning import record_method_experience
        fact = self.service.store.get(review_fact, 'operation_fact')
        if fact['intent']['episode'] != self.episode:
            raise ValueError('Review belongs to another episode')
        return record_method_experience(self.service, review_fact=review_fact, **experience)

    def begin_review(self, task):
        """Open actual evidence and its timing interval together; no native call.

        Elapsed review time is not model token usage. Start before inspecting the
        returned artifacts. Reopening the same basis reuses its running timer.
        """
        basis = self.review_basis(task)
        rows = [read_json(p) for p in (self.directory/'review-work').glob('*.json')]
        row = next((r for r in rows if r['basis'] == basis and not r.get('completed')), None)
        if row is None:
            token = self.begin_intervention(kind='evidence_interpretation', reason='Review returned evidence: '+task)
            row = {'task': task, 'basis': basis, 'timer': token, 'completed': False}
            write_json(self.directory/'review-work'/(token+'.json'), row)
        return {'basis': basis, 'timer': row['timer'], 'result': deepcopy(self.record['results'][task]['result'])}

    def _account_review(self, review):
        for path in (self.directory/'review-work').glob('*.json'):
            row = read_json(path)
            if row['basis'] == review['basis'] and not row.get('completed'):
                intervention = self.end_intervention(row['timer'], evidence=review['evidence'])
                row.update(completed=True, review=review['operation_handle'], intervention=intervention['operation_handle'])
                write_json(path, row)
                break
        # No inferred duration if the reviewer did not start an interval. The
        # metrics explicitly count this submission as unmeasured review work.

    def record_intervention(self, *, kind, reason, evidence, seconds=None, timer=None):
        """Retain actual Astra work; time is explicitly reported, never inferred."""
        from .session_metrics import INTERVENTION_KINDS
        if (kind not in INTERVENTION_KINDS or not reason or not evidence or
                seconds is not None and (type(seconds) not in (float, int)
                or not math.isfinite(seconds) or seconds < 0)):
            raise ValueError('Known intervention kind, reason, evidence and optional actual duration required')
        self._validate_episode(self.owner)
        pinned = [pin_link(self.service, link) for link in evidence]
        data = {'kind': kind, 'reason': reason, 'seconds': seconds, 'evidence': pinned}
        if timer is not None:
            data['timer'] = timer
        result = self.service._run_episode_callback(self.episode, 'record_astra_intervention', data,
            lambda: {**deepcopy(data), 'status': 'completed',
                     'submitted_at': datetime.now(timezone.utc).isoformat()})
        write_json(self.directory / 'interventions' / (result['operation_handle'] + '.json'), result)
        return result

    def metrics(self):
        return session_metrics(self.directory)

    def begin_intervention(self, *, kind, reason):
        """Start an explicit elapsed-work interval; survives owner process restart."""
        from .session_metrics import INTERVENTION_KINDS
        if kind not in INTERVENTION_KINDS or not reason:
            raise ValueError('Known intervention kind and concrete purpose required')
        self._validate_episode(self.owner)
        token = uuid.uuid4().hex
        write_json(self.directory / 'intervention-timers' / (token + '.json'), {
            'token': token, 'kind': kind, 'reason': reason, 'started_at': time.time(),
            'status': 'running', 'owner': self.owner, 'episode': self.episode})
        return token

    def end_intervention(self, token, *, evidence):
        if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise ValueError('Exact intervention timer token required')
        path = self.directory / 'intervention-timers' / (token + '.json')
        timer = read_json(path)
        if timer['owner'] != self.owner or timer['episode'] != self.episode:
            raise ValueError('Timer belongs to another operating session')
        if timer['status'] == 'completed':
            return read_json(self.directory / 'interventions' / (timer['operation_handle'] + '.json'))
        # Recover an interruption after the immutable fact/index was saved.
        prior = [read_json(p) for p in (self.directory / 'interventions').glob('*.json')]
        result = next((row for row in prior if row.get('timer') == token), None)
        if result is None:
            result = self.record_intervention(kind=timer['kind'], reason=timer['reason'],
                seconds=max(0., time.time() - timer['started_at']), evidence=evidence, timer=token)
        timer.update(status='completed', operation_handle=result['operation_handle'])
        write_json(path, timer)
        return result

    def recover(self, task):
        """Restore only an already returned, qualified result; never call a handler."""
        entry = self.record['results'][task]
        if (self.directory / 'controller' / 'controller.lock').exists():
            raise ValueError('Stop or reconcile the live controller before recovering its execution index')
        if entry['status'] in SETTLED:
            self._recover_controller(entry)
            return deepcopy(entry['result'])
        handle = entry.get('operation_handle')
        if not handle:
            raise ValueError('No reserved operation receipt; inspect original effects')
        outcome = read_json(self.service.store.root / 'calls' / handle / 'result.json')
        result = outcome.get('result', {})
        if outcome.get('status') != 'returned' or result.get('status') not in SETTLED or not result.get('workbench'):
            raise ValueError('Operation has no known qualified return; reconcile original capability-result/effects without replay')
        recovered = self.service.reconcile_operation(handle)
        entry.update(status=result['status'], result={**result, 'operation_handle': handle,
            'operation_fact': recovered['operation_fact']})
        write_json(self.path, self.record)
        self._recover_controller(entry)
        return deepcopy(entry['result'])

    def repair_report(self, task, report):
        """Repair interpretation/indexing of a known return without repeating it.

        The old raised result is preserved. This creates a new evidence-backed
        report operation and resolves only the original reporting failure.
        """
        if (self.directory / 'controller' / 'controller.lock').exists():
            raise ValueError('Stop or reconcile the live controller before report repair')
        entry = self.record['results'][task]
        projection_only = (entry['status'] in SETTLED and
                           entry.get('result', {}).get('workbench', {}).get('report_needs_repair') is True)
        if entry['status'] not in ('running', 'needs_reconciliation') and not projection_only:
            raise ValueError('Task does not need report reconciliation')
        handle = entry['operation_handle']
        folder = self.service.store.root / 'calls' / handle
        intent = read_json(folder / 'intent.json')
        source = folder / 'capability-result.json'
        raw = read_json(source)
        if (intent.get('episode') != self.episode or
                fingerprint(intent['arguments']['task']) != entry['definition'] or
                raw.get('status') not in SETTLED):
            raise ValueError('Report repair requires the exact known returned capability receipt')
        adapter = self.report_adapter
        try:
            self.report_adapter = lambda item, result: report
            repaired = self._report(intent['arguments'].get('compiled_task', intent['arguments']['task']), raw, intent['arguments'])
            if repaired['report_needs_repair']:
                raise ValueError('Repaired report still exceeds the public findings budget')
        finally:
            self.report_adapter = adapter
        result = self.service._run_episode_callback(self.episode, 'repair_modeling_report',
            {'original_handle': handle, 'evidence': [str(source)], 'report': repaired},
            lambda: {**raw, 'workbench': repaired, 'original_operation_handle': handle})
        if not projection_only:
            self.service.reconcile_operation(handle, observed={
                'effect_status': 'confirmed_returned', 'basis': 'Exact retained capability return; only later report validation/indexing failed',
                'report_repair': result['operation_handle']}, evidence_paths=[str(source)])
        entry.update(status=raw['status'], result=result)
        write_json(self.path, self.record)
        self._recover_controller(entry)
        return deepcopy(result)

    def _recover_controller(self, entry):
        path = self.directory / 'controller' / 'controller.json'
        if not path.exists():
            return
        controller = read_json(path)
        if controller['owner'] != self.owner:
            raise ValueError('Controller owner changed')
        for attempt in controller['attempts']:
            if (attempt['key'] == entry['attempt_key'] and
                    attempt['status'] in ('running', 'needs_reconciliation')):
                attempt.update(status=entry['status'], result=deepcopy(entry['result']),
                               recovery='Restored exact durable episode result without dispatch')
        write_json(path, controller)

    def recover_selection(self, *, expected_selection, observed, evidence):
        """Reconcile an old selection refusal from explicit no-dispatch evidence.

        This route only clears a proven undispatched selection. Historical requests
        and uncertain native effects retain their original receipts.
        """
        if (self.directory / 'controller' / 'controller.lock').exists():
            raise ValueError('Stop or reconcile the active controller before selection recovery')
        path = self.directory / 'controller' / 'controller.json'
        controller = read_json(path)
        selection = controller.get('selection')
        if (controller['owner'] != self.owner or selection is None or
                fingerprint(selection) != expected_selection):
            raise ValueError('Original pending selection identity changed')
        if (observed.get('effect_status') != 'confirmed_not_dispatched' or
                not observed.get('basis') or not evidence or any(
                    attempt['status'] in ('running', 'needs_reconciliation') for attempt in controller['attempts'])):
            raise ValueError('Require explicit no-dispatch evidence; uncertain operations or requests cannot be cleared')
        pinned = [pin_link(self.service, link) for link in evidence]
        result = self.service._run_episode_callback(self.episode, 'recover_modeling_selection',
            {'selection': selection, 'observed': observed, 'evidence': pinned},
            lambda: {'status': 'completed', 'selection': expected_selection,
                     'observed': observed, 'evidence': pinned, 'effects_replayed': False})
        current = read_json(path)
        if fingerprint(current.get('selection')) != expected_selection:
            raise ValueError('Selection changed while recording recovery')
        current['selection'] = None
        current.setdefault('selection_recoveries', []).append(result)
        write_json(path, current)
        return result
