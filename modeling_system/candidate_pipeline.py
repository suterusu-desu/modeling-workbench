"""Compile reusable candidate stages into the existing OperatingSession catalog.

Factories bind private scene capabilities to actual results. This is not another
executor: all effects, reviews and recovery remain in the existing session.
"""
from copy import deepcopy
from pathlib import Path
from .controller import fingerprint, read_json, write_json
from .operating_session import PROFILES


def capability_task(state, *, key, revision, lane, handler, description, completion,
                    profile, capability, method, bindings, reads, writes=(),
                    payload=None, native=False):
    """Bind a qualified capability in any work lane to current exact dependencies.

    The caller supplies real support and semantic roles. Names or descriptions do
    not qualify an operation. Use inside a pipeline factory to freeze this binding.
    """
    if profile not in PROFILES or not set(reads) <= state['values'].keys():
        raise ValueError('Known modeling profile and actual dependency revisions required')
    required = set(PROFILES[profile]['inputs']) | ({'adapter'} if native else set())
    if (not required <= bindings.keys() or not set(writes) <= set(reads)
            or any(not keys or not set(keys) <= set(reads) for keys in bindings.values())):
        raise ValueError('Complete input relationships and covered writes required')
    return {'id': key, 'revision': revision, 'lane': lane, 'handler': handler,
        'description': deepcopy(description), 'completion_condition': completion,
        'reads': {key: state['values'][key] for key in reads}, 'writes': list(writes),
        'requires': {}, 'payload': deepcopy(payload or {}),
        'workbench': {'profile': profile, 'capability': capability, 'method': method,
            'bindings': deepcopy(bindings), 'native': native}}


class CandidatePipeline:
    """Freeze each stage once; advance selected work without copying run scripts.

    Each factory(state, previous) returns qualified work items. previous contains
    the exact selected task and its settled result, or None for the first stage.
    Preparation and candidate stages may offer genuine alternatives. Verification,
    review preparation and retention are fixed obligations, with real image review
    before expensive verification (default) and before retention.
    """
    def __init__(self, directory, *, revision, candidates, verify, retain,
                 prepare=None, review=None, early_review=True):
        if (not revision or type(early_review) is not bool
                or not all(callable(f) for f in (candidates, verify, retain))):
            raise ValueError('Pipeline revision and explicit early-review policy required')
        self.path = Path(directory) / 'candidate-pipeline.json'
        self.revision, self.early_review = revision, early_review
        self.factories = {k: v for k, v in [('prepare', prepare), ('candidate', candidates),
            ('verify', verify), ('review', review), ('retain', retain)] if v is not None}
        if any(not callable(v) for v in self.factories.values()):
            raise ValueError('Executable stage factories required')
        self.binding = {'revision': revision, 'stages': list(self.factories), 'early_review': early_review}

    def adopt_completed_candidate(self, task, outcome):
        """Attach original settled evidence, without rewriting a task or replaying it."""
        if ('prepare' in self.factories or task.get('workbench', {}).get('profile') != 'appearance_edit'
                or outcome.get('status') != 'completed' or not outcome.get('result')
                or outcome.get('definition') != fingerprint(task)):
            raise ValueError('Exact completed candidate and a pipeline without preceding preparation required')
        # Prior execution retains its original method/runtime and qualification.
        # The existing session checks that this same result still owns the task.
        record = {'binding': self.binding, 'stages': {'candidate': [deepcopy(task)]},
            'adoption': {'task': task['id'], 'outcome': fingerprint(outcome),
                         'provenance': 'Existing execution; not run by this pipeline'}}
        if self.path.exists():
            current = read_json(self.path)
            if (current['binding'] != self.binding or current.get('adoption') != record['adoption']
                    or current['stages']['candidate'] != record['stages']['candidate']):
                raise ValueError('Existing pipeline differs from the original candidate adoption')
        else:
            write_json(self.path, record)

    def __call__(self, state, outcomes):
        record = read_json(self.path) if self.path.exists() else {'binding': self.binding, 'stages': {}}
        if record['binding'] != self.binding:
            raise ValueError('Pipeline changed; preserve this instance and start a new scoped pipeline')
        adopted = record.get('adoption')
        if adopted and fingerprint(outcomes.get(adopted['task'])) != adopted['outcome']:
            raise ValueError('Adopted original candidate outcome changed or is missing')
        items, previous = [], None
        for stage, factory in self.factories.items():
            if stage not in record['stages']:
                supplied = deepcopy(factory(deepcopy(state), deepcopy(previous)))
                if not isinstance(supplied, list) or not supplied or len(supplied) > 32:
                    raise ValueError('Stage needs a bounded nonempty catalog')
                fixed = stage in ('verify', 'review', 'retain')
                if fixed and len(supplied) != 1:
                    raise ValueError('Fixed follow-up stages require exactly one operation')
                for item in supplied:
                    contract = item['workbench']
                    if stage == 'candidate' and contract['profile'] != 'appearance_edit':
                        raise ValueError('Candidate must retain appearance-edit qualifications')
                    expected = {'verify': 'verification', 'retain': 'retention'}.get(stage)
                    if expected and contract['profile'] != expected:
                        raise ValueError('Stage profile does not match its modeling obligation')
                    if fixed and contract['profile'] == 'appearance_edit':
                        raise ValueError('A modeling pass cannot be disguised as fixed housekeeping')
                    item['required'] = fixed
                    item.setdefault('requires', {})
                    if previous:
                        parent = previous['task']
                        item['requires'][parent['id']] = ['completed']
                        contract.setdefault('consumes', {})[parent['id']] = list(
                            PROFILES[parent['workbench']['profile']]['outputs'])
                        if stage == 'retain' or stage == 'verify' and self.early_review:
                            contract['visual_review'] = parent['id']
                prior_ids = {item['id'] for rows in record['stages'].values() for item in rows}
                ids = [item['id'] for item in supplied]
                if len(set(ids)) != len(ids) or prior_ids.intersection(ids):
                    raise ValueError('Pipeline task IDs must be unique')
                record['stages'][stage] = supplied
                write_json(self.path, record)
            supplied = record['stages'][stage]
            attempted = [item for item in supplied if item['id'] in outcomes]
            if len(attempted) > 1:
                raise ValueError('More than one alternative ran in the same stage; reconcile original work')
            if not attempted:
                items.extend(deepcopy(supplied))
                break
            task = attempted[0]
            entry = outcomes[task['id']]
            if entry['definition'] != fingerprint(task):
                raise ValueError('Executed task definition differs from the frozen pipeline')
            # Unselected alternatives disappear from eligibility, not history.
            # A failed mechanism never causes the next variant to run implicitly.
            items.append(deepcopy(task))
            if entry['status'] != 'completed':
                break
            previous = {'task': deepcopy(task), 'result': deepcopy(entry['result'])}
        return items
