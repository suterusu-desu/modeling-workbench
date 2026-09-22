"""Reusable executable methods, with semantic choice left to Jev.

Factories qualify actual inputs. This registry does not infer anatomy or turn a
historical success into permission. Register mechanisms once, then bind a scope.
"""
from copy import deepcopy
from .experience import applicability
from .decision_budget import decision_budget


class MethodCatalog:
    def __init__(self, methods, *, context=None, budget=None):
        self.budget = decision_budget(budget)
        if (not isinstance(methods, list) or not methods or len(methods) > self.budget.max_methods
                or any(not m.get('id') or not m.get('description')
                       or not callable(m.get('build')) for m in methods)
                or len({m['id'] for m in methods}) != len(methods)):
            raise ValueError('Unique described executable methods required')
        self.methods = {m['id']: dict(m) for m in methods}
        for method in self.methods.values():
            remedies = method.get('remedy_methods', [])
            if (not isinstance(remedies, list) or len(set(remedies)) != len(remedies)
                    or not set(remedies) <= self.methods.keys() or method['id'] in remedies):
                raise ValueError('Remedy methods must name distinct other registered methods')
        self.context = context or (lambda state: state.get('public_state', {}))
        self.excluded = {}

    def _bind(self, method, rows, *, choose):
        if rows is None:
            return []
        rows = deepcopy(rows if isinstance(rows, list) else [rows])
        for row in rows:
            if row.get('required') and choose:
                raise ValueError('A method choice cannot be required housekeeping')
            row['workbench']['method_choice'] = method['id']
            row['description'] = {'operation': row['description'],
                                  'method': deepcopy(method['description'])}
            if choose:
                row['select_with_jev'] = True
                if method.get('method_checks'):
                    row.setdefault('decision', {})['method_checks'] = deepcopy(method['method_checks'])
                if method.get('remedy_methods'):
                    checks = row.setdefault('decision', {}).setdefault('method_checks', {})
                    checks['remedy_methods'] = deepcopy(method['remedy_methods'])
        return rows

    def __call__(self, state, previous=None):
        rows, self.excluded = [], {}
        context = self.context(deepcopy(state))
        for method in self.methods.values():
            fit = applicability(method.get('conditions', {}), context)
            if fit['status'] != 'applicable':
                self.excluded[method['id']] = fit
                continue
            supplied = method['build'](deepcopy(state), deepcopy(previous))
            rows.extend(self._bind(method, supplied, choose=True))
        if len(rows) > self.budget.max_tasks or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Bounded methods must produce unique task IDs')
        return rows

    def followup(self, stage):
        """Route a pipeline stage from the actual selected method and its result.

        A fixed apply after selected preparation needs no duplicate inference.
        Multiple offered follow-ups still use the queue's ordinary Jev choice.
        """
        def build(state, previous):
            key = previous['task']['workbench']['method_choice']
            method = self.methods[key]
            factory = method.get(stage)
            if not callable(factory):
                raise ValueError('Selected method has no executable ' + stage)
            rows = self._bind(method, factory(deepcopy(state), deepcopy(previous)), choose=False)
            for row in rows:
                row['workbench']['selected_method_from'] = previous['task']['id']
            return rows
        return build


class ParameterizedCatalog(MethodCatalog):
    """Bind registered implementations to qualified data, without copying factories.

    implementations maps mechanism IDs to bind(state, previous, parameters), or
    a dictionary with build and follow-up stage functions of that signature.
    Each binding supplies id, implementation, description, parameters and optional
    applicability conditions. Enumerate useful alternatives, not blind Cartesian
    parameter sweeps. Identical mechanism/parameter aliases are rejected.
    """
    def __init__(self, implementations, bindings, *, context=None, budget=None):
        from .controller import fingerprint
        seen, methods = set(), []
        for binding in bindings:
            key = binding['implementation']; parameters = deepcopy(binding['parameters'])
            implementation = implementations.get(key)
            factories = {'build': implementation} if callable(implementation) else implementation
            if (not isinstance(factories, dict) or not callable(factories.get('build'))
                    or not all(callable(fn) for fn in factories.values()) or not isinstance(parameters, dict)):
                raise ValueError('Registered executable implementation and named parameters required')
            identity = fingerprint({'implementation': key, 'parameters': parameters})
            if identity in seen:
                raise ValueError('Duplicate mechanism and parameters are not meaningful alternatives')
            seen.add(identity)
            def bind_factory(fn, row=deepcopy(binding), digest=identity):
                def build(state, previous):
                    tasks = fn(state, previous, deepcopy(row['parameters']))
                    if tasks is None:
                        return None
                    tasks = deepcopy(tasks if isinstance(tasks, list) else [tasks])
                    for task in tasks:
                        task['workbench']['recipe'] = {'implementation': row['implementation'],
                            'parameters_revision': digest}
                    return tasks
                return build
            methods.append({'id': binding['id'], 'description': binding['description'],
                            'conditions': binding.get('conditions', {}),
                            'method_checks': deepcopy(binding.get('method_checks', {})),
                            'remedy_methods': deepcopy(binding.get('remedy_methods', [])),
                            **{stage: bind_factory(fn) for stage, fn in factories.items()}})
        super().__init__(methods, context=context, budget=budget)


def bind_array_method(state, previous, parameters):
    """Ready-made ParameterizedCatalog implementation for saved-array recipes.

    Parameters carry task, revision, question, method, operation, inputs, reads,
    writes and optional operation parameters/lane/handler. File identities must
    be current task dependencies; semantic qualification remains in method text
    and its workspace evidence. It never loads Blender or guesses correspondence.
    """
    from .candidate_pipeline import capability_task
    from .preparation import ARRAY_OPERATIONS
    if parameters['operation'] not in ARRAY_OPERATIONS:
        raise ValueError('Known saved-array operation required')
    reads = parameters['reads']
    revisions = {state['values'][key] for key in reads}
    if any(ref['sha256'] not in revisions for ref in parameters['inputs'].values()):
        raise ValueError('Every saved-array input must be bound to the method reads')
    return capability_task(state, key=parameters['task'], revision=parameters['revision'],
        lane=parameters.get('lane', 'preparation'), handler=parameters.get('handler', 'arrays'),
        description=parameters['question'], completion='Recorded array result and exact input provenance.',
        profile='analysis', capability=parameters['operation'], method=parameters['method'],
        bindings={'evidence': reads}, reads=reads, writes=parameters.get('writes', []),
        payload={'operation': parameters['operation'], 'inputs': parameters['inputs'],
            'parameters': parameters.get('parameters', {}),
            'stop_on_no_effect': parameters.get('stop_on_no_effect', False)})
