"""Reusable executable methods, with semantic choice left to Jev.

Factories qualify actual inputs. This registry does not infer anatomy or turn a
historical success into permission. Register mechanisms once, then bind a scope.
"""
from copy import deepcopy
from .experience import applicability


class MethodCatalog:
    def __init__(self, methods, *, context=None):
        if (not isinstance(methods, list) or not methods or len(methods) > 32
                or any(not m.get('id') or not m.get('description')
                       or not callable(m.get('build')) for m in methods)
                or len({m['id'] for m in methods}) != len(methods)):
            raise ValueError('Unique described executable methods required')
        self.methods = {m['id']: dict(m) for m in methods}
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
        if len(rows) > 128 or len({row['id'] for row in rows}) != len(rows):
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
            return self._bind(method, factory(deepcopy(state), deepcopy(previous)), choose=False)
        return build
