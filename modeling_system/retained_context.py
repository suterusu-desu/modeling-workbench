"""Source-exact experience retrieval at the decision boundary, without native IO."""
from copy import deepcopy
from pathlib import Path
import json
from .controller import fingerprint, write_json
from .experience import retrieve
from .learning import public_experience
from .work_limits import work_limits


class RetainedContext:
    def __init__(self, service, *, query=None, project=None, sources=None, limit=None, context=None, budget=None):
        self.budget = work_limits(budget)
        limit = self.budget.context_passages if limit is None else limit
        if (project is not None and not callable(project)) or not 1 <= limit <= self.budget.context_passages:
            raise ValueError('Public projection and passage count within the explicit budget required')
        self.service, self.query, self.project = service, query, project
        self.sources, self.limit, self.context = sources, limit, context

    def __call__(self, state, items, outcomes):
        query = self.query(state, items, outcomes) if callable(self.query) else self.query
        if query is None:
            query = json.dumps({'situation': state.get('public_state', {}),
                                'offered_work': [item['description'] for item in items]})
        context = self.context(state) if self.context else state.get('public_state', {})
        projected = {}
        def project(row):
            summary = public_experience(row)
            if summary is None and self.project:
                summary = self.project(deepcopy(row))
            if summary is None:
                return False
            if not isinstance(summary, (dict, str)) or not summary:
                raise ValueError('Public experience must have deliberate semantic content')
            projected[fingerprint(row)] = summary
            return True
        result = retrieve(self.service.workspace, self.service.store, query,
                          self.budget.retrieval_candidates, context, self.sources, row_filter=project)
        public, exact, omitted, candidates = [], [], 0, []
        seen, duplicates = set(), 0
        for kind in ('authority', 'matches'):
            for row in result[kind]:
                # The workspace explicitly removes identities and locators. The
                # compact owner view links back to the exact local evidence.
                summary = projected[fingerprint(row)]
                identity = fingerprint(summary)
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                candidate = {'index': len(exact), 'role': kind, 'content': summary}
                if (len(candidates) < self.budget.retrieval_candidates
                        and len(json.dumps(candidates + [candidate]).encode()) <= self.budget.retrieval_bytes):
                    candidates.append(deepcopy(candidate))
                size = len(json.dumps(public + [candidate], ensure_ascii=True).encode())
                if len(public) >= self.limit or size > self.budget.context_bytes:
                    omitted += 1
                    exact.append({'index': None, 'source': row})
                    continue
                index = candidate['index']
                public.append(candidate)
                exact.append({'index': index, 'source': row})
        # Do not hide missing retrieval coverage or turn relevance into authority.
        return {'public': {'passages': public, 'coverage': {
                    'authority_matches': result['coverage']['authority_matches'],
                    'experience_matches': result['coverage']['experience_matches'],
                    'missing_source_count': len(result['coverage']['missing_sources']),
                    'excluded_by_projection': sum(result['coverage'][kind + '_matches'] - result['coverage']['eligible_' + kind]
                                                  for kind in ('authority', 'experience')),
                    'unreturned_eligible_passages': sum(result['coverage']['eligible_' + kind] - result['coverage']['returned_' + kind]
                                                        for kind in ('authority', 'experience')),
                    'duplicate_public_passages': duplicates,
                    'omitted_public_passages': omitted,
                    'semantic_candidates': len(candidates),
                    'unreturned_semantic_candidates': len(seen) - len(candidates)},
                'limits': 'Historical reviews and conditional lessons inform choices, not current state or user acceptance. Different prerequisites can change a method result; interpret the evidence rather than blindly repeat or ban it. Relevance never admits a guide or approves appearance.'},
            'candidates': candidates, 'exact': exact, 'query': query,
            'revision': fingerprint({'retrieval': result, 'public': public, 'candidates': candidates,
                                     'budget': self.budget.record()})}

    @staticmethod
    def retain(directory, record):
        path = Path(directory) / 'experience' / (record['revision'] + '.json')
        if not path.exists():
            write_json(path, record)
        return path
