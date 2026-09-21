"""Source-exact experience retrieval at the decision boundary, without native IO."""
from copy import deepcopy
from pathlib import Path
import json
from .controller import fingerprint, write_json
from .experience import retrieve


class RetainedContext:
    def __init__(self, service, *, query, project, sources=None, limit=4, context=None):
        if not callable(project) or not 1 <= limit <= 4:
            raise ValueError('Explicit public projection and one to four passages required')
        self.service, self.query, self.project = service, query, project
        self.sources, self.limit, self.context = sources, limit, context

    def __call__(self, state, items, outcomes):
        query = self.query(state, items, outcomes) if callable(self.query) else self.query
        context = self.context(state) if self.context else state.get('public_state', {})
        result = retrieve(self.service.workspace, self.service.store, query,
                          self.limit, context, self.sources)
        public, exact, omitted = [], [], 0
        used = 0
        for kind in ('authority', 'matches'):
            for row in result[kind]:
                # The workspace explicitly removes identities and locators. No
                # retrieved prose, filename or record goes on wire by default.
                summary = self.project(deepcopy(row))
                if summary is None:
                    continue
                if not isinstance(summary, (dict, str)) or not summary:
                    raise ValueError('Public experience must have deliberate semantic content')
                size = len(json.dumps(summary, ensure_ascii=True).encode())
                if used + size > 2400:
                    omitted += 1
                    exact.append({'index': None, 'source': row})
                    continue
                used += size
                index = len(public)
                public.append({'index': index, 'role': kind, 'content': summary})
                exact.append({'index': index, 'source': row})
        # Do not hide missing retrieval coverage or turn relevance into authority.
        return {'public': {'passages': public, 'coverage': {
                    'authority_matches': result['coverage']['authority_matches'],
                    'experience_matches': result['coverage']['experience_matches'],
                    'missing_source_count': len(result['coverage']['missing_sources']),
                    'omitted_public_passages': omitted},
                'limits': 'Historical methods and failures inform choices; current bindings govern applicability. Relevance never admits a guide or approves appearance.'},
            'exact': exact, 'query': query,
            'revision': fingerprint({'retrieval': result, 'public': public})}

    @staticmethod
    def retain(directory, record):
        path = Path(directory) / 'experience' / (record['revision'] + '.json')
        if not path.exists():
            write_json(path, record)
        return path
