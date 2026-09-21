"""Authority-first, source-exact retrieval. Lexical matches are not anatomy judgments."""
from pathlib import Path
import json
import re
from .store import digest


def terms(text):
    return set(re.findall(r'[a-z0-9]{3,}', str(text).lower())) - {'the','and','for','from','with','this','that','before','after','into','not','its','are','was','use'}


def passages(path, query, store=None):
    path = Path(path)
    raw = path.read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    wanted = terms(query)
    title, rows = '', []
    boundaries = {0, len(lines)}
    for i, line in enumerate(lines):
        if not line.strip() or line.startswith(('#', '|')):
            boundaries.update((i, i + 1))
    ordered, asset = sorted(boundaries), None
    for start, end in zip(ordered, ordered[1:]):
        block = '\n'.join(lines[start:end]).strip()
        if block.startswith('#'):
            title = block
            continue
        hits = wanted & terms(block)
        identifiers = set(re.findall(r'\bl\d+\b',query.lower()))
        exact_heading = bool(identifiers & terms(title))
        if exact_heading: hits |= identifiers
        if not block or not hits:
            continue
        if store and asset is None:
            asset = store.blob(path)
        # Keep the actual matching paragraph. Long paragraphs expose all matching
        # windows and exact offsets rather than dropping a decisive late sentence.
        spans = [[0, len(block)]]
        if len(block) > 2200:
            spans = []
            for m in re.finditer(r'[a-z0-9]{3,}', block.lower()):
                if m.group() not in wanted:
                    continue
                a, b = max(0, m.start()-350), min(len(block), m.end()+650)
                if spans and a <= spans[-1][1]:
                    spans[-1][1] = max(b, spans[-1][1])
                else:
                    spans.append([a, b])
        excerpts = [dict(text=block[a:b], char_start=a, char_end=b) for a,b in spans]
        rows.append(dict(source=str(path.resolve()), sha256=digest(raw), source_asset=asset,
                         line_start=start+1, line_end=end, title=title,
                         excerpt='\n[...]\n'.join(x['text'] for x in excerpts), passages=excerpts,
                         score=len(hits)+(20 if exact_heading else 0), matched_terms=sorted(hits),
                         coverage='matching paragraphs/windows; full pinned source available'))
    return rows


def applicability(required, context):
    missing, conflicts = [], []
    for key, expected in required.items():
        actual = context.get(key)
        acceptable = expected if isinstance(expected, list) else [expected]
        if actual is None:
            missing.append(key)
        elif not any(x in acceptable for x in (actual if isinstance(actual, list) else [actual])):
            # Free prose is evidence to interpret, not a closed-enum mismatch.
            if key in ('observed_failure','mechanism','support'):
                missing.append(key+' requires interpretation of supplied narrative')
            else:
                conflicts.append(dict(field=key, expected=acceptable, actual=actual))
    return dict(status='inapplicable' if conflicts else 'conditional' if missing else 'applicable',
                missing=missing, conflicts=conflicts)


def retrieve(workspace, store, query, limit=6, context=None, sources=None, row_filter=None):
    if not query.strip() or not 1 <= limit <= 20:
        raise ValueError('Query and a limit from one to twenty required')
    context = context or {}
    if sources is None:
        from .bindings import load_binding
        sources = load_binding(workspace).get('authority', [])
    authority, experience, missing = [], [], []
    for entry in sources:
        entry = dict(path=entry, role='authority') if isinstance(entry,str) else entry
        path = Path(workspace) / entry['path']
        if not path.is_file():
            missing.append(str(path)); continue
        rows = passages(path, query)
        for row in rows:
            row.update(role=entry.get('role','authority'), priority=entry.get('priority',10),
                       applicability='Read current scope; lexical relevance is not approval')
        (experience if entry.get('role') == 'experience' else authority).extend(rows)
    for key, value in store.records(('outcome','episode_judgment','procedure')):
        hit = len(terms(query) & terms(json.dumps(value)))
        if hit:
            experience.append(dict(record=key, source=str(store.root/'records'/(key+'.json')),
                                   score=hit, priority=20, role='retained experience', excerpt=value,
                                   applicability=applicability(value.get('conditions',{}),context)))
    from .learning import experience_rows
    for key, value in experience_rows(store):
        hit = len(terms(query) & terms(json.dumps(value['public'])))
        if hit:
            experience.append(dict(record=key, record_kind='modeling_experience',
                source=str(store.root/'records'/(key+'.json')), score=hit, priority=10,
                role=value['origin'], excerpt=value,
                applicability=applicability(value['public'].get('lesson', {}).get('conditions', {}), context)))
    order = lambda r: (r.get('priority',10), -r['score'], r.get('source',''),r.get('line_start',0))
    authority.sort(key=order)
    experience.sort(key=lambda r: (r.get('applicability',{}).get('status') == 'inapplicable'
                                  if isinstance(r.get('applicability'),dict) else False, -r['score'], order(r)))
    counts = {'authority_matches': len(authority), 'experience_matches': len(experience)}
    # Filter BEFORE limiting: private/unprojectable records must not starve a
    # lower-ranked usable passage. Ordinary source-exact retrieval is unchanged.
    if row_filter:
        authority = [row for row in authority if row_filter(row)]
        experience = [row for row in experience if row_filter(row)]
    return dict(query=query, authority=authority[:limit], matches=experience[:limit],
                coverage=dict(**counts, eligible_authority=len(authority), eligible_experience=len(experience),
                              returned_authority=min(limit,len(authority)),returned_experience=min(limit,len(experience)),
                              missing_sources=missing,expand='Increase limit or read full source_asset / read_record; absence is not proof of no rule'),
                method='authority before similarity; actual matching passages; explicit applicability')
