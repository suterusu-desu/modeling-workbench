"""Required evaluated outcomes carried through existing modeling operations.

Adapters are trusted, source-pinned geometric code, not a sandbox or an anatomy
solver. Their coverage/bind/measure functions must be qualified against actual
construction. Task-supplied pass flags and Jev confidence have no authority here.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

from .controller import fingerprint


def file_ref(path):
    path = Path(path).resolve(strict=True)
    with path.open('rb') as stream:
        return {'path': str(path), 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}


def checked(ref):
    actual = file_ref(ref['path'])
    if actual['sha256'] != ref['sha256']:
        raise ValueError('Required preservation source changed')
    return Path(actual['path'])


def same_ref(a, b):
    return (a.get('sha256') == b.get('sha256')
            and Path(a.get('path', '')).resolve() == Path(b.get('path', '')).resolve())


def _refs(value):
    if isinstance(value, dict):
        if {'path', 'sha256'} <= value.keys():
            yield value
        else:
            for child in value.values(): yield from _refs(child)
    elif isinstance(value, list):
        for child in value: yield from _refs(child)


def _key(ref):
    return 'preservation_file:' + fingerprint(str(Path(ref['path']).resolve()))


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def validate_requirements(document):
    if document.get('schema_version') != 1 or not document.get('construction') or not document.get('authority'):
        raise ValueError('Preservation needs current construction and authority sources')
    rows = document.get('outcomes')
    if not isinstance(rows, list) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Unique established outcome identities required')
    for row in rows:
        if not all(row.get(k) for k in ('id', 'description', 'baseline', 'guide', 'evidence', 'constraints', 'cells')):
            raise ValueError('Each outcome needs baseline, guide, evidence, constraint inputs and coverage cells')
        if not isinstance(row['constraints'], dict) or len({c['id'] for c in row['cells']}) != len(row['cells']):
            raise ValueError('Named constraints and unique region/pose/metric cells required')
        for cell in row['cells']:
            rule = cell.get('rule', {})
            if (not all(cell.get(k) for k in ('id', 'region', 'pose', 'metric'))
                    or not rule or set(rule)-{'maximum', 'max_increase'}
                    or any(not _number(v) or v < 0 for v in rule.values())):
                raise ValueError('Cells require explicit scope and finite nonnegative comparison bounds')
    for row in document.get('rejections', []):
        if not row.get('subject') or not row.get('evidence') or not row.get('reason'):
            raise ValueError('User rejection needs exact subject and source-backed reason')
    for ref in _refs(document): checked(ref)
    return deepcopy(document)


class PreservationPolicy:
    """One required-outcome ledger, independent of ranked historical passages.

    adapters[name] = {sources: [file refs], coverage: fn(item, document),
                     bind: fn(item, constraints), measure: fn(item, result, constraints)}.
    All callbacks must originate in a pinned source file. bind constructs actual
    executable payload arguments using the constraint refs. measure reads saved
    evaluated evidence; this module applies the non-waivable numeric rules.
    """
    def __init__(self, requirements, adapters):
        self.reference = deepcopy(requirements)
        self.document = validate_requirements(json.loads(checked(requirements).read_text(encoding='utf-8')))
        self.adapters = {key: dict(value) for key,value in adapters.items()}
        self.outcomes = {r['id']: r for r in self.document['outcomes']}
        self.references = [self.reference, *list(_refs(self.document))]
        for adapter in self.adapters.values():
            sources = adapter.get('sources', [])
            if not sources: raise ValueError('Geometric adapters must pin their source')
            for name in ('coverage', 'bind', 'measure'):
                fn = adapter.get(name)
                if not callable(fn) or Path(inspect.getsourcefile(fn)).resolve() not in {checked(r) for r in sources}:
                    raise ValueError('Every preservation callback must come from a pinned source file')
            self.references.extend(sources)
        self.adapter_identity = self._adapter_identity()
        self.revision = fingerprint({'requirements': self.reference, 'adapters': self.adapter_identity})

    def _adapter_identity(self):
        return {key: {'sources': value['sources'], 'subject': value.get('subject', ['subject']),
            'functions': {name: {'source': str(Path(inspect.getsourcefile(value[name])).resolve()),
                                'name': value[name].__qualname__} for name in ('coverage','bind','measure')}}
                for key,value in self.adapters.items()}

    def context_values(self):
        values = {}
        for ref in self.references:
            try: values[_key(ref)] = file_ref(ref['path'])['sha256']
            except OSError: values[_key(ref)] = None
        return values

    def fresh(self):
        for ref in self.references: checked(ref)
        if self._adapter_identity() != self.adapter_identity:
            raise ValueError('Preservation adapter binding changed')

    def bind_task(self, item, *, stage, adapter, previous=None):
        """Bind before a CandidatePipeline freezes its task definition."""
        self.fresh()
        if stage not in ('prepare', 'apply', 'verify', 'review', 'retain', 'recovery') or adapter not in self.adapters:
            raise ValueError('Qualified preservation stage and adapter required')
        row = deepcopy(item)
        coverage = self.adapters[adapter]['coverage'](deepcopy(row), deepcopy(self.document))
        ids = coverage.get('outcomes')
        if (not isinstance(ids, list) or len(set(ids)) != len(ids)
                or not set(ids) <= self.outcomes.keys() or not coverage.get('evidence')):
            raise ValueError('Adapter must resolve affected outcomes from actual construction evidence')
        for ref in coverage['evidence']:
            checked(ref)
            if not any(same_ref(ref, source) for source in self.references):
                raise ValueError('Coverage evidence is not a bound construction dependency')
        parent = (previous or {}).get('task', {}).get('workbench', {}).get('preservation')
        if parent:
            if parent['policy'] != self.revision: raise ValueError('Pipeline cannot change its required outcomes')
            ids = sorted(set(ids) | set(parent['outcomes']))
        row['workbench']['preservation'] = {'policy': self.revision, 'stage': stage,
            'adapter': adapter, 'outcomes': sorted(ids), 'coverage': deepcopy(coverage),
            'upstream': deepcopy((previous or {}).get('result', {}).get('preservation')),
            'upstream_task': (previous or {}).get('task', {}).get('id')}
        row['reads'].update({_key(ref): ref['sha256'] for ref in self.references})
        if stage in ('prepare', 'apply'):
            row.setdefault('decision', {}).setdefault('method_checks', {
                'method': 'Does this method apply to the current construction under the source-backed lesson conditions, without repeating a recorded failure unchanged?',
                'coverage': 'Does this proposed operation preserve the established evaluated shapes and relationships through the affected motion, including connected transitions?'})
        return row

    def relevant(self, item):
        return ('preservation' in item.get('workbench', {})
                or item.get('workbench', {}).get('profile') in ('appearance_edit', 'retention'))

    def constraints(self, item):
        self.fresh()
        binding = item.get('workbench', {}).get('preservation')
        if (not binding or binding['policy'] != self.revision or binding['adapter'] not in self.adapters
                or any(item['reads'].get(_key(ref)) != ref['sha256'] for ref in self.references)):
            raise ValueError('Affected operation lacks its current required preservation binding')
        coverage = self.adapters[binding['adapter']]['coverage'](deepcopy(item), deepcopy(self.document))
        if coverage != binding['coverage']:
            raise ValueError('Actual construction coverage changed since task binding')
        rows = [deepcopy(self.outcomes[key]) for key in binding['outcomes']]
        return {'policy': self.revision, 'stage': binding['stage'], 'outcomes': rows,
                'construction': deepcopy(self.document['construction']), 'authority': deepcopy(self.document['authority'])}

    def preflight(self, item, outcomes):
        if not self.relevant(item): return None
        constraints = self.constraints(item)
        binding = item['workbench']['preservation']
        if binding.get('upstream'):
            prior = binding['upstream']
            source = outcomes.get(binding.get('upstream_task'), {})
            if (binding.get('upstream_task') not in item.get('requires', {})
                    or source.get('result', {}).get('preservation') != prior):
                raise ValueError('Preservation prerequisite is not the actual preceding operation result')
            if prior.get('measurements'):
                self.verify_assessment(prior, source['task'], source['result'])
            elif binding['stage'] not in ('verify', 'review', 'recovery'):
                raise ValueError('Missing measurements require the relevant evidence operation')
            if binding['stage'] in ('apply', 'retain') and (prior['policy'] != self.revision or prior['status'] != 'pass'):
                raise ValueError('Relevant prerequisite has missing or failed preservation evidence')
        if binding['stage'] == 'retain':
            prior = binding.get('upstream')
            if not prior or prior.get('kind') != 'evaluated_output':
                raise ValueError('Retention needs source-bound final evaluated preservation evidence')
            if not same_ref(item.get('payload', {}).get('candidate', {}), prior['subject']):
                raise ValueError('Retention candidate differs from measured preservation subject')
            for rejected in self.document.get('rejections', []):
                if same_ref(prior['subject'], rejected['subject']):
                    raise ValueError('Current user rejection overrides earlier useful review')
        return constraints

    def prepare(self, item, outcomes):
        constraints = self.preflight(item, outcomes)
        if constraints is None: return deepcopy(item), None
        binding = item['workbench']['preservation']
        adapter = self.adapters[binding['adapter']]
        bound = adapter['bind'](deepcopy(item), deepcopy(constraints))
        if not isinstance(bound, dict) or set(bound) != {'payload', 'consumed'}:
            raise ValueError('Adapter must construct executable payload and exact consumed constraint inputs')
        expected = {r['id']: r['constraints'] for r in constraints['outcomes']}
        if bound['consumed'] != expected:
            raise ValueError('Preparation/application omitted required constraint inputs')
        # Exact files must be in the executable payload, not a detached prompt or
        # a satisfied=True declaration. Private native/solver code still needs
        # meaningful qualification showing it uses those arguments in its solve.
        payload_refs = list(_refs(bound['payload']))
        for ref in _refs(expected):
            if not any(same_ref(ref, actual) for actual in payload_refs):
                raise ValueError('Required constraints were not bound into actual executable arguments')
            if binding['stage'] == 'apply' and item['workbench'].get('native') and 'job' in bound['payload']:
                dependencies = bound['payload']['job'].get('dependency_hashes', {})
                pinned = {str(Path(path).resolve()): value for path,value in dependencies.items()}
                if pinned.get(str(Path(ref['path']).resolve())) != ref['sha256']:
                    raise ValueError('Native worker did not pin its consumed constraint inputs')
        prepared = deepcopy(item); prepared['payload'] = deepcopy(bound['payload'])
        return prepared, {'constraints': constraints, 'consumed': deepcopy(expected),
            'bound_payload': fingerprint(bound['payload']), 'adapter': binding['adapter']}

    def assess(self, item, result, consumption):
        """Recompute numeric verdicts from the registered saved-evidence reader."""
        if consumption is None: return None
        constraints = consumption['constraints']; binding = item['workbench']['preservation']
        if binding['policy'] != self.revision or constraints['policy'] != self.revision:
            raise ValueError('Actual operation and consumed requirements differ')
        report = self.adapters[binding['adapter']]['measure'](deepcopy(item), deepcopy(result), deepcopy(constraints))
        actual = result
        for key in self.adapters[binding['adapter']].get('subject', ['subject']): actual = actual[key]
        if not isinstance(actual, dict) or not same_ref(report.get('subject', {}), actual):
            raise ValueError('Measurement subject differs from the actual operation result')
        checked(actual)
        return self.evaluate_report(item, report, consumption)

    def evaluate_report(self, item, report, consumption):
        """The reader must bind its actual subject and final evaluated evidence."""
        binding = item['workbench']['preservation']
        self.fresh()
        if not report.get('subject') or not report.get('evidence'):
            raise ValueError('Preservation measurements require actual subject and saved evidence')
        checked(report['subject'])
        for ref in report['evidence']: checked(ref)
        expected_kind = 'prepared_output' if binding['stage'] == 'prepare' else 'evaluated_output'
        if report.get('kind') != expected_kind or not same_ref(report.get('construction', {}), self.document['construction']):
            raise ValueError('Preservation requires current construction and the declared evaluated stage')
        rows = report.get('outcomes', {})
        judgments = []
        for key in binding['outcomes']:
            expected = self.outcomes[key]; measured = rows.get(key, {})
            identity_ok = (same_ref(measured.get('baseline', {}), expected['baseline'])
                           and same_ref(measured.get('guide', {}), expected['guide']))
            cells = measured.get('cells', {})
            for cell in expected['cells']:
                measurement = cells.get(cell['id'], {})
                status = 'unknown'; value = measurement.get('observed'); base = measurement.get('baseline')
                if (identity_ok and _number(value) and measurement.get('region') == cell['region']
                        and measurement.get('pose') == cell['pose'] and measurement.get('metric') == cell['metric']):
                    rule = cell['rule']
                    if 'max_increase' not in rule or _number(base):
                        good = (('maximum' not in rule or value <= rule['maximum'])
                                and ('max_increase' not in rule or value-base <= rule['max_increase']))
                        status = 'pass' if good else 'fail'
                judgments.append({'outcome': key, 'cell': cell['id'], 'status': status,
                    'observed': value, 'baseline': base, 'rule': deepcopy(cell['rule'])})
        status = 'fail' if any(r['status'] == 'fail' for r in judgments) else (
            'unknown' if any(r['status'] == 'unknown' for r in judgments) else 'pass')
        return {'policy': self.revision, 'status': status, 'kind': expected_kind,
            'subject': deepcopy(report['subject']), 'evidence': deepcopy(report['evidence']),
            'measurements': deepcopy(report), 'checks': judgments, 'consumption': deepcopy(consumption),
            'outcomes': deepcopy(binding['outcomes']), 'adapter': binding['adapter'],
            'appearance_acceptance': 'not implied'}

    def verify_assessment(self, assessment, item, result):
        if assessment['policy'] != self.revision: raise ValueError('Preservation policy changed')
        # Re-read actual saved arrays through the pinned geometric reader. A
        # copied/edited report with the same file names is not measurement proof.
        actual = self.assess(item, result, assessment['consumption'])
        if actual != assessment: raise ValueError('Preservation assessment was altered')
        return actual

    def public(self):
        return {'required_outcomes': [{'outcome': r['description'],
            'coverage': [{k:v for k,v in c.items() if k != 'id'} for c in r['cells']]} for r in self.document['outcomes']],
            'construction': deepcopy(self.document.get('public_construction', {})),
            'user_rejections': [r['reason'] for r in self.document.get('rejections', [])],
            'limits': 'Required outcomes cannot be omitted by ranking or waived by a judgment. Missing measurements remain unknown. Geometry checks do not judge likeness.'}
