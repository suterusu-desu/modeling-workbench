"""Source-bound visual targets and corrections, using the Workbench Store/Ledger.

These records communicate historical evidence. They neither drive native geometry
nor authenticate a human, a camera, an approval, or the state of a live scene.
"""
from datetime import datetime, timezone
from pathlib import Path
import math
import uuid
from PIL import Image
from .ledger import Conflict
from .store import atomic_write, canonical


KIND = 'visual_feedback'
COLLECTIONS = ('images', 'targets', 'plans', 'comparisons', 'feedback',
               'agreements', 'reference_roles', 'state_facts', 'presentation_events')
FACETS = ('execution', 'technical_verification', 'visually_useful', 'retained', 'owner_acceptance')
VIEW_FIELDS = ('camera', 'display_state', 'pose', 'lighting', 'framing')


def view_matching(before, after):
    mismatched, unknown = [], []
    for field in VIEW_FIELDS:
        left, right = before['metadata'].get(field), after['metadata'].get(field)
        if not left or not right:
            unknown.append(field)
        elif left != right:
            mismatched.append(field)
    if before['size'] != after['size']:
        mismatched.append('image dimensions')
    return mismatched, unknown


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(field + ' requires nonempty text')
    return value  # Exact wording, including intentional whitespace, is evidence.


def dated(value, field='date'):
    text(value, field)
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise ValueError(field + ' must be an ISO date or datetime') from None
    return value


def user_fact(value):
    if not isinstance(value, dict) or set(value) != {'actor', 'wording', 'date', 'source'}:
        raise ValueError('Explicit user fact requires actor, exact wording, date and source; generic review flags are insufficient')
    if value['actor'] != 'user':
        raise ValueError('Agreement/acceptance must be an explicit user fact')
    return dict(actor='user', wording=text(value['wording'], 'user wording'),
                date=dated(value['date']), source=text(value['source'], 'user fact source'))


def rectangle(value):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError('Region is [x, y, width, height] in normalized image coordinates')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError('Region coordinates must be finite numbers')
    x, y, w, h = value
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.00000001 or y + h > 1.00000001:
        raise ValueError('Region must lie inside the source image')
    return value


class VisualFeedback:
    def __init__(self, store, ledger):
        self.store, self.ledger = store, ledger

    def _board(self, board, expected=None):
        value = self.ledger.read(board)
        if value['kind'] != KIND:
            raise ValueError('Expected a visual feedback board')
        if expected is not None and value['revision'] != expected:
            raise Conflict('Visual feedback changed; reopen before submitting. Your wording has not been applied to a newer result.')
        return value

    def _record(self, value, collection, key):
        if key not in value['data'].get(collection, []):
            raise ValueError('Record does not belong to this board: ' + collection)
        return self.store.get(key, 'visual_' + collection)

    def _append(self, value, records):
        data = {}
        for collection, payloads in records.items():
            refs = [self.store.put('visual_' + collection, dict(p, recorded_at=datetime.now(timezone.utc).isoformat()))
                    for p in payloads]
            data[collection] = [*value['data'].get(collection, []), *refs]
        updated = self.ledger.update(value['handle'], value['revision'], 'open', data)
        return dict(board=updated['handle'], revision=updated['revision'], added={k:v[len(value['data'].get(k, [])):] for k,v in data.items()})

    def create(self, title, idempotency_key):
        value = self.ledger.create(KIND, dict(title=text(title, 'title')), idempotency_key)
        return dict(board=value['handle'], revision=value['revision'], reused=value.get('reused', False))

    def image(self, board, expected_revision, image_path, metadata):
        value = self._board(board, expected_revision)
        allowed = {'label', 'source', 'source_version', 'captured_at', 'camera', 'display_state', 'pose',
                   'lighting', 'framing', 'view_role',
                   'capture_record', 'image_sha256'}
        if not isinstance(metadata, dict) or set(metadata) - allowed:
            raise ValueError('Image metadata fields: ' + ', '.join(sorted(allowed)))
        for field in ('label', 'source', 'source_version', 'captured_at'):
            text(metadata.get(field), field)
        dated(metadata['captured_at'], 'captured_at')
        for field in VIEW_FIELDS:
            if metadata.get(field) is not None and not isinstance(metadata[field], (str, dict)):
                raise ValueError(field + ' must be text, an exact descriptor, or null (unknown)')
        if metadata.get('view_role') not in (None, 'eye_context', 'whole_face', 'detail'):
            raise ValueError('view_role must be eye_context, whole_face, detail, or null (unknown)')
        # Decode before retention; raw source bytes remain unchanged.
        with Image.open(image_path) as image:
            if image.format not in ('PNG', 'JPEG', 'WEBP'):
                raise ValueError('Use PNG, JPEG or WebP inspection images')
            image.verify()
        with Image.open(image_path) as image:
            size, media_type = list(image.size), Image.MIME[image.format]
        asset = self.store.blob(image_path)
        if metadata.get('image_sha256') and metadata['image_sha256'] != asset['sha256']:
            raise ValueError('Image bytes differ from the supplied source hash')
        capture = metadata.get('capture_record')
        if capture:
            self.store.get(capture)  # A linked receipt is retrievable, not inferred authentication.
        return self._append(value, {'images': [dict(asset=asset, metadata=metadata,
            size=size, media_type=media_type, authority='Recorded source assertion; no live/native camera or state authentication')]})

    def target(self, board, expected_revision, image, label, wording, date, source, region,
               target_id=None, supersedes=None):
        value = self._board(board, expected_revision)
        inspection = self._record(value, 'images', image)
        if supersedes:
            old = self._record(value, 'targets', supersedes)
            if old['target_id'] != target_id or self._latest(value, 'targets', target_id) != supersedes:
                raise Conflict('Only the current revision of this region can be revised')
        elif target_id:
            raise ValueError('A new region receives its own ID; revising requires supersedes')
        return self._append(value, {'targets': [dict(target_id=target_id or uuid.uuid4().hex,
            supersedes=supersedes, image=image, label=text(label, 'region label'),
            wording=text(wording, 'exact user wording'), date=dated(date), source=text(source, 'wording source'),
            region=rectangle(region), source_version=inspection['metadata']['source_version'],
            image_sha256=inspection['asset']['sha256'], coordinate_system='normalized x/y/width/height on pinned image')]})

    def _latest(self, value, collection, target_id):
        for key in reversed(value['data'].get(collection, [])):
            row = self.store.get(key, 'visual_' + collection)
            if row.get('target_id') == target_id:
                return key
        return None

    def _current_target(self, value, target):
        row = self._record(value, 'targets', target)
        if self._latest(value, 'targets', row['target_id']) != target:
            raise Conflict('Target revision is superseded; reopen this region')
        return row

    def _plan_payload(self, value, target, interpretation, preserved_features, method, expected_appearance, references, supersedes):
        row = self._current_target(value, target)
        if not isinstance(preserved_features, list) or not preserved_features:
            raise ValueError('Name the features to preserve')
        for feature in preserved_features:
            text(feature, 'preserved feature')
        if not isinstance(references, list):
            raise ValueError('References must be a list of qualified reference-role records')
        for ref in references:
            self._record(value, 'reference_roles', ref)
        latest = self._latest(value, 'plans', row['target_id'])
        if supersedes != latest:
            raise Conflict('Interpretation/proposal revision changed; supply its current supersedes reference')
        return dict(target_id=row['target_id'], target=target, supersedes=supersedes,
                    interpretation=text(interpretation, 'interpretation'), preserved_features=preserved_features,
                    method=text(method, 'operation/method'), expected_appearance=text(expected_appearance, 'expected appearance'), references=references)

    def plan(self, board, expected_revision, target, interpretation, preserved_features, method,
             expected_appearance, references=None, supersedes=None):
        value = self._board(board, expected_revision)
        payload = self._plan_payload(value, target, interpretation, preserved_features, method,
                                     expected_appearance, references or [], supersedes)
        return self._append(value, {'plans': [payload]})

    def reference_role(self, board, expected_revision, image, role, overall_status, rejection_reason,
                       supported, excluded, unknown, qualification):
        value = self._board(board, expected_revision)
        self._record(value, 'images', image)
        if overall_status not in ('accepted', 'rejected', 'unknown'):
            raise ValueError('Overall guide status: accepted, rejected, unknown')
        if overall_status == 'rejected':
            text(rejection_reason, 'original rejection reason')
        for field, entry in [('role', role), ('supported portions and reason', supported),
                             ('excluded portions and reason', excluded), ('unknown portions and reason', unknown),
                             ('local qualification and provenance', qualification)]:
            text(entry, field)
        return self._append(value, {'reference_roles': [dict(image=image, role=role, overall_status=overall_status,
            rejection_reason=rejection_reason, supported=supported, excluded=excluded, unknown=unknown,
            qualification=qualification, geometry_adopted=False)]})

    def _scope(self, value, target, plan):
        row = self._current_target(value, target)
        proposal = self._record(value, 'plans', plan)
        if proposal['target'] != target or self._latest(value, 'plans', row['target_id']) != plan:
            raise Conflict('Interpretation/proposal is superseded or belongs to another target revision')
        return row, proposal

    def compare(self, board, expected_revision, target, plan, baseline, trial, baseline_caption,
                trial_caption, changed, unchanged, baseline_region, trial_region, baseline_approval=None,
                context_views=None):
        value = self._board(board, expected_revision)
        row, proposal = self._scope(value, target, plan)
        before = self._record(value, 'images', baseline)
        after = self._record(value, 'images', trial)
        inspection = self._record(value, 'images', row['image'])
        if baseline == trial:
            raise ValueError('An isolated trial requires a distinct image record')
        areas = [rectangle(baseline_region), rectangle(trial_region)]
        mismatched, unknown = view_matching(before, after)
        # The marked inspection defines what these coordinates refer to. Equal
        # baseline/trial descriptors cannot qualify a region from another view.
        for side, image in (('baseline', before), ('trial', after)):
            for field in VIEW_FIELDS:
                source, compared = inspection['metadata'].get(field), image['metadata'].get(field)
                label = 'inspection/' + side + ' ' + field
                if not source or not compared:
                    unknown.append(label)
                elif source != compared:
                    mismatched.append(label)
            if inspection['size'] != image['size']:
                mismatched.append('inspection/' + side + ' image dimensions')
        if areas[0] != areas[1]:
            mismatched.append('region coordinates')
        if areas[0] != row['region'] or areas[1] != row['region']:
            mismatched.append('target annotation coordinates')
        if before['metadata']['source_version'] == after['metadata']['source_version']:
            unknown.append('distinct source/result versions (same declared version)')
        # Broader captures stay inside this exact review, with their own matching
        # verdict. They cannot borrow the target's coordinates from another view.
        views, roles = [], set()
        if context_views is not None and not isinstance(context_views, list):
            raise ValueError('context_views must be a list of role/baseline/trial objects')
        for view in context_views or []:
            if not isinstance(view, dict) or set(view) != {'role', 'baseline', 'trial'}:
                raise ValueError('Context view requires exactly role, baseline and trial')
            role = view['role']
            if role not in ('eye_context', 'whole_face') or role in roles:
                raise ValueError('Use at most one eye_context and one whole_face pair')
            roles.add(role)
            left = self._record(value, 'images', view['baseline'])
            right = self._record(value, 'images', view['trial'])
            if view['baseline'] == view['trial']:
                raise ValueError('A context trial requires a distinct image record')
            different, missing = view_matching(left, right)
            for side, image, primary in (('baseline', left, before), ('trial', right, after)):
                if image['metadata']['source_version'] != primary['metadata']['source_version']:
                    different.append(side + ' source/result version')
                if not image['metadata'].get('pose') or not primary['metadata'].get('pose'):
                    missing.append(side + ' primary pose')
                elif image['metadata']['pose'] != primary['metadata']['pose']:
                    different.append(side + ' primary pose')
                if not image['metadata'].get('view_role'):
                    missing.append(side + ' view extent')
                elif image['metadata']['view_role'] != role:
                    different.append(side + ' view extent')
            views.append(dict(**view, matching=dict(
                status='unmatched' if different else 'unknown' if missing else 'matched declared inputs',
                mismatched=different, unknown=missing,
                basis='Recorded context descriptors and exact baseline/trial versions; no native authentication or remapped target annotation')))
        match = 'unmatched' if mismatched else 'unknown' if unknown else 'matched declared inputs'
        return self._append(value, {'comparisons': [dict(target_id=row['target_id'], target=target, plan=plan,
            inspection=row['image'], source_version=row['source_version'], baseline=baseline, trial=trial,
            result_version=after['metadata']['source_version'], baseline_region=areas[0], trial_region=areas[1],
            baseline_caption=text(baseline_caption, 'baseline caption'), trial_caption=text(trial_caption, 'trial caption'),
            changed=text(changed, 'changed areas'), unchanged=text(unchanged, 'unchanged areas'),
            baseline_approval=user_fact(baseline_approval) if baseline_approval is not None else None,
            context_views=views,
            matching=dict(status=match, mismatched=mismatched, unknown=unknown,
                basis='Exact comparison of recorded descriptors and normalized annotations; caller source assertions, not native authentication'),
            panel_order=['baseline', 'trial'])]})

    def agreement(self, board, expected_revision, target, plan, fact):
        value = self._board(board, expected_revision)
        row, _ = self._scope(value, target, plan)
        return self._append(value, {'agreements': [dict(target_id=row['target_id'], target=target, plan=plan,
            fact=user_fact(fact), scope='Target and interpretation agreement only; no result approval')]})

    def feedback(self, board, expected_revision, comparison, target, plan, inspection, trial,
                 source_version, result_version, wording, date, source, correction=None, historical=False):
        value = self._board(board, expected_revision)
        review = self._record(value, 'comparisons', comparison)
        for field, supplied in [('target', target), ('plan', plan), ('inspection', inspection), ('trial', trial),
                                ('source_version', source_version), ('result_version', result_version)]:
            if review[field] != supplied:
                raise Conflict('Feedback ' + field + ' differs from the displayed comparison; it has not been applied')
        latest = (self._latest(value, 'targets', review['target_id']) == target and
                  self._latest(value, 'plans', review['target_id']) == plan and
                  self._latest(value, 'comparisons', review['target_id']) == comparison)
        if not latest and not historical:
            raise Conflict('This feedback is stale for the current region/interpretation/result. Reopen, or explicitly archive it on the historical comparison.')
        if historical and correction:
            raise ValueError('Historical feedback cannot activate a corrected interpretation')
        payload = dict(target_id=review['target_id'], comparison=comparison, target=target, plan=plan,
            inspection=inspection, trial=trial, source_version=source_version, result_version=result_version,
            wording=text(wording, 'feedback wording'), date=dated(date), source=text(source, 'feedback source'),
            historical=bool(historical), interpretation_correction=bool(correction))
        records = {'feedback': [payload]}
        if correction:
            allowed = {'interpretation', 'preserved_features', 'method', 'expected_appearance', 'references'}
            if not isinstance(correction, dict) or set(correction) != allowed:
                raise ValueError('Correction requires the complete interpretation, preserved_features, method, expected_appearance and references')
            new_plan = self._plan_payload(value, target, **correction, supersedes=plan)
            new_plan['correction_from_comparison'] = comparison
            new_plan['correction_wording'] = wording
            records['plans'] = [new_plan]
        return self._append(value, records)

    def state_fact(self, board, expected_revision, comparison, facet, value, evidence, fact=None):
        board_value = self._board(board, expected_revision)
        review = self._record(board_value, 'comparisons', comparison)
        choices = dict(execution=('proposed', 'running', 'built'), technical_verification=('unknown', 'passed', 'failed'),
                       visually_useful=('unknown', 'yes', 'no'), retained=('unknown', 'yes', 'no'), owner_acceptance=('unknown', 'accepted', 'rejected'))
        if facet not in choices or value not in choices[facet]:
            raise ValueError('Choose an independent modeling state facet/value')
        text(evidence, 'state fact evidence and limits')
        if facet == 'owner_acceptance' and value != 'unknown':
            fact = user_fact(fact)
        elif fact is not None:
            raise ValueError('User fact is only for explicit owner acceptance/rejection')
        return self._append(board_value, {'state_facts': [dict(comparison=comparison, target=review['target'],
            plan=review['plan'], trial=review['trial'], result_version=review['result_version'],
            facet=facet, value=value, evidence=evidence, fact=fact,
            authority='Reported historical fact; recording does not execute, retain or approve native work')]})

    def presentation(self, board, expected_revision, collection, record, archived, reason):
        value = self._board(board, expected_revision)
        if collection == 'board':
            if record != board:
                raise ValueError('Board archive must name this exact board')
        elif collection == 'images':
            self._record(value, collection, record)
        else:
            raise ValueError('Presentation archive supports boards and images')
        if not isinstance(archived, bool):
            raise ValueError('archived must be a boolean')
        return self._append(value, {'presentation_events': [dict(collection=collection,
            item=record, archived=archived, reason=text(reason, 'presentation reason'),
            effect='Presentation only; all source bytes, targets, references and history preserved')]})

    def _presentation(self, value):
        visibility = {'board_archived': False, 'archived_images': []}
        images = set()
        for key in value['data'].get('presentation_events', []):
            event = self.store.get(key, 'visual_presentation_events')
            if event['collection'] == 'board':
                visibility['board_archived'] = event['archived']
            elif event['archived']:
                images.add(event['item'])
            else:
                images.discard(event['item'])
        visibility['archived_images'] = sorted(images)
        return visibility

    def read(self, board=None, limit=50):
        if board is None:
            return dict(boards=[dict(board=v['handle'], revision=v['revision'], title=v['intent']['title'], updated=v['utc'])
                               | {'archived': self._presentation(v)['board_archived']}
                               for v in self.ledger.list(KIND, limit=limit)])
        value = self._board(board)
        rows = {name: [dict(self.store.get(k, 'visual_' + name), record=k) for k in value['data'].get(name, [])]
                for name in COLLECTIONS}
        current = {}
        for target in rows['targets']:
            current[target['target_id']] = dict(target=target['record'], plan=None, comparison=None, agreement=None)
        for plan in rows['plans']:
            active = current.get(plan['target_id'])
            if active and plan['target'] == active['target']:
                active.update(plan=plan['record'], comparison=None, agreement=None)
        for comparison in rows['comparisons']:
            active = current[comparison['target_id']]
            if comparison['target'] == active['target'] and comparison['plan'] == active['plan']:
                active['comparison'] = comparison['record']
        for agreement in rows['agreements']:
            active = current[agreement['target_id']]
            if agreement['target'] == active['target'] and agreement['plan'] == active['plan']:
                active['agreement'] = agreement['record']
        for feedback in rows['feedback']:
            active = current[feedback['target_id']]
            feedback['applicability'] = ('current exact comparison' if not feedback['historical'] and
                feedback['comparison'] == active['comparison'] and feedback['plan'] == active['plan'] and
                feedback['target'] == active['target'] else 'historical only; never inherited by newer results')
        return dict(board=board, revision=value['revision'], title=value['intent']['title'], updated=value['utc'],
                    current=current, presentation=self._presentation(value), **rows,
                    native_effects=False, mandatory_approval_gate=False)

    def summary(self, board):
        view = self.read(board)
        by_id = {row['record']: row for name in COLLECTIONS for row in view[name]}
        lines = ['# ' + view['title'], '', 'Historical visual feedback. No live/native authority or mandatory approval gate.',
                 '', 'Board: ' + board, 'Revision: ' + view['revision'], '']
        for active in view['current'].values():
            target = by_id[active['target']]
            lines += ['## ' + target['label'], '', 'Exact target (' + target['date'] + '):', target['wording'],
                      'Source: ' + target['source'], 'Target revision: ' + target['record'],
                      'Inspection: ' + target['image'] + ' / version ' + target['source_version'],
                      'Region: ' + str(target['region']), '']
            if active['plan']:
                plan = by_id[active['plan']]
                lines += ['Interpretation: ' + plan['interpretation'], 'Preserve: ' + '; '.join(plan['preserved_features']),
                          'Method: ' + plan['method'], 'Expected appearance: ' + plan['expected_appearance'],
                          'Interpretation revision: ' + plan['record'], '']
            lines += ['Explicit target agreement: ' + ('recorded at ' + by_id[active['agreement']]['fact']['date'] if active['agreement'] else 'not recorded for this revision'), '']
            if active['comparison']:
                comparison = by_id[active['comparison']]
                lines += ['Comparison: ' + comparison['matching']['status'], 'LEFT: ' + comparison['baseline_caption'],
                          'RIGHT: ' + comparison['trial_caption'], 'Changed: ' + comparison['changed'],
                          'Unchanged: ' + comparison['unchanged'], 'Result version: ' + comparison['result_version'], '']
                for context in comparison.get('context_views', []):
                    lines += ['Context ' + context['role'] + ': ' + context['matching']['status'],
                              'LEFT image: ' + context['baseline'], 'RIGHT image: ' + context['trial'], '']
        lines += ['## Retrievable history', '']
        for name in ('targets', 'plans', 'comparisons', 'feedback', 'agreements', 'reference_roles', 'state_facts', 'presentation_events'):
            for row in view[name]:
                lines += [name + ': ' + row['record'], canonical(row).decode('utf-8'), '']
        return '\n'.join(lines)

    def export(self, board, output):
        path = Path(output)
        if path.exists():
            raise FileExistsError('Preserve the existing target summary; choose a new output')
        atomic_write(path, self.summary(board).encode('utf-8'))
        return dict(board=board, output=str(path.resolve()), native_effects=False)
