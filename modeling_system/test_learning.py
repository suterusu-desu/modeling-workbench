"""Learning reaches later choices without native replay or private source leakage."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from . import test_operating_session as fixtures
from .learning import record_lesson, import_review_history, experience_rows, retain_review
from .retained_context import RetainedContext


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.base = fixtures.OperatingTests(); self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.service = self.base.service
        self.evidence = [{'kind': 'file', 'path': str(self.base.base.image), 'role': 'Actual comparison'}]

    def lesson(self, **extra):
        return {'mechanism': 'Shared material trajectory',
                'observation': 'The same timing failed before whole-sheet guide fitting and helped after it.',
                'next_use': 'Check the opposing surface fit before repeating or rejecting the timing method.',
                'limits': 'One local gain; neither nearest distance nor this case proves anatomical attachment.',
                'conditions': {'opposing_fit': 'whole_surface'}, **extra}

    def reviewed(self, *, disposition='rejected', reason='Material timing left a visible ledge.', lesson=None):
        self.base.items = [self.base.item('material_candidate')]
        session = self.base.session(); session.run(max_steps=1)
        review = session.record_review('material_candidate', expected_basis=session.review_basis('material_candidate'),
            judgment={'disposition': disposition, 'scope': 'upper junction', 'reason': reason,
                      'next_question': 'Resolve material correspondence against the guide.'},
            evidence=self.evidence, lesson=lesson)
        return session, review

    def test_review_flows_into_new_scope_without_history_projector(self):
        old, review = self.reviewed()
        self.assertEqual(len(experience_rows(self.service.store)), 1)
        # A new session directory, same private workspace. No copied history file.
        self.base.root = self.base.root / 'next_scope'
        self.base.items = [self.base.item('material_repair'), self.base.item('material_diagnosis')]
        session = self.base.session()
        captured = []
        def choose(state, actions, plan):
            captured.append(deepcopy(state['observations']['experience']))
            return actions[0]['id']
        session.select = choose
        session.run(max_steps=1)
        self.assertEqual(len(captured), 1)
        payload = json.dumps(captured)
        self.assertIn('visible ledge', payload)
        for private in (str(self.base.base.image), str(Path(self.base.base.image).resolve()),
                        review['operation_handle'], review['operation_fact']):
            self.assertNotIn(json.dumps(private)[1:-1], payload)   # JSON-escaped form, meaningful on Windows too
        self.assertTrue(captured[0]['passages'])

    def test_retrieval_leads_with_the_public_judgment_and_lesson(self):
        _, review = self.reviewed(lesson=self.lesson())
        found = self.service.retrieve_experience('shared material trajectory timing opposing surface', limit=3)
        match = next(row for row in found['matches'] if row.get('record_kind') == 'modeling_experience')
        keys = list(match)
        self.assertLess(keys.index('public'), keys.index('excerpt'))
        self.assertEqual(match['public']['lesson']['mechanism'], 'Shared material trajectory')
        self.assertIn('visible ledge', match['public']['review']['reason'])
        self.assertEqual(match['public']['condition_comparison']['status'], 'conditional')
        self.assertEqual(match['evidence_count'], len(match['excerpt']['evidence']))
        # The public projection carries no private locators; the full record still does. Use the
        # stored locator (the store resolves paths, e.g. Windows short names) in JSON-escaped form.
        locator = json.dumps(match['excerpt']['evidence'][0]['path'])[1:-1]
        self.assertIn(locator, json.dumps(match['excerpt']))
        self.assertNotIn(locator, json.dumps(match['public']))

    def test_conditional_lesson_keeps_failure_and_changed_prerequisite(self):
        record_lesson(self.service, lesson=self.lesson(), evidence=self.evidence)
        context = RetainedContext(self.service, query='material timing opposing surface', sources=[])
        missing = context({'public_state': {}}, [], {})
        different = context({'public_state': {'opposing_fit': 'private-different-value'}}, [], {})
        known = context({'public_state': {'opposing_fit': 'whole_surface'}}, [], {})
        content = lambda row: row['public']['passages'][0]['content']
        self.assertEqual(content(missing)['condition_comparison']['status'], 'conditional')
        self.assertEqual(content(different)['condition_comparison']['different'], ['opposing_fit'])
        self.assertNotIn('private-different-value', json.dumps(different['public']))
        self.assertEqual(content(known)['condition_comparison']['status'], 'applicable')
        self.assertIn('failed before', content(known)['lesson']['observation'])
        self.assertNotEqual(missing['revision'], known['revision'])

    def test_projection_happens_before_limit_even_after_many_private_matches(self):
        for i in range(30):
            self.service.store.put('outcome', {'private': f'{i} material timing guide surface ledge correspondence secret locator'})
        path = self.base.root/'public-history.md'
        path.write_text('Material timing needs a qualified opposing surface.', encoding='utf-8')
        context = RetainedContext(self.service, query='material timing guide surface ledge correspondence', limit=1,
            sources=[{'path': str(path), 'role': 'experience'}],
            project=lambda row: row['excerpt'] if row['source'] == str(path.resolve()) else None)
        result = context({}, [], {})
        self.assertIn('qualified opposing', result['public']['passages'][0]['content'])
        self.assertEqual(result['public']['coverage']['excluded_by_projection'], 30)
        self.assertNotIn('secret locator', json.dumps(result['public']))

    def test_existing_source_projector_also_receives_new_durable_lessons(self):
        record_lesson(self.service, lesson=self.lesson(), evidence=self.evidence)
        result = RetainedContext(self.service, query='material timing', sources=[], project=lambda row: None)({}, [], {})
        self.assertEqual(len(result['public']['passages']), 1)

    def test_corrected_review_replaces_only_same_basis_and_preserves_records(self):
        session, original = self.reviewed()
        corrected = session.record_review('material_candidate', expected_basis=session.review_basis('material_candidate'),
            judgment={'disposition': 'unresolved', 'scope': 'upper junction',
                      'reason': 'The comparison angle was insufficient.', 'next_question': 'Use matched evidence.'},
            evidence=self.evidence)
        rows = experience_rows(self.service.store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1]['public']['review']['disposition'], 'unresolved')
        self.assertEqual(len(list(self.service.store.records(('modeling_experience',)))), 2)
        self.assertEqual(self.service.store.get(original['operation_fact'])['outcome']['result']['judgment']['disposition'], 'rejected')

    def test_import_is_idempotent_and_repairs_index_without_touching_old_receipts(self):
        with patch('modeling_system.learning.retain_review', return_value='interrupted-index'):
            session, review = self.reviewed()
        self.assertEqual(experience_rows(self.service.store), [])
        def snapshot():
            return {str(p): p.read_bytes() for p in session.directory.rglob('*') if p.is_file()}
        before = snapshot()
        first = import_review_history(self.service, [session.directory])
        self.assertEqual(first, import_review_history(self.service, [session.directory]))
        self.assertEqual(before, snapshot())
        self.assertEqual(len(first['records']), 1)
        self.assertEqual(first['native_calls'] + first['provider_calls'], 0)
        forged = deepcopy(review); forged['judgment']['reason'] = 'Not in the receipt'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            retain_review(self.service, session.directory, forged)
        forged = deepcopy(review); forged['operation_handle'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            retain_review(self.service, session.directory, forged)

    def test_conditional_review_and_evidence_are_retained_together(self):
        session, review = self.reviewed(lesson=self.lesson())
        row = experience_rows(self.service.store)[0][1]
        self.assertEqual(row['public']['lesson'], self.lesson())
        self.assertEqual(row['source']['review_fact'], review['operation_fact'])
        source = Path(self.evidence[0]['path']); source.write_bytes(b'later edit')
        self.assertNotEqual(source.read_bytes(), self.service.store.resolve_blob(row['evidence'][0]['asset']).read_bytes())

    def test_bounded_context_reports_overflow_and_never_truncates_a_caveat(self):
        for i in range(22):
            record_lesson(self.service, lesson=self.lesson(observation=f'Material {i}. ' + 'Detail. '*75), evidence=self.evidence)
        from .work_limits import WorkLimits
        result = RetainedContext(self.service, query='material', sources=[], budget=WorkLimits(
            context_passages=4, context_bytes=2400, retrieval_candidates=20))({}, [], {})
        public = result['public']
        self.assertLessEqual(len(json.dumps(public['passages'], ensure_ascii=True).encode()), 2400)
        self.assertGreater(public['coverage']['unreturned_eligible_passages'], 0)
        self.assertGreater(public['coverage']['omitted_public_passages'], 0)
        self.assertTrue(all('proves anatomical attachment' in row['content']['lesson']['limits'] for row in public['passages']))

    def test_private_evidence_does_not_supply_lexical_relevance(self):
        record_lesson(self.service, lesson=self.lesson(), evidence=self.evidence)
        query = Path(self.evidence[0]['path']).stem
        result = RetainedContext(self.service, query=query, sources=[])({}, [], {})
        self.assertEqual(result['public']['passages'], [])

    def test_bad_lesson_refused_before_review_or_record(self):
        with self.assertRaises(ValueError):
            record_lesson(self.service, lesson=self.lesson(), evidence=[])
        with self.assertRaises(ValueError):
            record_lesson(self.service, lesson=self.lesson(conditions={'opaque': {'nested': 'private'}}), evidence=self.evidence)
        with self.assertRaises(ValueError):
            record_lesson(self.service, lesson=self.lesson(limits='x'*2000), evidence=self.evidence)
        self.assertEqual(experience_rows(self.service.store), [])


if __name__ == '__main__': unittest.main()
