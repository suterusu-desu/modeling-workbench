from copy import deepcopy
import json
import unittest
from .judgments import prepare_judgments, planning_choice, resolve_judgments, validate_answers


class JudgmentTests(unittest.TestCase):
    def setUp(self):
        self.binding = {'owner': 'owner', 'authority_revision': 'scope1',
                        'dependencies': {'reads': {'private-mesh': 'v1'}, 'writes': []}}
        self.decisions = [planning_choice('method', [
            {'id': 'private-method', 'description': 'Repair the recorded correspondence discontinuity'}]),
            {'id': 'private-relevance', 'type': 'noul', 'instructions': 'Does this retained lesson concern the observed failure?'},
            {'id': 'private-impact', 'type': 'score', 'instructions': 'How does the recorded defect affect function?',
             'criteria': ['Cosmetic only', 'Impaired motion', 'Motion impossible']}]
        self.packet = prepare_judgments({'failure': 'recorded discontinuity'}, self.decisions, self.binding)
        self.answers = {
            'q0': {'type': 'choice', 'choice': 'o0', 'probabilities': {'o0': .8, 'o1': .2}, 'confidence': .6},
            'q1': {'type': 'noul', 'noul': .7},
            'q2': {'type': 'score', 'score': 1.25, 'probabilities': {'0': 0., '1': .75, '2': .25},
                   'confidence': .5, 'legend': {'0': 'Cosmetic only', '1': 'Impaired motion', '2': 'Motion impossible'}}}

    def test_mixed_batch_keeps_private_ids_local_and_full_probabilities(self):
        wire = json.dumps({k: self.packet[k] for k in ('state', 'questions')})
        self.assertNotIn('private', wire)
        result = resolve_judgments(self.packet, self.answers, self.binding)
        self.assertEqual(result['judgments']['method']['choice'], 'private-method')
        self.assertEqual(result['judgments']['private-impact']['score'], 1.25)
        self.assertEqual(result['judgments']['private-relevance']['noul'], .7)
        self.assertFalse(result['native_dispatch'])
        self.assertFalse(result['appearance_accepted'])

    def test_abstention_is_an_actual_outcome(self):
        a = deepcopy(self.answers)
        a['q0'].update(choice='o1', probabilities={'o0': .1, 'o1': .9})
        self.assertEqual(resolve_judgments(self.packet, a, self.binding)['judgments']['method']['choice'], 'needs_astra')

    def test_changed_authority_or_dependency_refuses_result(self):
        for field in ('authority', 'mesh'):
            b = deepcopy(self.binding)
            if field == 'authority': b['authority_revision'] = 'scope2'
            else: b['dependencies']['reads']['private-mesh'] = 'v2'
            with self.assertRaises(ValueError): resolve_judgments(self.packet, self.answers, b)

    def test_invalid_distribution_and_numeric_values_fail_closed(self):
        mutations = [lambda a: a['q0'].update(choice='absent'),
                     lambda a: a['q0'].update(confidence=True),
                     lambda a: a['q1'].update(noul=float('nan')),
                     lambda a: a['q2'].update(score=1.8),
                     lambda a: a['q2']['legend'].update({'1': 'changed rubric'}),
                     lambda a: a.pop('q1')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                a = deepcopy(self.answers); mutate(a)
                with self.assertRaises(ValueError): validate_answers(self.packet['questions'], a)

    def test_structured_questions_and_limits(self):
        specs = deepcopy(self.decisions)
        specs[0]['instructions'] = {'question': 'Which method applies?', 'boundary': ['No new geometry authority']}
        specs[0]['options'][0]['description'] = {'mechanism': 'correspondence repair', 'excludes': 'unknown topology'}
        self.assertIsInstance(prepare_judgments({}, specs, self.binding)['questions']['q0']['instructions'], dict)
        with self.assertRaises(ValueError): prepare_judgments({}, specs * 5, self.binding)
        with self.assertRaises(ValueError): prepare_judgments({'irrelevant': 'x' * 32000}, specs, self.binding)

    def test_cent_serialized_choice_preserves_raw_answer_and_selection(self):
        questions = {'route': {'type': 'choice', 'instructions': 'Choose a route',
                     'criteria': {k: k for k in ('a', 'b', 'c', 'd')}}}
        for probabilities in ({'a': .62, 'b': .21, 'c': .12, 'd': .04},
                              {'a': .62, 'b': .21, 'c': .12, 'd': .06}):
            answers = {'route': {'type': 'choice', 'choice': 'a',
                       'confidence': .49, 'probabilities': probabilities}}
            original = deepcopy(answers)
            self.assertEqual(validate_answers(questions, answers), {'route': 'a'})
            self.assertEqual(answers, original)

    def test_cent_compatibility_rejects_other_distribution_defects(self):
        questions = {'route': {'type': 'choice', 'instructions': 'Choose a route',
                     'criteria': {k: k for k in ('a', 'b', 'c', 'd')}}}
        original = {'route': {'type': 'choice', 'choice': 'a', 'confidence': .49,
                             'probabilities': {'a': .62, 'b': .21, 'c': .12, 'd': .04}}}
        mutations = [lambda a: a['route']['probabilities'].update(d=.03),
                     lambda a: a['route']['probabilities'].update(d=.041),
                     lambda a: a['route']['probabilities'].pop('d'),
                     lambda a: a['route']['probabilities'].update(e=0.),
                     lambda a: a['route']['probabilities'].update(d=float('nan')),
                     lambda a: a['route']['probabilities'].update(d=True),
                     lambda a: a['route']['probabilities'].update(d=-.01),
                     lambda a: a['route'].update(choice='b'),
                     lambda a: a['route'].update(confidence=1.01)]
        for mutate in mutations:
            answers = deepcopy(original); mutate(answers)
            with self.subTest(answers=answers), self.assertRaises(ValueError):
                validate_answers(questions, answers)

    def test_cent_compatibility_does_not_change_score_contract(self):
        answers = deepcopy(self.answers)
        answers['q2']['probabilities']['0'] = .01
        with self.assertRaises(ValueError): validate_answers(self.packet['questions'], answers)


if __name__ == '__main__':
    unittest.main()
