import json
import unittest
from .decision_menu import factor_choices, resolve_choice


def option(identity, family, description):
    return dict(id=identity, family=family, family_description=description,
                target_description='Local measured region ' + identity[-1])


class MenuTests(unittest.TestCase):
    def test_selected_family_only_and_private_ids_not_sent(self):
        menu = factor_choices([option('private-a', 'private-inspect', 'Inspect'),
                               option('private-b', 'private-inspect', 'Inspect'),
                               option('private-c', 'private-repair', 'Repair'),
                               option('private-d', 'private-repair', 'Repair')], 'Choose next')
        self.assertEqual(len(menu['questions']), 3)
        self.assertNotIn('private', json.dumps(menu['questions']))
        self.assertEqual(resolve_choice(menu['mapping'], {'operation': 'f1', 'target_f1': 't0',
                                                         'target_f0': 'unused-invalid'}), 'private-c')

    def test_single_family_avoids_operation_question(self):
        menu = factor_choices([option('a', 'inspect', 'Inspect'), option('b', 'inspect', 'Inspect')], '')
        self.assertEqual(set(menu['questions']), {'target_f0'})
        self.assertEqual(resolve_choice(menu['mapping'], {'target_f0': 't1'}), 'b')

    def test_fixed_branches_need_no_target_questions(self):
        menu = factor_choices([option('a', 'inspect', 'Inspect'), option('b', 'repair', 'Repair')], '')
        self.assertEqual(set(menu['questions']), {'operation'})
        self.assertEqual(resolve_choice(menu['mapping'], {'operation': 'f1'}), 'b')
        single = factor_choices([option('a', 'inspect', 'Inspect')], '')
        self.assertEqual(single['questions'], {})
        self.assertEqual(resolve_choice(single['mapping'], {}), 'a')

    def test_wrong_target_and_inconsistent_family_are_rejected(self):
        menu = factor_choices([option('a', 'inspect', 'Inspect'), option('b', 'inspect', 'Inspect')], '')
        with self.assertRaises(KeyError):
            resolve_choice(menu['mapping'], {'target_f0': 'foreign'})
        with self.assertRaises(ValueError):
            factor_choices([option('a', 'inspect', 'Inspect'), option('b', 'inspect', 'Different')], '')
