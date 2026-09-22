import unittest
from .blender_controller_status import status_lines


class StatusTests(unittest.TestCase):
    def test_stale_activity_is_not_shown_as_running(self):
        lines = status_lines({'status': 'running', 'updated_at': 10, 'action': 'measure'}, now=50)
        self.assertIn('disconnected / stale', lines[0])
        self.assertIn('Action: measure', lines)
        self.assertIn('Work limits: unknown', lines)

    def test_actual_budget_and_terminal_result_remain_visible(self):
        lines = status_lines({'status': 'completed', 'updated_at': 10,
                              'work_limits': {'max_steps': 0}}, now=50)
        self.assertEqual(lines[0], 'Workbench: completed')
        self.assertIn('"max_steps": 0', lines[-1])
