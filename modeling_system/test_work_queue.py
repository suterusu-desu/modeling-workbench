from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from .controller import read_json
from .work_queue import WorkQueue


def work(key, lane='diagnosis', requires=None):
    return {'id': key, 'revision': 'v1', 'lane': lane, 'description': 'Useful ' + key,
        'handler': 'run', 'payload': {}, 'reads': {'source': 'v1'}, 'writes': [],
        'requires': requires or {}, 'completion_condition': 'Receipt records completed ' + key}


class WorkQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = {'owner': 'owner', 'authority_revision': 'scope1', 'stage': 'initial',
            'values': {'source': 'v1'}, 'active_operations': [], 'public_state': {'goal': 'Repair observed defect'}}
        self.items = [work('inspect'), work('fit', 'repair', {'inspect': ['completed']}),
                      work('capture', 'review', {'fit': ['completed']})]
        self.ran = []
        self.outcome = {}

    def queue(self, select=None, run=None):
        def execute(item, context):
            self.ran.append(item['id'])
            return {'status': self.outcome.get(item['id'], 'completed'), 'evidence': ['fixture']}
        return WorkQueue(self.root/'queue', owner='owner', goal={'objective': 'Repair defect'},
            observe_context=lambda: deepcopy(self.state), catalog=lambda state, results: deepcopy(self.items),
            handlers={'run': run or execute}, select=select or (lambda state, actions, plan: actions[0]['id']))

    def test_three_dependent_operations_complete_without_restarting_and_do_not_replay(self):
        result = self.queue().run(max_steps=8)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(self.ran, ['inspect', 'fit', 'capture'])
        self.queue().run(max_steps=8)
        self.assertEqual(len(self.ran), 3)

    def test_failed_diagnosis_routes_to_recovery_without_retry(self):
        self.outcome['inspect'] = 'failed'
        self.items.append(work('recover', 'recovery', {'inspect': ['failed']}))
        self.queue().run(max_steps=8)
        self.assertEqual(self.ran, ['inspect', 'recover'])

    def test_immutable_outputs_do_not_redefine_two_independent_preparations(self):
        self.items = [work('first', 'preparation'), work('second', 'preparation')]
        self.state['values']['output_directory'] = 'empty'
        def prepare(item, context):
            self.ran.append(item['id'])
            artifact = self.root/(item['id'] + '.json')
            with artifact.open('x') as handle: handle.write('{"prepared": true}')
            self.state['values']['output_directory'] = 'changed-' + item['id']
            return {'status': 'completed', 'evidence': [str(artifact)]}
        queue = self.queue(run=prepare)
        self.assertEqual(queue.run(max_steps=3)['status'], 'completed')
        self.assertEqual(self.ran, ['first', 'second'])
        self.assertEqual(queue.run(max_steps=3)['status'], 'completed')
        self.assertEqual(self.ran, ['first', 'second'])

    def test_stale_source_blocks_all_dependent_work(self):
        self.state['values']['source'] = 'changed'
        result = self.queue().run(max_steps=8)
        self.assertEqual(result['status'], 'idle')
        self.assertEqual(self.ran, [])

    def test_busy_native_lane_never_executes(self):
        self.state['active_operations'] = [{'id': 'other', 'writes': ['source']}]
        self.assertEqual(self.queue().run(max_steps=8)['status'], 'waiting_owner')
        self.assertEqual(self.ran, [])

    def test_completed_prerequisite_does_not_hide_changed_guide(self):
        self.state['values']['guide'] = 'guide1'
        self.items[0]['reads']['guide'] = 'guide1'
        queue = self.queue()
        queue.run(max_steps=1)
        self.assertEqual(self.ran, ['inspect'])
        self.state['values']['guide'] = 'guide2'
        self.assertEqual(queue.run(max_steps=8)['status'], 'idle')
        self.assertEqual(self.ran, ['inspect'])

    def test_interruption_cannot_be_replayed_by_changing_definition(self):
        def crash(item, context):
            self.ran.append(item['id'])
            raise RuntimeError('Lost worker')
        self.assertEqual(self.queue(run=crash).run(max_steps=8)['status'], 'needs_reconciliation')
        self.items[0]['revision'] = 'v2'
        self.queue().run(max_steps=8)
        self.assertEqual(self.ran, ['inspect'])

    def test_real_deferral_is_not_an_uncertain_provider_failure(self):
        self.items = [work('first'), work('second', 'review')]
        result = self.queue(select=lambda *args: {'status': 'needs_review', 'reason': 'Need visual interpretation'}).run(max_steps=8)
        self.assertEqual(result['status'], 'needs_review')
        record = read_json(self.root/'queue/controller/controller.json')
        self.assertIsNone(record['selection'])
        self.assertEqual(self.ran, [])

    def test_changed_operation_between_selection_and_execution_is_not_run(self):
        self.items = [work('first'), work('second')]
        def select(state, actions, plan):
            self.items[0]['payload']['scope'] = 'different'
            return 'first'
        self.assertEqual(self.queue(select=select).run(max_steps=8)['status'], 'invalidated')
        self.assertEqual(self.ran, [])

    def test_cycles_are_explicit_errors(self):
        self.items[0]['requires'] = {'capture': ['completed']}
        self.assertEqual(self.queue().run(max_steps=8)['status'], 'error')
        self.assertEqual(self.ran, [])
