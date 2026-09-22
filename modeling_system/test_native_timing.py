import unittest
from .native_timing import NativeTimings


class NativeTimingTests(unittest.TestCase):
    def test_nested_calls_do_not_double_count_children(self):
        now = [0.]
        timer = NativeTimings(clock=lambda: now[0])
        with timer.measure('dispatch'):
            now[0] = 1.
            with timer.measure('fresh_observation'):
                now[0] = 3.
            now[0] = 5.
        report = timer.snapshot()
        self.assertEqual(report['total_ms'], 5000.)
        self.assertEqual([r['elapsed_ms'] for r in report['calls']], [5000., 2000.])
        self.assertEqual([r['exclusive_ms'] for r in report['calls']], [3000., 2000.])
        self.assertEqual([r['parent'] for r in report['calls']], [None, 0])

    def test_failed_native_call_is_retained_and_never_swallowed(self):
        updates = []
        timer = NativeTimings(sink=updates.append)
        with self.assertRaisesRegex(RuntimeError, 'native failed'):
            with timer.measure('save'):
                raise RuntimeError('native failed')
        self.assertEqual(updates[0]['calls'][0]['status'], 'running')
        self.assertEqual(updates[-1]['calls'][0]['status'], 'failed')
        self.assertFalse(timer.stack)

    def test_bounded_telemetry_does_not_interrupt_later_effects(self):
        timer = NativeTimings()
        effects = []
        for i in range(257):
            with timer.measure('step'): effects.append(i)
        self.assertEqual(len(effects), 257)
        self.assertEqual(len(timer.snapshot()['calls']), 256)
        self.assertEqual(timer.snapshot()['dropped_calls'], 1)

    def test_failed_telemetry_does_not_turn_a_known_result_into_unknown_effects(self):
        def unavailable(report): raise OSError('unavailable')
        timer = NativeTimings(sink=unavailable)
        effects = []
        with timer.measure('save'): effects.append('saved once')
        self.assertEqual(effects, ['saved once'])
        self.assertEqual(timer.snapshot()['calls'][0]['status'], 'returned')
        self.assertEqual(timer.snapshot()['sink_errors'], ['OSError', 'OSError'])


if __name__ == '__main__':
    unittest.main()
