"""Nested native-call timing without imports, dispatch, or outcome changes.

Adapters can load this file independently inside Blender. Inclusive durations
overlap; exclusive durations subtract measured children only. Neither proves
what uninstrumented code did or establishes native success.
"""
from contextlib import contextmanager
from copy import deepcopy
import re
import time


class NativeTimings:
    def __init__(self, *, clock=time.perf_counter, sink=None):
        self.clock, self.sink = clock, sink
        self.started = clock()
        self.rows, self.stack = [], []
        self.sink_errors = []
        self.dropped_calls = 0

    def snapshot(self):
        return {'schema_version': 1, 'total_ms': round((self.clock() - self.started) * 1000, 3),
                'calls': deepcopy(self.rows), 'sink_errors': list(self.sink_errors), 'dropped_calls': self.dropped_calls,
                'limits': 'Nested inclusive times overlap; exclusive times exclude measured children only. Timing is not native outcome evidence.'}

    def _emit(self):
        if self.sink is not None:
            try:
                self.sink(self.snapshot())
            except Exception as error:
                # A telemetry write must not relabel or repeat a known native effect.
                if len(self.sink_errors) < 8:
                    self.sink_errors.append(type(error).__name__)

    @contextmanager
    def measure(self, name):
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name):
            raise ValueError('A named native timing phase is required')
        if len(self.rows) >= 256:
            self.dropped_calls += 1
            yield
            return
        row = {'name': name, 'parent': self.stack[-1][0] if self.stack else None,
               'status': 'running', 'elapsed_ms': None, 'exclusive_ms': None}
        index = len(self.rows)
        self.rows.append(row)
        frame = [index, 0.]
        self.stack.append(frame)
        self._emit()
        started = self.clock()
        try:
            yield
        except BaseException:
            row['status'] = 'failed'
            raise
        else:
            row['status'] = 'returned'
        finally:
            duration = self.clock() - started
            row['elapsed_ms'] = round(duration * 1000, 3)
            row['exclusive_ms'] = round(max(0., duration - frame[1]) * 1000, 3)
            self.stack.pop()
            if self.stack:
                self.stack[-1][1] += duration
            self._emit()
