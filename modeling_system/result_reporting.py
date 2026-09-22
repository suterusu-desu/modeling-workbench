"""Strict numerical JSON and compact public reports with complete private detail."""
from collections.abc import Mapping
import json
import math
from pathlib import Path
import numpy as np


def json_data(value):
    """Convert numerical values without stringifying errors or nonfinite numbers."""
    if isinstance(value, np.ndarray):
        return json_data(value.tolist())
    if isinstance(value, np.generic):
        return json_data(value.item())
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite values need an explicit missing-data representation')
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise TypeError('JSON object keys must be strings')
        return {k: json_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_data(v) for v in value]
    raise TypeError('Unsupported evidence value: ' + type(value).__name__)


def compact_summary(operation, metrics, *, limits, max_bytes=1500):
    """Retain complete selected metrics; never cut a sentence or a limitation."""
    result = {'operation': operation, 'result': {}, 'limits': limits,
              'omitted_metric_count': len(metrics), 'detail': 'Complete result retained in the linked preparation evidence.'}
    def size():
        return len(json.dumps(result, allow_nan=False).encode('utf-8'))
    if size() > max_bytes:
        raise ValueError('Operation and complete limitations exceed the public summary budget')
    for key, value in json_data(metrics).items():
        result['result'][key] = value
        result['omitted_metric_count'] -= 1
        if size() > max_bytes:
            del result['result'][key]
            result['omitted_metric_count'] += 1
    return json.dumps(result, allow_nan=False)
