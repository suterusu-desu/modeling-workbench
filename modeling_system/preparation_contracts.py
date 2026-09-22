"""Qualified pure-array operations and actionable preflight failures."""
from dataclasses import dataclass, field
import importlib.util
import inspect
import re
import numpy as np


def check_arrays(arguments, contracts):
    for name, rule in contracts.items():
        if name not in arguments:
            raise ValueError('Missing contracted array: ' + name)
        value = np.asarray(arguments[name])
        shape = rule.get('shape')
        if shape is not None and (value.ndim != len(shape) or any(
                expected is not None and (type(expected) is not int or expected < 0 or actual != expected)
                for actual, expected in zip(value.shape, shape))):
            raise ValueError(f'{name}: expected shape {shape}, received {list(value.shape)}')
        if value.dtype.kind not in rule.get('kinds', 'biuf'):
            raise ValueError(name + ': unsupported numerical dtype')
        if rule.get('nonempty', True) and not value.size:
            raise ValueError(name + ': empty array has no qualified support')
        if rule.get('finite', True) and not np.isfinite(value).all():
            raise ValueError(name + ': nonfinite array values')


@dataclass(frozen=True)
class PreparationOperation:
    """A workspace registers qualified pure math, not a script or native executor.

    The function receives arrays and declared parameters, returns arrays/JSON,
    and does not perform external effects. The existing handler owns persistence.
    """
    function: object
    arrays: dict = field(default_factory=dict)
    requires: tuple = ()

    def bind(self, arguments):
        if not callable(self.function):
            raise ValueError('A qualified preparation callable is required')
        try:
            inspect.signature(self.function).bind(**arguments)
        except TypeError as error:
            raise ValueError('Preparation argument contract: ' + str(error)) from error
        for module in self.requires:
            if not isinstance(module, str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', module):
                raise ValueError('Declared importable dependency names required')
            try:
                available = importlib.util.find_spec(module) is not None
            except (ImportError, ValueError, AttributeError):
                available = False
            if not available:
                raise ValueError('Preparation dependency unavailable: ' + module)

    def __call__(self, **arguments):
        self.bind(arguments)
        check_arrays(arguments, self.arrays)
        return self.function(**arguments)
