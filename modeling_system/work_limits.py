"""Bounds for local catalogs and retained context; no inference or spending policy."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WorkLimits:
    version: int = 1
    context_passages: int = 8
    context_bytes: int = 8000
    retrieval_candidates: int = 32
    retrieval_bytes: int = 16000
    max_methods: int = 128
    max_tasks: int = 128

    def __post_init__(self):
        if self.version != 1 or any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Positive integer limits and supported version required')
        if self.context_passages > self.retrieval_candidates or self.context_bytes > self.retrieval_bytes:
            raise ValueError('Retained context must fit within retrieval limits')

    def record(self):
        return asdict(self)


def work_limits(value=None):
    return value if isinstance(value, WorkLimits) else WorkLimits(**(value or {}))
