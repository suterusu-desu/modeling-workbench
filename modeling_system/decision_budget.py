"""One local decision budget, distinct from token limits and spending authority."""
from dataclasses import asdict, dataclass
import json


def encoded_size(value):
    return len(json.dumps(value, allow_nan=False).encode('utf-8'))


@dataclass(frozen=True)
class DecisionBudget:
    version: int = 1
    max_questions: int = 96
    max_request_bytes: int = 60000
    max_state_question_bytes: int = 30000
    context_passages: int = 8
    context_bytes: int = 8000
    retrieval_candidates: int = 32
    retrieval_bytes: int = 16000
    max_methods: int = 128
    max_tasks: int = 128

    def __post_init__(self):
        if self.version != 1 or any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('Positive integer limits and supported decision budget version required')
        if (self.max_questions > 1024 or self.max_request_bytes > 64000
                or self.max_state_question_bytes > 32000 or self.max_methods > 254
                or self.max_tasks > 254 or self.retrieval_candidates > 254
                or self.context_passages > self.retrieval_candidates
                or self.context_bytes > self.retrieval_bytes
                or self.retrieval_bytes > self.max_request_bytes):
            raise ValueError('Decision budget exceeds supported conservative byte or menu envelope')

    def record(self):
        return asdict(self)

    def check(self, state, questions, model='jev-1.13.0'):
        if not 1 <= len(questions) <= self.max_questions:
            raise ValueError('Useful question count exceeds the explicit decision budget')
        wire = {'model': model, 'state': state, 'questions': questions}
        size = encoded_size(wire)
        longest = max(encoded_size({'state': state, 'question': q}) for q in questions.values())
        if size > self.max_request_bytes or longest > self.max_state_question_bytes:
            raise ValueError('Decision byte budget exceeded; narrow relevant state or split independent work')
        return {'request_bytes': size, 'state_longest_question_bytes': longest,
                'question_count': len(questions), 'token_count': None,
                'basis': 'Conservative encoded-byte envelope, not a measured tokenizer count'}


def decision_budget(value=None):
    if isinstance(value, DecisionBudget):
        return value
    return DecisionBudget(**(value or {}))


def bound_budget(binding):
    return decision_budget(binding.get('decision_budget'))
