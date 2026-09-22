"""Direct owner-controlled sessions with preservation, evidence and native recovery.

No inference transport, credential lookup or provider ledger is involved. Supply
an authorized qualified catalog, then an explicit task order or local selector
when more than one operation is eligible. A singleton can run directly.
"""
from .operating_session import OperatingSession


def create_session(directory, *, service, episode, owner, goal, observe_context,
                   catalog, handlers, task_order=None, select=None, report=None,
                   budget=None, experience=None, preservation=None):
    return OperatingSession(directory, service=service, episode=episode,
        owner=owner, goal=goal, observe_context=observe_context, catalog=catalog,
        handlers=handlers, task_order=task_order, select=select, report=report,
        budget=budget, experience=experience, preservation=preservation,
        require_preservation=True)
