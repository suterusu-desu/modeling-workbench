"""Resolve a raised task from known partial effects without replaying its handler."""
from copy import deepcopy
from pathlib import Path

from .controller import fingerprint, read_json, write_json
from .episodes import pin_link


def reconcile_failed_task(session, task, *, expected_handle, observed, evidence, report):
    """Record a resolved failure; remaining work must be a separate qualified task.

    This never claims that a handler returned when it raised. Native receipts and
    the original raised outcome remain unchanged, including completed substeps.
    """
    lock = session.directory / 'controller' / 'controller.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write('known partial-effect reconciliation')
    try:
        session.record = read_json(session.path)
        entry = session.record['results'][task]
        handle = entry.get('operation_handle')
        if handle != expected_handle or entry['status'] not in ('running', 'needs_reconciliation'):
            raise ValueError('Exact unresolved operation handle required')
        current = session._context()
        if current['active_operations']:
            raise ValueError('Stop the native lane before metadata-only reconciliation')
        folder = session.service.store.root / 'calls' / handle
        intent = read_json(folder / 'intent.json')
        if (intent.get('episode') != session.episode or
                fingerprint(intent['arguments']['task']) != entry['definition']):
            raise ValueError('Original task and episode identity required')
        if (folder / 'capability-result.json').exists():
            raise ValueError('A known capability return must use result/report recovery')
        if (observed.get('effect_status') != 'resolved_failed' or not observed.get('basis')
                or observed.get('uncertain_effects') != [] or not evidence
                or not isinstance(observed.get('completed_effects'), list)
                or not isinstance(observed.get('unapplied_effects'), list)
                or not (observed['completed_effects'] or observed['unapplied_effects'])):
            raise ValueError('Actual completed/unapplied effects and no remaining uncertainty required')
        pinned = [pin_link(session.service, link) for link in evidence]
        paths = [str(Path(link['path'])) for link in evidence if link.get('kind') == 'file']
        if len(paths) != len(evidence):
            raise ValueError('Recovery requires the actual retained file receipts')
        result = {'status': 'failed', 'original_operation_handle': handle,
                  'observed': deepcopy(observed), 'evidence': pinned,
                  'effects_replayed': False, 'provider_calls_added': 0}
        adapter = session.report_adapter
        try:
            session.report_adapter = lambda item, value: report
            result['workbench'] = session._report(intent['arguments']['task'], result, intent['arguments'])
        finally:
            session.report_adapter = adapter
        recovered = session.service._run_episode_callback(session.episode, 'reconcile_failed_modeling_task',
            {'original_handle': handle, 'observed': observed, 'evidence': pinned}, lambda: result)
        session.service.reconcile_operation(handle, observed={**observed,
            'recovery_operation': recovered['operation_handle']}, evidence_paths=paths)
        entry.update(status='failed', result=recovered)
        write_json(session.path, session.record)
        session._recover_controller(entry)
        return deepcopy(recovered)
    finally:
        lock.unlink()
