"""Compose private qualified saved-array scopes in one installed Jev session.

No Blender imports or native handlers. Exact source scopes remain preserved;
the manifest explicitly binds this runtime and its new shared run directory.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from .jev_session import create_session
from .runtime import source_manifest
from modeling_system.candidate_pipeline import capability_task
from modeling_system.controller import read_json, write_json, fingerprint
from modeling_system.cooperation import compose_catalogs
from modeling_system.method_catalog import MethodCatalog
from modeling_system.preparation import ArrayPreparation, ARRAY_OPERATIONS
from modeling_system.preparation_contracts import PreparationOperation
from modeling_system.service import ModelingService


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    workspace = (manifest_path.parent/manifest['workspace']).resolve()
    directory = (workspace/manifest['directory']).resolve()
    if source_manifest()['revision'] != manifest['runtime_revision']:
        raise ValueError('Composed scope requires this exact installed package revision')
    files = {'authority': workspace/manifest.get('authority_file', 'AGENTS.md'),
             'state': workspace/manifest.get('state_file', 'PROJECT.md'),
             'workspace_binding': workspace/'modeling-workspace.json',
             'factory': Path(__file__), 'manifest': manifest_path}
    scopes, handlers, catalogs = [], {}, []
    for ordinal, reference in enumerate(manifest['scopes']):
        path = (workspace/reference['path']).resolve()
        if sha(path) != reference['sha256']:
            raise ValueError('Qualified source scope changed')
        scope = read_json(path)
        if scope.get('native_job'):
            raise ValueError('This supporting-work runner has no native capability')
        prefix = f'scope{ordinal}:'
        local = {prefix+k: (workspace/v).resolve() for k, v in scope['files'].items()}
        local[prefix+'source_scope'] = path
        files.update(local)
        for ref in scope['inputs'].values():
            ref['path'] = str((workspace/ref['path']).resolve())
            if Path(ref['path']) not in local.values() or sha(ref['path']) != ref['sha256']:
                raise ValueError('Array inputs must be exact qualified scope dependencies')
        scope['_reads'] = list(local) + ['authority','state','workspace_binding','runtime','factory','manifest']
        scope['_bindings'] = {role: [prefix+k for k in keys] for role, keys in scope['bindings'].items()}
        operations = {}
        if scope['operation'] not in ARRAY_OPERATIONS:
            implementation = (workspace/scope['files']['implementation']).resolve()
            # These are explicitly qualified private numerical modules, not model-generated code.
            for parent in (implementation.parent, implementation.parent.parent):
                if str(parent) not in sys.path: sys.path.insert(0, str(parent))
            spec = importlib.util.spec_from_file_location('qualified_array_scope_'+str(ordinal), implementation)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            operations[scope['operation']] = PreparationOperation(getattr(module, scope['function']),
                requires=tuple(scope.get('requires', [])))
        handler = f'arrays{ordinal}'
        handlers[handler] = ArrayPreparation(directory/handler, operations=operations)
        scope['_handler'] = handler; scope['_reference'] = deepcopy(reference)
        scopes.append(scope)

    def observe():
        # One actual read per unique file per observation, even across many roles.
        # Nothing persists across freshness boundaries or caches mutable scene state.
        digests = {path.resolve(): sha(path) for path in set(files.values())}
        return {'owner': manifest['owner'], 'authority_revision': digests[files['authority'].resolve()],
            'values': {**{key: digests[path.resolve()] for key, path in files.items()},
                       'runtime': source_manifest()['revision']},
            'active_operations': [], 'public_state': {'objective': manifest['objective'],
                'supporting_work': [{'description': s['description'], 'facts': s['facts'],
                    'limits': s['limits'], 'source': s['source_label']} for s in scopes]}}

    initial = observe()
    tasks = []
    for scope in scopes:
        task = capability_task(initial, key=scope['task'], revision=fingerprint(scope['_reference']),
            lane=scope.get('lane','diagnosis'), handler=scope['_handler'], description=scope['description'],
            completion=scope['completion'], profile='analysis', capability=scope['capability'],
            method=scope['method'], bindings=scope['_bindings'], reads=scope['_reads'],
            payload={'operation':scope['operation'], 'inputs':scope['inputs'], 'parameters':scope.get('parameters',{})})
        tasks.append(task)
    frozen = directory/'qualified-tasks.json'
    identity = {'manifest':sha(manifest_path), 'runtime':manifest['runtime_revision']}
    if frozen.exists():
        retained = read_json(frozen)
        if retained['binding'] != identity:
            raise ValueError('Preserve this run; its scope/runtime binding changed')
        tasks = retained['tasks']  # Changed inputs block, never redefine settled tasks.
    else:
        write_json(frozen, {'binding':identity,'tasks':tasks})
    for task in tasks:
        catalogs.append(MethodCatalog([{'id': task['id'], 'description':task['description'],
            'build':lambda state, outcomes, frozen_task=deepcopy(task): deepcopy(frozen_task)}]))
    service = ModelingService(workspace=str(workspace))
    session = create_session(directory, service=service, episode=manifest['episode'], owner=manifest['owner'],
        goal={'objective':manifest['objective']}, observe_context=observe,
        catalog=compose_catalogs(*catalogs), handlers=handlers,
        public_projection=lambda state, actions, plan: {'state':state['observations']['public'],
            'descriptions':{a['id']:a['description'] for a in actions}},
        ledger_directory=(workspace/manifest['ledger_directory']).resolve())
    return session, manifest


def main():
    session, manifest = create(sys.argv[1])
    def handoff(value):
        print(json.dumps({'handoff':str(session.directory/'handoff.json'),
                          'requests':len(value.get('requests',[]))}),flush=True)
    result = session.run(max_steps=manifest['max_steps'], feedback_timeout=manifest.get('feedback_timeout',0),
                         on_handoff=handoff)
    write_json(session.directory/'result.json', result)
    print(json.dumps({k:result.get(k) for k in ('status','reason','completed_operations','timings_ms')}),flush=True)


if __name__ == '__main__': main()
