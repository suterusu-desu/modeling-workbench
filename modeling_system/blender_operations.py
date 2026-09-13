"""Dispatch to an explicitly selected private Blender adapter; no character defaults.

Adapters are trusted local code, not a sandbox. A configured adapter must implement
native ownership, freshness, preservation, checkpoint and rollback contracts.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
from .native_bridge import NativeBridgeError, validate_arguments


def execute(operation, arguments):
    try:
        public={k:v for k,v in arguments.items() if not k.startswith('_')}
        validate_arguments(operation, public)
        workspace=Path(arguments['_workspace']).resolve()
        binding=json.loads((workspace/'modeling-workspace.json').read_text(encoding='utf-8-sig'))
        config=binding.get('native_configuration',{})
        if binding.get('native_adapter')!='blender_json_v1' or operation not in config.get('operations',[]):
            raise NativeBridgeError('adapter_unconfigured','This native operation has no qualified workspace adapter',
                'Bind the intended character adapter and verify its advertised operation before use.')
        refs=[config['entrypoint'],*config.get('dependencies',[])]
        paths=[]
        for ref in refs:
            path=(workspace/ref['path']).resolve()
            if hashlib.sha256(path.read_bytes()).hexdigest()!=ref['sha256']:
                raise NativeBridgeError('adapter_changed','A pinned native adapter dependency changed',
                    'Review the changed adapter and refresh its binding; do not invoke stale code.')
            paths.append(path)
        name='_modeling_workspace_adapter_'+refs[0]['sha256']
        spec=importlib.util.spec_from_file_location(name,paths[0]); module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result=module.execute(operation,arguments)
        if not isinstance(result,dict) or not isinstance(result.get('ok'),bool):
            raise ValueError('Adapter must return the standard ok/result or ok/error envelope')
        return result
    except NativeBridgeError as exc:
        return {'ok':False,'error':exc.as_dict()}
    except Exception as exc:
        return {'ok':False,'error':{'code':'adapter_error','message':str(exc),
            'recovery':'Inspect the bound adapter and any retained result before repeating effects.'}}
