"""Local JSON transport for the installed Blender addon and native domain operations.

No mutation is retried automatically: a lost response may follow a committed edit.
The socket endpoint must be the user's local Blender addon, never a provider URL.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
import sys


class NativeBridgeError(RuntimeError):
    def __init__(self, code, message, recovery, details=None):
        super().__init__(message)
        self.code, self.message, self.recovery = code, message, recovery
        self.details = details or {}

    def as_dict(self):
        return dict(code=self.code, message=self.message,
                    recovery=self.recovery, details=self.details)


def _object(properties=None, required=(), **extra):
    return dict(type='object', properties=properties or {}, required=list(required),
                additionalProperties=False, **extra)


_S = {'type': 'string', 'minLength': 1}
_N = {'type': 'number'}
_B = {'type': 'boolean'}
_V = {'type': 'array', 'items': _N, 'minItems': 3, 'maxItems': 3}
_PATHREF = _object({'path': _S, 'sha256': _S, 'role': _S}, ('path', 'sha256'))
_CONTROLS = {'type': 'object', 'additionalProperties': _N}
_VIEWPORT = _object({'window': {'type': 'integer', 'minimum': 0},
                     'area': {'type': 'integer', 'minimum': 0}})
_CROP = {'type': 'array', 'items': {'type': 'integer', 'minimum': 0},
         'minItems': 4, 'maxItems': 4, 'description': 'x, y, width, height in native region pixels, top-left origin.'}
_POSE = _object({'controls': _CONTROLS, 'guide': _S, 'time': _N, 'label': _S,
                 'source_video_time': _N}, ('controls',))
_SHAPE = _object({'object': _S, 'key': _S, 'indices': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 1},
                  'coordinates': {'type': 'array', 'items': _V, 'minItems': 1},
                  'create': _B, 'relative_key': _S}, ('object', 'key', 'indices', 'coordinates'))
_PROPOSAL = _object({'kind': {'enum': ['shape_key', 'script']},
                     'label': _S, 'feature': {'enum': ['eyes', 'mouth']},
                     'mismatch': _S, 'mechanism': _S,
                     'constraints': {'type': 'array', 'items': _PATHREF, 'minItems': 1},
                     'allowed_objects': {'type': 'array', 'items': _S, 'minItems': 1},
                     'allowed_datablocks': {'type': 'array', 'items': _S},
                     'protected_objects': {'type': 'array', 'items': _S},
                     'repair_invalid_bindings': _B,
                     'shape_keys': {'type': 'array', 'items': _SHAPE, 'minItems': 1},
                     'script': _PATHREF},
                    ('kind', 'label', 'feature', 'mismatch', 'mechanism', 'constraints', 'allowed_objects'))
_QUALIFICATION = _object({'target_id': _S, 'object_name': _S, 'label': _S,
    'registered_npz': _PATHREF, 'source_state_id': _S,
    'feature': {'enum': ['eyes', 'mouth']}, 'pose': _CONTROLS,
    'region_support': {'type': 'object'}, 'target_role': _S,
    'reviewed': {'const': True}, 'review_record': _PATHREF,
    'provenance': _object({'raw_mesh': _PATHREF, 'artwork': {'type': 'array', 'items': _PATHREF, 'minItems': 1},
                          'registration_record': _PATHREF, 'derivation_record': _PATHREF,
                          'anatomy_basis': {'type': 'array', 'items': _PATHREF, 'minItems': 1}},
                         ('raw_mesh', 'artwork', 'registration_record', 'derivation_record', 'anatomy_basis'))},
    ('target_id', 'registered_npz', 'source_state_id', 'feature', 'pose', 'region_support',
     'target_role', 'reviewed', 'review_record', 'provenance'))
_EXPECTED = {'expected_state': _S}
_RETENTION_VIEW = _object({'rotation': {'type': 'array', 'items': _N, 'minItems': 4, 'maxItems': 4},
    'location': _V, 'distance': {'type': 'number', 'exclusiveMinimum': 0},
    'perspective': {'enum': ['PERSP', 'ORTHO', 'CAMERA']}, 'lens': {'type': 'number', 'exclusiveMinimum': 0}})
_SCHEMAS = {
 'inspect_live': _object({'refresh_scene': _B}),
 'bootstrap': _object({**_EXPECTED, 'expected_file': _PATHREF}, ('expected_state', 'expected_file')),
 'owner_release': _object(_EXPECTED, ('expected_state',)),
 'open_checkpoint': _object({**_EXPECTED, 'source': _PATHREF, 'load_ui': _B}, ('expected_state', 'source')),
 'retain_checkpoint': _object({**_EXPECTED, 'transaction_id': _S, 'source': _PATHREF,
    'candidate': _PATHREF, 'reopen': _PATHREF, 'target': _S, 'label': _S,
    'pose': _object({'controls': _CONTROLS, 'guide': _S, 'refresh': {'type': 'boolean', 'const': False}}),
    'display': _object({'mode': {'const': 'GUIDE_WIRE'}, 'through': _B, 'parts': _B,
        'viewport': _VIEWPORT, 'view': _RETENTION_VIEW}, ('mode',))},
    ('expected_state', 'transaction_id', 'source', 'candidate', 'reopen', 'target', 'label', 'display')),
 'inspect_feature': _object({'feature': {'enum': ['eyes', 'mouth']}, 'refresh': _B, **_EXPECTED}),
 'prepare_state': _object({'feature': {'enum': ['eyes', 'mouth']}, 'output_dir': _S, **_EXPECTED}, ('expected_state',)),
 'capture_view': _object({**_EXPECTED, 'output_dir': _S, 'feature': {'enum': ['eyes', 'mouth']},
    'viewport': _VIEWPORT, 'crop': _CROP, 'context': _B,
    'channels': {'type': 'array', 'items': {'enum': ['surface', 'sections', 'viewport_depth']}, 'uniqueItems': True}}, ('expected_state',)),
 'set_controls': _object({**_EXPECTED, 'controls': _CONTROLS, 'guide': _S, 'refresh': _B}, ('expected_state',)),
 'set_display': _object({**_EXPECTED, 'mode': {'enum': ['GUIDE_WIRE', 'WORK_WIRE', 'SURFACES', 'DISTANCE', 'SECTION', 'CLAY', 'MATERIALS']},
    'through': _B, 'parts': _B, 'axis': {'enum': ['HORIZONTAL', 'VERTICAL']}, 'section_value': _N,
    'viewport': _VIEWPORT, 'view': _object({'rotation': {'type': 'array', 'items': _N, 'minItems': 4, 'maxItems': 4},
        'location': _V, 'distance': {'type': 'number', 'exclusiveMinimum': 0},
        'perspective': {'enum': ['PERSP', 'ORTHO', 'CAMERA']}, 'lens': {'type': 'number', 'exclusiveMinimum': 0}})}, ('expected_state',)),
 'apply_proposal': _object({**_EXPECTED, 'proposal': _PROPOSAL}, ('expected_state', 'proposal')),
 'rollback_trial': _object({**_EXPECTED, 'receipt_path': _S}, ('expected_state', 'receipt_path')),
 'integrate_guide': _object({**_EXPECTED, 'qualification': _QUALIFICATION, 'expected_registry': _PATHREF}, ('expected_state', 'qualification', 'expected_registry')),
 'save_checkpoint': _object({**_EXPECTED, 'path': _S, 'label': _S, 'copy': _B,
                             'expected_file_sha256': _S}, ('expected_state', 'label')),
 'export_native': _object({**_EXPECTED, 'path': _S, 'format': {'enum': ['BLEND', 'NPZ', 'GLB', 'OBJ', 'FBX']},
    'objects': {'type': 'array', 'items': _S, 'minItems': 1}}, ('expected_state', 'path', 'format')),
 'prepare_reopen': _object({**_EXPECTED, 'source': _PATHREF,
    'poses': {'type': 'array', 'items': _POSE, 'minItems': 1, 'maxItems': 64},
    'objects': {'type': 'array', 'items': _S, 'minItems': 1},
    'dispatch': _B, 'blender': _S, 'runner': _S, 'python': _S}, ('expected_state', 'source')),
 'capture_motion': _object({**_EXPECTED, 'output_dir': _S,
    'poses': {'type': 'array', 'items': _POSE, 'minItems': 1, 'maxItems': 64},
    'feature': {'enum': ['eyes', 'mouth']}, 'viewport': _VIEWPORT, 'crop': _CROP,
    'context': _B, 'channels': {'type': 'array', 'items': {'enum': ['surface', 'sections', 'viewport_depth']}, 'uniqueItems': True}}, ('expected_state', 'poses')),
}


def operation_schemas():
    """JSON Schema argument contracts used locally and exposed unchanged by MCP."""
    return copy.deepcopy(_SCHEMAS)


def validate_arguments(operation, arguments):
    if operation not in _SCHEMAS:
        raise NativeBridgeError('unknown_operation', operation, 'Choose an operation from operation_schemas().')
    def visit(value, schema, path):
        def fail(message):
            raise NativeBridgeError('invalid_arguments', path + ': ' + message, 'Correct the arguments using operation_schemas().')
        if 'const' in schema and value != schema['const']: fail('unexpected constant')
        if 'enum' in schema and value not in schema['enum']: fail('value outside enum')
        kind = schema.get('type')
        match = {'object': isinstance(value, dict), 'array': isinstance(value, list),
                 'string': isinstance(value, str), 'boolean': isinstance(value, bool),
                 'number': isinstance(value, (int, float)) and not isinstance(value, bool),
                 'integer': isinstance(value, int) and not isinstance(value, bool)}
        if kind and not match[kind]: fail('expected ' + kind)
        if kind in ('number', 'integer'):
            if not math.isfinite(value): fail('number must be finite')
            if 'minimum' in schema and value < schema['minimum']: fail('below minimum')
            if 'exclusiveMinimum' in schema and value <= schema['exclusiveMinimum']: fail('below exclusive minimum')
        if kind == 'string' and len(value) < schema.get('minLength', 0): fail('empty string')
        if kind == 'object':
            for key in schema.get('required', []):
                if key not in value: fail('missing ' + key)
            for key, child in value.items():
                if key in schema.get('properties', {}): visit(child, schema['properties'][key], path + '.' + key)
                elif schema.get('additionalProperties') is False: fail('unexpected ' + key)
                elif isinstance(schema.get('additionalProperties'), dict): visit(child, schema['additionalProperties'], path + '.' + key)
        if kind == 'array':
            if len(value) < schema.get('minItems', 0) or len(value) > schema.get('maxItems', float('inf')): fail('invalid array length')
            if schema.get('uniqueItems') and len({json.dumps(x, sort_keys=True) for x in value}) != len(value): fail('duplicate array items')
            for index, child in enumerate(value): visit(child, schema.get('items', {}), path + '[' + str(index) + ']')
    visit(arguments, _SCHEMAS[operation], operation)
    return True


class NativeBridge:
    def __init__(self, workspace, host='127.0.0.1', port=9876, timeout=240., max_response_bytes=64 * 1024 * 1024):
        if host not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('Native bridge only supports a local Blender endpoint')
        self.workspace = Path(workspace).resolve()
        self.host, self.port, self.timeout = host, int(port), float(timeout)
        self.max_response_bytes = max_response_bytes

    def call(self, operation, arguments, owner=None):
        validate_arguments(operation, arguments)
        token = '__MODELING_NATIVE_' + uuid.uuid4().hex + '__'
        payload = dict(arguments, _workspace=str(self.workspace), _owner=owner)
        package=Path(__file__).resolve().parent
        namespace='_modeling_native_'+hashlib.sha256(str(package).encode()).hexdigest()[:20]
        # sys.path insertion cannot replace a package already held by Blender.
        # Isolate this installation without reloading another owner's modules.
        code = ('import sys,json,importlib,types\n'
                f'_p={str(package)!r}\n_n={namespace!r}\n'
                'if _n not in sys.modules:\n'
                ' _package=types.ModuleType(_n);_package.__path__=[_p];_package.__package__=_n\n'
                ' sys.modules[_n]=_package\n'
                'if list(sys.modules[_n].__path__)!=[_p]: raise RuntimeError("Native package namespace collision")\n'
                'importlib.invalidate_caches()\n'
                'for _dependency in (_n+".native_bridge",_n+".geometry",_n+".blender_capture"):\n'
                ' if _dependency in sys.modules: importlib.reload(sys.modules[_dependency])\n'
                '_m=importlib.import_module(_n+".blender_operations")\n'
                '_m=importlib.reload(_m)\n'
                f'_r=_m.execute({operation!r},json.loads({json.dumps(payload, allow_nan=False)!r}))\n'
                f'print({token!r}+json.dumps(_r,allow_nan=False)+{token!r})\n')
        packet = json.dumps({'type': 'execute_code', 'params': {'code': code}}, ensure_ascii=False).encode('utf-8')
        sent = False
        chunks = bytearray()
        deadline = time.monotonic() + self.timeout
        try:
            with socket.create_connection((self.host, self.port), timeout=min(self.timeout, 10.)) as client:
                client.sendall(packet); sent = True
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0: raise TimeoutError('Blender response deadline expired')
                    client.settimeout(remaining)
                    part = client.recv(65536)
                    if not part: raise ConnectionError('Blender closed the connection before a complete response')
                    chunks.extend(part)
                    if len(chunks) > self.max_response_bytes:
                        raise NativeBridgeError('response_too_large', 'Blender response exceeded the bounded transport limit',
                            'Inspect the persisted receipt; request compact file references instead of arrays.', {'operation_may_have_run': True})
                    try: response = json.loads(chunks.decode('utf-8'))
                    except (UnicodeDecodeError, json.JSONDecodeError): continue
                    break
        except NativeBridgeError:
            raise
        except (OSError, TimeoutError) as exc:
            raise NativeBridgeError('uncertain_execution' if sent else 'blender_unavailable', str(exc),
                'Inspect live state and the latest native receipt before retrying; no automatic mutation retry was made.' if sent else
                'Open the verified working Blender file and start its installed local bridge.',
                {'operation': operation, 'operation_may_have_run': sent}) from exc
        if response.get('status') != 'success':
            raise NativeBridgeError('addon_error', response.get('message', 'Blender addon rejected the command'),
                'Inspect live state and the native receipt before retrying.', {'operation_may_have_run': True})
        stdout = response.get('result', {}).get('result', '')
        pieces = stdout.split(token)
        if len(pieces) != 3:
            raise NativeBridgeError('invalid_response', 'Missing or ambiguous native response frame',
                'Inspect live state and native receipts; do not blindly retry.', {'operation_may_have_run': True})
        try: envelope = json.loads(pieces[1])
        except ValueError as exc:
            raise NativeBridgeError('invalid_response', str(exc), 'Inspect persisted evidence before retrying.') from exc
        if not envelope.get('ok'):
            err = envelope.get('error', {})
            raise NativeBridgeError(err.get('code', 'native_error'), err.get('message', 'Native operation failed'),
                                    err.get('recovery', 'Inspect the receipt and current Blender state.'), err.get('details'))
        result = envelope['result']
        if operation == 'prepare_reopen' and arguments.get('dispatch', False):
            result['dispatch'] = self._dispatch_reopen(result, arguments)
        return result

    def _dispatch_reopen(self, result, arguments):
        """Run the installed profile-isolating runner outside the UI process."""
        task = Path(result['job_path']); folder = task.parent
        python = Path(arguments.get('python', sys.executable))
        runner = Path(arguments.get('runner', ''))
        if not arguments.get('runner') or not python.is_file() or not runner.is_file():
            raise NativeBridgeError('runner_unavailable', 'Prepared reopen job; isolated runner/runtime missing',
                'Supply the installed runner and Python paths and dispatch the prepared job.', {'job_path': str(task)})
        path = folder / 'dispatch.json'
        with path.open('x', encoding='utf-8') as handle:
            log = folder / 'dispatch.log'
            with log.open('xb') as output:
                process = subprocess.Popen([str(python), str(runner), str(task)], cwd=str(self.workspace),
                    stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            receipt = {'status': 'running', 'pid': process.pid, 'job_path': str(task), 'log': str(log),
                       'output_root': result['output_root'], 'completion_evidence': 'runs/*/inventory.json and receipt.json'}
            json.dump(receipt, handle, indent=2)
        return dict(receipt, receipt_path=str(path))
