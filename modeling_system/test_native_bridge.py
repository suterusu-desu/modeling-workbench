"""Transport/schema regressions; real native acceptance receipts live in migration/native.

Run with: python -m unittest modeling_system.test_native_bridge
No test launches Blender or mutates a user's scene automatically.
"""
import json
import socket
import threading
import unittest
from unittest.mock import patch

from .native_bridge import NativeBridge, NativeBridgeError, operation_schemas, validate_arguments


class SchemaTests(unittest.TestCase):
    def test_all_operations_have_closed_argument_contracts(self):
        schemas = operation_schemas()
        self.assertEqual(len(schemas), 17)
        self.assertTrue(all(s['additionalProperties'] is False for s in schemas.values()))
        schemas['inspect_live']['properties']['bad'] = {}
        self.assertNotIn('bad', operation_schemas()['inspect_live']['properties'])

    def test_state_and_proposal_fields_are_required(self):
        for name, args in [('apply_proposal', {}), ('capture_view', {}),
                           ('set_controls', {'expected_state':'x', 'controls':{'blink':float('nan')}}),
                           ('inspect_live', {'_owner':'injected'}),
                           ('set_display', {'expected_state':'x', 'view':{'distance':0}})]:
            with self.subTest(operation=name), self.assertRaises(NativeBridgeError):
                validate_arguments(name, args)

    def test_display_can_set_named_parts_aside(self):
        self.assertTrue(validate_arguments('set_display', {'expected_state':'abc', 'mode':'GUIDE_WIRE',
            'hide':['upper lash', 'lower lash']}))
        self.assertTrue(validate_arguments('set_display', {'expected_state':'abc', 'mode':'GUIDE_WIRE', 'hide':[]}))
        for hide in (['lash', 'lash'], [''], 'upper lash', [3], ['part'] * 65):
            with self.subTest(hide=hide), self.assertRaises(NativeBridgeError):
                validate_arguments('set_display', {'expected_state':'abc', 'mode':'GUIDE_WIRE', 'hide':hide})

    def test_structured_controls_and_region_capture(self):
        self.assertTrue(validate_arguments('set_controls', {'expected_state':'abc', 'controls':{'blink':.95}, 'guide':'posed_corner_guide'}))
        self.assertTrue(validate_arguments('capture_view', {'expected_state':'abc', 'viewport':{'window':0,'area':3},
            'crop':[100,80,200,160], 'channels':['sections','viewport_depth']}))
        with self.assertRaises(NativeBridgeError):
            validate_arguments('capture_motion', {'expected_state':'x', 'poses':[{'controls':{}}]*65})


class TransportTests(unittest.TestCase):
    def server(self, responder):
        server = socket.socket(); server.bind(('127.0.0.1', 0)); server.listen(1)
        errors = []
        def serve():
            try:
                with server, server.accept()[0] as client:
                    data = bytearray()
                    while True:
                        data.extend(client.recv(4096))
                        try: request = json.loads(data.decode()); break
                        except ValueError: pass
                    responder(client, request)
            except Exception as exc: errors.append(exc)
        worker = threading.Thread(target=serve, daemon=True); worker.start()
        self.addCleanup(worker.join, 2.)
        return NativeBridge('.', port=server.getsockname()[1], timeout=.5), errors

    @staticmethod
    def frame(request, value):
        import ast
        tree = ast.parse(request['params']['code'])
        token = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                     and isinstance(node.value, str) and node.value.startswith('__MODELING_NATIVE_'))
        return json.dumps({'status':'success', 'result':{'executed':True,
            'result':'script output before frame\n'+token+json.dumps(value, ensure_ascii=False)+token+'\n'}}, ensure_ascii=False).encode()

    def test_partial_utf8_response_and_stdout_are_framed(self):
        def response(client, request):
            packet = self.frame(request, {'ok':True, 'result':{'label':'Example Character — 眼'}})
            for byte in packet: client.sendall(bytes([byte]))
        bridge, errors = self.server(response)
        self.assertEqual(bridge.call('inspect_live', {}), {'label':'Example Character — 眼'})
        self.assertFalse(errors)

    def test_typed_native_recovery_error(self):
        def response(client, request):
            client.sendall(self.frame(request, {'ok':False, 'error':{
                'code':'stale_state','message':'user edit','recovery':'refresh','details':{'later':True}}}))
        bridge, errors = self.server(response)
        with self.assertRaises(NativeBridgeError) as caught: bridge.call('inspect_live', {})
        self.assertEqual(caught.exception.code, 'stale_state')
        self.assertTrue(caught.exception.details['later'])

    def test_preloaded_package_cannot_redirect_the_bound_native_dispatcher(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from . import blender_operations as prior
        from .store import digest
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory);entry=workspace/'adapter.py'
            entry.write_text("def execute(operation, arguments):\n    return {'ok':True,'result':{'adapter':'bound synthetic adapter','owner':arguments['_owner']}}\n")
            (workspace/'modeling-workspace.json').write_text(json.dumps(dict(native_adapter='blender_json_v1',
                native_configuration=dict(operations=['inspect_live'],entrypoint=dict(path='adapter.py',sha256=digest(entry.read_bytes()))))))
            def response(client,request):
                stdout=io.StringIO()
                with contextlib.redirect_stdout(stdout):exec(request['params']['code'],{})
                client.sendall(json.dumps({'status':'success','result':{'result':stdout.getvalue()}}).encode())
            bridge,errors=self.server(response);bridge.workspace=workspace;bridge.timeout=5.
            with patch.object(prior,'execute',side_effect=AssertionError('Stale package selected')) as stale:
                result=bridge.call('inspect_live',{},owner='fixture owner')
                self.assertEqual(result,{'adapter':'bound synthetic adapter','owner':'fixture owner'})
                stale.assert_not_called()
                self.assertIs(prior.execute,stale)
            self.assertFalse(errors)

    def test_disconnect_after_send_is_ambiguous_and_not_retried(self):
        count = []
        bridge, errors = self.server(lambda client, request: count.append(request))
        with self.assertRaises(NativeBridgeError) as caught: bridge.call('set_controls', {'expected_state':'x'}, owner='test')
        self.assertEqual(caught.exception.code, 'uncertain_execution')
        self.assertTrue(caught.exception.details['operation_may_have_run'])
        self.assertEqual(len(count), 1)

    def test_connection_failure_is_distinct_from_uncertain_execution(self):
        with patch('socket.create_connection', side_effect=ConnectionRefusedError('closed')):
            with self.assertRaises(NativeBridgeError) as caught: NativeBridge('.').call('inspect_live', {})
        self.assertEqual(caught.exception.code, 'blender_unavailable')


if __name__ == '__main__':
    unittest.main()
