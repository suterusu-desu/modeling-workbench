"""Serve the workbench's native bridge from inside Blender, for one character workspace.

Run by Blender, usually through `python -m modeling_system.live launch`:

    blender [file.blend] --python live_bridge.py -- --workspace <character workspace> [--port N]

It claims the live lane for the binding's `native_owner`, then answers `{"type": "execute_code"}` requests from
`NativeBridge` on 127.0.0.1 (the binding's `native_bridge_port` unless `--port` is given), running each on Blender's
main thread and returning its printed output. In a visible Blender requests run from a timer, so the window stays live
and shows the work; in `--background` the bridge serves on the main thread until a `{"type": "shutdown"}` request. The
3D view header names the owner, file, controls and the guide shown (reference adapter). It never saves by itself.
Local, trusted code only: anything that can reach the port can run code in this Blender.
"""
import contextlib
import io
import json
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import bpy


def _arguments():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    options = {}
    for flag in ('--workspace', '--port'):
        if flag in argv:
            options[flag[2:]] = argv[argv.index(flag) + 1]
    if 'workspace' not in options:
        raise SystemExit('live_bridge needs -- --workspace <character workspace>')
    return options


def _adapter():
    import importlib.util
    path = Path(__file__).with_name('reference_adapter.py')
    spec = importlib.util.spec_from_file_location('_modeling_workbench_live_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


OPTIONS = _arguments()
WORKSPACE = Path(OPTIONS['workspace']).resolve()
BINDING = json.loads((WORKSPACE / 'modeling-workspace.json').read_text(encoding='utf-8-sig'))
OWNER = BINDING['native_owner']
PORT = int(OPTIONS.get('port') or BINDING['native_bridge_port'])
REFERENCE = _adapter()
LANE = bpy.app.driver_namespace.setdefault(REFERENCE.lane_key(WORKSPACE), {})
LANE.update(owner=OWNER, session=uuid.uuid4().hex, status='live bridge ready')
REQUESTS = queue.Queue()
STOP = threading.Event()


def _run(message):
    kind = message.get('type')
    if kind == 'shutdown':
        STOP.set()
        return {'status': 'success', 'result': {'result': 'stopping'}}
    if kind != 'execute_code':
        return {'status': 'error', 'message': 'Only execute_code and shutdown are supported'}
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            exec(message['params']['code'], {'__name__': '__modeling_workbench_bridge__'})
        return {'status': 'success', 'result': {'result': out.getvalue()}}
    except Exception:
        return {'status': 'error', 'message': traceback.format_exc()}
    finally:
        _header()


def _header():
    config = BINDING.get('native_configuration', {}).get('reference')
    if config and config.get('working'):
        try:
            REFERENCE.header(WORKSPACE, config)
        except Exception:
            pass


def _receive(conn):
    data = b''
    while True:
        part = conn.recv(65536)
        if not part:
            return None
        data += part
        try:
            return json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue


def _listen():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT)); server.listen(8)
    return server


def _session_file(mode):
    folder = WORKSPACE / 'runtime' / 'live'; folder.mkdir(parents=True, exist_ok=True)
    (folder / 'session.json').write_text(json.dumps({
        'owner': OWNER, 'port': PORT, 'session': LANE['session'], 'file': bpy.data.filepath, 'mode': mode,
        'started': time.time()}, indent=1), encoding='utf-8')


def serve_background():
    """Answer requests one at a time on the main thread until a shutdown request."""
    server = _listen(); _session_file('background')
    print(json.dumps({'live_bridge': 'ready', 'port': PORT, 'owner': OWNER}), flush=True)
    while not STOP.is_set():
        conn, _ = server.accept()
        with conn:
            message = _receive(conn)
            if message is not None:
                conn.sendall(json.dumps(_run(message)).encode('utf-8'))
    server.close()


def serve_window():
    """Accept on a thread; run each request on the main thread from a timer, so the window keeps drawing."""
    server = _listen()

    def accept():
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()

    def handle(conn):
        with conn:
            message = _receive(conn)
            if message is None:
                return
            box, done = {}, threading.Event(); REQUESTS.put((message, box, done)); done.wait(1800)
            conn.sendall(json.dumps(box.get('response', {'status': 'error', 'message': 'main-thread timeout'})).encode('utf-8'))

    def pump():
        while not REQUESTS.empty():
            message, box, done = REQUESTS.get()
            try:
                box['response'] = _run(message)
            finally:
                done.set()
        return .05

    threading.Thread(target=accept, daemon=True).start()
    bpy.app.timers.register(pump, first_interval=.2, persistent=True)
    bpy.app.timers.register(lambda: (_header(), None)[-1], first_interval=1.)
    _session_file('window')


if bpy.app.background:
    serve_background()
else:
    serve_window()
