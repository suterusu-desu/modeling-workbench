"""Manually started loopback UI for the existing ModelingService; stdlib only."""
import argparse
import base64
import binascii
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
from urllib.parse import urlsplit, parse_qs
from .service import ModelingService


UI = Path(__file__).with_name('visual_feedback_ui')
OPERATIONS = {'visual_feedback_' + suffix for suffix in (
    'create', 'image', 'video', 'motion', 'target', 'plan', 'reference', 'compare', 'agreement', 'submit', 'state', 'presentation')}
MAX_BODY = 32 * 1024 * 1024


def make_server(service, port=8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not print private image/source/target data into terminal logs.

        def send(self, status, body, media='application/json; charset=utf-8'):
            data = json.dumps(body, allow_nan=False).encode('utf-8') if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header('Content-Type', media)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none'")
            try:
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # A closed browser does not undo an already journaled operation.
                # Do not retry it, send a second response, or dump private paths.
                return

        def local_request(self):
            # Reject browser cross-origin writes and DNS rebinding. No relay or network exposure.
            expected = '127.0.0.1:' + str(self.server.server_port)
            if self.headers.get('Host') != expected:
                self.send(403, {'summary': 'Use the printed loopback URL'})
                return False
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + expected:
                self.send(403, {'summary': 'Local same-origin requests only'})
                return False
            return True

        def video_bytes(self, path):
            # Browser seek needs bounded byte ranges. Resolve/hash the immutable
            # asset before serving any range, including a reopened historical clip.
            length = path.stat().st_size
            start, end, status = 0, length - 1, 200
            header = self.headers.get('Range')
            if header:
                import re
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
                if not match or not any(match.groups()):
                    self.send(416, {'summary': 'Use a single byte range'})
                    return
                left, right = match.groups()
                if left:
                    start = int(left)
                    end = min(int(right), length - 1) if right else length - 1
                else:
                    start = max(0, length - int(right))
                if start >= length or end < start:
                    self.send(416, {'summary': 'Range outside the pinned video'})
                    return
                status = 206
            self.send_response(status)
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if status == 206:
                self.send_header('Content-Range', f'bytes {start}-{end}/{length}')
            try:
                self.end_headers()
                with path.open('rb') as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        chunk = stream.read(min(65536, remaining))
                        if not chunk: break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return

        def do_GET(self):
            if not self.local_request():
                return
            url = urlsplit(self.path)
            try:
                if url.path == '/api/boards':
                    self.send(200, service.visual_feedback_read())
                elif url.path == '/api/board':
                    self.send(200, service.visual_feedback_read(parse_qs(url.query)['board'][0]))
                elif url.path == '/api/summary':
                    board = parse_qs(url.query)['board'][0]
                    self.send(200, service.visual_feedback_summary(board)['markdown'].encode('utf-8'), 'text/markdown; charset=utf-8')
                elif url.path.startswith('/api/image/'):
                    board, record = url.path[len('/api/image/'):].split('/')
                    feedback = service._visual_feedback()
                    row = feedback._record(feedback._board(board), 'images', record)
                    self.send(200, service.store.resolve_blob(row['asset']).read_bytes(), row['media_type'])
                elif url.path.startswith('/api/video/'):
                    board, record = url.path[len('/api/video/'):].split('/')
                    feedback = service._visual_feedback()
                    row = feedback._record(feedback._board(board), 'videos', record)
                    self.video_bytes(service.store.resolve_blob(row['asset']))
                elif url.path in ('/', '/index.html', '/app.js', '/style.css', '/workflow.json'):
                    name = 'index.html' if url.path == '/' else url.path[1:]
                    media = {'index.html': 'text/html; charset=utf-8', 'app.js': 'text/javascript; charset=utf-8', 'style.css': 'text/css; charset=utf-8', 'workflow.json': 'application/json; charset=utf-8'}[name]
                    self.send(200, (UI / name).read_bytes(), media)
                else:
                    self.send(404, {'summary': 'Unknown route'})
            except (ValueError, KeyError, OSError, RuntimeError) as error:
                self.send(400, {'summary': str(error)})

        def do_POST(self):
            if not self.local_request():
                return
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length <= MAX_BODY or self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    raise ValueError('Send a bounded JSON request')
                body = json.loads(self.rfile.read(length))
                if self.path in ('/api/upload', '/api/video-upload'):
                    raw = base64.b64decode(body.pop('data'), validate=True)
                    suffix = Path(body.pop('filename')).suffix.lower()
                    video = self.path == '/api/video-upload'
                    if suffix not in (('.mp4',) if video else ('.png', '.jpg', '.jpeg', '.webp')):
                        raise ValueError('Choose a PNG, JPEG or WebP image')
                    # Transient upload bytes go into the same private store, then into content-addressed assets.
                    with tempfile.TemporaryDirectory(dir=service.store.root) as folder:
                        path = Path(folder) / ('inspection' + suffix)
                        path.write_bytes(raw)
                        result = service.execute('visual_feedback_video' if video else 'visual_feedback_image',
                            dict(body, **{('video_path' if video else 'image_path'): str(path)}))
                elif self.path == '/api/operation':
                    operation = body['operation']
                    if operation not in OPERATIONS:
                        raise ValueError('This UI only records visual feedback; native operations are unavailable')
                    result = service.execute(operation, body['arguments'])
                else:
                    self.send(404, {'summary': 'Unknown route'})
                    return
                self.send(409 if result.get('status') == 'conflicting' else 400 if result.get('status') == 'failed' else 200, result)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError, binascii.Error) as error:
                self.send(400, {'summary': str(error)})

    return HTTPServer(('127.0.0.1', port), Handler)


def serve(workspace=None, store=None, port=8765):
    service = ModelingService(workspace, store)
    server = make_server(service, port)
    print('Visual feedback: http://127.0.0.1:' + str(server.server_port) + '/', flush=True)
    print('Private historical evidence only. Stop with Ctrl+C when finished.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, help='Explicit private workspace; never the package checkout')
    parser.add_argument('--store', help='Optional existing Workbench evidence store')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    serve(args.workspace, args.store, args.port)


if __name__ == '__main__':
    main()
