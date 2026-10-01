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
    'create', 'image', 'target', 'plan', 'reference', 'compare', 'agreement', 'submit', 'state', 'presentation')}
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
            self.end_headers()
            self.wfile.write(data)

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
                if self.path == '/api/upload':
                    raw = base64.b64decode(body.pop('data'), validate=True)
                    suffix = Path(body.pop('filename')).suffix.lower()
                    if suffix not in ('.png', '.jpg', '.jpeg', '.webp'):
                        raise ValueError('Choose a PNG, JPEG or WebP image')
                    # Transient upload bytes go into the same private store, then into content-addressed assets.
                    with tempfile.TemporaryDirectory(dir=service.store.root) as folder:
                        path = Path(folder) / ('inspection' + suffix)
                        path.write_bytes(raw)
                        result = service.execute('visual_feedback_image', dict(body, image_path=str(path)))
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
