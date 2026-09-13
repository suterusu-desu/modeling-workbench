"""Content-addressed evidence storage, separate from the human decision ledger."""
from pathlib import Path
import hashlib
import json
import os
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Do not append to the destination name: a valid Windows path can then
    # exceed MAX_PATH, and a valid long basename can exceed NAME_MAX on POSIX.
    temp = path.with_name(uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class Store:
    def __init__(self, path):
        self.root = Path(path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def blob(self, path):
        path = Path(path)
        data = path.read_bytes()
        sha = digest(data)
        dest = self.root / 'assets' / sha[:2] / (sha + path.suffix.lower())
        if dest.exists():
            if digest(dest.read_bytes()) != sha:
                raise ValueError('Stored evidence was modified: ' + str(dest))
        else:
            atomic_write(dest, data)
        return {'sha256': sha, 'path': str(dest.relative_to(self.root)), 'source': str(path.resolve()), 'bytes': len(data)}

    def resolve_blob(self, blob):
        path = (self.root / blob['path'].replace('\\','/')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Evidence path leaves the store')
        if digest(path.read_bytes()) != blob['sha256']:
            raise ValueError('Evidence integrity mismatch: ' + str(path))
        return path

    def put(self, kind, payload):
        value = {'schema_version': 1, 'kind': kind, 'payload': payload}
        key = digest(canonical(value))
        dest = self.root / 'records' / (key + '.json')
        data = canonical(value)
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError('Record integrity mismatch')
        if not dest.exists():
            atomic_write(dest, data)
        return key

    def get(self, key, kind=None):
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ValueError('Invalid record reference')
        data = (self.root / 'records' / (key + '.json')).read_bytes()
        if digest(data) != key:
            raise ValueError('Record integrity mismatch')
        value = json.loads(data)
        if kind and value['kind'] != kind:
            raise ValueError('Expected ' + kind + ', received ' + value['kind'])
        return value['payload']

    def current(self):
        path = self.root / 'current.json'
        return json.loads(path.read_text(encoding='utf-8'))['question'] if path.exists() else None

    def records(self, kinds):
        """Verified small records by kind, without loading unrelated geometry."""
        prefixes = tuple(('{"kind":"' + kind + '"').encode() for kind in kinds)
        for path in sorted((self.root / 'records').glob('*.json')):
            with path.open('rb') as stream:
                relevant = stream.read(128).startswith(prefixes)
            if relevant:
                yield path.stem, self.get(path.stem)

    def set_current(self, question, expected):
        self.get(question, 'question')
        lock = self.root / 'current.lock'
        try:
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
        except FileExistsError as exc:
            raise RuntimeError('Current-question update is already in progress; inspect the owner before retrying') from exc
        try:
            if self.current() != expected:
                raise RuntimeError('Current question changed; preserve the newer question and reconcile')
            atomic_write(self.root / 'current.json', canonical({'question': question}))
        finally:
            lock.unlink()
