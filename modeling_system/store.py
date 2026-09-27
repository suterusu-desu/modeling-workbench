"""Content-addressed evidence storage, separate from the human decision ledger."""
from pathlib import Path
import hashlib
import json
import os
import time
import uuid


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def native_path(path):
    """Absolute extended Windows path for filesystem/media I/O, ordinary path elsewhere."""
    value=os.path.abspath(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value='\\\\?\\UNC\\'+value[2:] if value.startswith('\\\\') else '\\\\?\\'+value
    return Path(value)


def atomic_write(path, data):
    path = native_path(path)
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


def write_once(path, data, *, wait=5.):
    """Write content-addressed bytes that other processes may be writing at the same moment: when the replace fails
    because another process just wrote (or is reading) the same file, the file is accepted if it holds these bytes."""
    path = native_path(path)
    deadline = time.monotonic() + wait
    while True:
        if path.exists():
            try:
                if path.read_bytes() == data:
                    return
            except PermissionError:
                pass
            else:
                raise ValueError('Stored evidence was modified: ' + str(path))
        try:
            atomic_write(path, data)
            return
        except OSError:                       # Windows refuses a replace while another process holds the file
            if time.monotonic() > deadline:
                raise
            time.sleep(.02)


def append_line(path, line, *, wait=10., stale=60.):
    """Append one line to a shared journal from any number of processes: a lock file serializes the appends (append
    mode alone is not atomic across processes on Windows). A lock older than `stale` seconds is taken as abandoned."""
    path = native_path(path); path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + '.lock'); deadline = time.monotonic() + wait
    while True:
        try:
            with lock.open('x', encoding='utf-8') as stream:
                stream.write(str(os.getpid()))
            break
        except (FileExistsError, PermissionError):     # Windows: a lock being deleted by another process refuses creation
            try:
                if time.time() - lock.stat().st_mtime > stale:
                    lock.unlink(missing_ok=True); continue
            except (FileNotFoundError, PermissionError):
                pass
            if time.monotonic() > deadline:
                raise RuntimeError('Journal ' + str(path) + ' stays locked; inspect ' + str(lock)) from None
            time.sleep(.01)
    try:
        with path.open('a', encoding='utf-8') as stream:
            stream.write(line.rstrip('\n') + '\n'); stream.flush(); os.fsync(stream.fileno())
    finally:
        lock.unlink(missing_ok=True)


class Store:
    def __init__(self, path):
        self.root = native_path(path).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._record_headers = {}

    def blob(self, path):
        source=str(Path(path).resolve())
        path = native_path(path)
        data = path.read_bytes()
        sha = digest(data)
        dest = self.root / 'assets' / sha[:2] / (sha + path.suffix.lower())
        write_once(dest, data)
        return {'sha256': sha, 'path': str(dest.relative_to(self.root)), 'source': source, 'bytes': len(data)}

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
        try:
            write_once(dest, data)
        except ValueError:
            raise ValueError('Record integrity mismatch') from None
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
        """Fresh verified records; cache only unrelated-file classification.

        The content-addressed store is append-only. A directory scan discovers
        new/deleted files on every call; changed metadata invalidates a cached
        header. Every selected record still goes through get(), which reads and
        hashes its actual bytes even when timestamps have been preserved. No
        payload, modeling dependency or native-state check is cached here.
        """
        prefixes = tuple(('{"kind":"' + kind + '"').encode() for kind in kinds)
        directory = self.root / 'records'
        if not directory.exists():
            self._record_headers.clear()
            return
        # scandir supplies directory metadata cheaply on Windows; avoid opening
        # thousands of unrelated receipts at every decision validation boundary.
        with os.scandir(directory) as entries:
            files = sorted((entry for entry in entries if entry.name.endswith('.json') and entry.is_file()),
                           key=lambda entry: entry.name)
        current = {}
        for entry in files:
            stat = entry.stat()
            identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._record_headers.get(entry.name)
            if cached is not None and cached[0] == identity:
                header = cached[1]
            else:
                with open(entry.path, 'rb') as stream:
                    header = stream.read(128)
            current[entry.name] = (identity, header)
            if header.startswith(prefixes):
                key = entry.name[:-5]
                yield key, self.get(key)
        self._record_headers = current

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
