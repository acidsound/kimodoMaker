"""SQLite-backed, CPU-only receipts for artifacts published by an external runner."""
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import sqlite3
import stat
import tempfile


class EvidenceError(ValueError):
    """Cannot prove this run or operation matches its frozen evidence."""


def _json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _key(unit):
    return unit['arm'], unit['prompt_id'], unit['seed']


def _require(test, message):
    if not test:
        raise EvidenceError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip()) and '\x00' not in value


def _rel(value):
    _require(_text(value), 'expected relative file path')
    parts = value.split('/')
    for p in parts:
        _require(p and not p.endswith(('.', ' ')) and not re.search(r'[<>:"\\|?*\x00-\x1f]', p)
                 and not PureWindowsPath(p).is_reserved(), f'unsafe path: {value!r}')
    _require(parts[0].casefold() != '.v2-journal', 'reserved journal path')
    return value


def _path(root, rel):
    path = root / _rel(rel)
    _require(path.resolve().is_relative_to(root) and path.resolve() == path,
             f'redirected path: {rel}')
    return path


def _hash(path):
    try:
        _require(path.is_file(), f'missing or non-file: {path}')
        with path.open('rb') as f:
            before = os.fstat(f.fileno())
            h = hashlib.sha256()
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
            after = path.stat()
            _require(stat.S_ISREG(before.st_mode) and os.path.samestat(before, after)
                     and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                     f'file changed during verification: {path}')
            return h.hexdigest()
    except OSError as e:
        raise EvidenceError(f'cannot verify file: {path}') from e


def _validate(s):
    fields = 'experiment_id stage candidate_id protocol candidate inputs argv cwd budget trial units'.split()
    _require(isinstance(s, dict) and set(s) == set(fields) | ({'test_id'} if s.get('stage') == 'final' else set()),
             'invalid contract fields')
    _require('test_id' not in s or _text(s['test_id']), 'final requires a frozen test ID')
    _require(all(_text(s[k]) for k in ('experiment_id', 'stage', 'candidate_id', 'cwd')) and
             Path(s['cwd']).is_absolute() and isinstance(s['argv'], list) and s['argv'] and
             all(map(_text, s['argv'])) and isinstance(s['budget'], dict) and s['budget'] and
             type(s['trial']) is int and s['trial'] > 0, 'invalid run identity/command/budget/trial')
    inputs = s['inputs']
    _require(isinstance(inputs, dict) and all(_text(h) and re.fullmatch('[a-f0-9]{64}', h)
             for h in inputs.values()) and s['protocol'] in inputs and s['candidate'] in inputs,
             'protocol, candidate and all inputs require pinned SHA-256')
    _require(isinstance(s['units'], list) and s['units'], 'no expected units')
    paths = list(inputs)
    keys = set()
    for u in s['units']:
        _require(isinstance(u, dict) and set(u) == {'arm','prompt_id','seed','outputs','failure'}
                 and _text(u['arm']) and _text(u['prompt_id']) and type(u['seed']) is int
                 and isinstance(u['outputs'], list) and u['outputs'], 'invalid expected unit')
        _require(_key(u) not in keys, 'duplicate expected unit')
        keys.add(_key(u))
        paths.extend([*u['outputs'], u['failure']])
    names = {_rel(p).casefold() for p in paths}
    _require(len(names) == len(paths), 'file path reused across contract')
    for p in names:
        _require(not any(str(parent).replace('\\', '/') in names for parent in Path(p).parents),
                 'file/directory collision in contract')


class Journal:
    def __init__(self, root, spec, *, slot=None):
        self.root = Path(root).resolve(strict=True)
        self._spec = _json(spec)
        self._contract = json.loads(self._spec)
        _validate(self._contract)
        final = self._contract['stage'] == 'final'
        if slot is None:
            slot = 'final' if final else 'run'
        _require(re.fullmatch('[a-zA-Z0-9_-]+', _rel(slot)), 'unsafe journal slot')
        _require(slot == 'final' if final else slot.casefold() != 'final', 'reserved final slot')
        self.directory = self.root / '.v2-journal' / slot
        self._units = {_key(u): u for u in self._contract['units']}

    @contextmanager
    def _transaction(self, create=False):
        db = self.directory / 'receipts.sqlite3'
        _require(db.resolve() == db, 'redirected journal')
        uri = db.as_uri() + ('?mode=rwc' if create else '?mode=rw')
        with closing(sqlite3.connect(uri, uri=True, timeout=15)) as conn, conn:
            conn.execute('PRAGMA synchronous=FULL')
            conn.execute('BEGIN IMMEDIATE')
            if create:
                conn.execute('CREATE TABLE IF NOT EXISTS contract (id INTEGER PRIMARY KEY CHECK(id=1), spec TEXT NOT NULL)')
                conn.execute('CREATE TABLE IF NOT EXISTS units (key TEXT PRIMARY KEY NOT NULL, receipt TEXT NOT NULL, sha TEXT NOT NULL)')
                conn.execute('CREATE TABLE IF NOT EXISTS publication (id INTEGER PRIMARY KEY CHECK(id=1), marker TEXT NOT NULL, sha TEXT NOT NULL)')
                conn.execute('INSERT OR IGNORE INTO contract VALUES(1, ?)', (self._spec,))
            row = conn.execute('SELECT spec FROM contract').fetchall()
            if row != [(self._spec,)]:
                raise EvidenceError('contract differs from frozen run')
            for name, expected in self._contract['inputs'].items():
                _require(_hash(_path(self.root, name)) == expected, f'input hash mismatch: {name}')
            yield conn

    @classmethod
    def create(cls, root, spec, *, slot=None):
        journal = cls(root, spec, slot=slot)
        journal.directory.mkdir(parents=True, exist_ok=True)
        with journal._transaction(create=True):
            pass
        return journal

    def _receipt(self, key, outcome, hashes):
        _require(isinstance(key, tuple) and len(key) == 3 and type(key[2]) is int
                 and key in self._units, 'unexpected unit')
        u = self._units[key]
        _require(outcome in ('ok', 'failed'), 'invalid outcome')
        wanted, other = u['outputs'], [u['failure']]
        if outcome == 'failed':
            wanted, other = other, wanted
        _require(isinstance(hashes, dict) and set(hashes) == set(wanted), 'wrong artifact set')
        _require(not any(_path(self.root, n).exists() for n in other), 'conflicting outcome artifacts')
        for name in wanted:
            _require(_hash(_path(self.root, name)) == hashes[name], f'output hash mismatch: {name}')
        return {'contract_sha256': _digest(self._spec), 'unit': list(key),
                'outcome': outcome, 'files': hashes}

    def seal(self, key, outcome, hashes):
        """Commit immutable externally produced files against the producer's SHA-256 map."""
        with self._transaction() as conn:
            return self._seal(conn, key, outcome, hashes)

    def _seal(self, conn, key, outcome, hashes):
        receipt = self._receipt(key, outcome, hashes)
        text = _json(receipt)
        row = (_json(key), text, _digest(text))
        conn.execute('INSERT OR IGNORE INTO units VALUES(?, ?, ?)', row)
        _require(conn.execute('SELECT * FROM units WHERE key=?', row[:1]).fetchall() == [row],
                 'conflicting or corrupt existing receipt')
        return receipt

    @contextmanager
    def request(self, key):
        """Serialize cooperating producers, leaving a crash after file publication recoverable."""
        with self._transaction() as conn:
            _require(self._audit_units(conn)[0].get(key) == 'pending', 'unit is not pending')
            committed = False
            def seal(outcome, hashes):
                nonlocal committed
                _require(not committed, 'request sealed twice')
                for name, expected in self._contract['inputs'].items():
                    _require(_hash(_path(self.root, name)) == expected, f'input hash mismatch: {name}')
                receipt = self._seal(conn, key, outcome, hashes)
                committed = True
                return receipt
            yield seal
            _require(committed, 'producer exited without a receipt')

    def _audit_units(self, conn):
        states, receipts = {}, {}
        for key_text, text, sha in conn.execute('SELECT key, receipt, sha FROM units'):
            key, receipt = tuple(json.loads(key_text)), json.loads(text)
            _require(key_text == _json(key) and key not in states and _digest(text) == sha,
                     'duplicate or corrupt receipt')
            actual = self._receipt(key, receipt['outcome'], receipt['files'])
            _require(_json(actual) == text, 'receipt binding differs')
            states[key] = receipt['outcome']
            receipts[key] = sha
        for key, u in self._units.items():
            if key not in states:
                orphan = any(_path(self.root, n).exists() for n in [*u['outputs'], u['failure']])
                states[key] = 'recovery_required' if orphan else 'pending'
        return states, receipts

    def _binding(self, receipts):
        _require(receipts.keys() == self._units.keys(), 'not all expected units are sealed')
        s = self._contract
        return {**{k: s[k] for k in ('experiment_id', 'stage', 'candidate_id', 'trial')},
                **({'test_id': s['test_id']} if 'test_id' in s else {}),
                'contract_sha256': _digest(self._spec), 'protocol_sha256': s['inputs'][s['protocol']],
                'candidate_sha256': s['inputs'][s['candidate']],
                'evidence_sha256': _digest(_json(sorted(receipts.items())))}

    def report_binding(self):
        """Embed this exact object as `binding` in a JSON report alongside `decision`."""
        with self._transaction() as conn:
            return self._binding(self._audit_units(conn)[1])

    def _report(self, receipts, report, sha):
        binding = self._binding(receipts)
        data = _path(self.root, report).read_bytes()
        _require(hashlib.sha256(data).hexdigest() == sha, 'report hash mismatch')
        doc = json.loads(data)
        _require(isinstance(doc, dict) and _json(doc.get('binding')) == _json(binding)
                 and _text(doc.get('decision')), 'wrong report binding/decision')
        _require(binding['stage'] != 'final' or doc['decision'] in ('Accepted', 'Rejected', 'Blocked'),
                 'invalid final decision')
        return {'binding': binding, 'report': report, 'report_sha256': sha, 'decision': doc['decision']}

    def _publication(self, conn, receipts):
        rows = conn.execute('SELECT marker, sha FROM publication').fetchall()
        path = self.directory / 'done.json'
        _require(path.resolve() == path, 'redirected completion marker')
        if not rows:
            _require(not path.exists(), 'uncommitted completion marker')
            return None, False
        _require(len(rows) == 1, 'duplicate publication')
        text, sha = rows[0]
        _require(_digest(text) == sha, 'corrupt publication')
        saved = json.loads(text)
        expected = self._report(receipts, saved['report'], saved['report_sha256'])
        _require(_json(expected) == text, 'publication no longer matches evidence')
        exists = path.exists()
        if exists:
            _require(path.is_file() and path.read_bytes() == text.encode(), 'corrupt completion marker')
        return expected, exists

    def status(self):
        """Rehash disk evidence. Only 'pending' units may be regenerated automatically."""
        with self._transaction() as conn:
            states, receipts = self._audit_units(conn)
            marker, exists = self._publication(conn, receipts)
            return {'units': states, 'done': exists, 'publication': marker}

    def finish(self, report, sha256):
        """Latch the result once; repeat with identical report to recover a missing done.json."""
        with self._transaction() as conn:
            receipts = self._audit_units(conn)[1]
            requested = self._report(receipts, report, sha256)
            saved, exists = self._publication(conn, receipts)
            _require(saved is None or saved == requested, 'a different report is already latched')
            if exists:
                return saved
            if saved is None:
                text = _json(requested)
                conn.execute('INSERT INTO publication VALUES(1, ?, ?)', (text, _digest(text)))
        # The decision must be durable *before* a marker can become visible.
        with self._transaction() as conn:
            saved, exists = self._publication(conn, self._audit_units(conn)[1])
            _require(saved == requested, 'publication intent changed')
            if not exists:
                self._publish(saved)
            return saved

    def _publish(self, marker):
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix='.pending-')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(_json(marker).encode())
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, self.directory / 'done.json')  # atomic, never replaces existing evidence
        finally:
            os.unlink(tmp)
