"""P1.4 CPU-only process crash drill. --output must name a fresh JSON file.

Its adjacent .fixtures directory is deliberately retained for inspection. Never
pass an experiment journal: this tool creates all of its inputs from literals.
"""
import argparse
from contextlib import closing, contextmanager
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys

import v2_run_state as state

KEY = ('fixture', 'request', 1)
OUTPUT = 'artifact.bin'
TIMEOUT = 30


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + '\n').encode('utf-8')


def write_new(path, data):
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def fixture(root):
    root.mkdir()
    inputs = {'protocol.json': dump({'fixture': 'protocol'}),
              'candidate.json': dump({'fixture': 'candidate'})}
    spec = dict(experiment_id='restart-drill-fixture', stage='pilot', candidate_id='fixture',
                protocol='protocol.json', candidate='candidate.json',
                inputs={name: digest(data) for name, data in inputs.items()},
                argv=['fixture-producer'], cwd=str(root), budget={'requests': 1}, trial=1,
                units=[dict(arm=KEY[0], prompt_id=KEY[1], seed=KEY[2], outputs=[OUTPUT],
                            failure='failure.json')])
    for name, data in inputs.items():
        write_new(root / name, data)
    write_new(root / 'spec.json', dump(spec))
    return state.Journal.create(root, spec), spec


def receipt_rows(journal):
    uri = (journal.directory / 'receipts.sqlite3').as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as db:
        return {tuple(json.loads(k)): (text, sha) for k, text, sha in db.execute(
            'SELECT key,receipt,sha FROM units')}


def payload(writer):
    return f'local fixture artifact from {writer}\n'.encode('ascii')


def checkpoint(point, stop, conn):
    if point == stop:
        conn.send((point, os.getpid()))
        check(conn.recv() == 'continue', 'unexpected checkpoint release')


def produce(root, spec, writer, stop, conn):
    # No command discovery, PIDs from disk, subprocess launchers or external jobs.
    with conn:
        journal = state.Journal(root, spec)
        with journal.request(KEY) as seal:
            data = payload(writer)
            stage = root / ('.staged-' + writer)
            write_new(stage, data)
            checkpoint('before_publish', stop, conn)
            check(not (root / OUTPUT).exists(), 'refusing to overwrite an artifact')
            stage.rename(root / OUTPUT)
            checkpoint('after_publish', stop, conn)
            seal('ok', {OUTPUT: digest(data)})
        conn.send(('committed', os.getpid()))


@contextmanager
def producer(root, spec, writer, stop=None):
    # spawn and the OS process HANDLE target our own Python child (including in
    # a Windows venv); killing a shell/venv launcher would not test a DB crash.
    ctx = mp.get_context('spawn')
    parent, child = ctx.Pipe()
    process = ctx.Process(target=produce, args=(root, spec, writer, stop, child))
    process.start()
    child.close()
    try:
        yield process, parent
    finally:
        parent.close()
        if process.is_alive():
            process.kill()
        process.join(TIMEOUT)
        check(not process.is_alive(), 'fixture producer did not exit')
        process.close()


def expect(worker, event):
    proc, conn = worker
    check(conn.poll(TIMEOUT), f'{proc.name} timed out waiting for {event}')
    actual = conn.recv()
    check(actual == (event, proc.pid), f'expected {event} from own producer; got {actual}')


def join_ok(worker):
    worker[0].join(TIMEOUT)
    check(worker[0].exitcode == 0, f'producer exit code {worker[0].exitcode}')


def complete(journal):
    report = dump({'binding': journal.report_binding(), 'decision': 'fixture-only'})
    write_new(journal.root / 'report.json', report)
    marker = journal.finish('report.json', digest(report))
    done = journal.directory / 'done.json'
    check(journal.status()['done'] and json.loads(done.read_bytes()) == marker, 'invalid done')
    stamp = done.stat().st_mtime_ns
    check(journal.finish('report.json', digest(report)) == marker and
          done.stat().st_mtime_ns == stamp, 'finish was not idempotent')
    return digest(done.read_bytes())


def refused(operation, reason):
    try:
        operation()
    except state.EvidenceError as exc:
        check(str(exc) == reason, f'wrong rejection: {exc!r}; expected {reason!r}')
        return str(exc)
    raise RuntimeError(f'operation did not reject: {reason}')


def retry_request(journal, key=KEY):
    with journal.request(key):
        raise RuntimeError(f'producer was admitted for {key}')


def fingerprint(path):
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns, digest(path.read_bytes())


def crash_case(root, point):
    journal, spec = fixture(root)
    with producer(root, spec, 'interrupted', point) as worker:
        expect(worker, point)
        worker[0].kill()
        worker[0].join(TIMEOUT)
        exitcode = worker[0].exitcode
        check(exitcode not in (None, 0), 'producer was not hard-killed')
    journal = state.Journal(root, spec)  # new connection, only durable state counts
    status = journal.status()
    at_restart = dict(unit=status['units'][KEY], done=status['done'],
                      artifact=(root / OUTPUT).exists(), receipt_count=int(KEY in receipt_rows(journal)))
    expected_unit = {'before_publish': 'pending', 'after_publish': 'recovery_required'}[point]
    check(at_restart == dict(unit=expected_unit, done=False, artifact=point != 'before_publish', receipt_count=0),
          f'unexpected restart state: {at_restart}')
    record = dict(killed_exitcode=exitcode, at_restart=at_restart)
    if point == 'before_publish':
        check((root / '.staged-interrupted').read_bytes() == payload('interrupted'), 'bad staging')
        with producer(root, spec, 'retry') as worker:
            expect(worker, 'committed')
            join_ok(worker)
        check((root / OUTPUT).read_bytes() == payload('retry'), 'wrong resumed output')
        record['recovery'] = 'retry_pending'
    else:
        before = fingerprint(root / OUTPUT)
        record['retry_rejected'] = refused(lambda: retry_request(journal), 'unit is not pending')
        record['wrong_hash_rejected'] = refused(lambda: journal.seal(KEY, 'ok', {OUTPUT: '0' * 64}),
                                                 f'output hash mismatch: {OUTPUT}')
        check(KEY not in receipt_rows(journal), 'incorrect hash committed')
        # Recompute from the producer fixture recipe, never bless the orphan's own hash.
        journal.seal(KEY, 'ok', {OUTPUT: digest(payload('interrupted'))})
        check(fingerprint(root / OUTPUT) == before, 'orphan rewritten')
        record['recovery'] = 'seal_original'
    record['done_sha256'] = complete(journal)
    return record


def run(output):
    output = Path(output).absolute()
    check(output.suffix == '.json' and not output.exists(), 'output must be a fresh .json file')
    workspace = output.with_suffix('.fixtures')
    workspace.mkdir()  # exclusive; never touch or clean an earlier run
    cases = {name: crash_case(workspace / name, name) for name in ('before_publish', 'after_publish')}
    result = dict(schema=1, scope='P1.4 generated journal fixtures only; not a training restart',
                  passed=True, fixtures=workspace.name, cases=cases,
                  argv=[sys.executable, '-B', str(Path(__file__).resolve()), '--output', str(output)],
                  source_sha256={p.name: digest(p.read_bytes()) for p in
                                 (Path(__file__), Path(state.__file__))})
    staged = workspace / '.result-pending'
    write_new(staged, dump(result))
    os.link(staged, output)  # publish the complete report without replacing earlier evidence
    staged.unlink()
    return {'output': str(output), 'sha256': digest(output.read_bytes())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output)))
