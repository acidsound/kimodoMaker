"""CPU-only crash drill; every journal below lives in a newly created fixture."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'v2_restart_drill.py'


class RestartDrillTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='restart-drill-test-')
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name) / 'restart-drill-journal.json'
        self.fixtures = self.out.with_suffix('.fixtures')

    def run_drill(self):
        self.assertTrue(SCRIPT.is_file(), 'missing restart drill CLI')
        return subprocess.run([sys.executable, '-B', str(SCRIPT), '--output', str(self.out)],
                              capture_output=True, text=True, timeout=120)

    def verified_cases(self):
        proc = self.run_drill()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['sha256'],
                         hashlib.sha256(self.out.read_bytes()).hexdigest())
        doc = json.loads(self.out.read_bytes())
        self.assertIs(doc['passed'], True)
        return doc['cases']

    def receipts(self, case):
        directory = self.fixtures / case / '.v2-journal/run'
        with closing(sqlite3.connect((directory / 'receipts.sqlite3').as_uri() + '?mode=ro',
                                     uri=True)) as db:
            rows = db.execute('SELECT receipt,sha FROM units').fetchall()
            for text, sha in rows:
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), sha)
            return directory, [json.loads(text) for text, _ in rows]

    def test_before_publish_hard_kill_leaves_pending_and_only_retry_commits(self):
        name = 'before_publish'
        case = self.verified_cases()[name]
        self.assertNotEqual(case['killed_exitcode'], 0)
        self.assertEqual(case['at_restart'], {'unit': 'pending', 'receipt_count': 0,
                                              'artifact': False, 'done': False})
        directory, rows = self.receipts(name)
        good, = [row for row in rows if row['outcome'] == 'ok']
        path = self.fixtures / name / 'artifact.bin'
        self.assertIn(b'retry', path.read_bytes())
        self.assertEqual(good['files'], {'artifact.bin': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.assertEqual(json.loads((directory / 'done.json').read_bytes())['report_sha256'],
                         hashlib.sha256((self.fixtures / name / 'report.json').read_bytes()).hexdigest())

    def test_orphan_is_never_regenerated_and_requires_independent_producer_hash(self):
        name = 'after_publish'
        cases = self.verified_cases()
        self.assertIn(name, cases)
        case = cases[name]
        self.assertNotEqual(case['killed_exitcode'], 0)
        self.assertEqual(case['at_restart'], {'unit': 'recovery_required', 'receipt_count': 0,
                                              'artifact': True, 'done': False})
        self.assertEqual(case['recovery'], 'seal_original')
        self.assertEqual(case['retry_rejected'], 'unit is not pending')
        self.assertEqual(case['wrong_hash_rejected'], 'output hash mismatch: artifact.bin')
        _, rows = self.receipts(name)
        good, = [row for row in rows if row['outcome'] == 'ok']
        data = (self.fixtures / name / 'artifact.bin').read_bytes()
        self.assertIn(b'interrupted', data)
        self.assertEqual(good['files'], {'artifact.bin': hashlib.sha256(data).hexdigest()})
        self.assertEqual(case['done_sha256'], hashlib.sha256(
            (self.fixtures / name / '.v2-journal/run/done.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
