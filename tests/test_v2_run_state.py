"""CPU-only contract tests; runnable directly or with unittest discovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'docs/experiment/v2_run_state.py'
sys.path.insert(0, str(MODULE.parent))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_spec(root):
    inputs = {'lock.json': b'protocol', 'candidate.json': b'candidate', 'requests.jsonl': b'requests'}
    for rel, data in inputs.items():
        (root / rel).write_bytes(data)
    return dict(experiment_id='fixture', stage='pilot', candidate_id='can-A',
                protocol='lock.json', candidate='candidate.json',
                inputs={name: digest(data) for name, data in inputs.items()},
                argv=['producer', '--cpu'], cwd=str(root), budget={'requests': 2}, trial=1,
                units=[dict(arm='ref', prompt_id='walk', seed=i, outputs=[f'u{i}/motion', f'u{i}/meta'],
                            failure=f'u{i}/error') for i in (17, 43)])


class JournalTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.spec = make_spec(self.root)
        self.key = ('ref', 'walk', 17)

    def module(self):
        self.assertTrue(MODULE.is_file(), 'no v2 journal implementation')
        import v2_run_state
        return v2_run_state

    def produce(self, seed=17, outcome='ok'):
        files = self.spec['units'][0 if seed == 17 else 1]
        names = files['outputs'] if outcome == 'ok' else [files['failure']]
        for name in names:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(name.encode())
        return {name: digest(name.encode()) for name in names}

    def test_unsealed_files_require_explicit_recovery_and_exact_producer_hashes(self):
        mod = self.module()
        j = mod.Journal.create(self.root, self.spec)
        hashes = self.produce()
        self.assertEqual(j.status()['units'][self.key], 'recovery_required')
        for key, outcome, proposed in [(self.key, 'ok', {}), (self.key, 'ok', {**hashes, 'extra': 'x'}),
                                      (self.key, 'ok', {**hashes, 'u17/meta': '0'*64}),
                                      (('ref','walk',True), 'ok', hashes), (('ref','other',17), 'ok', hashes),
                                      (self.key, 'failed', hashes), (self.key, 'arbitrary', hashes)]:
            with self.subTest(key=key, outcome=outcome, proposed=proposed), self.assertRaises(mod.EvidenceError):
                j.seal(key, outcome, proposed)
        (self.root / 'u17/meta').unlink()
        with self.assertRaises(mod.EvidenceError):
            j.seal(self.key, 'ok', hashes)
        self.assertEqual(j.status()['units'][self.key], 'recovery_required')
        self.produce()
        receipt = j.seal(self.key, 'ok', hashes)
        self.assertEqual(j.seal(self.key, 'ok', hashes), receipt)
        fkey = ('ref', 'walk', 43)
        failure = self.produce(43, 'failed')
        j.seal(fkey, 'failed', failure)
        self.assertEqual(j.status()['units'], {self.key:'ok', fkey:'failed'})
        self.assertEqual(j.seal(fkey, 'failed', failure)['outcome'], 'failed')
        self.produce(43)
        with self.assertRaises(mod.EvidenceError):
            j.seal(fkey, 'ok', {n:digest(n.encode()) for n in self.spec['units'][1]['outputs']})
        with self.assertRaises(mod.EvidenceError):
            j.status()
        for n in self.spec['units'][1]['outputs']:
            (self.root / n).unlink()
        (self.root / 'u17/meta').write_bytes(b'swapped')
        with self.assertRaises(mod.EvidenceError):
            j.status()
        (self.root / 'u17/meta').unlink()
        with self.assertRaises(mod.EvidenceError):
            j.status()

    def complete(self, journal):
        journal.seal(self.key, 'ok', self.produce())
        journal.seal(('ref','walk',43), 'failed', self.produce(43, 'failed'))

    def report(self, doc, name='report.json'):
        data = json.dumps(doc).encode()
        (self.root / name).write_bytes(data)
        return digest(data)

    def test_report_is_bound_to_exact_verified_units_and_reapplication_is_a_noop(self):
        mod = self.module()
        j = mod.Journal.create(self.root, self.spec)
        self.assertTrue(hasattr(j, 'report_binding'), 'no complete-set report binding')
        with self.assertRaises(mod.EvidenceError):
            j.report_binding()
        self.complete(j)
        binding = j.report_binding()
        for field in ('experiment_id','stage','candidate_id','candidate_sha256','protocol_sha256',
                      'evidence_sha256','contract_sha256','trial'):
            bad = {'binding': {**binding, field: 'wrong'}, 'decision': 'Rejected'}
            with self.subTest(field=field), self.assertRaises(mod.EvidenceError):
                j.finish('report.json', self.report(bad))
        doc = {'binding': binding, 'decision': 'Rejected', 'measurements': [3]}
        sha = self.report(doc)
        with self.assertRaises(mod.EvidenceError):
            j.finish('report.json', '0'*64)
        marker = j.finish('report.json', sha)
        path = j.directory / 'done.json'
        stamp = path.stat().st_mtime_ns
        self.assertEqual(json.loads(path.read_bytes()), marker)
        self.assertTrue(j.status()['done'])
        self.assertEqual(marker['decision'], 'Rejected')
        self.assertEqual(marker['binding']['trial'], 1)
        self.assertEqual(j.finish('report.json', sha), marker)
        self.assertEqual(stamp, path.stat().st_mtime_ns)
        other = self.report({**doc, 'decision': 'Accepted'}, 'other.json')
        with self.assertRaises(mod.EvidenceError):
            j.finish('other.json', other)
        path.unlink()
        self.assertFalse(j.status()['done'])
        with self.assertRaises(mod.EvidenceError):
            j.finish('other.json', other)
        self.assertEqual(j.finish('report.json', sha), marker)
        path.write_bytes(b'{')
        for operation in (j.status, lambda: j.finish('report.json', sha)):
            with self.assertRaises(mod.EvidenceError):
                operation()
        self.assertEqual(path.read_bytes(), b'{')
        path.unlink()
        j.finish('report.json', sha)
        self.report({**doc, 'measurements': [4]})
        with self.assertRaises(mod.EvidenceError):
            j.status()

    def test_final_test_identity_is_reserved_before_work_and_has_only_one_slot(self):
        mod = self.module()
        self.spec['stage'] = 'final'
        with self.assertRaises(mod.EvidenceError):
            mod.Journal.create(self.root, self.spec)
        self.spec['test_id'] = 'sealed-01'
        with self.assertRaises(mod.EvidenceError):
            mod.Journal.create(self.root, self.spec, slot='other')
        j = mod.Journal.create(self.root, self.spec)
        for over in ({'candidate_id':'B'}, {'test_id':'sealed-02'}):
            with self.assertRaises(mod.EvidenceError):
                mod.Journal.create(self.root, {**self.spec, **over})
        with self.assertRaises(mod.EvidenceError):
            mod.Journal.create(self.root, {**self.spec, 'stage':'pilot'}, slot='FINAL')
        self.complete(j)
        binding = j.report_binding()
        self.assertEqual(binding['test_id'], 'sealed-01')
        bad = self.report({'binding':binding, 'decision':'Frozen'})
        with self.assertRaises(mod.EvidenceError):
            j.finish('report.json', bad)
        good = self.report({'binding':binding, 'decision':'Blocked'})
        self.assertEqual(j.finish('report.json', good), j.finish('report.json', good))

    def test_cooperating_producer_can_only_run_pending_units_after_input_verification(self):
        mod = self.module()
        j = mod.Journal.create(self.root, self.spec)
        self.assertTrue(hasattr(j, 'request'), 'no serialized request entry point')
        with self.assertRaises(mod.EvidenceError):
            with j.request(self.key):
                pass
        with j.request(self.key) as seal:
            seal('ok', self.produce())
        with self.assertRaises(mod.EvidenceError):
            with j.request(self.key):
                self.fail('completed request repeated')
        second = ('ref','walk',43)
        with self.assertRaises(mod.EvidenceError):
            with j.request(second) as seal:
                hashes = self.produce(43, 'failed')
                (self.root / 'candidate.json').write_bytes(b'input drift')
                seal('failed', hashes)
        (self.root / 'candidate.json').write_bytes(b'candidate')
        self.assertEqual(j.status()['units'][second], 'recovery_required')
        with self.assertRaises(mod.EvidenceError):
            with j.request(second):
                self.fail('orphan overwritten')

    def test_contract_rejects_ambiguous_units_paths_or_unverified_identity(self):
        import copy
        mod = self.module()
        alterations = [lambda s: s['units'].append(s['units'][0]),
                       lambda s: s.update(units=[]), lambda s: s.update(protocol='unhashed'),
                       lambda s: s.update(candidate='unhashed'), lambda s: s.update(experiment_id=''),
                       lambda s: s['inputs'].update({'lock.json': 'A' * 64}),
                       lambda s: s['inputs'].update({'requests.jsonl': '0' * 64}),
                       lambda s: s['units'][1].update(seed=True),
                       lambda s: s['units'][1].update(outputs=['u17/Motion']),
                       lambda s: s['units'][1].update(outputs=['lock.json']),
                       lambda s: s['units'][1].update(outputs=['u43/error']),
                       lambda s: s['units'][1].update(outputs=['../escape']),
                       lambda s: s['units'][1].update(outputs=['.v2-journal/evil']),
                       lambda s: s['units'][1].update(outputs=[]),
                       lambda s: s.update(trial=True), lambda s: s.update(argv=[]),
                       lambda s: s.update(budget=None)]
        for i, change in enumerate(alterations):
            with self.subTest(case=i):
                spec = copy.deepcopy(self.spec)
                change(spec)
                with self.assertRaises(mod.EvidenceError):
                    mod.Journal.create(self.root, spec, slot=f'case{i}')
        with self.assertRaises(mod.EvidenceError):
            mod.Journal.create(self.root, self.spec, slot='../bad')
        j = mod.Journal.create(self.root, self.spec)
        (self.root / 'candidate.json').write_bytes(b'replaced')
        with self.assertRaises(mod.EvidenceError):
            j.status()

    def test_reopen_verifies_frozen_contract_and_starts_with_all_units_pending(self):
        mod = self.module()
        mod.Journal.create(self.root, self.spec)
        reopened = mod.Journal(self.root, self.spec)
        self.assertEqual(reopened.status()['units'], {self.key: 'pending', ('ref','walk',43): 'pending'})
        self.assertFalse(reopened.status()['done'])
        self.spec['argv'].append('--changed')
        with self.assertRaises(mod.EvidenceError):
            mod.Journal.create(self.root, self.spec)


if __name__ == '__main__':
    unittest.main()
