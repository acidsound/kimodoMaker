import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import v1_freeze


def write(path, rows):
    path.write_text(''.join(json.dumps(x) + '\n' for x in rows), encoding='utf-8')


class FreezeTests(unittest.TestCase):
    def fixture(self, home):
        paths = {name: home / (name + '.jsonl') for name in ('train', 'dev', 'test')}
        rows = {'train': [{'id': 'train-only', 'group_id': 'group-train-only', 'split': 'train', 'en': 'Walk', 'ko': '걷는다'}]}
        for split in ('dev', 'test'):
            rows[split] = []
            for kind in (1, 2, 3, 4):
                for i in range(16):
                    ident = f'{split}-{kind}-{i}'
                    rows[split].append({'id': ident, 'group_id': f'{split}-{kind}-pair-{i // 2}',
                                        'split': split, 'kind': kind, 'seconds': 3.0,
                                        'unseen': i == 0, 'footing': True, 'intent': {'type': 'hand'},
                                        'en': f'en {ident}', 'ko': f'ko {ident}'})
        for name, path in paths.items():
            write(path, rows[name])
        return paths, rows

    def test_freeze_rejects_cross_split_semantic_groups_and_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, rows = self.fixture(root)
            for row in rows['dev'][:2]:
                row['group_id'] = 'group-train-only'
            write(paths['dev'], rows['dev'])
            with self.assertRaisesRegex(ValueError, 'group.*leak'):
                v1_freeze.freeze(paths, {}, {}, root / 'lock.json')
            paths, rows = self.fixture(root)
            rows['dev'][0]['en'] = ' WALK! '
            write(paths['dev'], rows['dev'])
            with self.assertRaisesRegex(ValueError, 'EN.*leak'):
                v1_freeze.freeze(paths, {}, {}, root / 'lock.json')

    def test_dev_never_reads_sealed_test_and_detects_asset_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = self.fixture(root)
            code = root / 'code.py'
            code.write_text('a = 1', encoding='utf-8')
            lock = root / 'lock.json'
            doc = v1_freeze.freeze(paths, {'source': code}, {'counts': {'original': 1}}, lock)
            self.assertEqual(doc['manifests']['test']['rows'], 64)
            self.assertEqual(doc['manifests']['dev']['sha256'], hashlib.sha256(paths['dev'].read_bytes()).hexdigest())
            paths['test'].unlink()
            self.assertEqual(v1_freeze.verify(lock, 'Dev')['manifests']['dev']['rows'], 64)
            with self.assertRaises(FileNotFoundError):
                v1_freeze.verify(lock, 'Frozen')
            code.write_text('a = 2', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                v1_freeze.verify(lock, 'Dev')

    def test_malformed_pair_count_is_never_locked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, rows = self.fixture(root)
            rows['test'].pop()
            write(paths['test'], rows['test'])
            lock = root / 'lock.json'
            with self.assertRaisesRegex(ValueError, '64 rows'):
                v1_freeze.freeze(paths, {}, {}, lock)
            self.assertFalse(lock.exists())


if __name__ == '__main__':
    unittest.main()
