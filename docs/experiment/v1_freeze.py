"""Validate and seal experiment inputs; the Dev reader never opens the final manifest."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


SPLITS = ('train', 'dev', 'test')
KINDS = (1, 2, 3, 4)
TRIAL = {'seeds': [17, 43, 97], 'steps': 100, 'postprocessing': True, 'constraints': [],
         'model': 'nvidia/Kimodo-SOMA-RP-v1.2', 'model_revision': '0e3c89024d896064b6851201373f7834784a386b',
         'student': 'google/embeddinggemma-2', 'student_revision': 'f9972ac22e696a105c482202646979e33e4b5a1e',
         'student_dtype': 'bfloat16', 'backend_revision': '809f03f4f613ce75d205cca1af2c44746f353ab4'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def norm(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def freeze(paths, assets, provenance, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if set(paths) != set(SPLITS):
        raise ValueError('Require exactly train, dev, and test manifests')
    seen = {key: {} for key in ('id', 'group', 'EN', 'KO')}
    entries = {}
    for split in SPLITS:
        path = Path(paths[split]).resolve()
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.decode('utf-8').splitlines()]
        if split != 'train' and len(rows) != 64:
            raise ValueError(f'{split} requires exactly 64 rows')
        if not rows:
            raise ValueError(f'{split} is empty')
        ids = set()
        groups = defaultdict(list)
        kinds = Counter()
        unseen = 0
        for row in rows:
            if not isinstance(row, dict) or row.get('split') != split:
                raise ValueError(f'Mislabeled {split} row')
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('id', 'group_id', 'en', 'ko')):
                raise ValueError(f'Invalid id, group, or text in {split}')
            ident = row['id']
            if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', ident) or ident in ids:
                raise ValueError(f'Unsafe or duplicate id: {ident}')
            ids.add(ident)
            for key, value in (('id', ident), ('group', row['group_id']), ('EN', norm(row['en'])), ('KO', norm(row['ko']))):
                if not value:
                    raise ValueError(f'Empty normalized {key} in {split}')
                if seen[key].setdefault(value, split) != split:
                    raise ValueError(f'{key} leakage across splits')
            if split == 'train':
                continue
            kind, seconds = row.get('kind'), row.get('seconds')
            if type(kind) is not int or kind not in KINDS:
                raise ValueError(f'Invalid kind for {ident}')
            if type(seconds) not in (int, float) or seconds <= 0 or not math.isfinite(seconds):
                raise ValueError(f'Invalid duration for {ident}')
            if any(type(row.get(k)) is not bool for k in ('unseen', 'footing')) or not isinstance(row.get('intent'), dict):
                raise ValueError(f'Missing predeclared evaluation fields for {ident}')
            groups[row['group_id']].append(row)
            kinds[kind] += 1
            unseen += row['unseen']
        if split != 'train':
            if any(kinds[k] != 16 for k in KINDS) or not unseen:
                raise ValueError(f'{split} requires 16 rows per kind and marked unseen combinations')
            for group in groups.values():
                if len(group) != 2 or group[0]['kind'] != group[1]['kind']:
                    raise ValueError(f'{split} requires two same-kind instructions per minimal pair')
                if any(norm(group[0][lang]) == norm(group[1][lang]) for lang in ('en', 'ko')):
                    raise ValueError(f'{split} contains a minimal pair with duplicate text')
        entries[split] = {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(),
                          'rows': len(rows), 'kinds': dict(kinds), 'unseen': unseen}
    document = {'schema': 'kimodo.encoder-freeze.v1', 'protocol': TRIAL, 'manifests': entries,
                'assets': {k: {'path': str(Path(v).resolve()), 'sha256': sha256(v)} for k, v in assets.items()},
                'provenance': provenance}
    with output.open('x', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    return document


def verify(lock_path, stage):
    if stage not in ('Dev', 'Frozen'):
        raise ValueError('Cannot read a manifest outside Dev or Frozen')
    manifest = json.loads(Path(lock_path).read_text(encoding='utf-8'))
    if manifest.get('schema') != 'kimodo.encoder-freeze.v1' or manifest.get('protocol') != TRIAL:
        raise ValueError('Experiment protocol drift')
    splits = ('train', 'dev') if stage == 'Dev' else ('test',)
    items = [*manifest['assets'].items(), *((k, manifest['manifests'][k]) for k in splits)]
    for label, record in items:
        if sha256(record['path']) != record['sha256']:
            raise ValueError(f'{label} SHA-256 mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for split in SPLITS:
        parser.add_argument(f'--{split}', type=Path, required=True)
    parser.add_argument('--asset', type=Path, action='append', default=[])
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    freeze({k: getattr(args, k) for k in SPLITS}, {str(p): p for p in args.asset},
           json.loads(args.provenance.read_text(encoding='utf-8')), args.out)
    print('Frozen experiment input SHA-256:', sha256(args.out))


if __name__ == '__main__':
    main()
