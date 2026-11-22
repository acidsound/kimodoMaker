"""Measure conditioning-vector distances between arms; explains why instruction text cannot matter."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def l2(a, b):
    return float(np.linalg.norm(a - b))


def collect(run, prefix, groups, seeds):
    out = {}
    for group in groups:
        for seed in seeds:
            p = run / f'{prefix}.{group}.{seed}' / 'condition.npy'
            if p.is_file():
                out[(group, seed)] = np.load(p).ravel().astype('d')
    return out


def compare(a, b, name):
    keys = sorted(set(a) & set(b))
    cos = [cosine(a[k], b[k]) for k in keys]
    dists = [l2(a[k], b[k]) for k in keys]
    norms = [np.linalg.norm(a[k]) for k in keys]
    return dict(name=name, n=len(keys),
                cos_mean=float(np.mean(cos)), cos_min=float(np.min(cos)),
                l2_mean=float(np.mean(dists)), l2_max=float(np.max(dists)),
                vector_norm_mean=float(np.mean(norms)))


def run(main_run, sweep_run, out_path):
    from v2_dev_predicates import MAPPED
    seeds = (17, 43, 97)
    groups = sorted(MAPPED)
    stock = collect(main_run / 'units', 'ref', groups, seeds)
    v1 = collect(main_run / 'units', 'candidate', groups, seeds)
    contact = collect(sweep_run / 'units', 'contact', groups, seeds)
    minimal = collect(sweep_run / 'units', 'minimal', groups, seeds)
    if not (len(stock) == len(v1) == len(contact) == len(minimal) == len(groups) * len(seeds)):
        raise ValueError('collection incomplete')
    report = dict(schema='kimodo.v2.embedding-distances.1', n=len(stock),
                  signals=[compare(stock, v1, 'stock-vs-instruction-v1'),
                           compare(stock, contact, 'stock-vs-contact'),
                           compare(stock, minimal, 'stock-vs-minimal'),
                           compare(v1, contact, 'instruction-v1-vs-contact'),
                           compare(v1, minimal, 'instruction-v1-vs-minimal'),
                           compare(contact, minimal, 'contact-vs-minimal')])
    out_path.write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    for s in report['signals']:
        print(f"{s['name']:28s} n={s['n']} cos_mean={s['cos_mean']:.6f} cos_min={s['cos_min']:.6f} "
              f"l2_mean={s['l2_mean']:.4f} l2_max={s['l2_max']:.4f}")
    print(f"vector_norm_mean={report['signals'][0]['vector_norm_mean']:.4f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--main-run', type=Path, required=True)
    ap.add_argument('--sweep-run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    run(a.main_run, a.sweep_run, a.out)
