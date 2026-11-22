"""Automated motion-quality proxies for the frozen dev split; a diagnostic, never the release metrics."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HIPS, SPINE2, L_HAND, R_HAND = 0, 2, 14, 42
TOE_ENDS, FOOT_CONTACT_COLUMNS = (71, 76), (2, 5)  # L toe end, R toe end in foot_contacts
METRICS = ('foot_slide_mps', 'foot_float_m', 'hip_jitter_cm', 'trunk_wobble_deg_per_frame', 'wrist_energy_mps')


def measures(npz):
    joints = npz['posed_joints'].astype('d')
    contacts = npz['foot_contacts']
    fps, frames = 30.0, len(joints)
    roots = joints[:, HIPS]
    feet = {name: joints[:, i] for name, i in zip(('l', 'r'), TOE_ENDS)}
    out = {}
    slide, floats, contact_frames = [], [], 0
    for foot, col in zip(('l', 'r'), FOOT_CONTACT_COLUMNS):
        on = contacts[:, col].astype(bool)
        contact_frames += int(on.sum())
        floor = feet[foot][:, 1].min()
        if on.sum():
            slide.append(np.linalg.norm(np.diff(feet[foot][on], axis=0), axis=1).sum() / (on.sum() / fps))
            floats.append(float((feet[foot][on][:, 1] - floor).mean()))
    out['foot_slide_mps'] = float(np.mean(slide)) if slide else float('nan')
    out['foot_float_m'] = float(np.mean(floats)) if floats else float('nan')
    out['hip_jitter_cm'] = float(np.sqrt(np.square(np.diff(roots[:, 1], n=2)).mean()) * 100)
    trunk = joints[:, SPINE2] - roots
    trunk = trunk / np.linalg.norm(trunk, axis=1, keepdims=True)
    cosangle = np.clip((trunk[:-1] * trunk[1:]).sum(1), -1, 1)
    out['trunk_wobble_deg_per_frame'] = float(np.degrees(np.arccos(cosangle)).mean())
    wrists = np.concatenate([joints[:, L_HAND], joints[:, R_HAND]], 1)
    out['wrist_energy_mps'] = float(np.linalg.norm(np.diff(wrists, axis=0), axis=1).sum() / (frames / fps))
    out['contact_frames'] = contact_frames
    out['frames'] = frames
    return out


def _parse(unit_dir):
    arm, group, seed = unit_dir.name.split('.')
    done = json.loads((unit_dir / 'done.json').read_bytes())
    return arm, group, int(seed), done['identity']['en']


def pairs(run):
    units = {}
    for unit in sorted((run / 'units').iterdir()):
        arm, group, seed, caption = _parse(unit)
        units.setdefault((group, seed, caption), {})[arm] = unit / 'motion.npz'
    for key, arms in units.items():
        if set(arms) != {'ref', 'candidate'}:
            raise ValueError(f'pair incomplete: {key} {sorted(arms)}')
        yield key, arms


def compute(run):
    rows = []
    for (group, seed, caption), arms in pairs(run):
        measured = {}
        for arm, motion in arms.items():
            with np.load(motion) as z:
                measured[arm] = measures(z)
        if measured['ref']['frames'] != measured['candidate']['frames']:
            raise ValueError(f'frame count mismatch in {group}.{seed}')
        rows.append(dict(group=group, seed=seed, caption=caption, measures=measured))
    return rows


def summarise(rows):
    from scipy.stats import wilcoxon
    summary = {}
    for metric in METRICS:
        diffs = [r['measures']['candidate'][metric] - r['measures']['ref'][metric] for r in rows]
        diffs = [d for d in diffs if np.isfinite(d)]
        stat, p = wilcoxon(diffs) if len(diffs) > 5 else (float('nan'), float('nan'))
        summary[metric] = dict(n=len(diffs), median_diff=float(np.median(diffs)),
                               mean_diff=float(np.mean(diffs)), wilcoxon_p=float(p),
                               worse=float(np.mean([d > 0 for d in diffs])))
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = compute(a.run)
    report = dict(schema='kimodo.v2.dev-diagnostic.1', run=str(a.run.resolve()), n_pairs=len(rows),
                  metrics=METRICS, summary=summarise(rows),
                  limit='automated proxy diagnostic only; release requires the human panel')
    a.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: {m: round(v[m], 5) for m in ('n', 'median_diff', 'wilcoxon_p', 'worse')} for k, v in report['summary'].items()}, indent=1))
