import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2_dev_diagnostic import METRICS, compute, measures, pairs, summarise

RUN = Path(os.environ.get('KIMODO_V2_TEST_RUN') or '../kimodoMaker-runs/v2-encoder-4090/english-conditioning-01/dev-split-01')


class DiagnosticContract(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(os.environ['KIMODO_RUN'])

    def _fake_npz(self, path, frames=90, slide=0.0, jitter=0.0, float_height=0.0, energy=1.0):
        rng = np.random.default_rng(0)
        joints = rng.normal(0, 0.3, (frames, 77, 3))
        joints[:, 0, 1] = 0.9
        contacts = np.zeros((frames, 6), bool)
        contacts[:, 2] = True
        contacts[:, 5] = True
        for col, toe, phase in ((71, 0.0, 0.0), (76, 0.5, 1.0)):
            joints[:, col, 1] = 0.01 + float_height * (0.5 + 0.5 * np.sin(np.arange(frames) / 5.0 + phase))
        jitter = np.concatenate([[0], rng.normal(0, max(jitter, 1e-9), frames - 1)]).cumsum() if jitter else None
        if jitter is not None:
            joints[:, 0, 1] += jitter
        joints[:, 14] += np.array([energy * np.sin(t) for t in range(frames)])[:, None] / 10
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, posed_joints=joints.astype('f4'), foot_contacts=contacts)

    def test_pairs_cover_both_arms_once_per_group_seed(self):
        keys = list(pairs(self.run_dir))
        self.assertEqual(len(keys), 192)
        self.assertEqual(len({k[0][0] for k in keys}), 64)
        self.assertEqual({k[0][1] for k in keys}, {17, 43, 97})
        self.assertEqual(len({(k[0][0], k[0][1]) for k in keys}), 192)

    def test_measures_are_finite_and_ordered(self):
        with np.load(next(pairs(self.run_dir))[1]['ref']) as z:
            m = measures(z)
        for metric in METRICS:
            self.assertTrue(np.isfinite(m[metric]), metric)
        self.assertGreaterEqual(m['foot_slide_mps'], 0)
        self.assertGreaterEqual(m['foot_float_m'], 0)
        self.assertGreater(m['wrist_energy_mps'], 0)

    def _fake_pair(self, base, rel, **kw):
        folder = base / 'units' / rel
        self._fake_npz(folder / 'motion.npz', **kw)
        (folder / 'done.json').write_text(json.dumps({'identity': {'arm': rel.split('.')[0], 'key': folder.name, 'en': 'test'}}, indent=1), encoding='utf-8')

    def test_measures_detect_synthetic_defects(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._fake_pair(base, 'ref.1.17')
            self._fake_pair(base, 'candidate.1.17', jitter=0.01, float_height=0.2, energy=3.0)
            rows = compute(base)
            self.assertEqual(len(rows), 1)
            m = rows[0]['measures']
            self.assertGreater(m['candidate']['hip_jitter_cm'], m['ref']['hip_jitter_cm'])
            self.assertGreater(m['candidate']['foot_float_m'], m['ref']['foot_float_m'])
            self.assertGreater(m['candidate']['wrist_energy_mps'], m['ref']['wrist_energy_mps'])

    def test_summarise_reports_p_values_and_signs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            rng = np.random.default_rng(1)
            for i in range(12):
                self._fake_pair(base, f'ref.g{i}.17', jitter=float(rng.normal(0, 0.001)))
                self._fake_pair(base, f'candidate.g{i}.17', jitter=0.01)
            summary = summarise(compute(base))
            self.assertTrue(all(metric in summary for metric in METRICS))
            self.assertTrue(np.isfinite(summary['hip_jitter_cm']['wilcoxon_p']))
            self.assertGreaterEqual(summary['hip_jitter_cm']['worse'], 0.9)
            self.assertEqual(summary['hip_jitter_cm']['n'], 12)

    def test_compute_on_real_run_measures_all_pairs(self):
        rows = compute(self.run_dir)
        self.assertEqual(len(rows), 192)
        for row in rows:
            for arm in ('ref', 'candidate'):
                self.assertTrue(all(np.isfinite(row['measures'][arm][m]) for m in METRICS))


if __name__ == '__main__':
    unittest.main()
