"""Diagnostic distance is measured at one physical frame, not per-axis minima."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class AppProbeTests(unittest.TestCase):
    def test_distance_and_height_come_from_same_frame(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_app_probe'), 'Missing app joint probe')
        from v2_app_probe import gap
        target = np.zeros((3, 3))
        observed = np.array([[0, -1, 0], [.03, -.04, 0], [1, 0, 1]])
        stats = gap(observed, target)
        self.assertEqual(stats['closest_frame'], 1)
        self.assertAlmostEqual(stats['closest_m'], .05)
        self.assertAlmostEqual(stats['dy_at_closest_m'], -.04)
        self.assertAlmostEqual(stats['max_dy_m'], 0)

    def test_mismatched_frames_do_not_broadcast(self):
        from v2_app_probe import gap
        with self.assertRaises(ValueError):
            gap(np.ones((5, 3)), np.ones((1, 3)))
        with self.assertRaises(ValueError):
            gap(np.full((5, 3), np.nan), np.zeros((5, 3)))

    @unittest.skipUnless(os.environ.get('KIMODO_V2_VERIFY_BROWSER'), 'requires local take and VRM fixtures')
    def test_real_soma_and_vrm_per_frame_evidence(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'evidence'
            subprocess.run([sys.executable, '-B', str(root / 'docs/experiment/v2_app_probe.py'),
                            '--vrm', str(root.parent / 'kimodoMaker-data/fixtures/Seed-san.vrm'),
                            '--out', str(out), str(root / 'backend/takes/5705ce5f10a7425eabed3d28cc18c3ee')], check=True)
            self.assertTrue((out / 'report.json').is_file(), 'missing end-to-end pose evidence')
            record = json.loads((out / 'report.json').read_bytes())['takes'][0]
            self.assertEqual(record['frames'], 120)
            self.assertEqual(set(record['measurements']), {'soma', 'vrm'})
            self.assertEqual(record['downloads_checked'], 3)


if __name__ == '__main__':
    unittest.main()
