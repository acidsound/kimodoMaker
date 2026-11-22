import json
import os
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2_dev_montage import EXPOSED, build, frames, sample

RUN = Path(os.environ.get('KIMODO_V2_TEST_REVIEW') or '../kimodoMaker-runs/v2-encoder-4090/english-conditioning-01/dev-split-01/dev-review')


class MontageContract(unittest.TestCase):
    def setUp(self):
        run_dir = Path(os.environ['KIMODO_RUN'])
        self.run_dir = run_dir if (run_dir / 'public').is_dir() else run_dir / 'dev-review'
        self.out = Path(os.environ['TMPDIR']) / 'montage-test'

    def test_sample_is_deterministic_unexposed_and_within_budget(self):
        ids = sample(self.run_dir, 12)
        self.assertEqual(sorted(ids), sorted(sample(self.run_dir, 12)))
        self.assertFalse(any(u.split('.')[0] in EXPOSED for u in ids))
        self.assertLessEqual(len(ids), 12)

    def test_frames_return_three_full_frames(self):
        raw = frames(self.run_dir / 'public' / f'{sample(self.run_dir, 1)[0]}.mp4', (15, 45, 75))
        self.assertEqual([len(r) for r in raw], [640 * 320 * 3] * 3)

    def test_build_produces_montages_and_empty_ballot(self):
        shutil.rmtree(self.out, ignore_errors=True)
        rows = build(self.run_dir, self.out, 3)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertTrue((self.out / r['montage']).is_file())
            self.assertTrue((self.out / 'ratings.csv').is_file())
        ballot = (self.out / 'ratings.csv').read_text(encoding='utf-8').splitlines()
        self.assertEqual(ballot[0], 'unit,a_action,a_quality,b_action,b_quality')
        self.assertEqual(len(ballot), 4)


if __name__ == '__main__':
    unittest.main()
