"""CPU-only contract for the blind dev review builder."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


class DevReviewTests(unittest.TestCase):
    def test_assignment_is_deterministic_balanced_and_secret_dependent(self):
        from v2_dev_review import assignment
        a = bytes.fromhex('a' * 64)
        b = bytes.fromhex('b' * 64)
        self.assertEqual(assignment(a, 'g1', 17), assignment(a, 'g1', 17))
        orders = {assignment(a, f'g{i}', s) for i in range(40) for s in (17, 43, 97)}
        self.assertEqual(orders, {('ref', 'candidate'), ('candidate', 'ref')})
        self.assertTrue(any(assignment(a, f'g{i}', 17) != assignment(b, f'g{i}', 17) for i in range(32)))

    def test_pair_frames_match_clip_length_and_canvas_and_skip_ended(self):
        from v2_dev_review import frames_pair, SIZE
        from v1_compare_video import Clip
        names = ('Hips', 'Head', 'LeftArm')
        clip = Clip(np.random.default_rng(1).normal(size=(12, 3, 3)).astype('float64'), names, (-1, 0, 0), 30)
        short = Clip(np.random.default_rng(2).normal(size=(5, 3, 3)).astype('float64'), names, (-1, 0, 0), 30)
        frames = list(frames_pair([clip, short]))
        self.assertEqual(len(frames), 12)
        self.assertEqual(len(frames[0]), 2 * SIZE * SIZE * 3)

    def test_build_refuses_existing_output_and_bad_rows(self):
        import tempfile
        from v2_dev_review import build
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'dev-rows.jsonl').write_text(
                ''.join(json_line(i) for i in range(63)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unexpected dev rows'):
                build(root, root / 'out')
            (root / 'dev-rows.jsonl').write_text(
                ''.join(json_line(i) for i in range(64)), encoding='utf-8')
            out = root / 'out'
            out.mkdir()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                build(root, out)


def json_line(i):
    import json
    return json.dumps(dict(group_id=f'g{i:03d}', stratum='travel_whole_body', en=f'a caption {i}')) + '\n'


if __name__ == '__main__':
    unittest.main()
