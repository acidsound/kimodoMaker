"""CPU-only contract for the dev split generation; never touches the sealed set."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows(n=64):
    strata = ['travel_whole_body', 'posture_floor', 'direction_path', 'limb_gesture']
    return [dict(group_id=f'g{i:03d}', stratum=strata[i % 4], en=f'a caption number {i}') for i in range(n)]


class DevGenerateTests(unittest.TestCase):
    def test_plan_covers_64_groups_3_seeds_2_arms_exactly_once(self):
        from v2_dev_generate import plan
        jobs = plan(rows())
        self.assertEqual(len(jobs), 384)
        keys = {(j['arm'], j['group_id'], j['seed']) for j in jobs}
        self.assertEqual(len(keys), 384)
        self.assertEqual({j['seed'] for j in jobs}, {17, 43, 97})
        self.assertEqual(len({j['group_id'] for j in jobs}), 64)
        self.assertEqual(len({j['en'] for j in jobs}), 64)

    def test_plan_rejects_duplicate_or_short_splits(self):
        from v2_dev_generate import plan
        for n in (63, 65):
            with self.subTest(n=n), self.assertRaises(ValueError):
                plan(rows(n))
        with self.assertRaises(ValueError):
            plan(rows() + rows())

    def test_real_frozen_dev_rows_are_covered_exactly_once(self):
        gate = Path('../kimodoMaker-runs/v2-encoder-4090/english-conditioning-01/evaluation/dev.jsonl')
        if not gate.is_file():
            self.skipTest('frozen dev split absent')
        import json
        from v2_dev_generate import plan
        frozen = [json.loads(line) for line in gate.read_bytes().decode('utf-8-sig').splitlines()]
        jobs = plan(frozen)
        self.assertEqual(len(jobs), 384)
        self.assertEqual({j['en'] for j in jobs}, {r['en'] for r in frozen})


if __name__ == '__main__':
    unittest.main()
