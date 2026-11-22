import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2_sweep import MAPPED, VARIANTS, evaluate_variant, plan, summarise_variant

HERE = Path(__file__).resolve().parent
INSTRUCTION_V1 = ('Represent the human movement in this caption for motion generation. '
                  'Distinguish the direction of travel relative to the person from the direction they face. '
                  'Preserve the stated action and do not infer additional actions:')


class SweepContract(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(os.environ['KIMODO_RUN'])

    def test_variants_are_fixed_and_distinct_from_v1(self):
        self.assertEqual(sorted(VARIANTS), ['contact', 'minimal'])
        for text in VARIANTS.values():
            self.assertIsInstance(text, str)
            self.assertTrue(text.endswith(':'))
            self.assertNotEqual(text, INSTRUCTION_V1)
        self.assertIn('contact', VARIANTS['contact'].lower())

    def test_plan_covers_mapped_groups_once_per_seed_per_variant(self):
        rows = [json.loads(line) for line in (self.run_dir / 'dev-rows.jsonl').read_bytes().decode('utf-8-sig').splitlines()]
        jobs = plan(rows, MAPPED)
        self.assertEqual(len(jobs), len(VARIANTS) * len(MAPPED) * 3)
        keys = {(j['variant'], j['group_id'], j['seed']) for j in jobs}
        self.assertEqual(len(keys), len(jobs))
        self.assertEqual({j['seed'] for j in jobs}, {17, 43, 97})
        self.assertTrue(all(j['group_id'] in MAPPED for j in jobs))

    def test_evaluate_variant_pairs_every_check_against_the_frozen_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            groups = sorted(MAPPED)
            from v2_dev_predicates import SPEC
            rng = np.random.default_rng(0)
            frames = 90
            heading = np.tile([1.0, 0.0], (frames, 1))
            for arm_dir, jitter in ((base / 'ref', 0.0), (base / 'cand', 0.0)):
                for group in groups:
                    for seed in (17, 43, 97):
                        joints = rng.normal(0, 0.3, (frames, 77, 3))
                        joints[:, 0, 1] = 0.9
                        joints[:, 14, 1] = 0.7  # hand raised: both_hands_up ~ finite
                        folder = arm_dir / f'ref.{group}.{seed}' if False else arm_dir / f'{arm_dir.name}.{group}.{seed}'
                        folder.mkdir(parents=True)
                        np.savez(folder / 'motion.npz', posed_joints=joints.astype('f4'),
                                 foot_contacts=np.ones((frames, 6), bool), global_root_heading=heading)
            # place the candidate arms into the sweep layout units/<variant>.<group>.<seed>
            for variant in VARIANTS:
                for group in groups:
                    for seed in (17, 43, 97):
                        src = base / 'cand' / f'cand.{group}.{seed}'
                        dst = base / 'units' / f'{variant}.{group}.{seed}'
                        dst.mkdir(parents=True, exist_ok=True)
                        np.savez(dst / 'motion.npz', posed_joints=np.load(src / 'motion.npz')['posed_joints'],
                                 foot_contacts=np.ones((frames, 6), bool), global_root_heading=heading)
            from v2_sweep import SEEDS
            rows_v = evaluate_variant(base, base / 'ref', variant='minimal')
            expected = sum(len(SPEC[g][1]) for g in groups) * 3
            self.assertEqual(len(rows_v), expected)

    def test_summarise_counts_rescue_and_loss(self):
        rows = [dict(ref_hit=True, cand_hit=True), dict(ref_hit=False, cand_hit=False),
                dict(ref_hit=False, cand_hit=True), dict(ref_hit=True, cand_hit=False)]
        s = summarise_variant(rows)
        self.assertEqual((s['n_checks'], s['ref_hits'], s['cand_hits'], s['rescue'], s['loss'], s['agree']),
                         (4, 2, 2, 1, 1, 2))


if __name__ == '__main__':
    import unittest.mock
    unittest.main()
