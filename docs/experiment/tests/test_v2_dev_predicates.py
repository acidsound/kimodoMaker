import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2_dev_predicates import MAPPED, REGISTRY, SPEC, UNMAPPED_REASON, evaluate, p_heading_change_deg, p_net_forward

RUN = Path(os.environ.get('KIMODO_V2_TEST_RUN') or '../kimodoMaker-runs/v2-encoder-4090/english-conditioning-01/dev-split-01')


def _clip(run, arm, group, seed):
    return np.load(run / 'units' / f'{arm}.{group}.{seed}' / 'motion.npz')


class PredicateContract(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(os.environ['KIMODO_RUN'])

    def test_every_group_is_mapped_or_reasoned(self):
        groups = {p.name.split('.')[1] for p in (self.run_dir / 'units').iterdir()}
        self.assertEqual(len(groups), 64)
        unmapped = groups - MAPPED
        for group in sorted(unmapped):
            self.assertIn(group, UNMAPPED_REASON, f'group {group} neither mapped nor reasoned')

    def test_spec_entries_reference_valid_predicates_and_ops(self):
        ops = {'ge', 'le', 'abs_ge', 'abs_le'}
        for group, (label, checks) in SPEC.items():
            self.assertTrue(label)
            for predicate, args, op, limit in checks:
                self.assertIn(predicate, REGISTRY, group)
                self.assertIn(op, ops, group)
                self.assertIsInstance(limit, (int, float))

    def test_heading_calibration_holds_on_caption_controls(self):
        # Walk forward: net forward travel must be positive and large.
        z = _clip(self.run_dir, 'ref', '6b6ba74d5e716e7556659943', 17)
        self.assertGreater(p_net_forward(z['posed_joints'].astype('d'), z['global_root_heading'].astype('d')), 1.0)
        # Walk rightward: lateral net displacement must be negative (right = -X).
        j = z['posed_joints'].astype('d')
        z2 = _clip(self.run_dir, 'ref', 'bb60ab37fe7a6a595bf18b06', 17)
        j2 = z2['posed_joints'].astype('d')
        self.assertLess(j2[-1, 0, 0] - j2[0, 0, 0], -1.0)
        # Turn left caption: signed heading change must be negative.
        z3 = _clip(self.run_dir, 'ref', 'ff9f75a9f004b85840d8d93e', 17)
        turn = p_heading_change_deg(z3['posed_joints'].astype('d'), z3['global_root_heading'].astype('d'))
        self.assertLess(turn, 0, f'turn left caption produced {turn} deg')

    def test_left_right_hand_indices_come_from_rest_pose(self):
        from kimodo.skeleton import SOMASkeleton77
        neutral = SOMASkeleton77().neutral_joints.detach().cpu().numpy()
        self.assertGreater(neutral[14, 0], 0)
        self.assertLess(neutral[42, 0], 0)

    def test_evaluate_reads_both_arms_of_every_mapped_group(self):
        from v2_dev_predicates import compute
        rows = compute(self.run_dir)
        self.assertEqual({r['group'] for r in rows}, MAPPED)
        for row in rows:
            for check in row['checks']:
                for arm in ('ref', 'candidate'):
                    self.assertIn(f'{arm}_value', check)
                    self.assertIsInstance(check[f'{arm}_hit'], bool)


if __name__ == '__main__':
    unittest.main()
