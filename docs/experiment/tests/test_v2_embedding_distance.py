import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v2_embedding_distance import collect, compare


class EmbeddingDistanceContract(unittest.TestCase):
    def setUp(self):
        self.main_run = Path(os.environ['KIMODO_RUN'])
        self.sweep_run = Path(os.environ.get('KIMODO_SWEEP', ''))
        if not (self.sweep_run / 'units').is_dir():
            self.skipTest('KIMODO_SWEEP not set')
        self.groups = ['039f75b5e790c62e4842dab5', '2e3876de9dded51dc08da02a']
        self.seeds = (17, 43, 97)

    def test_collect_reads_all_four_arms_for_shared_groups(self):
        stock = collect(self.main_run / 'units', 'ref', self.groups, self.seeds)
        v1 = collect(self.main_run / 'units', 'candidate', self.groups, self.seeds)
        contact = collect(self.sweep_run / 'units', 'contact', self.groups, self.seeds)
        minimal = collect(self.sweep_run / 'units', 'minimal', self.groups, self.seeds)
        for arm in (stock, v1, contact, minimal):
            self.assertEqual(len(arm), len(self.groups) * len(self.seeds))
        for k in stock:
            self.assertEqual(stock[k].shape, (4096,))
        same = {k: bool(np.allclose(stock[k], v1[k])) for k in stock}
        self.assertFalse(any(same.values()), 'stock and instruction-v1 must differ')

    def test_compare_reports_finite_statistics(self):
        a = {('g', 17): np.ones(4096)}
        b = {('g', 17): np.full(4096, 2.0)}
        s = compare(a, b, 'x')
        self.assertEqual(s['name'], 'x')
        self.assertEqual(s['n'], 1)
        self.assertTrue(all(np.isfinite(v) for k, v in s.items() if isinstance(v, float)))


if __name__ == '__main__':
    unittest.main()
