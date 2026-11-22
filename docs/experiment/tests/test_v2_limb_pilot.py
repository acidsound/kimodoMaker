import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

class LimbProbeTests(unittest.TestCase):
    def test_named_target_not_wrong_side_and_distance_at_same_frame(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_limb_pilot'),'missing real limb pilot')
        from v2_limb_pilot import metrics
        head=np.tile([0.,1.6,0],(20,1)); right=np.zeros_like(head); left=right.copy()
        left[7]=[.15,1.5,0]
        wrong=metrics(head,right,left)
        self.assertFalse(wrong['hit']); self.assertTrue(wrong['wrong_side_hit'])
        good=metrics(head,left,right); self.assertTrue(good['hit']); self.assertEqual(good['closest_frame'],7)
        right[7]=left[7]; self.assertFalse(metrics(head,left,right)['hit'])
        with self.assertRaises(ValueError): metrics(head,left[:1],right)

    def test_control_residual_is_measured_on_token_seen_by_motion_model(self):
        from v2_limb_pilot import Capture
        import torch
        def encoder(text): return torch.ones(len(text),1,4096),[1]*len(text)
        capture=Capture(encoder,'Walk.')
        got,_=capture(['Walk.']); got.zero_()
        np.testing.assert_array_equal(capture.token, np.ones((1,1,4096)))
        with self.assertRaises(ValueError): capture(['Wrong.'])

if __name__=='__main__': unittest.main()
