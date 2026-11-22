import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class LimbConditioningTests(unittest.TestCase):
    def test_route_requires_an_explicit_side_and_target_no_extra_actions(self):
        from importlib.util import find_spec
        self.assertIsNotNone(find_spec('v2_limb'), 'missing anatomical contrast encoder')
        from v2_limb import intent
        self.assertEqual(intent('Raise right foot beside head.'), 'right')
        self.assertEqual(intent('Lift your left leg up next to your left ear!'), 'left')
        self.assertEqual(intent('A person lifts the right foot beside the head.'), 'right')
        for text in ['Raise left hand by left ear.', 'Raise left leg by right ear.', 'Raise left leg by left ear then sit down.', 'Do not raise left leg by left ear.', 'He sits and lifts his right foot beside his head.', '', 'a person claps its hands']:
            self.assertIsNone(intent(text),text)

    def test_bounded_normalized_token_and_byte_identical_fallback(self):
        import torch
        from v2_limb import LimbEncoder, anchors
        src='Raise right foot beside head.'; control='Clap your hands.'
        pos, neg = anchors('right')
        vectors={src:[3,4,0],pos:[1,5,2],neg:[1,5,-2],control:[4,2,1]}
        def encode(batch): return torch.tensor([vectors[t] for t in batch],dtype=torch.float32)[:,None,:], [1]*len(batch)
        encoder=LimbEncoder(encode)
        orig,lens=encode([src,control]); got,newlens=encoder([src,control]); self.assertEqual(lens,newlens)
        self.assertTrue(torch.equal(orig[1],got[1])); self.assertGreater(got[0,0,2].item(),0)
        self.assertTrue(torch.allclose(orig.norm(dim=-1),got.norm(dim=-1),atol=1e-5))
        self.assertTrue(torch.isfinite(got).all())

if __name__=='__main__': unittest.main()
