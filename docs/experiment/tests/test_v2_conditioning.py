"""Verify the instruction channel without loading weights in unit tests."""
import unittest
import importlib.util


class ConditioningTest(unittest.TestCase):
    def test_official_boundary_receives_pairs_not_rewritten_captions(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_conditioning'), 'missing conditioning implementation')
        from v2_conditioning import InstructionEncoder, INSTRUCTION
        seen = []
        def official(batch):
            seen.append(batch)
            return 'official-result'
        result = InstructionEncoder(official, {'Shuffle to your left.'})(['Shuffle to your left.'])
        self.assertEqual(result, 'official-result')
        self.assertEqual(seen, [[[INSTRUCTION, 'Shuffle to your left.']]])
        self.assertTrue(INSTRUCTION.strip())

    def test_out_of_scope_null_or_mixed_batch_is_unmodified(self):
        from v2_conditioning import InstructionEncoder
        wrapper = InstructionEncoder(lambda x:x, {'walk'})
        for texts in ([''], ['unseen'], ['walk', 'unseen'], 'unseen'):
            with self.subTest(texts=texts): self.assertIs(wrapper(texts), texts)
