"""Pilot-only instruction text conditioning; no model weights or captions are edited."""

INSTRUCTION = ('Represent the human movement in this caption for motion generation. '
               'Distinguish the direction of travel relative to the person from the direction they face. '
               'Preserve the stated action and do not infer additional actions:')


class InstructionEncoder:
    """Use the official Kimodo wrapper and its underlying LLM2Vec [instruction, text] channel."""
    def __init__(self, official, allowed):
        self.official = official
        self.allowed = frozenset(allowed)

    def __call__(self, texts):
        if not isinstance(texts, list) or not texts or not all(t and t in self.allowed for t in texts):
            return self.official(texts)
        return self.official([[INSTRUCTION, text] for text in texts])
