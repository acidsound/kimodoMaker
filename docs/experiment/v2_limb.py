"""Experimental conditioning for an explicit unilateral foot-to-head target, never an app default."""
import re

_ART = r'(?:(?:the|your|their|his|her|one\'s) )?'
_ROUTE = re.compile(r"(?:(?:a person|the person|he|she|they) )?(?:rais(?:e|es)|lift(?:s)?) (?:up )?" + _ART +
                    r'(?P<side>left|right) (?:foot|leg)(?: up)? (?:beside|by|next to) ' + _ART +
                    r'(?:(?P<target>left|right) )?(?:head|ear)[.!?]?')


def intent(text):
    if not isinstance(text, str): return None
    match = _ROUTE.fullmatch(' '.join(text.lower().split()))
    if match and match['target'] in (None,match['side']): return match['side']
    return None


def anchors(side):
    if side not in ('left','right'): raise ValueError('unexpected side')
    def phrase(s):
        return f'A person lifts the {s} leg high to the side until the {s} foot is beside their head.'
    return phrase(side), phrase('left' if side=='right' else 'right')


class LimbEncoder:
    """Fixed residual in the unchanged LLM2Vec space; no search, labels, weights or extra motion samples."""
    def __init__(self, official): self.official = official

    def __call__(self, texts):
        import torch
        if isinstance(texts,str):
            if not intent(texts): return self.official(texts)
            out,lens=self([texts]); return out[0],lens[0]
        base,lengths = self.official(texts)
        sides = [intent(t) for t in texts]
        if not any(sides): return base,lengths
        result = base.clone()
        for side in sorted(set(filter(None,sides))):
            ab, lens = self.official(list(anchors(side)))
            if lens != [1,1] or ab.shape != (2,1,base.shape[-1]): raise ValueError('wrong anchor shape')
            mask=[i for i,s in enumerate(sides) if s==side]
            original=base[mask].float(); scale=original.norm(dim=-1,keepdim=True)
            def norm(x): return torch.nn.functional.normalize(x,dim=-1,eps=1e-12)
            pos,neg = norm(ab.float()).unbind(0)
            fused=.40*norm(original)+.60*pos+.25*(pos-neg)
            if not torch.isfinite(fused).all() or torch.any(scale==0) or torch.any(fused.norm(dim=-1)<1e-8):
                raise ValueError('degenerate condition')
            result[mask]=(scale*norm(fused)).to(base.dtype)
        return result,lengths
