"""Original EN/KO motion prompts for the Kimodo adapter v1 protocol (stdlib only).

    python3 docs/experiment/v1_corpus.py --stage train-dev --out PUBLIC
    python3 docs/experiment/v1_corpus.py --stage test --out SEALED --public-out PUBLIC
    python3 docs/experiment/v1_corpus.py --stage audit --out AUDITOR_ONLY

Use new, non-overlapping directories; keep SEALED and AUDITOR_ONLY outside the
training workspace. The train/default command does not open either evaluation
recipe, and dev opens only its own. This is a curator's generator, not the Dev
pass gate; the final evaluator is responsible for access after the gate. The
checked-in test recipe is visible to repository readers; path separation and
hashes are not encryption/access control. Do not put it on a training worker.

2.5–3.5 s are authored starting durations, not pilot measurements. For a
pilot-adjusted freeze use --durations on a new dev or test output. The input is
an optional partial JSON object of pair group_id -> seconds (2.5–3.5), never a
per-arm edit. Freeze only after the separate motion pilot described in README.

Source: newly written for this project; no clips, translations of external
captions, BONES records, model output, or external datasets are used. A row is
one bilingual instruction, not an observed motion. Original clause authoring,
mechanical composition and authored lexical cue audits are not human annotation
or an independent translation review. Semantic guarantees apply to the closed
inventory and equivalences declared here, not arbitrary incoming captions.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import re
import unicodedata


SOURCE = "v1-project-authored"
KINDS = {1: "hand", 2: "turn", 3: "order", 4: "height"}
SIDES = ("left", "right")
HEIGHTS = ("lower", "rise")
LANGS = ("en", "ko")
BASE_KEYS = {"id", "group_id", "split", *LANGS}
EVAL_KEYS = BASE_KEYS | {"kind", "seconds", "unseen", "footing", "intent"}

# Each item contains two deliberately authored phrasings per locale. Keeping the
# non-target motion, setup and timing fixed is enforced at the event level, not
# by a token edit-distance heuristic (e.g. a lower/rise pair must change start pose).
HANDS = {
    "up": (("raise your {s} hand straight overhead", "reach straight up with your {s} hand"),
           ("{s}을 머리 위로 곧게 들어 올리세요", "{s}을 위로 쭉 뻗으세요")),
    "forward": (("reach forward and up with your {s} hand", "lift your {s} hand high in front of you"),
                ("{s}을 앞쪽 위로 뻗으세요", "{s}을 몸 앞에서 높이 들어 올리세요")),
    "out": (("reach out to the side and up with your {s} hand", "lift your {s} hand high out to the side"),
            ("{s}을 옆으로 뻗어 높이 드세요", "{s}을 옆쪽 위로 들어 올리세요")),
    "diagonal": (("reach diagonally forward and up with your {s} hand", "lift your {s} hand high on a forward diagonal"),
                 ("{s}을 앞쪽 대각선 위로 뻗으세요", "{s}을 앞쪽 대각선으로 높이 들어 올리세요")),
    "across": (("raise your {s} hand high across your body", "reach up across your chest with your {s} hand"),
               ("{s}을 몸 앞에서 반대편 위쪽으로 들어 올리세요", "{s}을 가슴 앞으로 가로질러 높이 뻗으세요")),
    "bent": (("lift your {s} hand beside your head with your elbow bent", "bring your {s} hand up beside your ear, keeping the elbow bent"),
             ("팔꿈치를 굽힌 채 {s}을 머리 옆까지 들어 올리세요", "팔꿈치를 굽혀 {s}을 귀 옆으로 들어 올리세요")),
    "crown": (("lift your {s} hand and rest it on the crown of your head", "place your {s} hand flat on top of your head"),
              ("{s}을 들어 정수리에 얹으세요", "{s}을 정수리에 가볍게 대세요")),
    "back": (("bring your {s} hand to the upper back of your head", "lift your {s} hand to the back of your head, near the crown"),
             ("{s}으로 뒤통수 위쪽을 짚으세요", "{s}을 들어 정수리 바로 아래 뒤통수에 대세요")),
    "wave": (("lift your {s} hand above your head and give a small wave", "wave your {s} hand once above your head"),
             ("{s}을 머리보다 높이 들어 가볍게 흔드세요", "머리보다 높은 곳에서 {s}을 한 번 흔드세요")),
}
# Shape setup (for height start) is not an extra movement. The lower/rise
# sign changes the initial level even if another action is performed first.
HEIGHT = {
    "squat": {
        "lower": (("lower into a squat", "bend your knees and sink into a squat"),
                  ("무릎을 굽혀 몸을 낮추세요", "무릎을 굽혀 쪼그려 앉으세요")),
        "rise": (("stand up from the squat", "straighten your legs and rise from the squat"),
                 ("쪼그려 앉은 자세에서 일어서세요", "쪼그려 앉은 자세에서 무릎을 펴고 일어나세요")),
    },
    "wide": {
        "lower": (("lower into a squat with your feet wide apart", "sink into a wide-stance squat"),
                  ("두 발을 넓게 벌린 채 몸을 낮추세요", "두 발을 넓게 벌린 채 무릎을 굽혀 앉으세요")),
        "rise": (("rise from the squat with your feet wide apart", "stand up from the wide-stance squat"),
                 ("두 발을 넓게 벌린 채 일어서세요", "두 발을 넓게 벌리고 쪼그려 앉은 자세에서 일어나세요")),
    },
    "split": {
        "lower": (("with one foot slightly ahead of the other, lower by bending both knees", "sink by bending both knees in a split stance"),
                  ("한 발을 살짝 앞에 둔 채 두 무릎을 굽혀 몸을 낮추세요", "발을 앞뒤로 살짝 벌린 채 두 무릎을 굽혀 몸을 낮추세요")),
        "rise": (("with one foot slightly ahead of the other, straighten both knees and rise", "stand up from the bent knees without changing your split stance"),
                 ("한 발을 살짝 앞에 둔 채 두 무릎을 펴고 일어서세요", "발을 앞뒤로 살짝 벌린 채 두 무릎을 펴고 일어나세요")),
    },
    "narrow": {
        "lower": (("lower into a squat with your feet close together", "sink into a narrow-stance squat"),
                  ("두 발을 가까이 둔 채 몸을 낮추세요", "두 발을 좁게 벌린 채 무릎을 굽혀 앉으세요")),
        "rise": (("rise from the squat with your feet close together", "stand up from the narrow-stance squat"),
                 ("두 발을 가까이 둔 채 일어서세요", "두 발을 좁게 벌리고 쪼그려 앉은 자세에서 일어나세요")),
    },
    "toes": {
        "lower": (("lower into a squat with your toes angled outward", "sink into a squat, keeping your toes pointed out"),
                  ("발끝을 바깥으로 향한 채 몸을 낮추세요", "발끝이 바깥을 향한 채 무릎을 굽혀 앉으세요")),
        "rise": (("rise from the squat with your toes angled outward", "stand up from the squat, keeping your toes pointed out"),
                 ("발끝을 바깥으로 향한 채 일어서세요", "발끝이 바깥을 향한 채 쪼그려 앉은 자세에서 일어나세요")),
    },
    "hips": {
        "lower": (("lower by sitting your hips back", "push your hips back and sink by bending your knees"),
                  ("엉덩이를 뒤로 빼며 몸을 낮추세요", "엉덩이를 뒤로 빼고 무릎을 굽혀 몸을 낮추세요")),
        "rise": (("rise from the position with your hips back", "stand up from the hips-back squat"),
                 ("엉덩이를 뒤로 뺀 자세에서 일어서세요", "엉덩이를 뒤로 빼고 낮춘 자세에서 일어나세요")),
    },
    "upright": {
        "lower": (("lower into a squat while keeping your torso upright", "sink into a squat with your torso vertical"),
                  ("상체를 세운 채 무릎을 굽혀 몸을 낮추세요", "상체를 곧게 세운 채 무릎을 굽혀 앉으세요")),
        "rise": (("rise from the squat while keeping your torso upright", "stand up from the squat with your torso vertical"),
                 ("상체를 세운 채 쪼그려 앉은 자세에서 일어서세요", "상체를 곧게 세운 채 쪼그려 앉은 자세에서 일어나세요")),
    },
    "deep": {
        "lower": (("lower into a deep squat", "sink low into a deep squat"),
                  ("무릎을 깊게 굽혀 몸을 낮추세요", "무릎을 깊게 굽혀 쪼그려 앉으세요")),
        "rise": (("stand up from the deep squat", "rise from the deep squat"),
                 ("무릎을 깊게 굽혀 앉은 자세에서 일어서세요", "무릎을 깊게 굽힌 자세에서 일어나세요")),
    },
    "lunge": {
        "lower": (("lower into a lunge with your back knee just above the floor", "sink into a lunge until your back knee nearly touches the floor"),
                  ("뒤쪽 무릎이 바닥에 거의 닿을 때까지 몸을 낮추세요", "뒤쪽 무릎이 바닥에 닿기 직전까지 몸을 낮추세요")),
        "rise": (("rise from the lunge with your back knee just above the floor", "stand up from the lunge without taking a step"),
                 ("뒤쪽 무릎이 바닥에 거의 닿은 자세에서 일어서세요", "뒤쪽 무릎이 바닥에 가까운 자세에서 일어나세요")),
    },
}
TURN = {
    "45": (("turn 45 degrees to your {s}", "turn an eighth of a circle to the {s}"),
           ("{s}쪽으로 45도 도세요", "{s}쪽으로 비스듬히 45도만 도세요")),
    "90": (("turn 90 degrees to your {s}", "make a quarter turn to the {s}"),
           ("{s}쪽으로 90도 도세요", "{s}쪽으로 4분의 1바퀴 도세요")),
    "180": (("turn 180 degrees to your {s}", "make a half turn to the {s}"),
            ("{s}쪽으로 180도 도세요", "{s}쪽으로 반 바퀴 도세요")),
}
# Style -> phrase variant, scoped pace (first movement only), separator. The
# `next` style marks both slots; the other styles still require sequential cues.
STYLES = {
    "plain": (0, "", "", ". Then ", ". 그런 다음 "),
    "slow": (0, "Slowly ", "천천히 ", ". Then ", ". 그런 다음 "),
    "quick": (0, "Quickly ", "빠르게 ", ". Then ", ". 그런 다음 "),
    "steady": (1, "At an even pace, ", "일정한 속도로 ", ". After that, ", ". 이어서 "),
    "next": (1, "", "", ". Next, ", ". 그다음 "),
    "smooth": (1, "Smoothly ", "부드럽게 ", ". Next, ", ". 그다음 "),
}
POSE = {
    "squat": ("a squat", "쪼그려 앉은 자세에서"),
    "wide": ("a squat with your feet wide apart", "두 발을 넓게 벌리고 쪼그려 앉은 자세에서"),
    "split": ("a low stance with one foot slightly ahead of the other", "한 발을 살짝 앞에 두고 두 무릎을 굽힌 자세에서"),
    "narrow": ("a squat with your feet close together", "두 발을 가까이 두고 쪼그려 앉은 자세에서"),
    "toes": ("a squat with your toes angled outward", "발끝을 바깥으로 향하게 하고 쪼그려 앉은 자세에서"),
    "hips": ("a low stance with your hips back", "엉덩이를 뒤로 빼고 몸을 낮춘 자세에서"),
    "upright": ("a squat with your torso upright", "상체를 세운 채 쪼그려 앉은 자세에서"),
    "deep": ("a deep squat", "무릎을 깊게 굽혀 앉은 자세에서"),
    "lunge": ("a low lunge with your back knee just above the floor", "두 발을 앞뒤로 벌리고 뒤쪽 무릎이 바닥에 거의 닿을 만큼 낮춘 자세에서"),
}
UPRIGHT_STANCE = {
    "wide": (" with your feet wide apart", "두 발을 넓게 벌리고 똑바로 선 자세에서 시작하세요."),
    "split": (" with one foot slightly ahead of the other", "한 발을 살짝 앞에 두고 똑바로 선 자세에서 시작하세요."),
    "narrow": (" with your feet close together", "두 발을 가까이 두고 똑바로 선 자세에서 시작하세요."),
    "toes": (" with your toes angled outward", "발끝을 바깥으로 향하게 하고 똑바로 선 자세에서 시작하세요."),
}
SUPPORT = (("Keep both feet planted.", "두 발을 바닥에 붙인 채 움직이세요."),
           ("Stay in place.", "자리를 옮기지 말고 제자리에서 움직이세요."))
OFFSIDE = ("Keep your other hand down throughout.", "반대쪽 손은 내린 채로 두세요.")
HOLD = ("Hold your final position briefly.", "마지막 자세를 잠깐 유지하세요.")


@dataclass(frozen=True)
class Event:
    type: str
    shape: str
    sign: str

    def code(self):
        return f"{self.type}:{self.shape}:{self.sign}"

    def unsigned(self):
        return f"{self.type}:{self.shape}"

    def intent(self):
        if self.type == "hand":
            return {"type": "hand", "side": self.sign}
        if self.type == "height":
            return {"type": "height", "direction": self.sign}
        if self.type == "turn" and self.shape == "90":
            return {"type": "turn", "direction": self.sign, "min_degrees": 60, "max_degrees": 120}
        raise ValueError(f"Not a scorable v1 event: {self}")

    def clause(self, lang, variant):
        column = LANGS.index(lang)
        if self.type == "height":
            return HEIGHT[self.shape][self.sign][column][variant]
        alternatives = (HANDS if self.type == "hand" else TURN)[self.shape][column]
        side = self.sign if lang == "en" else {"left": "왼", "right": "오른"}[self.sign]
        if lang == "ko" and self.type == "hand":
            side += "손"
        return alternatives[variant].format(s=side)


@dataclass(frozen=True)
class Entry:
    row: dict
    events: tuple[Event, ...]
    style: str
    target: int | None = None

    def signed(self):
        return tuple(e.code() for e in self.events)

    def equivalent(self):
        return tuple(sorted(e.unsigned() for e in self.events))

    def types(self):
        return tuple(sorted(e.type for e in self.events))


def has_type(events, type_name):
    return any(e.type == type_name for e in events)


def capital(text):
    return text[0].upper() + text[1:]


def setup(height, lang, before_turn):
    if height is None:
        return ""
    col = LANGS.index(lang)
    if height.sign == "rise":
        pos = POSE[height.shape][col]
        return f"Start in {pos}." if lang == "en" else f"{pos} 시작하세요."
    # If the turn precedes a descent, don't pre-fix a stance that must also be
    # held while stepping to turn; the descent itself names the stance.
    en, ko = UPRIGHT_STANCE.get(height.shape if not before_turn else None,
                                ("", "똑바로 선 자세에서 시작하세요."))
    return f"Start standing upright{en}." if lang == "en" else ko


def render(events, style, lang):
    variant, en_speed, ko_speed, en_sep, ko_sep = STYLES[style]
    col = LANGS.index(lang)
    speed, sep = (en_speed, en_sep) if lang == "en" else (ko_speed, ko_sep)
    height = next((e for e in events if e.type == "height"), None)
    turns = has_type(events, "turn")
    start = setup(height, lang, turns and events[0].type == "turn")
    first = "First, " if lang == "en" else "먼저 "
    acts = [e.clause(lang, variant) for e in events]
    if style == "next" and len(acts) > 1:
        speed = first
    actions = (speed + acts[0]) if speed else capital(acts[0])
    if len(acts) > 1:
        actions += sep + acts[1]
    tail = [OFFSIDE[col]] if has_type(events, "hand") else []
    return " ".join(part for part in [start, SUPPORT[turns][col], actions + ".", *tail, HOLD[col]] if part)


def make_entry(split, key, group, events, style, target=None, **evaluation):
    row = {"id": key, "group_id": group, "split": split,
           **{lang: render(events, style, lang) for lang in LANGS}, **evaluation}
    return Entry(row, events, style, target)


# Train contains all signed single actions and either order of included combinations.
TRAIN_ATOMS = {"hand": ("up", "forward", "out", "diagonal", "wave"),
               "turn": ("45", "90", "180"), "height": ("squat", "hips", "upright", "deep")}


def train_plan():
    atoms = [(kind, shape) for kind, shapes in TRAIN_ATOMS.items() for shape in shapes]
    allowed = [(item,) for item in atoms]
    omitted = Counter()
    for a, b in combinations(atoms, 2):
        if a[0] == b[0]:
            continue  # Not candidates: repeated core types are outside this plan.
        if a[0] == "turn" and b[0] == "height":
            omitted["all_turn_plus_height_combinations_reserved"] += 1
        elif b == ("turn", "90"):
            omitted["quarter_turn_plus_hand_combinations_reserved"] += 1
        elif b in {("height", "hips"), ("height", "upright")}:
            omitted["hips_back_or_upright_squat_plus_hand_reserved"] += 1
        else:
            allowed.append((a, b))
    return allowed, omitted


def build_train():
    entries, provenance = [], []
    for motif in train_plan()[0]:
        slug = "--".join("-".join(e) for e in sorted(motif))
        for order_no, order in enumerate((motif, motif[::-1]) if len(motif) == 2 else (motif,)):
            for options in product(*(HEIGHTS if k == "height" else SIDES for k, _ in order)):
                events = tuple(Event(*a, sign) for a, sign in zip(order, options))
                signed = "-".join(e.sign for e in events)
                for style in STYLES:
                    group = f"v1-train-{slug}"
                    key = f"{group}-{order_no + 1}-{signed}-{style}"
                    entry = make_entry("train", key, group, events, style)
                    entries.append(entry)
                    provenance.append({"id": key, "group_id": group, "source": SOURCE,
                                       "origin": "original bilingual clauses composed by v1_corpus.py",
                                       "external_source_ids": [], "ordered_events": entry.signed(),
                                       "equivalence_signature": entry.equivalent(), "style": style,
                                       "phrase_variant_index": STYLES[style][0],
                                       "first_event_delivery": style if STYLES[style][1] else "unspecified"})
    return entries, provenance


def valid_duration(seconds):
    return type(seconds) in (int, float) and 2.5 <= seconds <= 3.5


def evaluated_intent(kind, events, target):
    if kind == 3:
        return {"type": "order", "first": events[0].intent(), "second": events[1].intent()}
    return events[target].intent()


def build_evaluation(split, durations=None, source=None):
    if split not in ("dev", "test"):
        raise ValueError("Evaluation split must be dev or test")
    if durations is None:
        durations = {}
    if not isinstance(durations, dict) or not all(valid_duration(v) for v in durations.values()):
        raise ValueError("Use a JSON object mapping group IDs to 2.5–3.5 second durations")
    if source is None:
        source = Path(__file__).with_name(f"v1_corpus_{split}.json").read_bytes()
    recipe = json.loads(source)
    if recipe.get("source") != f"{SOURCE}-{split}":
        raise ValueError(f"Incorrect source tag for {split}")
    train_footprints = {tuple(sorted(f"{k}:{s}" for k, s in motif)) for motif in train_plan()[0]}
    entries, seen_groups = [], set()
    for name, kind, specs, style, default_seconds in recipe["pairs"]:
        group = f"v1-{split}-k{kind}-{name}"
        events = tuple(Event(*item.split(":")) for item in specs)
        stars = [i for i, e in enumerate(events) if e.sign == "*"]
        if (not re.fullmatch(r"[a-z0-9-]+", name) or group in seen_groups or type(kind) is not int
                or kind not in KINDS or style not in STYLES or not valid_duration(default_seconds)
                or len(events) not in (1, 2) or len({e.type for e in events}) != len(events)
                or (kind == 3 and (len(events) != 2 or stars))
                or (kind != 3 and (len(stars) != 1 or events[stars[0]].type != KINDS[kind]))):
            raise ValueError(f"Invalid minimal pair recipe: {group}")
        if kind == 3:
            variants, target = (events, events[::-1]), None
        else:
            target = stars[0]
            e = events[target]
            variants = [events[:target] + (Event(e.type, e.shape, sign),) + events[target + 1:]
                        for sign in (HEIGHTS if kind == 4 else SIDES)]
        footing = not has_type(events, "turn")
        unseen = len(events) > 1 and tuple(sorted(e.unsigned() for e in events)) not in train_footprints
        for suffix, variant in zip(("a", "b"), variants):
            entries.append(make_entry(split, f"{group}-{suffix}", group, variant, style, target,
                                      kind=kind, seconds=durations.get(group, default_seconds), unseen=unseen,
                                      footing=footing, intent=evaluated_intent(kind, variant, target)))
        seen_groups.add(group)
    if durations.keys() - seen_groups:
        raise ValueError(f"Duration override names unknown pair(s): {sorted(durations.keys() - seen_groups)}")
    return entries


# Independent readers of realized EN and KO clauses. These are deliberately
# not the renderer's phrases, slot values, or bilingual string substitutions.
# They audit the closed authoring vocabulary; they are not general NLP parsers.
EN_HAND_SHAPES = {
    "wave": r"\bwave\b", "back": r"\bback of your head\b", "crown": r"\bon (?:the crown of|top of) your head\b",
    "bent": r"\belbow bent\b|\bbent elbow\b", "across": r"\bacross your (?:body|chest)\b",
    "diagonal": r"\bdiagonal(?:ly)?\b", "forward": r"(?<!diagonally )\bforward and up\b|\bin front of you\b",
    "out": r"\bout to the side\b", "up": r"\b(?:straight overhead|straight up)\b",
}
KO_HAND_SHAPES = {
    "wave": r"흔드", "back": r"뒤통수", "crown": r"정수리에", "bent": r"팔꿈치를 굽힌|팔꿈치를 굽혀",
    "across": r"반대편 위쪽|가로질러", "diagonal": r"대각선", "forward": r"앞쪽 위로|몸 앞에서 높이",
    "out": r"옆으로 뻗어|옆쪽 위로", "up": r"머리 위로 곧게|위로 쭉",
}
EN_HEIGHT_SHAPES = {
    "wide": r"\bwide[- ]stance\b|\bwide apart\b", "split": r"\bone foot slightly ahead\b|\bsplit stance\b",
    "narrow": r"\bclose together\b|\bnarrow[- ]stance\b", "toes": r"\btoes (?:angled outward|pointed out)\b",
    "hips": r"\b(?:your )?hips back\b|\bhips-back\b", "upright": r"\btorso (?:vertical|upright)\b",
    "deep": r"\bdeep squat\b", "lunge": r"\blunge\b",
}
KO_HEIGHT_SHAPES = {
    "wide": r"발을 넓게 벌", "split": r"한 발을 살짝 앞|발을 앞뒤로 살짝",
    "narrow": r"발을 가까이 둔|발을 가까이 두|발을 좁게 벌", "toes": r"발끝(?:을 바깥으로|이 바깥을) 향",
    "hips": r"엉덩이를 뒤로 (?:빼|뺀)", "upright": r"상체를 (?:곧게 )?세운",
    "deep": r"무릎을 깊게 굽", "lunge": r"뒤쪽 무릎이 바닥에",
}
EN_ANGLES = {"45": r"\b45 degrees\b|\ban eighth of a circle\b",
             "90": r"\b90 degrees\b|\ba quarter turn\b",
             "180": r"\b180 degrees\b|\ba half turn\b"}
KO_ANGLES = {"45": r"(?<!\d)45도", "90": r"(?<!\d)90도|4분의 1바퀴", "180": r"(?<!\d)180도|반 바퀴"}


def read_one(patterns, clause, default=None):
    matches = [k for k, pattern in patterns.items() if re.search(pattern, clause)]
    if len(matches) == 1:
        return matches[0]
    if matches or default is None:
        raise ValueError(f"Missing/conflicting cue: {clause!r} ({matches})")
    return default


def english_event(clause):
    clause = clause.casefold()
    if re.search(r"\b(?:don't|do not|never|simultaneously|meanwhile)\b", clause):
        raise ValueError(f"Negated/concurrent English event: {clause!r}")
    hands = {s: rf"\b{s} hand\b" for s in SIDES}
    turns = {s: rf"\bto (?:your|the) {s}\b" for s in SIDES}
    if re.search(r"\b(?:turn|pivot)\b", clause):
        return Event("turn", read_one(EN_ANGLES, clause), read_one(turns, clause))
    if re.search(r"\bhand\b", clause):
        return Event("hand", read_one(EN_HAND_SHAPES, clause), read_one(hands, clause))
    level = read_one({"lower": r"\b(?:lower|sink)\b", "rise": r"\b(?:rise|stand up)\b"}, clause)
    generic = "squat" if re.search(r"\bsquat\b", clause) else None
    return Event("height", read_one(EN_HEIGHT_SHAPES, clause, generic), level)


def korean_event(clause):
    if re.search(r"하지 마|않으세요|동시에|그러면서", clause):
        raise ValueError(f"Negated/concurrent Korean event: {clause!r}")
    if re.search(r"도세요", clause):
        direction = read_one({"left": r"왼쪽으로", "right": r"오른쪽으로"}, clause)
        return Event("turn", read_one(KO_ANGLES, clause), direction)
    if re.search(r"왼손|오른손", clause):
        return Event("hand", read_one(KO_HAND_SHAPES, clause), read_one({"left": "왼손", "right": "오른손"}, clause))
    level = read_one({"lower": r"낮추세요|굽혀 앉으세요|쪼그려 앉으세요", "rise": r"일어서세요|일어나세요"}, clause)
    generic = "squat" if re.search(r"쪼그려|무릎을 굽혀 몸을", clause) else None
    return Event("height", read_one(KO_HEIGHT_SHAPES, clause, generic), level)


# Header/footer audit uses the allowlisted full stance/support phrases, not a
# semantic claim about arbitrary stance synonyms. Pace and separators below
# are independently scanned from the realized movement sentences.
PACE = {
    "en": {"slow": r"\bslowly\b", "quick": r"\bquickly\b", "steady": r"\bat an even pace\b", "smooth": r"\bsmoothly\b"},
    "ko": {"slow": "천천히", "quick": "빠르게", "steady": "일정한 속도로", "smooth": "부드럽게"},
}
ORDER_MARKERS = {
    "en": {"then": r"^Then ", "after": r"^After that, ", "next": r"^Next, "},
    "ko": {"then": r"^그런 다음 ", "after": r"^이어서 ", "next": r"^그다음 "},
}
FIRST_MARKERS = {"en": "First, ", "ko": "먼저 "}


def audit_cues(entry, lang):
    text = entry.row[lang]
    events = entry.events
    col = LANGS.index(lang)
    turns, hands = (has_type(events, typ) for typ in ("turn", "hand"))
    height = next((event for event in events if event.type == "height"), None)
    start = setup(height, lang, turns and events[0].type == "turn")
    prefix = (f"{start} " if start else "") + SUPPORT[turns][col] + " "
    suffix = (" " + OFFSIDE[col] if hands else "") + " " + HOLD[col]
    if not (text.startswith(prefix) and text.endswith(suffix)):
        raise ValueError(f"Setup, footing, offside, or end-hold cue changed: {entry.row['id']} {lang}")
    actions = text[len(prefix):-len(suffix)].removesuffix(".").split(". ")
    if len(actions) != len(events):
        raise ValueError(f"Action sentence count changed: {entry.row['id']} {lang}")
    expect_pace = entry.style if entry.style in PACE[lang] else "unspecified"
    first_pace = read_one(PACE[lang], actions[0].casefold(), "unspecified")
    if first_pace != expect_pace or any(read_one(PACE[lang], line.casefold(), "unspecified") != "unspecified"
                                       for line in actions[1:]):
        raise ValueError(f"Incorrect first-event pace: {entry.row['id']} {lang}")
    first = FIRST_MARKERS[lang]
    if actions[0].startswith(first) != (len(events) == 2 and entry.style == "next"):
        raise ValueError(f"Incorrect first marker: {entry.row['id']} {lang}")
    if len(events) == 2:
        expected = {"steady": "after", "next": "next", "smooth": "next"}.get(entry.style, "then")
        if read_one(ORDER_MARKERS[lang], actions[1]) != expected:
            raise ValueError(f"Incorrect order connector: {entry.row['id']} {lang}")
    actual = tuple((english_event if lang == "en" else korean_event)(action) for action in actions)
    if actual != events:
        raise ValueError(f"Semantic cue mismatch: {entry.row['id']} {lang}: {actual} != {events}")
    counts = Counter({"rows": 1, "setup_or_support": 1, "final_hold": 1,
                      "explicit_initial_level": int(height is not None), "offside_hand_down": int(hands),
                      "ordered_sequence": int(len(events) == 2), "explicit_speed": int(first_pace in ("slow", "quick", "steady")),
                      "smooth_delivery": int(first_pace == "smooth")})
    counts.update(channel for event in actual for channel in {"hand": ("hand_side", "hand_shape"),
                  "turn": ("turn_direction", "turn_angle"), "height": ("height_direction", "height_shape")}[event.type])
    return counts


def normalized(text):
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in folded if unicodedata.category(c)[0] in "LN")


def validate_pair(pair):
    a, b = pair
    row = a.row
    kind = row["kind"]
    if a.style != b.style or any(row[k] != b.row[k] for k in ("kind", "seconds", "unseen", "footing")):
        raise ValueError(f"Non-target pair metadata changed: {row['group_id']}")
    if kind == 3:
        okay = len(a.events) == 2 and a.events[::-1] == b.events and a.target is b.target is None
    else:
        okay = (type(a.target) is int and a.target == b.target and len(a.events) == len(b.events)
                and a.events[a.target].type == KINDS[kind]
                and [i for i, (e, f) in enumerate(zip(a.events, b.events)) if e != f] == [a.target]
                and a.events[a.target].unsigned() == b.events[b.target].unsigned()
                and {a.events[a.target].sign, b.events[b.target].sign} == set(HEIGHTS if kind == 4 else SIDES))
    if not okay or row["intent"] == b.row["intent"]:
        raise ValueError(f"Pair must change just its declared cue: {row['group_id']}")
    if kind == 3 and a.style not in ("plain", "next"):
        raise ValueError(f"Order contrast cannot change an explicitly paced action: {row['group_id']}")


def audit(sets):
    seen = {key: {} for key in ("ids", "group_ids", "normalized_en", "normalized_ko", "signed_signatures", "equivalence_signatures")}
    cues = {lang: Counter() for lang in LANGS}
    summaries = {}
    motifs, omissions = train_plan()
    train_atoms = {f"{k}:{s}" for k, shapes in TRAIN_ATOMS.items() for s in shapes}
    train_types = {tuple(sorted(k for k, _ in motif)) for motif in motifs}
    train_equiv = {tuple(sorted(f"{k}:{s}" for k, s in motif)) for motif in motifs}
    for split, entries in sets.items():
        if split not in ("train", "dev", "test") or not entries:
            raise ValueError(f"Unexpected or empty split: {split}")
        for entry in entries:
            r = entry.row
            values = (r["id"], r["group_id"], normalized(r["en"]), normalized(r["ko"]), entry.signed(), entry.equivalent())
            for (label, index), value in zip(seen.items(), values):
                if not value:
                    raise ValueError(f"Empty {label}: {r['id']}")
                if value in index:
                    prior_split, prior_group = index[value]
                    if split != prior_split:
                        raise ValueError(f"Cross-split {label} overlap at {r['id']}")
                    if label in ("ids", "normalized_en", "normalized_ko"):
                        raise ValueError(f"Duplicate {label} at {r['id']}")
                    if prior_group != r["group_id"]:
                        raise ValueError(f"Equivalent signatures must share group: {r['id']}")
                else:
                    index[value] = (split, r["group_id"])
    for split, entries in sets.items():
        groups, types, kinds = defaultdict(list), Counter(), Counter()
        per_lang, seen_styles = {lang: Counter() for lang in LANGS}, Counter()
        atoms, signed, unsigned = set(), set(), set()
        unseen = unseen_types = known_atom_novel = support = 0
        for entry in entries:
            r = entry.row
            if r["split"] != split or set(r) != (BASE_KEYS if split == "train" else EVAL_KEYS):
                raise ValueError(f"Unexpected row schema or split: {r['id']}")
            if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", r[k]) for k in ("id", "group_id")):
                raise ValueError(f"Invalid ID: {r['id']}")
            for lang in LANGS:
                per_lang[lang].update(audit_cues(entry, lang))
            groups[r["group_id"]].append(entry)
            types["+".join(entry.types())] += 1
            seen_styles[entry.style] += 1
            atoms.update(e.unsigned() for e in entry.events)
            signed.add(entry.signed())
            unsigned.add(entry.equivalent())
            if split == "train":
                continue
            kinds[r["kind"]] += 1
            is_novel = len(entry.events) > 1 and entry.equivalent() not in train_equiv
            if (type(r["kind"]) is not int or not valid_duration(r["seconds"]) or r["unseen"] is not is_novel
                    or r["footing"] is not ("turn" not in entry.types())
                    or r["intent"] != evaluated_intent(r["kind"], entry.events, entry.target)
                    or any(e.type == "turn" and e.shape != "90" for e in entry.events)):
                raise ValueError(f"Invalid v1 evaluation annotation: {r['id']}")
            unseen += is_novel
            unseen_types += len(entry.events) > 1 and entry.types() not in train_types
            known_atom_novel += is_novel and all(e.unsigned() in train_atoms for e in entry.events)
            support += r["footing"]
        if any(len({entry.equivalent() for entry in group}) != 1 for group in groups.values()):
            raise ValueError(f"Mixed equivalence signature in a {split} group")
        info = {"rows": len(entries), "groups": len(groups), "source": f"{SOURCE}-{split}" if split != "train" else SOURCE,
                "unique_signed_signatures": len(signed), "equivalence_signatures": len(unsigned),
                "unsigned_atoms": sorted(atoms), "action_type_combinations": types, "styles": seen_styles, "cue_checks": per_lang}
        if split != "train":
            if kinds != dict.fromkeys(KINDS, 16) or len(groups) != 32 or not (known_atom_novel and unseen_types):
                raise ValueError(f"{split} needs 32 balanced pairs and held-out combinations, including novel combinations of known atoms")
            for pair in groups.values():
                if len(pair) != 2:
                    raise ValueError(f"Expected exactly two members: {pair[0].row['group_id']}")
                validate_pair(pair)
            info.update(kinds=kinds, pairs_per_kind={k: n // 2 for k, n in kinds.items()}, unseen_combination_rows=unseen,
                        unseen_type_combinations=unseen_types, known_atom_novel_combinations=known_atom_novel,
                        static_footing_rows=support, turning_rows=len(entries) - support)
        for lang in LANGS:
            cues[lang].update(per_lang[lang])
        summaries[split] = info
    report = {"schema": "kimodo.project-corpus-audit.v1", "splits": summaries, "cue_checks": cues,
              "limitations": ["Closed inventory; no independent human linguistic review or motion measurement.",
                              "Lexical cue tests do not measure physical fidelity; finer shapes, pace and non-target events are not v1 scorer targets.",
                              "Equivalence ignores phrase, mirror, pace, height sign and order within an explicit action-shape family.",
                              "No third-party captions, external clip IDs, model-generated prompts or motion targets."]}
    if len(sets) > 1:
        report["cross_split_overlaps"] = {k: 0 for k in seen if k != "ids"}
    if "train" in sets:
        report["train_considered"] = {"candidate_motifs": len(motifs) + sum(omissions.values()), "selected_motifs": len(motifs),
                                     "omitted_motifs_by_reason": omissions,
                                     "rows_excluded_by_motif_policy_before_rendering": sum(omissions.values()) * 8 * len(STYLES),
                                     "other_not_considered": ["External datasets and translations; no access or license assumed.",
                                                              "Repeated action types, jumps, locomotion, objects and scene constraints."],
                                     "evaluation_pairs_considered": {s: 32 for s in summaries if s != "train"},
                                     "evaluation_pairs_dropped": {s: 0 for s in summaries if s != "train"}}
    return report


RULES = {
    "protocol": "EncoderTrial v1 / docs/experiment/README.md sections 2–3",
    "seeds": [17, 43, 97], "steps": 100, "postprocessing": True, "external_constraints": [],
    "reference_frame": "actor-relative; SOMA-77 world coordinates in meters",
    "window_seconds": 0.2, "window_frames": "ceil(0.2 * fps)",
    "hand": {"wrist_above_same_side_shoulder_m": 0.10, "other_wrist": "at or below same-side shoulder"},
    "turn": {"min_degrees": 60, "max_degrees": 120,
             "terminal_delta": "last window mean unwrapped horizontal pelvis yaw minus first frame"},
    "height": {"min_pelvis_delta_m": 0.20, "end_stability_m": 0.05,
               "endpoint_estimator": "median pelvis height in first/last windows"},
    "order": {"min_first_event_gap_seconds": 0.20,
              "early_events": "use first occurrence for each event; do not skip an early wrong-order onset; missing, reversed or simultaneous fail"},
    "quality": {"joints": 77, "finite": True, "frames": "expected from supplied duration and generation fps",
                "floor_level_m": 0, "penetration_below_m": -0.05, "max_penetrating_frame_fraction": 0.02,
                "footing_contact_height_m": 0.06, "footing_max_slip_m": 0.10,
                "footing": "applies only to no-turn rows; each maximal contact window, at least one qualifying foot"},
}


def json_bytes(value, *, lines=False):
    def line(item):
        return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    text = "".join(map(line, value)) if lines else json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return text.encode("utf-8")


def read_json(blob):
    def no_duplicate_fields(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("JSON contains a duplicate key")
        return fields
    return json.loads(blob, object_pairs_hook=no_duplicate_fields)


def audit_pilot(blob, sets):
    """Read the separate pilot's literal prompt list without importing it."""
    tree = ast.parse(blob.decode("utf-8-sig"))
    lists = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(n, ast.Name) and n.id == "prompts" for n in node.targets)]
    if len(lists) != 1 or not lists[0]:
        raise ValueError("Expected one nonempty literal 'prompts' list in the pilot script")
    keys = {normalized(text) for _, text in lists[0]}
    if len(keys) != len(lists[0]) or "" in keys:
        raise ValueError("Duplicate or empty pilot prompt")
    for split in ("dev", "test"):
        if keys & {normalized(entry.row["en"]) for entry in sets[split]}:
            raise ValueError(f"Normalized English prompt overlap with the separate motion pilot: {split}")
    return {"sha256": sha256(blob).hexdigest(), "pilot_english_prompts_compared": len(keys),
            "normalized_english_eval_overlaps": {"dev": 0, "test": 0},
            "method": "AST literal extraction only; no pilot outputs used to tune templates"}


def disjoint(a, b):
    a, b = a.resolve(), b.resolve()
    return a != b and a not in b.parents and b not in a.parents


def emit(args):
    out = args.out
    if out.exists():
        raise ValueError(f"Output already exists; refusing to overwrite a bundle: {out}")
    if args.stage == "test":
        if not args.public_out or not disjoint(out, args.public_out) or not disjoint(out, Path(__file__).parent):
            raise ValueError("Test --out must be distinct from, outside and not contain --public-out or the source directory")
    elif args.public_out:
        raise ValueError("--public-out is only for the curator's test export")
    if args.durations and args.stage not in ("dev", "test"):
        raise ValueError("--durations requires a standalone dev or test stage")
    if args.pilot_source and args.stage != "audit":
        raise ValueError("--pilot-source is only for the curator's all-source audit")

    entries, blobs = {}, {}
    sources = {"v1_corpus.py": sha256(Path(__file__).read_bytes()).hexdigest()}
    if args.stage in ("train", "train-dev", "audit"):
        entries["train"], provenance = build_train()
        if args.stage != "audit":
            blobs["train_provenance.jsonl"] = (json_bytes(provenance, lines=True), len(provenance))
    for split in ("dev", "test"):
        if args.stage != split and args.stage != "audit" and not (args.stage == "train-dev" and split == "dev"):
            continue
        source_path = Path(__file__).with_name(f"v1_corpus_{split}.json")
        raw = source_path.read_bytes()
        sources[source_path.name] = sha256(raw).hexdigest()
        durations = None
        if args.durations:
            duration_blob = args.durations.read_bytes()
            durations = read_json(duration_blob)
            if not isinstance(durations, dict):
                raise ValueError("Duration override must be a JSON object")
            sources["duration_overlay"] = sha256(duration_blob).hexdigest()
        entries[split] = build_evaluation(split, durations, raw)
    report = audit(entries)
    if args.stage == "audit":
        if args.pilot_source:
            report["separate_pilot_comparison"] = audit_pilot(args.pilot_source.read_bytes(), entries)
        blobs["audit.json"] = (json_bytes(report), None)
    else:
        for split, records in entries.items():
            blobs[f"{split}.jsonl"] = (json_bytes([r.row for r in records], lines=True), len(records))
    manifest = {"schema": "kimodo.project-corpus-manifest.v1", "stage": args.stage, "sources_sha256": sources,
                "files": {name: {"sha256": sha256(blob).hexdigest(), "bytes": len(blob),
                                 **({"rows": count} if count is not None else {})}
                          for name, (blob, count) in blobs.items()},
                "audit": report}
    if args.stage != "train":
        manifest["rules"] = RULES
    # Validate before writing, refuse in-place updates, and write manifest last
    # so a partial output never appears as a completed freeze.
    out.mkdir(parents=True, exist_ok=False)
    for name, (blob, _) in blobs.items():
        with (out / name).open("xb") as file:
            file.write(blob)
    manifest_blob = json_bytes(manifest)
    with (out / "manifest.json").open("xb") as file:
        file.write(manifest_blob)
    print(f"{args.stage}: " + ", ".join(f"{k}={len(records)}" for k, records in entries.items()) +
          f"; manifest SHA-256 {sha256(manifest_blob).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("train", "dev", "train-dev", "test", "audit"), default="train")
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--public-out", type=Path, help="Public bundle path, required for test path separation")
    parser.add_argument("--durations", type=Path, help="Optional JSON file of adjusted pair durations, dev or test only")
    parser.add_argument("--pilot-source", type=Path, help="Optional separate pilot Python script to compare literal English prompts in audit stage")
    args = parser.parse_args()
    try:
        emit(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
