"""CPU-only tests for the experiment's NumPy scorer and TLA+ port."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


def load_scorer():
    source = Path(__file__).resolve().parents[1] / "v1_score.py"
    if not source.is_file():
        raise AssertionError("v1_score.py has not been implemented")
    spec = importlib.util.spec_from_file_location("v1_score", source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    return scorer


def good_report():
    arms = ("ref", "en", "ko")
    return {
        "hits": {a: [42] * 4 for a in arms},
        "usable": dict.fromkeys(arms, 181),
        "peak": {"ref": 20000, "en": 16000, "ko": 16000},
        "p95": {"ref": 1000, "en": 1200, "ko": 1200},
        "evidence": True,
        "standalone": True,
        "exports": True,
    }


# Scored bone names come from SOMASkeleton77; placeholder names fill unused
# joints. The deliberately shuffled order prevents reliance on fixed indexes.
BONES = ("RightToeEnd", "RightFoot", "LeftShoulder", "Hips", "RightHand",
         "LeftToeEnd", "LeftFoot", "RightShoulder", "LeftHand") + tuple(f"Unused{i}" for i in range(68))
LEFT = {"type": "hand", "side": "left"}
RIGHT = {"type": "hand", "side": "right"}
TURN_LEFT = {"type": "turn", "direction": "left", "min_degrees": 60, "max_degrees": 120}
TURN_RIGHT = {**TURN_LEFT, "direction": "right"}
LOWER = {"type": "height", "direction": "lower"}
RISE = {"type": "height", "direction": "rise"}


def y(motion, bone):
    return motion["posed_joints"][:, BONES.index(bone), 1]


def clip(frames=30):
    j = np.zeros((frames, 77, 3), dtype=np.float32)
    m = {"posed_joints": j}
    y(m, "Hips")[:] = 1.0
    for side in ("Left", "Right"):
        y(m, f"{side}Shoulder")[:] = 1.2
        y(m, f"{side}Hand")[:] = 1.0
        y(m, f"{side}Foot")[:] = 0.1
    return m


def apply_yaw(motion, degrees):
    radians = np.deg2rad(degrees)
    c, s = np.cos(radians), np.sin(radians)
    rotations = np.broadcast_to(np.eye(3, dtype=np.float32), (len(degrees), 77, 3, 3)).copy()
    hips = rotations[:, BONES.index("Hips")]
    hips[:, 0, 0], hips[:, 2, 2] = c, c
    hips[:, 0, 2], hips[:, 2, 0] = s, -s
    motion["global_rot_mats"] = rotations


def order(first, second):
    return {"type": "order", "first": first, "second": second}


class ClipTests(unittest.TestCase):
    def setUp(self):
        self.v1 = load_scorer()

    def run_clip(self, motion, intent=LEFT, fps=21, expected=30, sign=1, footing=True):
        return self.v1.score_clip(motion, BONES, fps, expected, intent, sign, footing)

    def test_hand_window_uses_ceil_and_requires_opposite_hand_down(self):
        m = clip()
        y(m, "LeftHand")[4:8] = 1.31  # At 21 Hz: 4 is too few, 5 passes.
        self.assertFalse(self.run_clip(m)["hit"])
        y(m, "LeftHand")[8] = 1.31
        y(m, "RightHand")[4:9] = 1.2  # Equality is allowed.
        result = self.run_clip(m)
        self.assertEqual((result["hit"], result["usable"], result["window_frames"]), (True, True, 5))
        self.assertEqual(result["intent"]["onset_frame"], 4)
        self.assertFalse(self.run_clip(m, RIGHT)["hit"])
        y(m, "RightHand")[6] = 1.201
        self.assertFalse(self.run_clip(m)["hit"])

    def test_mirrored_wrists_flip_side_at_ten_centimeters(self):
        m = clip()
        y(m, "RightShoulder")[:] = 0.5
        y(m, "RightHand")[:] = 0.6
        self.assertTrue(self.run_clip(m, RIGHT)["hit"])
        self.assertFalse(self.run_clip(m, LEFT)["hit"])

    def test_bad_generation_fails_both_scores(self):
        m = clip()
        for bad in ({}, {"posed_joints": m["posed_joints"][None]},
                    {"posed_joints": m["posed_joints"][:, :-1]}, clip(29)):
            score = self.run_clip(bad)
            self.assertEqual((score["hit"], score["usable"]), (False, False))
            self.assertTrue(score["quality"]["reasons"])
        y(m, "Hips")[0] = np.nan
        self.assertEqual(self.run_clip(m)["quality"]["reasons"], ["nonfinite_posed_joints"])

    def test_penetration_counts_frames_and_includes_ankles(self):
        m = clip(100)
        y(m, "LeftHand")[:] = 1.4
        y(m, "RightFoot")[:2] = -0.051
        y(m, "LeftToeEnd")[:2] = -0.07
        y(m, "LeftFoot")[2] = -0.05  # At, not below, the boundary.
        result = self.run_clip(m, expected=100)
        self.assertTrue(result["usable"])
        self.assertEqual(result["quality"]["penetration_frames"], 2)
        y(m, "RightToeEnd")[3] = -0.051
        result = self.run_clip(m, expected=100, footing=False)
        self.assertFalse(result["usable"])
        self.assertTrue(result["hit"])
        self.assertIn("floor_penetration", result["quality"]["reasons"])

    def test_each_maximum_toe_run_is_checked_from_its_first_position(self):
        m = clip()
        y(m, "RightToeEnd")[:] = 0.061
        y(m, "LeftToeEnd")[:] = 0.061
        y(m, "LeftToeEnd")[2:7] = 0.06
        y(m, "LeftToeEnd")[10:20] = 0.06
        toe = m["posed_joints"][:, BONES.index("LeftToeEnd"), :]
        toe[6, 0] = 0.1
        toe[14, 2] = 0.101  # Late slide and return within second planted run.
        result = self.run_clip(m)
        self.assertFalse(result["usable"])
        self.assertIn("foot_slide", result["quality"]["reasons"])
        self.assertEqual([(r["start_frame"], r["end_frame_exclusive"]) for r in result["quality"]["contact_runs"]],
                         [(2, 7), (10, 20)])
        toe[14, 2] = 0.0
        self.assertTrue(self.run_clip(m)["usable"])

    def test_no_contact_fails_only_when_footing_is_required(self):
        m = clip()
        for side in ("Left", "Right"):
            y(m, f"{side}ToeEnd")[:] = 0.07
            y(m, f"{side}ToeEnd")[:4] = 0.0
        self.assertIn("no_toe_contact", self.run_clip(m)["quality"]["reasons"])
        self.assertTrue(self.run_clip(m, footing=False)["usable"])

    def test_turn_unwraps_a_branch_crossing_and_uses_calibrated_sign(self):
        m = clip()
        degrees = 160 + np.minimum(np.arange(30) * 6, 90)
        apply_yaw(m, degrees)
        result = self.run_clip(m, TURN_LEFT)
        self.assertTrue(result["hit"])
        self.assertAlmostEqual(result["intent"]["delta_degrees"], 90, places=5)
        self.assertFalse(self.run_clip(m, TURN_RIGHT)["hit"])
        self.assertTrue(self.run_clip(m, TURN_RIGHT, sign=-1)["hit"])
        apply_yaw(m, -degrees)
        self.assertTrue(self.run_clip(m, TURN_RIGHT)["hit"])

    def test_turn_uses_tail_mean_not_last_frame_or_peak(self):
        m = clip()
        degrees = np.zeros(30)
        degrees[8:14] = 90
        degrees[-1] = 90
        apply_yaw(m, degrees)
        self.assertFalse(self.run_clip(m, TURN_LEFT)["hit"])
        for angle, hit in ((59.99, False), (60, True), (120, True), (120.01, False), (450, False)):
            with self.subTest(angle=angle):
                apply_yaw(m, np.r_[np.linspace(0, angle, 25), [angle] * 5])
                self.assertEqual(self.run_clip(m, TURN_LEFT)["hit"], hit)

    def test_turn_without_defined_pelvis_axis_has_no_hit(self):
        m = clip()
        self.assertFalse(self.run_clip(m, TURN_LEFT)["hit"])
        apply_yaw(m, np.zeros(30))
        m["global_rot_mats"][0, BONES.index("Hips")] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        result = self.run_clip(m, TURN_LEFT)
        self.assertFalse(result["hit"])
        self.assertEqual(result["intent"]["reason"], "undefined_pelvis_heading")

    def test_height_uses_first_and_last_window_medians_and_inclusive_change(self):
        m = clip()
        y(m, "Hips")[0] = 0.8  # One beginning outlier leaves the median at 1.0.
        y(m, "Hips")[-5:] = [0.80, 0.80, 0.81, 0.79, 0.83]
        self.assertTrue(self.run_clip(m, LOWER)["hit"])
        self.assertFalse(self.run_clip(m, RISE)["hit"])
        m["posed_joints"] = m["posed_joints"][::-1].copy()
        y(m, "Hips")[-1] = 0.99
        self.assertTrue(self.run_clip(m, RISE)["hit"])

    def test_height_rejects_an_unstable_tail_even_with_sufficient_median_change(self):
        m = clip()
        y(m, "Hips")[-5:] = [0.81, 0.76, 0.77, 0.76, 0.76]
        self.assertTrue(self.run_clip(m, LOWER)["hit"])
        y(m, "Hips")[-5] = 0.812
        result = self.run_clip(m, LOWER)
        self.assertFalse(result["hit"])
        self.assertEqual(result["intent"]["reason"], "unstable_end_height")

    def test_order_requires_first_onsets_to_be_separated_by_ceil_window(self):
        m = clip()
        y(m, "LeftHand")[4:9] = 1.4
        y(m, "RightHand")[9:14] = 1.4
        result = self.run_clip(m, order(LEFT, RIGHT))
        self.assertTrue(result["hit"])
        self.assertEqual(result["intent"]["gap_frames"], 5)
        self.assertFalse(self.run_clip(m, order(RIGHT, LEFT))["hit"])
        m["posed_joints"] = m["posed_joints"][::-1].copy()
        self.assertFalse(self.run_clip(m, order(LEFT, RIGHT))["hit"])
        self.assertTrue(self.run_clip(m, order(RIGHT, LEFT))["hit"])

    def test_order_can_detect_transient_height_and_turn_events(self):
        m = clip()
        y(m, "Hips")[7:12] = 0.76  # Returns to initial height before clip end.
        degrees = np.zeros(30)
        degrees[12:17] = 90  # Returns to initial yaw before clip end.
        apply_yaw(m, degrees)
        result = self.run_clip(m, order(LOWER, TURN_LEFT))
        self.assertTrue(result["hit"])
        self.assertEqual((result["intent"]["first"]["onset_frame"], result["intent"]["second"]["onset_frame"]), (7, 12))
        self.assertFalse(self.run_clip(m, LOWER)["hit"])
        self.assertFalse(self.run_clip(m, TURN_LEFT)["hit"])
        y(m, "Hips")[7:12] = 1.0
        y(m, "Hips")[8:13] = 0.76
        self.assertEqual(self.run_clip(m, order(LOWER, TURN_LEFT))["intent"]["reason"], "events_too_close")
        y(m, "Hips")[8:13] = 1.0
        y(m, "Hips")[12:17] = 0.76
        self.assertEqual(self.run_clip(m, order(LOWER, TURN_LEFT))["intent"]["reason"], "events_too_close")
        y(m, "Hips")[12:17] = 1.0
        self.assertEqual(self.run_clip(m, order(LOWER, TURN_LEFT))["intent"]["reason"], "missing_event")

    def test_order_early_policy_rejects_instead_of_ignoring_first_occurrence(self):
        m = clip()
        y(m, "RightHand")[:5] = 1.4
        y(m, "LeftHand")[6:11] = 1.4
        y(m, "RightHand")[15:20] = 1.4
        self.assertEqual(self.run_clip(m, order(LEFT, RIGHT))["intent"]["reason"], "wrong_order")
        result = self.run_clip(m, order(LEFT, {**RIGHT, "not_before_frame": 5}))
        self.assertEqual(result["intent"]["reason"], "early_event")
        self.assertEqual(result["intent"]["second"]["onset_frame"], 0)

    def test_invalid_manifest_options_raise_instead_of_marking_motion_as_failure(self):
        m = clip()
        for intent in ({**TURN_LEFT, "min_degrees": 121}, {**TURN_LEFT, "max_degrees": np.inf},
                       {**LOWER, "direction": "down"}, order(LEFT, order(LOWER, RISE)),
                       order(LEFT, {**RIGHT, "not_before_frame": -1})):
            with self.subTest(intent=intent), self.assertRaises(ValueError):
                self.run_clip(m, intent)
        for args in ({"fps": 0}, {"expected": True}, {"sign": 0}, {"footing": "false"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.run_clip(m, **args)


class GateTests(unittest.TestCase):
    def test_tla_boundary_examples(self):
        v1 = load_scorer()
        good = good_report()
        self.assertTrue(v1.valid(good))
        self.assertTrue(v1.gate(good))
        self.assertEqual(v1.score(good), 7 * 385 + 336)
        changes = [
            ("evidence", None, False), ("exports", None, False),
            ("standalone", None, False), ("p95", "en", 1201),
            ("peak", "ko", 16001), ("hits", "ko", [37, 42, 42, 42]),
            ("hits", "ko", [39] * 4), ("usable", "en", 171),
            ("hits", None, dict.fromkeys(good["hits"], [38] * 4)),
            ("hits", "en", [45] * 4), ("usable", "ko", 193),
        ]
        for key, arm, value in changes:
            with self.subTest(key=key, arm=arm, value=value):
                m = deepcopy(good)
                if arm is None:
                    m[key] = value
                else:
                    m[key][arm] = value
                self.assertFalse(v1.gate(m))
        m = good_report()
        m["exports"] = False
        self.assertEqual(v1.score(m), v1.score(good))
        m["p95"]["en"] += 1
        self.assertEqual(v1.score(m), v1.score(good) - 385)

    def test_exact_integer_hit_and_quality_boundaries(self):
        v1 = load_scorer()
        m = good_report()
        m["hits"] = {"ref": [38, 41, 42, 42], "en": [34, 39, 40, 41], "ko": [38, 41, 42, 42]}
        m["usable"] = {"ref": 182, "en": 173, "ko": 182}
        self.assertTrue(v1.gate(m))  # 154 vs. 163, 34 vs. 38, 173 vs. 182.
        below = deepcopy(m)
        below["hits"]["en"][0] = 33
        below["hits"]["en"][1] += 1
        self.assertFalse(v1.gate(below))  # Same total, failing kind.
        below = deepcopy(m)
        below["hits"]["ref"][1] += 1
        self.assertFalse(v1.gate(below))  # Ref lead = 10.
        below = deepcopy(m)
        below["hits"]["ko"][1] += 1
        self.assertFalse(v1.gate(below))  # Language gap = 10; others pass.
        m["usable"]["en"] -= 1
        self.assertFalse(v1.gate(m))

    def test_valid_rejects_bad_types_domains_and_bounds(self):
        v1 = load_scorer()
        malformed = [
            {"hits": {"en": [42] * 4}}, {"evidence": 1}, {"exports": "true"},
            {"peak": {"ref": 0, "en": 1, "ko": 1}}, {"p95": {"ref": 1., "en": 1, "ko": 1}},
            {"hits": {**good_report()["hits"], "ref": [True, 0, 0, 0]}},
            {"hits": {**good_report()["hits"], "ref": [49, 0, 0, 0]}},
            {"hits": {**good_report()["hits"], "ref": [0, 0, 0]}},
            {"usable": {**good_report()["usable"], "other": 1}},
            {"usable": {**good_report()["usable"], "ko": -1}},
        ]
        for updates in malformed:
            with self.subTest(updates=updates):
                m = {**good_report(), **updates}
                self.assertFalse(v1.valid(m))
                self.assertFalse(v1.gate(m))
                with self.assertRaises(ValueError):
                    v1.score(m)
        self.assertFalse(v1.valid(None))


class StateTests(unittest.TestCase):
    def test_only_strict_score_improvement_resets_patience(self):
        v1 = load_scorer()
        m = good_report()
        m["standalone"] = False
        m["hits"]["en"][1] -= 1
        state = v1.init_state()
        self.assertEqual(state, dict(phase="Dev", trial=0, best=-1, stale=0, locked=0, tests=0, last="none"))
        state = v1.advance(state, m)
        self.assertEqual((state["trial"], state["stale"], state["best"]), (1, 0, v1.score(m)))
        state = v1.advance(state, m)
        self.assertEqual(state["stale"], 1)
        m["hits"]["en"][1] += 1
        state = v1.advance(state, m)
        self.assertEqual(state["stale"], 0)
        for stale in range(1, 4):
            state = v1.advance(state, m)
            self.assertEqual((state["phase"], state["stale"]), ("Dev" if stale < 3 else "Stopped", stale))
        self.assertEqual(v1.advance(state, good_report()), state)

    def test_eighth_trial_stops_unless_gate_passes(self):
        v1 = load_scorer()
        m = good_report()
        m["exports"] = False
        state = v1.init_state()
        for count in range(35, 42):
            m["hits"]["en"][0] = count
            state = v1.advance(state, m)
            self.assertEqual((state["phase"], state["stale"]), ("Dev", 0))
        stopped = v1.advance(state, m)
        self.assertEqual((stopped["phase"], stopped["trial"], stopped["locked"]), ("Stopped", 8, 0))
        frozen = v1.advance(state, good_report())
        self.assertEqual((frozen["phase"], frozen["trial"], frozen["locked"]), ("Frozen", 8, 8))
        self.assertEqual(state["trial"], 7)

    def test_confirmation_consumes_exactly_one_report(self):
        v1 = load_scorer()
        frozen = v1.advance(v1.init_state(), good_report())
        for phase, updates in (("Accepted", {}), ("Rejected", {"exports": False}),
                               ("Blocked", {"evidence": False}), ("Blocked", {"usable": None})):
            with self.subTest(phase=phase, updates=updates):
                m = {**good_report(), **updates}
                final = v1.advance(frozen, m)
                self.assertEqual((final["phase"], final["tests"], final["last"]), (phase, 1, m))
                for key in ("trial", "best", "stale", "locked"):
                    self.assertEqual(final[key], frozen[key])
                m["hits"]["en"][0] = 0
                self.assertEqual(final["last"]["hits"]["en"][0], 42)
                self.assertEqual(v1.advance(final, good_report()), final)

    def test_invalid_dev_blocks_without_updating_ranking(self):
        v1 = load_scorer()
        m = {**good_report(), "exports": False}
        state = v1.advance(v1.advance(v1.init_state(), m), m)
        for report in ({**m, "evidence": False}, {**m, "usable": None}):
            blocked = v1.advance(state, report)
            self.assertEqual(blocked, {**state, "phase": "Blocked", "trial": 3})
            self.assertEqual(v1.advance(blocked, good_report()), blocked)


if __name__ == "__main__":
    unittest.main()
