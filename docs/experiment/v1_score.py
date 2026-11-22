"""NumPy SOMA-77 clip scorer and integer port of EncoderTrial.tla."""

from collections.abc import Mapping
from copy import deepcopy
from math import ceil, isfinite
from numbers import Integral, Real

import numpy as np

__all__ = ["score_clip", "valid", "gate", "score", "init_state", "advance"]

_ARMS = frozenset(("ref", "en", "ko"))
_STUDENTS = ("en", "ko")
_FLAGS = ("evidence", "standalone", "exports")
_KINDS, _N = 4, 48
_TOTAL = _KINDS * _N
_MAX_TRIALS, _PATIENCE = 8, 3  # EncoderTrial.cfg
_JOINT_COUNT = 77
_SIDES = ("left", "right")
# Kimodo SOMASkeleton77 names; Foot is the ankle, ToeEnd the toe tip.
_BONES = {"Hips"} | {s.title() + b for s in _SIDES for b in ("Shoulder", "Hand", "Foot", "ToeEnd")}
_FLOOR_BONES = tuple(s.title() + b for s in _SIDES for b in ("Foot", "ToeEnd"))
# A decimal boundary can land a fraction of a micrometre away after float32
# NPZ quantization. Keep the rounding margin separate from the v1 thresholds.
_METRE_EPS = 1e-7
_ANGLE_EPS = 1e-5
_INTENT_FIELDS = {
    "hand": {"type", "side"},
    "turn": {"type", "direction", "min_degrees", "max_degrees"},
    "height": {"type", "direction"},
    "order": {"type", "first", "second"},
}


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def valid(measurement):
    """True iff the TLA+ fields have valid types, arm domains and bounds."""
    if not isinstance(measurement, Mapping):
        return False
    if not all(type(measurement.get(key)) is bool for key in _FLAGS):
        return False
    for key in ("hits", "usable", "peak", "p95"):
        arms = measurement.get(key)
        if not isinstance(arms, Mapping) or arms.keys() != _ARMS:
            return False
        for value in arms.values():
            if key == "hits":
                if (not isinstance(value, list) or len(value) != _KINDS
                        or not all(_integer(n) and 0 <= n <= _N for n in value)):
                    return False
            elif not _integer(value) or (not 0 <= value <= _TOTAL if key == "usable" else value <= 0):
                return False
    return True


def _checks(measurement):
    # Cast integral scalars to Python ints before arithmetic (no NumPy overflow).
    hits = {arm: tuple(map(int, measurement["hits"][arm])) for arm in _ARMS}
    totals = {arm: sum(row) for arm, row in hits.items()}
    usable, peak, p95 = (
        {arm: int(measurement[key][arm]) for arm in _ARMS}
        for key in ("usable", "peak", "p95")
    )
    passed = [100 * abs(totals["en"] - totals["ko"]) <= 5 * _TOTAL]
    for arm in _STUDENTS:
        passed.append(
            100 * totals[arm] >= 80 * _TOTAL
            and 100 * (totals[arm] - totals["ref"]) >= -5 * _TOTAL
            and all(100 * h >= 70 * _N and 100 * (h - r) >= -10 * _N
                    for h, r in zip(hits[arm], hits["ref"]))
        )
        passed.append(
            100 * usable[arm] >= 90 * _TOTAL
            and 100 * (usable[arm] - usable["ref"]) >= -5 * _TOTAL
        )
        passed.append(100 * peak[arm] <= 80 * peak["ref"] and 100 * p95[arm] <= 120 * p95["ref"])
    return passed, totals["en"] + totals["ko"]


def gate(measurement):
    """Integer acceptance Gate; malformed or incomplete reports fail closed."""
    return valid(measurement) and all(measurement[k] for k in _FLAGS) and all(_checks(measurement)[0])


def score(measurement):
    """Integer Dev ranking score (not an acceptance result); invalid raises ValueError."""
    if not valid(measurement):
        raise ValueError("invalid measurement")
    passed, student_hits = _checks(measurement)
    return sum(passed) * (2 * _TOTAL + 1) + student_hits


def init_state():
    """New JSON-compatible trial state with the TLA+ Init field names."""
    return dict(phase="Dev", trial=0, best=-1, stale=0, locked=0, tests=0, last="none")


def advance(state, measurement):
    """Consume one report per Dev/Frozen step, or stutter in a terminal phase.

    Pass a state from init_state/advance or its persisted equivalent. Callers
    enforce manifest and candidate identity, persistence, and once-only final
    execution across processes; this function performs only the TLA+ step.
    """
    phase = state["phase"]
    result = dict(state)
    if phase in ("Accepted", "Rejected", "Stopped", "Blocked"):
        return result
    if phase == "Dev" and state["trial"] < _MAX_TRIALS:
        result["trial"] += 1
        if not valid(measurement) or not measurement["evidence"]:
            result["phase"] = "Blocked"
            return result
        ranking = score(measurement)
        if ranking > state["best"]:
            result.update(best=ranking, stale=0)
        else:
            result["stale"] += 1
        if gate(measurement):
            result.update(phase="Frozen", locked=result["trial"])
        else:
            result["locked"] = 0
            if result["trial"] == _MAX_TRIALS or result["stale"] >= _PATIENCE:
                result["phase"] = "Stopped"
        return result
    if phase == "Frozen" and state["tests"] == 0:
        if not valid(measurement) or not measurement["evidence"]:
            final = "Blocked"
        else:
            final = "Accepted" if gate(measurement) else "Rejected"
        result.update(phase=final, tests=1, last=deepcopy(measurement))
        return result
    raise ValueError("state has no enabled transition")


def _runs(mask):
    """Yield all maximal true intervals with exclusive end indices."""
    edges = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return ((int(start), int(end)) for start, end in edges.reshape(-1, 2))


def _array(motion, key):
    try:
        data = np.asarray(motion[key])
    except (KeyError, TypeError, ValueError):
        return None
    if data.dtype.kind not in "fiu":  # No coercion from strings, objects, bools or complex.
        return None
    return data


def _quality(joints, indices, window, footing):
    heights = joints[:, [indices[b] for b in _FLOOR_BONES], 1]
    penetration = int(np.count_nonzero(np.any(heights < -0.05 - _METRE_EPS, axis=1)))
    reasons = []
    if 100 * penetration > 2 * len(joints):
        reasons.append("floor_penetration")
    runs = []
    if footing:
        for side in _SIDES:
            toe = joints[:, indices[side.title() + "ToeEnd"]]
            for start, end in _runs(np.abs(toe[:, 1]) <= 0.06 + _METRE_EPS):
                if end - start < window:
                    continue
                xz = toe[start:end, (0, 2)]
                dx, dz = (xz - xz[0]).T
                runs.append(dict(side=side, start_frame=start, end_frame_exclusive=end,
                                 max_displacement_m=float(np.max(np.hypot(dx, dz)))))
        if not runs:
            reasons.append("no_toe_contact")
        elif any(run["max_displacement_m"] > 0.10 + _METRE_EPS for run in runs):
            reasons.append("foot_slide")
    return dict(reasons=reasons, penetration_frames=penetration, footing_checked=footing, contact_runs=runs)


def _finite_real(value):
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _validate_intent(intent, nested=False):
    kind = intent.get("type") if isinstance(intent, Mapping) else None
    if not isinstance(kind, str) or kind not in _INTENT_FIELDS or (nested and kind == "order"):
        raise ValueError("invalid intent type")
    fields = intent.keys() - ({"not_before_frame"} if nested else set())
    if fields != _INTENT_FIELDS[kind]:
        raise ValueError(f"invalid {kind} intent fields")
    minimum_frame = intent.get("not_before_frame", 0)
    if not _integer(minimum_frame) or minimum_frame < 0:
        raise ValueError("not_before_frame must be a nonnegative integer")
    if kind == "hand" and intent["side"] not in _SIDES:
        raise ValueError("hand side must be left or right")
    if kind == "turn":
        low, high = intent["min_degrees"], intent["max_degrees"]
        if (intent["direction"] not in _SIDES or not _finite_real(low) or not _finite_real(high)
                or not 0 < low <= high):
            raise ValueError("turn requires left/right and 0 < min_degrees <= max_degrees")
    if kind == "height" and intent["direction"] not in ("lower", "rise"):
        raise ValueError("height direction must be lower or rise")
    if kind == "order":
        _validate_intent(intent["first"], nested=True)
        _validate_intent(intent["second"], nested=True)


def _event(onset=None, reason="ok", **details):
    return dict(onset_frame=onset, reason=reason, **details)


def _hand(joints, indices, spec, window):
    side = spec["side"].title()
    opposite = ("Right" if side == "Left" else "Left")
    y = joints[:, :, 1]
    raised = y[:, indices[side + "Hand"]] - y[:, indices[side + "Shoulder"]] >= 0.10 - _METRE_EPS
    down = y[:, indices[opposite + "Hand"]] <= y[:, indices[opposite + "Shoulder"]] + _METRE_EPS
    longest = 0
    onset = None
    for start, end in _runs(raised & down):
        longest = max(longest, end - start)
        if onset is None and end - start >= window:
            onset = start
    return _event(onset, "ok" if onset is not None else "no_sustained_hand_raise", longest_hold_frames=longest)


def _first_true(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) if found.size else None


def _turn(motion, frames, hips, spec, window, left_yaw_sign, nested):
    rotations = _array(motion, "global_rot_mats")
    if rotations is None or rotations.shape != (frames, _JOINT_COUNT, 3, 3):
        return _event(reason="invalid_global_rot_mats")
    forward = rotations[:, hips, :, 2].astype(np.float64, copy=False)  # R @ [0, 0, 1]
    if not np.isfinite(forward).all():
        return _event(reason="nonfinite_pelvis_forward")
    x, z = forward[:, 0], forward[:, 2]
    if np.any(np.hypot(x, z) <= 1e-8):
        return _event(reason="undefined_pelvis_heading")
    yaw = np.unwrap(np.arctan2(x, z))
    delta = np.rad2deg(yaw - yaw[0])
    direction = left_yaw_sign if spec["direction"] == "left" else -left_yaw_sign

    def in_range(degrees):
        directed = direction * degrees
        return (directed > 0) & (directed >= spec["min_degrees"] - _ANGLE_EPS) & (directed <= spec["max_degrees"] + _ANGLE_EPS)

    onset = _first_true(in_range(delta))
    if nested:
        if onset is None:
            return _event(reason="no_turn_event")
        return _event(onset, delta_degrees=float(delta[onset]))
    if frames < window:
        return _event(reason="short_window")
    tail_delta = float(np.mean(delta[-window:]))
    reason = "ok" if in_range(tail_delta) else "turn_out_of_range"
    return _event(onset, reason, delta_degrees=tail_delta)


def _height(heights, spec, window, nested):
    if len(heights) < window:
        return _event(reason="short_window")
    initial = float(np.median(heights[:window]))
    sign = -1 if spec["direction"] == "lower" else 1
    if nested:
        windows = np.lib.stride_tricks.sliding_window_view(heights, window)
        medians = np.median(windows, axis=1)
        spans = np.ptp(windows, axis=1)
        # Require the directional threshold at the start as well as for the
        # window median: later samples cannot backdate an event's onset.
        threshold = 0.20 - _METRE_EPS
        onset = _first_true((sign * (medians - initial) >= threshold)
                            & (sign * (windows[:, 0] - initial) >= threshold)
                            & (spans <= 0.05 + _METRE_EPS))
        if onset is None:
            return _event(reason="no_stable_height_event", initial_height_m=initial)
        final, span = float(medians[onset]), float(spans[onset])
        return _event(onset, delta_m=final - initial, end_span_m=span)

    final = float(np.median(heights[-window:]))
    delta, span = final - initial, float(np.ptp(heights[-window:]))
    reason = "ok"
    if sign * delta < 0.20 - _METRE_EPS:
        reason = "height_change_out_of_range"
    elif span > 0.05 + _METRE_EPS:
        reason = "unstable_end_height"
    return _event(reason=reason, initial_height_m=initial, end_height_m=final, delta_m=delta, end_span_m=span)


def _intent(motion, joints, indices, spec, window, sign, nested=False):
    kind = spec["type"]
    if kind == "order":
        first, second = (_intent(motion, joints, indices, spec[key], window, sign, nested=True)
                         for key in ("first", "second"))
        details = dict(first=first, second=second, gap_frames=None)
        if any(event["reason"] == "early_event" for event in (first, second)):
            return _event(reason="early_event", **details)
        if any(event["reason"] != "ok" for event in (first, second)):
            return _event(reason="missing_event", **details)
        details["gap_frames"] = gap = second["onset_frame"] - first["onset_frame"]
        if gap < 0:
            return _event(reason="wrong_order", **details)
        if gap < window:
            return _event(reason="events_too_close", **details)
        return _event(first["onset_frame"], **details)

    if kind == "hand":
        event = _hand(joints, indices, spec, window)
    elif kind == "turn":
        event = _turn(motion, len(joints), indices["Hips"], spec, window, sign, nested)
    else:  # height
        event = _height(joints[:, indices["Hips"], 1], spec, window, nested)
    minimum_frame = spec.get("not_before_frame", 0)
    if nested and event["onset_frame"] is not None and event["onset_frame"] < minimum_frame:
        event.update(reason="early_event", not_before_frame=int(minimum_frame))
    return event


def score_clip(motion, joint_names, fps, expected_frames, intent, left_yaw_sign, footing=True):
    """Score one unbatched SOMA-77 NPZ motion without loading Kimodo or Torch.

    Returns a dict with independent booleans ``hit`` and ``usable``, the
    ``window_frames`` count, ``intent`` details (including ``reason`` and
    zero-based ``onset_frame``), and ``quality`` details (including ``reasons``).
    Invalid generated coordinates fail both; a footing failure does not erase
    an otherwise valid intent hit. Missing or unusable heading data contributes
    no turn event; coordinate-only quality is evaluated separately.

    ``left_yaw_sign`` is the calibrated sign of atan2(forward_x, forward_z)
    displacement for a left turn. Standalone turns use the last window mean;
    standalone height uses the first/last medians and last-window range.
    Order checks onset, not final state: a sustained hand raise, first in-range
    signed turn displacement, or first qualifying stable height window relative
    to the initial median. Each member may specify ``not_before_frame`` to
    reject its first onset before that zero-based frame (never retry later).
    Invalid manifest/caller settings raise ValueError.
    """
    if not _finite_real(fps) or fps <= 0 or not _integer(expected_frames) or expected_frames <= 0:
        raise ValueError("fps and expected_frames must be positive; expected_frames must be integral")
    if not isinstance(left_yaw_sign, Real) or isinstance(left_yaw_sign, bool) or left_yaw_sign not in (-1, 1):
        raise ValueError("left_yaw_sign must be a calibrated -1 or +1")
    if type(footing) is not bool:
        raise ValueError("footing must be boolean")
    _validate_intent(intent)
    window = max(1, ceil(fps / 5))
    joints = _array(motion, "posed_joints")
    try:
        names = tuple(joint_names)
    except TypeError:
        names = ()

    invalid = None
    if joints is None:
        invalid = "missing_or_invalid_posed_joints"
    elif joints.shape != (expected_frames, _JOINT_COUNT, 3):
        invalid = "invalid_posed_joints_shape"
    elif (len(names) != _JOINT_COUNT or not all(isinstance(name, str) for name in names)
          or len(set(names)) != _JOINT_COUNT or not _BONES.issubset(names)):
        invalid = "invalid_soma_joint_names"
    else:
        joints = joints.astype(np.float64, copy=False)
        if not np.isfinite(joints).all():
            invalid = "nonfinite_posed_joints"
    if invalid:
        return dict(hit=False, usable=False, window_frames=window, intent=_event(reason=invalid),
                    quality=dict(reasons=[invalid], penetration_frames=None, footing_checked=False, contact_runs=[]))

    indices = {name: i for i, name in enumerate(names)}
    quality = _quality(joints, indices, window, footing)
    result = _intent(motion, joints, indices, intent, window, int(left_yaw_sign))
    return dict(hit=result["reason"] == "ok", usable=not quality["reasons"],
                window_frames=window, intent=result, quality=quality)
