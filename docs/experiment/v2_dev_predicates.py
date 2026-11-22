"""Geometric predicate checks for caption comprehension on the frozen dev split.

A diagnostic: predicates and thresholds are fixed from body-scale constants and the
skeleton rest pose before any arm values are read. Release decisions stay with the human panel.

Axis convention (calibrated from SOMASkeleton77 neutral_joints): +X = character LEFT,
-Y down; forward from global_root_heading. Distances in metres.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

HIPS, SPINE2, HEAD = 0, 2, 6
L_HAND, R_HAND = 14, 42
L_SHOULDER, R_SHOULDER = 11, 39
L_LEG, R_LEG = 67, 72  # upper-leg marker below the hip
L_FOOT, R_FOOT, L_TOE, R_TOE = 69, 74, 71, 76


def _feet_floors(j):
    return j[:, L_TOE][:, 1].min(), j[:, R_TOE][:, 1].min()


def p_min_dist(j, a, b):
    return float(np.linalg.norm(j[:, a] - j[:, b], axis=1).min())


def p_any_hand_near_head(j):
    d = np.minimum(np.linalg.norm(j[:, L_HAND] - j[:, HEAD], axis=1),
                   np.linalg.norm(j[:, R_HAND] - j[:, HEAD], axis=1))
    return float(d.min())


def p_max_both_hands_above_hips(j):
    return float((np.minimum(j[:, L_HAND][:, 1], j[:, R_HAND][:, 1]) - j[:, HIPS][:, 1]).max())


def p_max_hand_above_hips(j, hand):
    return float((j[:, hand][:, 1] - j[:, HIPS][:, 1]).max())


def p_hands_spread(j):
    return float(np.minimum(np.abs(j[:, L_HAND][:, 0] - j[:, HIPS][:, 0]),
                            np.abs(j[:, R_HAND][:, 0] - j[:, HIPS][:, 0])).max())


def p_hands_joined_above(j):
    raised = (j[:, L_HAND][:, 1] > j[:, HIPS][:, 1] + 0.4) & (j[:, R_HAND][:, 1] > j[:, HIPS][:, 1] + 0.4)
    if not raised.any():
        return 0.0
    return float(np.linalg.norm(j[raised, L_HAND] - j[raised, R_HAND], axis=1).min())


def p_foot_raised_fraction(j, foot, other):
    floor_f, floor_o = _feet_floors(j)
    if foot == L_TOE:
        floor, other_floor = floor_f, floor_o
        o = R_TOE
    else:
        floor, other_floor = floor_o, floor_f
        o = L_TOE
    up = (j[:, foot][:, 1] >= floor + 0.05) & (j[:, o][:, 1] <= other_floor + 0.03)
    return float(up.mean())


def p_both_airborne_fraction(j):
    floor_l, floor_r = _feet_floors(j)
    up = (j[:, L_TOE][:, 1] >= floor_l + 0.08) & (j[:, R_TOE][:, 1] >= floor_r + 0.08)
    return float(up.mean())


def p_min_hips_y(j):
    return float(j[:, HIPS][:, 1].min())


def p_min_hand_y(j):
    return float(np.minimum(j[:, L_HAND][:, 1], j[:, R_HAND][:, 1]).min())


def p_net_lateral_x(j):
    return float(j[:, HIPS][:, 0][-1] - j[:, HIPS][:, 0][0])


def p_net_forward(j, heading):
    # global_root_heading points to the character's LEFT; forward = (-h_z, h_x) in (x, z).
    h = heading[0] / np.linalg.norm(heading[0])
    fwd = np.array([-h[1], h[0]])
    return float((j[:, HIPS][-1, [0, 2]] - j[:, HIPS][0, [0, 2]]) @ fwd)


def p_heading_change_deg(j, heading):
    # Positive change = turning toward the character's RIGHT (calibrated from the rest pose).
    a, b = heading[0] / np.linalg.norm(heading[0]), heading[-1] / np.linalg.norm(heading[-1])
    return float(math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])))


def p_foot_glide_while_low(j, foot):
    floor = j[:, foot][:, 1].min()
    low = j[:, foot][:, 1] <= floor + 0.05
    if low.sum() < 3:
        return 0.0
    return float(np.linalg.norm(np.diff(j[low, foot][:, [0, 2]], axis=0), axis=1).sum())


def p_hips_lateral_extent(j):
    return float(j[:, HIPS][:, 0].max() - j[:, HIPS][:, 0].min())


def p_stance_width(j):
    return float(np.abs(j[:, L_FOOT][:, 0] - j[:, R_FOOT][:, 0]).max())


def p_foot_forward_offset(j, foot, other):
    return float((j[:, foot][:, 2] - j[:, other][:, 2]).max())


def p_max_leg_above_floor(j, leg):
    floor = min(_feet_floors(j))
    return float((j[:, leg][:, 1] - floor).max())


# Predicate registry: name -> (function, needs_heading)
REGISTRY = {
    'dist': (p_min_dist, False),
    'any_hand_head': (p_any_hand_near_head, False),
    'both_hands_up': (p_max_both_hands_above_hips, False),
    'hand_up': (p_max_hand_above_hips, False),
    'spread': (p_hands_spread, False),
    'joined': (p_hands_joined_above, False),
    'foot_up_frac': (p_foot_raised_fraction, False),
    'airborne': (p_both_airborne_fraction, False),
    'hips_low': (p_min_hips_y, False),
    'hand_low': (p_min_hand_y, False),
    'lateral': (p_net_lateral_x, False),
    'forward': (p_net_forward, True),
    'turn_deg': (p_heading_change_deg, True),
    'glide': (p_foot_glide_while_low, False),
    'hips_extent': (p_hips_lateral_extent, False),
    'stance_width': (p_stance_width, False),
    'foot_forward': (p_foot_forward_offset, False),
    'leg_up': (p_max_leg_above_floor, False),
}

# Spec: group -> (short caption, checks). Each check: (predicate, args, op, limit).
# op 'ge' means hit when value >= limit; 'le' means hit when value <= limit.
SPEC = {
    '039f75b5e790c62e4842dab5': ('sways side to side hands up', [('hips_extent', (), 'ge', 0.15)]),
    '08b96f38d5400b8a36576441': ('kneels on hands and knees', [('hips_low', (), 'le', 0.55)]),
    '0dd9131aadeb1fe468c88288': ('relaxed dance wide stance', [('stance_width', (), 'ge', 0.35)]),
    '1664a9c3890a2cd1e5be9b53': ('high five right hand', [('hand_up', (R_HAND,), 'ge', 0.45)]),
    '167e7dfa5ccb382bbdc92fa3': ('turn over right shoulder', [('turn_deg', (), 'ge', 120.0)]),
    '1b3bfb61b22ee8f03daf1fc6': ('bent arms to shoulder height', [('both_hands_up', (), 'ge', 0.25)]),
    '2169084e10e2212bf7e8679c': ('hopping victory dance', [('airborne', (), 'ge', 0.15)]),
    '25e0d0da92138ab03d5dcdea': ('crouch and turn over right shoulder', [('turn_deg', (), 'ge', 90.0), ('hips_low', (), 'le', 0.65)]),
    '2e3876de9dded51dc08da02a': ('covers eyes with palms', [('dist', (L_HAND, HEAD), 'le', 0.15), ('dist', (R_HAND, HEAD), 'le', 0.15)]),
    '2fc955d82f65afda207e4ae2': ('dance turning to the sides', [('turn_deg', (), 'abs_ge', 60.0)]),
    '33ed6da2c34ce0a020ef41dc': ('vogue catwalk toward front', [('forward', (), 'ge', 0.3)]),
    '357f00355d7506891adf7610': ('spinning quickly in square', [('turn_deg', (), 'abs_ge', 120.0)]),
    '2ccfa81aa57c3168bde2dfa0': ('hands joined above head', [('joined', (), 'le', 0.20), ('both_hands_up', (), 'ge', 0.45)]),
    '4d36b42e891aa99cdd3b86d5': ('stand on left leg right foot up', [('foot_up_frac', (R_TOE, L_TOE), 'ge', 0.2)]),
    '47b46ab7c5a306e46d3d1b3a': ('wide leg movements', [('stance_width', (), 'ge', 0.4)]),
    '48d82d7b9952e2919b7413b6': ('arms up runs and turns', [('both_hands_up', (), 'ge', 0.4), ('turn_deg', (), 'abs_ge', 45.0)]),
    '5b2a18115b8ef1bbd005ad3b': ('gliding foot sideways along floor', [('glide', (R_TOE,), 'ge', 0.3), ('glide', (L_TOE,), 'ge', 0.3)]),
    '6b6ba74d5e716e7556659943': ('relaxed walk forward', [('forward', (), 'ge', 0.5)]),
    '6d73db107c9a1515a6a6584f': ('pointing upwards right hand', [('hand_up', (R_HAND,), 'ge', 0.55)]),
    '70f59468145e5d15b189cf2a': ('backward swing of both legs', [('leg_up', (R_LEG,), 'ge', 0.2), ('leg_up', (L_LEG,), 'ge', 0.2)]),
    '73e0e55ed56f13b64815859f': ('fingers to forehead', [('dist', (L_HAND, HEAD), 'le', 0.2), ('dist', (R_HAND, HEAD), 'le', 0.2)]),
    '74b83f3073afd1f7b6e76350': ('stretch out arms warmup', [('spread', (), 'ge', 0.35)]),
    '7553155090ade392616bb72b': ('low five right hand at waist', [('dist', (R_HAND, HIPS), 'le', 0.25)]),
    '7b927c609d8846985e1419e6': ('scratching head one hand', [('dist', (L_HAND, HEAD), 'le', 0.2), ('dist', (R_HAND, HEAD), 'le', 0.2)]),
    '828b0e914ca2501900f8130b': ('dance with feet and hands out', [('spread', (), 'ge', 0.3), ('stance_width', (), 'ge', 0.3)]),
    'a19824544b4898f53641b0d3': ('hip hop pointing at head', [('dist', (L_HAND, HEAD), 'le', 0.22), ('dist', (R_HAND, HEAD), 'le', 0.22)]),
    'ab5b5df348eca31e79d50642': ('walk in an arc', [('forward', (), 'ge', 0.3), ('turn_deg', (), 'abs_ge', 45.0)]),
    'ae1894a10f00eae33ad96838': ('spreading arms gesture', [('spread', (), 'ge', 0.35)]),
    'b5e2b54ba8b47075416b03d3': ('stand up from lying on stomach', [('hips_low', (), 'le', 0.35)]),
    'b741b0471a230b6a9ffbabb0': ('improvised forward walk', [('forward', (), 'ge', 0.5)]),
    'bb60ab37fe7a6a595bf18b06': ('walking rightward full cycle', [('lateral', (), 'le', -0.3)]),
    'c03095187d76d5a0c9f3bba9': ('jogs quickly in arched path', [('forward', (), 'ge', 0.4), ('turn_deg', (), 'abs_ge', 30.0)]),
    'c971a9d2b695c2dedcb8f303': ('dance swaying moving backward', [('forward', (), 'le', -0.2)]),
    'ca92b0bb362c5d3964a76cec': ('scratch right thigh right hand', [('dist', (R_HAND, R_LEG), 'le', 0.3)]),
    'cf5a7994f445d063dddc8600': ('slides backward with left leg', [('forward', (), 'le', -0.2)]),
    'd54f0f24ae37c977bda2f6b5': ('scratch left shoulder right hand', [('dist', (R_HAND, L_SHOULDER), 'le', 0.25)]),
    'dfca8074e0412d55803f70bf': ('one leg forward then hand near head', [('any_hand_head', (), 'le', 0.25)]),
    'd6b172e6b09dac97f6816c13': ('hips to sides hands in out', [('hips_extent', (), 'ge', 0.15)]),
    'd6b34c5a7ca1f2ec5df60dc8': ('bend touch ground vogue', [('hand_low', (), 'le', 0.25)]),
    'd718b2f9a33573fb8fb0d340': ('jumping hands in air', [('airborne', (), 'ge', 0.1), ('both_hands_up', (), 'ge', 0.35)]),
    'e2e6603c5f13741b6387ba88': ('relaxed walk forward low energy', [('forward', (), 'ge', 0.5)]),
    'e5e7306308ff8f9f8973a191': ('sitting on left leg right knee up', [('hips_low', (), 'le', 0.5), ('leg_up', (R_LEG,), 'ge', 0.25)]),
    'f593729cdbc575576d23cb02': ('looks back over shoulders', [('turn_deg', (), 'abs_ge', 60.0)]),
    'f5e0b60897eead37ad0154a7': ('hips sideways in place facing forward', [('hips_extent', (), 'ge', 0.12)]),
    'f8a32ad92c0f290949f88ca7': ('pulling shoulder while walking', [('forward', (), 'ge', 0.2)]),
    'fb30b138d641a12e1b0be6c8': ('vogue stepping to the right', [('lateral', (), 'le', -0.2)]),
    'ff6df37e9a4a4f3f3b47734e': ('step back extending arms', [('forward', (), 'le', -0.15), ('spread', (), 'ge', 0.25)]),
    'ff9f75a9f004b85840d8d93e': ('turn left vogue body shake', [('turn_deg', (), 'le', -90.0)]),
    'ffb632a5fc0c11a1abf01b98': ('standing to seated bent legs', [('hips_low', (), 'le', 0.5)]),
}

SPEC = {k: v for k, v in SPEC.items() if v[0]}

MAPPED = set(SPEC)
UNMAPPED_REASON = {
    '2d8afbcfa77a7530fe9485e7': 'only qualitative dance style claims',
    '29c6647efeca943a5b0313d6': 'relative body-side motion without a stable frame',
    '2895f94d32f9a945ba07accf': 'worm-like steps, no single checkable fact',
    '47a8133676d53da0c189cb60': 'two-person interaction, unmeasurable solo',
    '4c4e46aba639558ceef47037': 'subtle postural offset below noise floor',
    '531b8e4f80688101a12b65f8': 'only qualitative dance style claims',
    '5a5f5035b8bc8ecc1efe00f8': 'only qualitative dance style claims',
    '6e18fdea4fc631c90547e19b': 'named vogue pose not formally specified',
    '74e3e49c56d0052b1ddbc002': 'varied steps in different directions, no one fact',
    '75fb14d5b03fc7c38456b881': 'only qualitative dance style claims',
    '82b31193566868055536987b': 'only qualitative dance style claims',
    '94af6d1b1fcf3c644e254f08': 'only qualitative dance style claims',
    'a6ef120d9ce8d3ec7cc50576': 'disco finger detail below joint resolution',
    'c8fbeee575b8e8fb7b7ad487': 'diagonal behind-pointing has no stable ground-truth frame',
    'f9748895a1da98454f5dbeaa': 'only qualitative bounce/rhythm claims',
}


def evaluate(npz, check):
    predicate, args, op, limit = check
    fn, needs_heading = REGISTRY[predicate]
    kwargs = {}
    if needs_heading:
        kwargs['heading'] = npz['global_root_heading'].astype('d')
    value = fn(npz['posed_joints'].astype('d'), *args, **kwargs)
    magnitude = abs(value)
    if op == 'ge':
        hit = bool(value >= limit)
    elif op == 'le':
        hit = bool(value <= limit)
    elif op == 'abs_ge':
        hit = bool(magnitude >= limit)
    elif op == 'abs_le':
        hit = bool(magnitude <= limit)
    else:
        raise ValueError(f'unknown op {op}')
    return value, hit


def compute(run):
    from v2_dev_diagnostic import pairs
    rows = []
    for (group, seed, caption), arms in pairs(run):
        if group not in SPEC:
            continue
        checks = []
        for name, check in zip((c[0] for c in SPEC[group][1]), SPEC[group][1]):
            entry = dict(predicate=name, op=check[2], limit=check[3])
            for arm, path in arms.items():
                with np.load(path) as z:
                    value, hit = evaluate(z, check)
                entry[f'{arm}_value'] = value
                entry[f'{arm}_hit'] = hit
            checks.append(entry)
        rows.append(dict(group=group, seed=seed, caption=caption, checks=checks))
    return rows


def summarise(rows):
    from scipy.stats import binomtest
    checks = [c for r in rows for c in r['checks']]
    a = sum(1 for x in checks if x['ref_hit'] and x['candidate_hit'])
    b = sum(1 for x in checks if not x['ref_hit'] and x['candidate_hit'])
    c_ = sum(1 for x in checks if x['ref_hit'] and not x['candidate_hit'])
    d = sum(1 for x in checks if not x['ref_hit'] and not x['candidate_hit'])
    discordant = b + c_
    # exact McNemar: two-sided binomial test of the discordant pairs
    p = float(binomtest(b, discordant, 0.5).pvalue) if discordant else 1.0
    return dict(n_pairs=len(rows), n_checks=len(checks),
                ref_hits=a + c_, candidate_hits=a + b, both_hit=a, both_miss=d,
                rescue=b, loss=c_, agree=a + d, mcnemar_exact_p=p)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    rows = compute(a.run)
    print(json.dumps(summarise(rows), indent=1))
    a.out.write_text(json.dumps(dict(schema='kimodo.v2.dev-predicates.1', run=str(a.run.resolve()),
                                     rows=rows, summary=summarise(rows),
                                     limit='caption-fact diagnostic, not the release gate'), indent=1) + '\n',
                     encoding='utf-8')
