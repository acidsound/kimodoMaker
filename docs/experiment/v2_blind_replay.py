"""P1.3: blinded CPU-only SOMA review of the two pinned *regression* manifests.

Only historical-review/public is a reviewer handoff. No independent evaluation,
VRM playback, motion generation, inferred intent labels, or human ratings.
"""
import argparse
import csv
from hashlib import sha256
import hmac
import html
import io
from itertools import permutations
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

import numpy as np

import v1_compare_video as renderer
import v2_replay as audit

DATASETS = (
    ('xval', 'xval/manifest.jsonl', 5, 'caeab80c1572faaa61cab034326aabf987da59def802e1a42d03625bf7ed1eef'),
    ('xval-hist', 'xval/hist-manifest.jsonl', 12, '736b02c844837693aee2671d5040756c6040be005e3f7663717a34ae450021e7'),
)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def checked(root, relative):
    path = root / relative
    check(path.resolve() == path and path.is_relative_to(root), f'redirected source/output: {relative}')
    return path


def inventory(v1):
    """Exact manifest × seed × journal terminals, with content hashes of all exports."""
    v1 = Path(v1).resolve(strict=True)
    inputs, groups = {}, []
    for dataset, manifest, count, pin in DATASETS:
        path = checked(v1, manifest)
        check(audit.digest(path) == pin, f'{dataset}: manifest SHA-256 mismatch')
        inputs[manifest] = pin
        rows = audit.manifest(path, count)
        cfg = f'{dataset}/config.json'
        config = audit.read(checked(v1, cfg))
        essentials = dict(schema='kimodo.v1-generate.config.v1', split='xval', pilot=True,
                          manifest_sha256=pin, seeds=list(renderer.SEEDS), fps=renderer.FPS,
                          steps=100, planned_requests=count * len(renderer.SEEDS), exports=True,
                          constraints=[], post_processing=True, limit=None)
        check(all(config.get(k) == val for k, val in essentials.items()), f'{dataset}: config mismatch')
        inputs[cfg] = audit.digest(v1 / cfg)
        journals = {}
        for arm in renderer.ARMS:
            name = f'{dataset}/{arm}/requests.jsonl'
            path = checked(v1, name)
            journals[arm] = audit.journal(path, arm, rows, config['steps'], config['fps'])
            inputs[name] = audit.digest(path)
        for ident, original in rows.items():
            # Whitelist only what read_clip needs. The historical intent/footing/etc.
            # fields were not audited and must never be turned into answer keys.
            row = {k: original[k] for k in ('id', 'en', 'ko', 'seconds')}
            for seed in renderer.SEEDS:
                folders = {}
                for arm in renderer.ARMS:
                    folder = f'{dataset}/{arm}/{ident}-{seed}'
                    for n in audit.FILES:
                        checked(v1, f'{folder}/{n}')
                    rec = journals[arm][ident, seed]
                    hashes = audit.fingerprints(v1 / folder, rec.get('artifact_sha256', {}))
                    inputs.update({f'{folder}/{n}': h for n, h in hashes.items()})
                    folders[arm] = folder
                groups.append(dict(dataset=dataset, row=row, seed=seed, folders=folders,
                                   journal_hashes={arm: journals[arm][ident, seed].get('artifact_sha256', {})
                                                  for arm in renderer.ARMS}))
    return groups, inputs


def assignment(secret, key, choices):
    """Domain-separated pseudonyms and per-trio permutations; never put keys in public."""
    payload = json.dumps(key, separators=(',', ':')).encode('utf-8')
    name = hmac.digest(secret, b'P1.3/id/' + payload, 'sha256').hex()[:24]
    bits = hmac.digest(secret, b'P1.3/order/' + payload, 'sha256')
    orders = tuple(permutations(choices))
    return name, orders[int.from_bytes(bits, 'big') % len(orders)]


def ballots(units):
    buf = io.StringIO(newline='')
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerow(('unit', 'panel', 'action', 'visual_quality', 'evidence_time_and_reason'))
    for u in units:
        for label in renderer.LABELS:
            writer.writerow((u['unit'], label, '', '', ''))
    return buf.getvalue()


def media_evidence(path, expected_frames):
    """No header-only check: require one video stream, then decode and hash every frame."""
    def output(cmd):
        proc = subprocess.run(cmd, capture_output=True, check=False)
        check(proc.returncode == 0 and not proc.stderr.strip(),
              f'{path.name} {cmd[0]}: {proc.stderr.decode("utf-8", "replace")}')
        return proc.stdout

    probe = output(['ffprobe', '-v', 'error', '-count_frames', '-show_entries',
                    'stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames',
                    '-of', 'json', str(path)])
    streams = json.loads(probe)['streams']
    check(len(streams) == 1, 'expected one video stream and nothing else')
    s = streams[0]
    expected = dict(codec_type='video', codec_name='h264', width=3*renderer.SIZE, height=renderer.SIZE,
                    pix_fmt='yuv420p', r_frame_rate='30/1', avg_frame_rate='30/1')
    check(all(s.get(k) == v for k, v in expected.items()), f'{path.name}: media format mismatch: {s}')
    check(s.get('nb_read_frames') == str(expected_frames), f'{path.name}: probed frame count mismatch')
    raw = output(['ffmpeg', '-v', 'error', '-xerror', '-nostdin', '-hwaccel', 'none', '-threads', '1',
                  '-i', str(path), '-map', '0:v:0', '-vsync', '0', '-pix_fmt', 'yuv420p', '-threads', '1',
                  '-f', 'framehash', '-hash', 'sha256', 'pipe:1'])
    rows = [line.split(b',') for line in raw.splitlines() if line and not line.startswith(b'#')]
    check(len(rows) == expected_frames, f'{path.name}: decoded frame count mismatch')
    chain = sha256()
    for i, row in enumerate(rows):
        check(len(row) == 6 and [int(x) for x in row[:5]] == [0, i, i, 1, s['width']*s['height']*3//2],
              f'{path.name}: unexpected decoded frame {i}')
        frame_hash = bytes.fromhex(row[5].strip().decode('ascii'))
        check(len(frame_hash) == 32, 'bad frame digest')
        chain.update(frame_hash)
    return dict(**expected, probed_frames=int(s['nb_read_frames']), decoded_frames=len(rows),
                video_sha256=audit.digest(path), decoded_frames_sha256=chain.hexdigest())


CONTROL_PROMPTS = {
    'hand': 'Raise only your left hand above the top of your head.',
    'turn': 'Turn half a revolution (180 degrees) about the vertical axis.',
    'stance': 'Stand still with both feet planted; do not slide or jump in position.',
}


def make_controls(anchor):
    """Analytic perturbations of ONE initial pose. Not generated success/failure labels."""
    pose = anchor.positions[0]
    t = np.linspace(0, 1, 60)
    t = t*t*(3-2*t)  # ease both endpoints
    idx = anchor.names.index

    def still():
        return np.repeat(pose[None], len(t), axis=0)

    def rotate(indices, pivot, axes, angle):
        a, b = axes
        arr = still()
        rel = pose[indices] - pose[pivot]
        c, s = np.cos(angle*t)[:, None], np.sin(angle*t)[:, None]
        arr[:, indices, a] = pose[pivot, a] + rel[:, a]*c - rel[:, b]*s
        arr[:, indices, b] = pose[pivot, b] + rel[:, a]*s + rel[:, b]*c
        return arr

    def raise_hand(side, sign):
        pivot = idx(f'{side}Arm')
        descendants = [pivot]
        for i, p in enumerate(anchor.parents):
            if p in descendants:
                descendants.append(i)
        return rotate(descendants, pivot, (0, 1), sign*np.pi)

    def turn(degrees):
        return rotate(list(range(len(pose))), idx('Hips'), (2, 0), np.radians(degrees))

    slide, pop = still(), still()
    slide[:, :, 0] += .6*t[:, None]
    pop[len(t)//2:, :, 0] += .6
    kinds = dict(hand=dict(left_overhead=raise_hand('Left', 1), right_overhead=raise_hand('Right', -1), still=still()),
                 turn=dict(half_turn=turn(180), under_turn=turn(45), still=still()),
                 stance=dict(planted=still(), slide=slide, pop=pop))
    return {case: {name: renderer.Clip(poses, anchor.names, anchor.parents, renderer.FPS)
                   for name, poses in kinds[case].items()} for case in kinds}


def control_observation(case, clip):
    """Separate from construction. Geometric discriminators apply ONLY to calibration fixtures."""
    p, idx = clip.positions, clip.names.index
    excursion = float(np.linalg.norm(p-p[0], axis=2).max())
    if case == 'hand':
        left, right = (float((p[:, idx(s+'Hand'), 1]-p[:, idx('HeadEnd'), 1]).max()) for s in ('Left','Right'))
        label = ('still' if excursion < .001 else 'left_overhead' if left > .05 and right < 0
                 else 'right_overhead' if right > .05 and left < 0 else 'other')
        return label, dict(left_clearance_m=left, right_clearance_m=right, max_excursion_m=excursion)
    if case == 'turn':
        axle = p[:, idx('LeftArm')] - p[:, idx('RightArm')]
        yaw = np.degrees(np.unwrap(np.arctan2(-axle[:, 2], axle[:, 0])))
        delta = float(yaw[-1]-yaw[0])
        label = ('still' if excursion < .001 else 'half_turn' if 175 <= delta <= 185
                 else 'under_turn' if 35 <= delta <= 55 else 'other')
        return label, dict(signed_turn_deg=delta, max_excursion_m=excursion)
    check(case == 'stance', 'unexpected control')
    feet = p[:, [idx('LeftFoot'), idx('RightFoot')]]
    drift = float(np.linalg.norm((feet-feet[0])[:, :, [0, 2]], axis=2).max())
    step = float(np.linalg.norm(np.diff(p[:, idx('Hips')], axis=0), axis=1).max())
    lift = float(np.abs(feet[:, :, 1]-feet[0, :, 1]).max())
    label = ('planted' if excursion < .001 else 'pop' if step > .4 else
             'slide' if drift > .4 and lift < .001 and step < .03 else 'other')
    return label, dict(horizontal_foot_drift_m=drift, vertical_foot_excursion_m=lift, max_root_step_m=step)


# Author-written observation cues from the literal English captions, not the historical
# 'intent' field, outcome labels, tuned thresholds, or motion-derived targets.
CUES = {
    'xval-roll-forward': 'Look for a forward body roll along the ground, rather than a jump or backward roll.',
    'xval-sit-crosslegged': 'Look for a descent into a floor-seated position with the legs crossed.',
    'xval-run-sideflip': 'Look for running followed by an airborne sideways body rotation.',
    'xval-two-steps-turn': 'Count the two steps, look for both feet planting, then a turn to the actor’s left.',
    'xval-control-hand': 'Look for the actor’s left hand reaching straight up overhead.',
    'xval-h-01': 'Look for a forward roll and a distinct hero pose. No particular hero pose is specified.',
    'xval-h-02': 'Look for a seated posture with the legs crossed.',
    'xval-h-03': 'Look for a cross-legged seated posture with hands resting on knees. Note if relaxation is illegible.',
    'xval-h-04': 'Look for running and a forward airborne flip; distinguish the flip from a grounded roll.',
    'xval-h-05': 'Look for a sideways flip with one hand involved; note whether hand use is visible.',
    'xval-h-06': 'Look for a kick using the actor’s right leg. No target height or direction is specified.',
    'xval-h-07': 'Look for the actor’s left leg being raised high. No exact angle or height is specified.',
    'xval-h-08': 'Look for the head tipping in a nod rather than the whole body simply bending.',
    'xval-h-09': 'Look for the hands coming together in a clap. No clap count or sound is required.',
    'xval-h-10': 'Look for a forceful downward foot stomp; note if the expressive component is illegible.',
    'xval-h-11': 'Look for a standing posture with arms crossed.',
    'xval-h-12': 'Look for a hand-to-mouth kiss gesture. Note ambiguity rather than requiring facial animation.',
    'hand': 'Distinguish the actor’s two hands; look for only the left hand rising above the head.',
    'turn': 'Compare start and end facing and follow the intervening rotation; look for a half turn.',
    'stance': 'Look for stationary stance, uninterrupted foot support and no sudden change of location.',
}
RUBRIC = (
    'Judge each A, B and C independently, not by rank. Use pass, fail or unjudgeable for each dimension. '
    'Action: pass only if the requested observable action, body side, direction and amount (when specified) '
    'are shown; fail for a visible omission, contradiction or no action when movement was requested. '
    'Visual quality: separately judge whether the motion is physically coherent and continuous. Fail for '
    'visible pops, implausible support, sliding where contact should hold, or distracting jitter; do not '
    'penalize intentional travel or flight. A wrong action may look good and a correct one may look bad. '
    'Use unjudgeable when this view, ambiguous wording, or missing detail prevents a decision. '
    'Enter a timestamp and observable reason for each dimension in a personal copy of ballot.csv. '
    'Do not infer a model or inspect the other reviewer’s ratings. Compare disagreements only after submission. '
    'Left/right refer to the actor: cyan bones are anatomical left, orange are right. '
    'The camera, scale, ground root trail and 0.5 m axes are shared within each trio. This is skeleton motion, not avatar playback.'
)


def catalog(units):
    cards = ''.join(f'<section><h2>{u["unit"]}</h2><p>{html.escape(u["prompt"])}</p>'
                    f'<p>{html.escape(u["action_cue"])}</p><video controls preload="none" width="960" '
                    f'src="{u["video"]}"></video></section>\n' for u in units)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Anonymous motion review</title>'
            '<style>body{font:16px system-ui;max-width:980px;margin:2em auto}video{max-width:100%}'
            'section{padding:1em 0;border-top:1px solid #aaa}h2{font:1em monospace}</style>'
            '<h1>Anonymous motion review</h1><p>' + html.escape(RUBRIC) + '</p><p>'
            '<a href="ballot.csv" download>Blank ballot (CSV)</a></p>' + cards + '</html>\n')


def measurements(clip):
    """Non-scoring world-coordinate descriptors, in metres; frame offsets are zero-based."""
    p = clip.positions
    idx = clip.names.index
    hips, lh, rh = (p[:, idx(n)] for n in ('Hips', 'LeftHand', 'RightHand'))
    def extreme(x, fn):
        i = int(fn(x))
        return dict(m=float(x[i]), frame=i)
    delta = np.diff(hips, axis=0)
    return dict(frames=len(p), fps=clip.fps,
                hips_horizontal_path_m=float(np.linalg.norm(delta[:, [0, 2]], axis=1).sum()),
                hips_height_min=extreme(hips[:, 1], np.argmin), hips_height_max=extreme(hips[:, 1], np.argmax),
                hand_separation_min=extreme(np.linalg.norm(lh-rh, axis=1), np.argmin),
                left_wrist_above_head_end_max=extreme(lh[:, 1]-p[:, idx('HeadEnd'), 1], np.argmax),
                right_wrist_above_head_end_max=extreme(rh[:, 1]-p[:, idx('HeadEnd'), 1], np.argmax))


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def review(mode, v1, run):
    """Build only in a new directory; verify is read-only and replays every source join and decode."""
    check(mode in ('build', 'verify'), 'expected build or verify')
    build = mode == 'build'
    v1, run = Path(v1).resolve(strict=True), Path(run).resolve(strict=True)
    root = checked(run, 'historical-review')
    check(not root.is_relative_to(v1) and not v1.is_relative_to(root), 'input and output overlap')
    if build:
        check(not root.exists(), 'historical-review already exists; no overwrite, use verify')
    else:
        check(checked(root, 'done.json').is_file(), 'incomplete review: no done.json')
    check(all(shutil.which(t) for t in ('ffmpeg', 'ffprobe')), 'requires CPU ffmpeg and ffprobe')
    groups, inputs = inventory(v1)
    if build:
        root.mkdir()  # exclusive ownership; an interruption never publishes a done marker
        for name in ('public', 'private', 'private/units', 'private/source'):
            checked(root, name).mkdir()
    hashes = {}

    def material(rel, data=None):
        path = checked(root, rel)
        if data is not None:
            if build:
                with path.open('xb') as f:
                    f.write(data)
            check(path.read_bytes() == data, f'{rel}: evidence mismatch')
        hashes[rel] = audit.digest(path)

    # Freeze build provenance and reconstructable script bytes; verifying a package
    # does not execute its archived sources. The key and private provenance never ship.
    protocol_rel = 'private/protocol.json'
    scripts = [Path(m.__file__) for m in (sys.modules[__name__], renderer, audit)]
    if build:
        import PIL
        protocol = dict(schema='kimodo.historical-review.protocol.v1', secret_hex=secrets.token_hex(32),
                        argv=[sys.executable, *sys.orig_argv[1:]], cwd=str(Path.cwd()), source=str(v1),
                        budget='51 historical trios and 3 constructed calibration trios; CPU only',
                        handoff='public only; save completed ballots outside this immutable directory',
                        versions=dict(python=sys.version, pillow=PIL.__version__, numpy=np.__version__,
                                      **{t: subprocess.check_output([t, '-version']).decode().splitlines()[0]
                                         for t in ('ffmpeg', 'ffprobe')}),
                        limits='Regression, not held-out evaluation. Geometry checks only the constructed controls. '
                               'No human reviews, no VRM, no regenerated model outputs, no acceptance decision.')
        material(protocol_rel, canonical(protocol))
    else:
        protocol = audit.read(checked(root, protocol_rel))
        material(protocol_rel)
    secret = bytes.fromhex(protocol['secret_hex'])
    check(len(secret) == 32, 'bad private assignment key')
    for script in scripts:
        material('private/source/' + script.name, script.read_bytes() if build else None)
    material('private/input-hashes.json', canonical(inputs))
    material('private/rubric.json', canonical(dict(action_cues=CUES, rubric=RUBRIC)))

    anchor = next(g for g in groups if (g['dataset'], g['row']['id'], g['seed']) == ('xval','xval-roll-forward',17))
    anchor_dir = anchor['folders']['ref']
    anchor_clip = renderer.read_clip(v1/anchor_dir, anchor['row'], anchor['seed'], 'ref')
    controls = make_controls(anchor_clip)
    jobs = []
    for g in groups:
        row = g['row']
        unit, order = assignment(secret, (g['dataset'], row['id'], g['seed']), renderer.ARMS)
        jobs.append((unit, order, g, None))
    for case, variants in controls.items():
        unit, order = assignment(secret, ('control', case), tuple(variants))
        jobs.append((unit, order, None, case))
    check(len({j[0] for j in jobs}) == len(jobs), 'duplicate reviewer unit')
    public = []
    counts = {d: 0 for d, *_ in DATASETS}
    frames = control_count = historical_panels = recorded_count = 0
    ffmpeg = shutil.which('ffmpeg')
    for unit, order, g, case in sorted(jobs):  # source order also hidden in filesystem creation times
        panels, clips = {}, []
        if g:
            row, seed, dataset = g['row'], g['seed'], g['dataset']
            prompt, cue = row['en'], CUES[row['id']]
            private = dict(kind='historical', dataset=dataset, id=row['id'], seed=seed, seconds=row['seconds'])
            for label, arm in zip(renderer.LABELS, order):
                folder = g['folders'][arm]
                clip = renderer.read_clip(v1/folder, row, seed, arm)
                generation = audit.read(v1/folder/'motion.json')['generation']
                audit.fields(generation, dict(id=row['id'], arm=arm, seed=seed, steps=100, constraintCount=0,
                             model='Kimodo-SOMA-RP-v1.2', modelRevision='0e3c89024d896064b6851201373f7834784a386b'), folder)
                panels[label] = dict(arm=arm, folder=folder, generation=generation, geometry=measurements(clip),
                                     source_sha256={n: inputs[f'{folder}/{n}'] for n in audit.FILES},
                                     journal_sha256=g['journal_hashes'][arm])
                recorded_count += len(g['journal_hashes'][arm])
                clips.append(clip)
            counts[dataset] += 1
            historical_panels += len(panels)
        else:
            prompt, cue = CONTROL_PROMPTS[case], CUES[case]
            private = dict(kind='analytic_calibration', case=case, anchor_directory=anchor_dir, anchor_frame=0,
                           anchor_npz_sha256=inputs[f'{anchor_dir}/motion.npz'])
            for label, intended in zip(renderer.LABELS, order):
                clip = controls[case][intended]
                observed, values = control_observation(case, clip)
                check(observed == intended, f'{case}/{intended}: control geometry classified as {observed}')
                panels[label] = dict(constructed=intended, geometrically_observed=observed, measures=values,
                                     geometry=measurements(clip))
                clips.append(clip)
                control_count += 1
        check(all(c.names == clips[0].names and c.parents == clips[0].parents and c.fps == renderer.FPS
                  and len(c.positions) == len(clips[0].positions) for c in clips), 'mismatched panel skeleton/timing')
        filename = unit + '.mp4'
        rel = 'public/' + filename
        path = checked(root, rel)
        if build:
            renderer.encode(ffmpeg, path, clips, ['ok']*3, smoke=False)
        media = media_evidence(path, len(clips[0].positions))
        hashes[rel] = media['video_sha256']
        frames += media['decoded_frames']
        private.update(unit=unit, prompt=prompt, action_cue=cue, media=media, panels=panels)
        material('private/units/' + unit + '.json', canonical(private))
        public.append(dict(unit=unit, prompt=prompt, action_cue=cue, video=filename))

    check(counts == {d: n*len(renderer.SEEDS) for d, _, n, _ in DATASETS} and control_count == 9,
          'incomplete set of historical trios or controls')
    material('public/units.json', canonical(public))
    material('public/ballot.csv', ballots(public).encode())
    material('public/index.html', catalog(public).encode())
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual_files == set(hashes) | (set() if build else {'done.json'}), 'unexpected/missing package files')
    summary = dict(schema='kimodo.historical-review.done.v1', historical_trios=counts,
                   historical_panels=historical_panels, historical_source_artifacts=historical_panels*len(audit.FILES),
                   original_journal_artifact_hashes=recorded_count, control_panels_verified=control_count,
                   videos=len(public), decoded_comparison_frames=frames, human_ratings=0,
                   quality_decision='not assessed', file_sha256=hashes)
    data = canonical(summary)
    if build:
        pending = root/'done.partial'
        with pending.open('xb') as f:
            f.write(data)
        check(pending.read_bytes() == data, 'incomplete receipt write')
        pending.rename(root/'done.json')
    check((root/'done.json').read_bytes() == data, 'completion receipt mismatch')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['build','verify'])
    p.add_argument('--v1', type=Path, required=True, help='read-only v1 run; only xval and xval-hist are read')
    p.add_argument('--run', type=Path, required=True, help='existing v2 run; writes only historical-review/')
    args = p.parse_args()
    try:
        done = review(args.mode, args.v1, args.run)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as e:
        p.exit(1, f'Review {args.mode} failed: {e}\n')
    print(json.dumps({k: v for k, v in done.items() if k != 'file_sha256'}, sort_keys=True))


if __name__ == '__main__':
    main()
