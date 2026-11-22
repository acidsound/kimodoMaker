"""Build the blind dev review package from generated dev units; CPU only, no scoring here."""
import argparse
from hashlib import sha256
import hmac
import html
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v1_compare_video import AXES, BASIS, Clip, LIGHT, SIZE, TRACE, read_clip

SEEDS = (17, 43, 97)
RUBRIC = (
    'Judge each panel independently with pass, fail or unjudgeable on two axes. '
    'Action adherence: pass only if every explicit action, body side, direction, path, amount and pace cue in the '
    'caption is visible; fail on a visible omission, contradiction or reversed side. '
    'Visual quality: pass only when balance, contact and timing are coherent, with no material foot slip, jerk, pop '
    'or discontinuity. A wrong action may look good and a correct one may look bad. '
    'Use unjudgeable when the view, caption ambiguity or a failed render blocks a decision. '
    'Do not infer which conditioning produced a panel and do not discuss panels before submitting. '
    'Cyan bones are the actor\'s anatomical left, orange are right; camera, scale and ground trail are shared.')


def frames_pair(clips):
    axes = np.vstack((np.zeros(3), AXES)) @ BASIS
    posed = [clip.positions @ BASIS for clip in clips]
    ground = [(clip.positions[:, 0] * [1, 0, 1]) @ BASIS for clip in clips]
    points = np.concatenate([axes, *(p.reshape(-1, 2) for p in posed + ground)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = min(264 / max(1., hi[0] - lo[0]), 225 / max(1., hi[1] - lo[1]))
    mid = (lo + hi) / 2

    def pixel(pts):
        return np.rint((pts - mid) * scale + (160, 166)).astype(int)

    axis, posed, ground = pixel(axes), [pixel(p) for p in posed], [pixel(g) for g in ground]
    font = ImageFont.load_default()
    base = Image.new('RGB', (2 * SIZE, SIZE), (19, 25, 35))
    draw = ImageDraw.Draw(base)
    palette = ((224, 138, 128), (157, 221, 152), (139, 177, 228))
    edges = [[(child, parent, (121, 212, 243) if name.startswith('Left') else
               (245, 175, 114) if name.startswith('Right') else LIGHT)
              for child, (parent, name) in enumerate(zip(clip.parents, clip.names)) if parent >= 0] for clip in clips]
    nframes = max(len(clip.positions) for clip in clips)
    for index, label in enumerate('AB'):
        left = index * SIZE
        if index:
            draw.line((left, 0, left, SIZE), fill=(53, 63, 80))
        draw.rounded_rectangle((left + 15, 12, left + 45, 37), 5, fill=(43, 57, 77))
        draw.text((left + 30, 24), label, fill=LIGHT, font=font, anchor='mm')
        origin = tuple(axis[0] + (left, 0))
        for name, pos, color in zip('XYZ', axis[1:], palette):
            end = tuple(pos + (left, 0))
            draw.line((origin, end), fill=color)
            draw.text((end[0] + 2, end[1] + 1), '+' + name, fill=color, font=font)
        draw.text((left + 15, 301), '0.5 m axes | ground root trail', fill=(159, 175, 197), font=font)
    for frame in range(nframes):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for index, clip in enumerate(clips):
            if frame >= len(clip.positions):
                draw.text((index * SIZE + SIZE // 2, SIZE // 2), 'ENDED', fill=LIGHT, font=font, anchor='mm')
                continue
            path = [tuple(pt) for pt in ground[index] + (index * SIZE, 0)]
            draw.line(path[:frame + 1], fill=TRACE, width=2)
            x, y = path[frame]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=TRACE)
            pts = [tuple(pt) for pt in posed[index][frame] + (index * SIZE, 0)]
            for child, parent, color in edges[index]:
                draw.line((pts[child], pts[parent]), fill=color, width=2)
            x, y = pts[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=LIGHT)
        yield image.tobytes()


def encode_pair(ffmpeg, path, clips):
    temporary = path.with_suffix('.partial.mp4')
    cmd = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
           '-video_size', f'{2 * SIZE}x{SIZE}', '-framerate', str(clips[0].fps), '-i', 'pipe:0', '-an',
           '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'veryfast', '-threads', '2', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    with __import__('tempfile').TemporaryFile(dir=path.parent) as errors, \
            subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors) as proc:
        try:
            with proc.stdin:
                for frame in frames_pair(clips):
                    proc.stdin.write(frame)
        except BrokenPipeError:
            pass
        if proc.wait():
            errors.seek(0)
            raise RuntimeError(errors.read().decode('utf-8', 'replace').strip() or 'ffmpeg failed')
    temporary.rename(path)


def assignment(secret, group, seed):
    payload = json.dumps(['dev-review', group, seed], separators=(',', ':')).encode()
    return ('candidate', 'ref') if hmac.digest(secret, payload, 'sha256')[-1] & 1 else ('ref', 'candidate')


def unit_hashes(run, group, seed, arm):
    folder = run / 'units' / f'{arm}.{group}.{seed}'
    return {n: sha256((folder / n).read_bytes()).hexdigest() for n in ('motion.json', 'motion.npz')}


def build(run, out):
    run, out = run.resolve(strict=True), out.resolve()
    if out.exists():
        raise ValueError('dev review already exists; use verify')
    dev_rows = [json.loads(line) for line in (run / 'dev-rows.jsonl').read_bytes().decode('utf-8-sig').splitlines()]
    if len({r['group_id'] for r in dev_rows}) != 64:
        raise ValueError('unexpected dev rows')
    out.mkdir(parents=True)
    (out / 'public').mkdir()
    (out / 'private').mkdir()
    (out / 'private' / 'units').mkdir()
    secret = secrets.token_bytes(32)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg required')
    rows = {r['group_id']: r for r in dev_rows}
    units = run / 'units'
    entries = []
    missing = []
    decoded = frames = 0
    for group in sorted(rows):
        row = rows[group]
        caption = row['en']
        for seed in SEEDS:
            arms = assignment(secret, group, seed)
            folders = []
            ok = True
            for arm in arms:
                folder = units / f'{arm}.{group}.{seed}'
                if not (folder / 'motion.npz').is_file():
                    ok = False
                    missing.append(f'{arm}.{group}.{seed}')
                folders.append(folder)
            if not ok:
                continue
            clips = [read_clip(f) for f in folders]
            if clips[0].names != clips[1].names or len(clips[0].positions) != len(clips[1].positions):
                raise ValueError('skeleton/timing mismatch')
            name = f'{group}.{seed}.mp4'
            path = out / 'public' / name
            encode_pair(ffmpeg, path, clips)
            probe = json.loads(subprocess.check_output(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_streams', '-of', 'json',
                 str(path)], encoding='utf-8'))
            stream = probe['streams'][0]
            if stream['codec_name'] != 'h264' or stream['width'] != 2 * SIZE or stream['height'] != SIZE:
                raise ValueError('unexpected video encoding')
            frames += int(stream['nb_read_frames'])
            decoded += 1
            (out / 'private' / 'units' / f'{group}.{seed}.json').write_text(json.dumps(dict(
                group=group, seed=seed, panels=dict(A=arms[0], B=arms[1]),
                caption=caption, video=name, source_sha256=unit_hashes(run, group, seed, arms[0]) |
                unit_hashes(run, group, seed, arms[1])), sort_keys=True, indent=2) + '\n', encoding='utf-8')
            entries.append(dict(unit=f'{group}.{seed}', caption=caption, video=name))
    ballot = 'unit,caption,a_action,a_quality,b_action,b_quality,notes\n'
    for e in entries:
        ballot += f'"{e["unit"]}","{e["caption"]}",,,,,\n'
    (out / 'public' / 'ballot.csv').write_text(ballot, encoding='utf-8')
    catalog = ''.join(f'<section><h2>{html.escape(e["unit"])}</h2><p>{html.escape(e["caption"])}</p>'
                      f'<video controls preload="none" width="640" src="{e["video"]}"></video></section>\n' for e in entries)
    (out / 'public' / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Blind dev motion review</title>'
        '<style>body{font:16px system-ui;max-width:700px;margin:2em auto}video{max-width:100%}'
        'section{padding:1em 0;border-top:1px solid #aaa}h2{font:1em monospace}</style>'
        '<h1>Blind dev motion review</h1><p>' + html.escape(RUBRIC) + '</p><p>'
        '<a href="ballot.csv" download>Ballot (CSV)</a></p>' + catalog + '</html>\n', encoding='utf-8')
    (out / 'private' / 'key.json').write_text(json.dumps(dict(secret_hex=secret.hex())), encoding='utf-8')
    summary = dict(schema='kimodo.dev-review.v1', videos=len(entries), frames=frames, missing=missing,
                   human_ratings=0, decision='not assessed', rows_hash=file_sha_of(run / 'dev-rows.jsonl'),
                   units_root=str(run / 'units'), rubric=RUBRIC,
                   scope='development only; blinded arms, no ratings collected, not a release gate')
    data = (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    pending = out / 'done.partial'
    with pending.open('xb') as f:
        f.write(data)
    pending.rename(out / 'done.json')
    print(json.dumps({k: v for k, v in summary.items() if k != 'rubric'}, sort_keys=True))


def file_sha_of(path):
    with path.open('rb') as f:
        return sha256(f.read()).hexdigest()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    build(a.run, a.out)
