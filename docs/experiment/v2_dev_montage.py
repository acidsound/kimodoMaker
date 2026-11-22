"""Extract blind montages from the dev review videos; ratings stay unblinded until the key is used."""
import argparse
from hashlib import sha256
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SAMPLE_SEED = 96173
FRAMES = (15, 45, 75)
EXPOSED = {'039f75b5e790c62e4842dab5'}


def units_of(review):
    return sorted(p.stem for p in (review / 'public').glob('*.mp4'))


def sample(review, n, seed=SAMPLE_SEED):
    ids = [u for u in units_of(review) if u.split('.')[0] not in EXPOSED]
    if len(ids) < n:
        raise ValueError('not enough unexposed units')
    rng = random.Random(seed)
    return rng.sample(ids, n)


def frames(path, timestamps):
    cmd = [shutil.which('ffmpeg') or 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(path)]
    for i, t in enumerate(timestamps):
        cmd += ['-vf', f"select=eq(n\\,{t})", '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'rawvideo',
                '-pix_fmt', 'rgb24', '-']
    data = subprocess.run(cmd, capture_output=True, check=True).stdout
    size = 640 * 320 * 3
    if len(data) != size * len(timestamps):
        raise ValueError(f'frame capture failed for {path}')
    return [data[i * size:(i + 1) * size] for i in range(len(timestamps))]


def montage(review, unit, out):
    video = review / 'public' / f'{unit}.mp4'
    shots = frames(video, FRAMES)
    width, height = 320, 320
    from PIL import Image
    image = Image.new('RGB', (width * len(FRAMES), height * 2), (12, 16, 24))
    for row, name in enumerate(('A', 'B')):
        for col, raw in enumerate(shots):
            panel = Image.frombytes('RGB', (640, 320), raw).crop((row * 320, 0, (row + 1) * 320, 320))
            image.paste(panel, (col * width, row * height))
    image.save(out, format='PNG')
    return sha256(out.read_bytes()).hexdigest()


def build(review, out, n):
    review, out = review.resolve(strict=True), out.resolve()
    if out.exists():
        raise ValueError('montage directory already exists')
    out.mkdir(parents=True)
    ids = sample(review, n)
    rows = []
    for unit in ids:
        digest = montage(review, unit, out / f'{unit}.png')
        rows.append(dict(unit=unit, montage=f'{unit}.png', montage_sha256=digest, rows='top=A, bottom=B',
                         frames=str(FRAMES)))
    (out / 'units.json').write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8')
    (out / 'ratings.csv').write_text('unit,a_action,a_quality,b_action,b_quality\n' +
                                     ''.join(f'{u},,,,,\n' for u in ids), encoding='utf-8')
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', type=int, default=12)
    a = ap.parse_args()
    rows = build(a.review, a.out, a.n)
    print(json.dumps(dict(montages=len(rows), out=str(a.out.resolve()))))
