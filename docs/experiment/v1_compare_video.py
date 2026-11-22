"""Render blinded Kimodo v1 motion comparisons on CPU (NumPy, Pillow, ffmpeg/libx264).

Run from the repository root for one complete frozen split (192 videos when all
reference/English/Korean takes are available):

  python docs/experiment/v1_compare_video.py --manifest DEV.jsonl --motions RUN/out \
      --output-dir RUN/blind-review --key-out PRIVATE/dev-video-key.json

Append --id ID and/or --seed 17 (repeatable) to select a subset. The motions
root contains ref, en, and ko directories. To reproduce the panel order, pass
the prior private artifact as --key-in and use a new --key-out. Share only the
blind directory (MP4s and report.json), never the mapping or source files.

For a rendering smoke test only, replace --manifest and --motions with
--smoke-clips CLIP_DIR CLIP_DIR CLIP_DIR. This is explicitly not comparable
evidence. No mode computes a human score or acceptance gate.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import hmac
from itertools import permutations
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
from tempfile import TemporaryFile
from zipfile import BadZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ARMS = ("ref", "en", "ko")
LABELS = "ABC"
KEY_SCHEMA = "kimodo.v1-comparison-video.key.v1"
SEEDS = (17, 43, 97)
FPS = 30
SIZE = 320  # per panel
AXES = np.eye(3) * 0.5  # metres
ORDERS = tuple(permutations(ARMS))
ANGLE, ELEVATION = np.radians((25, 12))
BASIS = np.array([[-np.cos(ANGLE), 0, np.sin(ANGLE)],
                  [np.sin(ELEVATION) * np.sin(ANGLE), -np.cos(ELEVATION),
                   np.sin(ELEVATION) * np.cos(ANGLE)]]).T
LIGHT = (218, 227, 239)
TRACE = (130, 222, 177)


@dataclass
class Clip:
    positions: np.ndarray
    names: tuple[str, ...]
    parents: tuple[int, ...]
    fps: float


def read_manifest(path):
    raw = path.read_bytes()
    rows = {}
    seen = set()
    split = None
    for line in raw.decode("utf-8-sig").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError("Expected one JSON object per manifest line")
        ident = row.get("id")
        if (not isinstance(ident, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", ident)
                or ident.lower() in seen):
            raise ValueError(f"Invalid or duplicate manifest id: {ident!r}")
        for lang in ("en", "ko"):
            if not isinstance(row.get(lang), str) or not row[lang].strip():
                raise ValueError(f"Missing {lang} prompt for {ident}")
        seconds = row.get("seconds")
        if type(seconds) not in (int, float) or not 0.5 <= seconds <= 15:
            raise ValueError(f"Invalid seconds for {ident}")
        row_split = row.get("split")
        if row_split not in ("dev", "test") or split not in (None, row_split):
            raise ValueError("Expected one dev or test manifest")
        split = row_split
        rows[ident] = row
        seen.add(ident.lower())
    if not rows:
        raise ValueError("Empty manifest")
    return rows, split, sha256(raw).hexdigest()


def maker_prompt(text):
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    return text if text[-1] in ".!?" else text + "."


def read_clip(folder, row=None, seed=None, arm=None):
    """Read native world joints; never recompute or repair a missing pose."""
    doc = json.loads((folder / "motion.json").read_bytes())
    if not isinstance(doc, dict) or any(doc.get(k) != v for k, v in {
        "schema": "kimodo-maker.soma-motion.v1", "skeleton": "SOMA-77", "units": "m",
        "up": "+Y", "forward": "+Z",
    }.items()):
        raise ValueError("Expected SOMA-77 world coordinates in metres, +Y up, +Z forward")
    geometry = doc.get("joints")
    if not isinstance(geometry, list) or len(geometry) != 77:
        raise ValueError("Expected 77 joint definitions")
    names, parents = [], []
    for i, joint in enumerate(geometry):
        if not isinstance(joint, dict):
            raise ValueError("Invalid joint definition")
        name, parent = joint.get("name"), joint.get("parent")
        if (not isinstance(name, str) or not name or name in names or type(parent) is not int
                or (parent != -1 if i == 0 else not 0 <= parent < i)):
            raise ValueError("Invalid names or hierarchy")
        names.append(name)
        parents.append(parent)
    if names[0] != "Hips":
        raise ValueError("Expected Hips as the root joint")
    fps, total = doc.get("fps"), doc.get("frameCount")
    if (type(fps) not in (int, float) or not 1 <= fps <= 120
            or type(total) is not int or not 1 <= total <= 1800):
        raise ValueError("Invalid FPS/frame count")
    if row is not None:
        text = maker_prompt(row["ko" if arm == "ko" else "en"])
        generation = doc.get("generation")
        if (not isinstance(generation, dict) or type(generation.get("seed")) is not int
                or generation["seed"] != seed or maker_prompt(generation.get("prompt")) != text
                or type(generation.get("duration")) not in (int, float)
                or generation.get("duration") != row["seconds"]):
            raise ValueError("Generation seed, prompt or duration disagrees with manifest")
        if generation.get("id", row["id"]) != row["id"] or generation.get("arm", arm) != arm:
            raise ValueError("Generation ID or arm disagrees with its expected identity")
        if fps != FPS or total != round(row["seconds"] * FPS):
            raise ValueError("FPS or frame count disagrees with manifest")
    with np.load(folder / "motion.npz", allow_pickle=False) as motion:
        joints = motion["posed_joints"]
    if (joints.shape != (total, 77, 3) or joints.dtype.kind != "f"
            or not np.isfinite(joints).all()):
        raise ValueError("Invalid posed_joints shape, dtype or coordinates")
    return Clip(joints.astype(np.float64), tuple(names), tuple(parents), fps)


def shared_view(clips):
    """One orthographic world camera, centered/scaled once over the entire trio."""
    axes = np.vstack((np.zeros(3), AXES)) @ BASIS
    posed = [(clip.positions @ BASIS if clip else None) for clip in clips]
    ground = [((clip.positions[:, 0] * [1, 0, 1]) @ BASIS if clip else None) for clip in clips]
    points = np.concatenate([axes, *(pts.reshape(-1, 2) for pts in posed + ground if pts is not None)])
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = min(264 / max(1., hi[0] - lo[0]), 225 / max(1., hi[1] - lo[1]))
    midpoint = (lo + hi) / 2

    def pixel(pts):
        return np.rint((pts - midpoint) * scale + (160, 166)).astype(int)

    return (pixel(axes), [pixel(p) if p is not None else None for p in posed],
            [pixel(p) if p is not None else None for p in ground])


def video_frames(clips, statuses, *, smoke):
    axes, posed, ground = shared_view(clips)
    axis_palette = ((224, 138, 128), (157, 221, 152), (139, 177, 228))
    font = ImageFont.load_default()
    base = Image.new("RGB", (3 * SIZE, SIZE), (19, 25, 35))
    draw = ImageDraw.Draw(base)
    paths, edges = [], []
    for index, (label, clip, status) in enumerate(zip(LABELS, clips, statuses)):
        left = index * SIZE
        if index:
            draw.line((left, 0, left, SIZE), fill=(53, 63, 80))
        draw.rounded_rectangle((left + 15, 12, left + 45, 37), 5, fill=(43, 57, 77))
        draw.text((left + 30, 24), label, fill=LIGHT, font=font, anchor="mm")
        if smoke:
            draw.text((left + 58, 24), "SMOKE TEST - NOT COMPARABLE", fill=LIGHT, font=font, anchor="lm")
        if clip is None:
            draw.text((left + SIZE // 2, SIZE // 2), status.upper(), fill=LIGHT, font=font, anchor="mm")
            paths.append(None)
            edges.append(None)
            continue
        origin = tuple(axes[0] + (left, 0))
        for name, pos, color in zip("XYZ", axes[1:], axis_palette):
            end = tuple(pos + (left, 0))
            draw.line((origin, end), fill=color)
            draw.text((end[0] + 2, end[1] + 1), f"+{name}", fill=color, font=font)
        draw.text((left + 15, 301), "0.5 m axes | ground root trail", fill=(159, 175, 197), font=font)
        paths.append([tuple(pt) for pt in ground[index] + (left, 0)])
        edges.append([(child, parent, (121, 212, 243) if name.startswith("Left") else
                       (245, 175, 114) if name.startswith("Right") else LIGHT)
                      for child, (parent, name) in enumerate(zip(clip.parents, clip.names)) if parent >= 0])
    nframes = max(len(clip.positions) for clip in clips if clip is not None)
    for frame in range(nframes):
        image = base.copy()
        draw = ImageDraw.Draw(image)
        for index, clip in enumerate(clips):
            if clip is None:
                continue
            if frame >= len(clip.positions):
                draw.text((index * SIZE + SIZE // 2, SIZE // 2), "ENDED", fill=LIGHT, font=font, anchor="mm")
                continue
            path = paths[index]
            if frame:
                draw.line(path[:frame + 1], fill=TRACE, width=2)
            x, y = path[frame]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=TRACE)
            points = [tuple(pt) for pt in posed[index][frame] + (index * SIZE, 0)]
            for child, parent, color in edges[index]:
                draw.line((points[child], points[parent]), fill=color, width=2)
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=LIGHT)
        yield image.tobytes()


def encode(ffmpeg, path, clips, statuses, *, smoke):
    fps = next(clip.fps for clip in clips if clip is not None)
    temporary = path.with_suffix(".partial.mp4")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
           "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{3 * SIZE}x{SIZE}",
           "-framerate", str(fps), "-i", "pipe:0", "-an", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "veryfast", "-threads", "2", "-crf", "20",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    try:
        with TemporaryFile(dir=path.parent) as errors, subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errors) as proc:
            try:
                with proc.stdin:
                    for frame in video_frames(clips, statuses, smoke=smoke):
                        proc.stdin.write(frame)
            except BrokenPipeError:
                pass  # report ffmpeg's actual error below
            if proc.wait():
                errors.seek(0)
                raise RuntimeError(errors.read().decode("utf-8", "replace").strip() or "ffmpeg failed")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def panel_order(secret, split, ident, seed):
    payload = json.dumps(["kimodo-v1-compare", split, ident, seed], separators=(",", ":")).encode()
    digest = hmac.digest(secret, payload, "sha256")
    return ORDERS[int.from_bytes(digest, "big") % len(ORDERS)]


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_journal(path, arm):
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    records = {}
    # Ignore a possibly torn last write, but require valid complete records.
    for line in raw.split(b"\n")[:-1]:
        rec = json.loads(line)
        if (not isinstance(rec, dict) or rec.get("arm") != arm
                or not isinstance(rec.get("id"), str) or type(rec.get("seed")) is not int
                or rec.get("status") not in ("started", "ok", "failed")):
            raise ValueError(f"Malformed request journal: {path}")
        key = (rec["id"], rec["seed"])
        if records.get(key, {}).get("status") in ("ok", "failed"):
            raise ValueError(f"Conflicting terminal request journal record for {key}: {path}")
        records[key] = rec
    return records


def render_group(args, ffmpeg, secret, split, ident, seed, sources, *, row=None, journals=None):
    smoke = row is None
    name = ident if smoke else f"{ident}-{seed}"
    order = panel_order(secret, split, ident, seed)
    public = dict(id=ident, seed=seed, panels={}, video=None)
    private = dict(id=ident, seed=seed, panels={})
    clips = []
    for label, arm in zip(LABELS, order):
        folder = sources[arm]
        detail = dict(directory=str(folder))
        if not smoke:
            detail["arm"] = arm
        log = journals.get(arm) if journals is not None else None
        rec = log.get((ident, seed), {}) if log is not None else None
        clip = None
        try:
            if rec is not None and rec.get("status") == "failed":
                raise ValueError(f"Generation failed: {rec.get('error', 'no error recorded')}")
            if rec is not None and rec.get("status") != "ok":
                raise FileNotFoundError("No terminal success recorded in generation journal")
            clip = read_clip(folder, row, seed, arm)
        except FileNotFoundError as exc:
            status = "missing"
            detail["reason"] = str(exc)
        except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as exc:
            status = "failed"
            detail["reason"] = str(exc)
        else:
            status = "ok"
        public["panels"][label] = detail["status"] = status
        private["panels"][label] = detail
        clips.append(clip)
    rates = {clip.fps for clip in clips if clip is not None}
    if len(rates) > 1:
        public["video_status"] = "encode_failed"
        private["encode_error"] = "Input FPS differ; refusing to resample motion"
    elif rates:
        filename = f"{name}.mp4"
        try:
            encode(ffmpeg, args.output_dir / filename, clips, list(public["panels"].values()), smoke=smoke)
        except (OSError, RuntimeError) as exc:
            public["video_status"] = "encode_failed"
            private["encode_error"] = str(exc)
        else:
            public.update(video=filename, video_status="rendered")
    else:
        public["video_status"] = "no_motion"
    return public, private


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path, help="frozen dev/test JSONL")
    inputs.add_argument("--smoke-clips", type=Path, nargs=3, metavar="CLIP_DIR",
                        help="three distinct clips, for an explicitly non-comparable renderer smoke test")
    parser.add_argument("--motions", type=Path, help="root containing ref/en/ko, required with --manifest")
    parser.add_argument("--output-dir", type=Path, required=True, help="new public blind-review directory")
    parser.add_argument("--key-out", type=Path, required=True, help="new private JSON outside the blind directory")
    parser.add_argument("--key-in", type=Path, help="prior private JSON to reproduce the panel assignments")
    parser.add_argument("--id", action="append", help="one expected ID; repeat to select more")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append", help="one fixed v1 seed; repeat to select more")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg with the CPU libx264 encoder")
    args = parser.parse_args()
    try:
        journals = None
        smoke = args.smoke_clips is not None
        if smoke:
            if args.motions or args.id or args.seed:
                raise ValueError("--motions, --id and --seed require --manifest")
            folders = sorted(path.resolve() for path in args.smoke_clips)
            if len(set(folders)) != 3:
                raise ValueError("Supply three distinct directories for --smoke-clips")
            mode, split, digest = "smoke", None, None
            selected = [("render-smoke", None, None, dict(zip(ARMS, folders)))]
        else:
            if args.motions is None:
                raise ValueError("--manifest requires --motions")
            mode = "manifest"
            rows, split, digest = read_manifest(args.manifest)
            if args.id and set(args.id) - rows.keys():
                raise ValueError("Selected ID is not in the manifest")
            root = args.motions.resolve()
            config_path = root / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_bytes())
                if (not isinstance(config, dict) or config.get("manifest_sha256") != digest
                        or config.get("split") != split or config.get("pilot") is not False):
                    raise ValueError("Generation config does not match the non-pilot manifest")
            journals = {arm: read_journal(root / arm / "requests.jsonl", arm) for arm in ARMS}
            selected = [(ident, seed, row, {arm: root / arm / f"{ident}-{seed}" for arm in ARMS})
                        for ident, row in rows.items() if not args.id or ident in args.id
                        for seed in SEEDS if not args.seed or seed in args.seed]
        args.output_dir = args.output_dir.resolve()
        args.key_out = args.key_out.resolve()
        if args.key_out.is_relative_to(args.output_dir):
            raise ValueError("Keep --key-out outside the public blind-review directory")
        if args.output_dir.exists() or args.key_out.exists():
            raise ValueError("Both the output directory and private key path must be new")
        ffmpeg = shutil.which(args.ffmpeg)
        if not ffmpeg:
            raise ValueError("ffmpeg executable not found")
        if args.key_in:
            old_key = json.loads(args.key_in.read_bytes())
            if (not isinstance(old_key, dict) or old_key.get("schema") != KEY_SCHEMA
                    or old_key.get("mode") != mode or old_key.get("manifest_sha256") != digest
                    or not isinstance(old_key.get("secret"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", old_key["secret"])):
                raise ValueError("--key-in does not match the current manifest or mode")
            secret = bytes.fromhex(old_key["secret"])
        else:
            secret = secrets.token_bytes(32)
        args.output_dir.mkdir(parents=True)
        args.key_out.parent.mkdir(parents=True, exist_ok=True)
        report = dict(schema="kimodo.v1-comparison-video.report.v1", mode=mode, human_score=False,
                      note="Visual evidence only; not a human score or an acceptance gate.",
                      split=split, manifest_sha256=digest, expected_groups=len(selected),
                      expected_panels=3 * len(selected), groups=[])
        if smoke:
            report.update(comparable=False, note="CPU rendering smoke test from unrelated takes. "
                          "Not comparable and not a human score or an acceptance gate.")
        key = dict(schema=KEY_SCHEMA, mode=mode, manifest_sha256=digest,
                   secret=secret.hex(), groups=[])
        for ident, seed, row, sources in selected:
            public, private = render_group(args, ffmpeg, secret, split, ident, seed, sources,
                                           row=row, journals=journals)
            report["groups"].append(public)
            key["groups"].append(private)
        report["panel_counts"] = dict(Counter(status for group in report["groups"]
                                              for status in group["panels"].values()))
        report["video_counts"] = dict(Counter(group["video_status"] for group in report["groups"]))
        write_json(args.key_out, key)
        write_json(args.output_dir / "report.json", report)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"{len(selected)} expected groups: {report['video_counts']}; see report.json")
    return int(bool(report["video_counts"].get("encode_failed")))


if __name__ == "__main__":
    sys.exit(main())
