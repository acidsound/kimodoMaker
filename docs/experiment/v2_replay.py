"""P1.1: 기존 자연 회귀 테이크를 CPU로 감사한다. 모델 로드·재생성·채점 없음."""
import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = ('ref', 'en', 'ko')
SEEDS = (17, 43, 97)
FILES = ('motion.npz', 'motion.json', 'motion.bvh')
SETS = {'xval': ('xval/manifest.jsonl', 5), 'xval-hist': ('xval/hist-manifest.jsonl', 12)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def fingerprints(directory, recorded):
    require(isinstance(recorded, dict) and set(recorded) <= set(FILES),
            f'{directory}: invalid artifact hash map')
    observed = {}
    for name in FILES:
        path = directory / name
        require(path.stat().st_size > 0, f'{path}: empty artifact')
        observed[name] = digest(path)
        if name in recorded:
            require(observed[name] == recorded[name], f'{path}: recorded SHA-256 mismatch')
    return observed


def unique_fields(pairs):
    obj = dict(pairs)
    require(len(obj) == len(pairs), 'duplicate JSON field')
    return obj


def decode(raw):
    def invalid(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    return json.loads(raw, object_pairs_hook=unique_fields, parse_constant=invalid)


def read(path):
    return decode(path.read_bytes())


def lines(path, *, journal_file=False):
    raw = path.read_bytes()
    if journal_file:
        require(raw.endswith(b'\n'), f'{path}: torn or empty journal')
    for i, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        try:
            obj = decode(line)
            require(isinstance(obj, dict), 'expected object')
        except ValueError as exc:
            raise ValueError(f'{path}:{i}: {exc}') from exc
        yield obj


def manifest(path, count):
    rows, seen = {}, set()
    for row in lines(path):
        ident = row.get('id')
        require(isinstance(ident, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,100}', ident)
                and ident.casefold() not in seen, f'{path}: invalid/duplicate id {ident!r}')
        require(row.get('split') == 'xval' and type(row.get('seconds')) in (float, int)
                and .5 <= row['seconds'] <= 15 and type(row.get('kind')) is int
                and all(isinstance(row.get(k), str) and row[k].strip() for k in ('en', 'ko')),
                f'{path}: invalid manifest fields for {ident}')
        seen.add(ident.casefold())
        rows[ident] = row
    require(len(rows) == count, f'{path}: expected {count} distinct prompts; got {len(rows)}')
    return rows


def journal(path, arm, rows, steps, fps):
    expected = {(ident, seed) for ident in rows for seed in SEEDS}
    starts, oks = {}, {}
    for rec in lines(path, journal_file=True):
        ident, seed = key = (rec.get('id'), rec.get('seed'))
        where = f'{path}: {key}'
        require(type(seed) is int and key in expected and rec.get('arm') == arm,
                f'{where}: unexpected key/arm')
        row = rows[ident]
        require(type(rec.get('steps')) is int and rec['steps'] == steps
                and rec.get('seconds') == row['seconds'] and type(rec.get('seconds')) in (int, float)
                and rec.get('kind') == row['kind'] and type(rec.get('kind')) is int,
                f'{where}: options do not match manifest/config')
        if rec.get('status') == 'started':
            require(key not in starts, f'{where}: duplicate start')
            starts[key] = rec
            continue
        require(rec.get('status') == 'ok' and key in starts and key not in oks,
                f'{where}: missing start, duplicate or non-ok terminal')
        for field in ('session_id', 'started_utc', 'started_monotonic_ns'):
            require(field in rec and starts[key].get(field) == rec[field], f'{where}: mismatched {field}')
        require(rec.get('fps') == fps and type(rec.get('frame_count')) is int
                and rec['frame_count'] == round(fps * row['seconds']), f'{where}: frame count/fps mismatch')
        require(rec.get('artifacts') == {n: f'{ident}-{seed}/{n}' for n in FILES},
                f'{where}: artifact path/key mismatch')
        oks[key] = rec
    require(set(oks) == expected, f'{path}: missing terminal(s) {sorted(expected - oks.keys())}')
    return oks


def coordinates(path: Path, frames: int):
    with np.load(path, allow_pickle=False) as data:
        if len(set(data.files)) != len(data.files):
            raise ValueError(f'{path}: duplicate NPZ member')
        def coord(name, shape):
            if name not in data:
                raise ValueError(f'{path}: missing {name}')
            value = data[name]
            if value.shape != shape or value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
                raise ValueError(f'{path}: malformed or nonfinite {name}')
            return value.astype(np.float64)
        if type(frames) is not int or frames < 1:
            raise ValueError(f'{path}: invalid frame count {frames!r}')
        joints = coord('posed_joints', (frames, 77, 3))
        root = coord('root_positions', (frames, 3))
    return joints, root


def load_pose(path: Path, frames: int):
    """Return float64 SOMA-77 joint positions minus each frame's NPZ root (metres)."""
    joints, root = coordinates(path, frames)
    return joints - root[:, None, :]


def rms_cm(a, b):
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (77, 3) or len(a) == 0:
        raise ValueError('RMS requires matching nonempty frames × 77 × 3 arrays')
    delta = np.asarray(a, dtype=np.float64) - b
    value = 100 * np.sqrt(np.mean(np.sum(delta**2, axis=-1)))
    if not np.isfinite(value):
        raise ValueError('nonfinite RMS')
    return float(value)


def fields(actual, expected, where):
    for k, v in expected.items():
        require(k in actual and actual[k] == v and not (isinstance(v, (bool, int, float)) and
                isinstance(v, bool) != isinstance(actual[k], bool)), f'{where}: mismatch: {k}')


def json_soma_positions(data, frames, fps, path):
    """Integer-frame, root-motion-on FK before VRM, in the NPZ world origin."""
    from scipy.spatial.transform import Rotation

    fields(data, dict(schema='kimodo-maker.soma-motion.v1', skeleton='SOMA-77', units='m',
                      up='+Y', forward='+Z', quaternionOrder='xyzw', frameCount=frames, fps=fps), path)
    joints = data['joints']
    require(len(joints) == 77 and len({j['name'] for j in joints}) == 77 and
            joints[0]['name'] == 'Hips', f'{path}: bad skeleton')
    def finite(array, shape):
        value = np.asarray(array)
        require(value.shape == shape and value.dtype.kind in 'fiu' and np.isfinite(value).all(),
                f'{path}: invalid JSON coordinates/rotations')
        return value.astype(np.float64)
    root = finite(data['rootPositions'], (frames, 3))
    quat = finite(data['localRotations'], (frames, 77*4)).reshape(-1, 4)
    offsets = finite([j['restOffset'] for j in joints], (77, 3))
    rot = Rotation.from_quat(quat).as_matrix().reshape(frames, 77, 3, 3)
    pos = np.empty((frames, 77, 3))
    pos[:, 0] = root
    for i, joint in enumerate(joints):
        parent = joint['parent']
        require(type(parent) is int and (parent == -1 if i == 0 else 0 <= parent < i),
                f'{path}: invalid parent at joint {i}')
        if i:
            pos[:, i] = pos[:, parent] + rot[:, parent] @ offsets[i]
            rot[:, i] = rot[:, parent] @ rot[:, i]
    return pos


def check_bvh(path, frames, fps):
    header, marker, motion = path.read_text(encoding='ascii').partition('\nMOTION\n')
    rows = motion.splitlines()
    require(marker and header.startswith('HIERARCHY\n') and len(rows) == frames+2 and
            rows[0].split() == ['Frames:', str(frames)] and rows[1].startswith('Frame Time:') and
            math.isclose(float(rows[1].partition(':')[2]), 1/fps, rel_tol=1e-5),
            f'{path}: BVH frames/fps mismatch')
    width = sum(int(s.split()[1]) for s in header.splitlines() if s.lstrip().startswith('CHANNELS '))
    require(width > 0 and all(len(s.split()) == width for s in rows[2:]), f'{path}: BVH channels mismatch')
    values = np.fromstring('\n'.join(rows[2:]), sep=' ')
    require(values.size == frames * width and np.isfinite(values).all(), f'{path}: invalid BVH data')


def inspect_clip(directory, generation, frames, fps, recorded):
    hashes = fingerprints(directory, recorded)
    jp = directory/'motion.json'
    data = read(jp)
    fields(data['generation'], generation, jp)
    fk = json_soma_positions(data, frames, fps, jp)
    joints, roots = coordinates(directory/'motion.npz', frames)
    check_bvh(directory/'motion.bvh', frames, fps)
    result = dict(artifact_sha256=hashes, recorded_artifact_sha256=recorded,
                  npz_to_soma_json_fk_rms_cm=rms_cm(joints, fk),
                  npz_to_soma_json_hips_max_cm=float(100*np.linalg.norm(fk[:, 0]-joints[:, 0], axis=-1).max()),
                  frame_count=frames, fps=fps, generation=data['generation'])
    return result, joints - roots[:, None, :], tuple(j['name'] for j in data['joints'])


# Pinned evidence identities, not expected results for the RMS calculation.
LOCK = '0d2931ce78520d28af9faae7532a7faa313e912d7e07e0a064aaa47f30079e87'
MANIFEST_HASHES = {'xval': 'caeab80c1572faaa61cab034326aabf987da59def802e1a42d03625bf7ed1eef',
                   'xval-hist': '736b02c844837693aee2671d5040756c6040be005e3f7663717a34ae450021e7'}
CHECKPOINT = 'd5742dcfbfd41c635d9d0c874937d0ed8112e4cedfbe227b19bcc3a62f9d8a07'


def source(root, name):
    path = root / name
    require(path.is_relative_to(root) and path.resolve() == path, f'{path}: redirected source')
    return path


def stats(values):
    return dict(pairs=len(values), min=min(values), median=float(np.median(values)), max=max(values))


def read_regression(v1):
    sources = {}

    def hashed(name, pin=None):
        path = source(v1, name)
        value = digest(path)
        require(pin is None or value == pin, f'{path}: SHA-256 mismatch')
        sources[name] = value
        return path

    lock = read(hashed('lock.json', LOCK))
    protocol = lock['protocol']
    hashed('adapter-mlp-04/adapter.safetensors', CHECKPOINT)
    hashed('adapter-mlp-04/run.json')
    runner = lock['assets'][r'docs\experiment\v1_generate.py']['sha256']
    # Use the frozen runner's pure formatting function only; model loading is never invoked.
    import v1_generate
    require(digest(Path(v1_generate.__file__)) == runner, 'archived v1 runner source hash mismatch')
    student_entry = lock['assets'][r'docs\experiment\v1_student.py']['sha256']
    model = [protocol['model'], protocol['model_revision']]
    sets, differences = {}, {arm: [] for arm in ARMS if arm != 'ref'}
    for group, (manifest_name, count) in SETS.items():
        rows = manifest(hashed(manifest_name, MANIFEST_HASHES[group]), count)
        config = read(hashed(f'{group}/config.json'))
        expected = count * len(SEEDS)
        fields(config, dict(schema='kimodo.v1-generate.config.v1', pilot=True, split='xval',
                            manifest_sha256=MANIFEST_HASHES[group], seeds=list(SEEDS), fps=30,
                            planned_requests=expected, steps=protocol['steps'], exports=True, limit=None,
                            constraints=protocol['constraints'], post_processing=protocol['postprocessing'],
                            motion_model=model, runner_sha256=runner), group)
        arms, journals, requests = {}, {}, []
        student = read(hashed(f'{group}/student.json'))
        fields(student, dict(backbone=[protocol['student'], protocol['student_revision']],
                             dtype=protocol['student_dtype'], checkpoint_sha256=CHECKPOINT,
                             checkpoint_run_json_sha256=sources['adapter-mlp-04/run.json'],
                             entry_sha256=student_entry), group)
        for arm in ARMS:
            base = f'{group}/{arm}'
            cfg = read(hashed(f'{base}/config.json'))
            fields(cfg, dict(arm=arm), base)
            require(Path(cfg['motion_snapshot'].replace('\\', '/')).name == model[1], f'{base}: revision mismatch')
            encoder = cfg['encoder']
            if arm == 'ref':
                fields(encoder, dict(wrapper='Kimodo LLM2VecEncoder'), base)
                enc = encoder
            else:
                require(encoder == student, f'{base}: student config mismatch')
                enc = {k: encoder[k] for k in ('backbone','dtype','checkpoint_sha256','entry_sha256')}
            log = f'{base}/requests.jsonl'
            oks = journal(hashed(log), arm, rows, config['steps'], config['fps'])
            expected_dirs = {f'{ident}-{seed}' for ident, seed in oks}
            found_dirs = {p.name for p in source(v1, base).iterdir() if p.is_dir() and p.name != 'sessions'}
            require(expected_dirs == found_dirs, f'{base}: archive directories differ from journal keys')
            summary = read(hashed(f'{base}/summary.json'))
            fields(summary, dict(schema='kimodo.v1-generate.summary.v1', arm=arm, expected=expected,
                                 pilot=True, completed=len(oks), ok=len(oks), failed=0), base)
            require(set(summary['sessions']) == {rec['session_id'] for rec in oks.values()},
                    f'{base}: session mismatch')
            arms[arm] = dict(encoder=enc, ok=len(oks), journal=log, sessions=summary['sessions'])
            journals[arm] = oks
        for ident, row in rows.items():
            for seed in SEEDS:
                clips, poses = {}, {}
                for arm in ARMS:
                    rec = journals[arm][ident, seed]
                    relative = f'{group}/{arm}/{ident}-{seed}'
                    for name in FILES:
                        source(v1, f'{relative}/{name}')
                    prompt = row['ko' if arm == 'ko' else 'en'].strip()
                    gen = dict(prompt=prompt, seed=seed, duration=row['seconds'], steps=config['steps'],
                               model=model[0].split('/')[-1], modelRevision=model[1], arm=arm, id=ident,
                               constraintCount=len(config['constraints']))
                    clip, poses[arm], names = inspect_clip(v1/relative, gen, rec['frame_count'],
                                                          rec['fps'], rec.get('artifact_sha256', {}))
                    if arm == 'ref':
                        ref_names = names
                    require(names == ref_names, f'{relative}: joint order mismatch')
                    clips[arm] = dict(clip, directory=relative, session_id=rec['session_id'],
                                      reconstructed_runner_prompt=v1_generate.maker_prompt(row, arm))
                rms = {arm: rms_cm(poses['ref'], poses[arm]) for arm in differences}
                for arm, value in rms.items():
                    differences[arm].append(value)
                requests.append(dict(id=ident, seed=seed, en=row['en'], ko=row['ko'], seconds=row['seconds'],
                                     clips=clips, ref_relative_rms_cm=rms))
        sets[group] = dict(manifest=manifest_name, model=model, steps=config['steps'], fps=config['fps'],
                           constraints=config['constraints'], post_processing=config['post_processing'],
                           arms=arms, requests=requests)
    return sets, {arm: stats(values) for arm, values in differences.items()}, sources


def read_app(takes, regression):
    entries, nonready = [], []
    for directory in sorted(takes.iterdir()):
        if not directory.is_dir():
            continue
        meta = source(takes, directory/'take.json')
        take = read(meta)
        require(take['id'] == directory.name, f'{meta}: take id mismatch')
        if take['status'] != 'ready':
            nonready.append(dict(take_id=take['id'], status=take['status'], take_sha256=digest(meta)))
            continue
        gen = {k: take[k] for k in ('prompt', 'duration', 'seed', 'steps', 'model')}
        gen['constraintCount'] = take['constraint_count']
        constraint_path = source(takes, directory/'constraints.json')
        constraints = read(constraint_path) if constraint_path.exists() else []
        require(isinstance(constraints, list) and len(constraints) == gen['constraintCount'],
                f'{meta}: constraint count mismatch')
        frames, fps = take['frame_count'], take['fps']
        require(frames == round(gen['duration']*fps), f'{meta}: duration/frame mismatch')
        for name in FILES:
            source(takes, directory/name)
        clip, _, _ = inspect_clip(directory, gen, frames, fps, take.get('artifact_sha256', {}))
        matches = {}
        for group, data in regression.items():
            for req in data['requests']:
                ref = req['clips']['ref']
                if gen['prompt'] != ref['generation']['prompt']:
                    continue
                key = (group, req['id'])
                match = matches.setdefault(key, dict(dataset=group, id=req['id'],
                                                     matched_ref_seeds_by_recorded_options=[]))
                if (all(ref['generation'][k] == gen[k] for k in gen) and data['constraints'] == constraints
                        and ref['fps'] == fps and ref['frame_count'] == frames):
                    match['matched_ref_seeds_by_recorded_options'].append(req['seed'])
        entries.append(dict(clip, take_id=directory.name, take_sha256=digest(meta), constraints=constraints,
                            prompt_matches=list(matches.values())))
    require(entries, f'{takes}: no ready app takes')
    return dict(takes=entries, nonready=nonready, root=takes.as_posix(),
                scope='App store, not v1 arms. Ready is export status, not a success label. Prompt matching is exact. '
                      'Matching recorded options does not establish encoder, model revision or runtime identity.')


def link_app_poses(path, takes_root, takes):
    """Link an existing browser measurement to exact app exports, never to a v1 arm."""
    data = read(path)
    fields(data, dict(schema='kimodo.v2.app-probe.1', diagnostic_only=True, browser_errors=[]), path)
    index = {t['take_id']: t for t in takes}
    linked = {}
    for pose in data['takes']:
        folder = Path(pose['numeric']['path']).resolve()
        ident = folder.name
        require(folder.parent == takes_root and ident in index and ident not in linked, f'{path}: wrong app take')
        original = index[ident]
        require(pose['generation'] == original['generation'] and pose['frames'] == original['frame_count'],
                f'{path}: app generation mismatch')
        expected = {name.split('.')[1]: h for name, h in original['artifact_sha256'].items()}
        require(pose['numeric']['sha256'] == expected, f'{path}: app artifact hash mismatch')
        for k in ('trace', 'screenshot'):
            name = pose[k]
            require(Path(name).name == name and digest(source(path.parent, name)) == pose[k+'_sha256'],
                    f'{path}: {k} mismatch')
        linked[ident] = {k: pose[k] for k in ('measurements','trace_sha256','screenshot_sha256')}
        prompt = original['generation']['prompt']
        values = pose['measurements']
        if prompt == 'raise right foot beside head':
            soma, vrm = (values[k]['RightFoot-Head'] for k in ('soma', 'vrm'))
            linked[ident]['observation'] = (
                f"Minimum right foot-to-head distance: {soma['closest_m']:.3f} m in SOMA before retargeting, "
                f"{vrm['closest_m']:.3f} m on this VRM. Peak signed foot-minus-head height was "
                f"{soma['max_dy_m']:.3f} m and {vrm['max_dy_m']:.3f} m respectively (+Y is up).")
        elif prompt == 'a person claps its hands':
            soma, vrm = (values[k]['RightHand-LeftHand']['closest_m'] for k in ('soma', 'vrm'))
            linked[ident]['observation'] = (f'Minimum hand-to-hand distance: {soma:.3f} m in SOMA, '
                                            f'{vrm:.3f} m on this VRM. Proximity alone does not establish clapping.')
    require(linked, f'{path}: empty app pose evidence')
    return dict(path=path.as_posix(), sha256=digest(path), takes=linked,
                scope='Separate original app browser diagnostics, not VRM playback of the archived v1 arms. '
                      'Distances are observations, not human labels or an independent evaluation.')


def audit(v1, takes, app_poses=None):
    v1, takes = (Path(p).resolve(strict=True) for p in (v1, takes))
    groups, rms, sources = read_regression(v1)
    app = read_app(takes, groups)
    if app_poses is not None:
        app['browser_evidence'] = link_app_poses(Path(app_poses).resolve(strict=True), takes, app['takes'])
    reqs = [r for group in groups.values() for r in group['requests']]
    clips = [c for r in reqs for c in r['clips'].values()]
    return dict(schema='kimodo.v2.v1-replay.1', scope='P1.1 read-only CPU audit of exposed regression data; '
                'no new generations, human ratings, independent evaluation or release decision.',
                auditor_sha256=digest(Path(__file__)), archive_root=v1.as_posix(), source_sha256=sources,
                regression=groups, original_app=app, rms_cm=rms,
                totals=dict(regression_prompts=len({(g,r['id']) for g, d in groups.items() for r in d['requests']}),
                            regression_prompt_seeds=len(reqs), regression_clips=len(clips),
                            regression_artifacts=sum(len(c['artifact_sha256']) for c in clips),
                            regression_recorded_artifact_hashes=sum(len(c['recorded_artifact_sha256']) for c in clips),
                            original_app_ready=len(app['takes']),
                            original_app_exact_prompt_matches=sum(bool(c['prompt_matches']) for c in app['takes'])),
                diagnosis=dict(vrm_attribution='undetermined', attribution_scope='v1 regression arms only. '
                               'SOMA NPZ and exported SOMA JSON FK are measured before retargeting. '
                               'No v1 VRM arm playback or semantic success/failure labels were audited.',
                               archive_max_npz_to_json_fk_rms_cm=max(c['npz_to_soma_json_fk_rms_cm'] for c in clips),
                               rms_definition='100 * sqrt(mean over frames and SOMA-77 joints of squared 3D distance '
                                              'between corresponding ref and student joints after each frame subtracts '
                                              'its own NPZ root_positions in metres). Not a quality score or repeat-run tolerance.',
                               hash_scope='Current SHA-256 fingerprints are not retrospective integrity proof when '
                                          'the original request journal has no corresponding digest.'))


def publish(report, output, protected, verify_only=False):
    """Publish exactly once: fsync a sibling, then link atomically without clobbering."""
    output = Path(output).resolve()
    require(not any(output.is_relative_to(Path(p).resolve()) for p in protected), f'{output}: protected source')
    payload = (json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+'\n').encode('utf-8')
    if not verify_only and not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix='.v1-replay-', delete=False) as f:
            tmp = Path(f.name)
        try:
            with tmp.open('wb') as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(tmp, output)
            except FileExistsError:
                pass  # compare the winning writer's exact bytes below
        finally:
            tmp.unlink()
    require(output.read_bytes() == payload, f'{output}: existing report differs; refusing to overwrite')
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1', type=Path, default=REPO.parent/'kimodoMaker-runs/v1-encoder-4090')
    parser.add_argument('--takes', type=Path, default=REPO/'backend/takes')
    parser.add_argument('--app-poses', type=Path, help='Optional separate original-app-poses/report.json to bind by hash')
    parser.add_argument('--output', type=Path, required=True, help='Run diagnostics/v1-replay.json')
    parser.add_argument('--verify-only', action='store_true', help='Re-audit all sources and compare; never write')
    args = parser.parse_args()
    try:
        report = audit(args.v1, args.takes, args.app_poses)
        protected = [args.v1, args.takes] + ([args.app_poses.parent] if args.app_poses else [])
        sha = publish(report, args.output, protected, args.verify_only)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'v1 replay audit failed: {exc}\n')
    print(json.dumps(dict(report_sha256=sha, totals=report['totals'], rms_cm=report['rms_cm']), sort_keys=True))


if __name__ == '__main__':
    main()
