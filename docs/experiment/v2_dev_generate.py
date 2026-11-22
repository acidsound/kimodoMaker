"""Generate the frozen dev split with the unchanged Maker engine for both conditioning arms."""
import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from v1_generate import (ROOT, MOTION, TEACHER, TEACHER_FILES, Nvml, backend_settings, freeze_json, load_pinned_motion,
                         load_teacher, output_lock, pinned_snapshot, sha256, utc_now, write_json)
from v2_pilot import unit

SEEDS = (17, 43, 97)
SETTINGS = dict(duration=3., steps=100, fps=30, constraints=[], samples=1, post_processing=True)
NAMES = ['motion.json', 'motion.bvh', 'motion.npz', 'condition.npy', 'telemetry.json']
DEV = 'evaluation/dev.jsonl'
SOURCE = ['docs/experiment/v2_dev_generate.py', 'docs/experiment/v2_pilot.py', 'docs/experiment/v1_generate.py',
          'docs/experiment/v1_export_check.py', 'docs/experiment/v2_conditioning.py', 'docs/experiment/v2_gate.py',
          'backend/engine.py']


def plan(rows):
    jobs = []
    for row in rows:
        for seed in SEEDS:
            for arm in ('ref', 'candidate'):
                jobs.append(dict(arm=arm, group_id=row['group_id'], stratum=row['stratum'], seed=seed, en=row['en']))
    keys = {f'{j["arm"]}.{j["group_id"]}.{j["seed"]}' for j in jobs}
    if len(keys) != len(jobs) or len(keys) != 384 or len({j['group_id'] for j in jobs}) != 64:
        raise ValueError('invalid dev plan')
    if len({(j['stratum'], j['seed'], j['arm']) for j in jobs}) != 4 * 3 * 2 or len({j['en'] for j in jobs}) != 64:
        raise ValueError('strata or caption duplication')
    return jobs


def lock_inputs(run_dir, parent_lock, cache, gpu, rows_raw):
    base = json.loads(parent_lock.read_bytes())
    def inventory(folder, skip_pyc=False):
        return {p.relative_to(folder).as_posix(): sha256(p) for p in sorted(folder.rglob('*'))
                if p.is_file() and not (skip_pyc and p.suffix == '.pyc')}
    if base['gpu'] != gpu or base['backend'] != backend_settings():
        raise ValueError('runtime differs from pinned reference lock')
    if inventory(Path(find_spec('kimodo').origin).parent, True) != base['kimodo']:
        raise ValueError('installed Kimodo differs from pinned reference')
    models = {}
    for k, ident in {'motion': MOTION, **TEACHER}.items():
        snap = pinned_snapshot(ident, cache, TEACHER_FILES.get(k))
        saved = base['models'][k]
        if list(ident) != saved['id'] or inventory(snap) != saved['files']:
            raise ValueError(f'model bytes changed: {k}')
        models[k] = snap
    code = {}
    for rel in SOURCE:
        p = ROOT / rel
        target = run_dir / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        if sha256(target) != sha256(p):
            raise ValueError(f'source drift: {rel}')
        code[rel] = sha256(p)
    lock = dict(schema='dev-split.lock.1', parent_lock=str(parent_lock), parent_lock_sha256=sha256(parent_lock),
                code=code, models={k: base['models'][k] for k in base['models']}, gpu=gpu, backend=backend_settings(),
                cache=str(cache), cwd=str(ROOT), dev_rows_sha256=sha256_bytes(rows_raw))
    freeze_json(run_dir / 'lock.json', lock)
    return sha256(run_dir / 'lock.json'), Path(models['motion'])


def sha256_bytes(raw):
    import hashlib
    return hashlib.sha256(raw).hexdigest()


def run(run_dir, parent_lock, cache):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if os.getenv('KIMODO_TEXT_ENCODER', '').strip().lower() == 'student':
        raise ValueError('stock teacher required')
    sys.path.insert(0, str(ROOT))
    from v2_run_state import Journal
    from backend.engine import KimodoEngine
    from v1_export_check import verify_take
    from kimodo.sanitize import sanitize_text
    from v2_conditioning import InstructionEncoder
    eval_rows_raw = (run_dir.parent / DEV).read_bytes()
    rows = [json.loads(line) for line in eval_rows_raw.decode('utf-8-sig').splitlines()]
    jobs = plan(rows)
    (run_dir / 'dev-rows.jsonl').write_bytes(eval_rows_raw)
    protocol = dict(schema='dev-split.protocol.1', split='dev', rows=sha256_bytes(eval_rows_raw),
                    population='BONES-SEED English captions; proxy only, disclosed in the gate protocol',
                    arms=['ref', 'candidate'], candidate='instruction-v1 on every dev caption',
                    same_settings=SETTINGS, seeds=list(SEEDS), budget=dict(units=384, warmup=2,
                    max_gpu_generation_wall_minutes=90, min_free_GiB=20, retries=0),
                    scoring='none here; blind review and ratings come from v2_dev_review.py and the human panel')
    if (run_dir / 'protocol.json').exists():
        if json.loads((run_dir / 'protocol.json').read_bytes()) != protocol:
            raise ValueError('protocol drift; no overwrite')
    else:
        write_json(run_dir / 'protocol.json', protocol)
    argv = [sys.executable, '-B', '-u', 'docs/experiment/v2_dev_generate.py', '--run', run_dir.as_posix(),
            '--model-lock', parent_lock.as_posix(), '--hf-cache', cache.as_posix()]
    units_spec = [dict(arm=j['arm'], prompt_id=j['group_id'], seed=j['seed'],
                       outputs=[f'units/{j["arm"]}.{j["group_id"]}.{j["seed"]}/{n}' for n in NAMES],
                       failure=f'units/{j["arm"]}.{j["group_id"]}.{j["seed"]}/error.json') for j in jobs]
    spec = dict(experiment_id='english-conditioning-01', stage='dev', candidate_id='instruction-v1',
                protocol='protocol.json', candidate='dev-rows.jsonl',
                inputs={n: sha256(run_dir / n) for n in ('protocol.json', 'dev-rows.jsonl')},
                argv=argv, cwd=str(ROOT), budget={'requests': 384}, trial=1, units=units_spec)
    journal = Journal.create(run_dir, spec)
    units = run_dir / 'units'
    units.mkdir(exist_ok=True)
    engine = teacher = candidate = None
    status = {'done': 0, 'failed': 0}
    telemetry = {a: [] for a in ('ref', 'candidate')}

    def progress(phase, **more):
        write_json(run_dir / 'progress.json', dict(phase=phase, utc=utc_now(), done=status['done'],
                   failed=status['failed'], expected=384, argv=argv, cwd=str(ROOT), **more))

    def generate(arm, en, seed, stage):
        nonlocal engine, candidate, teacher
        import torch
        if engine is None:
            progress('loading official local model')
            if torch.cuda.device_count() != 1 or 'RTX 4090' not in torch.cuda.get_device_name(0):
                raise RuntimeError('wrong CUDA topology')
            teacher = load_teacher(cache, run_dir)
            engine = KimodoEngine()
            engine.model = load_pinned_motion(model_path, teacher)
            candidate = InstructionEncoder(teacher, {sanitize_text(r['en']) for r in rows})
            if not teacher.model.skip_instruction or teacher.model.pooling_mode != 'mean':
                raise RuntimeError('unexpected teacher mode')
        active = candidate if arm == 'candidate' else teacher
        engine.model.text_encoder = active
        options = dict(prompt=en, seed=seed, steps=SETTINGS['steps'], duration=SETTINGS['duration'], constraints=[])
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.generate(options, stage, lambda phase: None)
        latency_ms = (time.perf_counter() - start) * 1000
        peak = int(torch.cuda.max_memory_allocated())
        text = sanitize_text(en)
        vec, lens = active([text])
        arr = np.asarray(vec.detach().cpu())
        if arr.shape != (1, 1, 4096) or lens != [1] or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError('unexpected conditioning token')
        np.save(stage / 'condition.npy', arr, allow_pickle=False)
        checked = verify_take(stage)
        doc = json.loads((stage / 'motion.json').read_bytes())
        # The Maker metadata keeps the request prompt exactly as received (period fixing happens downstream).
        if doc['generation'] != dict(prompt=en, seed=seed, duration=SETTINGS['duration'],
                                     steps=SETTINGS['steps'], model=MOTION[0].split('/')[-1], constraintCount=0):
            raise ValueError('generation metadata mismatch')
        if checked['frames'] != SETTINGS['duration'] * SETTINGS['fps'] or checked['fps'] != 30:
            raise ValueError('export mismatch')
        write_json(stage / 'telemetry.json', dict(latency_ms=latency_ms, peak_bytes=peak, arm=arm,
                                                  device=torch.cuda.get_device_name(0)))
        return dict(export_frames=checked['frames'], latency_ms=latency_ms, peak_bytes=peak)

    with output_lock(run_dir), Nvml() as nv:
        try:
            progress('verifying inputs')
            lock, model_path = lock_inputs(run_dir, parent_lock, cache, nv.identity, eval_rows_raw)
            requests = run_dir / 'requests.jsonl'
            if not requests.exists():
                requests.write_bytes(b'')
            ident = dict(lock=lock)
            # Warm up each arm once so the first CUDA graphs do not enter the cost sample.
            first = jobs[0]
            for arm in ('ref', 'candidate'):
                warm = units.parent / 'warmup' / f'{arm}.{first["group_id"]}.{first["seed"]}'
                warm.parent.mkdir(parents=True, exist_ok=True)
                if not warm.exists():
                    unit(warm, {**ident, 'arm': arm, 'warmup': True}, NAMES,
                         lambda stage, arm=arm: generate(arm, first['en'], first['seed'], stage))
            for j in jobs:
                key = f'{j["arm"]}.{j["group_id"]}.{j["seed"]}'
                folder = units / key
                key_tuple = (j['arm'], j['group_id'], j['seed'])
                record = dict(key=key, arm=j['arm'])
                rel = lambda n: f'units/{key}/{n}'
                if (folder / 'done.json').is_file():
                    hashes = {rel(n): h for n, h in json.loads((folder / 'done.json').read_bytes())['hashes'].items()}
                    outcome = 'ok'
                    record['status'] = 'reused'
                    archived = json.loads((folder / 'telemetry.json').read_bytes())
                    telemetry[j['arm']].append(dict(latency_ms=archived['latency_ms'], peak_bytes=archived['peak_bytes']))
                elif (folder / 'error.json').is_file():
                    hashes = {rel('error.json'): sha256(folder / 'error.json')}
                    outcome = 'failed'
                    record['status'] = 'failed'
                else:
                    try:
                        info = unit(folder, {**ident, 'key': key, 'arm': j['arm'], 'en': j['en']}, NAMES,
                                    lambda stage, j=j: generate(j['arm'], j['en'], j['seed'], stage))
                    except Exception as exc:
                        folder.mkdir(exist_ok=True)
                        freeze_json(folder / 'error.json', dict(key=key, reason=repr(exc), utc=utc_now()))
                        status['failed'] += 1
                        record['status'] = 'failed'
                        hashes = {rel('error.json'): sha256(folder / 'error.json')}
                        outcome = 'failed'
                    else:
                        hashes = {rel(n): sha256(folder / n) for n in NAMES}
                        result = info['result']
                        telemetry[j['arm']].append(dict(latency_ms=result['latency_ms'], peak_bytes=result['peak_bytes']))
                        status['done'] += 1
                        record['status'] = 'ok'
                        outcome = 'ok'
                # unit() publishes atomically; the receipt is separately idempotent.
                journal.seal(key_tuple, outcome, hashes)
                with requests.open('a', encoding='utf-8') as rq:
                    rq.write(json.dumps(record, separators=(',', ':')) + '\n')
                progress('generating')
            def stats(rows):
                if not rows:
                    return dict(n=0)
                lat = sorted(r['latency_ms'] for r in rows)
                return dict(n=len(rows), p95_latency_ms=lat[int(.95 * len(lat)) - 1],
                            max_peak_bytes=max(r['peak_bytes'] for r in rows))
            binding = journal.report_binding()
            write_json(run_dir / 'report.json', dict(binding=binding, complete=True, generated=status['done'],
                       failed=status['failed'], telemetry={a: stats(v) for a, v in telemetry.items()},
                       device=dict(name=nv.identity['name'], uuid=nv.identity['uuid'], driver=nv.identity['driver'],
                                   runtime='torch cuda as pinned in lock'),
                       scope='dev generation only; no ratings, no acceptance'))
            progress('complete')
        except BaseException as exc:
            progress('blocked/interrupted', error=repr(exc))
            raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--model-lock', type=Path, required=True)
    ap.add_argument('--hf-cache', type=Path, required=True)
    a = ap.parse_args()
    run(a.run.resolve(), a.model_lock.resolve(strict=True), a.hf_cache.resolve(strict=True))
