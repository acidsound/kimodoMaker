"""Sweep instruction-text variants on the predicate-mapped dev groups, candidate arm only.

The reference arm is the frozen dev-split-01 output (receipt-sealed, unchanged hashes).
Variants are fixed before generation; no selection is done until all variants are evaluated.
"""
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
from v2_dev_predicates import MAPPED, SPEC, evaluate

SEEDS = (17, 43, 97)
SETTINGS = dict(duration=3., steps=100, fps=30, constraints=[], samples=1, post_processing=True)
NAMES = ['motion.json', 'motion.bvh', 'motion.npz', 'condition.npy', 'telemetry.json']
SOURCE = ['docs/experiment/v2_sweep.py', 'docs/experiment/v2_pilot.py', 'docs/experiment/v1_generate.py',
          'docs/experiment/v1_export_check.py', 'docs/experiment/v2_dev_predicates.py', 'backend/engine.py']

# Fixed a priori from the instruction-v1 measured pattern: gross-action rescues,
# hand-to-head contact losses. Variants test one hypothesis each.
VARIANTS = {
    'contact': ('Represent the human movement in this caption for motion generation. '
                'Keep stated contacts between body parts exact and do not infer additional actions:'),
    'minimal': ('Describe the human movement in this caption for motion generation:'),
}


class VariantEncoder:
    """Same LLM2Vec [instruction, text] channel as instruction-v1, with a different fixed instruction."""

    def __init__(self, official, allowed, instruction):
        self.official = official
        self.allowed = frozenset(allowed)
        self.instruction = instruction

    def __call__(self, texts):
        if not isinstance(texts, list) or not texts or not all(t and t in self.allowed for t in texts):
            return self.official(texts)
        return self.official([[self.instruction, text] for text in texts])


def plan(rows, mapped):
    jobs = []
    for variant in VARIANTS:
        for row in rows:
            if row['group_id'] not in mapped:
                continue
            for seed in SEEDS:
                jobs.append(dict(variant=variant, group_id=row['group_id'], seed=seed, en=row['en']))
    keys = {f'{j["variant"]}.{j["group_id"]}.{j["seed"]}' for j in jobs}
    if len(keys) != len(jobs) or len(keys) != len(VARIANTS) * len(mapped) * len(SEEDS):
        raise ValueError('invalid sweep plan')
    return jobs


def lock_inputs(run_dir, parent_lock, cache, gpu, rows_raw, mapped):
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
    import hashlib
    lock = dict(schema='sweep.lock.1', parent_lock=str(parent_lock), parent_lock_sha256=sha256(parent_lock),
                code=code, models={k: base['models'][k] for k in base['models']}, gpu=gpu,
                backend=backend_settings(), cache=str(cache), cwd=str(ROOT),
                dev_rows_sha256=hashlib.sha256(rows_raw).hexdigest(),
                reference_arm=str(run_dir.parent / 'dev-split-01' / 'units'),
                variants=VARIANTS, mapped_groups=sorted(mapped))
    freeze_json(run_dir / 'lock.json', lock)
    return sha256(run_dir / 'lock.json'), Path(models['motion'])


def evaluate_variant(run_dir, ref_run, variant):
    rows = []
    for group, (label, checks) in SPEC.items():
        for seed in SEEDS:
            ref_path = ref_run / f'ref.{group}.{seed}' / 'motion.npz'
            cand_path = run_dir / 'units' / f'{variant}.{group}.{seed}' / 'motion.npz'
            if not ref_path.is_file() or not cand_path.is_file():
                return None
            with np.load(ref_path) as zr, np.load(cand_path) as zc:
                for check in checks:
                    value_r, hit_r = evaluate(zr, check)
                    value_c, hit_c = evaluate(zc, check)
                    rows.append(dict(group=group, seed=seed, label=label, check=check[0],
                                     ref_value=value_r, cand_value=value_c, ref_hit=hit_r, cand_hit=hit_c))
    return rows


def summarise_variant(rows):
    from scipy.stats import binomtest
    a = sum(1 for x in rows if x['ref_hit'] and x['cand_hit'])
    b = sum(1 for x in rows if not x['ref_hit'] and x['cand_hit'])
    c_ = sum(1 for x in rows if x['ref_hit'] and not x['cand_hit'])
    d = sum(1 for x in rows if not x['ref_hit'] and not x['cand_hit'])
    discordant = b + c_
    p = float(binomtest(b, discordant, 0.5).pvalue) if discordant else 1.0
    return dict(n_checks=len(rows), ref_hits=a + c_, cand_hits=a + b, rescue=b, loss=c_,
                agree=a + d, mcnemar_exact_p=p)


def run(run_dir, parent_lock, cache):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(ROOT))
    from v2_run_state import Journal
    from backend.engine import KimodoEngine
    from v1_export_check import verify_take
    from kimodo.sanitize import sanitize_text

    ref_run = run_dir.parent / 'dev-split-01' / 'units'
    if not (ref_run / f'ref.{sorted(MAPPED)[0]}.17' / 'motion.npz').is_file():
        raise ValueError('frozen reference arm missing')
    eval_rows_raw = (run_dir.parent / 'evaluation' / 'dev.jsonl').read_bytes()
    rows = [json.loads(line) for line in eval_rows_raw.decode('utf-8-sig').splitlines()]
    jobs = plan(rows, MAPPED)
    (run_dir / 'dev-rows.jsonl').write_bytes(eval_rows_raw)
    argv = [sys.executable, '-B', '-u', 'docs/experiment/v2_sweep.py', '--run', run_dir.as_posix(),
            '--model-lock', parent_lock.as_posix(), '--hf-cache', cache.as_posix()]
    units_spec = [dict(arm=j['variant'], prompt_id=j['group_id'], seed=j['seed'],
                       outputs=[f'units/{j["variant"]}.{j["group_id"]}.{j["seed"]}/{n}' for n in NAMES],
                       failure=f'units/{j["variant"]}.{j["group_id"]}.{j["seed"]}/error.json') for j in jobs]
    spec = dict(experiment_id='english-conditioning-01', stage='sweep', candidate_id='instruction-variants',
                protocol='lock.json', candidate='dev-rows.jsonl',
                inputs={'lock.json': 'pending', 'dev-rows.jsonl': sha256(run_dir / 'dev-rows.jsonl')},
                argv=argv, cwd=str(ROOT), budget={'requests': len(jobs)}, trial=1, units=units_spec)
    engine = None
    model_path = None
    encoders = {}

    def generate(variant, en, seed, stage):
        nonlocal engine, model_path
        import torch
        if engine is None:
            if torch.cuda.device_count() != 1 or 'RTX 4090' not in torch.cuda.get_device_name(0):
                raise RuntimeError('wrong CUDA topology')
            teacher = load_teacher(cache, run_dir)
            engine = KimodoEngine()
            engine.model = load_pinned_motion(model_path, teacher)
            for name, text in VARIANTS.items():
                encoders[name] = VariantEncoder(teacher, {sanitize_text(r['en']) for r in rows}, text)
        engine.model.text_encoder = encoders[variant]
        options = dict(prompt=en, seed=seed, steps=SETTINGS['steps'], duration=SETTINGS['duration'], constraints=[])
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.generate(options, stage, lambda phase: None)
        latency_ms = (time.perf_counter() - start) * 1000
        peak = int(torch.cuda.max_memory_allocated())
        active = encoders[variant]
        text = sanitize_text(en)
        vec, lens = active([text])
        arr = np.asarray(vec.detach().cpu())
        if arr.shape != (1, 1, 4096) or lens != [1] or arr.dtype != np.float32 or not np.isfinite(arr).all():
            raise ValueError('unexpected conditioning token')
        np.save(stage / 'condition.npy', arr, allow_pickle=False)
        checked = verify_take(stage)
        doc = json.loads((stage / 'motion.json').read_bytes())
        if doc['generation'] != dict(prompt=en, seed=seed, duration=SETTINGS['duration'],
                                     steps=SETTINGS['steps'], model=MOTION[0].split('/')[-1], constraintCount=0):
            raise ValueError('generation metadata mismatch')
        if checked['frames'] != SETTINGS['duration'] * SETTINGS['fps'] or checked['fps'] != 30:
            raise ValueError('export mismatch')
        write_json(stage / 'telemetry.json', dict(latency_ms=latency_ms, peak_bytes=peak, variant=variant,
                                                  device=torch.cuda.get_device_name(0)))
        return dict(export_frames=checked['frames'], latency_ms=latency_ms, peak_bytes=peak)

    with output_lock(run_dir), Nvml() as nv:
        try:
            lock, model_path = lock_inputs(run_dir, parent_lock, cache, nv.identity, eval_rows_raw, MAPPED)
            spec['inputs']['lock.json'] = lock
            journal = Journal.create(run_dir, spec)
            (run_dir / 'units').mkdir(exist_ok=True)
            for j in jobs:
                key = f'{j["variant"]}.{j["group_id"]}.{j["seed"]}'
                folder = run_dir / 'units' / key
                if not (folder / 'done.json').is_file():
                    info = unit(folder, {'lock': lock, 'key': key, 'variant': j['variant'], 'en': j['en']}, NAMES,
                                lambda stage, j=j: generate(j['variant'], j['en'], j['seed'], stage))
                journal.seal((j['variant'], j['group_id'], j['seed']), 'ok',
                             {f'units/{key}/{n}': sha256(folder / n) for n in NAMES})
            results = {}
            for variant in VARIANTS:
                rows_v = evaluate_variant(run_dir, ref_run, variant)
                if rows_v is None:
                    raise ValueError(f'evaluation incomplete for {variant}')
                results[variant] = dict(rows=rows_v, summary=summarise_variant(rows_v))
            write_json(run_dir / 'sweep-report.json',
                       dict(binding=journal.report_binding(), variants={k: v['summary'] for k, v in results.items()},
                            limit='dev sweep, candidate arm only; selection on dev data is disclosed; '
                                  'sealed split untouched'))
            (run_dir / 'sweep-rows.json').write_text(
                json.dumps({k: v['rows'] for k, v in results.items()}, indent=1), encoding='utf-8')
        except BaseException as exc:
            write_json(run_dir / 'progress.json', dict(phase='blocked/interrupted', error=repr(exc), utc=utc_now()))
            raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--model-lock', type=Path, required=True)
    ap.add_argument('--hf-cache', type=Path, required=True)
    a = ap.parse_args()
    run(a.run.resolve(), a.model_lock.resolve(strict=True), a.hf_cache.resolve(strict=True))
