"""Pilot-only, revision-pinned, resume-safe Kimodo conditioning comparison."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from v1_generate import sha256, write_json


def unit(final, identity, names, make):
    """Only a fully flushed, validated staging directory is renamed into the units namespace."""
    def validate(directory):
        marker = directory/'done.json'
        if (set(x.name for x in directory.iterdir()) != {*names, 'done.json','done.sha256'}
                or (directory/'done.sha256').read_text('ascii') != sha256(marker)):
            raise ValueError(f'incomplete/changed unit {directory}')
        data = json.loads(marker.read_bytes())
        if data['identity'] != identity or data['hashes'] != {n:sha256(directory/n) for n in names}:
            raise ValueError(f'unit identity/artifacts mismatch: {directory}')
        return data
    if final.exists():
        return validate(final)
    stage = final.parent.parent/'staging'/f'{final.name}-{uuid4().hex}'
    stage.mkdir(parents=True)
    info = make(stage)
    for name in names:
        with (stage/name).open('r+b') as f:
            if not os.fstat(f.fileno()).st_size:
                raise ValueError(f'empty artifact: {name}')
            os.fsync(f.fileno())
    write_json(stage/'done.json', dict(identity=identity, hashes={n:sha256(stage/n) for n in names}, result=info))
    with (stage/'done.sha256').open('x',encoding='ascii') as f:
        f.write(sha256(stage/'done.json')); f.flush(); os.fsync(f.fileno())
    validate(stage)
    stage.rename(final)
    return validate(final)


def plan(proto):
    import re
    jobs = []
    def add(arm, row, seed, settings):
        key = f"{arm}.{row['id']}.{seed}"
        if not re.fullmatch(r'[a-z]+\.[a-z-]+\.[0-9]+',key):
            raise ValueError('invalid key')
        jobs.append(dict(key=key,arm=arm,id=row['id'],group=row.get('group'),direction=row.get('direction'),
                         options=dict(prompt=row['prompt'],seed=seed,steps=settings['steps'],
                                      duration=settings['duration'],constraints=[])))
    w = proto['witness']; add('witness', w, w['seed'], w)
    for seed in proto['seeds']:
        for row in proto['rows']:
            add('ref',row,seed,proto['settings'])
            if len(jobs)==2: add('repeat',row,seed,proto['settings'])
            add('candidate',row,seed,proto['settings'])
    if len({j['key'] for j in jobs}) != len(jobs) or len(jobs)!=proto['budget']['maximum_committed_units'] or len(jobs)-2!=proto['budget']['paired_clips']:
        raise ValueError('duplicate key or wrong budget')
    return jobs


def measure(roots, left, right, direction):
    roots, left, right = (np.asarray(a, dtype=np.float64) for a in (roots, left, right))
    if (roots.shape != left.shape or roots.shape != right.shape or roots.ndim != 2
            or roots.shape[1] != 3 or len(roots) < 10 or not all(np.isfinite(a).all() for a in (roots,left,right))):
        raise ValueError('invalid root/hip coordinates')
    lateral = (left-right)[:, [0,2]]
    initial = lateral[:5].mean(0)
    if min(np.linalg.norm(initial), np.linalg.norm(lateral, axis=1).min()) < .01:
        raise ValueError('undefined hip axis')
    initial /= np.linalg.norm(initial)
    forward = np.array([-initial[1],initial[0]])
    delta = (roots[-5:].mean(0) - roots[:5].mean(0))[[0,2]]
    axis = {'left':initial, 'right':-initial, 'forward':forward, 'backward':-forward}[direction]
    along = float(delta @ axis)
    off = abs(float(delta @ [-axis[1],axis[0]]))
    angles = np.rad2deg(np.arccos(np.clip((lateral @ initial) / np.linalg.norm(lateral,axis=1), -1, 1)))
    p90 = float(np.percentile(angles,90))
    return dict(along_m=along, off_axis_m=off, heading_abs_p90_deg=p90,
                hit=along >= .30 and along >= off and p90 <= 45)


def summarize(jobs, data):
    pairs=[]
    for j in jobs:
        if j['arm']!='candidate': continue
        ck=j['key']; rk=ck.replace('candidate.','ref.',1)
        if ck not in data or rk not in data: continue
        r,c=(data[k]['result']['semantic'] for k in (rk,ck))
        pairs.append(dict(id=j['id'],group=j['group'],direction=j['direction'],seed=j['options']['seed'],ref=r,candidate=c,
                          delta_along_m=c['along_m']-r['along_m']))
    def stats(rows):
        return dict(pairs=len(rows),ref_hits=sum(x['ref']['hit'] for x in rows),candidate_hits=sum(x['candidate']['hit'] for x in rows),
                    rescues=sum(not x['ref']['hit'] and x['candidate']['hit'] for x in rows),losses=sum(x['ref']['hit'] and not x['candidate']['hit'] for x in rows),
                    mean_delta_along_m=float(np.mean([x['delta_along_m'] for x in rows])) if rows else None)
    groups={g:stats([r for r in pairs if r['group']==g]) for g in sorted({j['group'] for j in jobs if j['arm']=='candidate'})}
    return dict(scope='nonblind pilot directional diagnostic only; no quality or release verdict',complete=set(data)=={j['key'] for j in jobs},
                units=len(data),expected_units=len(jobs),expected_pairs=sum(j['arm']=='candidate' for j in jobs),
                groups=groups,matched_rows=pairs,**stats(pairs))


def freeze_inputs(pilot,cache,proto,gpu):
    """Hash actual local model bytes (not just revision names), source, historical witness and runtime before ANY skip."""
    import shutil, sys
    from importlib.util import find_spec
    from v1_generate import ROOT, MOTION, TEACHER, TEACHER_FILES, backend_settings, pinned_snapshot, freeze_json
    from v2_conditioning import INSTRUCTION
    if proto['instruction'] != INSTRUCTION or proto['settings'] != dict(duration=3.,steps=100,fps=30,constraints=[],samples=1,post_processing=True,cfg='unchanged Maker defaults'):
        raise ValueError('implementation/settings do not match protocol')
    code={}
    for rel in ('docs/experiment/v2_pilot.py','docs/experiment/v2_conditioning.py','docs/experiment/v1_generate.py','docs/experiment/v1_export_check.py','backend/engine.py','docs/experiment/tests/test_v2_conditioning.py','docs/experiment/tests/test_v2_pilot.py'):
        p=ROOT/rel; dest=pilot/'source'/rel; want=sha256(p)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest)
        if sha256(dest)!=want: raise ValueError(f'changed source: {rel}')
        code[rel]=want
    def inventory(base,exclude_pyc=False):
        return {p.relative_to(base).as_posix():sha256(p) for p in sorted(base.rglob('*')) if p.is_file() and not (exclude_pyc and p.suffix=='.pyc')}
    models={}
    for k,identity in {'motion':MOTION,**TEACHER}.items():
        base=pinned_snapshot(identity,cache,TEACHER_FILES.get(k)); models[k]=dict(id=list(identity),path=str(base),files=inventory(base))
    installation=Path(find_spec('kimodo').origin).parent
    lock=dict(schema='direction-pilot.lock.1',protocol_sha256=sha256(pilot/'protocol.json'),code=code,models=models,
              historical_sha256=sha256(proto['witness']['path']),kimodo=inventory(installation,True),backend=backend_settings(),
              gpu=gpu,cache=str(cache),argv=[sys.executable,'-B','-u','docs/experiment/v2_pilot.py','--pilot',pilot.as_posix(),'--hf-cache',cache.as_posix()],
              cwd=str(ROOT), offline=True, temporal_budget='30 minute window from first loading; successful units remain verifiable subsequently',
              conditioning='bf16 frozen official weights; output float32 [B,1,4096]. No reranking, cache or guidance changes.')
    freeze_json(pilot/'lock.json',lock)
    return sha256(pilot/'lock.json'),Path(models['motion']['path'])


def motion_parity(a,b):
    with np.load(a,allow_pickle=False) as first,np.load(b,allow_pickle=False) as second:
        out={}
        for key in ('root_positions','posed_joints'):
            x,y=first[key],second[key]
            if x.shape!=y.shape: raise ValueError('parity shape mismatch')
            v=float(np.sqrt(np.square(x.astype('d')-y.astype('d')).sum(-1).mean()))
            out[key+'_rms_m']=v
        out['pass']=max(out.values())<=.001
        return out


def run(pilot,cache):
    import sys,time,shutil
    from v1_generate import ROOT,Nvml,output_lock,load_teacher,load_pinned_motion,utc_now,freeze_json
    from v2_conditioning import InstructionEncoder
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if os.environ.get('KIMODO_TEXT_ENCODER','').strip().lower()=='student': raise ValueError('stock teacher required')
    sys.path.insert(0,str(ROOT))
    from backend.engine import KimodoEngine
    from v1_export_check import verify_take
    from kimodo.sanitize import sanitize_text
    proto=json.loads((pilot/'protocol.json').read_bytes()); jobs=plan(proto)
    units=pilot/'units'; units.mkdir(exist_ok=True); names=['motion.json','motion.npz','motion.bvh','condition.npy']
    cmd=[sys.executable,'-B','-u','docs/experiment/v2_pilot.py','--pilot',pilot.as_posix(),'--hf-cache',cache.as_posix()]
    data={}; parity={}; engine=None; key=None
    def progress(status,**more):
        write_json(pilot/'progress.json',dict(status=status,utc=utc_now(),last_verified=next(reversed(data),None),
            verified_this_pass=len(data),expected=len(jobs),next_key=key,argv=cmd,cwd=str(ROOT),**more))
    with output_lock(pilot), Nvml() as nv:
        progress('freezing or checking inputs')
        try:
            digest,model_path=freeze_inputs(pilot,cache,proto,nv.identity)
            if set(p.name for p in units.iterdir())-{j['key'] for j in jobs}: raise ValueError('unexpected units')
            def make(stage):
                nonlocal engine,teacher,candidate
                if (pilot/'failure.json').exists(): raise RuntimeError('terminal failure preserved; no rerun')
                window=pilot/'gpu-window.json'
                if not window.exists(): freeze_json(window,dict(started=time.time(),lock=digest))
                window_info=json.loads(window.read_bytes())
                if window_info['lock']!=digest or time.time()-window_info['started'] >= proto['budget']['maximum_gpu_generation_wall_minutes']*60: raise RuntimeError('GPU time budget exceeded')
                limit=proto['budget']['disk_budget_GiB']*2**30
                if sum(p.stat().st_size for p in pilot.rglob('*') if p.is_file())+32*2**20 > limit or shutil.disk_usage(pilot).free < 256*2**20: raise RuntimeError('disk budget exceeded')
                if engine is None:
                    progress('loading official local model')
                    import torch
                    if torch.cuda.device_count()!=1 or 'RTX 4090' not in torch.cuda.get_device_name(0): raise RuntimeError('wrong CUDA topology')
                    teacher=load_teacher(cache,pilot); engine=KimodoEngine(); engine.model=load_pinned_motion(model_path,teacher)
                    if not teacher.model.skip_instruction or teacher.model.pooling_mode!='mean' or any(p.requires_grad for p in teacher.model.parameters()): raise RuntimeError('unexpected teacher pooling or trainable parameters')
                    candidate=InstructionEncoder(teacher,{sanitize_text(r['prompt']) for r in proto['rows']})
                active=candidate if j['arm']=='candidate' else teacher
                engine.model.text_encoder=active
                progress('generating/exporting',lock=digest)
                start=time.perf_counter()
                try:
                    info=engine.generate(j['options'],stage,lambda phase:None)
                except Exception as e:
                    freeze_json(pilot/'failure.json',dict(key=key,lock=digest,error=repr(e),stage=stage.name)); raise
                elapsed=time.perf_counter()-start
                # Diagnostic re-encoding after the actual, unchanged Maker call (never injected or cached).
                text=sanitize_text(j['options']['prompt']); vec,lens=active([text]); arr=vec.detach().cpu().numpy()
                if arr.shape!=(1,1,4096) or lens!=[1] or arr.dtype != np.float32 or not np.isfinite(arr).all(): raise ValueError('unexpected conditioning token')
                np.save(stage/'condition.npy',arr,allow_pickle=False)
                checked=verify_take(stage); checked.pop('path')
                if checked['frames']!=round(j['options']['duration']*30) or checked['fps']!=30.: raise ValueError('export timing mismatch')
                motion=json.loads((stage/'motion.json').read_bytes()); expected={k:v for k,v in j['options'].items() if k!='constraints'}
                if motion['generation']!=dict(expected,model='Kimodo-SOMA-RP-v1.2',constraintCount=0): raise ValueError('wrong exported request')
                pose_names=[x['name'] for x in motion['joints']]; semantic=None
                if j['direction']:
                    with np.load(stage/'motion.npz',allow_pickle=False) as z:
                        hips=z['posed_joints']; semantic=measure(z['root_positions'],hips[:,pose_names.index('LeftLeg')],hips[:,pose_names.index('RightLeg')],j['direction'])
                return dict(export=checked,engine=info,semantic=semantic,generate_and_export_s=elapsed,encoder_probe='separate post-generation re-encode; not a cost benchmark',utc=utc_now())
            teacher=candidate=None
            for j in jobs:
                key=j['key']; progress('verifying or starting',lock=digest)
                identity={'lock':digest,**j}; data[key]=unit(units/key,identity,names,make)
                if j['arm'] in ('witness','repeat'):
                    other=Path(proto['witness']['path']) if j['arm']=='witness' else units/key.replace('repeat.','ref.',1)/'motion.npz'
                    parity[key]=motion_parity(units/key/'motion.npz',other)
                    if not parity[key]['pass']:
                        raise ValueError(f'parity gate blocked: {key} {parity[key]}')
                rep=summarize(jobs,data); rep.update(lock_sha256=digest,parity=parity)
                write_json(pilot/'partial.json',rep); print(f'verified {len(data)}/{len(jobs)} {key}',flush=True)
            for r in rep['matched_rows']:
                tail=f"{r['id']}.{r['seed']}"; a,b=[np.load(units/f'{arm}.{tail}'/'condition.npy',allow_pickle=False).ravel().astype('d') for arm in ('ref','candidate')]
                r['encoder_cosine']=float((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)))
            freeze_json(pilot/'report.json',rep); key=None; progress('complete: pilot diagnostic only',lock=digest,report_sha256=sha256(pilot/'report.json'))
            return rep
        except BaseException as e:
            progress('blocked/interrupted',error=repr(e),parity=parity); raise


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--pilot',type=Path,required=True); ap.add_argument('--hf-cache',type=Path,required=True)
    a=ap.parse_args(); r=run(a.pilot.resolve(),a.hf_cache.resolve()); print({k:r[k] for k in ('pairs','ref_hits','candidate_hits','rescues','losses')})
