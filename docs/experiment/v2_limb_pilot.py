"""Run the predeclared unilateral limb diagnostic through the real, unchanged Maker generator."""
import argparse
from importlib.util import find_spec
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from v1_generate import (ROOT, TEACHER, TEACHER_FILES, MOTION, Nvml, backend_settings, freeze_json, load_pinned_motion,
                         load_teacher, output_lock, pinned_snapshot, sha256, utc_now, write_json)
from v2_limb import LimbEncoder, intent
from v2_pilot import motion_parity, unit


def metrics(head, target, opposite):
    h,t,o=(np.asarray(x,dtype='d') for x in (head,target,opposite))
    if h.shape!=t.shape or h.shape!=o.shape or h.ndim!=2 or h.shape[1]!=3 or not len(h) or not all(np.isfinite(x).all() for x in (h,t,o)):
        raise ValueError('joint arrays must match')
    def gap(a,b):
        d=np.linalg.norm(a-h,axis=1); i=int(d.argmin()); height=float(a[i,1]-b[i,1]); distance=float(d[i])
        return i,distance,height,distance<=.45 and height>=.30
    i,d,height,hit=gap(t,o); _,other,_,wrong=gap(o,t)
    return dict(closest_frame=i,min_gap_m=d,opposite_min_gap_m=other,ankle_difference_m=height,hit=hit,
                wrong_side_hit=wrong, peak_target_height_m=float(t[:,1].max()),
                opposite_lift_from_start_m=float(o[:,1].max()-o[0,1]))


class Capture:
    def __init__(self,encoder,expected): self.encoder,self.expected,self.token=encoder,expected,None
    def __call__(self,texts):
        if texts!=[self.expected]: raise ValueError(f'unexpected input at conditioning boundary: {texts!r}')
        out,lens=self.encoder(texts); token=out.detach().cpu().numpy().copy()
        if token.dtype!=np.float32 or token.shape!=(1,1,4096) or lens!=[1] or not np.isfinite(token).all():
            raise ValueError('invalid conditioning token')
        if self.token is not None and not np.array_equal(self.token,token): raise ValueError('condition changed within clip')
        self.token=token; return out,lens


def inputs(out, parent_lock, cache, gpu, proto):
    base=json.loads(parent_lock.read_bytes()); models={}
    if base['gpu']!=gpu or base['backend']!=backend_settings(): raise ValueError('runtime differs from pinned pilot')
    def inventory(folder, skip_pyc=False):
        return {p.relative_to(folder).as_posix():sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and not (skip_pyc and p.suffix=='.pyc')}
    if inventory(Path(find_spec('kimodo').origin).parent, True)!=base['kimodo']: raise ValueError('installed Kimodo differs')
    for k,ident in {'motion':MOTION,**TEACHER}.items():
        snap=pinned_snapshot(ident,cache,TEACHER_FILES.get(k)); saved=base['models'][k]
        if list(ident)!=saved['id'] or inventory(snap)!=saved['files']: raise ValueError(f'model bytes changed: {k}')
        models[k]=snap
    code={}
    for rel in ['backend/engine.py',*['docs/experiment/'+s for s in ['v1_generate.py','v1_export_check.py','v2_limb.py','v2_pilot.py','v2_limb_pilot.py','tests/test_v2_limb.py','tests/test_v2_limb_pilot.py']]]:
        p=ROOT/rel; target=out/'source'/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(p,target)
        if sha256(target)!=sha256(p) or (rel in base['code'] and sha256(p)!=base['code'][rel]): raise ValueError(f'source drift: {rel}')
        code[rel]=sha256(p)
    originals={}
    for row in proto['rows']:
        if intent(row['prompt'])!=row['side']: raise ValueError(f'route / rubric mismatch: {row["id"]}')
        if 'take_id' not in row: continue
        folder=ROOT/'backend/takes'/row['take_id']; doc=json.loads((folder/'take.json').read_bytes())
        expected=dict(prompt=row['prompt'],duration=row['seconds'],steps=row['steps'],seed=row['seeds'][0],constraint_count=0,status='ready',model=MOTION[0].split('/')[-1])
        if any(doc.get(k)!=v for k,v in expected.items()): raise ValueError('historical request mismatch')
        originals[row['id']]={name:sha256(folder/name) for name in ['take.json','motion.npz']}
    lock=dict(protocol_sha256=sha256(out/'protocol.json'), model_lock=str(parent_lock),model_lock_sha256=sha256(parent_lock),
              app_witnesses=originals, code=code, gpu=gpu, backend=backend_settings(), cache=str(cache),cwd=str(ROOT))
    freeze_json(out/'lock.json',lock); return sha256(out/'lock.json'),models['motion']


def run(out,parent_lock,cache):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if os.getenv('KIMODO_TEXT_ENCODER','').lower().strip()=='student': raise ValueError('disable student path')
    sys.path.insert(0,str(ROOT))
    from backend.engine import KimodoEngine
    from kimodo.sanitize import sanitize_text
    from v1_export_check import verify_take
    proto=json.loads((out/'protocol.json').read_bytes()); budget=proto['budget']
    pairs=[(r,s) for r in proto['rows'] for s in r['seeds']]; keys={f'{a}.{r["id"]}.{s}' for r,s in pairs for a in ['ref','candidate']}
    if len(keys)!=len(pairs)*2 or len(keys)!=budget['units']: raise ValueError('invalid budget/duplicate')
    engine=teacher=None; results={}; comparison=[]; parity={}; key=None
    def progress(phase,**kwargs):
        write_json(out/'progress.json',dict(phase=phase,key=key,verified=len(results),expected=len(keys),utc=utc_now(),**kwargs))
    units=out/'units'; units.mkdir(exist_ok=True)
    with output_lock(out), Nvml() as nv:
        try:
            progress('verifying inputs'); lock,motion=inputs(out,parent_lock,cache,nv.identity,proto)
            if {p.name for p in units.iterdir()}-keys: raise ValueError('unexpected unit')
            for r,seed in pairs:
                options=dict(prompt=r['prompt'],seed=seed,steps=r['steps'],duration=r['seconds'],constraints=[])
                for arm in ['ref','candidate']:
                    key=f'{arm}.{r["id"]}.{seed}'; progress('verifying or generating')
                    def make(stage):
                        nonlocal engine,teacher
                        if (out/'failure.json').exists(): raise RuntimeError('terminal failure preserved')
                        if shutil.disk_usage(out).free < budget['min_free_gib']*(2**30): raise RuntimeError('disk budget')
                        # Reserve a per-request ceiling before a fresh generation. Crashes retain their charge.
                        meter=out/'budget.json'; used=json.loads(meter.read_bytes())['charged_s'] if meter.exists() else 0.
                        limit=budget['max_active_wall_minutes']*60
                        allowance=min(limit-used, 240.0)
                        if allowance<=0: raise RuntimeError('GPU budget exhausted')
                        write_json(meter,dict(charged_s=used+allowance,reserved_for=key)); start=time.monotonic()
                        try:
                            if engine is None:
                                teacher=load_teacher(cache,out); engine=KimodoEngine(); engine.model=load_pinned_motion(motion,teacher)
                            active=Capture(LimbEncoder(teacher) if arm=='candidate' else teacher,sanitize_text(options['prompt']))
                            engine.model.text_encoder=active; engine.generate(options,stage,lambda _:None)
                            if active.token is None: raise ValueError('encoder was bypassed')
                            np.save(stage/'condition.npy',active.token,allow_pickle=False)
                            checked=verify_take(stage); checked.pop('path'); doc=json.loads((stage/'motion.json').read_bytes())
                            expected={k:v for k,v in options.items() if k!='constraints'}|dict(model=MOTION[0].split('/')[-1],constraintCount=0)
                            if doc['generation']!=expected or checked['frames']!=r['seconds']*30 or checked['fps']!=30: raise ValueError('export mismatch')
                            sem=None
                            if r['side']:
                                names=[j['name'] for j in doc['joints']]; ix=names.index; side=r['side'].title(); opp='Left' if side=='Right' else 'Right'
                                with np.load(stage/'motion.npz',allow_pickle=False) as z:
                                    joints=z['posed_joints']; sem=metrics(joints[:,ix('Head')],joints[:,ix(side+'Foot')],joints[:,ix(opp+'Foot')])
                            elapsed=time.monotonic()-start
                            if elapsed>allowance: raise RuntimeError('per-unit budget exceeded')
                            return dict(export=checked,semantic=sem,elapsed_s=elapsed)
                        except Exception as e:
                            freeze_json(out/'failure.json',dict(key=key,lock=lock,reason=repr(e))); raise
                        finally:
                            write_json(meter,dict(charged_s=used+time.monotonic()-start,last_attempt=key))
                    root=units/key
                    results[key]=unit(root,dict(lock=lock,arm=arm,request=options,id=r['id']),['motion.json','motion.bvh','motion.npz','condition.npy'],make)
                    if r.get('take_id') and seed==r['seeds'][0] and arm=='ref':
                        par=motion_parity(root/'motion.npz',ROOT/'backend/takes'/r['take_id']/'motion.npz'); parity[key]=par
                        if not par['pass']: raise ValueError(f'historical parity failed: {key} {par}')
                ref=key.replace('candidate.','ref.',1); sem=[results[k]['result']['semantic'] for k in (ref,key)]
                if r['side'] is None:
                    par=motion_parity(units/ref/'motion.npz',units/key/'motion.npz'); parity['fallback']=par
                    if not par['pass'] or results[ref]['hashes']['condition.npy']!=results[key]['hashes']['condition.npy']: raise ValueError('fallback not identical')
                comparison.append(dict(id=r['id'],seed=seed,ref=sem[0],candidate=sem[1]))
                write_json(out/'partial.json',dict(pairs=comparison,parity=parity))
            target=[x for x in comparison if x['ref'] is not None]; counts={arm:sum(x[arm]['hit'] for x in target) for arm in ['ref','candidate']}
            if len(target)!=budget['target_pairs'] or len(comparison)-len(target)!=budget['identity_pairs'] or set(results)!=keys: raise ValueError('missing results')
            freeze_json(out/'report.json',dict(lock_sha256=lock,complete=True,scope='diagnostic only',pairs=comparison,parity=parity,hits=counts,
                wrong_side_hits={arm:sum(x[arm]['wrong_side_hit'] for x in target) for arm in counts},
                rescues=sum(not x['ref']['hit'] and x['candidate']['hit'] for x in target),losses=sum(x['ref']['hit'] and not x['candidate']['hit'] for x in target)))
            progress('complete: diagnostic only',report_sha256=sha256(out/'report.json'))
        except BaseException as e:
            progress('blocked/interrupted',error=repr(e),parity=parity); raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True); ap.add_argument('--model-lock',type=Path,required=True); ap.add_argument('--hf-cache',type=Path,required=True)
    a=ap.parse_args(); run(a.run.resolve(),a.model_lock.resolve(strict=True),a.hf_cache.resolve(strict=True))
