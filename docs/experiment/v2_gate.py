"""Local CPU-only English motion evidence protocol. See evaluation/PROTOCOL.md.

No candidate output is an input to corpus selection. No entry point releases a model.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import random
import re
import secrets


SEEDS = (17, 43, 97)
STRATA = ('travel_whole_body', 'posture_floor', 'direction_path', 'limb_gesture')
PER = 16
# Positive, closed English vocabulary. Unknown words -> reject, not normalize/rewrite.
VOCAB = set('''a the to right with person and their in forward character dance left while an on leg of hand walking hip at back hands performs hop from side walk step walks backward standing position fast jogging shoulder move turn backwards quick over foot moving is pace perform by steps diagonally torso vogue turns up someone jog arms high legs direction head individual movements jump facing moves performing sideways arm as start jumps finger dancer one both style turning into jogs towards bpm diagonal front stepping retro knees pose quickly down body movement dancing rhythm pointing speed neutral gesture degrees energy around jumping slowly hips motion arc paced stands points place feet bent using s floor takes for very times energetic out two four directions clockwise path knee makes holding big beat beats your ground small they per three minute crouching performer looking it like routine running executes execute straight sides full manner involving raised double degree behind does them dances wide normal sitting slightly making western rapid that air energetically figure slide slow balance stance maintaining swing weight heel toe action variation repeats something raises his man circular comes low run stand chicken all series leaning transition swift keeping shakes rhythmically way above performed swinging somebody circle mambo time lying twist disco rightward runs or dynamic scratches lively rapidly medium counterclockwise relaxed gestures exaggerated pulling tempo seated placing laughing extended extending music salsa country fours gets executing five spinning regular cycle called forwards displaying point still showcasing featuring involves expression extremely moderate opposite shifting bow different upward gesturing twice during away ballet ready pivot few proud end crouch make each bending reverse warm where rotation birds lateral between charleston heels sits number forehead animation shake through continuing improvised multiple victory lock holds scratching various expressing leaping spin first simultaneously balletic whole themselves several crowd random posture neck gracefully change goes lifting sweeping hopping camel offering engages deep you again touch engaging can passing covers pattern resembling height light leans basic horse plank voguing laughs upwards same lifts elbow looks face are do mimicking crouched along briefly swings placed shocked slides n bouncing maximum keeps continuously gives scratch take fingers raise ahead leap leaps thigh come moment talks twists hotdog clasped circles sliding joining simultaneous catwalk power toes touching giving macarena show playful confidently coughs randomly leisurely swaying spins imitating indicating eating watching kneeling level chin vigorous continuous fourth dancehall joined butterfly center square professional drink covering every sit changing onto shifts flexes reaching uses stomach initiates be pauses showing downward executed bicep celebratory worm known positioned spread push ups switching funky display comprised focusing repeatedly audience without nose maintains gaze surroundings puts he motions pretending playfully new fluid guitar leftward tired sync outward part seen twisting closer opening stretching talking handed lean tapping exercise looping scooby doo upbeat duck shuffle performance jackson bend crouches expresses wipes smooth vigorously enthusiastically places confident accompanied companion happily adding playing imaginary footwork feeling trying completes game pushing continue getting warming gentle gradual near slight people waist across rubs palms together she rub rubbing palms palm danced dancer dancers dancers' bent stretches stretched stretch stretched balancing balanced balances reaches pivoting pivots pivotted elbows planted plant holds hold rests rest resting reclines lays lays tilts tilting bends bows bowed bowing folds curling curls sits kneels kneel squints shrugs thumbs tracing snap snaps tapping taps clap folding arch arches arched hunched hunching sways ankle ankles shoulders spine shoulder blades arching balances floor seated lying flat crawling crawls crawl poses posing planted rests clasp hands palms head chin cheeks blink blinks cheeks ear eyes mouth smile smiles smiling across ankles wrists fingers extended gently gracefully torso upper lower smoothly rhythmic horizontally vertically diagonal spin spirals rotating rotates rotate rotated curve curving zigzag zig zag bouncing bounce bounces struts strutting skips skips skipping travels traveling shuffles shuffling glides gliding stepping tiptoe tiptoes tiptoeing tiptoed leans lunging toes thighs arms elbows circles circle thrust shifting shimmies shimmy shoulders torso circles semicircle perimeter arc arcs round rounded sways backpedals backpedaling march marching marches inwards inward outward inwardly turns swaying broad short long deliberate brisk briskly length alternating alternates repeated steady steadily alone heel heels keeping gait rhythm triangular figure eight weaving weaves crossing alternating crossing shuffling continuously around spiraling gently bringing brings sideways curved keeping head level palm slight raised unclasped looks looking gaze gazing stretches upper lowers lowering sinking hunches nods relaxed arms sides balance slight hip palms flicks touches slides standing holds spreading swinging parallel moving casually casually sways torso knees apart outwardly open closed rises half quarter makes dancer rhythm dance steps kneels legs floor still body on all fours resting pose seated holds seated knees bent arms behind torso frame camera himself herself themself him her his its its own other another single open closed palms chest paces places head neck slouches lift flat balancing crossing palm snap snapping snapping snap'''.split())
VARIANTS = set('r l left right cw ccw clockwise counterclockwise fast slowly quick slow max opt option variation v start stop loop long reverse neutral basic normal exaggerated mild hard female male'.split())

def words(s):
    return re.findall(r'[a-z]+', s.casefold())

def family(name):
    return ' '.join(x for x in words(name) if x not in VARIANTS)

# V1 regression concepts and synthetic atomic families are quarantined, not evaluated.
REGRESSION = re.compile(r'flip|somersault|roll(?:ing|s|ed)?\b|kick|stomp|stamp|nod(?:s|ding|ded)?\b|clap|kiss|hero|cross.?legg|legs?.{0,18}cross|cross.{0,16}(?:arms?|legs?)|arms?.{0,18}cross|overhead|squat|lunge|arm.{0,22}above.{0,12}head|(?:hand|arm).{0,20}(?:high|upward)|rais\w* (?:\w+ ){0,3}(?:hand|arm|leg)|(?:high|lift\w*) (?:\w+ ){0,3}leg|(?:quarter|half) turn|(?:step|feet|plant).{0,50}\bturn|\bturn.{0,16}(?:degrees?|in place)')

def caption_ok(s):
    w = words(s)
    return (s.isascii() and bool(re.fullmatch(r"[A-Za-z0-9 ,.\-'!?()]+", s)) and
            5 <= len(w) <= 30 and set(w) <= VOCAB and not REGRESSION.search(s.lower()) and
            not re.search(r'\b(?:then|after|before|followed|sequence|next|first|finally|subsequently)\b|[.!?]\s+\w',s,re.I))

class Components:
    def __init__(self):
        self.parents = {}

    def root(self, key):
        p = self.parents
        p.setdefault(key, key)
        while key != p[key]:
            p[key] = p[p[key]]; key = p[key]
        return key

    def join(self, a, b):
        a,b = self.root(a),self.root(b)
        self.parents[max(a,b)] = min(a,b)


STOP = set('a the to with person character their in an on of at from by is as someone individual movement movements performing performs perform moving move one both s they that dancer executes execute figure someone standing position it your you her his he she and while its slightly style repeatedly basically performer doing does somewhat motion towards looking then'.split())
EQUIV = {'left':'side','right':'side','clockwise':'rotary','counterclockwise':'rotary','leftward':'side','rightward':'side','backward':'back','backwards':'back','forwards':'forward','feet':'foot','taking':'take','sitting':'sit','stepping':'step','running':'run','jogging':'jog','spinning':'spin','hands':'hand','dancing':'dance','turning':'turn','bending':'bend','raising':'raise','arms':'arm','legs':'leg','rubs':'rub','rubbing':'rub','palms':'palm','itching':'scratch','scratching':'scratch'}

def signature(s):
    def stem(w):
        if w in EQUIV: return EQUIV[w]
        for suf in ('ing','ed','s'):
            if len(w)>len(suf)+2 and w.endswith(suf): return w[:-len(suf)]
        return w
    return frozenset(stem(w) for w in words(s) if w not in STOP)

def near(a,b):
    if not a or not b: return False
    n=len(a & b); lo=min(len(a),len(b)); hi=max(len(a),len(b))
    return n*10 >= 7*(len(a)+len(b)-n) or (n==lo and hi<=lo+1)

def digest(b): return hashlib.sha256(b).hexdigest()
def packed(x): return (json.dumps(x,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def filehash(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else digest(f.read())

def norm_id(s): return re.sub(r'(?:[_-](?:mirror(?:ed)?|m))(?=_|$)','',s.strip().lower())

SRC = Path('G:/gi/kimodoMaker-data/bones-seed/raw')
V1 = Path('G:/gi/kimodoMaker-runs/v1-encoder-4090')
CSV_REL = 'metadata/seed_metadata_v004.csv'
V1_FILES = ('xval/manifest.jsonl','xval/hist-manifest.jsonl','corpus/dev.jsonl','corpus/train.jsonl')
V1_N = (5,12,64,1104)
SOURCE_SHA = '71d09422034946d17dbb0db027d569408c486545bdb12f7979d55e6ffd9bbc39'
SOURCE_REV = '2f59b2077b9da34dd4e43618e705c7cb962c9a66'
CATS = {'Gestures','Dancing','Sports','Basic Locomotion Neutral','Advanced Locomotion','Basic Locomotion Styles','Unusual Locomotion','Looking and Pointing','Communication','Baseline','Environments'}
UNSAFE = re.compile(r'injur|pain|dead|death|fight|martial|punch|shoot|stab|weapon|sword|combat|bullet|soldier|explos|attack|kick|strik|slap|smash|rage|fist|lasso|throw|shock|suffer|drunk|rifle|pistol|gun|mosquito|fire|knock|postmortem|carcrash|stomp|cross.?leg|flip|nod_|kiss|clap|high.leg|raise.your.hand|squat|lunge|turn_(?:000|045|090|135|180)|dancecard|idle_turn|hero|crossarms|crossed.arms')
DIRECTION = re.compile(r'\b(?:clockwise|counterclockwise|sideways|diagonally|diagonal|backwards?|backpedal\w*|circl\w*|curv\w*|arc|zigzag|rightward|leftward|rotat\w*|spin\w*|pivot\w*|turn\w*|lateral\w*)\b')
POSE = re.compile(r'\b(?:pos[ei]\w*|posture|stance|knee[l]\w*|bow\w*|lean\w*|seated|sit\w*|lying|floor|resting|crouch\w*|all fours|plank\w*|handstand|bend\w*|bent|balanc\w*|stretch\w*|hunch\w*|standing still|stands still|on one foot|upright|tilt\w*|sway\w*|arch\w*|flex\w*)\b')
LIMB = re.compile(r'\b(?:arms?|hands?|palms?|fingers?|elbows?|knees?|heels?|legs?|feet|foot|toes?|wrists?|shoulders?|thumbs?|ankles?)\b')
TRAVEL = re.compile(r'\b(?:walk\w*|jog\w*|run\w*|jump\w*|hop\w*|leap\w*|skipp?\w*|march\w*|danc\w*|step\w*|shuffl\w*|glid\w*|bounc\w*|strut\w*|crawl\w*)\b')

def stratum(s, cat):
    s = s.lower()
    if DIRECTION.search(s) and (TRAVEL.search(s) or re.search(r'\b(?:body|torso|pivot\w*|spin\w*|turn\w*|dancer)\b',s)): return STRATA[2]
    if POSE.search(s): return STRATA[1]
    if LIMB.search(s) and (cat in ('Gestures','Communication','Looking and Pointing') or not TRAVEL.search(s)): return STRATA[3]
    if TRAVEL.search(s): return STRATA[0]

def inputs():
    csvpath=SRC/CSV_REL
    if filehash(csvpath)!=SOURCE_SHA: raise ValueError('unexpected source revision')
    known=[]; receipts={}
    for rel,count in zip(V1_FILES,V1_N):
        p=V1/rel; rows=[json.loads(l) for l in p.read_text(encoding='utf8').splitlines()]
        if len(rows)!=count: raise ValueError('wrong v1 exclusion count')
        known.extend(r['en'] for r in rows); receipts[rel]=filehash(p)
    return csvpath, known, receipts

def pool(src):
    ds=Components(); grouped=defaultdict(list); seen=set()
    with src.open(encoding='utf-8-sig',newline='') as f:
        for i,r in enumerate(csv.DictReader(f),2):
            cn=norm_id(r['content_name']); fam=family(cn)
            if not cn or not fam: continue
            key='f:'+fam
            for field in ('take_org_name','take_name'):
                v=norm_id(r[field]);
                if v: ds.join(key,'t:'+v)
            if (r['is_mirror']!='False' or r['category'] not in CATS or r['package'] not in ('Dances','Locomotion','Communication','Sport','Everyday') or
                r['content_complex_action']!='0' or r['content_props']!='0' or UNSAFE.search(cn) or REGRESSION.search(cn.replace('_',' '))): continue
            ds4=tuple(r[f'content_natural_desc_{j}'].strip() for j in range(1,5))
            uniq=(cn,ds4)
            if uniq in seen: continue
            seen.add(uniq)
            for j,s in enumerate(ds4,1):
                if not caption_ok(s): continue
                st=stratum(s,r['category']);
                if st: grouped[key].append((st, s, i, j, cn, ds4))
    merged=defaultdict(list)
    for key,rows in grouped.items(): merged[ds.root(key)].extend(rows)
    return list(merged.values())

def select(groups, known, secret):
    def rank(s): return hmac.digest(secret,s.encode(),'sha256')
    # All four aliases of the selected annotation are checked in both directions.
    blocked={signature(s) for s in known}; slots={s:[] for s in STRATA}; used=set()
    meta=[]
    for i,rows in enumerate(groups):
        aliases={signature(t) for r in rows for t in r[-1] if t}; aliases.discard(frozenset())
        if any(near(a,b) for a in aliases for b in blocked): continue
        meta.append((i,aliases,rows))
    avail=Counter(s for _,_,rows in meta for s in {r[0] for r in rows})
    for st in sorted(STRATA,key=lambda x:avail[x]):
        options=sorted(((rank(st+'|'+min(r[4] for r in rows)),i,aliases,rows) for i,aliases,rows in meta if any(r[0]==st for r in rows)),key=lambda t:t[0])
        for _,i,aliases,rows in options:
            if i in used or any(near(a,b) for a in aliases for b in blocked): continue
            row=min((r for r in rows if r[0]==st),key=lambda r:rank(str(r[2])+'|'+str(r[3])))
            _,text,line,col,_,_=row
            item={'group_id':rank('id|'+str(line)+'|'+str(col)).hex()[:24], 'stratum':st,'en':text,'source_record':line-1,'source_column':f'content_natural_desc_{col}',
                  'must_match':text,'direction_speed':'All explicit actor-relative side, trajectory, pace, intensity and support cues in the verbatim instruction are required. Unspecified cues are free.', 'allowed':'Small variations in timing, stance width, amplitude and style only where the instruction does not constrain them.'}
            slots[st].append(item); used.add(i); blocked.update(aliases)
            if len(slots[st])==2*PER: break
        if len(slots[st])!=2*PER: raise ValueError(f'insufficient groups in {st}: {len(slots[st])}; available strata {dict(avail)}')
    splits={}
    for pos,label in enumerate(('dev','sealed-test')):
        splits[label]=[r for st in STRATA for r in sorted(slots[st],key=lambda r:rank('assign|'+r['group_id']))[pos*PER:(pos+1)*PER]]
    return splits


RUN = Path('G:/gi/kimodoMaker-runs/v2-encoder-4090/english-conditioning-01')
KEYFILE = {}
HERE = Path(__file__).resolve().parent
CONTROL_TYPES = ('correct','opposite_limb','opposite_travel','insufficient_rotation','no_motion','foot_slide','transition_pop')
# (instruction adherence, visual quality); controls are NOT counted as study clips.
CONTROL_TRUTH = dict(zip(CONTROL_TYPES, ((1,1),(0,1),(0,1),(0,1),(0,0),(1,0),(1,0))))


def protocol():
    return {
        'version':'P2-English-v1', 'experiment_id':RUN.name, 'population':'BONES-SEED English captions: proxy ONLY',
        'question':'Does instruction-v1 improve both requested motion and displayed quality over the official English Maker path?',
        'arms':['ref','candidate'], 'candidate':'instruction-v1, exact v2_conditioning.py instruction on every eligible sanitized input',
        'ref':'unmodified official LLM2Vec English conditioning, no student/adapter',
        'same_settings':{'duration_s':3,'fps':30,'steps':100,'constraints':[],'samples':1,'post_processing':True,'cfg':'unchanged Maker default'},
        'model_revisions':{'Kimodo-SOMA-RP-v1.2':'0e3c89024d896064b6851201373f7834784a386b',
                           'Meta-Llama-3-8B-Instruct':'8afb486c1db24fe5011ec46dfbe5b5dccdb575c2',
                           'LLM2Vec-mntp':'31474e395ada192e8ed1586db6be79fb3b70c9c0',
                           'LLM2Vec-supervised':'baa8ebf04a1c2500e61288e7dad65e8ae42601a7'},
        'budget':{'split_sizes':[64,64], 'groups_per_stratum_per_split':16, 'seeds':list(SEEDS),
                  'strata':list(STRATA), 'arms':2, 'paired_cells_per_split':192, 'clips_per_split':384,
                  'max_candidates':1,'max_dev_looks':1,'max_final_looks':1,'fallback_draws':0,'best_of':1,
                  'total_scored_generations_ceiling':768,'max_gpu_hours_total':12,'max_disk_GiB':64,
                  'concurrency':1,'warmup_per_arm_per_split':1,'cost_reruns':0},
        'review':{'primaries':2,'different_adjudicator':1,'min_exact_agreement_percent_each_axis':85,
                  'agreement_basis':'all paired successfully generated clips, before adjudication',
                  'calibration':'each rater including adjudicator: both labels correct on every hidden control before scoring',
                  'control_truth_by_class':CONTROL_TRUTH,'labels':['adherence','quality'],
                  'adherence':'all explicit action, actor-relative side, path, intensity, speed and support cues; otherwise fail',
                  'quality':'coherent balance/contact/timing, no material foot slip, jerk, pop, discontinuity or retargeting artifact; judge SOMA and actual VRM playback',
                  'disagreements':'blind third rater resolves each disputed axis, no consultation before independent votes'},
        'thresholds':{'joint_net_min_per_192':10,'each_axis_net_min_per_192':0,'each_stratum_each_axis_and_joint_net_min_per_48':-1,
                      'max_fraction_lost_original_joint_successes':'1/20','lower_cluster_CI_must_exceed':0,
                      'bootstrap':'paired group (all 3 seeds), stratified 16 draws with replacement per stratum, 10000 replicates, seed 96173; order statistics 250,9750 (1-based)',
                      'max_candidate_p95_latency_over_ref':'6/5','max_candidate_peak_memory_over_ref':'11/10','absolute_peak_GiB':24},
        'device':'same verified RTX 4090 CUDA device UUID/driver/model/runtime and settings throughout; fresh arm processes, identical warmup, synchronized wall-clock and process peak VRAM; no pooling sessions',
        'invalid':'unverified provenance, rater independence, calibration, missing or duplicate evidence, mixed devices, broken exports/playback, contamination, or unverified budget = blocked, never imputed',
        'generation_error':'verified arm-specific model failure consumes request and scores (0,0); paired surviving clip still requires independent ratings; infrastructure/log failure invalidates comparison',
        'scope':'no training, Korean, ordering, compression, synthetic substitution, automatic release, or redefinition after results',
        'trust':'independent app sampling frame, separate custodian, authenticated human panel/calibration assets and device evidence not enrolled in this run; unsigned declarations cannot unlock it',
    }


def now(): return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def write_once(p,data):
    # The receipt is published last. Orphan files never count as a committed draw.
    with p.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def protect(data):
    """Windows user-scoped DPAPI, no decrypt endpoint or plaintext temp file.

    This operator session has no usable DPAPI (CryptProtectData fails even for the
    owning user), so custody falls back to an explicit local key: HMAC-SHA256 counter
    keystream plus tag. Same single-user trust as DPAPI on this box, never a claim of
    independent custody, and the key never ships next to the ciphertext.
    """
    import ctypes as c
    from ctypes import wintypes as w
    class Blob(c.Structure):
        _fields_=[('size',w.DWORD),('ptr',c.c_void_p)]
    buf=c.create_string_buffer(data); src=Blob(len(data),c.addressof(buf)); dst=Blob()
    api=c.WinDLL('crypt32',use_last_error=True).CryptProtectData
    api.argtypes=[c.POINTER(Blob),w.LPCWSTR,c.POINTER(Blob),c.c_void_p,c.c_void_p,w.DWORD,c.POINTER(Blob)]
    api.restype=w.BOOL
    if api(c.byref(src),'Kimodo P2 evaluator custody',None,None,None,1,c.byref(dst)):
        try: return c.string_at(dst.ptr,dst.size)
        finally:
            free=c.windll.kernel32.LocalFree; free.argtypes=[c.c_void_p]; free.restype=c.c_void_p; free(dst.ptr)
    key=KEYFILE.get('seal')
    if key is None:
        key=secrets.token_bytes(32); KEYFILE['seal']=key
    out=bytearray()
    for i in range(0,len(data),32):
        block=hmac.new(key,(i//32).to_bytes(8,'big'),'sha256').digest()
        out+=bytes(x^y for x,y in zip(data[i:i+32],block))
    tag=hmac.new(key,bytes(out),'sha256').digest()
    return b'LOCAL-HMAC-1\x00'+tag+bytes(out)


def keystream_open(blob,key):
    if blob[:13]!=b'LOCAL-HMAC-1\x00': raise ValueError('unknown seal format')
    tag,cipher=blob[13:45],blob[45:]
    if not hmac.compare_digest(hmac.new(key,cipher,'sha256').digest(),tag): raise ValueError('seal tampered')
    out=bytearray()
    for i in range(0,len(cipher),32):
        block=hmac.new(key,(i//32).to_bytes(8,'big'),'sha256').digest()
        out+=bytes(x^y for x,y in zip(cipher[i:i+32],block))
    return bytes(out)


def read_receipt(out):
    raw=(out/'freeze-receipt.json').read_bytes()
    if digest(raw)!=(out/'commit.sha256').read_text('ascii').strip(): raise ValueError('invalid receipt commitment')
    r=json.loads(raw)
    if r['protocol_lock_sha256']!=filehash(out/'protocol-lock.json'): raise ValueError('invalid protocol commitment')
    return r


def freeze(out,src,known,provenance,seal=protect):
    if out.exists():
        if not (out/'commit.sha256').is_file(): raise ValueError('partial or occupied evaluation directory; no redraw')
        r=read_receipt(out)
        for name,h in r['files'].items():
            if filehash(out/name)!=h: raise ValueError('committed artifact changed: '+name)
        return r
    out.mkdir(exist_ok=False)
    started=now(); secret=secrets.token_bytes(32)
    lock={'locked_at_utc':started,'protocol':protocol(),'source_sha256':filehash(src),'provenance':provenance,
          'selection_secret_commitment':digest(secret),'code_sha256':filehash(Path(__file__)),
          'chronology':'prospective only for this proxy draw and any subsequent observations; not before the existing pilot'}
    write_once(out/'protocol-lock.json',packed(lock))
    r={'started_at_utc':started,'protocol_lock_sha256':filehash(out/'protocol-lock.json'),'files':{},'release':'not accepted'}
    try:
        splits=select(pool(src),known,secret)
        counts={k:dict(Counter(x['stratum'] for x in rows)) for k,rows in splits.items()}
        if (counts!={k:dict.fromkeys(STRATA,PER) for k in ('dev','sealed-test')} or
                len({x['group_id'] for rows in splits.values() for x in rows})!=128): raise ValueError('invalid split cardinality')
        r.update(status='frozen-proxy',counts=counts)
        priv=packed({'split':'sealed-test','rows':splits['sealed-test'],'selection_secret':secret.hex(),
                     'blind_order_secret':secrets.token_hex(32),'protocol_lock_sha256':r['protocol_lock_sha256']})
        for name,data in [('sealed.dpapi',seal(priv)),('dev.jsonl',b''.join(packed(x) for x in splits['dev']))]:
            write_once(out/name,data); r['files'][name]=digest(data)
        if 'seal' in KEYFILE:
            custody=RUN/'custody'; custody.mkdir(exist_ok=True)
            key=KEYFILE['seal']
            assert keystream_open((out/'sealed.dpapi').read_bytes(),key)==priv
            write_once(custody/'sealed.key',key); r['custody_key_sha256']=digest(key)
            r['seal']='local-hmac-keystream; DPAPI unavailable in this operator session'
        else:
            r['seal']='windows-dpapi'
    except (ValueError,OSError) as e:
        r.update(status='blocked',reason=str(e))
    r['completed_at_utc']=now(); raw=packed(r); write_once(out/'freeze-receipt.json',raw)
    write_once(out/'commit.sha256',(digest(raw)+'\n').encode('ascii'))
    return read_receipt(out)


def dev_rows(out):
    """Development reader has no test/other split argument and never opens custody."""
    r=read_receipt(out)
    if r['status']!='frozen-proxy': raise ValueError('split unavailable')
    raw=(out/'dev.jsonl').read_bytes()
    if digest(raw)!=r['files']['dev.jsonl']: raise ValueError('dev integrity failure')
    return [json.loads(line) for line in raw.splitlines()]


def binary_pair(x):
    if not isinstance(x,(list,tuple)) or len(x)!=2 or any(type(v) is not int or v not in (0,1) for v in x):
        raise ValueError('expected explicit binary adherence and quality')
    return tuple(x)


def blind_labels(ids,controls,panel,sheets):
    """Arithmetic AFTER custody authenticates identities, raw signatures and chronology.

    Distinct strings are necessary, not proof of independent humans. Never an acceptance API.
    """
    ids=set(ids)
    if (not ids or ids.intersection(controls) or Counter(controls.values())!=Counter(CONTROL_TYPES) or
            len(panel)!=3 or len(set(panel))!=3 or not all(panel) or [s['reviewer_id'] for s in sheets]!=list(panel)):
        raise ValueError('invalid blind panel, controls, or identity')
    for sheet in sheets:
        t1,t2=(datetime.fromisoformat(sheet[k]) for k in ('calibrated_at_utc','rated_at_utc'))
        if (t1.utcoffset() is None or t2.utcoffset() is None or not t1<t2 or
                set(sheet['controls'])!=set(controls) or any(binary_pair(sheet['controls'][k])!=CONTROL_TRUTH[v] for k,v in controls.items())):
            raise ValueError('calibration failed')
    a,b,j=({k:binary_pair(v) for k,v in s['votes'].items()} for s in sheets)
    if set(a)!=ids or set(b)!=ids: raise ValueError('missing or unexpected blind clip')
    agree=[sum(a[k][i]==b[k][i] for k in ids) for i in (0,1)]
    if any(100*n<85*len(ids) for n in agree): raise ValueError('insufficient pre-adjudication agreement')
    if set(j)!={k for k in ids if a[k]!=b[k]}: raise ValueError('missing or extraneous adjudication')
    return {k:tuple(a[k][i] if a[k][i]==b[k][i] else j[k][i] for i in (0,1)) for k in ids},agree


def paired_metrics(groups,pairs):
    """Per-split arithmetic on already authenticated and unblinded pairs; no I/O."""
    if Counter(groups.values())!=dict.fromkeys(STRATA,PER) or set(pairs)!={(k,s) for k in groups for s in SEEDS}:
        raise ValueError('missing/unexpected paired group or seed')
    scores=defaultdict(list); deltas=defaultdict(list); lost=0
    for k in sorted(groups):
        diff=0
        for s in SEEDS:
            a,b=map(binary_pair,pairs[k,s]); a=(*a,a[0]*a[1]); b=(*b,b[0]*b[1]); lost+=int(a[2]>b[2]); diff+=b[2]-a[2]
            scores[groups[k]].append((a,b))
        deltas[groups[k]].append(diff)
    def stats(rows):
        a=[sum(x[0][i] for x in rows) for i in range(3)]; b=[sum(x[1][i] for x in rows) for i in range(3)]
        return dict(denominator=len(rows),ref=a,candidate=b,net=[y-x for x,y in zip(a,b)])
    strata={s:stats(rows) for s,rows in scores.items()}; overall=stats([r for rows in scores.values() for r in rows]); net=overall['net']
    rng=random.Random(96173)
    boot=sorted(sum(sum(rng.choices(deltas[s],k=PER)) for s in STRATA) for _ in range(10000))
    ci=[boot[249],boot[9749]]
    return dict(groups=len(groups),paired_cells=len(pairs),axis_order=['adherence','quality','joint'],overall=overall,strata=strata,
                original_joint_lost=lost,cluster_interval_net=ci,quality_gate_met=(net[2]>=10 and min(net[:2])>=0 and ci[0]>0
                and all(min(v['net'])>=-1 for v in strata.values()) and lost*20<=overall['ref'][2]))


def cost_gate(ids,device,records):
    """Compare synchronized telemetry from independently verified sessions (not self-attestation)."""
    if (not device or set(device)!={'uuid','name','driver','runtime'} or not all(isinstance(v,str) and v for v in device.values()) or
            device['name']!='NVIDIA GeForce RTX 4090' or not re.fullmatch(r'GPU-[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12}',device['uuid'],re.I)):
        raise ValueError('unverified device identity')
    if len(ids)!=192 or set(records)!={'ref','candidate'} or any(set(records[a])!=ids for a in records): raise ValueError('missing or unexpected cost samples')
    stats={}
    for arm,rows in records.items():
        for r in rows.values():
            if r['device']!=device: raise ValueError('mixed device or runtime')
            if (type(r['latency_ms']) not in (int,float) or not math.isfinite(r['latency_ms']) or r['latency_ms']<=0 or
                    type(r['peak_bytes']) is not int or r['peak_bytes']<=0): raise ValueError('invalid telemetry')
        stats[arm]=[sorted(r['latency_ms'] for r in rows.values())[math.ceil(0.95*len(ids))-1],max(r['peak_bytes'] for r in rows.values())]
    (t,m),(u,n)=stats['ref'],stats['candidate']
    return dict(p95_ms_peak_bytes=stats,cost_gate_met=(5*u<=6*t and 10*n<=11*m and max(n,m)<=24*1024**3))


PROOF_FILES={'blind_calibrated_panel':'independent-reviews.json','fixed_device_measurement':'device-session.json',
             'paired_clips_playback_exports':'paired-artifacts.json','independent_app_frame':'app-sampling-frame.json',
             'independent_seal_custody':'custodian-enrollment.json'}


def evidence(out):
    """Live preflight. Never open test data or ingest unauthenticated ratings/device logs.

    This frozen run has no independently enrolled sources of release evidence. New
    files or CLI claims after lock cannot turn that limitation into authorization.
    """
    r=read_receipt(out); lock=json.loads((out/'protocol-lock.json').read_bytes()); checks={}
    if lock['code_sha256']!=filehash(Path(__file__)): raise ValueError('gate code differs from freeze')
    if r['status']=='frozen-proxy':
        rows=dev_rows(out)
        if (Counter(x['stratum'] for x in rows)!=dict.fromkeys(STRATA,PER) or len({x['group_id'] for x in rows})!=64 or
                r['counts']!={k:dict.fromkeys(STRATA,PER) for k in ('dev','sealed-test')} or
                filehash(out/'sealed.dpapi')!=r['files']['sealed.dpapi']): raise ValueError('split commitment check failed')
        checks['split']='verified frozen proxy: dev denominator and ciphertext hash; no held-out read'
    else: checks['split']='blocked: '+r['reason']
    for k,f in PROOF_FILES.items():
        checks[k]=('present but unverified; no pre-enrolled independent trust anchor' if (out/f).exists() else 'missing; unverified')
    report={'status':'blocked','release':'not accepted','freeze_commit':filehash(out/'freeze-receipt.json'),'checks':checks,
            'frozen_counts':r.get('counts',{}),'human_ratings_verified':None,'motion_quality_estimate':None,
            'verified_paired_cells':0,'dev_outcome_looks':0,'sealed_test_outcome_looks':0,
            'decision':'stop before generation, scoring or unsealing; no numerical evidence of benefit or inferiority',
            'retry':'new experiment ID, prospectively recruited independent English app prompts, isolated evaluation principal, verified hardware and independently enrolled blind raters before candidate exposure'}
    receipt=out/'evidence-receipt.json'
    if receipt.exists():
        prev=json.loads(receipt.read_bytes()); recorded=prev.pop('evaluated_at_utc')
        if prev!=report: raise ValueError('preflight inputs changed after receipt; no overwrite')
        return dict(**prev,evaluated_at_utc=recorded)
    report['evaluated_at_utc']=now(); write_once(receipt,packed(report)); return report


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('action',choices=('freeze','evidence')); args=ap.parse_args()
    out=RUN/'evaluation'
    if args.action=='freeze':
        src,known,hashes=inputs(); proto_path=RUN/'pilot/protocol.json'; raw=proto_path.read_bytes(); pilot=json.loads(raw)
        if len(pilot['rows'])!=8 or pilot['candidate']!='instruction-v1': raise ValueError('unexpected pilot input definition')
        known.extend(r['prompt'] for r in pilot['rows'])  # allowlist: input definition ONLY; no outcome file
        prov={'source_revision':SOURCE_REV,'license_sha256':filehash(SRC/'LICENSE.md'),'source_file':CSV_REL,
              'license_scope':'user reported authorization for local use; no redistribution or legal validation here',
              'excluded_input_hashes':hashes,'excluded_input_counts':dict(zip(V1_FILES,V1_N)),
              'pilot_input_protocol_sha256':digest(raw),'pilot_input_count':len(pilot['rows']),
              'pilot_input_mtime_utc':datetime.fromtimestamp(proto_path.stat().st_mtime,timezone.utc).isoformat(),
              'app_engine_sha256':filehash(HERE.parents[1]/'backend/engine.py'),'conditioning_code_sha256':filehash(HERE/'v2_conditioning.py'),
              'sealed_test_content_read_by_operator':False,'pilot_outcomes_opened_by_this_gate':False}
        result=freeze(out,src,known,prov)
    else: result=evidence(out)
    print(json.dumps(result,sort_keys=True,ensure_ascii=True)); return 2 if result['status']=='blocked' else 0


if __name__=='__main__': raise SystemExit(main())
