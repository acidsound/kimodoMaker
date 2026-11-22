"""Run one frozen Kimodo v1 arm locally; invoke separately for ref, en and ko."""

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
from importlib import import_module, metadata
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
KIMODO_CODE = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
MOTION = ("nvidia/Kimodo-SOMA-RP-v1.2", "0e3c89024d896064b6851201373f7834784a386b")
STUDENT = ("google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e")
TEACHER = {
    "llama": ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
    "mntp": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    "supervised": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
}
TEACHER_FILES = {
    "llama": ["config.json", "model.safetensors.index.json", "model-*.safetensors"],
    "mntp": ["config.json", "adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json"],
    "supervised": ["adapter_config.json", "adapter_model.safetensors"],
}
SEEDS = (17, 43, 97)
STEPS = 100
FPS = 30
POLL_NS = 20_000_000
ARTIFACTS = ("motion.json", "motion.bvh", "motion.npz")
WARMUP = {
    "id": "warmup", "seconds": 2.0,
    "en": "The person sits still and slowly nods twice.",
    "ko": "사람이 가만히 앉아서 천천히 두 번 고개를 끄덕인다.",
}


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_manifest(raw, *, pilot):
    rows, seen = [], set()
    for line, text in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        row = json.loads(text, parse_constant=reject_constant)
        if not isinstance(row, dict):
            raise ValueError(f"Line {line}: expected an object")
        if not all(isinstance(row.get(k), str) and row[k].strip()
                   for k in ("id", "group_id", "split", "en", "ko")):
            raise ValueError(f"Line {line}: invalid id/group_id/split/en/ko")
        key = row["id"].lower()
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", row["id"]) or key in seen:
            raise ValueError(f"Line {line}: unsafe or duplicate id: {row['id']!r}")
        if type(row.get("kind")) is not int or row["kind"] not in range(1, 5):
            raise ValueError(f"Line {line}: kind must be an integer in 1..4")
        if type(row.get("seconds")) not in (int, float) or not 0.5 <= row["seconds"] <= 15:
            raise ValueError(f"Line {line}: seconds outside Maker's [0.5, 15] interval")
        if (type(row.get("unseen")) is not bool or type(row.get("footing")) is not bool
                or not isinstance(row.get("intent"), dict)):
            raise ValueError(f"Line {line}: missing or invalid unseen/footing/intent")
        if any(len(row[k].strip()) > 1000 for k in ("en", "ko")):
            raise ValueError(f"Line {line}: prompt exceeds Maker's 1000-character bound")
        seen.add(key)
        rows.append(row)
    splits = {r["split"] for r in rows}
    if len(splits) != 1 or "train" in splits:
        raise ValueError("Use one nonempty, non-training evaluation split per output root")
    if not pilot and (splits - {"dev", "test"} or Counter(r["kind"] for r in rows) != {k: 16 for k in range(1, 5)}):
        raise ValueError("v1 requires one dev/test split with 16 rows for each kind; use --pilot for a probe")
    return rows


def request_plan(rows):
    return [(row, seed) for row in rows for seed in SEEDS]


def maker_prompt(row, arm):
    prompt = row["ko" if arm == "ko" else "en"].strip()
    return prompt if prompt[-1] in ".!?" else prompt + "."


def ceil_unit(value, unit):
    return (value + unit - 1) // unit


def nearest_rank_p95_ms(values):
    if not values:
        return None
    return ceil_unit(sorted(values)[ceil_unit(95 * len(values), 100) - 1], 1_000_000)


def decode_student_response(raw, count):
    if not raw.endswith(b"\n"):
        raise ValueError("Student response must be one complete JSON line")
    response = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(response, dict):
        raise ValueError("Student response is not an object")
    if "error" in response:
        raise RuntimeError(f"Student encoder: {response['error']}")
    vectors, lengths = response.get("vectors"), response.get("lengths")
    if (not isinstance(vectors, list) or len(vectors) != count
            or not isinstance(lengths, list) or len(lengths) != count
            or any(type(n) is not int or n != 1 for n in lengths)):
        raise ValueError("Student response batch size/lengths do not match")
    if any(not isinstance(v, list) or len(v) != 4096 or any(type(x) not in (int, float) for x in v)
           for v in vectors):
        raise ValueError("Expected 4096 numeric values per student text")
    return vectors, lengths


class StudentProvider:
    """One sequential request per JSONL line; no embedding cache or fallback."""

    def __init__(self, python, script, checkpoint, stderr_path, env, timeout, *, device="cuda:0"):
        self.command = [str(python), str(script), "serve", "--checkpoint", str(checkpoint)]
        self.stderr_path, self.env, self.timeout, self.device = stderr_path, env, timeout, device
        self.inbox = Queue()

    def __enter__(self):
        with self.stderr_path.open("xb") as err:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=err, env=self.env)
        self.reader = threading.Thread(target=self._read, name="student-stdout", daemon=True)
        self.reader.start()
        return self

    def _read(self):
        try:
            for line in self.proc.stdout:
                self.inbox.put(line)
        finally:
            self.inbox.put(None)

    def __exit__(self, *_):
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.reader.join()
        self.proc.stdout.close()

    def __call__(self, texts):
        import torch

        line = json.dumps({"texts": texts}, ensure_ascii=False).encode("utf-8") + b"\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        try:
            raw = self.inbox.get(timeout=self.timeout)
        except Empty as exc:
            self.proc.kill()  # no late replies may be associated with a later request
            self.proc.wait()
            raise TimeoutError(f"Student encoder timed out after {self.timeout}s; see {self.stderr_path}") from exc
        if raw is None:
            raise RuntimeError(f"Student encoder closed stdout; see {self.stderr_path}")
        try:
            vectors, lengths = decode_student_response(raw, len(texts))
        except ValueError:
            self.proc.kill()
            self.proc.wait()
            raise  # invalid framing may shift subsequent messages; never reuse this child
        data = torch.tensor(vectors, dtype=torch.float32, device="cpu").reshape(len(texts), 1, 4096)
        if not torch.isfinite(data).all().item():
            raise ValueError("Student encoder emitted vectors not representable as finite float32")
        return data.to(self.device), lengths


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_json(path, obj):
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("xb") as file:
        file.write(json_bytes(obj))
        file.flush()
        os.fsync(file.fileno())
    tmp.replace(path)


def freeze_json(path, obj):
    if path.exists():
        previous = json.loads(path.read_bytes())
        if previous != obj:
            different = sorted(k for k in previous.keys() | obj.keys() if previous.get(k) != obj.get(k))
            raise ValueError(f"Incompatible frozen settings in {path}: {', '.join(different)}")
    else:
        write_json(path, obj)


def append_jsonl(path, obj):
    with path.open("ab") as file:
        file.write(json_bytes(obj))
        file.flush()
        os.fsync(file.fileno())


def completed_requests(path):
    if not path.exists():
        return {}
    result = {}
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        # Preserve evidence of a torn write; keep the journal parseable on restart.
        valid, _, tail = content.rpartition(b"\n")
        kept = valid + b"\n" if valid else b""
        with path.with_name(f"{path.name}.torn-{uuid.uuid4().hex}").open("xb") as archive:
            archive.write(tail)
            archive.flush()
            os.fsync(archive.fileno())
        with path.open("r+b") as file:
            file.truncate(len(kept))
            os.fsync(file.fileno())
        content = kept
    for line in content.split(b"\n")[:-1]:
        rec = json.loads(line)
        key = (rec["id"], rec["seed"])
        if key in result:
            raise ValueError(f"Journal has a later entry for a terminal request: {key}")
        if rec["status"] in ("ok", "failed"):
            result[key] = rec
        elif rec["status"] != "started":
            raise ValueError(f"Unknown journal status: {rec['status']}")
    return result


@contextmanager
def output_lock(out):
    import msvcrt

    with (out / ".run.lock").open("a+b") as file:
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Another arm has the output root locked: {out}") from exc
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


class Nvml:
    """Query device-wide framebuffer accounting without starting a CUDA context."""

    class Memory(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]

    def __enter__(self):
        dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvml.dll"
        self.dll = ctypes.WinDLL(str(dll))
        self.check("nvmlInit_v2")
        try:
            count = ctypes.c_uint()
            self.check("nvmlDeviceGetCount_v2", ctypes.byref(count))
            if count.value != 1:
                raise RuntimeError(f"v1 needs exactly one physical NVIDIA GPU; NVML returned {count.value}")
            self.device = ctypes.c_void_p()
            self.check("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(0), ctypes.byref(self.device))
            cur, pending = ctypes.c_uint(), ctypes.c_uint()
            self.check("nvmlDeviceGetDriverModel", self.device, ctypes.byref(cur), ctypes.byref(pending))
            self.identity = {
                "name": self.string("nvmlDeviceGetName", self.device),
                "uuid": self.string("nvmlDeviceGetUUID", self.device),
                "driver": self.string("nvmlSystemGetDriverVersion"),
                "driver_model": {0: "WDDM", 1: "TCC"}.get(cur.value, f"unknown-{cur.value}"),
                "nvml_total_bytes": self.memory().total,
            }
            if "RTX 4090" not in self.identity["name"]:
                raise RuntimeError(f"Expected an RTX 4090, found {self.identity['name']}")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        self.check("nvmlShutdown")

    def check(self, name, *args):
        rc = getattr(self.dll, name)(*args)
        if rc:
            raise RuntimeError(f"{name} returned NVML error {rc}")

    def string(self, name, *args):
        buffer = ctypes.create_string_buffer(128)
        self.check(name, *args, buffer, ctypes.c_uint(len(buffer)))
        return buffer.value.decode("ascii")

    def memory(self):
        data = self.Memory()
        self.check("nvmlDeviceGetMemoryInfo", self.device, ctypes.byref(data))
        return data

    def used_bytes(self):
        return self.memory().used


class MemoryTrace:
    def __init__(self, path, read):
        self.path, self.read = path, read
        self.stop = threading.Event()
        self.stats = {"interval_ns": POLL_NS, "samples": 0, "peak_used_bytes": 0, "max_gap_ns": 0, "errors": 0}
        self.previous = None

    def sample(self, phase):
        now = time.perf_counter_ns()
        sample = {"monotonic_ns": now, "phase": phase}
        try:
            used = self.read()
            sample["used_bytes"] = used
            if phase != "idle":
                self.stats["peak_used_bytes"] = max(self.stats["peak_used_bytes"], used)
            else:
                self.stats["idle_used_bytes"] = used
        except Exception as exc:
            sample["error"] = f"{type(exc).__name__}: {exc}"
            self.stats["errors"] += 1
        if self.previous is not None:
            self.stats["max_gap_ns"] = max(self.stats["max_gap_ns"], now - self.previous)
        self.previous = now
        self.stats["samples"] += 1
        self.file.write(json_bytes(sample))

    def __enter__(self):
        self.file = self.path.open("xb", buffering=0)
        self.timer = ctypes.WinDLL("winmm.dll") if os.name == "nt" else None
        self.stats["windows_one_ms_timer"] = bool(self.timer and self.timer.timeBeginPeriod(1) == 0)
        self.sample("idle")
        if self.stats["errors"]:
            if self.stats["windows_one_ms_timer"]:
                self.timer.timeEndPeriod(1)
            self.file.close()
            raise RuntimeError(f"Idle NVML sample failed; see {self.path}")
        self.sample("start")
        self.thread = threading.Thread(target=self.poll, name="nvml-20ms", daemon=True)
        self.thread.start()
        return self

    def poll(self):
        try:
            target = time.perf_counter_ns() + POLL_NS
            while not self.stop.wait(max(0, target - time.perf_counter_ns()) / 1e9):
                self.sample("poll")
                target += POLL_NS
                now = time.perf_counter_ns()
                if target < now:
                    target += ceil_unit(now - target, POLL_NS) * POLL_NS
        except Exception as exc:
            self.stats["thread_error"] = f"{type(exc).__name__}: {exc}"

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        try:
            with self.file:
                self.sample("end")
                os.fsync(self.file.fileno())
        finally:
            if self.stats["windows_one_ms_timer"]:
                self.timer.timeEndPeriod(1)
        self.stats["peak_used_mib"] = ceil_unit(self.stats["peak_used_bytes"], 1024 * 1024)


def inventory():
    fields = "pid,process_name,gpu_uuid,used_gpu_memory"
    try:
        proc = subprocess.run(["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
        return {"utc": utc_now(), "return_code": proc.returncode,
                "compute_apps_csv": proc.stdout, "stderr": proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"utc": utc_now(), "error": str(exc)}


def pinned_snapshot(identity, cache, files=None):
    from huggingface_hub import snapshot_download

    repo, revision = identity
    path = Path(snapshot_download(repo, revision=revision, local_files_only=True, cache_dir=cache,
                                  allow_patterns=files))
    if path.name != revision or not path.is_dir():
        raise RuntimeError(f"Missing exact local snapshot: {repo}@{revision}")
    return path.absolute()


@contextmanager
def override_environment(updates):
    previous = {key: os.environ.get(key) for key in updates}
    for key, val in updates.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val
    try:
        yield
    finally:
        for key, val in previous.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val


def load_pinned_motion(snapshot, encoder):
    # The pinned public loader takes a registry name, not a checkpoint path. Override
    # only its path resolver, then use its real injection and construction code.
    loader = import_module("kimodo.model.load_model")
    resolver = loader._resolve_hf_model_path

    def resolve(name):
        if loader.MODEL_NAMES.get(name) != MOTION[0]:
            raise RuntimeError(f"Unexpected Kimodo registry entry: {name}")
        return snapshot

    with override_environment({"CHECKPOINT_DIR": None}):
        loader._resolve_hf_model_path = resolve
        try:
            model = loader.load_model(MOTION[0].split("/")[1], device="cuda:0", text_encoder=encoder)
        finally:
            loader._resolve_hf_model_path = resolver
    if model.text_encoder is not encoder or model.fps != FPS:
        raise RuntimeError("Wrong Kimodo encoder/FPS")
    return model


def load_teacher(cache, session):
    from kimodo.model import LLM2VecEncoder

    snap = {key: pinned_snapshot(pinned, cache, TEACHER_FILES[key]) for key, pinned in TEACHER.items()}
    # The mntp adapter refers to the *unpinned* Llama by repo ID. Copy only that
    # adapter and point it to the exact local Llama, keeping tokenizer/config intact.
    with TemporaryDirectory(prefix="teacher-mntp-", dir=session) as temp:
        mntp = Path(temp)
        for file in snap["mntp"].iterdir():
            if file.is_file() and file.suffix in (".json", ".safetensors"):
                shutil.copyfile(file, mntp / file.name)
        config_path = mntp / "adapter_config.json"
        config = json.loads(config_path.read_bytes())
        if config.get("base_model_name_or_path") != TEACHER["llama"][0]:
            raise RuntimeError("Unexpected base model in official mntp adapter")
        config.update(base_model_name_or_path=str(snap["llama"]), revision=None)
        config_path.write_bytes(json_bytes(config))
        with override_environment({"TEXT_ENCODERS_DIR": None, "TEXT_ENCODER_DEVICE": "cuda:0",
                                   "HUGGINGFACE_CACHE_DIR": None}):
            encoder = LLM2VecEncoder(str(mntp), str(snap["supervised"]), "bfloat16", 4096, device="cuda:0")
        if encoder.model.model.config._name_or_path != TEACHER["llama"][0]:
            raise RuntimeError("Official Llama prompt formatting changed")
        return encoder


def generate_clip(model, row, arm, seed, steps, directory, stage):
    import numpy as np
    import torch
    from backend.engine import export_take
    from kimodo.tools import seed_everything

    stage("generation")
    prompt = maker_prompt(row, arm)
    frames = round(row["seconds"] * model.fps)
    seed_everything(seed)
    with torch.no_grad():
        batch = model([prompt], [frames], num_denoising_steps=steps, constraint_lst=[],
                      num_samples=1, multi_prompt=True, post_processing=True, return_numpy=True)
    output = {name: value[0] for name, value in batch.items()}
    if output["posed_joints"].shape != (frames, 77, 3):
        raise ValueError(f"Expected {frames} frames with 77 joints, got {output['posed_joints'].shape}")
    if any(not np.isfinite(data).all() for data in output.values()):
        raise ValueError("Kimodo generated non-finite motion data")
    if directory is None:
        return {"fps": float(model.fps), "frame_count": frames}
    stage("export")
    data = export_take(output, model.output_skeleton, model.fps, {
        "prompt": row["ko" if arm == "ko" else "en"].strip(), "duration": row["seconds"],
        "seed": seed, "steps": steps, "model": MOTION[0].split("/")[1], "constraintCount": 0,
        "modelRevision": MOTION[1], "arm": arm, "id": row["id"],
    }, directory)
    for name in ARTIFACTS:
        with (directory / name).open("r+b") as file:
            if not os.fstat(file.fileno()).st_size:
                raise ValueError(f"Empty export: {name}")
            os.fsync(file.fileno())
    return data


def run_request(model, row, seed, args, arm_dir, session, *, warmup=False):
    journal = session / "warmup.jsonl" if warmup else arm_dir / "requests.jsonl"
    key = f"{row['id']}-{seed}"
    staged = session / "staged" / key
    final = session / key if warmup else arm_dir / key
    # Only missing terminal records are passed here. Retain any exports left
    # behind if the last process died after publish but before journalling.
    if final.exists():
        final.rename(session / f"interrupted-{key}")
    start = time.perf_counter_ns()
    common = {"arm": args.arm, "id": row["id"], "seed": seed, "seconds": row["seconds"],
              "steps": args.steps, "session_id": session.name, "kind": row.get("kind"),
              "started_monotonic_ns": start, "started_utc": utc_now()}
    append_jsonl(journal, common | {"status": "started"})
    phase = "prepare"

    def stage(value):
        nonlocal phase
        phase = value

    try:
        if not args.no_export:
            staged.mkdir()
        info = generate_clip(model, row, args.arm, seed, args.steps, None if args.no_export else staged, stage)
        phase = "publish"
        if not args.no_export:
            staged.rename(final)
        bin_paths = {} if args.no_export else {n: (final / n).relative_to(arm_dir).as_posix() for n in ARTIFACTS}
        record = {"status": "ok", "artifacts": bin_paths} | info
    except Exception as exc:
        record = {"status": "failed", "phase": phase, "error_type": type(exc).__name__, "error": str(exc),
                  "partial_directory": staged.relative_to(arm_dir).as_posix() if staged.exists() else None}
    end = time.perf_counter_ns()
    record.update(common, ended_monotonic_ns=end, latency_ns=end - start,
                  latency_ms=ceil_unit(end - start, 1_000_000))
    append_jsonl(journal, record)
    print(f"{args.arm} {key} {record['status']} {record['latency_ms']} ms"
          + (f": {record['error']}" if record["status"] == "failed" else ""), flush=True)
    return record


def student_settings(args):
    if args.arm == "ref":
        return None, None
    script = Path(__file__).with_name("v1_student.py")
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}}
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    if args.student_hf_cache:
        env["HF_HUB_CACHE"] = str(args.student_hf_cache)
        env["HUGGINGFACE_HUB_CACHE"] = str(args.student_hf_cache)
        env["TRANSFORMERS_CACHE"] = str(args.student_hf_cache)
    command = [str(args.student_python), "-B", "-c", "import sys,json;from importlib.metadata import distributions;"
               "print(json.dumps({'python':list(sys.version_info[:3]), 'packages':"
               "{d.metadata['Name']:d.version for d in distributions() if d.metadata['Name']}}))"]
    student_env = json.loads(subprocess.check_output(command, env=env, timeout=30))
    if student_env["python"][:2] != [3, 13]:
        raise RuntimeError("v1 student must use its separate Python 3.13 environment")
    meta = args.checkpoint.with_name("run.json")
    return {"python_executable": str(args.student_python), "environment": student_env,
            "entry_sha256": sha256(script), "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "checkpoint_run_json_sha256": sha256(meta) if meta.is_file() else None,
            "dtype": "bfloat16", "offloading": "none", "backbone": list(STUDENT),
            "cache_env": {name: env.get(name) for name in ("HF_HUB_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE",
                                                           "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")},
            "ipc_timeout_seconds": args.student_timeout}, env


def backend_settings():
    dist = metadata.distribution("kimodo")
    if json.loads(dist.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id") != KIMODO_CODE:
        raise RuntimeError("Install Kimodo at the pinned revision in backend/requirements.txt")
    if sys.version_info[:2] != (3, 10):
        raise RuntimeError("v1 parent must use the Maker Python 3.10 environment")
    return {"python_executable": str(Path(sys.executable).resolve()), "python": list(sys.version_info[:3]),
            "packages": {k: metadata.version(k) for k in ("torch", "numpy", "scipy", "transformers", "peft",
                                                          "huggingface-hub", "kimodo", "safetensors")},
            "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF",
                    "PYTORCH_CUDA_ALLOC_CONF", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE",
                    "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def summarize(arm_dir, plan, current, pilot):
    records = completed_requests(arm_dir / "requests.jsonl")
    sessions_dir = arm_dir / "sessions"
    sessions = [json.loads(p.read_bytes()) for p in sorted(sessions_dir.glob("*/session.json"))]
    measurements = [s["memory"] for s in sessions if s.get("memory")]
    count = len(records)
    ok = sum(r["status"] == "ok" for r in records.values())
    peak = max((m.get("peak_used_mib", 0) for m in measurements), default=0)
    measurement_complete = (count == len(plan) and len(sessions) == 1 and sessions[0]["status"] == "complete"
                            and len(measurements) == 1 and not measurements[0]["errors"]
                            and "thread_error" not in measurements[0] and measurements[0]["samples"] >= 4)
    report = {"schema": "kimodo.v1-generate.summary.v1", "arm": arm_dir.name, "pilot": pilot,
              "expected": len(plan), "completed": count, "ok": ok, "failed": count - ok,
              "p95_ms": nearest_rank_p95_ms([r["latency_ns"] for r in records.values()]),
              "p95_basis": "all terminal requests including failures, excludes warmup; pilots may skip export",
              "peak_used_mib": peak or None, "single_session_measurement_complete": measurement_complete,
              "sessions": [s["session_id"] for s in sessions], "latest_session": current}
    write_json(arm_dir / "summary.json", report)
    return report


def run(args):
    # Must be set before importing either torch/transformers or the Hub.
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    raw = args.manifest.read_bytes()
    rows = read_manifest(raw, pilot=args.pilot)
    if args.limit is not None:
        rows = rows[:args.limit]
    if maker_prompt(WARMUP, args.arm) in {maker_prompt(row, args.arm) for row in rows}:
        raise ValueError("The fixed warmup prompt must be unrelated to the evaluated requests")
    plan = request_plan(rows)
    keys = {(row["id"], seed) for row, seed in plan}
    backend = backend_settings()
    student, child_env = student_settings(args)
    snapshot = pinned_snapshot(MOTION, args.hf_cache)
    args.out.mkdir(parents=True, exist_ok=True)
    with output_lock(args.out), Nvml() as gpu:
        common = {"schema": "kimodo.v1-generate.config.v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                  "split": rows[0]["split"], "pilot": args.pilot, "limit": args.limit, "steps": args.steps,
                  "planned_requests": len(plan), "seeds": list(SEEDS), "fps": FPS, "concurrency": 1,
                  "exports": not args.no_export, "post_processing": True, "constraints": [],
                  "motion_model": list(MOTION), "motion_config_sha256": sha256(snapshot / "config.yaml"),
                  "motion_precision": "official checkpoint with default Kimodo/PyTorch inference (no autocast)",
                  "motion_code": KIMODO_CODE, "runner_sha256": sha256(__file__),
                  "backend_engine_sha256": sha256(ROOT / "backend" / "engine.py"), "backend": backend,
                  "accelerator": gpu.identity, "poll_interval_ns": POLL_NS, "embedding_cache": False,
                  "warmup": WARMUP, "warmup_seed": SEEDS[0]}
        freeze_json(args.out / "config.json", common)
        if student:
            freeze_json(args.out / "student.json", student)
        arm_dir = args.out / args.arm
        arm_dir.mkdir(exist_ok=True)
        freeze_json(arm_dir / "config.json", {"arm": args.arm, "motion_snapshot": str(snapshot),
                    "encoder": student or {"dtype": "bfloat16", "models": {k:list(v) for k,v in TEACHER.items()},
                                            "offloading": "none", "wrapper": "Kimodo LLM2VecEncoder"}})
        completed = completed_requests(arm_dir / "requests.jsonl")
        if completed.keys() - keys:
            raise ValueError("Journal contains unexpected request IDs or seeds")
        for rec in completed.values():
            if rec["status"] == "ok" and not args.no_export:
                directory = arm_dir / f"{rec['id']}-{rec['seed']}"
                if any(not (directory / name).is_file() or not (directory / name).stat().st_size for name in ARTIFACTS):
                    raise RuntimeError(f"Terminal request has missing exports; will not replace: {directory}")
        pending = [(row, seed) for row, seed in plan if (row["id"], seed) not in completed]
        if not pending:
            report = summarize(arm_dir, plan, None, args.pilot)
            return report

        session = arm_dir / "sessions" / uuid.uuid4().hex
        (session / "staged").mkdir(parents=True)
        session_info = {"session_id": session.name, "started_utc": utc_now(), "pid": os.getpid(),
                        "argv": [sys.executable, *sys.argv], "cwd": str(Path.cwd()), "pending_at_start": len(pending),
                        "motion_snapshot": str(snapshot), "processes_before": inventory(), "status": "running",
                        "gpu_exclusivity": "Not enforced. Close other GPU workloads; WDDM desktop/graphics and other GPU users count. Process snapshots may not identify all GPU usage.",
                        "nvml_api": "nvmlDeviceGetMemoryInfo (v1), device allocated + reserved framebuffer bytes; no idle subtraction"}
        write_json(session / "session.json", session_info)
        trace = MemoryTrace(session / "nvml.jsonl", gpu.used_bytes)
        try:
            with ExitStack() as resources, trace:
                import torch

                if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
                    raise RuntimeError("Exactly one visible CUDA GPU with native BF16 is required")
                if "RTX 4090" not in torch.cuda.get_device_name(0):
                    raise RuntimeError("CUDA:0 does not match the physical RTX 4090")
                encoder = (resources.enter_context(StudentProvider(args.student_python,
                    Path(__file__).with_name("v1_student.py"), args.checkpoint, session / "student.stderr.log",
                    child_env, args.student_timeout)) if student else load_teacher(args.hf_cache, session))
                if student:
                    session_info["student_pid"] = encoder.proc.pid
                model = load_pinned_motion(snapshot, encoder)
                warmup = run_request(model, WARMUP, SEEDS[0], args, arm_dir, session, warmup=True)
                if warmup["status"] != "ok":
                    raise RuntimeError(f"Warmup failed before evaluation: {warmup['error']}")
                for row, seed in pending:
                    run_request(model, row, seed, args, arm_dir, session)
            session_info["status"] = "complete"
        except BaseException as exc:
            session_info.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "blocked",
                                error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            session_info.update(ended_utc=utc_now(), memory=trace.stats, processes_after=inventory())
            write_json(session / "session.json", session_info)
            report = summarize(arm_dir, plan, session.name, args.pilot)
        return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("ref", "en", "ko"))
    parser.add_argument("--manifest", required=True, type=Path, help="UTF-8 manifest JSONL")
    parser.add_argument("--out", required=True, type=Path, help="Shared run root; use a new root for each pilot/eval")
    parser.add_argument("--hf-cache", type=Path, help="Local Hugging Face hub directory for motion and official teacher")
    parser.add_argument("--student-python", type=Path, help="Python 3.13 executable (en/ko only)")
    parser.add_argument("--checkpoint", type=Path, help="Trained adapter safetensors (en/ko only)")
    parser.add_argument("--student-hf-cache", type=Path, help="Optional separate Hugging Face hub directory for en/ko")
    parser.add_argument("--student-timeout", type=float, default=300, help="Maximum seconds per child IPC response")
    parser.add_argument("--pilot", action="store_true", help="Mark diagnostics as non-v1 evidence")
    parser.add_argument("--limit", type=int, help="Pilot only: first N manifest rows, all 3 seeds")
    parser.add_argument("--steps", type=int, help="Pilot only: override 100 denoising steps")
    parser.add_argument("--no-export", action="store_true", help="Pilot only: bypass Maker exports")
    args = parser.parse_args(argv)
    if not args.pilot and (args.limit is not None or args.steps is not None or args.no_export):
        parser.error("--limit, --steps, and --no-export require --pilot")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.steps is None:
        args.steps = STEPS
    if not 1 <= args.steps <= STEPS:
        parser.error("--steps must be in 1..100")
    if not math.isfinite(args.student_timeout) or args.student_timeout <= 0:
        parser.error("--student-timeout must be finite and positive")
    if args.arm == "ref" and (args.student_python or args.checkpoint or args.student_hf_cache):
        parser.error("--student-python, --checkpoint and --student-hf-cache are for en/ko only")
    if args.arm != "ref" and not (args.student_python and args.checkpoint):
        parser.error("en/ko require --student-python and --checkpoint")
    for key in ("manifest", "out", "hf_cache", "student_python", "checkpoint", "student_hf_cache"):
        path = getattr(args, key)
        if path is not None:
            setattr(args, key, path.resolve())
    return args


def main():
    sys.path.insert(0, str(ROOT))
    args = parse_args()
    result = run(args)
    print(json_bytes(result).decode("utf-8"), end="", flush=True)
    return int(result["failed"] > 0 or result["completed"] != result["expected"])


if __name__ == "__main__":
    raise SystemExit(main())