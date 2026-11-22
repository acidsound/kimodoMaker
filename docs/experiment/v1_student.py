"""Train and serve the formal Kimodo v1 EmbeddingGemma 2 text adapter."""

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from zipfile import ZipFile

# Keep stdout a protocol-only FD, including for native-library output. Set this
# before importing torch or anything that can write directly to fd 1.
PROTOCOL = None
if __name__ == "__main__" and sys.argv[1:2] == ["serve"] and not {"-h", "--help"}.intersection(sys.argv):
    sys.stdout.flush()
    PROTOCOL = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

import numpy as np
import torch

INPUT_DIM = 768
OUTPUT_DIM = 4096
MLP_HIDDEN_DIM = 1024
ARCHITECTURES = ("linear", "mlp")
LANGUAGES = ("en", "ko")
SEED = 17
MODEL_ID = "google/embeddinggemma-2"
MODEL_REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"
PROMPT_NAME = "SentenceSimilarity"
PROMPT_PREFIX = "task: sentence similarity | query: "
MAX_TOKENS = 512
ENCODER_BATCH = 1  # No BF16 padding/batch geometry changes between training and serving.
ADAPTER_SCHEMA = "kimodo.v1.student-adapter"
BUNDLE_MIB_LIMITS = {"train.jsonl": 256, "targets.npz": 1200, "teacher.json": 1}
PIPELINE = {
    "student": MODEL_ID, "student_revision": MODEL_REVISION,
    "student_parameter_dtype": "bfloat16", "attention": "eager", "vision": False, "audio": False,
    "feature_dim": INPUT_DIM, "feature_storage_dtype": "float32 (BF16 values widened without re-normalizing)",
    "preprocess": "Maker terminal punctuation and pinned Kimodo sanitize_text for both EN and KO; no Unicode normalization",
    "prompt_name": PROMPT_NAME, "prompt_prefix": PROMPT_PREFIX, "max_tokens_including_prefix": MAX_TOKENS,
    "pooling": "official projected tokens; mean with prompt included; checkpoint L2 plus encode L2 (both BF16)",
    "encoder_batch_size": ENCODER_BATCH, "blank": "whitespace-only inputs bypass encoder and return exact zeros",
    "languages": list(LANGUAGES),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def load_training_bundle(path):
    """Validate teacher export; align raw vectors to the train manifest by ID."""
    path = Path(path)
    if path.is_file():
        with ZipFile(path) as archive, tempfile.TemporaryDirectory(prefix="kimodo-v1-teacher-") as temp:
            members = archive.infolist()
            if len(members) != len(BUNDLE_MIB_LIMITS) or {m.filename for m in members} != BUNDLE_MIB_LIMITS.keys():
                raise ValueError("Teacher ZIP must contain exactly train.jsonl, targets.npz and teacher.json at root")
            root = Path(temp)
            for member in members:
                if member.file_size > BUNDLE_MIB_LIMITS[member.filename] * (1 << 20):
                    raise ValueError(f"Teacher ZIP member too large: {member.filename}")
                with archive.open(member) as source, (root / member.filename).open("wb") as dest:
                    shutil.copyfileobj(source, dest)
            return load_training_bundle(root)
    if not path.is_dir():
        raise FileNotFoundError(path)
    manifest, cache, provenance = (path / name for name in BUNDLE_MIB_LIMITS)
    for name, limit in BUNDLE_MIB_LIMITS.items():
        if (path / name).stat().st_size > limit * (1 << 20):
            raise ValueError(f"Teacher bundle member too large: {name}")
    meta = json.loads(provenance.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("schema") != "kimodo.teacher.v1" or meta.get("teacher") != "kimodo-llm2vec":
        raise ValueError("Expected a kimodo.teacher.v1 LLM2Vec provenance file")
    for file, key in ((manifest, "manifest_sha256"), (cache, "targets_sha256")):
        if sha256(file) != meta.get(key):
            raise ValueError(f"{file.name} SHA-256 does not match teacher.json")
    rows, seen = [], set()
    with manifest.open(encoding="utf-8") as stream:
        for line, data in enumerate(stream, 1):
            row = json.loads(data)
            if not isinstance(row, dict) or row.get("split") != "train":
                raise ValueError(f"Line {line}: only train rows are allowed")
            if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("id", "group_id", *LANGUAGES)):
                raise ValueError(f"Line {line}: missing id, group_id, en or ko")
            if any(len(row[k]) > 600 for k in LANGUAGES):
                raise ValueError(f"Line {line}: review text over 600 characters before training")
            if row["id"] in seen:
                raise ValueError(f"Line {line}: duplicate id")
            seen.add(row["id"])
            rows.append(row)
            if len(rows) > 65536:
                raise ValueError("Too many teacher pairs (maximum 65536)")
    if not rows or type(meta.get("pairs")) is not int or meta["pairs"] != len(rows):
        raise ValueError("Teacher pair count does not match a nonempty train.jsonl")
    with np.load(cache, allow_pickle=False) as data:
        if set(data.files) != {"ids", "vectors"} or len(data.files) != 2:
            raise ValueError("Expected exactly ids and vectors in targets.npz")
        ids, vectors = data["ids"], data["vectors"]
    if ids.shape != (len(rows),) or ids.dtype.kind != "U" or set(ids.tolist()) != seen:
        raise ValueError("Teacher ids must match unique train.jsonl ids")
    if vectors.shape != (len(rows), OUTPUT_DIM) or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Teacher vectors must be finite float32 [pairs,4096]")
    order = {key: i for i, key in enumerate(ids.tolist())}
    indices = [order[row["id"]] for row in rows]
    return rows, (vectors if indices == list(range(len(rows))) else vectors[indices]), meta


def new_adapter(architecture="linear"):
    def linear(a, b):
        return torch.nn.Linear(a, b, device="cpu", dtype=torch.float32)

    if architecture == "linear":
        return linear(INPUT_DIM, OUTPUT_DIM)
    if architecture == "mlp":
        return torch.nn.Sequential(linear(INPUT_DIM, MLP_HIDDEN_DIM), torch.nn.GELU(), linear(MLP_HIDDEN_DIM, OUTPUT_DIM))
    raise ValueError(f"Unknown adapter architecture: {architecture}")


def fit_adapter(features, targets, *, device, epochs, batch_size, lr, seed=SEED, architecture="linear"):
    """Fit FP32 on [EN, KO] features against one *raw* target per pair."""
    n = len(targets)
    if targets.shape != (n, OUTPUT_DIM) or features.shape != (2, n, INPUT_DIM) or not n:
        raise ValueError("Expected features [2,pairs,768] and targets [pairs,4096]")
    if any(a.dtype != np.float32 or not np.isfinite(a).all() for a in (features, targets)):
        raise ValueError("Features and targets must be finite float32 arrays")
    if epochs < 1 or batch_size < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive and finite")
    x = torch.from_numpy(features).flatten(0, 1)
    y = torch.from_numpy(targets)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = new_adapter(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0, fused=False, foreach=False)

    @torch.inference_mode()
    def diagnostics():
        report = {}
        for arm, offset in zip(LANGUAGES, (0, n)):
            totals = dict(mse=0.0, cosine=0.0, norm_mae=0.0, target_norm_mean=0.0, predicted_norm_mean=0.0)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                gold = y[start:end].to(device)
                pred = model(x[offset + start:offset + end].to(device))
                pn, gn = pred.norm(dim=-1), gold.norm(dim=-1)
                totals["mse"] += (pred - gold).square().mean(dim=-1).sum().item()
                totals["cosine"] += torch.nn.functional.cosine_similarity(pred, gold).sum().item()
                totals["norm_mae"] += (pn - gn).abs().sum().item()
                totals["target_norm_mean"] += gn.sum().item()
                totals["predicted_norm_mean"] += pn.sum().item()
            if not all(math.isfinite(v) for v in totals.values()):
                raise RuntimeError("Non-finite adapter diagnostics")
            report[arm] = {key: value / n for key, value in totals.items()}
        return report

    before = diagnostics()
    history = []
    for _ in range(epochs):
        total = 0.0
        for indices in torch.randperm(2 * n, generator=rng).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x[indices].to(device))
            loss = torch.nn.functional.mse_loss(prediction, y[indices % n].to(device))
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite adapter loss")
            loss.backward()
            optimizer.step()
            total += value * len(indices)
        history.append(total / (2 * n))
    return model.eval(), {"before": before, "after": diagnostics(), "epoch_mse": history}


def preprocess_text(text):
    # Maker (engine.py) adds terminal ASCII punctuation; mirror it before the
    # pinned Kimodo canonicalization so training sees the same strings as serving.
    text = text.strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    # Adapted from NVIDIA's kimodo/sanitize.py at the teacher's Kimodo revision.
    # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    # SPDX-License-Identifier: Apache-2.0
    text = " ".join(text.split())
    first = next((i for i, c in enumerate(text) if c.isalnum()), len(text) - 1)
    text = text[first:].capitalize()
    last = next((i for i in range(len(text) - 1, -1, -1) if text[i].isalnum() or text[i] in ".!?\"])'"), 0)
    text = text[:last + 1]
    if text[-1] not in ".!?":
        text += "."
    for mark in ".!?":
        text = (mark + " ").join(piece[:1].capitalize() + piece[1:] for piece in map(str.strip, text.split(mark))).strip()
    return text


def save_adapter(path, model, config):
    from safetensors.torch import save_file

    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path, metadata={
        "schema": ADAPTER_SCHEMA, "pipeline_sha256": fingerprint(PIPELINE), "config": canonical_json(config),
        "architecture": config.get("architecture", "linear"), "dtype": "float32",
    })


def load_adapter(path):
    """Load only the candidate’s own data-only, shape-checked safetensors adapter."""
    from safetensors import safe_open

    path = Path(path)
    if path.suffix != ".safetensors" or path.stat().st_size > (24 << 20):
        raise ValueError("Expected a v1 adapter .safetensors file (maximum 24 MiB)")
    with safe_open(path, framework="pt", device="cpu") as data:
        meta = data.metadata() or {}
        architecture = meta.get("architecture")
        if ((meta.get("schema"), meta.get("pipeline_sha256"), meta.get("dtype")) != (ADAPTER_SCHEMA, fingerprint(PIPELINE), "float32")
                or architecture not in ARCHITECTURES or not isinstance(meta.get("config"), str)):
            raise ValueError("Expected v1 adapter provenance for the pinned student")
        config = json.loads(meta["config"])
        if not isinstance(config, dict) or config.get("architecture", "linear") != architecture:
            raise ValueError("Invalid adapter run provenance")
        model = new_adapter(architecture)
        state = model.state_dict()
        if set(data.keys()) != state.keys():
            raise ValueError("Unexpected adapter tensor names")
        weights = {key: data.get_tensor(key) for key in state}
    for key, value in weights.items():
        if value.shape != state[key].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid FP32 adapter tensor: {key}")
    model.load_state_dict(weights, strict=True)
    return model.eval().requires_grad_(False), config


@torch.inference_mode()
def predict(texts, encode, model):
    prompts = [preprocess_text(text) for text in texts]
    active = [i for i, text in enumerate(prompts) if text]
    vectors = torch.zeros((len(prompts), OUTPUT_DIM), dtype=torch.float32)
    if active:
        features = encode([prompts[i] for i in active])
        if features.shape != (len(active), INPUT_DIM) or features.dtype != torch.float32:
            raise RuntimeError("Unexpected serving feature shape or dtype")
        # Serving is also singleton for the FP32 adapter GEMM (list size cannot
        # introduce a different GEMM kernel for an otherwise identical prompt).
        projected = torch.cat([model(row) for row in features.split(1)]).cpu()
        if not torch.isfinite(projected).all():
            raise RuntimeError("Non-finite candidate output")
        vectors[active] = projected
    return {"vectors": vectors.tolist(), "lengths": [1] * len(prompts)}


def serve_jsonl(requests, responses, encode, model):
    for line in requests:
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"texts"} or not isinstance(request["texts"], list):
                raise ValueError('expected {"texts": [<strings>]}')
            if not all(isinstance(text, str) for text in request["texts"]):
                raise ValueError("texts must contain only strings")
            response = predict(request["texts"], encode, model)
        except ValueError as exc:
            response = {"error": str(exc)}
        print(json.dumps(response, ensure_ascii=True, separators=(",", ":"), allow_nan=False), file=responses, flush=True)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def save_feature_cache(path, ids, features, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v1-features-", dir=path.parent) as temp:
        cache = Path(temp) / "features.npz"
        meta = cache.with_suffix(".json")
        np.savez(cache, ids=ids, features=features)
        write_json(meta, {"config": config, "config_sha256": fingerprint(config), "sha256": sha256(cache)})
        # Fail instead of clobbering another run's cache. A missing sidecar is
        # rejected rather than guessed on the next attempt after interruption.
        if path.exists() or path.with_suffix(".json").exists():
            raise FileExistsError(f"Feature cache already exists: {path}")
        cache.rename(path)
        meta.rename(path.with_suffix(".json"))


def load_feature_cache(path, ids, config):
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or meta.get("config") != config:
        raise ValueError("Feature cache provenance does not match this run")
    if sha256(path) != meta.get("sha256"):
        raise ValueError("Feature cache SHA-256 mismatch")
    if meta.get("config_sha256") != fingerprint(config):
        raise ValueError("Feature cache provenance fingerprint mismatch")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"ids", "features"} or len(data.files) != 2:
            raise ValueError("Invalid feature cache fields")
        archived_ids, features = data["ids"], data["features"]
    if archived_ids.dtype.kind != "U" or not np.array_equal(archived_ids, ids):
        raise ValueError("Feature cache ID order does not match train.jsonl")
    if features.dtype != np.float32 or features.shape != (2, len(ids), INPUT_DIM) or not np.isfinite(features).all():
        raise ValueError("Invalid finite float32 feature cache shape or dtype")
    return features


DEVICE = "cuda:0"
VERSIONS = ("torch", "transformers", "sentence-transformers", "huggingface-hub", "numpy", "safetensors", "tokenizers")


def runtime_environment():
    packages = {p: version(p) for p in VERSIONS}
    if packages["sentence-transformers"] != "6.3.0" or packages["transformers"] != "5.24.0":
        raise RuntimeError("Use the separate v1 environment: sentence-transformers==6.3.0 and transformers==5.24.0")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("v1 requires CUDA with native BF16; no CPU/FP16/FP32 encoder fallback")
    torch.set_float32_matmul_precision("highest")  # FP32 adapter, no implicit TF32 shortcuts.
    return {"packages": packages, "python": platform.python_version(), "executable": sys.executable,
            "platform": platform.platform(), "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)), "native_bf16": True,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def pinned_snapshot():
    from huggingface_hub import snapshot_download

    # The verified cache contains these runtime files, not the model card or Git attributes.
    root = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                                  allow_patterns=["*.json", "model.safetensors", "tokenizer.model", "chat_template.jinja"]))
    if root.name != MODEL_REVISION or not (root / "model.safetensors").is_file():
        raise RuntimeError("Expected the pinned local official EmbeddingGemma 2 safetensors snapshot")
    return root


class FrozenEncoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.base.modules.normalize import Normalize
        from sentence_transformers.sentence_transformer.modules.pooling import Pooling

        root = pinned_snapshot()
        self.model = SentenceTransformer(
            str(root), device=DEVICE, trust_remote_code=False, local_files_only=True,
            config_kwargs={"vision_config": None, "audio_config": None},
            model_kwargs={"dtype": torch.bfloat16, "attn_implementation": "eager", "use_safetensors": True},
        ).eval().requires_grad_(False)
        backbone, pooling = self.model[0].auto_model, self.model[1]
        if backbone.vision_tower is not None or backbone.audio_tower is not None:
            raise RuntimeError("Expected text-only EmbeddingGemma 2 (both towers disabled)")
        parameters = [p for p in self.model.parameters() if p.is_floating_point()]
        if not parameters or any(p.dtype != torch.bfloat16 or str(p.device) != DEVICE or p.requires_grad for p in parameters):
            raise RuntimeError("Expected actual frozen CUDA BF16 encoder parameters")
        pooling_config = pooling.get_config_dict()
        if (len(self.model) != 3 or not isinstance(pooling, Pooling) or pooling_config.get("pooling_mode") != "mean"
                or pooling_config.get("include_prompt") is not True or not isinstance(self.model[2], Normalize)
                or self.model.get_embedding_dimension() != INPUT_DIM or self.model.prompts.get(PROMPT_NAME) != PROMPT_PREFIX):
            raise RuntimeError("Unexpected pinned model projection, mean pooling, normalization or prompt")
        self.model.max_seq_length = MAX_TOKENS

    @torch.inference_mode()
    def __call__(self, prompts):
        token_ids = self.model.tokenizer([PROMPT_PREFIX + p for p in prompts], truncation=False)["input_ids"]
        if any(len(ids) > MAX_TOKENS for ids in token_ids):
            raise ValueError(f"Prompt exceeds {MAX_TOKENS} tokens including the shared EN/KO prefix; refusing to truncate")
        features = self.model.encode(prompts, prompt_name=PROMPT_NAME, batch_size=ENCODER_BATCH,
                                     normalize_embeddings=True, convert_to_tensor=True, convert_to_numpy=False,
                                     show_progress_bar=False)
        if (features.shape != (len(prompts), INPUT_DIM) or features.dtype != torch.bfloat16
                or str(features.device) != DEVICE or not torch.isfinite(features).all()):
            raise RuntimeError("Unexpected, non-finite or non-BF16 student feature output")
        return features.float()


def encode_train_rows(rows):
    encoder = FrozenEncoder()
    features = np.empty((2, len(rows), INPUT_DIM), dtype=np.float32)
    for arm, language in enumerate(LANGUAGES):
        for i in range(0, len(rows), 128):
            batch = rows[i:i + 128]
            prompts = [preprocess_text(r[language]) for r in batch]
            try:
                features[arm, i:i + len(batch)] = encoder(prompts).cpu().numpy()
            except ValueError as exc:
                raise ValueError(f"{language} train rows {i + 1}-{i + len(batch)}: {exc}") from exc
            print(f"v1 {language} cached: {i + len(batch)}/{len(rows)}", file=sys.stderr, flush=True)
    return features


def feature_diagnostics(features):
    # Cheap training-only retrieval proxy; cap pairwise work, never open dev/test.
    en, ko = features[:, :512]
    en = en / np.maximum(np.linalg.norm(en, axis=1, keepdims=True), 1e-12)
    ko = ko / np.maximum(np.linalg.norm(ko, axis=1, keepdims=True), 1e-12)
    scores = en @ ko.T
    ids = np.arange(len(en))
    return {"pairs_first_in_manifest": len(en), "paired_cosine_mean": float(np.diag(scores).mean()),
            "en_to_ko_exact_row_top1": float((scores.argmax(axis=1) == ids).mean()),
            "ko_to_en_exact_row_top1": float((scores.argmax(axis=0) == ids).mean())}


TEACHER_KIMODO_REVISION = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
TEACHER_MODELS = {
    "llama": ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
    "mntp": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    "supervised": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
}


def train(args):
    if args.epochs < 1 or args.batch < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("--epochs, --batch and --lr must be positive and finite")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"--out must be a new directory: {out}")
    rows, targets, teacher = load_training_bundle(args.bundle)
    if (teacher.get("kimodo_revision") != TEACHER_KIMODO_REVISION or teacher.get("dtype") != "bfloat16"
            or teacher.get("models") != {k: {"repo": model, "revision": rev} for k, (model, rev) in TEACHER_MODELS.items()}):
        raise ValueError("Formal v1 requires the pinned official Kimodo BF16 LLM2Vec teacher export")
    env = runtime_environment()
    code_hash = sha256(__file__)
    ids = np.array([r["id"] for r in rows])
    cache_config = {"schema": "kimodo.v1.student-features", "manifest_sha256": teacher["manifest_sha256"],
                    "pipeline": PIPELINE, "code_sha256": code_hash, "runtime": env}
    cache_id = fingerprint(cache_config)
    cache = args.feature_cache or out.parent / f"v1-features-{cache_id}.npz"
    cache = cache.resolve()
    if cache.suffix != ".npz" or cache.is_relative_to(out):
        raise ValueError("--feature-cache must name an .npz outside the new output directory")
    reused = cache.exists() or cache.with_suffix(".json").exists()
    encoder_peak = None
    if not reused:
        torch.cuda.reset_peak_memory_stats(0)
        features = encode_train_rows(rows)
        encoder_peak = torch.cuda.max_memory_allocated(0)
        gc.collect()
        torch.cuda.empty_cache()
        save_feature_cache(cache, ids, features, cache_config)
        del features
    features = load_feature_cache(cache, ids, cache_config)
    print(f"v1 feature cache {'reused' if reused else 'written'}: {cache}", file=sys.stderr, flush=True)
    torch.cuda.reset_peak_memory_stats(0)
    model, diagnostics = fit_adapter(features, targets, device=DEVICE, epochs=args.epochs, batch_size=args.batch,
                                     lr=args.lr, architecture=args.architecture)
    config = {
        "schema": "kimodo.v1.student-run", "student": MODEL_ID, "student_revision": MODEL_REVISION, "pipeline": PIPELINE,
        "pipeline_sha256": fingerprint(PIPELINE), "code_sha256": code_hash, "runtime": env, "runtime_sha256": fingerprint(env),
        "architecture": args.architecture, "shape": [INPUT_DIM, OUTPUT_DIM], "adapter_dtype": "float32",
        "hidden_dim": MLP_HIDDEN_DIM if args.architecture == "mlp" else None,
        "activation": "GELU" if args.architecture == "mlp" else None,
        "target": "shared raw LLM2Vec float32 by pair ID; no target or output normalization", "teacher": teacher,
        "features_sha256": sha256(cache), "feature_config_sha256": cache_id, "pairs": len(rows),
        "optimizer": "AdamW(weight_decay=0, foreach=False, fused=False)", "loss": "mse",
        "epochs": args.epochs, "lr": args.lr, "adapter_batch_size": args.batch, "seed": SEED,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v1-student-", dir=out.parent) as temp:
        pending = Path(temp) / "candidate"
        pending.mkdir()
        checkpoint = pending / "adapter.safetensors"
        save_adapter(checkpoint, model, config)
        identity = {**config, "checkpoint_sha256": sha256(checkpoint)}
        candidate_id = fingerprint(identity)
        report = {
            **identity, "candidate_id": candidate_id, "run_id": candidate_id,
            "feature_cache": str(cache), "feature_cache_reused": reused,
            "encoder_peak_torch_allocated_bytes": encoder_peak,
            "adapter_peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "train_diagnostics_before": diagnostics["before"], "train_diagnostics_after": diagnostics["after"],
            "epoch_mse": diagnostics["epoch_mse"], "train_feature_diagnostics": feature_diagnostics(features),
        }
        write_json(pending / "run.json", report)
        if out.exists():
            raise FileExistsError(out)
        pending.rename(out)
    print(json.dumps({"candidate_id": candidate_id, "checkpoint_sha256": identity["checkpoint_sha256"],
                      "checkpoint": str(out / checkpoint.name), "run": str(out / "run.json"),
                      "train_diagnostics_after": diagnostics["after"]}, ensure_ascii=True, allow_nan=False), flush=True)


def serve(checkpoint, inputs, responses):
    model, config = load_adapter(checkpoint)
    if config.get("schema") != "kimodo.v1.student-run" or config.get("code_sha256") != sha256(__file__):
        raise ValueError("Checkpoint was not built by this exact v1 student code")
    env = runtime_environment()
    training_env = config.get("runtime")
    if not isinstance(training_env, dict) or any(
        training_env.get(key) != env[key] for key in ("packages", "cuda_build", "float32_matmul_precision")
    ):
        raise RuntimeError("Serving must match the training software and FP32 matmul configuration")
    encoder = FrozenEncoder()
    model = model.to(DEVICE)
    print(canonical_json({"event": "v1-student-ready", "runtime": env, "pipeline": PIPELINE,
                          "checkpoint_sha256": sha256(checkpoint)}), file=sys.stderr, flush=True)
    serve_jsonl(inputs, responses, encoder, model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Fit a formal v1 adapter from a train-only teacher bundle")
    training.add_argument("--bundle", required=True, type=Path, help="Teacher directory or its kimodo-train-input.zip")
    training.add_argument("--out", required=True, type=Path, help="New directory for adapter.safetensors and run.json")
    training.add_argument("--feature-cache", type=Path, help="Optional reusable .npz path, paired with a .json sidecar")
    training.add_argument("--epochs", type=int, default=12)
    training.add_argument("--lr", type=float, default=0.002)
    training.add_argument("--batch", type=int, default=64, help="Adapter training batch only; encoder always batch 1")
    training.add_argument("--architecture", choices=ARCHITECTURES, default="linear",
                          help="First candidate is linear; fixed GELU mlp is opt-in for a later dev round")
    serving = commands.add_parser("serve", help="Serve newline-delimited UTF-8 JSON on stdin/stdout")
    serving.add_argument("--checkpoint", required=True, type=Path, help="Trusted adapter.safetensors from v1 train")
    args = parser.parse_args(argv)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    if args.command == "train":
        train(args)
    else:
        sys.stdin.reconfigure(encoding="utf-8")
        serve(args.checkpoint, sys.stdin, PROTOCOL or sys.stdout)


if __name__ == "__main__":
    main()
