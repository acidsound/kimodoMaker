"""Cache the official Kimodo LLM2Vec targets locally, without loading motion weights."""

import argparse
import hashlib
from importlib.metadata import distribution, version
import json
import os
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile


KIMODO_REVISION = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"
MODELS = {
    "llama": ("meta-llama/Meta-Llama-3-8B-Instruct", "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"),
    "mntp": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", "31474e395ada192e8ed1586db6be79fb3b70c9c0"),
    "supervised": ("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised", "baa8ebf04a1c2500e61288e7dad65e8ae42601a7"),
}


def train_rows(manifest_bytes):
    rows, seen = [], set()
    for number, line in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or row.get("split") != "train":
            raise ValueError(f"Line {number}: only train rows are allowed")
        if not all(isinstance(row.get(key), str) and row[key].strip() for key in ("id", "group_id", "en", "ko")):
            raise ValueError(f"Line {number}: missing id, group_id, en, or ko")
        if row["id"] in seen:
            raise ValueError(f"Line {number}: duplicate id")
        if any(len(row[k]) > 600 for k in ("en", "ko")):
            raise ValueError(f"Line {number}: review text over 600 characters before training")
        seen.add(row["id"])
        rows.append(row)
        if len(rows) > 65536:
            raise ValueError("This Colab pilot supports at most 65536 training pairs per run")
    if not rows:
        raise ValueError("Empty training manifest")
    return rows


def teacher_prompts(rows):
    from kimodo.sanitize import sanitize_texts

    prompts = []
    for row in rows:
        text = row["en"].strip()
        prompts.append(text if text[-1] in ".!?" else text + ".")
    return sanitize_texts(prompts)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_upload_zip(bundle):
    archive_path = bundle / "kimodo-train-input.zip"
    with ZipFile(archive_path, "x") as archive:
        for name in ("train.jsonl", "targets.npz", "teacher.json"):
            archive.write(bundle / name, name)
    return archive_path


def export_bundle(manifest, out, batch):
    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from kimodo.model import LLM2VecEncoder

    if out.exists():
        raise FileExistsError(out)
    manifest_bytes = manifest.read_bytes()
    rows = train_rows(manifest_bytes)
    prompts = teacher_prompts(rows)
    identity = json.loads(distribution("kimodo").read_text("direct_url.json") or "{}")
    if identity.get("vcs_info", {}).get("commit_id") != KIMODO_REVISION:
        raise RuntimeError("Install the pinned Kimodo revision from backend/requirements.txt")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Teacher cache requires a CUDA GPU with native BF16 (e.g. the RTX 4090)")
    if batch < 1:
        raise ValueError("batch must be positive")
    files = {
        "llama": ["config.json", "model.safetensors.index.json", "model-*.safetensors"],
        "mntp": ["config.json", "adapter_config.json", "adapter_model.safetensors",
                 "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "supervised": ["adapter_config.json", "adapter_model.safetensors"],
    }
    snapshots = {
        name: Path(snapshot_download(repo, revision=revision, allow_patterns=files[name],
                                     local_files_only=HF_HUB_OFFLINE))
        for name, (repo, revision) in MODELS.items()
    }
    # Pin the transitive Llama dependency without changing the official tokenizer,
    # config.json (_name_or_path drives chat formatting), or adapters in the HF cache.
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kimodo-teacher-", dir=out.parent) as tmp:
        tmp = Path(tmp)
        mntp = tmp / "mntp"
        mntp.mkdir()
        for file in snapshots["mntp"].iterdir():
            if file.suffix in (".json", ".safetensors") and file.is_file():
                shutil.copyfile(file, mntp / file.name)
        config_path = mntp / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["base_model_name_or_path"] != MODELS["llama"][0]:
            raise RuntimeError("Unexpected Llama model in mntp adapter config")
        config["base_model_name_or_path"] = str(snapshots["llama"].resolve())
        config["revision"] = None
        config_path.write_text(json.dumps(config), encoding="utf-8")
        os.environ.pop("TEXT_ENCODERS_DIR", None)
        os.environ["TEXT_ENCODER_DEVICE"] = "cuda:0"
        encoder = LLM2VecEncoder(str(mntp), str(snapshots["supervised"]), "bfloat16", 4096, device="cuda:0")
        if encoder.model.model.config._name_or_path != MODELS["llama"][0]:
            raise RuntimeError("LLM2Vec prompt formatting no longer matches the official Llama preset")
        data = np.empty((len(rows), 4096), dtype=np.float32)
        for start in range(0, len(rows), batch):
            end = min(start + batch, len(rows))
            vectors, lengths = encoder(prompts[start:end])  # wrapper preserves the internal batch_size=1
            if vectors.shape != (end - start, 1, 4096) or lengths != [1] * (end - start):
                raise RuntimeError("Unexpected Kimodo text encoder shape")
            data[start:end] = vectors[:, 0].cpu().float().numpy()
            print(f"teacher cache: {end}/{len(rows)}", flush=True)
        if not np.isfinite(data).all():
            raise RuntimeError("Non-finite teacher vectors")
        bundle = tmp / "bundle"
        bundle.mkdir()
        (bundle / "train.jsonl").write_bytes(manifest_bytes)
        target_file = bundle / "targets.npz"
        np.savez(target_file, ids=np.array([r["id"] for r in rows]), vectors=data)
        meta = {
            "schema": "kimodo.teacher.v1", "teacher": "kimodo-llm2vec", "dtype": "bfloat16",
            "kimodo_revision": KIMODO_REVISION, "gpu": torch.cuda.get_device_name(0),
            "models": {k: {"repo": repo, "revision": rev} for k, (repo, rev) in MODELS.items()},
            "preprocess": "Maker terminal punctuation + kimodo.sanitize.sanitize_texts + LLM2VecEncoder",
            "versions": {name: version(name) for name in ("torch", "transformers", "peft", "huggingface-hub")},
            "teacher_code_sha256": sha256_file(Path(__file__)),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "targets_sha256": sha256_file(target_file), "pairs": len(rows),
        }
        (bundle / "teacher.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        write_upload_zip(bundle)
        bundle.rename(out)
    print(f"Colab upload ZIP: {out.resolve() / 'kimodo-train-input.zip'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Frozen, authorized train-only UTF-8 JSONL")
    parser.add_argument("out", type=Path, help="New directory for the train-only Colab input bundle")
    parser.add_argument("--batch", type=int, default=8, help="Outer batch only; LLM2Vec always uses internal batch size 1")
    args = parser.parse_args()
    export_bundle(args.manifest, args.out, args.batch)


if __name__ == "__main__":
    main()
