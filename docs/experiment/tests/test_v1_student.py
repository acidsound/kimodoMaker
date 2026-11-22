"""CPU-only tests; never load a model snapshot or a CUDA device."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "v1_student.py"


def student():
    spec = importlib.util.spec_from_file_location("v1_student", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudentFitTest(unittest.TestCase):
    def test_linear_fit_learns_shared_raw_en_ko_target_scale(self):
        v1 = student()
        features = np.zeros((2, 3, 768), dtype=np.float32)
        for i in range(3):
            features[0, i, i] = 1
            features[1, i, i + 3] = 1
        targets = np.repeat(np.array([[0.4], [-0.6], [0.9]], dtype=np.float32), 4096, axis=1)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            model, stats = v1.fit_adapter(features, targets, device="cpu", epochs=35, batch_size=6, lr=0.05)
        finally:
            torch.set_num_threads(old_threads)
        self.assertIsInstance(model, torch.nn.Linear)
        self.assertEqual(model.weight.shape, (4096, 768))
        self.assertEqual(model.weight.dtype, torch.float32)
        for lang in ("en", "ko"):
            self.assertLess(stats["after"][lang]["mse"], stats["before"][lang]["mse"] * 0.03)
            self.assertGreater(stats["after"][lang]["target_norm_mean"], 30)
            self.assertLess(stats["after"][lang]["norm_mae"], 5)


def make_bundle(root, *, ids=("b", "a"), split="train", dtype=np.float32, nonfinite=False):
    rows = [{"id": k, "group_id": "pair-group", "split": split, "en": "Turn left.", "ko": "왼쪽으로 돈다."}
            for k in ("a", "b")]
    manifest = root / "train.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    targets = root / "targets.npz"
    values = np.array([3 if k == "a" else 7 for k in ids], dtype=dtype)
    if nonfinite:
        values[0] = np.nan
    np.savez(targets, ids=np.array(ids), vectors=np.repeat(values[:, None], 4096, axis=1))
    (root / "teacher.json").write_text(json.dumps({
        "schema": "kimodo.teacher.v1", "teacher": "kimodo-llm2vec", "pairs": len(rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(targets.read_bytes()).hexdigest(),
    }), encoding="utf-8")


class StudentBundleTest(unittest.TestCase):
    def test_teacher_ids_align_raw_float32_vectors_for_directory_and_zip(self):
        v1 = student()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            archive = root / "bundle.zip"
            with ZipFile(archive, "w") as output:
                for name in ("train.jsonl", "teacher.json", "targets.npz"):
                    output.write(root / name, name)
            for source in (root, archive):
                rows, targets, meta = v1.load_training_bundle(source)
                self.assertEqual([r["id"] for r in rows], ["a", "b"])
                self.assertEqual(targets.dtype, np.float32)
                np.testing.assert_array_equal(targets[:, 0], [3, 7])
                self.assertEqual(meta["teacher"], "kimodo-llm2vec")

    def test_teacher_bundle_rejects_invalid_split_ids_dtype_and_nonfinite(self):
        v1 = student()
        for options, error in (({"split": "test"}, "train"), ({"ids": ("a", "a")}, "ids"),
                               ({"ids": ("a", "x")}, "ids"), ({"dtype": np.float64}, "float32"),
                               ({"nonfinite": True}, "finite")):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                make_bundle(Path(tmp), **options)
                with self.assertRaisesRegex(ValueError, error):
                    v1.load_training_bundle(Path(tmp))

    def test_teacher_bundle_rejects_manifest_hash_mismatch(self):
        v1 = student()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_bundle(root)
            with (root / "train.jsonl").open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                v1.load_training_bundle(root)


class StudentInferenceTest(unittest.TestCase):
    def test_prompt_preprocessing_matches_app_for_both_languages(self):
        v1 = student()
        for text, canonical in (("  -TURN    left  ", "Turn left."), ("왼쪽으로 \t 돈다!", "왼쪽으로 돈다!"),
                                ("  jump?then turn  ", "Jump? Then turn."), ("\t ", ""), ("---", ".")):
            with self.subTest(text=text):
                self.assertEqual(v1.preprocess_text(text), canonical)
                self.assertEqual(v1.preprocess_text(canonical), canonical)

    def test_jsonl_protocol_preserves_order_blanks_and_length(self):
        v1 = student()
        model = v1.new_adapter()
        with torch.no_grad():
            model.weight.zero_()
            model.weight[:, 0].fill_(1)
            model.bias.fill_(5)
        responses = io.StringIO()
        prompts = {"Turn left.": 1, "왼쪽으로 돈다!": 2}

        def encode(texts):
            features = torch.zeros(len(texts), 768)
            for i, text in enumerate(texts):
                features[i, 0] = prompts[text]
            return features

        requests = io.StringIO('{"texts":["turn left", "", "왼쪽으로 돈다!", " "]}\n{"texts":[]}\n{"texts":[3]}\n')
        v1.serve_jsonl(requests, responses, encode, model)
        mixed, empty, invalid = map(json.loads, responses.getvalue().splitlines())
        self.assertEqual(mixed["lengths"], [1, 1, 1, 1])
        self.assertEqual(mixed["vectors"], [[6.] * 4096, [0.] * 4096, [7.] * 4096, [0.] * 4096])
        self.assertEqual(empty, {"vectors": [], "lengths": []})
        self.assertIn("error", invalid)

    def test_fixed_mlp_is_opt_in_and_roundtrips_only_its_own_shape(self):
        v1 = student()
        model = v1.new_adapter("mlp")
        self.assertIsInstance(model[1], torch.nn.GELU)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            v1.save_adapter(path, model, {"architecture": "mlp"})
            restored, _ = v1.load_adapter(path)
            probe = torch.linspace(-1, 1, 768).view(1, -1)
            with torch.no_grad():
                torch.testing.assert_close(restored(probe), model(probe), atol=0, rtol=0)
            self.assertEqual(restored(probe).shape, (1, 4096))

    def test_checkpoint_roundtrip_validates_provenance_and_preserves_fp32_weights(self):
        v1 = student()
        model = v1.new_adapter()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            run_config = {"input": "unit-test"}
            v1.save_adapter(path, model, run_config)
            restored, restored_config = v1.load_adapter(path)
            self.assertEqual(restored_config, run_config)
            torch.testing.assert_close(restored.weight, model.weight, atol=0, rtol=0)
            torch.testing.assert_close(restored.bias, model.bias, atol=0, rtol=0)
            from safetensors.torch import save_file
            untagged = path.with_name("untagged.safetensors")
            save_file(model.state_dict(), untagged)
            with self.assertRaisesRegex(ValueError, "provenance"):
                v1.load_adapter(untagged)


class StudentFeatureCacheTest(unittest.TestCase):
    def test_cache_checks_encoder_fingerprint_and_literal_row_order(self):
        v1 = student()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.array(["a", "b"])
            features = np.zeros((2, 2, 768), dtype=np.float32)
            features[1, 0, 7] = 0.5
            config = {"encoder": "test-only"}
            path = root / "features.npz"
            v1.save_feature_cache(path, ids, features, config)
            np.testing.assert_array_equal(v1.load_feature_cache(path, ids, config), features)
            with self.assertRaisesRegex(ValueError, "order"):
                v1.load_feature_cache(path, ids[::-1], config)
            with self.assertRaisesRegex(ValueError, "provenance"):
                v1.load_feature_cache(path, ids, {"encoder": "different"})
            path.with_suffix(".json").write_text(json.dumps({"config": config, "sha256": "bad"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                v1.load_feature_cache(path, ids, config)


if __name__ == "__main__":
    unittest.main()
