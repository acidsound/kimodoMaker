import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
import torch
from safetensors.torch import load_file


EXPERIMENT = Path(__file__).resolve().parents[1]
NOTEBOOK = EXPERIMENT / "Train_EmbeddingGemma2_Adapter_T4.ipynb"


class ColabAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        cls.core = {"__name__": "notebook_test"}
        exec(compile(cls.cells["adapter-core"], str(NOTEBOOK), "exec"), cls.core)

    def test_fits_one_raw_teacher_for_both_languages_and_exports(self):
        generator = torch.Generator().manual_seed(17)
        x = torch.randn(2, 3, 768, generator=generator) * 0.1
        y = torch.randn(3, 4096, generator=generator) * 3
        result = self.core["fit_adapter"](x.numpy(), y.numpy(), device="cpu", epochs=7, batch_size=6, lr=0.02, seed=17)
        self.assertLess(result["metrics"]["en"]["mse"], result["before"]["en"]["mse"])
        self.assertLess(result["metrics"]["ko"]["mse"], result["before"]["ko"]["mse"])
        self.assertAlmostEqual(result["metrics"]["ko"]["target_norm_mean"], y.norm(dim=-1).mean().item(), places=3)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            meta = self.core["save_result"](out, result, {"mode": "smoke", "code_sha256": "a" * 64})
            loaded = load_file(str(out / "adapter.safetensors"))
            self.assertEqual(tuple(loaded["weight"].shape), (4096, 768))
            self.assertEqual(meta["checkpoint_sha256"], hashlib.sha256((out / "adapter.safetensors").read_bytes()).hexdigest())
            with ZipFile(out / "kimodo-result.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"adapter.safetensors", "run.json"})
                self.assertEqual(archive.read("adapter.safetensors"), (out / "adapter.safetensors").read_bytes())
                self.assertEqual(json.loads(archive.read("run.json"))["run_id"], meta["run_id"])
            with self.assertRaises(FileExistsError):
                self.core["save_result"](out, result, {})

    def test_training_bundle_reorders_ids_and_rejects_sealed_rows_or_corruption(self):
        rows = [{"id": key, "group_id": key, "split": "train", "en": "One " + key, "ko": "동작 " + key} for key in ("a", "b")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "train.jsonl"
            targets = root / "targets.npz"
            meta_path = root / "teacher.json"

            def fixture(current_rows):
                manifest.write_text("\n".join(map(json.dumps, current_rows)), encoding="utf-8")
                np.savez(targets, ids=np.array(["b", "a"]), vectors=np.stack([np.ones(4096), np.full(4096, 2)]).astype(np.float32))
                meta = {"schema": "kimodo.teacher.v1", "teacher": "kimodo-llm2vec", "pairs": len(current_rows),
                        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(), "targets_sha256": hashlib.sha256(targets.read_bytes()).hexdigest()}
                meta_path.write_text(json.dumps(meta), encoding="utf-8")

            fixture(rows)
            decoded, aligned, _ = self.core["load_training_bundle"](root)
            self.assertEqual([r["id"] for r in decoded], ["a", "b"])
            np.testing.assert_array_equal(aligned[:, 0], [2, 1])
            package = root / "input.zip"
            with ZipFile(package, "w") as archive:
                for file in (manifest, targets, meta_path):
                    archive.write(file, file.name)
            decoded_zip, aligned_zip, _ = self.core["load_training_input"](package)
            self.assertEqual(decoded_zip, decoded)
            np.testing.assert_array_equal(aligned_zip, aligned)
            with ZipFile(package, "a") as archive:
                archive.writestr("../extra", "unexpected")
            with self.assertRaisesRegex(ValueError, "exactly"):
                self.core["load_training_input"](package)
            targets.write_bytes(targets.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "targets.*SHA"):
                self.core["load_training_bundle"](root)
            fixture([rows[0], {**rows[1], "split": "test"}])
            with self.assertRaisesRegex(ValueError, "train"):
                self.core["load_training_bundle"](root)

    def test_default_train_fails_on_missing_input_before_runtime_setup(self):
        state = dict(self.core)
        exec(self.cells["config"], state)
        self.assertEqual(state["MODE"], "train")
        exec(self.cells["run"].rsplit("\nmain()", 1)[0], state)
        with tempfile.TemporaryDirectory() as tmp:
            state["INPUT_PATH"] = Path(tmp) / "missing.zip"
            state["OUT_DIR"] = Path(tmp) / "out"
            state["runtime"] = lambda: self.fail("Runtime setup must not run before input verification")
            with self.assertRaisesRegex(FileNotFoundError, "missing.zip"):
                state["main"]()
            self.assertFalse(state["OUT_DIR"].exists())

    def test_core_cells_compile_without_notebook_magics(self):
        for cell_id, source in self.cells.items():
            with self.subTest(cell=cell_id):
                compile(source, cell_id, "exec")


if __name__ == "__main__":
    unittest.main()
