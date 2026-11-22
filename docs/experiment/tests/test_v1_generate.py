import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import v1_generate as runner


def item(**overrides):
    return {
        "id": "raise-l", "group_id": "raise", "split": "dev", "kind": 1,
        "en": "  Raise the left hand  ", "ko": "왼손을 들어!",
        "seconds": 2.5, "unseen": True, "footing": True, "intent": {},
    } | overrides


def manifest(*items):
    return "\n".join(json.dumps(row) for row in items).encode("utf-8")


class ProtocolTests(unittest.TestCase):
    def test_manifest_order_same_seeds_and_maker_punctuation_in_each_arm(self):
        rows = runner.read_manifest(manifest(item()), pilot=True)
        self.assertEqual([(r["id"], seed) for r, seed in runner.request_plan(rows)],
                         [("raise-l", 17), ("raise-l", 43), ("raise-l", 97)])
        self.assertEqual(runner.maker_prompt(rows[0], "ref"), "Raise the left hand.")
        self.assertEqual(runner.maker_prompt(rows[0], "en"), "Raise the left hand.")
        self.assertEqual(runner.maker_prompt(rows[0], "ko"), "왼손을 들어!")

    def test_manifest_rejects_collision_and_unbalanced_nonpilot(self):
        with self.assertRaises(ValueError):
            runner.read_manifest(manifest(item(), item(id="Raise-L")), pilot=True)
        with self.assertRaises(ValueError):
            runner.read_manifest(manifest(item(id="../bad")), pilot=True)
        with self.assertRaises(ValueError):
            runner.read_manifest(manifest(item()), pilot=False)

    def test_nearest_rank_reports_ceil_milliseconds_without_interpolating(self):
        self.assertEqual(runner.nearest_rank_p95_ms(list(range(1, 21))), 1)
        self.assertEqual(runner.nearest_rank_p95_ms([1_000_001] * 19 + [99_000_000]), 2)
        self.assertIsNone(runner.nearest_rank_p95_ms([]))

    def test_student_uses_real_json_lines_child_and_preserves_raw_norms(self):
        script = (
            "import json,sys\n"
            "assert sys.argv[1:] == ['serve', '--checkpoint', 'demo']\n"
            "for line in sys.stdin.buffer:\n"
            "    texts = json.loads(line)['texts']\n"
            "    print('child log', file=sys.stderr, flush=True)\n"
            "    print(json.dumps({'vectors': [[len(t)] * 4096 for t in texts], "
            "'lengths': [1] * len(texts)}), flush=True)\n"
        )
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            entry = Path(d) / "student.py"
            entry.write_text(script, encoding="utf-8")
            err = Path(d) / "stderr"
            with runner.StudentProvider(sys.executable, entry, "demo", err, None, 5, device="cpu") as child:
                tensor, lengths = child(["안녕", "hello"])
                self.assertEqual(tuple(tensor.shape), (2, 1, 4096))
                self.assertEqual(str(tensor.dtype), "torch.float32")
                self.assertEqual(tensor[:, 0, 0].tolist(), [2.0, 5.0])
                self.assertEqual(tensor[:, 0, :].norm(dim=1).tolist(), [128.0, 320.0])
                self.assertEqual(lengths, [1, 1])
            self.assertIn("child log", err.read_text(encoding="utf-8"))

    def test_ipc_rejects_bad_vectors_and_reports_child_errors(self):
        for payload in ({"vectors": [[1] * 4096], "lengths": [True]},
                        {"vectors": [[float("nan")] * 4096], "lengths": [1]},
                        {"vectors": [["1"] * 4096], "lengths": [1]}):
            with self.assertRaises(ValueError):
                runner.decode_student_response((json.dumps(payload) + "\n").encode("utf-8"), 1)
        with self.assertRaisesRegex(RuntimeError, "CUDA failed"):
            runner.decode_student_response(b'{"error":"CUDA failed"}\n', 1)

    def test_frozen_settings_do_not_allow_resuming_with_a_different_manifest(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            path = Path(d) / "run.json"
            runner.freeze_json(path, {"manifest_sha256": "a"})
            runner.freeze_json(path, {"manifest_sha256": "a"})
            with self.assertRaisesRegex(ValueError, "settings"):
                runner.freeze_json(path, {"manifest_sha256": "b"})
            self.assertEqual(json.loads(path.read_text())["manifest_sha256"], "a")

    def test_completed_failure_is_never_replaced_but_incomplete_work_is_pending(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            path = Path(d) / "requests.jsonl"
            first = {"id": "a", "seed": 17, "session_id": "original"}
            second = {"id": "a", "seed": 43, "session_id": "original"}
            runner.append_jsonl(path, first | {"status": "started"})
            runner.append_jsonl(path, first | {"status": "failed", "latency_ns": 100})
            runner.append_jsonl(path, second | {"status": "started"})
            self.assertEqual(runner.completed_requests(path), {("a", 17): first | {"status": "failed", "latency_ns": 100}})

    def test_sampler_records_idle_and_samples_through_last_request(self):
        counter = iter([10, 31] + [25] * 10000)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            path = Path(d) / "nvml.jsonl"
            with runner.MemoryTrace(path, lambda: next(counter)) as sampler:
                self.assertGreaterEqual(sampler.stats["peak_used_bytes"], 31)
            samples = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(samples[0]["phase"], "idle")
            self.assertEqual(samples[-1]["phase"], "end")
            self.assertGreater(samples[-1]["monotonic_ns"], samples[0]["monotonic_ns"])

    def test_installed_loader_uses_pinned_config_without_selecting_teacher(self):
        loader = runner.import_module("kimodo.model.load_model")
        provider = object()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
            path = Path(d)
            (path / "config.yaml").write_text("checkpoint_probe: ${oc.select:checkpoint_dir}/model.safetensors\n")

            def instantiate(cfg, overrides):
                self.assertEqual(cfg["checkpoint_probe"], f"{path}/model.safetensors")
                self.assertIsNone(cfg["text_encoder"])
                self.assertEqual(overrides, {"device": "cuda:0"})
                model = SimpleNamespace(fps=30)
                model.eval = lambda: model
                return model

            with patch.object(loader, "instantiate_from_dict", side_effect=instantiate), patch.object(
                loader, "_select_text_encoder_conf", side_effect=AssertionError("Must not select teacher/API")
            ), patch.dict(os.environ, {"CHECKPOINT_DIR": "unrelated"}):
                self.assertIs(runner.load_pinned_motion(path, provider).text_encoder, provider)
                self.assertEqual(os.environ["CHECKPOINT_DIR"], "unrelated")

    def test_request_runs_maker_exporter_and_retains_failed_request(self):
        import numpy as np
        from kimodo.skeleton import SOMASkeleton77

        class DummyMotion:
            output_skeleton = SOMASkeleton77().cpu()
            fps = 30

            def __call__(self, texts, frames, **kwargs):
                self.texts, self.kwargs = texts, kwargs
                if self.broken:
                    raise RuntimeError("fixture generation failed")
                return {"posed_joints": np.tile(self.output_skeleton.neutral_joints.numpy(), (1, frames[0], 1, 1)),
                        "global_rot_mats": np.tile(np.eye(3, dtype=np.float32), (1, frames[0], 77, 1, 1))}

        args = SimpleNamespace(arm="ko", no_export=False, steps=100)
        model = DummyMotion()
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d, patch("kimodo.tools.seed_everything") as reseed:
            arm = Path(d) / "ko"
            session = arm / "sessions" / "one"
            (session / "staged").mkdir(parents=True)
            model.broken = False
            success = runner.run_request(model, item(), 17, args, arm, session)
            reseed.assert_called_once_with(17)
            self.assertEqual(model.texts, ["왼손을 들어!"])
            self.assertEqual(model.kwargs, {"num_denoising_steps": 100, "constraint_lst": [], "num_samples": 1,
                                           "multi_prompt": True, "post_processing": True, "return_numpy": True})
            self.assertEqual(success["status"], "ok", success.get("error"))
            clip = arm / "raise-l-17"
            with np.load(clip / "motion.npz", allow_pickle=False) as motion:
                self.assertEqual(motion["posed_joints"].shape, (75, 77, 3))
            self.assertEqual(json.loads((clip / "motion.json").read_bytes())["frameCount"], 75)
            self.assertIn("Frames: 75", (clip / "motion.bvh").read_text(encoding="utf-8"))
            model.broken = True
            failure = runner.run_request(model, item(), 43, args, arm, session)
            self.assertEqual((failure["status"], failure["phase"]), ("failed", "generation"))
            self.assertGreaterEqual(failure["latency_ns"], 0)
            self.assertEqual(len(runner.completed_requests(arm / "requests.jsonl")), 2)