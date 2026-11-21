import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from server import FORMATS, TakeStore, validate_options


class BlockingEngine:
    def __init__(self):
        self.release = threading.Event()

    def generate(self, options, directory, report):
        report("generating")
        if not self.release.wait(2):
            raise TimeoutError("Test did not release generation")
        for extension in FORMATS:
            (directory / f"motion.{extension}").write_text("fixture", encoding="utf-8")
        return {"fps": 30, "frame_count": 120}


class TakeStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory_context = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory_context.cleanup)
        self.path = Path(self.directory_context.name)
        self.engine = BlockingEngine()
        self.options = validate_options({
            "prompt": " Walk and stop. ", "duration": 4, "seed": 0,
            "constraints": [{"type": "root2d"}],
        })

    def test_serializes_jobs_and_recovers_completed_history(self):
        store = TakeStore(self.path, self.engine)
        take = store.start(self.options)
        self.assertIsNone(store.start(self.options))
        self.engine.release.set()
        for _ in range(100):
            if store.get(take["id"])["status"] == "ready":
                break
            time.sleep(0.01)
        reloaded = TakeStore(self.path, self.engine)
        saved = reloaded.get(take["id"])
        self.assertEqual(saved["status"], "ready")
        self.assertEqual(saved["frame_count"], 120)
        self.assertEqual(saved["prompt"], "Walk and stop.")
        self.assertEqual(saved["seed"], 0)
        directory = self.path / take["id"]
        self.assertEqual(json.loads((directory / "constraints.json").read_text()), self.options["constraints"])

    def test_interrupted_jobs_do_not_block_generation_after_restart(self):
        directory = self.path / "abandoned-take"
        directory.mkdir()
        (directory / "take.json").write_text(json.dumps({
            "id": directory.name, "status": "generating", "created": "2026-11-21",
        }))
        store = TakeStore(self.path, self.engine)
        self.assertEqual(store.get(directory.name)["status"], "failed")

    def test_rejects_invalid_inputs(self):
        for overrides in ({"seed": True}, {"duration": float("nan")}, {"constraints": {}}, {"prompt": "  "}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    validate_options(dict(self.options, **overrides))


if __name__ == "__main__":
    unittest.main()
