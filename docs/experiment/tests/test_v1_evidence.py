import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import v1_evidence
import v1_freeze
import v1_score


def manifest_rows(split):
    rows = []
    for kind in range(1, 5):
        for i in range(16):
            ident = f"{split}-k{kind}-{i}"
            rows.append({"id": ident, "group_id": f"g-{split}-{kind}-{i // 2}", "split": split, "kind": kind,
                         "seconds": 3.0, "unseen": i == 0, "footing": True,
                         "intent": {"type": "hand", "side": "left"}, "en": f"en {ident}", "ko": f"ko {ident}"})
    return rows


def freeze_fixture(home):
    paths = {"train": [{"id": "t1", "group_id": "g-train", "split": "train", "en": "train en", "ko": "train ko"}]}
    for split in ("dev", "test"):
        paths[split] = manifest_rows(split)
    written = {}
    for name, rows in paths.items():
        path = home / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        written[name] = path
    lock = home / "lock.json"
    v1_freeze.freeze(written, {}, {"policy": "test"}, lock)
    return lock


def write_run(root, arm, rows, manifest_sha, checkpoint=None, motion_model=("nvidia/Kimodo-SOMA-RP-v1.2", "r")):
    run = Path(root)
    (run / arm).mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(
        {"manifest_sha256": manifest_sha, "steps": 100, "seeds": [17, 43, 97], "pilot": False, "exports": True,
         "concurrency": 1, "constraints": [], "motion_model": list(motion_model)}), encoding="utf-8")
    (run / arm / "summary.json").write_text(json.dumps(
        {"pilot": False, "completed": 192, "single_session_measurement_complete": True,
         "peak_used_mib": 1000, "p95_ms": 2000}), encoding="utf-8")
    (run / arm / "requests.jsonl").write_text(
        "".join(json.dumps({"id": row["id"], "seed": seed}) + "\n" for row in rows for seed in (17, 43, 97)),
        encoding="utf-8")
    for row in rows:
        for seed in (17, 43, 97):
            directory = run / arm / f"{row['id']}-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "motion.json").write_text("{}", encoding="utf-8")
    if checkpoint:
        (run / arm / "student.json").write_text(json.dumps({
            "checkpoint_sha256": checkpoint, "checkpoint_run_json_sha256": "run-json", "entry_sha256": "entry",
            "backbone": ["google/embeddinggemma-2", "rev"],
            "environment": {"python": [3, 13, 0]}, "cache_env": {}}), encoding="utf-8")
    return run


def report_dict(measurement, candidate, stage="Dev"):
    return {"stage": stage, "candidate": candidate, "measurement": measurement, "evidence_problems": []}


class MeasureTests(unittest.TestCase):
    def test_measure_records_evidence_problems_and_counts_all_clips(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            lock = freeze_fixture(home)
            rows = manifest_rows("dev")
            manifest = v1_freeze.verify(lock, "Dev")
            run = write_run(home / "run", "ref", rows, manifest["manifests"]["dev"]["sha256"])
            with mock.patch.object(v1_evidence, "clip_score", return_value=dict(hit=True, usable=True, reason="ok")):
                report = v1_evidence.measure(lock, run, "Dev", 1)
            self.assertFalse(report["measurement"]["evidence"])
            self.assertIn("no export evidence supplied", report["evidence_problems"])
            self.assertIn("en/ko checkpoints differ", report["evidence_problems"])
            self.assertEqual(report["measurement"]["hits"]["ref"], [48] * 4)
            self.assertEqual(report["measurement"]["usable"]["ref"], 192)
            self.assertEqual(len(report["clips"]), 192)  # only the ref arm exists in this run
            self.assertIn("en summary unreadable", "\n".join(report["evidence_problems"]))
            # Missing arms leave zero p95/peak: the measurement is invalid and fails closed.
            self.assertIsNone(report["dev_ranking"])

    def test_measure_reuses_ref_root_and_requires_protocol_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            lock = freeze_fixture(home)
            rows = manifest_rows("dev")
            manifest = v1_freeze.verify(lock, "Dev")
            run = write_run(home / "run", "ref", rows, manifest["manifests"]["dev"]["sha256"])
            other = write_run(home / "second", "ref", rows, manifest["manifests"]["dev"]["sha256"],
                              motion_model=("nvidia/Kimodo-SOMA-RP-v1.2", "other"))
            with mock.patch.object(v1_evidence, "clip_score", return_value=dict(hit=False, usable=False, reason="x")):
                report = v1_evidence.measure(lock, other, "Dev", 1, ref_root=run)
            self.assertIn("reused ref motion model differs", report["evidence_problems"])


class DecideTests(unittest.TestCase):
    def measurement(self, hits_en, hits_ko):
        return {"hits": {"ref": [48] * 4, "en": [hits_en] * 4, "ko": [hits_ko] * 4},
                "usable": {"ref": 192, "en": 192, "ko": 192},
                "peak": {"ref": 100, "en": 10, "ko": 10}, "p95": {"ref": 100, "en": 10, "ko": 10},
                "evidence": True, "standalone": True, "exports": True}

    def test_decide_tracks_candidate_and_freezes_when_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            report_path = home / "report.json"
            state_path = home / "state.json"
            candidate = {"checkpoint_sha256": "cp", "entry_sha256": "entry"}
            report_path.write_text(json.dumps(report_dict(self.measurement(0, 0), candidate)), encoding="utf-8")
            state, entry = v1_evidence.decide(report_path, state_path)
            self.assertEqual(state["phase"], "Dev")
            self.assertEqual(state["candidate_id"], "cp")
            self.assertEqual(entry["bundle_passes"], 5)
            report_path.write_text(json.dumps(report_dict(self.measurement(48, 48), candidate)), encoding="utf-8")
            state, entry = v1_evidence.decide(report_path, state_path)
            self.assertEqual(state["phase"], "Frozen")
            self.assertEqual(state["locked"], 2)
            self.assertTrue(entry["gate"])
            report = report_dict(self.measurement(48, 48), candidate)
            report["stage"] = "Frozen"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            state, entry = v1_evidence.decide(report_path, state_path)
            self.assertEqual(state["phase"], "Accepted")
            self.assertEqual(state["tests"], 1)
            state, entry = v1_evidence.decide(report_path, state_path)
            self.assertEqual(state["phase"], "Accepted")
            self.assertEqual(state["tests"], 1)

    def test_decide_blocks_on_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            report_path = home / "report.json"
            state_path = home / "state.json"
            candidate = {"checkpoint_sha256": "cp", "entry_sha256": "entry"}
            measurement = self.measurement(0, 0)
            measurement["evidence"] = False
            report_path.write_text(json.dumps(report_dict(measurement, candidate)), encoding="utf-8")
            state, entry = v1_evidence.decide(report_path, state_path)
            self.assertEqual(state["phase"], "Blocked")


if __name__ == "__main__":
    unittest.main()
