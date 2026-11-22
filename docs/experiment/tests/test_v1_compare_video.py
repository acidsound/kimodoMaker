"""CPU-only integration tests for the blind video artifact contract."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "v1_compare_video.py"
ARMS = {"ref": "Raise a hand.", "en": "Raise a hand.", "ko": "손을 든다."}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "need ffmpeg and ffprobe")
class CompareVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.manifest = self.base / "dev.jsonl"
        self.manifest.write_text(json.dumps(dict(id="raise", split="dev", en=ARMS["en"],
                                                 ko=ARMS["ko"], seconds=0.5)) + "\n",
                                 encoding="utf-8")
        self.motions = self.base / "out"

    def write_clip(self, arm, *, seed=17, prompt=None, ident="raise"):
        folder = self.motions / arm / "raise-17"
        folder.mkdir(parents=True)
        joints = np.zeros((15, 77, 3), dtype=np.float32)
        joints[:, :, 0] = np.linspace(0, 0.1, 15)[:, None]
        joints[:, :, 1] = np.linspace(1, 1.7, 77)
        np.savez(folder / "motion.npz", posed_joints=joints)
        document = dict(schema="kimodo-maker.soma-motion.v1", skeleton="SOMA-77", units="m",
                        up="+Y", forward="+Z", frameCount=15, fps=30,
                        generation=dict(seed=seed, prompt=prompt or ARMS[arm], duration=0.5,
                                        arm=arm, id=ident),
                        joints=[dict(name="Hips", parent=-1)] +
                        [dict(name=f"Bone{i}", parent=i - 1) for i in range(1, 77)])
        (folder / "motion.json").write_text(json.dumps(document), encoding="utf-8")
        return folder

    def render(self, *selection, tag="blind", extra=()):
        blind = self.base / tag
        key = self.base / f"{tag}-key.json"
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--manifest", str(self.manifest),
                                 "--motions", str(self.motions), "--output-dir", str(blind),
                                 "--key-out", str(key), *selection, *extra],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return blind, self.read(blind / "report.json"), self.read(key)

    @staticmethod
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def test_manifest_clip_encodes_at_native_fps_with_key_outside_blind_package(self):
        for arm in ARMS:
            self.write_clip(arm)
        blind, report, key = self.render("--id", "raise", "--seed", "17")
        self.assertFalse(report["human_score"])
        self.assertEqual(report["expected_groups"], 1)
        entry = report["groups"][0]
        self.assertEqual((entry["id"], entry["seed"]), ("raise", 17))
        self.assertEqual(entry["panels"], dict.fromkeys("ABC", "ok"))
        self.assertEqual({p["arm"] for p in key["groups"][0]["panels"].values()}, set(ARMS))
        self.assertEqual({p.name for p in blind.iterdir()}, {"report.json", "raise-17.mp4"})
        self.assertNotIn('"arm"', (blind / "report.json").read_text(encoding="utf-8"))
        probe = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                         "-count_frames", "-show_streams", "-of", "json",
                                         str(blind / entry["video"])], encoding="utf-8")
        s = json.loads(probe)["streams"][0]
        self.assertEqual((s["codec_name"], s["width"], s["height"], s["r_frame_rate"],
                          s["nb_read_frames"]), ("h264", 960, 320, "30/1", "15"))

    def test_default_enumerates_seeds_and_never_renders_missing_or_failed_motion(self):
        self.write_clip("ref")
        self.write_clip("en", seed=97)
        blind, report, key = self.render()
        self.assertEqual(report["expected_groups"], 3)
        by_seed = {group["seed"]: group for group in report["groups"]}
        self.assertEqual(set(by_seed), {17, 43, 97})
        mapping = {p["arm"]: p["status"] for p in key["groups"][0]["panels"].values()}
        self.assertEqual(mapping, {"ref": "ok", "en": "failed", "ko": "missing"})
        self.assertEqual(by_seed[17]["video_status"], "rendered")
        for seed in (43, 97):
            self.assertEqual(by_seed[seed]["video_status"], "no_motion")
            self.assertIsNone(by_seed[seed]["video"])
        self.assertEqual([p.name for p in blind.glob("*.mp4")], ["raise-17.mp4"])

    def test_logged_failure_overrides_motion_and_metadata_id_mismatch_fails(self):
        folder = self.write_clip("ref")
        (folder.parent / "requests.jsonl").write_text(
            json.dumps(dict(arm="ref", id="raise", seed=17, status="failed", error="generation failed")) + "\n",
            encoding="utf-8")
        self.write_clip("en", ident="different")
        _, report, key = self.render("--seed", "17")
        self.assertEqual(report["groups"][0]["video_status"], "no_motion")
        panels = {p["arm"]: p for p in key["groups"][0]["panels"].values()}
        self.assertIn("generation failed", panels["ref"]["reason"])
        self.assertEqual(panels["en"]["status"], "failed")

    def test_running_request_does_not_render_uncommitted_motion(self):
        folder = self.write_clip("ref")
        (folder.parent / "requests.jsonl").write_text(
            json.dumps(dict(arm="ref", id="raise", seed=17, status="started")) + "\n",
            encoding="utf-8")
        _, report, key = self.render("--seed", "17")
        self.assertEqual(report["groups"][0]["video_status"], "no_motion")
        ref = next(p for p in key["groups"][0]["panels"].values() if p["arm"] == "ref")
        self.assertEqual(ref["status"], "missing")

    def test_new_runs_can_reproduce_private_order_for_a_subset(self):
        self.write_clip("ref")
        _, _, first_key = self.render()
        _, _, second_key = self.render("--seed", "17", tag="rerun",
                                       extra=("--key-in", str(self.base / "blind-key.json")))
        self.assertEqual(first_key["groups"][0], second_key["groups"][0])

    def test_uncanonicalized_export_prompt_matches_generator_normalization(self):
        self.write_clip("en", prompt="Raise a hand")
        _, report, key = self.render("--seed", "17")
        self.assertEqual(report["groups"][0]["video_status"], "rendered")
        en = next(p for p in key["groups"][0]["panels"].values() if p["arm"] == "en")
        self.assertEqual(en["status"], "ok")

    def test_smoke_accepts_unrelated_takes_but_never_labels_them_comparable(self):
        folders = [self.write_clip(arm) for arm in ARMS]
        blind = self.base / "smoke"
        key = self.base / "smoke-key.json"
        result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--smoke-clips", *map(str, folders),
                                 "--output-dir", str(blind), "--key-out", str(key)],
                                capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = self.read(blind / "report.json")
        self.assertEqual(report["mode"], "smoke")
        self.assertFalse(report["comparable"])
        self.assertIsNone(report["groups"][0]["seed"])
        self.assertTrue((blind / report["groups"][0]["video"]).is_file())
        self.assertTrue(all("arm" not in p for p in self.read(key)["groups"][0]["panels"].values()))

    def test_whole_split_keeps_192_expected_groups_when_every_export_is_missing(self):
        sample = self.read(self.manifest)
        self.manifest.write_text("".join(json.dumps(sample | {"id": f"move-{i:02d}"}) + "\n"
                                         for i in range(64)), encoding="utf-8")
        digest = sha256(self.manifest.read_bytes()).hexdigest()
        self.motions.mkdir()
        (self.motions / "config.json").write_text(json.dumps(dict(manifest_sha256=digest, split="dev", pilot=False)),
                                                  encoding="utf-8")
        seed_key = self.base / "fixed-private-key.json"
        seed_key.write_text(json.dumps(dict(schema="kimodo.v1-comparison-video.key.v1",
                                            mode="manifest", manifest_sha256=digest, secret="00" * 32)),
                            encoding="utf-8")
        blind, report, key = self.render(extra=("--key-in", str(seed_key)))
        self.assertEqual((report["expected_groups"], report["expected_panels"]), (192, 576))
        self.assertEqual(report["panel_counts"], {"missing": 576})
        self.assertEqual(report["video_counts"], {"no_motion": 192})
        self.assertEqual(len({(g["id"], g["seed"]) for g in report["groups"]}), 192)
        orders = {tuple(group["panels"][label]["arm"] for label in "ABC") for group in key["groups"]}
        self.assertEqual(len(orders), 6)
        self.assertEqual([p.name for p in blind.iterdir()], ["report.json"])


if __name__ == "__main__":
    unittest.main()
