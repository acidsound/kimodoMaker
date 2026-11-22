"""CPU fixtures for the verifier; the CLI is also exercised on real pilot takes."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_take(directory):
    names = ["Hips"] + [f"Joint{i}" for i in range(1, 77)]
    offsets = [[0, 1.005, 0]] + [[0, .01, 0] for _ in names[1:]]
    joints = [dict(name=name, parent=-1 if i == 0 else 0, restOffset=offsets[i])
              for i, name in enumerate(names)]
    roots = np.array([[.2, 1., -.4], [.31, .98, -.2]])
    matrices = np.broadcast_to(np.eye(3), (2, 77, 3, 3)).copy()
    posed = roots[:, None, :] + np.array([[0, 0, 0], *offsets[1:]])
    motion = dict(schema="kimodo-maker.soma-motion.v1", skeleton="SOMA-77", units="m",
                  up="+Y", forward="+Z", quaternionOrder="xyzw", fps=30, frameCount=2,
                  rootPositions=roots.tolist(), localRotations=[[0, 0, 0, 1] * 77] * 2,
                  joints=joints, generation=dict(prompt="Fixture", duration=2/30,
                                                  seed=17, steps=100, model="fixture"))
    (directory / "motion.json").write_text(json.dumps(motion), encoding="utf-8")
    np.savez(directory / "motion.npz", posed_joints=posed, root_positions=roots,
             global_rot_mats=matrices, local_rot_mats=matrices)
    channels = "CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation"
    parts = ["HIERARCHY", "ROOT Root", "{", "OFFSET 0 0 0", channels,
             "JOINT Hips", "{", "OFFSET 0 100 0", channels]
    for name in names[1:]:
        parts += [f"JOINT {name}", "{", "OFFSET 0 1 0",
                  "CHANNELS 3 Zrotation Yrotation Xrotation", "}"]
    parts += ["}", "}", "MOTION", "Frames: 2", "Frame Time: 0.03333333333333333"]
    parts += ["0 0 0 0 0 0 " + " ".join(str(x) for x in root * 100) + " 0 0 0" * 77
              for root in roots]
    (directory / "motion.bvh").write_text("\n".join(parts) + "\n", encoding="utf-8")


class ExportCheckTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.take = Path(tmp.name)
        make_take(self.take)

    def test_checks_absolute_xyz_without_adding_bvh_hips_offset(self):
        from v1_export_check import verify_take

        result = verify_take(self.take)
        self.assertEqual((result["frames"], result["fps"]), (2, 30))
        self.assertLess(result["pelvis_max_mm"], 0.001)
        self.assertIsNone(result["npz_fps"])

    def test_rejects_constant_pelvis_shift_above_one_mm(self):
        from v1_export_check import verify_take

        path = self.take / "motion.json"
        motion = json.loads(path.read_bytes())
        for root in motion["rootPositions"]:
            root[2] += .0011
        path.write_text(json.dumps(motion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pelvis.*1 mm"):
            verify_take(self.take)

    def test_rejects_rotation_damage_despite_same_pelvis(self):
        from v1_export_check import verify_take

        path = self.take / "motion.json"
        motion = json.loads(path.read_bytes())
        motion["localRotations"][1][4:8] = [0, 0, 1, 0]
        path.write_text(json.dumps(motion), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "rotation"):
            verify_take(self.take)

    def test_rejects_bvh_fps_mismatch(self):
        from v1_export_check import verify_take

        path = self.take / "motion.bvh"
        path.write_text(path.read_text().replace("0.03333333333333333", "0.04"))
        with self.assertRaisesRegex(ValueError, "FPS"):
            verify_take(self.take)

    def test_rejects_extra_bvh_motion_row(self):
        from v1_export_check import verify_take

        path = self.take / "motion.bvh"
        with path.open("a") as file:
            file.write(path.read_text().splitlines()[-1] + "\n")
        with self.assertRaisesRegex(ValueError, "frame count"):
            verify_take(self.take)

    def test_sandbox_api_only_serves_preseeded_copies_and_disallows_writes(self):
        from v1_export_check import isolated_api, verify_take

        sandbox = self.take / "sandbox"
        sandbox.mkdir()
        before = verify_take(self.take)
        with isolated_api(sandbox, [before]) as (base, [take_id]):
            self.assertTrue(base.startswith("http://127.0.0.1:"))
            with urlopen(f"{base}/api/takes", timeout=5) as response:
                self.assertEqual([take["id"] for take in json.load(response)], [take_id])
            with urlopen(f"{base}/api/takes/{take_id}/bvh?download=1", timeout=5) as response:
                self.assertEqual(response.read(), (self.take / "motion.bvh").read_bytes())
                self.assertIn(f"{take_id[:8]}-motion.bvh", response.headers["Content-Disposition"])
            for verb in ("POST", "DELETE"):
                with self.assertRaises(HTTPError) as failure:
                    urlopen(Request(f"{base}/api/takes/{take_id}", method=verb), timeout=5)
                self.assertEqual(failure.exception.code, 405)
                failure.exception.close()
        self.assertEqual(verify_take(self.take), before)

    def test_playback_requires_actual_avatar_limb_movement_not_timeline_only(self):
        from v1_export_check import playback_delta

        start = {"frame": 0, "bones": [[0, 1, 0], [0.4, 1.3, 0], [-0.4, 1.3, 0]]}
        translated = {"frame": 12, "bones": [[0, 1, 1], [0.4, 1.3, 1], [-0.4, 1.3, 1]]}
        with self.assertRaisesRegex(ValueError, "VRM limbs did not move"):
            playback_delta(start, translated)
        translated["bones"][1][1] += .05
        self.assertAlmostEqual(playback_delta(start, translated), 50)
        translated["frame"] = 0
        with self.assertRaisesRegex(ValueError, "advance"):
            playback_delta(start, translated)
