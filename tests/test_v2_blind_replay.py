"""CPU-only P1.3 contract; never opens independent evaluation material."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1] / 'docs' / 'experiment'
sys.path.insert(0, str(EXPERIMENT))


class BlindReplayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_blind_replay'), 'missing historical blind replay')
        import v2_blind_replay as blind
        self.b = blind

    def test_manifest_pinned_before_using_any_motion_or_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / 'xval').mkdir()
            (p / 'xval/manifest.jsonl').write_text(json.dumps(dict(split='xval', intent='dummy')) + '\n')
            with self.assertRaisesRegex(ValueError, 'manifest SHA-256'):
                self.b.inventory(p)
        self.assertEqual([a[:3] for a in self.b.DATASETS], [
            ('xval', 'xval/manifest.jsonl', 5), ('xval-hist', 'xval/hist-manifest.jsonl', 12)])

    def test_units_have_opaque_ordered_keys_and_blind_independent_ballots(self):
        b = self.b
        self.assertTrue(hasattr(b, 'assignment'), 'missing keyed per-unit blinding')
        secret = bytes(range(32))
        arms = b.renderer.ARMS
        seen, orders = set(), set()
        for seed in range(50):
            name, order = b.assignment(secret, ('xval', 'private-source-id', seed), arms)
            self.assertEqual((name, order), b.assignment(secret, ('xval', 'private-source-id', seed), arms))
            self.assertRegex(name, '^[a-f0-9]{24}$')
            self.assertNotIn(name, seen)
            self.assertEqual(set(order), set(arms))
            seen.add(name)
            orders.add(order)
        self.assertEqual(len(orders), 6)
        form = b.ballots([dict(unit=name)])
        import csv, io
        rows = list(csv.DictReader(io.StringIO(form)))
        self.assertEqual([(r['unit'], r['panel']) for r in rows], [(name, s) for s in 'ABC'])
        self.assertTrue(all(r['action'] == r['visual_quality'] == r['evidence_time_and_reason'] == '' for r in rows))
        for token in (*arms, 'private-source-id'):
            self.assertNotIn(',' + token + ',', form)

    def test_media_checks_actual_frames_and_rejects_unplayable_video(self):
        b = self.b
        self.assertTrue(hasattr(b, 'media_evidence'), 'missing full media verification')
        import shutil, subprocess
        if not all(shutil.which(t) for t in ('ffmpeg', 'ffprobe')):
            self.skipTest('requires ffmpeg and ffprobe')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'test.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                            'color=s=960x320:r=30', '-frames:v', '3', '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', str(p)], check=True)
            result = b.media_evidence(p, 3)
            self.assertEqual((result['decoded_frames'], result['probed_frames']), (3, 3))
            self.assertEqual(result['video_sha256'], b.audit.digest(p))
            self.assertRegex(result['decoded_frames_sha256'], '^[a-f0-9]{64}$')
            with self.assertRaisesRegex(ValueError, 'frame count'):
                b.media_evidence(p, 2)
            p.write_bytes(p.read_bytes()[:99])
            with self.assertRaises((ValueError, subprocess.CalledProcessError)):
                b.media_evidence(p, 3)

    def test_real_evidence_contract_if_installed(self):
        b = self.b
        self.assertTrue(hasattr(b, 'review'), 'missing build/verify workflow')
        root = EXPERIMENT.parents[2]/'kimodoMaker-runs'
        run = root/'v2-encoder-4090/english-conditioning-01'
        p = run/'historical-review'
        if not (p/'done.json').exists():
            self.skipTest('requires completed historical review bundle')
        done = b.review('verify', root/'v1-encoder-4090', run)
        self.assertEqual(done['historical_trios'], {'xval': 15, 'xval-hist': 36})
        self.assertEqual((done['control_panels_verified'], done['videos'], done['historical_panels']), (9, 54, 153))
        self.assertEqual(done['human_ratings'], 0)
        units = json.loads((p/'public/units.json').read_text(encoding='utf-8'))
        self.assertEqual({tuple(sorted(u)) for u in units}, {('action_cue','prompt','unit','video')})
        for unit in units:
            rec = b.audit.read(p/'private/units'/f"{unit['unit']}.json")
            self.assertEqual(set(rec['panels']), set('ABC'))
            self.assertEqual(rec['unit'], unit['unit'])
            self.assertEqual(rec['media']['video_sha256'], b.audit.digest(p/'public'/unit['video']))
        with self.assertRaisesRegex(ValueError, 'already exists'):
            b.review('build', root/'v1-encoder-4090', run)

    def test_never_build_in_input_or_verify_incomplete_output(self):
        b = self.b
        self.assertTrue(hasattr(b, 'review'), 'missing protected publication boundary')
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'; source.mkdir()
            with self.assertRaisesRegex(ValueError, 'overlap'):
                b.review('build', source, source)
            self.assertEqual(list(source.iterdir()), [])
            run = Path(tmp)/'run'; (run/'historical-review').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                b.review('verify', source, run)

    def test_controls_distinguish_geometry_not_model_or_human_scores(self):
        import numpy as np
        b = self.b
        self.assertTrue(hasattr(b, 'make_controls'), 'missing geometric controls')
        names = ('Hips', 'HeadEnd', 'LeftArm', 'LeftHand', 'RightArm', 'RightHand', 'LeftFoot', 'RightFoot')
        pose = [[0,1,0],[0,1.7,0],[.2,1.4,0],[.25,.85,0],[-.2,1.4,0],[-.25,.85,0],[.1,.05,0],[-.1,.05,0]]
        clip = b.renderer.Clip(np.array([pose], dtype=float), names, (-1,0,0,2,0,4,0,0), 30)
        cases = b.make_controls(clip)
        self.assertEqual(set(cases), {'hand', 'turn', 'stance'})
        expected = [('hand', 'left_overhead'), ('hand', 'right_overhead'), ('hand', 'still'),
                    ('turn', 'half_turn'), ('turn', 'under_turn'), ('turn', 'still'),
                    ('stance', 'planted'), ('stance', 'slide'), ('stance', 'pop')]
        for case, intended in expected:
            c = cases[case][intended]
            observed, metrics = b.control_observation(case, c)
            self.assertEqual(observed, intended)
            self.assertTrue(metrics and all(np.isfinite(n) for n in metrics.values()))
            np.testing.assert_allclose(c.positions[0], clip.positions[0])
            self.assertEqual((len(c.positions), c.fps), (60, 30))
        self.assertNotEqual(b.control_observation('hand', cases['hand']['right_overhead'])[0], 'left_overhead')
        np.testing.assert_array_equal(clip.positions[0], pose)


if __name__ == '__main__':
    unittest.main()
