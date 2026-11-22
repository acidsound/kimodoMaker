"""CPU-only regression evidence checks; no model, server, or sealed material."""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_replay'), 'missing CPU replay auditor')
        import v2_replay
        self.r = v2_replay
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def test_exact_manifest_journal_pairs_and_paths(self):
        self.assertTrue(hasattr(self.r, 'journal'), 'missing read-only journal reconciliation')
        p = self.tmp / 'requests.jsonl'
        row = dict(id='example', kind=1, seconds=2, split='xval', en='jump', ko='점프')
        manifest = self.tmp / 'manifest.jsonl'
        manifest.write_text(json.dumps(row)+'\n', encoding='utf-8')
        self.assertEqual(self.r.manifest(manifest, 1), {'example': row})
        manifest.write_text((json.dumps(row)+'\n')*2, encoding='utf-8')
        with self.assertRaises(ValueError):
            self.r.manifest(manifest, 2)
        records = []
        for seed in (17, 43, 97):
            common = dict(id='example', arm='ref', seed=seed, kind=1, seconds=2,
                          session_id='session', started_monotonic_ns=123, started_utc='t', steps=100)
            paths = {name: f'example-{seed}/{name}' for name in self.r.FILES}
            records.extend([common | dict(status='started'), common | dict(status='ok', artifacts=paths,
                            frame_count=60, fps=30)])
        def check(rows, suffix=b'\n'):
            raw = b'\n'.join(json.dumps(v).encode() for v in rows) + suffix
            p.write_bytes(raw)
            try:
                return self.r.journal(p, 'ref', {'example': row}, 100, 30)
            finally:
                self.assertEqual(p.read_bytes(), raw)  # torn journals must never be repaired
        self.assertEqual(set(check(records)), {('example', s) for s in (17, 43, 97)})
        for bad in [records[:-2], records+[records[-1]], records[:1]+records,
                    records[1:], records[:-1]+[records[-1] | dict(seed=98)],
                    records[:-1]+[records[-1] | dict(seed=97.0)],
                    records[:-1]+[records[-1] | dict(arm='ko')],
                    records[:-1]+[records[-1] | dict(status='failed')],
                    records[:-1]+[records[-1] | dict(steps=99)],
                    records[:-1]+[records[-1] | dict(seconds=2.5)],
                    records[:-1]+[records[-1] | dict(session_id='different')],
                    records[:-1]+[records[-1] | dict(frame_count=59)],
                    records[:-1]+[records[-1] | dict(artifacts=records[1]['artifacts'])]]:
            with self.assertRaises(ValueError):
                check(bad)
        with self.assertRaises(ValueError):
            check(records, suffix=b'')

    def test_compare_recorded_digests_not_just_inventory(self):
        self.assertTrue(hasattr(self.r, 'fingerprints'), 'missing artifact verification')
        expected = {}
        for name in self.r.FILES:
            data = name.encode()
            (self.tmp / name).write_bytes(data)
            expected[name] = hashlib.sha256(data).hexdigest()
        self.assertEqual(self.r.fingerprints(self.tmp, {}), expected)
        self.assertEqual(self.r.fingerprints(self.tmp, expected), expected)
        (self.tmp / 'motion.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'motion.npz'):
            self.r.fingerprints(self.tmp, expected)
        (self.tmp / 'motion.bvh').unlink()
        with self.assertRaises(FileNotFoundError):
            self.r.fingerprints(self.tmp, {})

    def make_clip(self):
        g = dict(prompt='jump', seed=17, duration=2/30, steps=100, model='SOMA', constraintCount=0)
        joints = [dict(name='Hips' if i == 0 else str(i), parent=i-1, restOffset=[1, 0, 0]) for i in range(77)]
        roots = np.arange(6.).reshape(2, 3)
        pos = np.zeros((2,77,3)) + roots[:,None,:]
        pos[:,:,0] += np.arange(77)
        np.savez(self.tmp/'motion.npz', posed_joints=pos, root_positions=roots)
        j = dict(schema='kimodo-maker.soma-motion.v1', skeleton='SOMA-77', units='m', up='+Y',
                 forward='+Z', quaternionOrder='xyzw', generation=g, frameCount=2, fps=30, joints=joints,
                 rootPositions=roots.tolist(), localRotations=[[0, 0, 0, 1]*77]*2)
        (self.tmp/'motion.json').write_text(json.dumps(j))
        (self.tmp/'motion.bvh').write_text('HIERARCHY\nROOT Hips\n{\n  CHANNELS 3 Xposition Yposition Zposition\n}\nMOTION\nFrames: 2\nFrame Time: 0.03333333333333333\n0 0 0\n0 0 0\n')
        return g, j

    def test_app_export_pairing_before_retarget_not_visual_evidence(self):
        self.assertTrue(hasattr(self.r, 'inspect_clip'), 'missing NPZ/JSON pairing')
        g, j = self.make_clip()
        result, _, _ = self.r.inspect_clip(self.tmp, g, 2, 30, {})
        self.assertEqual(result['npz_to_soma_json_fk_rms_cm'], 0)
        self.assertEqual(result['npz_to_soma_json_hips_max_cm'], 0)
        j['joints'][1]['restOffset'] = [2, 0, 0]
        (self.tmp/'motion.json').write_text(json.dumps(j))
        self.assertGreater(self.r.inspect_clip(self.tmp, g, 2, 30, {})[0]['npz_to_soma_json_fk_rms_cm'], 0)
        for wrong in [g | dict(prompt='jump.'), g | dict(steps=99), g | dict(seed=43)]:
            with self.assertRaises(ValueError):
                self.r.inspect_clip(self.tmp, wrong, 2, 30, {})
        (self.tmp/'motion.bvh').write_text('MOTION\nFrames: 1\nFrame Time: 0.0333333333\n0\n')
        with self.assertRaises(ValueError):
            self.r.inspect_clip(self.tmp, g, 2, 30, {})

    def test_report_is_repeatable_and_cannot_overwrite_source_or_changed_evidence(self):
        self.assertTrue(hasattr(self.r, 'publish'), 'missing safe evidence publication')
        out = self.tmp/'out/diagnostics/v1-replay.json'
        source = self.tmp/'v1'
        source.mkdir()
        with self.assertRaises(ValueError):
            self.r.publish({'ok': True}, source/'diagnostics/v1-replay.json', (source,), False)
        with self.assertRaises(FileNotFoundError):
            self.r.publish({'ok': True}, out, (source,), True)
        self.r.publish({'ok': True}, out, (source,), False)
        self.r.publish({'ok': True}, out, (source,), True)
        first = out.read_bytes()
        with self.assertRaises(ValueError):
            self.r.publish({'ok': False}, out, (source,), False)
        self.assertEqual(first, out.read_bytes())

    def test_cli_rejects_unreadable_archive_without_publishing(self):
        import subprocess
        import sys
        out = self.tmp/'diagnostics/v1-replay.json'
        cmd = [sys.executable, '-B', self.r.__file__, '--v1', str(self.tmp/'missing'),
               '--takes', str(self.tmp/'takes'), '--output', str(out)]
        run = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn('missing', run.stderr)
        self.assertFalse(out.exists())

    def test_external_pose_link_must_use_the_same_app_artifacts(self):
        self.assertTrue(hasattr(self.r, 'link_app_poses'))
        p = self.tmp/'report.json'
        pose = dict(generation={'prompt': 'test'}, numeric={'path': str(self.tmp/'app/id'),
                    'sha256': {ext: '0'*64 for ext in ('npz', 'bvh', 'json')}}, frames=2)
        p.write_text(json.dumps(dict(schema='kimodo.v2.app-probe.1', diagnostic_only=True,
                                    takes=[pose], browser_errors=[])))
        take = dict(take_id='id', generation={'prompt':'test'}, frame_count=2,
                    artifact_sha256={f'motion.{ext}': '1'*64 for ext in ('npz','bvh','json')})
        with self.assertRaisesRegex(ValueError, 'artifact'):
            self.r.link_app_poses(p, self.tmp/'app', [take])

    def test_cli_binds_app_probe_but_keeps_it_out_of_archive_diagnosis(self):
        import subprocess
        import sys
        repo = Path(__file__).resolve().parents[3]
        runs = repo.parent/'kimodoMaker-runs'
        poses = runs/'v2-encoder-4090/english-conditioning-01/diagnostics/original-app-poses/report.json'
        if not poses.is_file() or not (runs/'v1-encoder-4090/xval/manifest.jsonl').is_file() or not (repo/'backend/takes').is_dir():
            self.skipTest('requires original local archive, app store and separate pose evidence')
        out = self.tmp/'diagnostics/v1-replay.json'
        cmd = [sys.executable, '-B', self.r.__file__, '--v1', str(runs/'v1-encoder-4090'),
               '--takes', str(repo/'backend/takes'), '--app-poses', str(poses), '--output', str(out)]
        receipt = json.loads(subprocess.check_output(cmd))
        self.assertEqual(receipt['report_sha256'], hashlib.sha256(out.read_bytes()).hexdigest())
        report = json.loads(out.read_bytes())
        self.assertEqual(report['diagnosis']['vrm_attribution'], 'undetermined')
        probe = report['original_app']['browser_evidence']
        self.assertEqual(probe['sha256'], hashlib.sha256(poses.read_bytes()).hexdigest())
        foot = probe['takes']['5705ce5f10a7425eabed3d28cc18c3ee']
        self.assertIn('before retargeting', foot['observation'])
        hist = next(r for r in report['regression']['xval-hist']['requests'] if r['id']=='xval-h-01')
        self.assertEqual(hist['clips']['ref']['generation']['prompt'], 'rolling forward and hero pose')
        self.assertEqual(hist['clips']['ref']['reconstructed_runner_prompt'], 'rolling forward and hero pose.')

    def test_real_audit_when_local_evidence_is_installed(self):
        repo = Path(__file__).resolve().parents[3]
        v1 = repo.parent/'kimodoMaker-runs/v1-encoder-4090'
        takes = repo/'backend/takes'
        if not (v1/'xval/manifest.jsonl').is_file() or not takes.is_dir():
            self.skipTest('requires original local v1 evidence')
        self.assertTrue(hasattr(self.r, 'audit'), 'missing integrated audit')
        report = self.r.audit(v1, takes)
        self.assertEqual(report['totals']['regression_clips'], 153)
        self.assertEqual(report['totals']['regression_artifacts'], 459)
        self.assertEqual(report['totals']['regression_recorded_artifact_hashes'], 0)
        for arm, expected in [('en', (4.95, 49.98, 73.47)), ('ko', (11.50, 55.85, 74.05))]:
            stats = report['rms_cm'][arm]
            self.assertEqual(stats['pairs'], 51)
            for field, value in zip(('min','median','max'), expected):
                self.assertAlmostEqual(stats[field], value, delta=.01)
        by_set = report['regression']
        self.assertEqual([len(by_set[s]['requests']) for s in ['xval','xval-hist']], [15,36])
        for s in by_set.values():
            self.assertEqual(set(s['arms']), set(self.r.ARMS))
            for req in s['requests']:
                self.assertEqual(set(req['clips']), set(self.r.ARMS))
                for arm, clip in req['clips'].items():
                    self.assertEqual(clip['generation']['prompt'], req['ko' if arm=='ko' else 'en'].strip())
        matches = [c for c in report['original_app']['takes'] if c['prompt_matches']]
        self.assertIn('aa22d98cf9ae41aca71f4d340fa46dbe', [c['take_id'] for c in matches])
        historical = next(c for c in matches if c['take_id']=='aa22d98cf9ae41aca71f4d340fa46dbe')
        self.assertEqual(historical['generation']['seed'], 123123)
        self.assertEqual(historical['generation']['steps'], 80)
        self.assertFalse(historical['prompt_matches'][0]['matched_ref_seeds_by_recorded_options'])
        self.assertEqual(report['diagnosis']['vrm_attribution'], 'undetermined')

    def test_root_aligned_3d_rms(self):
        root = np.arange(6, dtype=float).reshape(2, 3)
        pose = np.zeros((2, 77, 3)) + root[:, None, :]
        p = self.tmp / 'motion.npz'
        np.savez(p, posed_joints=pose, root_positions=root)
        a = self.r.load_pose(p, 2)
        self.assertEqual(self.r.rms_cm(a, a), 0)
        # Same pose after a world translation; then all joints move 3/4 cm in x/y.
        np.savez(p, posed_joints=pose+2, root_positions=root+2)
        self.assertEqual(self.r.rms_cm(a, self.r.load_pose(p, 2)), 0)
        self.assertAlmostEqual(self.r.rms_cm(a, a+[.03, .04, 0]), 5)
        with self.assertRaises(ValueError):
            self.r.rms_cm(a, a[:1])  # never broadcast unequal frame counts

    def test_reject_missing_wrong_nonfinite_or_duplicate_npz(self):
        import io
        import warnings
        import zipfile
        p = self.tmp / 'motion.npz'
        with self.assertRaises(FileNotFoundError):
            self.r.load_pose(p, 2)
        valid = dict(posed_joints=np.zeros((2, 77, 3)), root_positions=np.zeros((2, 3)))
        for invalid in [dict(joints=valid['posed_joints']), {'root_positions': valid['root_positions']},
                        valid | {'posed_joints': np.zeros((1, 77, 3))},
                        valid | {'root_positions': np.zeros((1, 3))},
                        valid | {'posed_joints': np.full((2, 77, 3), np.inf)},
                        valid | {'root_positions': np.full((2, 3), np.nan)}]:
            with self.subTest(keys=list(invalid)):
                np.savez(p, **invalid)
                with self.assertRaises(ValueError):
                    self.r.load_pose(p, 2)
        np.savez(p, **valid)
        buf = io.BytesIO()
        np.save(buf, valid['root_positions'])
        with warnings.catch_warnings(), zipfile.ZipFile(p, 'a') as archive:
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('root_positions.npy', buf.getvalue())
        with self.assertRaises(ValueError):
            self.r.load_pose(p, 2)
