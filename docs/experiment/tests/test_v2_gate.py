"""CPU-only contract tests. All examples here are fixtures, never evaluation rows."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'v2_gate.py'


def gate():
    if not MODULE.is_file():
        raise AssertionError('v2 evidence gate is missing')
    spec = importlib.util.spec_from_file_location('v2_gate', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SplitTests(unittest.TestCase):
    def test_family_normalization_collapses_mirror_pace_take_and_number(self):
        g = gate()
        self.assertEqual(g.family('walk_arc_090_R_fast'), g.family('walk_arc_270_L_slow'))
        self.assertEqual(g.family('itching_right_calf_R'), g.family('itching_left_calf'))
        self.assertNotEqual(g.family('knightly_bow'), g.family('finger_snap'))

    def test_known_regression_paraphrase_and_violence_are_out(self):
        g = gate()
        for s in ('She sits on the floor with her legs crossed.', 'Person claps their hands.',
                  'He kicks with his left foot.', 'She punches another person.', 'He fires a rifle.'):
            self.assertFalse(g.caption_ok(s), s)
        self.assertTrue(g.caption_ok('A person rubs their palms together.'))

    def test_take_alias_bridge_excludes_a_distant_paraphrase(self):
        g=gate(); d=g.Components()
        d.join('content A','take 1'); d.join('content B','take 1'); d.join('take 2','content B')
        self.assertEqual(d.root('take 2'),d.root('content A'))
        self.assertTrue(g.near(g.signature('A person is walking clockwise in a circle.'),
                               g.signature('The character walks counterclockwise in a circle.')))

    def test_selection_rejects_shortage_without_substitution(self):
        g=gate()
        with self.assertRaisesRegex(ValueError,'insufficient'):
            g.select([], [], b'a'*32)


def fixture_groups(g):
    return [[(st, f'action{i} family{i}', i+2, 1, f'component{i}', (f'fixture{chr(97+i//26)}{chr(97+i%26)} pattern',))]
            for i,st in enumerate(s for s in g.STRATA for _ in range(2*g.PER))]


class FreezeTests(unittest.TestCase):
    def test_freeze_writes_only_public_dev_and_sealed_counts_and_never_redraws(self):
        g=gate(); self.assertTrue(hasattr(g,'freeze'))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); src=d/'source'; src.write_bytes(b'fixture metadata'); out=d/'evaluation'
            # Opaque sealer is a fixture only; production uses Windows DPAPI.
            seal=lambda b: hashlib.sha256(b).digest()
            with patch.object(g,'pool',return_value=fixture_groups(g)) as pool:
                r=g.freeze(out, src, [], {'fixture':True}, seal)
                self.assertEqual(r['counts'],{k:{s:16 for s in g.STRATA} for k in ['dev','sealed-test']})
                self.assertEqual(len(g.dev_rows(out)),64)
                self.assertEqual(g.freeze(out,src,[],{'fixture':True},seal),r)
                self.assertEqual(pool.call_count,1)
                with patch.object(g,'filehash',wraps=g.filehash) as fh:
                    self.assertEqual(len(g.dev_rows(out)),64)
                    self.assertFalse(any('sealed' in str(call.args[0]) for call in fh.call_args_list))
                self.assertFalse(any('sealed-test.json' in p.name for p in out.rglob('*')))
                (out/'dev.jsonl').write_bytes(b'{}\n')
                with self.assertRaises(ValueError): g.dev_rows(out)

    def test_shortage_retains_reason_and_permanently_consumes_draw(self):
        g=gate(); self.assertTrue(hasattr(g,'freeze'))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); src=d/'source'; src.touch(); out=d/'evaluation'
            with patch.object(g,'pool',return_value=[]):
                r=g.freeze(out,src,[],{},lambda b:b'opaque')
                self.assertEqual(r['status'],'blocked'); self.assertIn('insufficient',r['reason'])
                self.assertFalse((out/'dev.jsonl').exists())
                self.assertEqual(g.freeze(out,src,[],{},lambda b:b'opaque'),r)


class EvidenceTests(unittest.TestCase):
    def test_calibration_agreement_and_independent_adjudication(self):
        g=gate(); self.assertTrue(hasattr(g,'blind_labels'))
        controls={f'c{i}':k for i,k in enumerate(g.CONTROL_TYPES)}; ids={f'b{i}' for i in range(20)}
        def sheet(who,observations):
            return {'reviewer_id':who,'calibrated_at_utc':'2026-11-20T11:00:00+00:00','rated_at_utc':'2026-11-20T12:00:00+00:00',
                    'controls':{k:list(g.CONTROL_TRUTH[v]) for k,v in controls.items()},'votes':observations}
        a=sheet('A',dict.fromkeys(ids,[1,1])); b=sheet('B',dict.fromkeys(ids,[1,1])); b['votes']['b0']=[0,1]
        j=sheet('J',{'b0':[0,1]}); panel=('A','B','J')
        self.assertEqual(g.blind_labels(ids,controls,panel,[a,b,j])[0]['b0'],(0,1))
        with self.assertRaises(ValueError): g.blind_labels(ids,controls,('A','A','J'),[a,a,j])
        for i in range(1,4): b['votes'][f'b{i}']=[0,1]
        with self.assertRaisesRegex(ValueError,'agreement'): g.blind_labels(ids,controls,panel,[a,b,j])
        b['controls']['c0']=[0,0]
        with self.assertRaisesRegex(ValueError,'calibration'): g.blind_labels(ids,controls,panel,[a,b,j])

    def test_no_seed_pseudoreplication_and_no_missing_cells(self):
        g=gate(); self.assertTrue(hasattr(g,'paired_metrics'))
        groups={f'{s}-{i}':s for s in g.STRATA for i in range(16)}
        pairs={(k,seed):((0,1),(int(k.endswith('-0') and groups[k]!=g.STRATA[-1]),1)) for k in groups for seed in g.SEEDS}
        m=g.paired_metrics(groups,pairs)
        self.assertEqual(m['overall']['net'],[9,0,9]); self.assertEqual(m['groups'],64)
        self.assertEqual(m['paired_cells'],192); self.assertEqual(m['cluster_interval_net'][0],0)
        self.assertFalse(m['quality_gate_met']); pairs.pop(next(iter(pairs)))
        with self.assertRaisesRegex(ValueError,'paired'): g.paired_metrics(groups,pairs)

    def test_cost_rejects_foreign_device_and_missing_samples(self):
        g=gate(); self.assertTrue(hasattr(g,'cost_gate'))
        device={'uuid':'GPU-11111111-1111-4111-8111-111111111111','name':'NVIDIA GeForce RTX 4090','driver':'fixture','runtime':'fixture'}
        ids=set(range(192)); records={a:{i:{'device':device,'latency_ms':20,'peak_bytes':1024} for i in ids} for a in ['ref','candidate']}
        self.assertTrue(g.cost_gate(ids,device,records)['cost_gate_met'])
        records['candidate'][3]={**records['candidate'][3],'device':{**device,'uuid':'different'}}
        with self.assertRaisesRegex(ValueError,'device'): g.cost_gate(ids,device,records)
        del records['candidate'][3]
        with self.assertRaisesRegex(ValueError,'missing'): g.cost_gate(ids,device,records)

    def test_live_preflight_cannot_release_on_self_reported_files(self):
        g=gate(); self.assertTrue(hasattr(g,'evidence'))
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); src=d/'src'; src.touch(); out=d/'evaluation'
            with patch.object(g,'pool',return_value=fixture_groups(g)):
                g.freeze(out,src,[],{},lambda b:b'opaque')
            (out/'independent-reviews.json').write_text('{"accepted":true}')
            r=g.evidence(out); self.assertEqual(r['status'],'blocked'); self.assertIsNone(r['motion_quality_estimate'])
            self.assertIn('unverified',r['checks']['blind_calibrated_panel']); self.assertIsNone(r['human_ratings_verified'])
            self.assertEqual(g.evidence(out),r)
