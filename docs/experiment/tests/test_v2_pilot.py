import importlib.util
import unittest

import numpy as np


class PilotTest(unittest.TestCase):
    def test_body_relative_travel_is_invariant_to_global_yaw(self):
        self.assertIsNotNone(importlib.util.find_spec('v2_pilot'), 'missing pilot implementation')
        from v2_pilot import measure
        # Half-turn from the neutral pose: anatomical left=-X, forward=-Z.
        left = np.tile([-0.1, 0, 0.0], (90, 1)); right = -left
        for direction, delta in [('left', [-1,0,0]), ('right', [1,0,0]),
                                 ('forward', [0,0,-1]), ('backward', [0,0,1])]:
            roots = np.linspace(np.zeros(3), delta, len(left))
            m = measure(roots, left+roots, right+roots, direction)
            self.assertGreater(m['along_m'], .90)
            self.assertAlmostEqual(m['off_axis_m'], 0.)
            self.assertTrue(m['hit'])
            self.assertFalse(measure(-roots, left-roots, right-roots, direction)['hit'])

    def test_committed_unit_rechecks_key_inputs_and_artifacts_before_skip(self):
        import tempfile
        from pathlib import Path
        import v2_pilot as p
        self.assertTrue(hasattr(p,'unit'), 'missing atomic resume unit')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); published = root/'ref.first.17'
            calls = []
            def create(stage):
                self.assertFalse(published.exists())
                calls.append(1); (stage/'motion.npz').write_bytes(b'unit fixture')
                return {'fixture':True}
            args = (published, {'lock':'abc','key':'ref.first.17'}, ['motion.npz'], create)
            first = p.unit(*args); self.assertEqual(first, p.unit(*args)); self.assertEqual(calls,[1])
            with self.assertRaises(ValueError): p.unit(published,{'lock':'changed'},args[2],create)
            (published/'motion.npz').write_bytes(b'other')
            with self.assertRaises(ValueError): p.unit(*args)
            self.assertEqual(calls,[1])

    def test_plan_matches_budget_and_controls_precede_candidate(self):
        import v2_pilot as p
        self.assertTrue(hasattr(p,'plan'), 'missing locked request planner')
        proto = example_protocol()
        jobs = p.plan(proto)
        self.assertEqual([j['key'] for j in jobs[:4]],['witness.historical.17','ref.a.17','repeat.a.17','candidate.a.17'])
        self.assertEqual(len(jobs),10)
        self.assertEqual(len({j['key'] for j in jobs}),len(jobs))
        for j in jobs:
            if j['arm']=='candidate':
                r=next(r for r in jobs if r['key']==j['key'].replace('candidate.','ref.'))
                self.assertEqual(j['options'],r['options'])
        proto['rows'][1]['id']='a'
        with self.assertRaises(ValueError): p.plan(proto)


    def test_partial_summary_pairs_keys_not_list_positions(self):
        import v2_pilot as p
        self.assertTrue(hasattr(p,'summarize'), 'missing matched report')
        jobs=p.plan(example_protocol()); data={}
        for j in jobs:
            # Missing candidate b/43: must be visibly incomplete, not silently a loss.
            if j['key']=='candidate.b.43': continue
            hit = j['arm']=='candidate' or j['id']=='b'
            data[j['key']]={'result':{'semantic':dict(hit=hit,along_m=1. if hit else -1.)}}
        rep=p.summarize(jobs,data)
        self.assertEqual([rep[k] for k in ('pairs','expected_pairs','ref_hits','candidate_hits','rescues','losses')],[3,4,1,3,2,0])
        self.assertEqual(rep['groups']['x']['pairs'],3)
        self.assertFalse(rep['complete'])

    @unittest.skipUnless(__import__('os').getenv('V2_PILOT_REAL'), 'opt-in real 4090 pilot; no default GPU allocation')
    def test_real_gpu_official_engine_exports(self):
        import os, json, subprocess, sys
        from pathlib import Path
        from v1_generate import ROOT
        out=Path(os.environ['V2_PILOT_REAL']); cmd=[sys.executable,'-B','-u','docs/experiment/v2_pilot.py','--pilot',str(out),'--hf-cache','H:/HF_CACHE/hub']
        subprocess.run(cmd,cwd=ROOT,check=True)
        self.assertTrue((out/'report.json').is_file(),'missing real paired pilot report')
        report=json.loads((out/'report.json').read_bytes())
        self.assertTrue(report['complete']); self.assertEqual(report['pairs'],24)
        self.assertTrue(all(v['pass'] for v in report['parity'].values()))


def example_protocol():
    return dict(rows=[dict(id='a',prompt='A.',group='x',direction='left'),dict(id='b',prompt='B.',group='x',direction='right')],
                seeds=[17,43],settings=dict(duration=3.,steps=100,constraints=[],fps=30,samples=1,post_processing=True,cfg='unchanged Maker defaults'),
                budget=dict(maximum_committed_units=10,paired_clips=8),
                witness=dict(id='historical',prompt='C.',seed=17,duration=2.5,steps=100,path='unused'))
