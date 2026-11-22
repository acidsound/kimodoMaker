"""Compare real SOMA and raw VRM bones on every frame; no semantic pass labels."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

import numpy as np


def gap(observed, target):
    a, b = np.asarray(observed, dtype=float), np.asarray(target, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or not len(a) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Expected matching nonempty finite frame-by-XYZ arrays')
    delta = a - b
    distance = np.linalg.norm(delta, axis=1)
    i = int(distance.argmin())
    return dict(closest_frame=i, closest_m=float(distance[i]), dy_at_closest_m=float(delta[i, 1]),
                max_dy_m=float(delta[:, 1].max()))


BONES = ('Hips', 'Head', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand')
PAIRS = ((3, 1), (2, 1), (5, 4))
# Build the actual UI into a temporary directory; only add a read-only viewer reference.
# Serving built assets by browser interception avoids Vite defaulting port 0 to 5173.
BUILD = """
import { build } from 'vite';
await build({configFile: false, envFile: false, publicDir: false, logLevel: 'error',
  plugins: [{name:'pose-probe', enforce:'pre', transform(src,id){
    if (id.replaceAll('\\\\','/').endsWith('/src/main.js'))
      return src+'\\nObject.defineProperty(globalThis,"__poseProbe",{get:()=>viewer});';
  }}], build:{outDir: process.argv[1]}});
"""
SAMPLE = """names => {
  const v = globalThis.__poseProbe;
  if (v.playing || !v.retarget || v.renderer.getContext().isContextLost()) throw Error('VRM not bound');
  const nodes = names.map(k => v.vrm.humanoid.getRawBoneNode(k));
  if (nodes.some(n => !n)) throw Error('Missing raw VRM bone');
  const result = [];
  for(let i=0;i<v.motion.frameCount;i++) {
    v.seek(i); v.vrm.update(0); v.vrm.scene.updateMatrixWorld(true);
    result.push(nodes.map(n => n.getWorldPosition(n.position.clone()).toArray()));
  }
  return result;
}"""


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def run(folders, vrm, chrome, evidence):
    from playwright.sync_api import expect, sync_playwright
    from v1_export_check import ROOT, isolated_api, verify_take
    from v1_generate import write_json

    if evidence.exists() or not folders or len(set(folders)) != len(folders):
        raise ValueError('Use a new evidence directory and distinct input takes')
    reports = [verify_take(p) for p in folders]
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=evidence.parent) as work:
        tmp = Path(work)
        out = tmp / 'evidence'
        out.mkdir()
        dist = tmp / 'dist'
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith('VITE_')}
        subprocess.run([shutil.which('node') or 'node', '--input-type=module', '-e', BUILD, str(dist)],
                       cwd=ROOT / 'frontend', env=env, check=True, timeout=90)
        records, errors = [], []
        with isolated_api(tmp, reports) as (api, ids), sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=str(chrome), args=['--use-gl=angle',
                '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-background-networking'])
            context = browser.new_context(accept_downloads=True)

            def local_only(route):
                url = route.request.url
                path = urlsplit(url).path
                if not url.startswith(api + '/') or route.request.method != 'GET':
                    return route.abort()
                if path.startswith('/api/'):
                    return route.continue_()
                if path == '/favicon.ico':
                    return route.fulfill(status=204)
                f = (dist / (path.lstrip('/') or 'index.html')).resolve()
                if not f.is_relative_to(dist) or not f.is_file():
                    return route.abort()
                mime = {'.js': 'text/javascript', '.html': 'text/html', '.css': 'text/css'}
                return route.fulfill(path=str(f), content_type=mime[f.suffix])

            context.route('**/*', local_only)
            page = context.new_page()
            page.on('pageerror', lambda err: errors.append(str(err)))
            page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
            page.goto(api)
            expect(page.locator('#connection')).to_have_attribute('data-state', 'online')
            page.locator('#vrm-file').set_input_files(str(vrm))
            expect(page.locator('#avatar-name')).to_have_text(vrm.name)
            page.locator('#skeleton-visible').uncheck()
            for folder, ident, numeric in zip(folders, ids, reports):
                page.locator(f'[data-take-id="{ident}"] .take-main').click()
                expect(page.locator('#take-info')).to_contain_text(ident[:8])
                avatar = np.asarray(page.evaluate(SAMPLE, [s[0].lower() + s[1:] for s in BONES]))
                info = json.loads((folder / 'motion.json').read_bytes())
                joint_names = [j['name'] for j in info['joints']]
                with np.load(folder / 'motion.npz', allow_pickle=False) as npz:
                    soma = npz['posed_joints'][:, [joint_names.index(name) for name in BONES]]
                if soma.shape != avatar.shape or soma.shape != (numeric['frames'], len(BONES), 3):
                    raise ValueError('Frame/joint coverage mismatch')
                stats = {arm: {f'{BONES[a]}-{BONES[b]}': gap(x[:, a], x[:, b]) for a, b in PAIRS}
                         for arm, x in [('soma', soma), ('vrm', avatar)]}
                name = str(len(records) + 1)
                traces = out / f'{name}-joints.npz'
                np.savez_compressed(traces, soma=soma, vrm=avatar)
                # Screenshot of the requested extremum, not a cherry-picked quality score.
                probe = 'LeftHand' if 'clap' in info['generation']['prompt'].lower() else 'RightFoot'
                pair = 'RightHand-LeftHand' if probe == 'LeftHand' else 'RightFoot-Head'
                frame = stats['vrm'][pair]['closest_frame']
                page.evaluate('i=>{const v=globalThis.__poseProbe;v.seek(i);v.vrm.update(0);v.renderer.render(v.scene,v.camera)}', frame)
                shot = out / f'{name}-vrm.png'
                page.locator('#viewport').screenshot(path=str(shot))
                for fmt, expected in numeric['sha256'].items():
                    with page.expect_download() as event:
                        page.locator(f'#exports a[href$="/{fmt}?download=1"]').click()
                    if digest(event.value.path()) != expected:
                        raise ValueError(f'{fmt} download changed')
                records.append(dict(generation=info['generation'], numeric=numeric, frames=numeric['frames'],
                                    measurements=stats, downloads_checked=len(numeric['sha256']),
                                    trace=traces.name, trace_sha256=digest(traces), screenshot=shot.name,
                                    screenshot_sha256=digest(shot), screenshot_frame=frame))
            browser.close()
        if errors:
            raise RuntimeError(f'Browser errors: {errors}')
        write_json(out / 'report.json', dict(schema='kimodo.v2.app-probe.1', diagnostic_only=True,
                   bones=BONES, vrm_sha256=digest(vrm), takes=records, browser_errors=errors))
        out.rename(evidence)


if __name__ == '__main__':
    from v1_export_check import DEFAULT_CHROME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('takes', type=Path, nargs='+')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--vrm', type=Path, required=True)
    parser.add_argument('--chrome', type=Path, default=DEFAULT_CHROME)
    args = parser.parse_args()
    run([p.resolve(strict=True) for p in args.takes], args.vrm.resolve(strict=True),
        args.chrome.resolve(strict=True), args.out.resolve())
    print(args.out / 'report.json')
