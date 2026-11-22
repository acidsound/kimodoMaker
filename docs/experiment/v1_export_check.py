"""Verify two existing Kimodo exports, then play/download them in a loopback app.

Use the backend venv (numpy/scipy/bvhio); browser mode also needs Playwright,
locally installed Chrome/Chromium and frontend node_modules. No model is loaded.
Only the disposable take store receives copies; the source take paths are read-only.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from http.server import ThreadingHTTPServer
import json
from itertools import combinations
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import urlopen
from uuid import uuid4

import bvhio
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
FORMATS = ("json", "bvh", "npz")
PELVIS_LIMIT_MM = 1.0
ROTATION_LIMIT_DEG = 0.1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def array(value, shape, label):
    raw = np.asarray(value)
    require(raw.shape == shape and raw.dtype.kind in "fiu" and np.isfinite(raw).all(),
            f"{label}: expected finite numbers with shape {shape}, got {raw.shape}")
    return raw.astype(np.float64, copy=False)


def max_distance(a, b):
    return float(np.linalg.norm(a - b, axis=-1).max())


def max_rotation_degrees(a, b):
    # Geodesic distance, not differences in Euler angles or signed quaternions.
    relative = (a @ np.swapaxes(b, -1, -2)).reshape(-1, 3, 3)
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude().max()))


def max_pair_distance(values):
    return max(max_distance(a, b) for a, b in combinations(values, 2))


def read_bvh(path, frames, fps, names, parents):
    body = path.read_text(encoding="utf-8")
    match = re.search(r"^MOTION\s*\n[ \t]*Frames:[ \t]*(\d+)[ \t]*\n"
                      r"[ \t]*Frame Time:[ \t]*(\S+)[ \t]*$", body, re.MULTILINE)
    require(match is not None, "BVH frame count/time header is missing")
    rows = body[match.end():].strip().splitlines()
    require(int(match[1]) == frames == len(rows), "BVH frame count differs from JSON or motion rows")
    header_time = float(match[2])
    require(math.isfinite(header_time) and header_time > 0, "Invalid BVH FPS (frame time)")
    parsed = bvhio.readAsBvh(str(path))
    # Older bvhio containers may not expose FrameTime; the text is authoritative.
    frame_time = getattr(parsed, "FrameTime", None)
    frame_time = header_time if frame_time is None else float(frame_time)
    require(math.isclose(frame_time, header_time, rel_tol=1e-7) and
            math.isclose(frame_time, 1 / fps, abs_tol=5e-7), "BVH/JSON FPS mismatch")
    require(parsed.FrameCount == frames, "BVH decoded frame count mismatch")
    wrapper = parsed.Root
    require(wrapper.Name == "Root" and len(wrapper.Children) == 1 and
            wrapper.Children[0].Name == "Hips", "Unexpected BVH wrapper/root hierarchy")
    nodes, bvh_parents = {}, {}

    def walk(node, parent):
        require(node.Name not in nodes, f"Duplicate BVH joint: {node.Name}")
        nodes[node.Name] = node
        bvh_parents[node.Name] = parent
        for child in node.Children:
            walk(child, node.Name)

    walk(wrapper, None)
    require(set(nodes) == {"Root", *names}, "BVH and JSON joint names differ")
    rot_channels = ["Zrotation", "Yrotation", "Xrotation"]
    pos_channels = ["Xposition", "Yposition", "Zposition"]
    for name, node in nodes.items():
        expected = pos_channels + rot_channels if name in ("Root", "Hips") else rot_channels
        require(list(node.Channels) == expected, f"Unexpected BVH channels at {name}")
        require(len(node.Keyframes) == frames, f"BVH keyframe count mismatch at {name}")
    width = sum(len(node.Channels) for node in nodes.values())
    require(all(len(row.split()) == width for row in rows), "BVH frame channel count mismatch")
    for name, parent in zip(names, parents):
        require(bvh_parents[name] == ("Root" if parent < 0 else names[parent]),
                f"BVH and JSON parents differ at {name}")

    def positions(node):
        return array([tuple(key.Position) for key in node.Keyframes], (frames, 3),
                     f"BVH {node.Name} translations") / 100

    def rotations(bones):
        quats = [[[key.Rotation.x, key.Rotation.y, key.Rotation.z, key.Rotation.w]
                   for key in bone.Keyframes] for bone in bones]
        raw = array(quats, (len(bones), frames, 4), "BVH rotations").transpose(1, 0, 2)
        return Rotation.from_quat(raw.reshape(-1, 4)).as_matrix().reshape(frames, len(bones), 3, 3)

    require(np.max(np.abs(positions(wrapper))) < 1e-7 and
            np.max(np.abs(rotations([wrapper]) - np.eye(3))) < 1e-7 and
            np.max(np.abs(tuple(wrapper.Offset))) < 1e-7,
            "BVH wrapper Root is not identity")
    # bvhio reads the XYZ position channels as the absolute Hips translation,
    # replacing (not adding) the BVH Hips rest offset. Both formats are +Y/+Z.
    offsets = array([tuple(nodes[name].Offset) for name in names], (77, 3), "BVH offsets") / 100
    return positions(nodes["Hips"]), rotations([nodes[name] for name in names]), offsets, 1 / frame_time


def forward_kinematics(roots, local, offsets, parents):
    world = local.copy()
    posed = np.empty(world.shape[:3], dtype=np.float64)
    posed[:, 0] = roots
    for i, parent in enumerate(parents[1:], 1):
        posed[:, i] = posed[:, parent] + world[:, parent] @ offsets[i]
        world[:, i] = world[:, parent] @ local[:, i]
    return posed, world


def digests(directory):
    return {fmt: sha256((directory / f"motion.{fmt}").read_bytes()).hexdigest() for fmt in FORMATS}


def verify_take(directory):
    """Decode all frames in all three serializations and compare in absolute metres."""
    motion = json.loads((directory / "motion.json").read_bytes())
    require(all(motion.get(k) == v for k, v in {
        "schema": "kimodo-maker.soma-motion.v1", "skeleton": "SOMA-77", "units": "m",
        "up": "+Y", "forward": "+Z", "quaternionOrder": "xyzw",
    }.items()), "Unexpected JSON SOMA-77 schema/axes/units")
    frames, fps = motion["frameCount"], motion["fps"]
    require(type(frames) is int and frames >= 2 and type(fps) in (int, float) and
            math.isfinite(fps) and fps > 0, "Invalid JSON frame count or FPS")
    joints = motion["joints"]
    names, parents = [j["name"] for j in joints], [j["parent"] for j in joints]
    require(len(names) == len(set(names)) == 77 and names[0] == "Hips" and
            all(isinstance(name, str) and name for name in names) and
            type(parents[0]) is int and parents[0] == -1 and
            all(type(p) is int and 0 <= p < i for i, p in enumerate(parents[1:], 1)),
            "Invalid JSON SOMA-77 joint hierarchy")
    json_offsets = array([j["restOffset"] for j in joints], (77, 3), "JSON offsets")
    json_hips = array(motion["rootPositions"], (frames, 3), "JSON pelvis")
    q = array(motion["localRotations"], (frames, 77 * 4), "JSON rotations").reshape(-1, 4)
    require(np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) < 1e-5, "Non-unit JSON rotations")
    shape = (frames, 77, 3, 3)
    json_rot = Rotation.from_quat(q).as_matrix().reshape(shape)
    bvh_hips, bvh_rot, bvh_offsets, bvh_fps = read_bvh(
        directory / "motion.bvh", frames, fps, names, parents)
    with np.load(directory / "motion.npz", allow_pickle=False) as data:
        posed = array(data["posed_joints"], (frames, 77, 3), "NPZ posed joints")
        npz_hips = array(data["root_positions"], (frames, 3), "NPZ pelvis")
        local = array(data["local_rot_mats"], shape, "NPZ local rotations")
        world = array(data["global_rot_mats"], shape, "NPZ world rotations")
        # The current Kimodo NPZ stores no timebase. Do not invent one for it.
        npz_fps = float(array(data["fps"], (), "NPZ FPS")) if "fps" in data else None
        require(npz_fps is None or math.isclose(npz_fps, fps, rel_tol=1e-6), "NPZ/JSON FPS mismatch")

    pelvis = {"json": json_hips, "bvh": bvh_hips, "npz_posed": posed[:, 0], "npz_root": npz_hips}
    pelvis_mm = {f"{a}/{b}": 1000 * max_distance(pelvis[a], pelvis[b])
                 for a, b in combinations(pelvis, 2)}
    pelvis_max = max(pelvis_mm.values())
    require(pelvis_max <= PELVIS_LIMIT_MM, f"Absolute pelvis differs by {pelvis_max:.6f} mm (> 1 mm)")
    # The JSON rest Hips is floor-derived; BVH uses a nominal offset that is
    # overridden by the position channels. Non-root rest positions must agree.
    require(max_distance(json_offsets[1:], bvh_offsets[1:]) <= 1e-5,
            "Non-root BVH/JSON rest offsets differ by > 0.01 mm")
    for label, rot in (("local", local), ("world", world)):
        require(np.max(np.abs(rot @ np.swapaxes(rot, -1, -2) - np.eye(3))) < 1e-4 and
                np.max(np.abs(np.linalg.det(rot) - 1)) < 1e-4, f"NPZ {label} rotations are invalid")
    npz_from_world = world.copy()
    npz_from_world[:, 1:] = np.swapaxes(world[:, parents[1:]], -1, -2) @ world[:, 1:]
    max_rotation = max(max_rotation_degrees(a, b) for a, b in
                       combinations((json_rot, bvh_rot, local, npz_from_world), 2))
    require(max_rotation <= ROTATION_LIMIT_DEG, f"Local rotation mismatch: {max_rotation:.6f} deg")
    json_fk, json_world = forward_kinematics(json_hips, json_rot, json_offsets, parents)
    bvh_fk, bvh_world = forward_kinematics(bvh_hips, bvh_rot, bvh_offsets, parents)
    fk_mm = 1000 * max_pair_distance((json_fk, bvh_fk, posed))
    require(fk_mm <= PELVIS_LIMIT_MM, f"World joint position mismatch: {fk_mm:.6f} mm")
    max_world_rotation = max(max_rotation_degrees(a, b) for a, b in
                             combinations((json_world, bvh_world, world), 2))
    require(max_world_rotation <= ROTATION_LIMIT_DEG,
            f"World rotation mismatch: {max_world_rotation:.6f} deg")
    return {"path": str(directory), "frames": frames, "joints": len(joints), "fps": fps,
            "bvh_fps": bvh_fps, "npz_fps": npz_fps, "pelvis_max_mm": pelvis_max,
            "pelvis_pairwise_mm": pelvis_mm, "all_joint_fk_max_mm": fk_mm,
            "local_rotation_max_deg": max_rotation, "world_rotation_max_deg": max_world_rotation,
            "sha256": digests(directory)}


@contextmanager
def isolated_api(sandbox, reports):
    """Run the real Maker read API on an OS-assigned port, in a fresh store."""
    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from server import TakeStore, handler_for
    finally:
        sys.path.pop(0)

    data = sandbox / "takes"
    data.mkdir(exist_ok=False)
    take_ids = []
    for index, report in enumerate(reports):
        source = Path(report["path"])
        take_id = uuid4().hex
        directory = data / take_id
        directory.mkdir()
        take_ids.append(take_id)
        for fmt in FORMATS:
            shutil.copyfile(source / f"motion.{fmt}", directory / f"motion.{fmt}")
        require(digests(directory) == report["sha256"], f"Source changed before staging: {source}")
        motion = json.loads((directory / "motion.json").read_bytes())
        info = motion.get("generation", {})
        take = dict(id=take_id, status="ready", order=index, fps=report["fps"],
                    frame_count=report["frames"], created=datetime.now(timezone.utc).isoformat(),
                    duration=report["frames"] / report["fps"], prompt=info.get("prompt") or source.name,
                    seed=info.get("seed"), steps=info.get("steps"), model=info.get("model"),
                    constraint_count=info.get("constraintCount", 0))
        (directory / "take.json").write_text(json.dumps(take, ensure_ascii=False), encoding="utf-8")

    # A read-only harness: no engine instance, no job threads, no production path.
    base_handler = handler_for(TakeStore(data, None))

    class ReadOnlyHandler(base_handler):
        def deny_write(self):
            self.send_json({"error": "Verification store is read-only"}, 405)

        do_POST = do_DELETE = deny_write

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), ReadOnlyHandler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", take_ids
        finally:
            server.shutdown()
            worker.join()


# Ignore the project's fixed-port proxy and .env files. Both actual servers use
# ephemeral loopback ports; Vite dependencies/cache write only to the sandbox.
VITE = """
import { createServer } from 'vite';
const [api, cacheDir] = process.argv.slice(1);
const server = await createServer({
  configFile: false, envFile: false, cacheDir,
  server: { host: '127.0.0.1', port: 0, strictPort: true, hmr: false,
            proxy: { '/api': api } },
});
await server.listen();
console.log('V1_EXPORT_URL=http://127.0.0.1:' + server.httpServer.address().port);
"""


@contextmanager
def isolated_frontend(sandbox, api, evidence):
    node = shutil.which("node")
    require(node is not None, "Node is not installed or on PATH")
    frontend = ROOT / "frontend"
    require((frontend / "node_modules" / "vite" / "package.json").is_file(),
            "Missing frontend dependencies; run npm ci in frontend before verifying")
    log = evidence / "vite.log"
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("VITE_")}
    with log.open("xb") as stream:
        proc = subprocess.Popen([node, "--input-type=module", "-e", VITE, api, str(sandbox / "vite")],
                                cwd=frontend, stdout=stream, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and proc.poll() is None:
                match = re.search(r"^V1_EXPORT_URL=(http://127\.0\.0\.1:[1-9]\d*)\s*$",
                                  log.read_text(encoding="utf-8", errors="replace"), re.MULTILINE)
                if match:
                    yield match[1]
                    return
                time.sleep(0.1)
            raise RuntimeError(f"Vite failed to start (exit {proc.poll()}); see {log}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


# Served instrumentation, not an on-disk application change. This appends one
# read-only reference to the actual Viewer created by main.js. All load,
# retargeting, draw, transport, and download behavior is the existing app's.
PROBE = "\nObject.defineProperty(globalThis, '__v1ExportViewer', { get: () => viewer });\n"
OBSERVE = """
() => {
  const v = globalThis.__v1ExportViewer;
  if (!v) throw new Error('Maker Viewer is unavailable (requires WebGL2)');
  const gl = v.renderer.getContext();
  const ext = gl.getExtension('WEBGL_debug_renderer_info');
  const names = ['hips', 'leftHand', 'rightHand', 'leftFoot', 'rightFoot', 'head'];
  const bones = names.map(name => {
    const node = v.vrm?.humanoid.getRawBoneNode(name);
    return node?.getWorldPosition(node.position.clone()).toArray() ?? null;
  });
  let skinnedMeshes = 0;
  v.vrm?.scene.traverse(node => { if (node.isSkinnedMesh && node.visible) skinnedMeshes++; });
  return {
    frame: v.frame, fps: v.motion?.fps, frames: v.motion?.frameCount, playing: v.playing,
    bones, skinnedMeshes, bindings: v.retarget?.links.length,
    triangles: v.renderer.info.render.triangles, renderFrame: v.renderer.info.render.frame,
    renderer: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null,
    contextLost: gl.isContextLost(),
  };
}
"""


def playback_delta(start, later):
    require(later["frame"] > start["frame"], "Maker timeline did not advance during playback")
    samples = []
    for obs in (start, later):
        bones = array(obs["bones"], (len(start["bones"]), 3), "VRM raw avatar bones")
        samples.append(bones[1:] - bones[0])
    delta_mm = 1000 * max_distance(*samples)
    require(delta_mm > 0.1, "VRM limbs did not move relative to the hips (>0.1 mm)")
    return delta_mm


def indicated_status(page):
    return page.evaluate("""() => Object.fromEntries(
      ['connection','avatar-status','preview-status','empty-stage'].map(id => {
        const node = document.getElementById(id);
        return [id, node && !node.hidden ? node.textContent : null];
      }))""")


def browser_checks(base, ids, reports, vrm, chrome, out, sandbox, result):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(chrome), headless=True, args=[
            "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
            "--disable-background-networking", "--no-first-run",
        ])
        try:
            result["chrome_version"] = browser.version
            context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 960})
            context.set_default_timeout(15_000)

            def local_only(route):
                url = route.request.url
                if not url.startswith(f"{base}/"):
                    route.abort()
                elif url.split("?", 1)[0] == f"{base}/src/main.js":
                    response = route.fetch(max_redirects=0)
                    route.fulfill(response=response, body=response.text() + PROBE)
                elif url == f"{base}/favicon.ico":
                    route.fulfill(status=204)
                else:
                    route.continue_()

            context.route("**/*", local_only)
            page = context.new_page()
            errors = result["page_errors"] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
            try:
                page.goto(base, wait_until="domcontentloaded")
                expect(page.locator("#connection")).to_have_attribute("data-state", "online")
                expect(page.locator("#take-list .take-item")).to_have_count(len(ids))
                expect(page.locator("#viewport canvas")).to_be_visible()
                observed = page.evaluate(OBSERVE)
                result["renderer"] = observed["renderer"]
                require(not observed["contextLost"] and "swiftshader" in (observed["renderer"] or "").lower(),
                        f"Expected WebGL2 software SwiftShader, got {observed['renderer']!r}")
                page.locator("#vrm-file").set_input_files(str(vrm))
                page.wait_for_function("""() => !document.querySelector('#avatar-name').hidden ||
                  document.querySelector('#avatar-status').dataset.error === 'true'""")
                status = indicated_status(page)
                require(not status["avatar-status"], f"Maker VRM load error: {status['avatar-status']}")
                expect(page.locator("#avatar-name")).to_have_text(vrm.name)
                # See the avatar alone rather than measuring the separate SOMA preview.
                page.locator("#skeleton-visible").uncheck()
                page.locator("#loop").uncheck()

                for index, (take_id, report) in enumerate(zip(ids, reports), 1):
                    run = {"id": take_id, "take": report["path"], "downloads": {}}
                    result["takes"].append(run)
                    row = page.locator(f'[data-take-id="{take_id}"] .take-main')
                    row.click()
                    expect(row).to_have_attribute("aria-pressed", "true")
                    expect(page.locator("#take-info")).to_contain_text(take_id[:8])
                    expect(page.locator("#timeline")).to_have_attribute("max", str(report["frames"] - 1))
                    expect(page.locator("#play")).to_be_enabled()
                    page.wait_for_function("""() => {
                      const v = globalThis.__v1ExportViewer;
                      return v?.retarget && v.renderer.info.render.triangles > 0;
                    }""")
                    start = page.evaluate(OBSERVE)
                    require(start["frame"] == 0 and start["frames"] == report["frames"] and
                            start["fps"] == report["fps"] and start["skinnedMeshes"] > 0 and
                            start["bindings"] >= 15 and not start["contextLost"],
                            f"Maker did not bind/render the selected take at frame 0: {start}")
                    page.locator("#play").click()
                    page.wait_for_function("target => +document.querySelector('#timeline').value >= target",
                                           arg=min(12, report["frames"] - 1))
                    later = page.evaluate(OBSERVE)
                    require(later["renderFrame"] > start["renderFrame"] and not later["contextLost"] and
                            later["triangles"] > 0, "No rendered avatar frame during playback")
                    run.update(played_from=start["frame"], played_to=later["frame"],
                               raw_vrm_limb_motion_mm=playback_delta(start, later),
                               skin_meshes=later["skinnedMeshes"], triangles=later["triangles"],
                               bound_bones=later["bindings"])
                    if later["playing"]:
                        page.locator("#play").click()
                    expect(page.locator("#play")).to_have_text("\uc7ac\uc0dd")
                    # Actual range control keyboard input (not a direct call to Viewer.seek).
                    timeline = page.locator("#timeline")
                    timeline.press("End")
                    expect(timeline).to_have_value(str(report["frames"] - 1))
                    expect(page.locator("#timecode")).to_have_text(
                        f"{(report['frames'] - 1) / report['fps']:.2f} s")
                    require(page.evaluate(OBSERVE)["frame"] == report["frames"] - 1,
                            "Maker did not seek to the last frame")
                    timeline.press("ArrowLeft")
                    screenshot = out / f"take-{index}-vrm.png"
                    png = page.locator("#viewport").screenshot(path=str(screenshot), timeout=15_000)
                    run["screenshot"] = str(screenshot)
                    run["screenshot_sha256"] = sha256(png).hexdigest()

                    for fmt in FORMATS:
                        link = page.locator(f'#exports a[href="/api/takes/{take_id}/{fmt}?download=1"]')
                        with page.expect_download() as event:
                            link.click()
                        download = event.value
                        filename = f"{take_id[:8]}-motion.{fmt}"
                        require(download.url == f"{base}/api/takes/{take_id}/{fmt}?download=1" and
                                download.suggested_filename == filename,
                                f"Unexpected {fmt} download: {download.url} ({download.suggested_filename})")
                        path = sandbox / filename
                        download.save_as(str(path))
                        payload = path.read_bytes()
                        digest = sha256(payload).hexdigest()
                        require(digest == report["sha256"][fmt], f"Browser download differs: {filename}")
                        run["downloads"][fmt] = {"bytes": len(payload), "sha256": digest}
                require(not errors, "Browser page/console errors: " + "; ".join(errors))
            except Exception:
                if not page.is_closed():
                    try:
                        result["ui_status"] = indicated_status(page)
                        failure = out / "browser-failure.png"
                        page.screenshot(path=str(failure), timeout=5_000)
                        result["failure_screenshot"] = str(failure)
                    except Exception:
                        pass  # Preserve the original exact browser failure.
                raise
        finally:
            browser.close()


SEED_SAN_SHA256 = "624d0d554bc205bbdc33e22a68a2c3c20edebb3e573011ead8878a65e5329b23"
SEED_SAN_COMMIT = "94e82dd346fa6cf0337c4421728640e5252dd38e"
DEFAULT_CHROME = Path("C:/Program Files/Google/Chrome/Application/chrome.exe")


def verify_browser(reports, vrm, chrome, evidence, result):
    require(vrm.is_file(), f"VRM file not found: {vrm}")
    require(chrome.is_file(), f"Chrome executable not found: {chrome}; pass --chrome")
    require(not evidence.is_relative_to(ROOT) and
            not any(evidence.is_relative_to(Path(report["path"])) for report in reports),
            "Evidence must be outside the repository and the input take directories")
    evidence.mkdir(exist_ok=False)
    digest = sha256(vrm.read_bytes()).hexdigest()
    result["vrm"] = {"path": str(vrm), "sha256": digest}
    if digest == SEED_SAN_SHA256:
        result["vrm"].update(name="Seed-san", author="VirtualCast, Inc.",
                             license="VRM Public License 1.0", source_commit=SEED_SAN_COMMIT)
    # No data store, downloads, browser profile, or model copied into the repo.
    with tempfile.TemporaryDirectory(prefix="sandbox-", dir=evidence) as tmp:
        sandbox = Path(tmp)
        with isolated_api(sandbox, reports) as (api, ids), isolated_frontend(sandbox, api, evidence) as base:
            result.update(api=api, frontend=base, takes=[])
            with urlopen(api + "/api/takes", timeout=5) as response:
                entries = json.load(response)
            require([entry["id"] for entry in entries] == ids and
                    all(entry["status"] == "ready" for entry in entries), "Isolated take listing mismatch")
            browser_checks(base, ids, reports, vrm, chrome, evidence, sandbox, result)
            with urlopen(api + "/api/takes", timeout=5) as response:
                require(json.load(response) == entries, "Sandbox take listing changed during playback")
    for report in reports:
        require(digests(Path(report["path"])) == report["sha256"], f"Source changed: {report['path']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("takes", type=Path, nargs=2, metavar="TAKE_DIR")
    parser.add_argument("--vrm", type=Path, help="local VRM test model; no download or redistribution")
    parser.add_argument("--evidence-dir", type=Path, help="new output directory outside the repo/inputs")
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME if DEFAULT_CHROME.exists() else
                        Path(shutil.which("google-chrome") or shutil.which("chromium") or "chrome.exe"))
    parser.add_argument("--numeric-only", action="store_true", help="do not start the app or claim playback/downloads")
    args = parser.parse_args()
    if not args.numeric_only and not (args.vrm and args.evidence_dir):
        parser.error("provide --vrm and --evidence-dir, or use --numeric-only")

    browser = {"status": "skipped" if args.numeric_only else "not_run"}
    result = {"ok": False, "numeric": {"status": "failed", "takes": []}, "browser": browser}
    reports = result["numeric"]["takes"]
    try:
        inputs = [path.resolve(strict=True) for path in args.takes]
        require(len(set(inputs)) == 2, "Provide two distinct take directories")
        for path in inputs:
            try:
                reports.append(verify_take(path))
            except Exception as exc:
                raise ValueError(f"{path}: {exc}") from exc
        require(all(reports[0]["sha256"][fmt] != reports[1]["sha256"][fmt] for fmt in FORMATS),
                "The take artifacts are not two distinct generated motions")
        result["numeric"]["status"] = "passed"
        if not args.numeric_only:
            browser["status"] = "failed"
            verify_browser(reports, args.vrm.resolve(), args.chrome.resolve(), args.evidence_dir.resolve(), browser)
            browser["status"] = "passed"
        result["ok"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
