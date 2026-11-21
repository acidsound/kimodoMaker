"""Serve the editor and serialize generation requests on the local GPU."""

import json
import logging
import secrets
import shutil
import threading
import uuid
from argparse import ArgumentParser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from mimetypes import guess_type
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from engine import MODEL, KimodoEngine

ROOT = Path(__file__).resolve().parent
MAX_REQUEST_BYTES = 2 * 1024 * 1024
FORMATS = {"json", "bvh", "npz"}
PENDING = {"queued", "loading", "generating", "exporting"}
LOG = logging.getLogger(__name__)


def validate_options(data):
    if not isinstance(data, dict):
        raise ValueError("요청 본문은 JSON 객체여야 합니다.")
    prompt = data.get("prompt")
    duration = data.get("duration")
    seed = data.get("seed")
    steps = data.get("steps", 100)
    constraints = data.get("constraints", [])
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1000:
        raise ValueError("프롬프트를 1~1000자로 입력하세요.")
    if type(duration) not in (int, float) or not 0.5 <= duration <= 15:
        raise ValueError("길이는 0.5~15초로 입력하세요.")
    if seed is None:
        seed = secrets.randbits(32)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("시드는 0~4294967295 정수여야 합니다.")
    if type(steps) is not int or not 10 <= steps <= 100:
        raise ValueError("추론 스텝은 10~100 정수여야 합니다.")
    if not isinstance(constraints, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("type"), str)
        for item in constraints
    ):
        raise ValueError("제약 파일은 Kimodo constraints.json 형식이어야 합니다.")
    return dict(prompt=prompt.strip(), duration=duration, seed=seed, steps=steps, constraints=constraints)


class TakeStore:
    def __init__(self, directory, engine):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.lock = threading.Lock()
        self.takes = {}
        for path in directory.glob("*/take.json"):
            try:
                take = json.loads(path.read_text(encoding="utf-8"))
                if take["id"] != path.parent.name:
                    continue
                if take["status"] in PENDING:
                    take.update(status="failed", error="서버가 종료되어 생성이 중단됐습니다.")
                    self.save(take)
                self.takes[take["id"]] = take
            except (OSError, ValueError, KeyError, TypeError):
                LOG.warning("Cannot read take metadata at %s", path)

    def save(self, take):
        path = self.directory / take["id"] / "take.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(take, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def list(self):
        with self.lock:
            return sorted(self.takes.values(), key=lambda take: take["created"], reverse=True)

    def get(self, take_id):
        with self.lock:
            return self.takes.get(take_id)

    def start(self, options):
        with self.lock:
            if any(take["status"] in PENDING for take in self.takes.values()):
                return None
            take_id = uuid.uuid4().hex
            directory = self.directory / take_id
            directory.mkdir()
            take = {key: value for key, value in options.items() if key != "constraints"}
            take.update(
                id=take_id,
                status="queued",
                created=datetime.now(timezone.utc).isoformat(),
                model=MODEL,
                constraint_count=len(options["constraints"]),
            )
            try:
                if options["constraints"]:
                    (directory / "constraints.json").write_text(
                        json.dumps(options["constraints"], ensure_ascii=False), encoding="utf-8"
                    )
                self.save(take)
            except OSError:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            self.takes[take_id] = take
            threading.Thread(target=self.run, args=(take_id, options), daemon=True).start()
            return take

    def run(self, take_id, options):
        def update(status, **metadata):
            with self.lock:
                take = dict(self.takes[take_id], status=status, **metadata)
                self.takes[take_id] = take
                try:
                    self.save(take)
                except OSError:
                    LOG.exception("Cannot save take %s", take_id)

        try:
            directory = self.directory / take_id
            result = self.engine.generate(options, directory, update)
            if not all((directory / f"motion.{ext}").stat().st_size for ext in FORMATS):
                raise RuntimeError("내보낼 파일이 비어 있습니다.")
            update("ready", **result)
        except Exception as error:
            LOG.exception("Generation failed for %s", take_id)
            update("failed", error=f"{type(error).__name__}: {error}"[:1200])


def handler_for(store, static):
    class Handler(BaseHTTPRequestHandler):
        def local_host(self):
            host = self.headers.get("Host", "").partition(":")[0].lower()
            if host in {"localhost", "127.0.0.1"}:
                return True
            self.send_json({"error": "로컬 호스트에서 연결하세요."}, 403)
            return False

        def send_json(self, data, status=200):
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_file(self, path, attachment=None):
            try:
                handle = path.open("rb")
            except OSError:
                self.send_json({"error": "파일을 찾을 수 없습니다."}, 404)
                return
            with handle:
                self.send_response(200)
                self.send_header("Content-Type", guess_type(path)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(path.stat().st_size))
                self.send_header("X-Content-Type-Options", "nosniff")
                if attachment:
                    self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
                self.end_headers()
                shutil.copyfileobj(handle, self.wfile)

        def do_GET(self):
            if not self.local_host():
                return
            url = urlsplit(self.path)
            if url.path == "/api/takes":
                self.send_json(store.list())
                return
            parts = url.path.strip("/").split("/")
            if parts[:2] == ["api", "takes"] and len(parts) in (3, 4):
                take_id = parts[2]
                take = store.get(take_id)
                if take and len(parts) == 3:
                    self.send_json(take)
                    return
                if take and len(parts) == 4:
                    fmt = parts[3]
                    if fmt == "constraints" or (take["status"] == "ready" and fmt in FORMATS):
                        filename = "constraints.json" if fmt == "constraints" else f"motion.{fmt}"
                        download = parse_qs(url.query).get("download") == ["1"]
                        self.send_file(
                            store.directory / take_id / filename,
                            f"{take_id[:8]}-{filename}" if download else None,
                        )
                        return
            if url.path.startswith("/api/"):
                self.send_json({"error": "테이크나 파일을 찾을 수 없습니다."}, 404)
                return
            path = (static / (url.path.strip("/") or "index.html")).resolve()
            if path.is_relative_to(static):
                self.send_file(path)
            else:
                self.send_json({"error": "파일을 찾을 수 없습니다."}, 404)

        def do_POST(self):
            if not self.local_host():
                return
            if self.path != "/api/takes":
                self.send_json({"error": "경로를 찾을 수 없습니다."}, 404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_json({"error": "application/json 요청이 필요합니다."}, 415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST_BYTES:
                    raise ValueError("JSON 요청은 최대 2 MB까지 허용합니다.")
                data = json.loads(self.rfile.read(size), parse_constant=reject_constant)
                options = validate_options(data)
                take = store.start(options)
            except (ValueError, RecursionError) as error:
                self.send_json({"error": str(error)}, 400)
                return
            except OSError:
                LOG.exception("Cannot create take")
                self.send_json({"error": "테이크를 저장할 수 없습니다. 서버 로그를 확인하세요."}, 500)
                return
            if take is None:
                self.send_json({"error": "다른 테이크를 생성하고 있습니다."}, 409)
                return
            self.send_json(take, 202)

    return Handler


def reject_constant(value):
    raise ValueError(f"JSON 숫자 {value}는 지원하지 않습니다.")


def main():
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "takes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    store = TakeStore(args.data_dir.resolve(), KimodoEngine())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(store, (ROOT / "dist").resolve())) as server:
        print(f"Kimodo Maker · http://localhost:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
