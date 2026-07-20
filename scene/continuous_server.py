"""Stdlib HTTP control server for continuous_train.py.

Runs in a daemon thread; shares a ControlState with the trainer.
All requests and responses are JSON.

Endpoints:
    POST /ingest        Two accepted payload shapes:
                        1. Home-server contract (app/api/trainer.py):
                           {"session_id": str, "image_name": str,
                            "image_path": str, "sparse_dir": str}
                           snapshot_dir is derived from sparse_dir (which must
                           end in sparse/<N>); idempotent on
                           (session_id, image_name) — a duplicate POST is a
                           200 no-op reporting the original request_id.
                        2. Legacy: {"snapshot_dir": "...", "request_id": "..."}
                        → 202 {"request_id": "..."} (200 on duplicate)
    POST /pause         → 200 {"paused": true, "vram_mb": <post-move MiB>}
                        Blocks (up to pause_timeout) until the trainer has
                        moved its model to CPU and freed VRAM. Idempotent: a
                        second /pause while already paused is a fast no-op
                        reporting the current vram_mb.
    POST /resume        → 200 {"paused": false}
                        Moves the model back to GPU; training continues from
                        the exact paused state. Idempotent.
    POST /checkpoint    → 200 {"path": "..."}  (blocks until write completes)
    GET  /health        → 200 {"state","iter","splat_count","queue_depth",
                                "last_ingest","vram_mb","paused", ...}
    GET  /status        → 200 {"gaussian_count":N, "monitor_state":"...", ...}
    GET  /status/{id}   → 200 {"state": "queued|integrating|training|converged|unknown"}
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple


def resolve_snapshot_dir(body: dict) -> Tuple[Optional[str], Optional[str]]:
    """Derive the snapshot directory from an ingest payload.

    Returns (snapshot_dir, error). Exactly one is non-None.

    The home-server posts sparse_dir = <session_root>/sparse/<N>; the
    snapshot root the trainer loads (images/ + sparse/0/) is <session_root>.
    """
    snap_dir = body.get("snapshot_dir")
    sparse_dir = body.get("sparse_dir")

    if not snap_dir and not sparse_dir:
        return None, "snapshot_dir or sparse_dir required"

    if not snap_dir:
        sp = Path(sparse_dir)
        # Expect .../<session_root>/sparse/<N>
        if sp.parent.name != "sparse":
            return None, (
                f"sparse_dir must end in sparse/<N>, got {sparse_dir!r}")
        snap_dir = str(sp.parent.parent)

    root = Path(snap_dir)
    sparse0 = root / "sparse" / "0"
    if not any((sparse0 / f"cameras{ext}").exists() for ext in (".bin", ".txt")):
        return None, f"no COLMAP reconstruction at {sparse0}"
    if not (root / "images").is_dir():
        return None, f"no images/ directory under {snap_dir}"
    return snap_dir, None


class _Handler(BaseHTTPRequestHandler):
    # ctrl, checkpoint_timeout and pause_timeout are injected by the factory
    ctrl = None
    checkpoint_timeout: float = 30.0
    pause_timeout: float = 30.0

    def log_message(self, fmt, *args):
        pass  # suppress default stderr logging

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return body if isinstance(body, dict) else None

    def do_POST(self):
        if self.path == "/ingest":
            body = self._read_json()
            if body is None:
                self._send_json(400, {"error": "invalid JSON"})
                return

            session_id = body.get("session_id")
            image_name = body.get("image_name")

            # One trainer process serves one session; reject cross-session
            # ingests loudly rather than silently mixing scenes.
            pinned = self.ctrl.session_id
            if session_id and pinned and session_id != pinned:
                self._send_json(409, {
                    "error": f"trainer is bound to session {pinned!r}",
                    "session_id": pinned,
                })
                return

            snap_dir, err = resolve_snapshot_dir(body)
            if err:
                self._send_json(400, {"error": err})
                return

            rid, duplicate = self.ctrl.enqueue_ingest(
                snap_dir, body.get("request_id"),
                session_id=session_id, image_name=image_name,
            )
            if duplicate:
                self._send_json(200, {"request_id": rid, "duplicate": True})
            else:
                self._send_json(202, {"request_id": rid})

        elif self.path == "/pause":
            # Idempotent: if already paused, report the current VRAM without
            # disturbing the (parked) trainer loop — the server re-pauses each
            # job in a backlog.
            if self.ctrl.is_paused():
                self._send_json(200, {
                    "paused": True, "vram_mb": self.ctrl.current_vram_mb()})
                return
            self.ctrl.request_pause()
            ok = self.ctrl.wait_until_paused(timeout=self.pause_timeout)
            if ok:
                self._send_json(200, {
                    "paused": True, "vram_mb": self.ctrl.current_vram_mb()})
            else:
                # Trainer didn't ack in time (e.g. mid-ingest on a huge scene).
                # Report not-paused so the caller can decide; it proceeds with
                # COLMAP regardless per the home-server's degrade-to-no-op rule.
                self._send_json(504, {"error": "pause timed out", "paused": False})

        elif self.path == "/resume":
            self.ctrl.request_resume()
            self.ctrl.wait_until_resumed(timeout=self.pause_timeout)
            self._send_json(200, {"paused": False})

        elif self.path == "/checkpoint":
            self.ctrl.request_checkpoint()
            path = self.ctrl.wait_for_checkpoint(timeout=self.checkpoint_timeout)
            if path is None:
                self._send_json(504, {"error": "checkpoint timed out"})
            else:
                self._send_json(200, {"path": path})

        else:
            self._send_json(404, {"error": "not found"})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, self.ctrl.health_snapshot())

        elif self.path == "/status":
            self._send_json(200, self.ctrl.status_snapshot())

        elif self.path.startswith("/status/"):
            rid = self.path[len("/status/"):]
            state = self.ctrl.get_request_state(rid)
            self._send_json(200, {"state": state})

        else:
            self._send_json(404, {"error": "not found"})


def start_server(host: str, port: int, ctrl, checkpoint_timeout: float = 30.0,
                 pause_timeout: float = 30.0):
    """Spin up ThreadingHTTPServer in a daemon thread; returns immediately."""

    # Build a handler class with the shared state baked in
    handler = type("Handler", (_Handler,), {
        "ctrl": ctrl,
        "checkpoint_timeout": checkpoint_timeout,
        "pause_timeout": pause_timeout,
    })

    server = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
