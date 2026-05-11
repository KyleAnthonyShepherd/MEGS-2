"""Stdlib HTTP control server for continuous_train.py.

Runs in a daemon thread; shares a ControlState with the trainer.
All requests and responses are JSON.

Endpoints:
    POST /ingest        body: {"snapshot_dir": "...", "request_id": "..."}
                        → 202 {"request_id": "..."}
    POST /checkpoint    → 200 {"path": "..."}  (blocks until write completes)
    GET  /status        → 200 {"gaussian_count":N, "monitor_state":"...", ...}
    GET  /status/{id}   → 200 {"state": "queued|integrating|training|converged|unknown"}
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional


class _Handler(BaseHTTPRequestHandler):
    # ctrl and checkpoint_timeout are injected by the factory below
    ctrl = None
    checkpoint_timeout: float = 30.0

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
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def do_POST(self):
        if self.path == "/ingest":
            body = self._read_json()
            if body is None:
                self._send_json(400, {"error": "invalid JSON"})
                return
            snap_dir = body.get("snapshot_dir")
            if not snap_dir:
                self._send_json(400, {"error": "snapshot_dir required"})
                return
            rid = self.ctrl.enqueue_ingest(snap_dir, body.get("request_id"))
            self._send_json(202, {"request_id": rid})

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
        if self.path == "/status":
            self._send_json(200, self.ctrl.status_snapshot())

        elif self.path.startswith("/status/"):
            rid = self.path[len("/status/"):]
            state = self.ctrl.get_request_state(rid)
            self._send_json(200, {"state": state})

        else:
            self._send_json(404, {"error": "not found"})


def start_server(host: str, port: int, ctrl, checkpoint_timeout: float = 30.0):
    """Spin up ThreadingHTTPServer in a daemon thread; returns immediately."""

    # Build a handler class with the shared state baked in
    handler = type("Handler", (_Handler,), {
        "ctrl": ctrl,
        "checkpoint_timeout": checkpoint_timeout,
    })

    server = ThreadingHTTPServer((host, port), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
