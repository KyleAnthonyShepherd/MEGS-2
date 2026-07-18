"""Tests for scene/control_state.py + scene/continuous_server.py.

Covers the home-server ingest contract (app/api/trainer.py on the
home-server side): payload shapes, idempotency, session pinning, /health
keys, and snapshot-dir resolution/validation.
"""

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tests._direct_import import load_scene_module

control_state = load_scene_module("control_state")
continuous_server = load_scene_module("continuous_server")

ControlState = control_state.ControlState
resolve_snapshot_dir = continuous_server.resolve_snapshot_dir
start_server = continuous_server.start_server


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def snapshot_root(tmp_path):
    """A minimal valid snapshot layout: images/ + sparse/0/cameras.bin."""
    (tmp_path / "images").mkdir()
    sparse0 = tmp_path / "sparse" / "0"
    sparse0.mkdir(parents=True)
    (sparse0 / "cameras.bin").write_bytes(b"\x00")
    return tmp_path


@pytest.fixture
def server(snapshot_root):
    ctrl = ControlState()
    srv = start_server("127.0.0.1", 0, ctrl, checkpoint_timeout=0.2)
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    yield ctrl, base
    srv.shutdown()


def post(base, path, payload):
    req = urllib.request.Request(
        base + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def get(base, path):
    try:
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


# ---------------------------------------------------------------------------
# resolve_snapshot_dir
# ---------------------------------------------------------------------------

def test_resolve_from_snapshot_dir(snapshot_root):
    d, err = resolve_snapshot_dir({"snapshot_dir": str(snapshot_root)})
    assert err is None and d == str(snapshot_root)


def test_resolve_from_sparse_dir(snapshot_root):
    d, err = resolve_snapshot_dir(
        {"sparse_dir": str(snapshot_root / "sparse" / "0")})
    assert err is None and d == str(snapshot_root)


def test_resolve_rejects_malformed_sparse_dir(snapshot_root):
    d, err = resolve_snapshot_dir({"sparse_dir": str(snapshot_root / "images")})
    assert d is None and "sparse/<N>" in err


def test_resolve_rejects_missing_reconstruction(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    d, err = resolve_snapshot_dir({"snapshot_dir": str(tmp_path)})
    assert d is None and "no COLMAP reconstruction" in err


def test_resolve_rejects_missing_images_dir(tmp_path):
    sparse0 = tmp_path / "sparse" / "0"
    sparse0.mkdir(parents=True)
    (sparse0 / "cameras.bin").write_bytes(b"\x00")
    d, err = resolve_snapshot_dir({"snapshot_dir": str(tmp_path)})
    assert d is None and "images/" in err


def test_resolve_requires_some_dir():
    d, err = resolve_snapshot_dir({})
    assert d is None and "required" in err


def test_resolve_accepts_text_reconstruction(tmp_path):
    (tmp_path / "images").mkdir()
    sparse0 = tmp_path / "sparse" / "0"
    sparse0.mkdir(parents=True)
    (sparse0 / "cameras.txt").write_text("")
    d, err = resolve_snapshot_dir({"snapshot_dir": str(tmp_path)})
    assert err is None and d == str(tmp_path)


# ---------------------------------------------------------------------------
# ControlState idempotency + session pinning
# ---------------------------------------------------------------------------

def test_enqueue_dedups_on_session_and_image():
    ctrl = ControlState()
    rid1, dup1 = ctrl.enqueue_ingest("/snap", session_id="s1", image_name="img_1.jpg")
    rid2, dup2 = ctrl.enqueue_ingest("/snap", session_id="s1", image_name="img_1.jpg")
    assert not dup1 and dup2
    assert rid1 == rid2
    assert len(ctrl.drain_ingest_queue()) == 1


def test_enqueue_without_identity_never_dedups():
    ctrl = ControlState()
    _, dup1 = ctrl.enqueue_ingest("/snap")
    _, dup2 = ctrl.enqueue_ingest("/snap")
    assert not dup1 and not dup2
    assert len(ctrl.drain_ingest_queue()) == 2


def test_first_tagged_ingest_pins_session():
    ctrl = ControlState()
    ctrl.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert ctrl.session_id == "s1"


def test_request_state_lifecycle():
    ctrl = ControlState()
    rid, _ = ctrl.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert ctrl.get_request_state(rid) == "queued"
    ctrl.set_request_state(rid, "training")
    assert ctrl.get_request_state(rid) == "training"
    assert ctrl.get_request_state("nope") == "unknown"


def test_wait_for_work_wakes_on_enqueue():
    ctrl = ControlState()
    woke = threading.Event()

    def waiter():
        ctrl.wait_for_work(timeout=10)
        woke.set()

    t = threading.Thread(target=waiter)
    t.start()
    ctrl.enqueue_ingest("/snap")
    t.join(timeout=5)
    assert woke.is_set()


def test_health_snapshot_contract_keys():
    """Keys the home-server's get_trainer_health() consumer may read."""
    ctrl = ControlState()
    h = ctrl.health_snapshot()
    for key in ("state", "iter", "splat_count", "queue_depth",
                "last_ingest", "vram_mb"):
        assert key in h
    assert h["state"] == "initializing"
    assert h["last_ingest"] is None


def test_update_status_tracks_iteration():
    ctrl = ControlState()
    ctrl.update_status(10, "improving", 5, True, iteration=123)
    assert ctrl.status_snapshot()["iter"] == 123
    assert ctrl.health_snapshot()["iter"] == 123


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

def test_ingest_home_server_payload(server, snapshot_root):
    ctrl, base = server
    payload = {
        "session_id": "sess1",
        "image_name": "img_0001.jpg",
        "image_path": str(snapshot_root / "images" / "img_0001.jpg"),
        "sparse_dir": str(snapshot_root / "sparse" / "0"),
    }
    status, body = post(base, "/ingest", payload)
    assert status == 202 and "request_id" in body
    items = ctrl.drain_ingest_queue()
    assert len(items) == 1
    assert items[0].snapshot_dir == str(snapshot_root)
    assert items[0].session_id == "sess1"
    assert items[0].image_name == "img_0001.jpg"


def test_ingest_duplicate_is_200_noop(server, snapshot_root):
    ctrl, base = server
    payload = {
        "session_id": "sess1", "image_name": "img_0001.jpg",
        "sparse_dir": str(snapshot_root / "sparse" / "0"),
    }
    s1, b1 = post(base, "/ingest", payload)
    s2, b2 = post(base, "/ingest", payload)
    assert s1 == 202 and s2 == 200
    assert b2["duplicate"] is True
    assert b1["request_id"] == b2["request_id"]
    assert len(ctrl.drain_ingest_queue()) == 1


def test_ingest_legacy_payload(server, snapshot_root):
    ctrl, base = server
    status, body = post(base, "/ingest", {"snapshot_dir": str(snapshot_root),
                                          "request_id": "rid-42"})
    assert status == 202 and body["request_id"] == "rid-42"


def test_ingest_session_mismatch_409(server, snapshot_root):
    ctrl, base = server
    payload = {"session_id": "sess1", "image_name": "a.jpg",
               "sparse_dir": str(snapshot_root / "sparse" / "0")}
    assert post(base, "/ingest", payload)[0] == 202
    payload2 = dict(payload, session_id="sess2")
    status, body = post(base, "/ingest", payload2)
    assert status == 409 and body["session_id"] == "sess1"


def test_ingest_validation_errors(server, snapshot_root):
    ctrl, base = server
    assert post(base, "/ingest", {})[0] == 400
    assert post(base, "/ingest", {"sparse_dir": "/nonexistent/sparse/0"})[0] == 400
    # non-dict JSON
    req = urllib.request.Request(
        base + "/ingest", data=b"[1,2]",
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 400


def test_health_endpoint(server, snapshot_root):
    ctrl, base = server
    ctrl.update_status(1234, "improving", 7, True, iteration=99)
    status, body = get(base, "/health")
    assert status == 200
    assert body["state"] == "improving"
    assert body["splat_count"] == 1234
    assert body["iter"] == 99
    assert body["queue_depth"] == 0


def test_status_and_per_request_status(server, snapshot_root):
    ctrl, base = server
    payload = {"session_id": "s", "image_name": "a.jpg",
               "sparse_dir": str(snapshot_root / "sparse" / "0")}
    _, body = post(base, "/ingest", payload)
    rid = body["request_id"]
    assert get(base, f"/status/{rid}")[1]["state"] == "queued"
    status, snap = get(base, "/status")
    assert status == 200 and snap["queue_depth"] == 1
    assert snap["session_id"] == "s"


def test_checkpoint_times_out_without_trainer(server):
    ctrl, base = server
    status, body = post(base, "/checkpoint", {})
    assert status == 504


def test_unknown_routes_404(server):
    ctrl, base = server
    assert get(base, "/nope")[0] == 404
    assert post(base, "/nope", {})[0] == 404
