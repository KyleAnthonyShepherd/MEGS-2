"""Tests for the GPU-serialization pause/resume path (MEGS-2 ingest contract
§2/§3): the ControlState handshake, the /pause + /resume HTTP endpoints, the
`paused` health key, and — when a GPU is present — the model-level
to_cpu()/to_cuda() round trip (Adam binding + bitwise equality + loss keeps
falling).

The pure-Python handshake/endpoint tests run everywhere (they load the
control/server modules by file path, no torch). The model round trip needs
CUDA (simple_knn, the cohort tensors live on the GPU) and skips without it.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from tests._direct_import import load_scene_module

control_state = load_scene_module("control_state")
continuous_server = load_scene_module("continuous_server")

ControlState = control_state.ControlState
start_server = continuous_server.start_server


# ---------------------------------------------------------------------------
# HTTP helpers (mirrors tests/test_continuous_server.py)
# ---------------------------------------------------------------------------

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


def _fake_trainer_loop(ctrl, stop, vram_mb=42.0):
    """Stand-in for continuous_training's pause handling: on a pause request,
    ack via mark_paused, park until resume, then mark_resumed."""
    while not stop.is_set():
        if ctrl.pause_requested():
            ctrl.mark_paused(vram_mb)
            while not stop.is_set() and not ctrl.wait_for_resume(timeout=0.02):
                pass
            if stop.is_set():
                break
            ctrl.mark_resumed()
        else:
            time.sleep(0.005)


# ---------------------------------------------------------------------------
# ControlState handshake (no torch)
# ---------------------------------------------------------------------------

def test_pause_resume_flag_transitions():
    ctrl = ControlState()
    assert not ctrl.is_paused()
    assert not ctrl.pause_requested()

    ctrl.request_pause()
    assert ctrl.pause_requested()           # trainer should pick this up
    assert not ctrl.is_paused()

    ctrl.mark_paused(12.5)
    assert ctrl.is_paused()
    assert ctrl.current_vram_mb() == 12.5
    assert not ctrl.pause_requested()       # consumed by the mark

    # A repeated pause while already paused must NOT re-trigger a pause.
    ctrl.request_pause()
    assert not ctrl.pause_requested()

    ctrl.request_resume()
    assert ctrl.wait_for_resume(timeout=0.5)
    ctrl.mark_resumed()
    assert not ctrl.is_paused()
    assert ctrl.current_vram_mb() is None


def test_health_and_status_report_paused():
    ctrl = ControlState()
    assert ctrl.health_snapshot()["paused"] is False
    assert ctrl.status_snapshot()["paused"] is False
    ctrl.request_pause()
    ctrl.mark_paused(5.0)
    assert ctrl.health_snapshot()["paused"] is True
    assert ctrl.status_snapshot()["paused"] is True


def test_wait_for_work_wakes_on_pause_request():
    ctrl = ControlState()
    woke = threading.Event()

    def waiter():
        ctrl.wait_for_work(timeout=10)
        woke.set()

    t = threading.Thread(target=waiter)
    t.start()
    ctrl.request_pause()
    t.join(timeout=5)
    assert woke.is_set()


def test_request_pause_clears_pending_resume_and_vice_versa():
    ctrl = ControlState()
    ctrl.request_resume()
    ctrl.request_pause()
    # A pending resume must not survive a fresh pause request.
    assert not ctrl.wait_for_resume(timeout=0.05)
    ctrl.request_resume()
    assert ctrl.wait_for_resume(timeout=0.5)


# ---------------------------------------------------------------------------
# /pause + /resume HTTP endpoints
# ---------------------------------------------------------------------------

def test_pause_resume_http_round_trip():
    ctrl = ControlState()
    stop = threading.Event()
    trainer = threading.Thread(
        target=_fake_trainer_loop, args=(ctrl, stop), daemon=True)
    trainer.start()

    srv = start_server("127.0.0.1", 0, ctrl,
                       checkpoint_timeout=0.2, pause_timeout=5.0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        s, b = post(base, "/pause", {})
        assert s == 200 and b["paused"] is True and b["vram_mb"] == 42.0
        assert ctrl.is_paused()
        assert get(base, "/health")[1]["paused"] is True

        # Idempotent: a second /pause while paused is an instant no-op that
        # still reports the current vram_mb.
        s2, b2 = post(base, "/pause", {})
        assert s2 == 200 and b2["paused"] is True and b2["vram_mb"] == 42.0

        s3, b3 = post(base, "/resume", {})
        assert s3 == 200 and b3["paused"] is False
        assert not ctrl.is_paused()
        assert get(base, "/health")[1]["paused"] is False

        # Resume again is a harmless no-op.
        assert post(base, "/resume", {})[1]["paused"] is False
    finally:
        stop.set()
        srv.shutdown()


def test_pause_times_out_without_trainer_ack():
    """No trainer thread to ack → /pause reports a timeout (504) rather than
    hanging; the home-server proceeds with COLMAP regardless."""
    ctrl = ControlState()
    srv = start_server("127.0.0.1", 0, ctrl,
                       checkpoint_timeout=0.2, pause_timeout=0.2)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        s, b = post(base, "/pause", {})
        assert s == 504 and b["paused"] is False
    finally:
        srv.shutdown()


# ---------------------------------------------------------------------------
# Model-level to_cpu()/to_cuda() round trip (needs CUDA)
# ---------------------------------------------------------------------------

def test_model_to_cpu_to_cuda_round_trip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("to_cpu()/to_cuda() round trip needs a GPU")
    try:
        import numpy as np
        from scene.spherical_gaussian_model import SphericalGaussianModel
        from scene.optim_guard import optimizer_binding_ok
        from utils.graphics_utils import BasicPointCloud
    except Exception as e:  # missing CUDA extensions / env
        pytest.skip(f"model stack not importable: {e}")

    from argparse import Namespace
    ta = Namespace(
        position_lr_init=1.6e-4, position_lr_final=1.6e-6,
        position_lr_delay_mult=0.01, position_lr_max_steps=30_000,
        feature_lr=2.5e-3, opacity_lr=0.05, scaling_lr=5e-3,
        rotation_lr=1e-3, percent_dense=0.01,
    )

    g = SphericalGaussianModel(max_sg_degree=3)
    N = 128
    rng = np.random.RandomState(0)
    pcd = BasicPointCloud(
        points=rng.randn(N, 3).astype("float32"),
        colors=rng.rand(N, 3).astype("float32"),
        normals=np.zeros((N, 3), dtype="float32"),
    )
    g.create_from_pcd(pcd, spatial_lr_scale=1.0, birth_iter=0)
    g.training_setup(ta)

    params = (
        g._xyz_cohorts + g._rgb_base_cohorts + g._opacity_cohorts
        + g._scaling_cohorts + g._rotation_cohorts + g._sg_directions_cohorts
        + g._sg_sharpness_cohorts + g._sg_rgb_cohorts
    )
    targets = [p.detach().clone() + 0.1 for p in params]

    def objective():
        return sum(((p - t) ** 2).mean() for p, t in zip(params, targets))

    # A few real Adam steps so exp_avg/exp_avg_sq are populated (no renderer
    # needed — the objective is differentiable in the params directly).
    for _ in range(5):
        objective().backward()
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)
    assert optimizer_binding_ok(g.optimizer)

    before = [p.detach().clone() for p in params]
    before_adam = {
        id(p): (st["exp_avg"].clone(), st["exp_avg_sq"].clone())
        for p, st in g.optimizer.state.items()
    }
    assert before_adam, "expected populated Adam state before the round trip"

    alloc_before = torch.cuda.memory_allocated()
    g.to_cpu()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Everything is on CPU and VRAM has dropped.
    assert all(p.data.device.type == "cpu" for p in params)
    assert g.max_radii2D.device.type == "cpu"
    for st in g.optimizer.state.values():
        assert st["exp_avg"].device.type == "cpu"
        assert st["exp_avg_sq"].device.type == "cpu"
    assert torch.cuda.memory_allocated() < alloc_before

    g.to_cuda()
    torch.cuda.synchronize()

    # Binding preserved (T1 invariant) and every tensor bit-identical.
    assert optimizer_binding_ok(g.optimizer)
    for p, b in zip(params, before):
        assert p.data.is_cuda
        assert torch.equal(p.data, b.cuda())
    for p, st in g.optimizer.state.items():
        exp_avg, exp_avg_sq = before_adam[id(p)]
        assert st["exp_avg"].is_cuda and st["exp_avg_sq"].is_cuda
        assert torch.equal(st["exp_avg"], exp_avg.cuda())
        assert torch.equal(st["exp_avg_sq"], exp_avg_sq.cuda())

    # Training continues and the loss keeps falling from the paused state.
    l0 = float(objective())
    for _ in range(25):
        objective().backward()
        g.optimizer.step()
        g.optimizer.zero_grad(set_to_none=True)
    l1 = float(objective())
    assert l1 < l0


def test_to_cpu_idempotent_without_optimizer():
    """to_cpu()/to_cuda() on a not-yet-bootstrapped model (no cohorts, no
    optimizer) is a safe no-op — the trainer may be paused before the first
    ingest."""
    torch = pytest.importorskip("torch")
    try:
        from scene.spherical_gaussian_model import SphericalGaussianModel
    except Exception as e:
        pytest.skip(f"model stack not importable: {e}")
    g = SphericalGaussianModel(max_sg_degree=3)
    g.to_cpu()   # must not raise
    g.to_cpu()   # idempotent
    if torch.cuda.is_available():
        g.to_cuda()
