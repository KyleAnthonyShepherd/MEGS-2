"""Unit tests for scene/skipgs.py — SkipGS backward-gating lifecycle."""

from tests._direct_import import load_scene_module

skipgs_mod = load_scene_module("skipgs")
SkipGSGate = skipgs_mod.SkipGSGate


def make_ready_gate(warmup=5, rho_lo=0.5, cam_losses=None):
    """Drive a gate through enable + warmup so it is_ready()."""
    g = SkipGSGate(warmup_steady_samples=warmup, rho_lo=rho_lo)
    g.notify_monitor_state("converged")  # unlock
    for _ in range(warmup):
        g.notify_monitor_state("improving")
    if cam_losses:
        for cam_id, loss in cam_losses.items():
            g.update_ema(cam_id, loss)
    return g


def test_disabled_until_converged_seen():
    g = SkipGSGate(warmup_steady_samples=2)
    for state in ("improving", "stalled", "wants_capacity"):
        g.notify_monitor_state(state)
    assert not g.is_ready()
    gate, will_backward = g.decide([0.5, 0.5])
    assert gate == [True, True] and will_backward


def test_warmup_requires_consecutive_improving():
    g = SkipGSGate(warmup_steady_samples=3)
    g.notify_monitor_state("converged")
    g.notify_monitor_state("improving")
    g.notify_monitor_state("improving")
    g.notify_monitor_state("stalled")  # breaks the streak
    assert not g.is_ready()
    for _ in range(3):
        g.notify_monitor_state("improving")
    assert g.is_ready()


def test_unknown_camera_has_infinite_deviation():
    g = SkipGSGate()
    assert g.deviation(42, 1.0) == float("inf")


def test_ema_update():
    g = SkipGSGate(beta=0.9)
    g.update_ema(1, 1.0)
    assert g._ema[1] == 1.0
    g.update_ema(1, 0.0)
    assert abs(g._ema[1] - 0.9) < 1e-9


def test_ready_gate_skips_below_baseline_views():
    g = make_ready_gate(cam_losses={1: 1.0, 2: 1.0})
    # Warm the budget so rho_cum starts healthy
    for _ in range(20):
        gate, fired = g.decide([2.0, 2.0])  # above baseline → backward
        g.record_backward(fired)
    gate, fired = g.decide([0.5, 2.0])  # cam1 below baseline, cam2 above
    assert gate == [False, True] and fired
    g.record_backward(fired)
    gate, fired = g.decide([0.5, 0.5])
    assert gate == [False, False] and not fired


def test_budget_floor_forces_backward():
    """When the cumulative backward ratio falls below rho_min, the gate must
    force a full backward regardless of deviations."""
    g = make_ready_gate(rho_lo=0.99, cam_losses={1: 1.0})
    # rho_min ≈ 0.99: the gate may let a single skip through while rho_cum
    # sits at exactly 1.0, but every subsequent step must be forced until the
    # ratio recovers — so at most one of the five steps skips.
    fired_seq = []
    for _ in range(5):
        gate, fired = g.decide([0.5])
        g.record_backward(fired)
        fired_seq.append(fired)
    assert g._backward_count >= 4
    # Never two consecutive skips under a 0.99 floor
    assert not any(not a and not b for a, b in zip(fired_seq, fired_seq[1:]))


def test_rho_min_calibrated_once_from_warmup_stats():
    g = make_ready_gate(rho_lo=0.5)
    g.update_ema(1, 1.0)
    # Pre-ready decides recorded warmup stats; drive some evaluable samples
    g2 = SkipGSGate(warmup_steady_samples=1, rho_lo=0.5)
    g2.notify_monitor_state("converged")
    g2.update_ema(1, 1.0)
    # Not ready yet: these count as warmup-evaluable
    g2.decide([2.0])   # would fire
    g2.decide([0.5])   # would not
    g2.notify_monitor_state("improving")
    assert g2.is_ready()
    g2.decide([2.0])
    # rho_hat = 1/2 → rho_min = 0.5 + 0.5*0.5 = 0.75
    assert abs(g2._rho_min - 0.75) < 1e-9
