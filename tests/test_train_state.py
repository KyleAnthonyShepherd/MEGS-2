"""Round-trip tests for the checkpoint/resume state serialization:
ConvergenceMonitor, SkipGSGate, and the ControlState ingest ledger.

The model/optimizer half of train_state.pt goes through the pre-existing
SphericalGaussianModel.capture()/restore() pair, which needs CUDA — its
round trip is exercised by the GPU kill/resume test in the regression
harness, not here.
"""

from tests._direct_import import load_scene_module

convergence = load_scene_module("convergence")
skipgs_mod = load_scene_module("skipgs")
control_state = load_scene_module("control_state")

ConvergenceMonitor = convergence.ConvergenceMonitor
SkipGSGate = skipgs_mod.SkipGSGate
ControlState = control_state.ControlState


def test_monitor_round_trip_preserves_behaviour():
    """A restored monitor must classify the exact same future stream the
    same way the original would."""
    m1 = ConvergenceMonitor(loss_window=20, densify_window=5)
    for i in range(15):
        m1.update_loss(1.0 - 0.01 * i)
    m1.update_densify(10, 100)
    m1.reset(fraction_changed=0.3)
    for i in range(8):
        m1.update_loss(0.8)
        m1.state()  # accumulate stalled counts

    m2 = ConvergenceMonitor(loss_window=99, densify_window=99)  # wrong ctor args
    m2.set_state(m1.get_state())
    assert m2.loss_history.maxlen == 20
    assert m2.cycle == m1.cycle

    # Same future stream → identical state classifications
    for _ in range(30):
        m1.update_loss(0.8)
        m2.update_loss(0.8)
        assert m1.state() == m2.state()


def test_monitor_round_trip_preserves_densify_aging():
    m1 = ConvergenceMonitor(loss_window=10, densify_window=5)
    for _ in range(5):
        m1.update_loss(1.0)
    m1.update_densify(10, 100)
    m2 = ConvergenceMonitor(loss_window=10, densify_window=5)
    m2.set_state(m1.get_state())
    assert m2.densify_saturation() == m1.densify_saturation()
    # Age past the horizon in both; must decay identically
    for _ in range(20):
        m1.update_loss(1.0)
        m2.update_loss(1.0)
    assert m1.densify_saturation() == m2.densify_saturation() == 0.0


def test_skipgs_round_trip():
    g1 = SkipGSGate(warmup_steady_samples=3, rho_lo=0.5)
    g1.notify_monitor_state("converged")
    for _ in range(3):
        g1.notify_monitor_state("improving")
    g1.update_ema(1, 1.0)
    g1.update_ema(2, 2.0)
    for _ in range(10):
        gate, fired = g1.decide([2.0, 3.0])
        g1.record_backward(fired)

    g2 = SkipGSGate(warmup_steady_samples=3, rho_lo=0.5)
    g2.set_state(g1.get_state())
    assert g2.is_ready() == g1.is_ready()
    assert g2._rho_min == g1._rho_min

    # Same future decisions
    for devs in ([0.5, 0.5], [2.0, 0.5], [0.5, 2.0]):
        d1 = g1.decide(list(devs))
        d2 = g2.decide(list(devs))
        assert d1 == d2
        g1.record_backward(d1[1])
        g2.record_backward(d2[1])


def test_ledger_round_trip_preserves_idempotency():
    """The whole point of persisting the ledger: an upstream retry of an
    already-integrated image must still be recognised after a restart."""
    c1 = ControlState()
    rid, dup = c1.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert not dup
    c1.drain_ingest_queue()

    c2 = ControlState()
    c2.restore_ledger(c1.ledger_state())
    assert c2.session_id == "s1"
    rid2, dup2 = c2.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert dup2 and rid2 == rid
    # New images still accepted
    _, dup3 = c2.enqueue_ingest("/snap", session_id="s1", image_name="b.jpg")
    assert not dup3


def test_ledger_state_is_json_serializable():
    import json
    c = ControlState()
    c.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    state = json.loads(json.dumps(c.ledger_state()))
    c2 = ControlState()
    c2.restore_ledger(state)
    _, dup = c2.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert dup


def test_restore_empty_ledger():
    c = ControlState()
    c.restore_ledger({})
    assert c.session_id is None
    _, dup = c.enqueue_ingest("/snap", session_id="s1", image_name="a.jpg")
    assert not dup
