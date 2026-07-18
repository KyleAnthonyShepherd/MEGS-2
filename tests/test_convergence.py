"""Unit tests for scene/convergence.py — ConvergenceMonitor state machine.

Includes pinned regressions for historical scheduler bugs from the commit log:
  - cc71724 "Age out densify_saturation by iter so converged is reachable"
  - 850f7c0 stalled-timeout promotion to converged
"""

from tests._direct_import import load_scene_module

convergence = load_scene_module("convergence")
ConvergenceMonitor = convergence.ConvergenceMonitor


def make_monitor(loss_window=20, densify_window=10):
    return ConvergenceMonitor(loss_window=loss_window, densify_window=densify_window)


def feed_losses(monitor, values):
    for v in values:
        monitor.update_loss(v)


def test_insufficient_history_reports_improving():
    m = make_monitor(loss_window=20)
    feed_losses(m, [1.0] * 5)
    assert m.relative_slope() == float("-inf")
    assert m.state() == "improving"


def test_monotone_improving_stream_reaches_converged():
    """Steadily but slowly decreasing loss must eventually read converged."""
    m = make_monitor(loss_window=20)
    # Slope per iter relative to loss: tiny (1e-6) — inside the converged band
    feed_losses(m, [1.0 - 1e-6 * i for i in range(20)])
    assert m.state() == "converged"


def test_steep_improvement_reads_improving():
    m = make_monitor(loss_window=20)
    feed_losses(m, [1.0 - 0.01 * i for i in range(20)])
    assert m.state() == "improving"


def test_plateau_reads_stalled_then_promotes_to_converged():
    """A flat-but-noisy loss with recent densify activity is 'stalled';
    after a full loss_window of consecutive stalled reads it must promote to
    converged (the stalled-timeout added in 850f7c0), so the trainer can idle
    instead of grinding forever just outside the converged band."""
    m = make_monitor(loss_window=20)
    feed_losses(m, [1.0] * 20)
    # Keep densify saturation just above the converged gate (d >= 0.01)
    # but below active_densify so state stays out of wants_capacity.
    m.update_densify(2, 100)  # 0.02
    states = [m.state() for _ in range(19)]
    assert set(states) == {"stalled"}
    # 20th consecutive stalled sample == loss_window → promoted
    assert m.state() == "converged"


def test_wants_capacity_when_densify_active_and_slope_steep():
    m = make_monitor(loss_window=20)
    feed_losses(m, [1.0 - 0.01 * i for i in range(20)])
    m.update_densify(10, 100)  # 0.10 > active_densify 0.05
    assert m.state() == "wants_capacity"


def test_densify_saturation_ages_out_by_iter():
    """Pinned regression (cc71724): a single past densify must not keep
    saturation pinned forever — entries older than loss_window iters are
    dropped, so converged becomes reachable again."""
    m = make_monitor(loss_window=20)
    feed_losses(m, [1.0 - 0.01 * i for i in range(10)])
    m.update_densify(10, 100)
    assert m.densify_saturation() == 0.10
    # Run well past the horizon with flat loss
    feed_losses(m, [0.9] * 40)
    assert m.densify_saturation() == 0.0
    # And with a flat slope + aged-out densify, converged is reachable
    assert m.state() == "converged"


def test_reset_scales_min_entries_with_fraction_changed():
    """A small structural change (5%) needs only ~10 fresh entries before the
    slope is meaningful again; a full change needs the whole window."""
    m = make_monitor(loss_window=100)
    feed_losses(m, [1.0] * 100)

    m.reset(fraction_changed=0.05)
    feed_losses(m, [1.0] * 10)  # only 10 entries, but min_entries = max(10, 5) = 10
    assert m.relative_slope() != float("-inf")

    m.reset(fraction_changed=1.0)
    feed_losses(m, [1.0] * 50)  # 50 < 100 required
    assert m.relative_slope() == float("-inf")


def test_reset_bumps_cycle_and_clears_stall_count():
    m = make_monitor(loss_window=10)
    feed_losses(m, [1.0] * 10)
    m.update_densify(2, 100)
    for _ in range(5):
        m.state()  # accumulate stalled count
    c0 = m.cycle
    m.reset(fraction_changed=0.5)
    assert m.cycle == c0 + 1
    # After reset the stalled counter starts over: feed a fresh flat window
    # sized to min_entries and confirm we do NOT immediately promote.
    feed_losses(m, [1.0] * 10)
    m.update_densify(2, 100)
    assert m.state() == "stalled"


def test_non_stalled_state_resets_stall_counter():
    """Interleaving an improving read must clear the consecutive-stall count,
    so promotion requires an *uninterrupted* stalled window."""
    m = make_monitor(loss_window=10)
    feed_losses(m, [1.0] * 10)
    m.update_densify(2, 100)
    for _ in range(9):
        assert m.state() == "stalled"
    # Steep improvement resets the counter
    feed_losses(m, [1.0 - 0.05 * i for i in range(10)])
    assert m.state() == "improving"
    # Back to plateau: needs a full 10 consecutive stalls again
    feed_losses(m, [0.5] * 10)
    m.update_densify(2, 100)
    for _ in range(9):
        assert m.state() == "stalled"
    assert m.state() == "converged"
