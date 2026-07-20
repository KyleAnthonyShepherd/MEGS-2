"""Unit tests for scene/triggers.py — trigger predicates in isolation.

Fake gaussians expose only the tensors each predicate reads, so every
branch (state gating, anti-thrash floor, evidence gating, cap breach) can be
exercised without a model or GPU.
"""

from types import SimpleNamespace

import torch

from tests._direct_import import load_scene_module

triggers = load_scene_module("triggers")


def make_gaussians(n=100, denom_val=20.0, grad_val=0.0, opacity_val=0.5,
                   sharpness_val=2.0, max_sg_degree=3):
    g = SimpleNamespace()
    g._xyz = torch.zeros(n, 3)
    g.denom = torch.full((n, 1), denom_val)
    g.xyz_gradient_accum = torch.full((n, 1), grad_val)
    g.get_opacity = torch.full((n, 1), opacity_val)
    g.get_sg_sharpness = torch.full((n, max_sg_degree, 1), sharpness_val)
    g.max_sg_degree = max_sg_degree
    return g


OPT = SimpleNamespace(densify_grad_threshold=0.01, min_opacity_threshold=0.005)


# ---------------------------------------------------------------------------
# evidence_settled
# ---------------------------------------------------------------------------

def test_evidence_settled_true_with_mature_observations():
    assert triggers.evidence_settled(make_gaussians(denom_val=20))


def test_evidence_settled_false_when_too_few_seen():
    g = make_gaussians(n=100, denom_val=0.0)
    g.denom[:10] = 20.0  # only 10% seen < min_seen_fraction 0.5
    assert not triggers.evidence_settled(g)


def test_evidence_settled_false_when_observations_immature():
    assert not triggers.evidence_settled(make_gaussians(denom_val=5))


# ---------------------------------------------------------------------------
# should_densify
# ---------------------------------------------------------------------------

def _densify(g, state="wants_capacity", iters_since=100, mask_blur=None, **kw):
    if mask_blur is None:
        mask_blur = torch.zeros(g._xyz.shape[0], dtype=torch.bool)
    return triggers.should_densify(state, g, OPT, mask_blur,
                                   iters_since_last=iters_since, **kw)


def test_densify_fires_on_high_grad_candidates():
    g = make_gaussians(grad_val=10.0)  # grads = 10/20 = 0.5 >> threshold
    assert _densify(g)


def test_densify_respects_require_states():
    g = make_gaussians(grad_val=10.0)
    assert not _densify(g, state="improving")
    assert not _densify(g, state="converged")
    assert _densify(g, state="stalled")


def test_densify_anti_thrash_floor():
    g = make_gaussians(grad_val=10.0)
    assert not _densify(g, iters_since=59)
    assert _densify(g, iters_since=60)


def test_densify_blocked_by_immature_evidence():
    g = make_gaussians(grad_val=10.0, denom_val=5)
    assert not _densify(g)


def test_densify_needs_enough_candidates():
    g = make_gaussians(n=1000, grad_val=0.0)
    # 4 candidates / 1000 = 0.004 < candidate_fraction 0.005
    g.xyz_gradient_accum[:4] = 10.0
    assert not _densify(g)
    g.xyz_gradient_accum[:6] = 10.0
    assert _densify(g)


def test_densify_counts_mask_blur_as_candidates():
    g = make_gaussians(n=1000, grad_val=0.0)
    mask_blur = torch.zeros(1000, dtype=torch.bool)
    mask_blur[:6] = True
    assert _densify(g, mask_blur=mask_blur)


# ---------------------------------------------------------------------------
# should_fast_prune
# ---------------------------------------------------------------------------

def test_fast_prune_fires_on_dead_fraction():
    g = make_gaussians(n=100, opacity_val=0.5)
    g.get_opacity[:3] = 0.001  # 3% dead > 2%
    assert triggers.should_fast_prune("stalled", g, OPT, iters_since_last=500)


def test_fast_prune_blocked_below_dead_fraction():
    g = make_gaussians(n=100, opacity_val=0.5)
    g.get_opacity[:1] = 0.001  # 1% < 2%
    assert not triggers.should_fast_prune("stalled", g, OPT, iters_since_last=500)


def test_fast_prune_state_and_floor_gating():
    g = make_gaussians(n=100)
    g.get_opacity[:50] = 0.001
    assert not triggers.should_fast_prune("improving", g, OPT, iters_since_last=500)
    assert not triggers.should_fast_prune("stalled", g, OPT, iters_since_last=100)
    assert triggers.should_fast_prune("converged", g, OPT, iters_since_last=500)


# ---------------------------------------------------------------------------
# should_lightweight_prune
# ---------------------------------------------------------------------------

def test_lightweight_prune_cap_breach_overrides_everything():
    """Cap breach fires regardless of state or anti-thrash floor — the VRAM
    ceiling is the one hard constraint."""
    g = make_gaussians(n=100)
    assert triggers.should_lightweight_prune(
        "improving", g, n_at_last_prune=100, soft_cap=50, iters_since_last=0)


def test_lightweight_prune_growth_requires_state():
    g = make_gaussians(n=100)
    kw = dict(n_at_last_prune=80, soft_cap=1000, iters_since_last=500)
    assert triggers.should_lightweight_prune("converged", g, **kw)
    assert not triggers.should_lightweight_prune("improving", g, **kw)


def test_lightweight_prune_growth_threshold_and_floor():
    g = make_gaussians(n=100)
    # 100/95 ≈ 5% growth < 10% threshold
    assert not triggers.should_lightweight_prune(
        "converged", g, n_at_last_prune=95, soft_cap=1000, iters_since_last=500)
    # floor not met
    assert not triggers.should_lightweight_prune(
        "converged", g, n_at_last_prune=80, soft_cap=1000, iters_since_last=50)


# ---------------------------------------------------------------------------
# should_cull_sg_axes
# ---------------------------------------------------------------------------

def test_cull_fires_when_enough_axes_dull():
    g = make_gaussians(sharpness_val=0.5)  # all below threshold 1.0
    assert triggers.should_cull_sg_axes(
        "converged", g, sharpness_threshold=1.0, iters_since_last=1000)


def test_cull_gated_by_state_floor_and_degree():
    g = make_gaussians(sharpness_val=0.5)
    assert not triggers.should_cull_sg_axes(
        "stalled", g, sharpness_threshold=1.0, iters_since_last=1000)
    assert not triggers.should_cull_sg_axes(
        "converged", g, sharpness_threshold=1.0, iters_since_last=100)
    g.max_sg_degree = 0
    assert not triggers.should_cull_sg_axes(
        "converged", g, sharpness_threshold=1.0, iters_since_last=1000)


def test_cull_needs_low_fraction():
    g = make_gaussians(n=100, sharpness_val=2.0, max_sg_degree=1)
    g.get_sg_sharpness[:10] = 0.5  # 10% < 20%
    assert not triggers.should_cull_sg_axes(
        "converged", g, sharpness_threshold=1.0, iters_since_last=1000)
    g.get_sg_sharpness[:30] = 0.5  # 30% > 20%
    assert triggers.should_cull_sg_axes(
        "converged", g, sharpness_threshold=1.0, iters_since_last=1000)
