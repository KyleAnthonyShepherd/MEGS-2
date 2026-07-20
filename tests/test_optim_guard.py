"""Tests for scene/optim_guard.py with a real (CPU) Adam optimizer."""

import pytest

torch = pytest.importorskip("torch")

from tests._direct_import import load_scene_module

og = load_scene_module("optim_guard")


def make_adam(n=4):
    p = torch.nn.Parameter(torch.zeros(n, 3))
    opt = torch.optim.Adam([{"params": [p], "name": "xyz"}], lr=1e-3)
    return p, opt


def step(p, opt):
    (p.sum()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_none_and_fresh_optimizer_ok():
    assert og.optimizer_binding_ok(None)
    _, opt = make_adam()
    assert og.optimizer_binding_ok(opt)  # no steps yet, empty state


def test_stepped_optimizer_ok():
    p, opt = make_adam()
    step(p, opt)
    assert len(opt.state) == 1
    assert og.optimizer_binding_ok(opt)


def test_correct_prune_style_rebind_ok():
    """The Adam-preserving pattern (_prune_optimizer): new param, state
    migrated to the new key."""
    p, opt = make_adam()
    step(p, opt)
    state = opt.state.pop(p)
    keep = torch.tensor([True, False, True, True])
    new_p = torch.nn.Parameter(p.data[keep])
    state["exp_avg"] = state["exp_avg"][keep]
    state["exp_avg_sq"] = state["exp_avg_sq"][keep]
    opt.param_groups[0]["params"][0] = new_p
    opt.state[new_p] = state
    assert og.optimizer_binding_ok(opt)


def test_stale_rebind_detected():
    """The T1 bug pattern: param rebound without migrating state — the
    state entry stays keyed by the dead parameter object."""
    p, opt = make_adam()
    step(p, opt)
    new_p = torch.nn.Parameter(p.data[:2])
    opt.param_groups[0]["params"][0] = new_p  # state still keyed by old p
    assert not og.optimizer_binding_ok(opt)
