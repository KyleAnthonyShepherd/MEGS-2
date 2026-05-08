"""Tests for expand_from_pcd and grace-period mechanism (Phase 4 acceptance tests)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def requires_cuda(fn):
    """Decorator to skip if CUDA not available."""
    import functools
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return fn(*args, **kwargs)
    return wrapper


def make_minimal_gaussians(n_pts=50, sg_degree=0):
    """Create a minimal SphericalGaussianModel with `n_pts` random Gaussians."""
    import torch
    from utils.graphics_utils import BasicPointCloud
    from scene.spherical_gaussian_model import SphericalGaussianModel
    from argparse import Namespace

    gaussians = SphericalGaussianModel(max_sg_degree=sg_degree)
    rng = np.random.default_rng(42)
    xyz = rng.uniform(-1, 1, (n_pts, 3)).astype(np.float32)
    rgb = rng.uniform(0, 1, (n_pts, 3)).astype(np.float32)
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros_like(xyz))
    gaussians.create_from_pcd(pcd, spatial_lr_scale=1.0)

    opt = Namespace(
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30000,
        feature_lr=0.0025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.001,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=100,
        opacity_reset_interval=3000,
        densify_from_iter=500,
        densify_until_iter=15000,
        densify_grad_threshold=0.0002,
        prune_ratio1=0.5,
        prune_ratio2=0.8,
        sharpness_ratio=0.7,
        optimizing_spa_interval=50,
        optimizing_spa_start_iter=25200,
        optimizing_spa_stop_iter=35200,
        optimizing_spa_sg_start_iter=25200,
        optimizing_spa_sg_stop_iter=35200,
        optimizing_spa=False,
        rho_lr=0.0005,
        lambda_sh_sparsity=0.01,
        random_background=False,
    )
    gaussians.training_setup(opt)
    return gaussians, opt


@requires_cuda
def test_expand_from_pcd_increases_count():
    """Appending new points should increase Gaussian count."""
    from utils.graphics_utils import BasicPointCloud
    from scene.spherical_gaussian_model import SphericalGaussianModel

    gaussians, _ = make_minimal_gaussians(n_pts=50, sg_degree=0)
    G0 = gaussians.get_xyz.shape[0]

    n_new = 30
    rng = np.random.default_rng(1)
    new_xyz = rng.uniform(2, 3, (n_new, 3)).astype(np.float32)
    new_rgb = rng.uniform(0, 1, (n_new, 3)).astype(np.float32)
    new_pcd = BasicPointCloud(
        points=new_xyz, colors=new_rgb, normals=np.zeros_like(new_xyz))
    mask = np.ones(n_new, dtype=bool)

    gaussians.expand_from_pcd(new_pcd, mask, spatial_lr_scale=1.0)
    G1 = gaussians.get_xyz.shape[0]

    assert G1 > G0, f"Gaussian count did not increase: {G0} → {G1}"
    assert G1 - G0 == n_new, f"Expected +{n_new} Gaussians, got +{G1 - G0}"


@requires_cuda
def test_expand_from_pcd_optimizer_shape_integrity():
    """After expand, each optimizer param group must match its model tensor."""
    import torch
    from utils.graphics_utils import BasicPointCloud

    gaussians, opt = make_minimal_gaussians(n_pts=20, sg_degree=0)

    n_new = 10
    rng = np.random.default_rng(2)
    new_xyz = rng.uniform(5, 6, (n_new, 3)).astype(np.float32)
    new_rgb = rng.uniform(0, 1, (n_new, 3)).astype(np.float32)
    new_pcd = BasicPointCloud(
        points=new_xyz, colors=new_rgb, normals=np.zeros_like(new_xyz))
    mask = np.ones(n_new, dtype=bool)

    gaussians.expand_from_pcd(new_pcd, mask, spatial_lr_scale=1.0)

    # Verify param group shapes match model tensor shapes
    for group in gaussians.optimizer.param_groups:
        name = group["name"]
        param = group["params"][0]
        if name == "xyz":
            expected = gaussians._xyz
        elif name == "rgb_base":
            expected = gaussians._rgb_base
        elif name == "opacity":
            expected = gaussians._opacity
        elif name == "scaling":
            expected = gaussians._scaling
        elif name == "rotation":
            expected = gaussians._rotation
        else:
            continue
        assert param.shape == expected.shape, \
            f"Param group '{name}' shape {param.shape} != model tensor {expected.shape}"


@requires_cuda
def test_no_nan_after_expansion_and_backward():
    """Training step should produce finite loss after expansion."""
    import torch
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import l1_loss

    gaussians, opt = make_minimal_gaussians(n_pts=20, sg_degree=0)

    n_new = 5
    rng = np.random.default_rng(3)
    new_xyz = rng.uniform(5, 6, (n_new, 3)).astype(np.float32)
    new_rgb = rng.uniform(0, 1, (n_new, 3)).astype(np.float32)
    new_pcd = BasicPointCloud(
        points=new_xyz, colors=new_rgb, normals=np.zeros_like(new_xyz))

    gaussians.expand_from_pcd(new_pcd, np.ones(n_new, dtype=bool), 1.0)

    # Simple loss on positions (not a real render loss, just a smoke test)
    fake_target = gaussians.get_xyz.detach() + 0.01
    loss = l1_loss(gaussians.get_xyz, fake_target)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    loss.backward()
    gaussians.optimizer.step()
    gaussians.optimizer.zero_grad(set_to_none=True)
    assert torch.isfinite(gaussians.get_xyz).all(), "NaN in xyz after optimizer step"


@requires_cuda
def test_mark_recently_added_and_grace_mask():
    """Grace period: protected indices are True before expiry, False after."""
    gaussians, _ = make_minimal_gaussians(n_pts=300, sg_degree=0)
    gaussians._grace_records = []

    gaussians.mark_recently_added(slice(100, 200), iteration=0, grace_iters=200)

    mask_50 = gaussians.get_grace_protected_mask(current_iter=50)
    assert mask_50[100:200].all(), "Indices 100-199 should be protected at iter 50"
    assert not mask_50[:100].any(), "Indices 0-99 should not be protected"
    assert not mask_50[200:].any(), "Indices 200+ should not be protected"

    mask_250 = gaussians.get_grace_protected_mask(current_iter=250)
    assert not mask_250.any(), "All protection should expire at iter 250"


@requires_cuda
def test_expand_with_sg_degree():
    """Expansion should work with max_sg_degree > 0."""
    import torch
    from utils.graphics_utils import BasicPointCloud

    gaussians, opt = make_minimal_gaussians(n_pts=10, sg_degree=2)

    n_new = 8
    rng = np.random.default_rng(5)
    new_xyz = rng.uniform(5, 6, (n_new, 3)).astype(np.float32)
    new_rgb = rng.uniform(0, 1, (n_new, 3)).astype(np.float32)
    new_pcd = BasicPointCloud(
        points=new_xyz, colors=new_rgb, normals=np.zeros_like(new_xyz))

    G_before = gaussians._xyz.shape[0]
    gaussians.expand_from_pcd(new_pcd, np.ones(n_new, dtype=bool), 1.0)
    G_after = gaussians._xyz.shape[0]

    assert G_after == G_before + n_new
    assert gaussians._sg_directions.shape == (G_after, 2, 3)
    assert gaussians._sg_sharpness.shape == (G_after, 2, 1)
    assert gaussians._sg_rgb.shape == (G_after, 2, 3)
    assert gaussians._sg_axis_count.shape == (G_after,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
