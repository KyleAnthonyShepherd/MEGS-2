import torch


def evidence_settled(gaussians, min_obs=10, min_seen_fraction=0.5):
    """True iff per-Gaussian gradients have enough observations to act on."""
    denom = gaussians.denom.squeeze()
    seen = denom > 0
    if seen.sum().item() < min_seen_fraction * len(denom):
        return False
    return denom[seen].min().item() >= min_obs


def should_densify(gaussians, opt, mask_blur,
                   candidate_fraction=0.005,
                   min_obs=10):
    """Fire densify when per-Gaussian gradient evidence shows enough candidates.

    Settling (post-densify cool-off) is handled externally by tracking denom.sum()
    rather than iteration counts — see train_window's densify_denom_settled.
    """
    if not evidence_settled(gaussians, min_obs=min_obs):
        return False
    grads = gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)
    n_clone = (grads.squeeze() >= opt.densify_grad_threshold).sum().item()
    n_split = mask_blur.sum().item()
    n_total = gaussians._xyz.shape[0]
    return (n_clone + n_split) / max(n_total, 1) > candidate_fraction


def should_fast_prune(gaussians, opt, dead_fraction=0.02):
    """Fire opacity/size prune when enough dead splats have accumulated.
    Cheap; can be checked every iter."""
    min_op = getattr(opt, 'min_opacity_threshold', 0.005)
    dead = (gaussians.get_opacity < min_op).sum().item()
    return dead / gaussians._xyz.shape[0] > dead_fraction


def should_lightweight_prune(gaussians, monitor, n_at_last_prune,
                              soft_cap, growth_threshold=0.10):
    """Fire importance-prune on cap breach OR stagnation-with-growth.
    Expensive predicate (caller gates frequency)."""
    n = gaussians._xyz.shape[0]
    if n > soft_cap:
        return True
    grew = (n - n_at_last_prune) / max(n_at_last_prune, 1) > growth_threshold
    if grew and monitor.state() == "stalled":
        return True
    return False


def should_cull_sg_axes(gaussians, sharpness_threshold,
                        fraction_low=0.20):
    """Fire SG axis culling once enough axes are below sharpness threshold."""
    if gaussians.max_sg_degree == 0:
        return False
    sharpness = gaussians.get_sg_sharpness  # (N, K, 1)
    low_fraction = (sharpness < sharpness_threshold).float().mean().item()
    return low_fraction > fraction_low
