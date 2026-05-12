import torch


def evidence_settled(gaussians, min_obs=10, min_seen_fraction=0.5):
    """True iff per-Gaussian gradients have enough observations to act on."""
    denom = gaussians.denom.squeeze()
    seen = denom > 0
    if seen.sum().item() < min_seen_fraction * len(denom):
        return False
    return denom[seen].min().item() >= min_obs


def should_densify(current_state, gaussians, opt, mask_blur, iters_since_last,
                   candidate_fraction=0.005, min_obs=10,
                   min_iters_between=60, require_states=("stalled", "wants_capacity")):
    """Fire densify when monitor is in a capacity-hungry state and evidence is sufficient.

    current_state is the monitor state snapshotted at iter start — passed
    explicitly so all triggers in the same iter see the same world, even
    after an earlier trigger called monitor.reset().

    iters_since_last is the anti-thrash floor — prevents back-to-back densify
    cycles from overwhelming the optimizer before new Gaussians have settled.
    """
    if current_state not in require_states:
        return False
    if iters_since_last < min_iters_between:
        return False
    if not evidence_settled(gaussians, min_obs=min_obs):
        return False
    grads = gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)
    n_clone = (grads.squeeze() >= opt.densify_grad_threshold).sum().item()
    n_split = mask_blur.sum().item()
    n_total = gaussians._xyz.shape[0]
    return (n_clone + n_split) / max(n_total, 1) > candidate_fraction


def should_fast_prune(current_state, gaussians, opt, dead_fraction=0.02,
                      iters_since_last=0, min_iters_between=200,
                      require_states=("stalled", "converged")):
    """Fire opacity/size prune when dead splats accumulate and state allows it."""
    if current_state not in require_states:
        return False
    if iters_since_last < min_iters_between:
        return False
    min_op = getattr(opt, 'min_opacity_threshold', 0.005)
    dead = (gaussians.get_opacity < min_op).sum().item()
    return dead / gaussians._xyz.shape[0] > dead_fraction


def should_lightweight_prune(current_state, gaussians, n_at_last_prune,
                              soft_cap, iters_since_last=0,
                              min_iters_between=100, growth_threshold=0.10,
                              require_states=("converged",)):
    """Fire importance-prune on cap breach OR stagnation-with-growth."""
    n = gaussians._xyz.shape[0]
    if n > soft_cap:
        return True
    if iters_since_last < min_iters_between:
        return False
    grew = (n - n_at_last_prune) / max(n_at_last_prune, 1) > growth_threshold
    if grew and current_state in require_states:
        return True
    return False


def should_cull_sg_axes(current_state, gaussians, sharpness_threshold,
                        fraction_low=0.20, iters_since_last=0,
                        min_iters_between=500, require_states=("converged",)):
    """Fire SG axis culling when state signals convergence and enough axes are dull."""
    if current_state not in require_states:
        return False
    if iters_since_last < min_iters_between:
        return False
    if gaussians.max_sg_degree == 0:
        return False
    sharpness = gaussians.get_sg_sharpness
    low_fraction = (sharpness < sharpness_threshold).float().mean().item()
    return low_fraction > fraction_low
