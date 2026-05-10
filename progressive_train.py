"""Progressive MEGS² training entry point.

Trains Gaussian Splatting scenes incrementally as new images arrive from an
upstream Structure-from-Motion pipeline, using MEGS-2's SphericalGaussianModel
and layering GS_On-The-Fly's progressive scheduling on top.

Usage:
    python progressive_train.py \
        --source_path_dir /path/to/snapshots \
        --model_path /path/to/output \
        --config configs/progressive.yaml \
        [--dense_init_enabled] \
        [--imp_metric outdoor]
"""

import gc
import logging
import math
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import List, Optional

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.spherical_gaussian_model import SphericalGaussianModel
from scene.progressive_scene import ProgressiveScene
from scene.match_matrix import parse_match_matrix, compute_image_weights
from scene.dense_init import (
    DepthAnythingV2Wrapper, align_depth_to_sfm, depth_to_points,
    transform_to_camera_frame, AlignmentFailed,
    DAv2Config, RansacConfig,
)
from scene.convergence import ConvergenceMonitor
from scene.triggers import (
    should_densify, should_fast_prune,
    should_lightweight_prune, should_cull_sg_axes,
)
from scene.skipgs import SkipGSGate
from spherical_gaussian_renderer import render_imp
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import BasicPointCloud
from utils.image_utils import psnr

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SkipGSConfig:
    enabled: bool = True
    phase: str = "final"
    warmup: int = 500
    beta: float = 0.95
    eps: float = 1e-8
    rho_lo: float = 0.5


@dataclass
class TrainingConfig:
    # Iteration caps (convergence drives early exit; these are safety belts)
    iter_initial: int = 3000
    iter_per_merge: int = 200
    iter_final: int = 4000

    # Single VRAM-bounded Gaussian ceiling
    num_max_ceiling: int = 800_000

    # Pruning ratios
    prune_ratio1: float = 0.05
    prune_ratio2: float = 0.05
    sharpness_threshold: float = 1.0
    imp_metric: str = "outdoor"

    # T5: camera subsample in update_imp_score (0 = all cameras)
    imp_score_camera_subsample: int = 0

    # T6: multi-view gradient accumulation (1 = original single-view)
    accumulation_views: int = 1

    # T7: trigger predicate knobs
    densify_min_obs: int = 10
    densify_min_settling: int = 50
    densify_max_interval: int = 500
    densify_candidate_fraction: float = 0.005
    fast_prune_dead_fraction: float = 0.02
    lightweight_prune_growth_threshold: float = 0.10
    sg_axis_cull_low_fraction: float = 0.20

    # T7: fast final compression ratio
    fast_final_prune_ratio: float = 0.10

    # T8: SkipGS gate config
    skipgs: SkipGSConfig = field(default_factory=SkipGSConfig)


@dataclass
class DenseInitConfig:
    enabled: bool = True
    skip_first_n_snapshots: int = 3
    min_sfm_points_for_alignment: int = 10
    min_sfm_depth_range_fraction: float = 0.10
    target_dense_points_per_image: int = 50_000
    novelty_distance_threshold: float = 0.01
    depth_disagreement_threshold: float = 0.10
    max_rejected_fraction: float = 0.5
    grace_iters: int = 200
    persist_model: bool = False         # T4: hold DAv2 across snapshots
    dav2: DAv2Config = field(default_factory=DAv2Config)
    ransac: RansacConfig = field(default_factory=RansacConfig)


@dataclass
class SnapshotConfig:
    source_path_dir: Optional[str] = None
    snapshot_skip: int = 1
    progressive_output: bool = True


@dataclass
class ProgressiveConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dense_init: DenseInitConfig = field(default_factory=DenseInitConfig)
    snapshots: SnapshotConfig = field(default_factory=SnapshotConfig)


def load_config(yaml_path: str) -> ProgressiveConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = ProgressiveConfig()

    if "training" in raw:
        for k, v in raw["training"].items():
            if k == "skipgs" and isinstance(v, dict):
                for sk, sv in v.items():
                    if hasattr(cfg.training.skipgs, sk):
                        setattr(cfg.training.skipgs, sk, sv)
            elif hasattr(cfg.training, k):
                setattr(cfg.training, k, v)

    if "dense_init" in raw:
        di = raw["dense_init"]
        for k, v in di.items():
            if k == "dav2":
                for dk, dv in v.items():
                    if hasattr(cfg.dense_init.dav2, dk):
                        setattr(cfg.dense_init.dav2, dk, dv)
            elif k == "ransac":
                for rk, rv in v.items():
                    if hasattr(cfg.dense_init.ransac, rk):
                        setattr(cfg.dense_init.ransac, rk, rv)
            elif hasattr(cfg.dense_init, k):
                setattr(cfg.dense_init, k, v)

    if "snapshots" in raw:
        for k, v in raw["snapshots"].items():
            if hasattr(cfg.snapshots, k):
                setattr(cfg.snapshots, k, v)

    return cfg


# ---------------------------------------------------------------------------
# Importance scoring (lifted from train.py)
# ---------------------------------------------------------------------------

def update_imp_score(cameras, gaussians, pipe, background, imp_metric="outdoor",
                     subsample_n: int = 0, weights=None):
    # T5: optional camera subsampling by weight
    active_cameras = list(cameras)
    if subsample_n > 0 and len(active_cameras) > subsample_n:
        if weights is not None and len(weights) == len(active_cameras):
            sorted_idx = np.argsort(weights)[::-1]
            active_cameras = [active_cameras[i] for i in sorted_idx[:subsample_n]]
        else:
            active_cameras = list(np.random.choice(active_cameras, size=subsample_n, replace=False))
    scale = len(cameras) / max(len(active_cameras), 1)

    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
    for view in active_cameras:
        render_pkg = render_imp(view, gaussians, pipe, background, is_training=True)
        accum_weights = render_pkg["accum_weights"]
        area_proj = render_pkg["area_proj"]
        area_max = render_pkg["area_max"]
        accum_area_max = accum_area_max + area_max
        if imp_metric == "outdoor":
            mask_t = area_max != 0
            temp = imp_score + accum_weights / area_proj
            imp_score[mask_t] = temp[mask_t]
        else:
            imp_score = imp_score + accum_weights
    imp_score[accum_area_max == 0] = 0
    if subsample_n > 0 and scale != 1.0:
        imp_score = imp_score * scale
    return imp_score


def update_sg_color_diff(gaussians):
    sg_color = gaussians.get_sg_rgb
    sg_sharpness = gaussians.get_sg_sharpness
    sg_color_diff = torch.abs(sg_color) * (1 - torch.exp(-2 * sg_sharpness))
    sg_color_diff = torch.mean(torch.abs(sg_color_diff), dim=2, keepdim=True)
    return sg_color_diff


# ---------------------------------------------------------------------------
# Camera selection (Phase 6 — match-graph weighted)
# ---------------------------------------------------------------------------

def select_cameras_weighted(
    all_cameras,
    image_weights: Optional[np.ndarray],
    n_top: int = 10,
    n_random: int = 10,
) -> list:
    """Select cameras for one training iteration.

    Top n_top by weight + n_random uniformly random from the rest.
    Falls back to uniform random when weights are None.
    """
    if image_weights is None or len(all_cameras) <= n_top + n_random:
        return list(all_cameras)

    weights = image_weights
    sorted_idx = np.argsort(weights)[::-1]
    top_idx = set(sorted_idx[:n_top].tolist())
    rest_idx = [i for i in range(len(all_cameras)) if i not in top_idx]

    selected = [all_cameras[i] for i in top_idx]
    if rest_idx:
        sampled = np.random.choice(
            rest_idx, size=min(n_random, len(rest_idx)), replace=False)
        selected += [all_cameras[i] for i in sampled]
    return selected


# ---------------------------------------------------------------------------
# Gaussian expansion from new sparse SfM points (Phase 4)
# ---------------------------------------------------------------------------

def expand_gaussians_from_new_points(
    gaussians: SphericalGaussianModel,
    prog_scene: ProgressiveScene,
    new_point_mask: np.ndarray,
    opt,
    distance_threshold: float = 1.0,
    distance_buffer: float = 1.5,
):
    """Seed new Gaussians for novel sparse SfM points.

    Uses cKDTree to confirm spatial novelty against existing Gaussians.
    """
    pcd = prog_scene.current_basic_pcd
    if pcd is None or new_point_mask.sum() == 0:
        return

    candidate_xyz = pcd.points[new_point_mask]
    if candidate_xyz.shape[0] == 0:
        return

    existing_xyz = gaussians.get_xyz.detach().cpu().numpy()
    if len(existing_xyz) > 0:
        tree = cKDTree(existing_xyz)
        threshold = distance_threshold * prog_scene.cameras_extent
        distances, _ = tree.query(candidate_xyz)
        novel_mask_local = distances > threshold / distance_buffer
    else:
        novel_mask_local = np.ones(len(candidate_xyz), dtype=bool)

    # Map local mask back to full pcd mask
    all_new_indices = np.where(new_point_mask)[0]
    novel_global = np.zeros(len(pcd.points), dtype=bool)
    novel_global[all_new_indices[novel_mask_local]] = True

    n_novel = novel_global.sum()
    if n_novel == 0:
        logger.info("[expand] No novel sparse points to add")
        return

    logger.info(f"[expand] Adding {n_novel} Gaussians from new sparse SfM points")
    gaussians.expand_from_pcd(pcd, novel_global, prog_scene.cameras_extent)


# ---------------------------------------------------------------------------
# Dense initialization for new images (Phase 5)
# ---------------------------------------------------------------------------

def dense_init_for_new_images(
    gaussians: SphericalGaussianModel,
    prog_scene: ProgressiveScene,
    new_cams: list,
    dense_cfg: DenseInitConfig,
    opt,
    current_iter: int,
    dav2_model=None,
):
    """Run DAv2 on each new image, align to SfM, filter, append as Gaussians.

    dav2_model: optional pre-loaded DepthAnythingV2Wrapper (T4 persistence).
                When provided, the caller owns the model lifecycle; this function
                will NOT open or close a context manager.
    """
    import time
    t_start = time.monotonic()
    accumulated_xyz = []
    accumulated_rgb = []
    scene_scale = prog_scene.cameras_extent

    pre_vram = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[dense-init] VRAM before DAv2 context: {pre_vram:.0f} MB")

    def _run_cameras(depth_model):
        for cam in new_cams:
            sfm_xyz_all, visible_mask = prog_scene.get_sfm_points_visible_to(cam)
            sfm_xyz_visible = sfm_xyz_all[visible_mask]

            if len(sfm_xyz_visible) < dense_cfg.min_sfm_points_for_alignment:
                logger.info(
                    f"[dense-init] skip {cam.image_name}: "
                    f"only {len(sfm_xyz_visible)} SfM points visible"
                )
                continue

            depths_in_cam = transform_to_camera_frame(sfm_xyz_visible, cam)[:, 2]
            depth_range = float(depths_in_cam.max() - depths_in_cam.min())
            if depth_range < dense_cfg.min_sfm_depth_range_fraction * scene_scale:
                logger.info(
                    f"[dense-init] skip {cam.image_name}: "
                    f"SfM depth range too narrow ({depth_range:.3f} < "
                    f"{dense_cfg.min_sfm_depth_range_fraction * scene_scale:.3f})"
                )
                continue

            depth_map = depth_model.predict(cam.original_image.cpu())

            try:
                result = align_depth_to_sfm(
                    depth_map, cam, sfm_xyz_visible,
                    dense_cfg.ransac, scene_scale=scene_scale,
                )
            except AlignmentFailed as e:
                logger.warning(f"[dense-init] alignment failed for {cam.image_name}: {e}")
                del depth_map
                continue

            outcome = depth_to_points(
                depth_map, cam, result.a, result.b,
                target_n_points=dense_cfg.target_dense_points_per_image,
                sanity_threshold=dense_cfg.depth_disagreement_threshold * scene_scale,
                sfm_xyz_visible=sfm_xyz_visible,
                image_rgb=cam.original_image.cpu(),
                max_rejected_fraction=dense_cfg.max_rejected_fraction,
            )
            del depth_map

            if outcome is None:
                logger.warning(
                    f"[dense-init] sanity filter rejected too much for "
                    f"{cam.image_name}, skipping"
                )
                continue

            xyz, rgb = outcome
            accumulated_xyz.append(xyz)
            accumulated_rgb.append(rgb)
            logger.info(
                f"[dense-init] {cam.image_name}: kept {len(xyz)} dense points "
                f"(a={result.a:.3f}, b={result.b:.3f}, inliers={result.n_inliers})"
            )

    if dav2_model is not None:
        _run_cameras(dav2_model)
    else:
        with DepthAnythingV2Wrapper(dense_cfg.dav2) as depth_model:
            _run_cameras(depth_model)

    post_vram = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(
        f"[dense-init] VRAM after DAv2 context: {post_vram:.0f} MB "
        f"(delta {post_vram - pre_vram:+.0f})"
    )
    if post_vram - pre_vram > 100:
        logger.warning(
            f"[dense-init] WARNING: DAv2 leaked {post_vram - pre_vram:.0f} MB of VRAM! "
            "Check for reference leaks."
        )

    if not accumulated_xyz:
        logger.info("[dense-init] no points produced for this snapshot")
        return

    all_xyz = np.concatenate(accumulated_xyz, axis=0)
    all_rgb = np.concatenate(accumulated_rgb, axis=0)
    del accumulated_xyz, accumulated_rgb

    # Novelty filter — only add points far from existing Gaussians
    existing_xyz = gaussians.get_xyz.detach().cpu().numpy()
    threshold = dense_cfg.novelty_distance_threshold * scene_scale
    if len(existing_xyz) > 0:
        tree = cKDTree(existing_xyz)
        distances, _ = tree.query(all_xyz)
        novel_mask = distances >= threshold
    else:
        novel_mask = np.ones(len(all_xyz), dtype=bool)

    n_novel = int(novel_mask.sum())
    if n_novel == 0:
        logger.info("[dense-init] all dense points covered by existing Gaussians — nothing added")
        return

    novel_pcd = BasicPointCloud(
        points=all_xyz[novel_mask],
        colors=all_rgb[novel_mask],
        normals=np.zeros((n_novel, 3), dtype=np.float32),
    )

    n_before = gaussians.get_xyz.shape[0]
    full_mask = np.ones(n_novel, dtype=bool)
    gaussians.expand_from_pcd(novel_pcd, full_mask, scene_scale)
    n_after = gaussians.get_xyz.shape[0]
    gaussians.mark_recently_added(
        slice(n_before, n_after),
        iteration=current_iter,
        grace_iters=dense_cfg.grace_iters,
    )
    elapsed = time.monotonic() - t_start
    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 2
    logger.info(
        f"[dense-init] added {n_novel} dense Gaussians "
        f"(rejected {len(all_xyz) - n_novel} as redundant); "
        f"wall={elapsed:.1f}s peak_vram={peak_vram:.0f}MB"
    )


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model_path: str, gaussians: SphericalGaussianModel, snapshot_idx=None, tag=None):
    if tag is None:
        tag = snapshot_idx
    point_cloud_path = os.path.join(model_path, f"point_cloud/iteration_{tag}")
    os.makedirs(point_cloud_path, exist_ok=True)
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    logger.info(f"[save] checkpoint → {point_cloud_path}/point_cloud.ply")


# ---------------------------------------------------------------------------
# Per-phase training window
# ---------------------------------------------------------------------------

def train_window(
    gaussians: SphericalGaussianModel,
    prog_scene: ProgressiveScene,
    opt,
    pipe,
    config: ProgressiveConfig,
    n_iters: int,
    phase: str,  # "initial" | "merge" | "final"
    new_cam_indices: Optional[List[int]] = None,
    global_iter_start: int = 0,
    diary_file=None,
    bg_white: bool = False,
):
    """Run up to `n_iters` training iterations for the given phase.

    Returns the number of iterations actually consumed (may be < n_iters when
    the convergence monitor reports early completion).
    """
    training_cfg = config.training
    K = max(1, training_cfg.accumulation_views)  # T6: views per optimizer step

    bg_color = [1, 1, 1] if bg_white else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # T7: convergence monitor + trigger state
    monitor = ConvergenceMonitor()
    last_densify_iter = 0
    n_at_last_prune = gaussians._xyz.shape[0]
    has_culled = False
    soft_cap = int(training_cfg.num_max_ceiling * 0.85)

    # T8: SkipGS view-adaptive backward gate (configured phase only)
    skipgs = None
    if training_cfg.skipgs.enabled and phase == training_cfg.skipgs.phase:
        skipgs = SkipGSGate(
            warmup=training_cfg.skipgs.warmup,
            beta=training_cfg.skipgs.beta,
            eps=training_cfg.skipgs.eps,
            rho_lo=training_cfg.skipgs.rho_lo,
        )

    # Position LR: in merge phase freeze at mid-schedule value to stabilise old Gaussians
    if phase == "merge":
        mid_lr = gaussians.xyz_scheduler_args(n_iters // 2)
        for pg in gaussians.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = mid_lr

    # In final phase, restart LR from iteration 0 of this phase
    if phase == "final":
        gaussians.update_learning_rate(0)

    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
    viewpoint_stack = None
    image_weights = prog_scene.image_weights if phase == "merge" else None

    progress_bar = tqdm(range(n_iters), desc=f"[{phase}]")
    ema_loss = 0.0
    phase_iter = 0  # track in case loop body never executes

    for phase_iter in range(1, n_iters + 1):
        global_iter = global_iter_start + phase_iter

        # ---- LR update ----
        if phase == "initial":
            gaussians.update_learning_rate(phase_iter)
        elif phase == "final":
            gaussians.update_learning_rate(phase_iter)
            if phase_iter % 1000 == 0 and phase_iter > n_iters // 4:
                gaussians.oneupSGdegree()
        # merge: xyz LR frozen at mid-schedule (set once above)

        # ---- T6: K-view inner sub-loop ----
        # per_view_packs: list of (loss_tensor, render_pkg, cam_uid, cam_name, cam_weight)
        per_view_packs = []
        for _sub in range(K):
            if phase == "merge":
                candidates = select_cameras_weighted(
                    prog_scene.train_cameras, image_weights, n_top=10, n_random=10,
                )
                if not candidates:
                    candidates = prog_scene.train_cameras
                vcam = candidates[randint(0, len(candidates) - 1)]
                cam_idx = prog_scene._cam_name_to_idx.get(vcam.image_name, 0)
                cam_w = (float(image_weights[cam_idx])
                         if image_weights is not None and cam_idx < len(image_weights)
                         else 1.0)
            else:
                if not viewpoint_stack:
                    viewpoint_stack = prog_scene.train_cameras.copy()
                vcam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                cam_idx = prog_scene._cam_name_to_idx.get(vcam.image_name, 0)
                cam_w = 1.0

            bg = torch.rand((3,), device="cuda") if opt.random_background else background
            render_pkg = render_imp(vcam, gaussians, pipe, bg, is_training=True)
            img = render_pkg["render"]
            gt_image = vcam.original_image.cuda()

            Ll1 = l1_loss(img, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_val = fused_ssim(img.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_val = ssim(img, gt_image)

            base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
            loss_k = cam_w * base_loss if phase == "merge" else base_loss

            per_view_packs.append((loss_k, render_pkg, vcam.uid, vcam.image_name, cam_w))

        # ---- T8: EMA update + gate decision ----
        if skipgs is not None:
            for loss_t, _, cam_id, _, _ in per_view_packs:
                skipgs.update_ema(cam_id, float(loss_t.detach()))
            devs = [skipgs.deviation(cam_id, float(loss_t.detach()))
                    for loss_t, _, cam_id, _, _ in per_view_packs]
            gate, _ = skipgs.decide(devs)
        else:
            gate = [True] * K

        contributing = [loss_t
                        for (loss_t, _, _, _, _), g in zip(per_view_packs, gate) if g]

        if not contributing:
            # Fully-skipped step — update visibility stats only, no backward/step
            if skipgs is not None:
                skipgs.record_backward(False)
            with torch.no_grad():
                for _, render_pkg, _, _, _ in per_view_packs:
                    vf = render_pkg["visibility_filter"]
                    rad = render_pkg["radii"]
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rad[vf])
                    am = render_pkg["area_max"]
                    ri = render_pkg["render"]
                    mbp = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                        mbp[:mask_blur.shape[0]] = mask_blur
                    mask_blur = torch.logical_or(mbp, am > (ri.shape[1] * ri.shape[2] / 5000))
            continue

        # ---- Backward (sum of contributing views) ----
        loss = sum(contributing)
        loss.backward()

        if skipgs is not None:
            skipgs.record_backward(True)

        # ---- Diary logging (merge phase lists all K cameras) ----
        if phase == "merge" and diary_file is not None:
            cam_line = ",".join(f"{name}:{w:.3f}" for _, _, _, name, w in per_view_packs)
            diary_file.write(f"iter={global_iter} cams=[{cam_line}]\n")

        # ---- no_grad: stats, T7 triggers, convergence check ----
        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if phase_iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}",
                                          "N": gaussians._xyz.shape[0]})
                progress_bar.update(10)

            monitor.update_loss(ema_loss)

            # Densification stats: visibility for all K views; gradient only for contributors
            for (_, render_pkg, _, _, _), g in zip(per_view_packs, gate):
                vf = render_pkg["visibility_filter"]
                rad = render_pkg["radii"]
                am = render_pkg["area_max"]
                ri = render_pkg["render"]

                gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rad[vf])

                mbp = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                    mbp[:mask_blur.shape[0]] = mask_blur
                mask_blur = torch.logical_or(mbp, am > (ri.shape[1] * ri.shape[2] / 5000))

                if g:  # T8: only accumulate gradient stats for views that contributed
                    gaussians.add_densification_stats(render_pkg["viewspace_points"], vf)

            # T7: fast opacity/size prune every iter (cheap)
            if should_fast_prune(gaussians, opt, training_cfg.fast_prune_dead_fraction):
                gaussians.opacity_size_prune(
                    min_opacity=0.005,
                    max_screen_size=20 if phase == "initial" else None,
                    extent=prog_scene.cameras_extent,
                )

            # T7: evidence-based densify (checked every 50 iters, gated by ceiling)
            if (phase_iter % 50 == 0
                    and gaussians._xyz.shape[0] < training_cfg.num_max_ceiling):
                if should_densify(gaussians, opt, mask_blur,
                                  last_densify_iter, phase_iter,
                                  min_settling=training_cfg.densify_min_settling,
                                  candidate_fraction=training_cfg.densify_candidate_fraction,
                                  max_interval=training_cfg.densify_max_interval):
                    n_before = gaussians._xyz.shape[0]
                    size_threshold = 20 if phase == "initial" else None
                    gaussians.densify_and_prune_split(
                        opt.densify_grad_threshold,
                        0.005,
                        prog_scene.cameras_extent,
                        size_threshold,
                        mask_blur[:gaussians.xyz_gradient_accum.shape[0]],
                    )
                    n_after = gaussians._xyz.shape[0]
                    monitor.update_densify(n_after - n_before, n_after)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    last_densify_iter = phase_iter

            # T7: lightweight importance prune (every 200 iters, non-initial phases)
            if phase_iter % 200 == 0 and phase != "initial":
                if should_lightweight_prune(gaussians, monitor,
                                            n_at_last_prune, soft_cap,
                                            growth_threshold=training_cfg.lightweight_prune_growth_threshold):
                    lightweight_prune(gaussians, prog_scene, opt, config, global_iter)
                    n_at_last_prune = gaussians._xyz.shape[0]

            # T7: late-phase SG axis cull (final phase only, triggered once at 85%+)
            if (phase == "final"
                    and phase_iter > n_iters * 0.85
                    and not has_culled
                    and should_cull_sg_axes(gaussians, training_cfg.sharpness_threshold,
                                           fraction_low=training_cfg.sg_axis_cull_low_fraction)):
                gaussians.cull_low_sharpness_axes(
                    sharpness_threshold=training_cfg.sharpness_threshold)
                has_culled = True
                torch.cuda.empty_cache()
                logger.info(f"[{phase}] SG axis cull triggered at iter {phase_iter}")

            # T7: convergence-driven early termination
            converged = monitor.state() == "converged"

        if converged:
            logger.info(f"[{phase}] converged at iter {phase_iter}/{n_iters}")
            break

        # Optimizer step (skip on the very last cap-terminated iter for consistency)
        if phase_iter < n_iters:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

    # T8: per-phase SkipGS summary
    if skipgs is not None and skipgs._step > 0:
        rho_cum = skipgs._backward_count / max(skipgs._step, 1)
        logger.info(
            f"[skipgs] rho_min={skipgs._rho_min:.3f} "
            f"steps={skipgs._step} backward={skipgs._backward_count} "
            f"rho_cum={rho_cum:.3f}"
        )

    progress_bar.close()
    return phase_iter  # actual iterations consumed


# ---------------------------------------------------------------------------
# Lightweight merge-phase prune
# ---------------------------------------------------------------------------

def lightweight_prune(gaussians, prog_scene, opt, config, global_iter):
    """Run a single low-intensity prune pass during merge phase."""
    logger.info("[prune] Running lightweight prune pass")
    pipe_dummy = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    subsample_n = config.training.imp_score_camera_subsample
    imp_score = update_imp_score(
        prog_scene.train_cameras, gaussians, pipe_dummy, background,
        imp_metric=config.training.imp_metric,
        subsample_n=subsample_n,
        weights=prog_scene.image_weights,
    )
    grace_mask = gaussians.get_grace_protected_mask(global_iter)
    threshold = int(config.training.prune_ratio1 * imp_score.shape[0])
    imp_sorted, _ = torch.sort(imp_score, 0)
    imp_thresh = imp_sorted[max(threshold - 1, 0)]
    prune_mask = (imp_score <= imp_thresh).squeeze()
    prune_mask = prune_mask & ~grace_mask
    before = gaussians._xyz.shape[0]
    gaussians.prune_points(prune_mask)
    after = gaussians._xyz.shape[0]
    logger.info(f"[prune] {before} → {after} Gaussians")
    torch.cuda.empty_cache()


def fast_final_compression(gaussians, prog_scene, opt, pipe, config, global_iter):
    """Single-shot importance prune + SG axis cull after the final phase.

    Replaces the historical destructive reinitial_pts reset + reconverge pass.
    Runs once after the convergence-driven final phase exits.
    """
    pipe_dummy = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    subsample_n = config.training.imp_score_camera_subsample
    imp_score = update_imp_score(
        prog_scene.train_cameras, gaussians, pipe_dummy, background,
        imp_metric=config.training.imp_metric,
        subsample_n=subsample_n,
        weights=prog_scene.image_weights,
    )
    grace_mask = gaussians.get_grace_protected_mask(global_iter)
    ratio = config.training.fast_final_prune_ratio
    threshold = int(ratio * imp_score.shape[0])
    imp_sorted, _ = torch.sort(imp_score, 0)
    cutoff = imp_sorted[max(threshold - 1, 0)]
    prune_mask = (imp_score <= cutoff).squeeze() & ~grace_mask
    n_before = gaussians._xyz.shape[0]
    gaussians.prune_points(prune_mask)
    gaussians.cull_low_sharpness_axes(
        sharpness_threshold=config.training.sharpness_threshold)
    torch.cuda.empty_cache()
    logger.info(f"[fast-final] {n_before} → {gaussians._xyz.shape[0]} Gaussians, SG axes culled")


# ---------------------------------------------------------------------------
# Main progressive training loop
# ---------------------------------------------------------------------------

def progressive_training(dataset, opt, pipe, args, config: ProgressiveConfig):
    source_path_dir = args.source_path_dir
    model_path = dataset.model_path
    os.makedirs(model_path, exist_ok=True)

    # Diary file for Phase 6 weight logging
    diary_path = os.path.join(model_path, "progressive_diary.txt")
    diary_file = open(diary_path, "w")

    gaussians = SphericalGaussianModel(dataset.sg_degree)
    prog_scene = ProgressiveScene(
        source_path_dir=source_path_dir,
        model_args=dataset,
        images_subdir=getattr(dataset, 'images', 'images'),
    )

    logger.info(f"Found {len(prog_scene.snapshot_indices)} snapshots: "
                f"{prog_scene.snapshot_indices}")

    # ---- Snapshot 0: initial training ----
    new_cams, new_point_mask, new_cam_indices = prog_scene.load_next_snapshot()
    logger.info(
        f"[snap 0] Loaded {len(prog_scene.train_cameras)} cameras, "
        f"{prog_scene.current_basic_pcd.points.shape[0]} SfM points"
    )
    gaussians.create_from_pcd(prog_scene.current_basic_pcd, prog_scene.cameras_extent)
    gaussians.training_setup(opt)

    # T4: optionally open DAv2 once and hold it for all snapshots
    _persistent_dav2 = None
    if config.dense_init.persist_model:
        logger.info("[dense-init] T4: opening persistent DAv2 model ...")
        _persistent_dav2 = DepthAnythingV2Wrapper(config.dense_init.dav2).__enter__()

    # Dense init for the initial snapshot (same logic as merge phase)
    if config.dense_init.enabled and new_cams:
        logger.info(f"[dense_init] Running on initial {len(new_cams)} cameras ...")
        dense_init_for_new_images(
            gaussians, prog_scene, new_cams,
            config.dense_init, opt, current_iter=0,
            dav2_model=_persistent_dav2,
        )

    bg_white = getattr(dataset, 'white_background', False)
    global_iter = 0
    global_iter += train_window(
        gaussians, prog_scene, opt, pipe, config,
        n_iters=config.training.iter_initial,
        phase="initial",
        global_iter_start=global_iter,
        diary_file=diary_file,
        bg_white=bg_white,
    )

    if config.snapshots.progressive_output:
        save_checkpoint(model_path, gaussians, snapshot_idx=0)

    # ---- Progressive ingestion loop ----
    snapshot_count = 0
    while prog_scene.has_more_snapshots():
        snapshot_count += 1
        snap_idx = prog_scene.snapshot_indices[prog_scene._current_snapshot_pos]

        logger.info(f"[snap {snap_idx}] Loading snapshot {snap_idx} ...")
        new_cams, new_point_mask, new_cam_indices = prog_scene.load_next_snapshot()
        logger.info(
            f"[snap {snap_idx}] +{len(new_cams)} new cameras, "
            f"{new_point_mask.sum()} new SfM points, "
            f"total cams={len(prog_scene.train_cameras)}"
        )

        # Phase 4: seed Gaussians from new sparse SfM points
        expand_gaussians_from_new_points(gaussians, prog_scene, new_point_mask, opt)

        # Phase 5: optionally seed dense Gaussians from monocular depth
        skip_n = config.dense_init.skip_first_n_snapshots
        if (config.dense_init.enabled
                and prog_scene.current_idx is not None
                and snapshot_count > skip_n
                and new_cams):
            dense_init_for_new_images(
                gaussians, prog_scene, new_cams,
                config.dense_init, opt, current_iter=global_iter,
                dav2_model=_persistent_dav2,
            )

        # Phase 6: parse match matrix and compute image weights
        matrix_path = prog_scene.snapshot_dir / "sparse/0/imageMatchMatrix.txt"
        names_path = prog_scene.snapshot_dir / "sparse/0/imagesNames.txt"
        if matrix_path.exists() and names_path.exists():
            ordered_names = [cam.image_name for cam in prog_scene.train_cameras]
            prog_scene.image_match_matrix = parse_match_matrix(
                str(matrix_path), str(names_path), ordered_names,
            )
            prog_scene.image_weights = compute_image_weights(
                prog_scene.image_match_matrix, new_cam_indices,
            )
            logger.info(
                f"[snap {snap_idx}] match-matrix parsed; "
                f"new-cam weights={[f'{prog_scene.image_weights[i]:.2f}' for i in new_cam_indices]}"
            )
        else:
            # No match matrix — use uniform weights with bias toward new cameras
            M = len(prog_scene.train_cameras)
            weights = np.ones(M, dtype=np.float32) * 0.5
            for i in new_cam_indices:
                weights[i] = 1.0
            prog_scene.image_weights = weights

        # Merge training window
        n_merge = max(1, config.training.iter_per_merge * max(len(new_cams), 1))
        global_iter += train_window(
            gaussians, prog_scene, opt, pipe, config,
            n_iters=n_merge,
            phase="merge",
            new_cam_indices=new_cam_indices,
            global_iter_start=global_iter,
            diary_file=diary_file,
            bg_white=bg_white,
        )

        # T7: cap-breach safety valve (replaces prune_every_n_snapshots)
        soft_cap = int(config.training.num_max_ceiling * 0.85)
        if gaussians._xyz.shape[0] > soft_cap:
            lightweight_prune(gaussians, prog_scene, opt, config, global_iter)

        if config.snapshots.progressive_output:
            save_checkpoint(model_path, gaussians, snapshot_idx=snap_idx)

    # T4: free persistent DAv2 before final phase (reclaim ~1.3 GB VRAM)
    if _persistent_dav2 is not None:
        _persistent_dav2.__exit__(None, None, None)
        _persistent_dav2 = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("[dense-init] T4: released persistent DAv2 model before final phase")

    # ---- Final refinement phase ----
    logger.info(f"[final] Starting final refinement (cap={config.training.iter_final} iters) ...")
    gaussians.update_learning_rate(0)

    global_iter += train_window(
        gaussians, prog_scene, opt, pipe, config,
        n_iters=config.training.iter_final,
        phase="final",
        global_iter_start=global_iter,
        diary_file=diary_file,
        bg_white=bg_white,
    )

    # T7: fast final compression — single-shot importance prune + SG axis cull
    logger.info("[final] Running fast_final_compression ...")
    fast_final_compression(gaussians, prog_scene, opt, pipe, config, global_iter)

    save_checkpoint(model_path, gaussians, snapshot_idx="final")
    diary_file.close()

    logger.info(
        f"Progressive training complete. "
        f"Final Gaussian count: {gaussians.get_xyz.shape[0]}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Progressive MEGS² training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--source_path_dir", required=True, type=str,
                        help="Root dir of numbered snapshot folders (e.g. 16/, 17/, ...)")
    parser.add_argument("--config", type=str, default="configs/progressive.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dense_init_enabled", action="store_true", default=None,
                        help="Override config: enable dense init")
    parser.add_argument("--dense_init_disabled", action="store_true", default=None,
                        help="Override config: disable dense init")
    parser.add_argument("--imp_metric", type=str, default=None,
                        help="Override importance metric: outdoor | indoor")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    config = load_config(args.config)

    if args.dense_init_enabled:
        config.dense_init.enabled = True
    if args.dense_init_disabled:
        config.dense_init.enabled = False
    if args.imp_metric:
        config.training.imp_metric = args.imp_metric

    config.snapshots.source_path_dir = args.source_path_dir

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if not dataset.model_path:
        dataset.model_path = os.path.join("./output/progressive", str(uuid.uuid4())[:8])

    os.makedirs(dataset.model_path, exist_ok=True)
    logger.info(f"Output → {dataset.model_path}")

    # Save run config for reproducibility
    with open(os.path.join(dataset.model_path, "run_config.yaml"), "w") as f:
        yaml.dump(
            {
                "source_path_dir": args.source_path_dir,
                "dense_init_enabled": config.dense_init.enabled,
                "imp_metric": config.training.imp_metric,
            },
            f,
        )

    progressive_training(dataset, opt, pipe, args, config)


if __name__ == "__main__":
    main()
