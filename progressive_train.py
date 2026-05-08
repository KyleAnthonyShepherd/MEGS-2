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
from spherical_gaussian_renderer import render_depth, render_imp
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import BasicPointCloud
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB
from optimizing_spa import OptimizingSpa
from optimizing_spa_sg import OptimizingSpaSG

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
class TrainingConfig:
    iter_initial: int = 3000
    iter_per_merge: int = 200
    iter_final: int = 4000
    num_max: int = 800_000
    prune_every_n_snapshots: int = 5
    simp_iteration1_frac: float = 0.40
    optimizing_spa_start_iter_frac: float = 0.50
    optimizing_spa_stop_iter_frac: float = 0.80
    optimizing_spa_sg_stop_iter_frac: float = 0.95
    prune_ratio1: float = 0.05
    prune_ratio2: float = 0.05
    sharpness_threshold: float = 1.0
    optimizing_spa_interval: int = 100
    merge_densification_interval: int = 100
    imp_metric: str = "outdoor"


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
            if hasattr(cfg.training, k):
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

def update_imp_score(cameras, gaussians, pipe, background, imp_metric="outdoor"):
    imp_score = torch.zeros(gaussians._xyz.shape[0]).cuda()
    accum_area_max = torch.zeros(gaussians._xyz.shape[0]).cuda()
    for view in cameras:
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
):
    """Run DAv2 on each new image, align to SfM, filter, append as Gaussians."""
    accumulated_xyz = []
    accumulated_rgb = []
    scene_scale = prog_scene.cameras_extent

    pre_vram = torch.cuda.memory_allocated() / 1024 ** 2
    logger.info(f"[dense-init] VRAM before DAv2 context: {pre_vram:.0f} MB")

    with DepthAnythingV2Wrapper(dense_cfg.dav2) as depth_model:
        for cam in new_cams:
            sfm_xyz_all, visible_mask = prog_scene.get_sfm_points_visible_to(cam)
            sfm_xyz_visible = sfm_xyz_all[visible_mask]

            if len(sfm_xyz_visible) < dense_cfg.min_sfm_points_for_alignment:
                logger.info(
                    f"[dense-init] skip {cam.image_name}: "
                    f"only {len(sfm_xyz_visible)} SfM points visible"
                )
                continue

            # Depth-range check (avoid planar-scene rank deficiency)
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
    logger.info(
        f"[dense-init] added {n_novel} dense Gaussians "
        f"(rejected {len(all_xyz) - n_novel} as redundant)"
    )


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model_path: str, gaussians: SphericalGaussianModel, tag):
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
):
    """Run `n_iters` training iterations for the given phase.

    Returns the number of global iterations consumed.
    """
    training_cfg = config.training

    bg_color = [1, 1, 1] if getattr(opt, 'white_background', False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Phase-specific derived thresholds (for final phase only)
    simp_iteration1 = int(n_iters * training_cfg.simp_iteration1_frac)
    optimizing_spa_start_iter = int(n_iters * training_cfg.optimizing_spa_start_iter_frac)
    optimizing_spa_stop_iter = int(n_iters * training_cfg.optimizing_spa_stop_iter_frac)
    optimizing_spa_sg_stop_iter = int(n_iters * training_cfg.optimizing_spa_sg_stop_iter_frac)
    optimizing_spa_sg_start_iter = optimizing_spa_start_iter  # same as SPA start

    # Position LR: in merge phase freeze at mid-warmup value to stabilise old Gaussians
    if phase == "merge":
        mid_lr = gaussians.xyz_scheduler_args(simp_iteration1 // 2)
        for pg in gaussians.optimizer.param_groups:
            if pg["name"] == "xyz":
                pg["lr"] = mid_lr

    # In final phase, restart LR from iteration 0 of this phase
    if phase == "final":
        gaussians.update_learning_rate(0)

    optimizingSpa = None
    optimizingSpaSg = None
    imp_score = torch.zeros(gaussians._xyz.shape[0], device="cuda")
    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
    viewpoint_stack = None

    # Build per-camera weight lookup for merge phase
    image_weights = prog_scene.image_weights if phase == "merge" else None

    desc = f"[{phase}]"
    progress_bar = tqdm(range(n_iters), desc=desc)
    ema_loss = 0.0

    for phase_iter in range(1, n_iters + 1):
        global_iter = global_iter_start + phase_iter

        # ---- LR update ----
        if phase in ("initial", "merge"):
            if phase == "initial":
                gaussians.update_learning_rate(phase_iter)
        else:  # final
            if phase_iter < simp_iteration1:
                gaussians.update_learning_rate(phase_iter)
            else:
                gaussians.update_learning_rate(phase_iter - simp_iteration1 + 5000)
            if phase_iter % 1000 == 0 and phase_iter > simp_iteration1:
                gaussians.oneupSGdegree()

        # ---- Camera selection ----
        if phase == "merge":
            candidates = select_cameras_weighted(
                prog_scene.train_cameras, image_weights,
                n_top=10, n_random=10,
            )
            if not candidates:
                candidates = prog_scene.train_cameras
            viewpoint_cam = candidates[randint(0, len(candidates) - 1)]
            cam_idx = prog_scene._cam_name_to_idx.get(viewpoint_cam.image_name, 0)
        else:
            if not viewpoint_stack:
                viewpoint_stack = prog_scene.train_cameras.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            cam_idx = prog_scene._cam_name_to_idx.get(viewpoint_cam.image_name, 0)

        # ---- Forward pass ----
        bg = torch.rand((3,), device="cuda") if opt.random_background else background
        render_pkg = render_imp(viewpoint_cam, gaussians, pipe, bg, is_training=True)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        area_max = render_pkg["area_max"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        base_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Phase 6: scale loss by camera weight in merge phase
        if phase == "merge" and image_weights is not None and cam_idx < len(image_weights):
            w = float(image_weights[cam_idx])
            loss = w * base_loss
        else:
            loss = base_loss

        # OptimizingSpa regulariser (final phase only)
        if phase == "final" and opt.optimizing_spa:
            if (optimizingSpa is not None
                    and phase_iter > optimizing_spa_start_iter
                    and phase_iter % training_cfg.optimizing_spa_interval == 0
                    and phase_iter < optimizing_spa_stop_iter):
                loss = optimizingSpa.append_spa_loss(loss)
            if (optimizingSpaSg is not None
                    and phase_iter > optimizing_spa_sg_start_iter
                    and phase_iter % training_cfg.optimizing_spa_interval == 0
                    and phase_iter < optimizing_spa_sg_stop_iter):
                loss = optimizingSpaSg.append_spa_loss_sg(loss)

        loss.backward()

        # ---- Diary logging (Phase 6) ----
        if phase == "merge" and diary_file is not None:
            w_log = float(image_weights[cam_idx]) if (image_weights is not None and cam_idx < len(image_weights)) else 1.0
            diary_file.write(f"iter={global_iter} cam={viewpoint_cam.image_name} weight={w_log:.4f}\n")

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if phase_iter % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}",
                                          "N": gaussians._xyz.shape[0]})
                progress_bar.update(10)

            # ---- Densification (initial and merge phases; lighter in merge) ----
            densify_interval = (
                training_cfg.merge_densification_interval
                if phase == "merge" else opt.densification_interval
            )
            if phase in ("initial", "merge") and gaussians._xyz.shape[0] < training_cfg.num_max:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                mask_blur_padded = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                    mask_blur_padded[:mask_blur.shape[0]] = mask_blur
                mask_blur = mask_blur_padded
                mask_blur = torch.logical_or(
                    mask_blur,
                    area_max > (image.shape[1] * image.shape[2] / 5000)
                )

                if (phase_iter > opt.densify_from_iter
                        and phase_iter % densify_interval == 0):
                    size_threshold = 20 if phase == "initial" else None
                    gaussians.densify_and_prune_split(
                        opt.densify_grad_threshold,
                        0.005,
                        prog_scene.cameras_extent,
                        size_threshold,
                        mask_blur[:gaussians.xyz_gradient_accum.shape[0]],
                    )
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")

            # ---- Final-phase MEGS-2 full schedule ----
            if phase == "final":
                if phase_iter < opt.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    mask_blur_padded = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                        mask_blur_padded[:mask_blur.shape[0]] = mask_blur
                    mask_blur = mask_blur_padded
                    mask_blur = torch.logical_or(
                        mask_blur,
                        area_max > (image.shape[1] * image.shape[2] / 5000)
                    )
                    if (phase_iter > opt.densify_from_iter
                            and phase_iter % opt.densification_interval == 0
                            and phase_iter % 5000 != 0
                            and gaussians._xyz.shape[0] < training_cfg.num_max):
                        size_threshold = 20 if phase_iter > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune_split(
                            opt.densify_grad_threshold,
                            0.005,
                            prog_scene.cameras_extent,
                            size_threshold,
                            mask_blur[:gaussians.xyz_gradient_accum.shape[0]],
                        )
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")

                # Depth-based reinit (final phase only, never in merge to protect new Gaussians)
                if phase_iter % 5000 == 0:
                    out_pts_list, gt_list = [], []
                    views = prog_scene.train_cameras
                    for view in views:
                        gt = view.original_image[0:3, :, :]
                        rdpkg = render_depth(view, gaussians, pipe, background)
                        out_pts = rdpkg["out_pts"]
                        accum_alpha = rdpkg["accum_alpha"]
                        prob = 1 - accum_alpha
                        prob = prob / prob.sum()
                        prob = prob.reshape(-1).cpu().numpy()
                        factor = 1 / (
                            image.shape[1] * image.shape[2] * len(views)
                            / getattr(opt, 'num_depth', 3_500_000)
                        )
                        N_xyz = prob.shape[0]
                        num_sampled = int(N_xyz * factor)
                        indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
                        out_pts = out_pts.permute(1, 2, 0).reshape(-1, 3)
                        gt = gt.permute(1, 2, 0).reshape(-1, 3)
                        out_pts_list.append(out_pts[indices])
                        gt_list.append(gt[indices])
                    out_pts_merged = torch.cat(out_pts_list)
                    gt_merged = torch.cat(gt_list)
                    gaussians.reinitial_pts(out_pts_merged, gt_merged)
                    gaussians.training_setup(opt)
                    mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    torch.cuda.empty_cache()
                    viewpoint_stack = None

                # OptimizingSpa init
                if opt.optimizing_spa:
                    if phase_iter == optimizing_spa_start_iter:
                        imp_score = update_imp_score(
                            prog_scene.train_cameras, gaussians, pipe, background,
                            imp_metric=training_cfg.imp_metric,
                        )
                        optimizingSpa = OptimizingSpa(
                            gaussians, opt, "cuda", imp_score_flag=True)
                        optimizingSpa.update(imp_score, update_u=False)

                    if phase_iter == optimizing_spa_sg_start_iter:
                        imp_sg_score = update_sg_color_diff(gaussians)
                        optimizingSpaSg = OptimizingSpaSG(
                            gaussians, opt, "cuda", imp_score_flag=True)
                        optimizingSpaSg.update(imp_sg_score, update_u=False)

                    if (phase_iter > optimizing_spa_start_iter
                            and phase_iter % training_cfg.optimizing_spa_interval == 0):
                        if phase_iter <= optimizing_spa_stop_iter:
                            imp_score = update_imp_score(
                                prog_scene.train_cameras, gaussians, pipe, background,
                                imp_metric=training_cfg.imp_metric,
                            )
                            optimizingSpa.update(imp_score)
                        if phase_iter <= optimizing_spa_sg_stop_iter:
                            imp_sg_score = update_sg_color_diff(gaussians)
                            optimizingSpaSg.update(imp_sg_score)

                # Prune at simp_iteration1
                if phase_iter == simp_iteration1:
                    imp_score = update_imp_score(
                        prog_scene.train_cameras, gaussians, pipe, background,
                        imp_metric=training_cfg.imp_metric,
                    )
                    grace_mask = gaussians.get_grace_protected_mask(global_iter)
                    prob = (imp_score + 1) / (imp_score + 1).sum()
                    prob = prob.cpu().numpy()
                    N_xyz = gaussians._xyz.shape[0]
                    num_sampled = int(N_xyz * (1 - training_cfg.prune_ratio1))
                    indices = np.random.choice(N_xyz, size=num_sampled, p=prob, replace=False)
                    mask = np.zeros(N_xyz, dtype=bool)
                    mask[indices] = True
                    # Never prune grace-protected Gaussians
                    grace_np = grace_mask.cpu().numpy()
                    mask = mask | grace_np
                    gaussians.prune_points(mask == False)
                    gaussians.max_sg_degree = gaussians.max_sg_degree
                    gaussians.reinitial_pts(gaussians._xyz, SH2RGB(gaussians._rgb_base))
                    gaussians.training_setup(opt)
                    torch.cuda.empty_cache()
                    viewpoint_stack = None

                # Second prune at optimizing_spa_stop_iter
                if phase_iter == optimizing_spa_stop_iter:
                    imp_score = update_imp_score(
                        prog_scene.train_cameras, gaussians, pipe, background,
                        imp_metric=training_cfg.imp_metric,
                    )
                    grace_mask = gaussians.get_grace_protected_mask(global_iter)
                    threshold = int(training_cfg.prune_ratio2 * imp_score.shape[0])
                    imp_sorted, _ = torch.sort(imp_score, 0)
                    imp_threshold = imp_sorted[max(threshold - 1, 0)]
                    prune_mask = (imp_score <= imp_threshold).squeeze()
                    # Protect grace-period Gaussians
                    prune_mask = prune_mask & ~grace_mask
                    logger.info(f"[final] Before 2nd prune: {gaussians.get_opacity.shape[0]}")
                    gaussians.prune_points(prune_mask)
                    logger.info(f"[final] After 2nd prune: {gaussians.get_opacity.shape[0]}")
                    torch.cuda.empty_cache()

                # SG axis culling
                if phase_iter == optimizing_spa_sg_stop_iter:
                    logger.info(f"[final] SG axis culling (threshold={training_cfg.sharpness_threshold})")
                    gaussians.cull_low_sharpness_axes(
                        sharpness_threshold=training_cfg.sharpness_threshold)
                    torch.cuda.empty_cache()

            # ---- Optimizer step ----
            if phase_iter < n_iters:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    progress_bar.close()
    return n_iters


# ---------------------------------------------------------------------------
# Lightweight merge-phase prune
# ---------------------------------------------------------------------------

def lightweight_prune(gaussians, prog_scene, opt, config, global_iter):
    """Run a single low-intensity prune pass during merge phase."""
    logger.info("[prune] Running lightweight prune pass")
    pipe_dummy = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    imp_score = update_imp_score(
        prog_scene.train_cameras, gaussians, pipe_dummy, background,
        imp_metric=config.training.imp_metric,
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

    global_iter = 0
    global_iter += train_window(
        gaussians, prog_scene, opt, pipe, config,
        n_iters=config.training.iter_initial,
        phase="initial",
        global_iter_start=global_iter,
        diary_file=diary_file,
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
        )

        # Phase 7: lightweight prune every N snapshots
        if snapshot_count % config.training.prune_every_n_snapshots == 0:
            lightweight_prune(gaussians, prog_scene, opt, config, global_iter)

        if config.snapshots.progressive_output:
            save_checkpoint(model_path, gaussians, snapshot_idx=snap_idx)

    # ---- Final refinement phase ----
    logger.info(f"[final] Starting final refinement ({config.training.iter_final} iters) ...")
    # Reset LR schedule for final phase
    gaussians.update_learning_rate(0)

    global_iter += train_window(
        gaussians, prog_scene, opt, pipe, config,
        n_iters=config.training.iter_final,
        phase="final",
        global_iter_start=global_iter,
        diary_file=diary_file,
    )

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
