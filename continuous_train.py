"""Continuous MEGS² training entry point — unbounded streaming mode.

Images arrive via POST /ingest from an external COLMAP service.  Training runs
until monitor.state()=="converged" with an empty ingest queue, then idles.
There are no phase caps; convergence is the only timing signal.

Usage:
    python continuous_train.py \\
        --model_path /path/to/output \\
        --config configs/continuous.yaml \\
        [--http_host 127.0.0.1] \\
        [--http_port 8765]
"""

import gc
import logging
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from pathlib import Path
from random import randint
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

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
from scene.control_state import ControlState
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
class TriggerConfig:
    min_iters_between: int = 60
    require_state: list = field(default_factory=lambda: ["wants_capacity"])


@dataclass
class TriggersConfig:
    densify: TriggerConfig = field(
        default_factory=lambda: TriggerConfig(60, ["stalled", "wants_capacity"]))
    fast_prune: TriggerConfig = field(
        default_factory=lambda: TriggerConfig(200, ["stalled", "converged"]))
    lightweight_prune: TriggerConfig = field(
        default_factory=lambda: TriggerConfig(100, ["stalled"]))
    cull_sg_axes: TriggerConfig = field(
        default_factory=lambda: TriggerConfig(500, ["converged"]))


@dataclass
class SkipGSConfig:
    enabled: bool = True
    warmup_steady_samples: int = 50
    beta: float = 0.95
    eps: float = 1e-8
    rho_lo: float = 0.5


@dataclass
class BootstrapConfig:
    min_images: int = 4


@dataclass
class ConvergenceConfig:
    loss_window: int = 200
    densify_window: int = 10
    converged_slope: float = -1e-4
    active_densify: float = 0.05
    active_slope: float = -1e-3


@dataclass
class TrainingConfig:
    accumulation_views: int = 4
    num_max_ceiling: int = 800_000
    prune_ratio1: float = 0.05
    prune_ratio2: float = 0.05
    sharpness_threshold: float = 1.0
    imp_metric: str = "outdoor"
    imp_score_camera_subsample: int = 0
    densify_min_obs: int = 10
    densify_candidate_fraction: float = 0.005
    fast_prune_dead_fraction: float = 0.02
    lightweight_prune_growth_threshold: float = 0.10
    sg_axis_cull_low_fraction: float = 0.20
    fast_final_prune_ratio: float = 0.10


@dataclass
class DenseInitConfig:
    enabled: bool = True
    min_sfm_points_for_alignment: int = 10
    min_sfm_depth_range_fraction: float = 0.10
    target_dense_points_per_image: int = 30_000
    novelty_distance_threshold: float = 0.01
    depth_disagreement_threshold: float = 0.10
    max_rejected_fraction: float = 0.5
    grace_iters: int = 20
    persist_model: bool = False
    dav2: DAv2Config = field(default_factory=DAv2Config)
    ransac: RansacConfig = field(default_factory=RansacConfig)


@dataclass
class HttpConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    checkpoint_timeout: float = 30.0


@dataclass
class ContinuousConfig:
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    triggers: TriggersConfig = field(default_factory=TriggersConfig)
    skipgs: SkipGSConfig = field(default_factory=SkipGSConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dense_init: DenseInitConfig = field(default_factory=DenseInitConfig)
    http: HttpConfig = field(default_factory=HttpConfig)


def _apply_dict(obj, d):
    for k, v in d.items():
        if hasattr(obj, k):
            setattr(obj, k, v)


def load_config(yaml_path: str) -> ContinuousConfig:
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    cfg = ContinuousConfig()

    if "convergence" in raw:
        _apply_dict(cfg.convergence, raw["convergence"])

    if "triggers" in raw:
        t = raw["triggers"]
        for name, sub in t.items():
            trig = getattr(cfg.triggers, name, None)
            if trig is not None and isinstance(sub, dict):
                _apply_dict(trig, sub)

    if "skipgs" in raw:
        _apply_dict(cfg.skipgs, raw["skipgs"])

    if "bootstrap" in raw:
        _apply_dict(cfg.bootstrap, raw["bootstrap"])

    if "training" in raw:
        _apply_dict(cfg.training, raw["training"])

    if "dense_init" in raw:
        di = raw["dense_init"]
        for k, v in di.items():
            if k == "dav2":
                _apply_dict(cfg.dense_init.dav2, v)
            elif k == "ransac":
                _apply_dict(cfg.dense_init.ransac, v)
            elif hasattr(cfg.dense_init, k):
                setattr(cfg.dense_init, k, v)

    if "http" in raw:
        _apply_dict(cfg.http, raw["http"])

    return cfg


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

def update_imp_score(cameras, gaussians, pipe, background, imp_metric="outdoor",
                     subsample_n: int = 0, weights=None):
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
# Camera selection
# ---------------------------------------------------------------------------

def select_cameras_weighted(all_cameras, image_weights, n_top=10, n_random=10):
    if image_weights is None or len(all_cameras) <= n_top + n_random:
        return list(all_cameras)
    sorted_idx = np.argsort(image_weights)[::-1]
    top_idx = set(sorted_idx[:n_top].tolist())
    rest_idx = [i for i in range(len(all_cameras)) if i not in top_idx]
    selected = [all_cameras[i] for i in top_idx]
    if rest_idx:
        sampled = np.random.choice(
            rest_idx, size=min(n_random, len(rest_idx)), replace=False)
        selected += [all_cameras[i] for i in sampled]
    return selected


# ---------------------------------------------------------------------------
# Gaussian expansion from new SfM points
# ---------------------------------------------------------------------------

def expand_gaussians_from_new_points(gaussians, prog_scene, new_point_mask, opt,
                                     distance_threshold=1.0, distance_buffer=1.5,
                                     birth_iter=0):
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
    all_new_indices = np.where(new_point_mask)[0]
    novel_global = np.zeros(len(pcd.points), dtype=bool)
    novel_global[all_new_indices[novel_mask_local]] = True
    n_novel = novel_global.sum()
    if n_novel == 0:
        return
    logger.info(f"[expand] Adding {n_novel} Gaussians from new SfM points")
    gaussians.expand_from_pcd(pcd, novel_global, prog_scene.cameras_extent, birth_iter=birth_iter)


# ---------------------------------------------------------------------------
# Dense initialization for new images
# ---------------------------------------------------------------------------

def dense_init_for_new_images(gaussians, prog_scene, new_cams, dense_cfg,
                               opt, current_iter, dav2_model=None, max_gaussians=0):
    import time
    t_start = time.monotonic()
    accumulated_xyz = []
    accumulated_rgb = []
    scene_scale = prog_scene.cameras_extent

    def _run_cameras(depth_model):
        for cam in new_cams:
            sfm_xyz_all, visible_mask = prog_scene.get_sfm_points_visible_to(cam)
            sfm_xyz_visible = sfm_xyz_all[visible_mask]

            if len(sfm_xyz_visible) < dense_cfg.min_sfm_points_for_alignment:
                continue

            depths_in_cam = transform_to_camera_frame(sfm_xyz_visible, cam)[:, 2]
            depth_range = float(depths_in_cam.max() - depths_in_cam.min())
            if depth_range < dense_cfg.min_sfm_depth_range_fraction * scene_scale:
                continue

            depth_map = depth_model.predict(cam.original_image.cpu())

            try:
                result = align_depth_to_sfm(
                    depth_map, cam, sfm_xyz_visible, dense_cfg.ransac,
                    scene_scale=scene_scale,
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

    if not accumulated_xyz:
        return

    all_xyz = np.concatenate(accumulated_xyz, axis=0)
    all_rgb = np.concatenate(accumulated_rgb, axis=0)
    del accumulated_xyz, accumulated_rgb

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
        return

    novel_xyz = all_xyz[novel_mask]
    novel_rgb = all_rgb[novel_mask]

    if max_gaussians > 0:
        capacity = max_gaussians - gaussians.get_xyz.shape[0]
        if capacity <= 0:
            logger.warning("[dense-init] already at capacity — skipping")
            return
        if n_novel > capacity:
            rng = np.random.default_rng(0)
            keep = rng.choice(n_novel, size=capacity, replace=False)
            keep.sort()
            novel_xyz = novel_xyz[keep]
            novel_rgb = novel_rgb[keep]
            n_novel = capacity

    novel_pcd = BasicPointCloud(
        points=novel_xyz, colors=novel_rgb,
        normals=np.zeros((n_novel, 3), dtype=np.float32),
    )

    n_before = gaussians.get_xyz.shape[0]
    gaussians.expand_from_pcd(novel_pcd, np.ones(n_novel, dtype=bool),
                               prog_scene.cameras_extent, birth_iter=current_iter)
    n_after = gaussians.get_xyz.shape[0]
    gaussians.mark_recently_added(
        slice(n_before, n_after), iteration=current_iter,
        grace_iters=dense_cfg.grace_iters,
    )
    elapsed = time.monotonic() - t_start
    logger.info(f"[dense-init] added {n_novel} dense Gaussians; wall={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Lightweight importance prune
# ---------------------------------------------------------------------------

def run_lightweight_prune(gaussians, prog_scene, opt, cfg, global_iter):
    pipe_dummy = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    imp_score = update_imp_score(
        prog_scene.train_cameras, gaussians, pipe_dummy, background,
        imp_metric=cfg.training.imp_metric,
        subsample_n=cfg.training.imp_score_camera_subsample,
        weights=prog_scene.image_weights,
    )
    grace_mask = gaussians.get_grace_protected_mask(global_iter)
    # Index-based bottom-K selection: with value-thresholding (<=), heavy
    # ties at the bottom of the importance distribution (e.g. unobserved
    # dense-init Gaussians sitting at zero) cause the mask to scoop up far
    # more than prune_ratio1 of the population. argsort selects exactly N
    # rows regardless of ties.
    threshold = int(cfg.training.prune_ratio1 * imp_score.shape[0])
    flat_imp = imp_score.squeeze()
    sorted_indices = torch.argsort(flat_imp)
    prune_mask = torch.zeros_like(flat_imp, dtype=torch.bool)
    prune_mask[sorted_indices[:threshold]] = True
    prune_mask = prune_mask & ~grace_mask
    before = gaussians._xyz.shape[0]
    gaussians.prune_points(prune_mask)
    after = gaussians._xyz.shape[0]
    logger.info(f"[prune] {before} → {after} Gaussians")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Atomic snapshot write
# ---------------------------------------------------------------------------

def write_snapshot_atomic(model_path: str, gaussians: SphericalGaussianModel) -> str:
    """Write current.ply atomically; return final path."""
    out_dir = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "current.ply.tmp"
    final = out_dir / "current.ply"
    gaussians.save_ply(str(tmp))
    os.replace(str(tmp), str(final))
    return str(final)


# ---------------------------------------------------------------------------
# Main continuous training loop
# ---------------------------------------------------------------------------

def continuous_training(dataset, opt, pipe, args, cfg: ContinuousConfig,
                        ctrl: ControlState):
    model_path = dataset.model_path
    os.makedirs(model_path, exist_ok=True)

    gaussians = SphericalGaussianModel(dataset.sg_degree)
    prog_scene = ProgressiveScene(
        source_path_dir=None,
        model_args=dataset,
        images_subdir=getattr(dataset, 'images', 'images'),
    )

    bg_color = [1, 1, 1] if getattr(dataset, 'white_background', False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    conv_cfg = cfg.convergence
    monitor = ConvergenceMonitor(
        loss_window=conv_cfg.loss_window,
        densify_window=conv_cfg.densify_window,
    )

    skipgs: Optional[SkipGSGate] = None
    if cfg.skipgs.enabled:
        skipgs = SkipGSGate(
            warmup_steady_samples=cfg.skipgs.warmup_steady_samples,
            beta=cfg.skipgs.beta,
            eps=cfg.skipgs.eps,
            rho_lo=cfg.skipgs.rho_lo,
        )

    tc = cfg.training
    soft_cap = int(tc.num_max_ceiling * 0.85)

    # Per-action iteration counters for anti-thrash floors
    iters_since_densify = 0
    iters_since_fast_prune = 0
    iters_since_lw_prune = 0
    iters_since_cull = 0

    n_at_last_prune = 0
    cycle_at_last_cull = -1  # allow first cull on cycle 0
    wrote_current_at_cycle = -1  # write current.ply once per convergence cycle
    mask_blur = None
    viewpoint_stack = None
    global_iter = 0
    ema_loss = 0.0
    bootstrapped = False

    _persistent_dav2 = None
    if cfg.dense_init.persist_model:
        _persistent_dav2 = DepthAnythingV2Wrapper(cfg.dense_init.dav2).__enter__()

    logger.info("[continuous] Trainer started; waiting for ≥4 images via /ingest")

    while True:
        # ---- 1. Handle pending checkpoint ----
        if ctrl.checkpoint_pending():
            path = write_snapshot_atomic(model_path, gaussians)
            ctrl.complete_checkpoint(path)
            logger.info(f"[checkpoint] wrote {path}")

        # ---- 2. Drain ingest queue ----
        pending = ctrl.drain_ingest_queue()
        for req in pending:
            ctrl.set_request_state(req.request_id, "integrating")
            logger.info(f"[ingest] {req.snapshot_dir} (id={req.request_id})")

            n_before_ingest = prog_scene.n_images
            new_cams, new_point_mask, new_cam_indices = \
                prog_scene.add_snapshot(req.snapshot_dir)

            # First snapshot: initialise Gaussians from sparse cloud
            if n_before_ingest == 0:
                gaussians.create_from_pcd(
                    prog_scene.current_basic_pcd, prog_scene.cameras_extent,
                    birth_iter=global_iter)
                gaussians.training_setup(opt)
                mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                n_at_last_prune = gaussians._xyz.shape[0]

            # Retrofit existing cohort LR schedules to the current scene scale,
            # then seed new Gaussians from sparse / dense points at this scale.
            if n_before_ingest > 0:
                gaussians.rescale_lr_scale_to(prog_scene.cameras_extent, opt)

            # Subsequent snapshots: seed from new sparse points
            if n_before_ingest > 0 and new_point_mask.sum() > 0:
                expand_gaussians_from_new_points(
                    gaussians, prog_scene, new_point_mask, opt,
                    birth_iter=global_iter)

            # Dense init for each newly ingested image
            if cfg.dense_init.enabled and new_cams:
                dense_init_for_new_images(
                    gaussians, prog_scene, new_cams,
                    cfg.dense_init, opt, current_iter=global_iter,
                    dav2_model=_persistent_dav2,
                    max_gaussians=tc.num_max_ceiling,
                )

            # Parse match matrix if available
            matrix_path = prog_scene.snapshot_dir / "sparse/0/imageMatchMatrix.txt"
            names_path = prog_scene.snapshot_dir / "sparse/0/imagesNames.txt"
            if matrix_path.exists() and names_path.exists():
                ordered_names = [cam.image_name for cam in prog_scene.train_cameras]
                prog_scene.image_match_matrix = parse_match_matrix(
                    str(matrix_path), str(names_path), ordered_names)
                prog_scene.image_weights = compute_image_weights(
                    prog_scene.image_match_matrix, new_cam_indices)
            else:
                M = len(prog_scene.train_cameras)
                weights = np.ones(M, dtype=np.float32) * 0.5
                for i in new_cam_indices:
                    weights[i] = 1.0
                prog_scene.image_weights = weights

            # Reset convergence monitor scaled by how much the scene changed.
            # A large ingest (many new images) resets more aggressively so the
            # monitor doesn't declare stale convergence on a structurally new scene.
            if prog_scene.n_images > 0:
                fraction_changed = len(new_cams) / max(prog_scene.n_images, 1)
                monitor.reset(fraction_changed=fraction_changed)

            if skipgs is not None:
                skipgs._enabled = False
                skipgs._steady_count = 0

            ctrl.set_request_state(req.request_id, "training")

        # ---- 3. Bootstrap gate ----
        n_images_now = prog_scene.n_images
        bootstrapped = n_images_now >= cfg.bootstrap.min_images

        ctrl.update_status(
            gaussian_count=gaussians._xyz.shape[0] if n_images_now > 0 else 0,
            monitor_state=monitor.state(
                conv_cfg.converged_slope, conv_cfg.active_densify, conv_cfg.active_slope,
            ) if n_images_now > 0 else "initializing",
            n_images=n_images_now,
            bootstrap_complete=bootstrapped,
        )

        if not bootstrapped:
            ctrl.wait_for_work(timeout=0.5)
            continue

        # ---- 4. Optimizer step ----
        K = max(1, tc.accumulation_views)
        gaussians.update_learning_rate(global_iter)

        per_view_packs = []
        for _sub in range(K):
            cameras = prog_scene.train_cameras
            if prog_scene.image_weights is not None:
                candidates = select_cameras_weighted(
                    cameras, prog_scene.image_weights, n_top=10, n_random=10)
                if not candidates:
                    candidates = cameras
                vcam = candidates[randint(0, len(candidates) - 1)]
                cam_idx = prog_scene._cam_name_to_idx.get(vcam.image_name, 0)
                cam_w = float(prog_scene.image_weights[cam_idx]) \
                    if cam_idx < len(prog_scene.image_weights) else 1.0
            else:
                if not viewpoint_stack:
                    viewpoint_stack = cameras.copy()
                vcam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
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
            per_view_packs.append((cam_w * base_loss, render_pkg, vcam.uid, vcam.image_name, cam_w))

        # SkipGS gating
        cur_state = monitor.state(
            conv_cfg.converged_slope, conv_cfg.active_densify, conv_cfg.active_slope)
        if skipgs is not None:
            skipgs.notify_monitor_state(cur_state)
            for loss_t, _, cam_id, _, _ in per_view_packs:
                skipgs.update_ema(cam_id, float(loss_t.detach()))
            devs = [skipgs.deviation(cam_id, float(loss_t.detach()))
                    for loss_t, _, cam_id, _, _ in per_view_packs]
            gate, _ = skipgs.decide(devs)
        else:
            gate = [True] * K

        contributing = [loss_t for (loss_t, _, _, _, _), g in zip(per_view_packs, gate) if g]

        if not contributing:
            if skipgs is not None:
                skipgs.record_backward(False)
            with torch.no_grad():
                for _, render_pkg, _, _, _ in per_view_packs:
                    vf = render_pkg["visibility_filter"]
                    rad = render_pkg["radii"]
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rad[vf])
                    am = render_pkg["area_max"]
                    ri = render_pkg["render"]
                    if mask_blur is None:
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    mbp = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                        mbp[:mask_blur.shape[0]] = mask_blur
                    mask_blur = torch.logical_or(mbp, am > (ri.shape[1] * ri.shape[2] / 5000))
        else:
            loss = sum(contributing)
            loss.backward()
            if skipgs is not None:
                skipgs.record_backward(True)

            with torch.no_grad():
                ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
                monitor.update_loss(ema_loss)

                for (_, render_pkg, _, _, _), g in zip(per_view_packs, gate):
                    vf = render_pkg["visibility_filter"]
                    rad = render_pkg["radii"]
                    am = render_pkg["area_max"]
                    ri = render_pkg["render"]
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], rad[vf])
                    if mask_blur is None:
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    mbp = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                    if mask_blur.shape[0] <= gaussians._xyz.shape[0]:
                        mbp[:mask_blur.shape[0]] = mask_blur
                    mask_blur = torch.logical_or(mbp, am > (ri.shape[1] * ri.shape[2] / 5000))
                    if g:
                        gaussians.add_densification_stats(render_pkg["viewspace_points"], vf)

                # ---- Densify ----
                if gaussians._xyz.shape[0] < tc.num_max_ceiling:
                    d_cfg = cfg.triggers.densify
                    if should_densify(
                        monitor, gaussians, opt, mask_blur,
                        iters_since_last=iters_since_densify,
                        candidate_fraction=tc.densify_candidate_fraction,
                        min_obs=tc.densify_min_obs,
                        min_iters_between=d_cfg.min_iters_between,
                        require_states=tuple(d_cfg.require_state),
                    ):
                        n_before = gaussians._xyz.shape[0]
                        gaussians.densify_and_prune_split(
                            opt.densify_grad_threshold, 0.005,
                            prog_scene.cameras_extent, None,
                            mask_blur[:gaussians.xyz_gradient_accum.shape[0]],
                        )
                        n_after = gaussians._xyz.shape[0]
                        monitor.update_densify(max(n_after - n_before, 0), n_after)
                        fraction_changed = abs(n_after - n_before) / max(n_after, 1)
                        monitor.reset(fraction_changed=fraction_changed)
                        mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
                        iters_since_densify = 0

                # ---- Fast prune ----
                fp_cfg = cfg.triggers.fast_prune
                if should_fast_prune(
                    gaussians, opt, tc.fast_prune_dead_fraction,
                    monitor=monitor,
                    iters_since_last=iters_since_fast_prune,
                    min_iters_between=fp_cfg.min_iters_between,
                    require_states=tuple(fp_cfg.require_state),
                ):
                    n_before_fp = gaussians._xyz.shape[0]
                    gaussians.opacity_size_prune(
                        min_opacity=0.005, max_screen_size=None,
                        extent=prog_scene.cameras_extent,
                    )
                    n_after_fp = gaussians._xyz.shape[0]
                    fraction_pruned = (n_before_fp - n_after_fp) / max(n_before_fp, 1)
                    monitor.reset(fraction_changed=fraction_pruned)
                    iters_since_fast_prune = 0

                # ---- Lightweight importance prune ----
                lw_cfg = cfg.triggers.lightweight_prune
                if should_lightweight_prune(
                    gaussians, monitor, n_at_last_prune, soft_cap,
                    iters_since_last=iters_since_lw_prune,
                    min_iters_between=lw_cfg.min_iters_between,
                    growth_threshold=tc.lightweight_prune_growth_threshold,
                    require_states=tuple(lw_cfg.require_state),
                ):
                    n_before_prune = gaussians._xyz.shape[0]
                    run_lightweight_prune(gaussians, prog_scene, opt, cfg, global_iter)
                    n_at_last_prune = gaussians._xyz.shape[0]
                    fraction_pruned = (n_before_prune - n_at_last_prune) / max(n_before_prune, 1)
                    monitor.reset(fraction_changed=fraction_pruned)
                    iters_since_lw_prune = 0

                # ---- SG axis cull ----
                # Fires at most once per convergence cycle. monitor.cycle bumps
                # on every reset (ingest, densify, prune); axis-cull is a heavy
                # appearance-basis change so we want a fresh look after it.
                if monitor.cycle > cycle_at_last_cull:
                    cull_cfg = cfg.triggers.cull_sg_axes
                    if should_cull_sg_axes(
                        gaussians, monitor, tc.sharpness_threshold,
                        fraction_low=tc.sg_axis_cull_low_fraction,
                        iters_since_last=iters_since_cull,
                        min_iters_between=cull_cfg.min_iters_between,
                        require_states=tuple(cull_cfg.require_state),
                    ):
                        gaussians.cull_low_sharpness_axes(
                            sharpness_threshold=tc.sharpness_threshold)
                        cycle_at_last_cull = monitor.cycle
                        monitor.reset(fraction_changed=0.3)
                        iters_since_cull = 0
                        torch.cuda.empty_cache()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            global_iter += 1
            iters_since_densify += 1
            iters_since_fast_prune += 1
            iters_since_lw_prune += 1
            iters_since_cull += 1

            if global_iter % 100 == 0:
                logger.info(
                    f"[iter {global_iter}] loss={ema_loss:.6f} "
                    f"N={gaussians._xyz.shape[0]} state={cur_state}"
                )

        # ---- 5. Converged + idle ----
        if (cur_state == "converged"
                and ctrl.queue_depth == 0
                and not ctrl.checkpoint_pending()):
            # The main iter loop exits to idle the moment "converged" fires,
            # so the cull check inside that loop never gets a converged state
            # to act on. Run it here instead — if it fires, monitor.reset()
            # bumps the cycle, training resumes, and we re-enter convergence
            # later with the culled model.
            if monitor.cycle > cycle_at_last_cull:
                cull_cfg = cfg.triggers.cull_sg_axes
                if should_cull_sg_axes(
                    gaussians, monitor, tc.sharpness_threshold,
                    fraction_low=tc.sg_axis_cull_low_fraction,
                    iters_since_last=iters_since_cull,
                    min_iters_between=cull_cfg.min_iters_between,
                    require_states=tuple(cull_cfg.require_state),
                ):
                    n_before_cull = gaussians._xyz.shape[0]
                    gaussians.cull_low_sharpness_axes(
                        sharpness_threshold=tc.sharpness_threshold)
                    logger.info(f"[cull] SG axes culled (N stays {n_before_cull})")
                    cycle_at_last_cull = monitor.cycle
                    monitor.reset(fraction_changed=0.3)
                    iters_since_cull = 0
                    torch.cuda.empty_cache()
                    continue

            if wrote_current_at_cycle != monitor.cycle:
                path = write_snapshot_atomic(model_path, gaussians)
                wrote_current_at_cycle = monitor.cycle
                logger.info(f"[converged] Wrote {path}; idling until next ingest or checkpoint")
            ctrl.update_status(
                gaussian_count=gaussians._xyz.shape[0],
                monitor_state="converged",
                n_images=prog_scene.n_images,
                bootstrap_complete=bootstrapped,
            )
            ctrl.wait_for_work(timeout=5.0)

    # (unreachable — loop runs until KeyboardInterrupt / SIGTERM)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = ArgumentParser(description="Continuous MEGS² training")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--config", type=str, default="configs/continuous.yaml")
    parser.add_argument("--http_host", type=str, default=None)
    parser.add_argument("--http_port", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    cfg = load_config(args.config)

    if args.http_host is not None:
        cfg.http.host = args.http_host
    if args.http_port is not None:
        cfg.http.port = args.http_port

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    if not dataset.model_path:
        dataset.model_path = os.path.join("./output/continuous", str(uuid.uuid4())[:8])

    os.makedirs(dataset.model_path, exist_ok=True)
    logger.info(f"Output → {dataset.model_path}")

    ctrl = ControlState()

    # Start HTTP server in background thread
    from scene.continuous_server import start_server
    start_server(cfg.http.host, cfg.http.port, ctrl, cfg.http.checkpoint_timeout)
    logger.info(f"[http] Listening on http://{cfg.http.host}:{cfg.http.port}")

    try:
        continuous_training(dataset, opt, pipe, args, cfg, ctrl)
    except KeyboardInterrupt:
        logger.info("Interrupted — exiting")


if __name__ == "__main__":
    main()
