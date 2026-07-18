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
import shutil
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
    DepthAnythingV2Wrapper, DepthAnything3Wrapper, validate_alignment,
    align_depth_to_sfm, depth_to_points,
    transform_to_camera_frame, AlignmentFailed,
    DAv2Config, DA3Config, RansacConfig,
)
from scene.convergence import ConvergenceMonitor
from scene.triggers import (
    should_densify, should_fast_prune,
    should_lightweight_prune, should_cull_sg_axes,
)
from scene.skipgs import SkipGSGate
from scene.control_state import ControlState
from scene.event_log import EventLog
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
    # Importance sampling: bias view selection toward currently-poorly-
    # rendered images via a per-image EMA of recent loss.
    image_error_weighting: bool = True
    image_error_ema_beta: float = 0.95
    # Crash resilience: write train_state.pt (model + optimizer + monitor +
    # ingest ledger) every N optimizer iters. 0 disables periodic saves
    # (a save still happens at each converged-idle transition).
    train_state_interval_iters: int = 500


@dataclass
class DenseInitConfig:
    enabled: bool = True
    # dav2 | da3 — default stays dav2 until tools/bench_dense_init.py shows
    # DA3 winning on quality at acceptable VRAM on the target machine.
    backend: str = "dav2"
    # When the DA3 fit fails its direct-depth sanity check for an image,
    # retry just that image with DAv2 (contexts run sequentially, so the
    # two models are never co-resident on GPU).
    da3_fallback_to_dav2: bool = True
    min_sfm_points_for_alignment: int = 10
    min_sfm_depth_range_fraction: float = 0.10
    target_dense_points_per_image: int = 30_000
    novelty_distance_threshold: float = 0.01
    depth_disagreement_threshold: float = 0.10
    max_rejected_fraction: float = 0.5
    grace_iters: int = 20
    persist_model: bool = False
    dav2: DAv2Config = field(default_factory=DAv2Config)
    da3: DA3Config = field(default_factory=DA3Config)
    ransac: RansacConfig = field(default_factory=RansacConfig)


@dataclass
class HttpConfig:
    host: str = "127.0.0.1"
    port: int = 8666
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
            elif k == "da3":
                _apply_dict(cfg.dense_init.da3, v)
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


def compute_effective_weights(image_weights, image_error_ema, cameras):
    """Combine match-matrix weights with per-image error EMA for selection bias.

    Returns image_weights scaled by each camera's normalised (mean=1) error
    EMA. Cameras with no recorded error yet get the mean error, so they're
    neither over- nor under-sampled until evidence accumulates.
    """
    if image_weights is None or not image_error_ema:
        return image_weights
    raw = np.array([
        image_error_ema.get(cam.image_name, -1.0) for cam in cameras
    ], dtype=np.float32)
    seen = raw >= 0.0
    if not seen.any():
        return image_weights
    mean_err = float(raw[seen].mean())
    if mean_err <= 0.0:
        return image_weights
    raw[~seen] = mean_err
    normalised = raw / mean_err
    return image_weights * normalised


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
    backend = getattr(dense_cfg, 'backend', 'dav2')
    # Cameras whose DA3 fit failed the direct-depth sanity check; retried
    # with DAv2 after the DA3 context closes (never co-resident on GPU).
    fallback_cams = []

    def _run_cameras(depth_model, cams, model_backend):
        pass_camera = getattr(depth_model, 'accepts_camera', False)
        for cam in cams:
            sfm_xyz_all, visible_mask = prog_scene.get_sfm_points_visible_to(cam)
            sfm_xyz_visible = sfm_xyz_all[visible_mask]

            if len(sfm_xyz_visible) < dense_cfg.min_sfm_points_for_alignment:
                continue

            depths_in_cam = transform_to_camera_frame(sfm_xyz_visible, cam)[:, 2]
            depth_range = float(depths_in_cam.max() - depths_in_cam.min())
            if depth_range < dense_cfg.min_sfm_depth_range_fraction * scene_scale:
                continue

            if pass_camera:
                depth_map = depth_model.predict(cam.original_image.cpu(), camera=cam)
            else:
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

            ok, reason = validate_alignment(model_backend, result)
            if not ok:
                logger.warning(f"[dense-init] {cam.image_name}: {reason}")
                fallback_cams.append(cam)
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

    if backend == "da3":
        if dav2_model is not None:
            logger.warning(
                "[dense-init] persist_model is only supported for the dav2 "
                "backend; ignoring persistent handle for da3")
        with DepthAnything3Wrapper(dense_cfg.da3) as depth_model:
            _run_cameras(depth_model, new_cams, "da3")
        if fallback_cams and getattr(dense_cfg, 'da3_fallback_to_dav2', True):
            logger.warning(
                f"[dense-init] retrying {len(fallback_cams)} image(s) with "
                "DAv2 after DA3 sanity failure")
            retry = list(fallback_cams)
            fallback_cams.clear()
            with DepthAnythingV2Wrapper(dense_cfg.dav2) as depth_model:
                _run_cameras(depth_model, retry, "dav2")
    elif dav2_model is not None:
        _run_cameras(dav2_model, new_cams, "dav2")
    else:
        with DepthAnythingV2Wrapper(dense_cfg.dav2) as depth_model:
            _run_cameras(depth_model, new_cams, "dav2")

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

def write_snapshot_atomic(model_path: str, gaussians: SphericalGaussianModel,
                          export_dir: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
    """Write current.ply atomically; return final path.

    When export_dir and session_id are set, also mirror the snapshot to
    <export_dir>/<session_id>/latest.ply — the path the home-server's
    viewer reads (app/api/trainer.py latest_splat_ply)."""
    out_dir = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "current.ply.tmp"
    final = out_dir / "current.ply"
    gaussians.save_ply(str(tmp))
    os.replace(str(tmp), str(final))

    if export_dir and session_id:
        try:
            exp_dir = Path(export_dir) / session_id
            exp_dir.mkdir(parents=True, exist_ok=True)
            exp_tmp = exp_dir / "latest.ply.tmp"
            shutil.copyfile(final, exp_tmp)
            os.replace(str(exp_tmp), str(exp_dir / "latest.ply"))
        except OSError as e:
            logger.warning(f"[snapshot] export mirror failed: {e}")

    return str(final)


# ---------------------------------------------------------------------------
# Train-state checkpoint (crash resilience / --resume)
# ---------------------------------------------------------------------------

TRAIN_STATE_NAME = "train_state.pt"


def save_train_state(model_path: str, gaussians, monitor, skipgs, ctrl,
                     trainer_state: dict) -> str:
    """Atomically persist everything needed to continue a session after a
    process restart: model tensors + Adam state (gaussians.capture()), grace
    records, convergence-monitor state, SkipGS gate state, the ingest
    idempotency ledger, and the trainer-loop counters."""
    ckpt = {
        "version": 1,
        "model": gaussians.capture(),
        "grace_records": [tuple(r) for r in getattr(gaussians, "_grace_records", [])],
        "monitor": monitor.get_state(),
        "skipgs": skipgs.get_state() if skipgs is not None else None,
        "ledger": ctrl.ledger_state(),
        "trainer": trainer_state,
    }
    out_dir = Path(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / (TRAIN_STATE_NAME + ".tmp")
    final = out_dir / TRAIN_STATE_NAME
    torch.save(ckpt, str(tmp))
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

    # Pre-tuple require_state lists once; called every iter from triggers.
    require_states_densify = tuple(cfg.triggers.densify.require_state)
    require_states_fast_prune = tuple(cfg.triggers.fast_prune.require_state)
    require_states_lw_prune = tuple(cfg.triggers.lightweight_prune.require_state)
    require_states_cull = tuple(cfg.triggers.cull_sg_axes.require_state)

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
    warned_no_match_matrix = False
    export_dir = getattr(args, 'export_dir', None) or None

    last_snapshot_dir = None
    evlog = EventLog(os.path.join(model_path, "events.jsonl"))

    def trainer_state_dict():
        return {
            "global_iter": global_iter,
            "ema_loss": ema_loss,
            "iters_since_densify": iters_since_densify,
            "iters_since_fast_prune": iters_since_fast_prune,
            "iters_since_lw_prune": iters_since_lw_prune,
            "iters_since_cull": iters_since_cull,
            "n_at_last_prune": n_at_last_prune,
            "cycle_at_last_cull": cycle_at_last_cull,
            "wrote_current_at_cycle": wrote_current_at_cycle,
            "last_snapshot_dir": last_snapshot_dir,
            "image_error_ema": dict(prog_scene.image_error_ema),
            "image_weights": (
                prog_scene.image_weights.tolist()
                if prog_scene.image_weights is not None else None),
        }

    # ---- Resume from a previous run's train state ----
    train_state_path = Path(model_path) / TRAIN_STATE_NAME
    if getattr(args, "resume", False):
        if train_state_path.exists():
            ckpt = torch.load(str(train_state_path), weights_only=False)
            ts = ckpt["trainer"]
            last_snapshot_dir = ts.get("last_snapshot_dir")
            if last_snapshot_dir and Path(last_snapshot_dir).exists():
                # Cumulative snapshot: re-adding the latest one rebuilds the
                # full camera set and sparse cloud.
                prog_scene.add_snapshot(last_snapshot_dir)
            else:
                logger.warning(
                    f"[resume] last snapshot dir missing ({last_snapshot_dir}); "
                    "cameras will rebuild on the next ingest")
            gaussians.restore(ckpt["model"], opt)
            gaussians._grace_records = [
                tuple(r) for r in ckpt.get("grace_records", [])]
            monitor.set_state(ckpt["monitor"])
            if skipgs is not None and ckpt.get("skipgs") is not None:
                skipgs.set_state(ckpt["skipgs"])
            ctrl.restore_ledger(ckpt.get("ledger", {}))
            global_iter = ts["global_iter"]
            ema_loss = ts["ema_loss"]
            iters_since_densify = ts["iters_since_densify"]
            iters_since_fast_prune = ts["iters_since_fast_prune"]
            iters_since_lw_prune = ts["iters_since_lw_prune"]
            iters_since_cull = ts["iters_since_cull"]
            n_at_last_prune = ts["n_at_last_prune"]
            cycle_at_last_cull = ts["cycle_at_last_cull"]
            wrote_current_at_cycle = ts["wrote_current_at_cycle"]
            prog_scene.image_error_ema = dict(ts.get("image_error_ema", {}))
            if ts.get("image_weights") is not None:
                prog_scene.image_weights = np.array(
                    ts["image_weights"], dtype=np.float32)
            evlog.emit("resume", iter=global_iter,
                       n_images=prog_scene.n_images,
                       splat_count=gaussians._xyz.shape[0])
            logger.info(
                f"[resume] restored iter={global_iter} "
                f"N={gaussians._xyz.shape[0]} images={prog_scene.n_images} "
                f"session={ctrl.session_id}")
        else:
            logger.warning(
                f"[resume] requested but {train_state_path} not found; "
                "starting fresh")

    _persistent_dav2 = None
    if cfg.dense_init.persist_model:
        if cfg.dense_init.backend == "da3":
            logger.warning(
                "[dense-init] persist_model not supported with backend=da3; "
                "DA3 loads per ingest")
        else:
            _persistent_dav2 = DepthAnythingV2Wrapper(cfg.dense_init.dav2).__enter__()

    logger.info("[continuous] Trainer started; waiting for ≥4 images via /ingest")

    while True:
        # ---- 1. Handle pending checkpoint ----
        if ctrl.checkpoint_pending():
            path = write_snapshot_atomic(model_path, gaussians,
                                         export_dir, ctrl.session_id)
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
                mask_blur = None
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
                if not warned_no_match_matrix:
                    logger.warning(
                        "[ingest] no imageMatchMatrix.txt/imagesNames.txt in "
                        f"{prog_scene.snapshot_dir}/sparse/0 — using uniform "
                        "weights with new-camera bias (L8 fallback); this "
                        "warning is logged once")
                    warned_no_match_matrix = True
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

            last_snapshot_dir = req.snapshot_dir
            evlog.emit("ingest", iter=global_iter,
                       snapshot_dir=req.snapshot_dir,
                       session_id=req.session_id,
                       image_name=req.image_name,
                       n_images=prog_scene.n_images,
                       n_new_cams=len(new_cams),
                       splat_count=gaussians._xyz.shape[0])
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
            iteration=global_iter,
        )

        if not bootstrapped:
            ctrl.wait_for_work(timeout=0.5)
            continue

        # ---- 4. Converged-idle gate (BEFORE iter work) ----
        # Compute state from the existing loss_history without doing any new
        # optimizer work. If we're converged, run pending cull/snapshot/idle
        # logic and skip the iteration entirely — otherwise the trainer would
        # burn GPU on optimizer steps that don't change anything meaningful.
        cur_state = monitor.state(
            conv_cfg.converged_slope, conv_cfg.active_densify, conv_cfg.active_slope)
        if (cur_state == "converged"
                and ctrl.queue_depth == 0
                and not ctrl.checkpoint_pending()):
            if monitor.cycle > cycle_at_last_cull:
                if should_cull_sg_axes(
                    cur_state, gaussians, tc.sharpness_threshold,
                    fraction_low=tc.sg_axis_cull_low_fraction,
                    iters_since_last=iters_since_cull,
                    min_iters_between=cfg.triggers.cull_sg_axes.min_iters_between,
                    require_states=require_states_cull,
                ):
                    n_unchanged = gaussians._xyz.shape[0]
                    gaussians.cull_low_sharpness_axes(
                        sharpness_threshold=tc.sharpness_threshold)
                    monitor.reset(fraction_changed=0.3)
                    # Capture cycle AFTER reset; reset bumps it, so this
                    # disables re-firing until an external event (ingest,
                    # densify, prune) bumps the cycle again.
                    cycle_at_last_cull = monitor.cycle
                    iters_since_cull = 0
                    torch.cuda.empty_cache()
                    evlog.emit("cull_sg_axes", iter=global_iter,
                               state=cur_state, splat_count=n_unchanged)
                    logger.info(
                        f"[cull] SG axes pruned; Gaussian count unchanged at {n_unchanged}"
                    )
                    continue

            if wrote_current_at_cycle != monitor.cycle:
                path = write_snapshot_atomic(model_path, gaussians,
                                             export_dir, ctrl.session_id)
                wrote_current_at_cycle = monitor.cycle
                evlog.emit("converged", iter=global_iter,
                           splat_count=gaussians._xyz.shape[0],
                           n_images=prog_scene.n_images, snapshot=path)
                save_train_state(model_path, gaussians, monitor, skipgs,
                                 ctrl, trainer_state_dict())
                logger.info(f"[converged] Wrote {path}; idling until next ingest or checkpoint")
            ctrl.update_status(
                gaussian_count=gaussians._xyz.shape[0],
                monitor_state="converged",
                n_images=prog_scene.n_images,
                bootstrap_complete=bootstrapped,
                iteration=global_iter,
            )
            ctrl.wait_for_work(timeout=5.0)
            continue

        # ---- 5. Optimizer step ----
        K = max(1, tc.accumulation_views)
        gaussians.update_learning_rate(global_iter)

        cameras = prog_scene.train_cameras
        # Computed once per iter; views within the same iter share the same
        # selection bias (cheap; avoids redundant per-view recomputation).
        if tc.image_error_weighting and prog_scene.image_weights is not None:
            selection_weights = compute_effective_weights(
                prog_scene.image_weights, prog_scene.image_error_ema, cameras)
        else:
            selection_weights = prog_scene.image_weights

        # Build the candidate pool once per outer iter. select_cameras_weighted
        # does np.argsort + set construction; doing it K times is wasted work
        # when the underlying weights don't change between micro-batches.
        if selection_weights is not None:
            candidates = select_cameras_weighted(
                cameras, selection_weights, n_top=10, n_random=10)
            if not candidates:
                candidates = cameras
        else:
            candidates = None

        per_view_packs = []
        with gaussians.stable_views():
            for _sub in range(K):
                if candidates is not None:
                    vcam = candidates[randint(0, len(candidates) - 1)]
                    cam_idx = prog_scene._cam_name_to_idx.get(vcam.image_name, 0)
                    # Loss weighting (cam_w) keeps using the raw match-matrix
                    # weight — error_ema only biases selection, not gradient
                    # contribution, to avoid double-counting.
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
                if tc.image_error_weighting:
                    ema = prog_scene.image_error_ema
                    prev = ema.get(vcam.image_name, float(base_loss.detach()))
                    beta = tc.image_error_ema_beta
                    ema[vcam.image_name] = beta * prev + (1.0 - beta) * float(base_loss.detach())
                per_view_packs.append((cam_w * base_loss, render_pkg, vcam.uid, vcam.image_name, cam_w))

        # SkipGS gating — uses cur_state snapshotted at top of loop
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

        # mask_blur survives across iters; resize/clear if N changed since last
        # iter (densify/prune in the prior iter would have done that). One bool
        # buffer reused via in-place |= avoids K size-N alloc+copy pairs per iter.
        n_total = gaussians._xyz.shape[0]
        if mask_blur is None or mask_blur.shape[0] != n_total:
            mask_blur = torch.zeros(n_total, dtype=torch.bool, device="cuda")

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
                    mask_blur |= am > (ri.shape[1] * ri.shape[2] / 5000)
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
                    mask_blur |= am > (ri.shape[1] * ri.shape[2] / 5000)
                    if g:
                        gaussians.add_densification_stats(render_pkg["viewspace_points"], vf)

                # ---- Densify ----
                if gaussians._xyz.shape[0] < tc.num_max_ceiling:
                    if should_densify(
                        cur_state, gaussians, opt, mask_blur,
                        iters_since_last=iters_since_densify,
                        candidate_fraction=tc.densify_candidate_fraction,
                        min_obs=tc.densify_min_obs,
                        min_iters_between=cfg.triggers.densify.min_iters_between,
                        require_states=require_states_densify,
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
                        evlog.emit("densify", iter=global_iter, state=cur_state,
                                   n_before=n_before, n_after=n_after)
                        # mask_blur is reallocated at the top of the next iter
                        # when n_total changes; no need to reset here.
                        iters_since_densify = 0

                # ---- Fast prune ----
                if should_fast_prune(
                    cur_state, gaussians, opt, tc.fast_prune_dead_fraction,
                    iters_since_last=iters_since_fast_prune,
                    min_iters_between=cfg.triggers.fast_prune.min_iters_between,
                    require_states=require_states_fast_prune,
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
                    evlog.emit("fast_prune", iter=global_iter, state=cur_state,
                               n_before=n_before_fp, n_after=n_after_fp)

                # ---- Lightweight importance prune ----
                if should_lightweight_prune(
                    cur_state, gaussians, n_at_last_prune, soft_cap,
                    iters_since_last=iters_since_lw_prune,
                    min_iters_between=cfg.triggers.lightweight_prune.min_iters_between,
                    growth_threshold=tc.lightweight_prune_growth_threshold,
                    require_states=require_states_lw_prune,
                ):
                    n_before_prune = gaussians._xyz.shape[0]
                    run_lightweight_prune(gaussians, prog_scene, opt, cfg, global_iter)
                    n_at_last_prune = gaussians._xyz.shape[0]
                    fraction_pruned = (n_before_prune - n_at_last_prune) / max(n_before_prune, 1)
                    monitor.reset(fraction_changed=fraction_pruned)
                    iters_since_lw_prune = 0
                    evlog.emit("lightweight_prune", iter=global_iter,
                               state=cur_state, n_before=n_before_prune,
                               n_after=n_at_last_prune)

                # SG axis cull is only handled in the converged-idle gate at
                # the top of the loop — its require_state is ["converged"],
                # and we never enter the iter body when converged, so an
                # in-loop check here would be dead code.

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            global_iter += 1
            iters_since_densify += 1
            iters_since_fast_prune += 1
            iters_since_lw_prune += 1
            iters_since_cull += 1

            if (tc.train_state_interval_iters > 0
                    and global_iter % tc.train_state_interval_iters == 0):
                path = save_train_state(model_path, gaussians, monitor,
                                        skipgs, ctrl, trainer_state_dict())
                evlog.emit("train_state", iter=global_iter, path=path)

            if global_iter % 100 == 0:
                logger.info(
                    f"[iter {global_iter}] loss={ema_loss:.6f} "
                    f"N={gaussians._xyz.shape[0]} state={cur_state}"
                )

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
    parser.add_argument(
        "--export_dir", type=str,
        default=os.environ.get("TRAINER_EXPORT_DIR", ""),
        help="Mirror snapshots to <export_dir>/<session_id>/latest.ply for the "
             "home-server viewer (defaults to $TRAINER_EXPORT_DIR)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Restore model/optimizer/monitor/ledger from "
             "<model_path>/train_state.pt and continue the session")
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
