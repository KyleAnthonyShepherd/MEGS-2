"""Dense initialization via Depth Anything v2.

Single-image path:
  DepthAnythingV2Wrapper  — context manager: loads DAv2 on __enter__, frees on __exit__
  align_depth_to_sfm      — RANSAC alignment of DAv2 depth to SfM ground-truth depths
  depth_to_points         — back-project aligned depth map to world-space point cloud

Multi-view window path (Plan 6a; see scene/covisibility.py for which images go
into each window):
  DepthAnything3Wrapper.predict_batch — one DA3 call over a window, anchor first
  normalize_window_extrinsics — §8 pose pre-normalisation, undone on the depth
  cross_view_consistency      — §6 per-pixel agreement count across the window
  window_scale_residual /
    resolve_scale_residual    — §9 one-scalar residual check against the SfM points
  depth_to_points_masked      — back-project the pixels the window kept
  fit_depth_to_reference /
    ReferenceFit              — incremental scale+offset alignment of a new
                                window onto the dense cloud already built from
                                the earlier images

Landmine notes:
  DL1: Camera FoVx/FoVy are in radians → fx = W/(2*tan(FoVx/2))
  DL2: DAv2 outputs *relative disparity* (not depth).  align_depth_to_sfm fits
       a linear model aligned = a*raw + b so the convention doesn't matter for
       fitting, but back_project MUST use the aligned (metric-ish) depth, not raw.
  DL3: COLMAP world-to-camera: p_cam = R_w2c @ p_world + t.
       world_view_transform is stored as Rt.T (transpose), so W2C = world_view_transform.T
"""

import gc
import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses (mirrors configs/progressive.yaml)
# ---------------------------------------------------------------------------

@dataclass
class DAv2Config:
    model_size: str = "small"     # small | base | large
    input_resolution: int = 518   # must be multiple of 14
    device: str = "cuda"
    fp16: bool = True


@dataclass
class DA3Config:
    """Depth Anything 3 backend config.

    model_name is a HuggingFace repo id. On the 6 GB target start with the
    any-view DA3-SMALL; DA3MONO-LARGE / DA3METRIC-LARGE exist only in LARGE
    and need headroom checks before use. All DA3 models predict *depth*
    directly (not disparity like DAv2), so the RANSAC-fitted scale `a`
    must come out positive — a negative fit means something is wrong
    (see validate_alignment).
    """
    model_name: str = "depth-anything/DA3-SMALL"
    device: str = "cuda"
    process_res: int = 504
    # Pass the session's COLMAP intrinsics as conditioning (FOV prior).
    conditioning: bool = True
    # Additionally pass COLMAP extrinsics. DA3 aligns its predicted pose
    # trajectory to the input poses with Umeyama Sim(3), which needs >= 2
    # views — a single-image call is rank-degenerate and raises
    # GeometryException. predict() runs one image at a time and only consumes
    # prediction.depth (RANSAC re-fits scale to SfM), so extrinsics add nothing
    # in that path; keep this OFF until a multi-view batch path exists.
    condition_extrinsics: bool = False
    use_ray_pose: bool = False
    # Multi-view window landmines (Plan 6a §8).
    # DA3 defaults to "saddle_balanced", which *reorders views*. We keep the
    # anchor at index 0 and read depths[0] back, so the reference view must be
    # the first one.
    ref_view_strategy: str = "first"
    # _normalize_extrinsics clamps the median camera distance at 0.1
    # (api.py:333). COLMAP scale is arbitrary, so a session whose inter-camera
    # spacing lands below that clamp would be silently mis-normalised. Divide
    # the translations by the window's own median camera-centre distance and
    # undo it on the returned depth. align_to_input_ext_scale should make this
    # a no-op — that is a prediction, not an observation (Plan 6a §12).
    prenormalize_extrinsics: bool = True


@dataclass
class RansacConfig:
    iterations: int = 200
    inlier_threshold: float = 0.05   # fraction of scene_scale
    min_inliers: int = 8


# ---------------------------------------------------------------------------
# Exceptions & result types
# ---------------------------------------------------------------------------

class AlignmentFailed(Exception):
    pass


@dataclass
class AlignmentResult:
    a: float
    b: float
    inlier_indices: np.ndarray
    n_inliers: int


# ---------------------------------------------------------------------------
# DAv2 context manager
# ---------------------------------------------------------------------------

class DepthAnythingV2Wrapper:
    """Loads DAv2 on __enter__, frees ALL GPU memory on __exit__.

    Usage:
        with DepthAnythingV2Wrapper(cfg) as depth_model:
            depth_map = depth_model.predict(image_chw)
        # model is gone here; GPU memory returned
    """

    _REPOS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, cfg: DAv2Config):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def __enter__(self):
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        repo = self._REPOS[self.cfg.model_size]
        self.processor = AutoImageProcessor.from_pretrained(repo)
        dtype = torch.float16 if self.cfg.fp16 else torch.float32
        self.model = AutoModelForDepthEstimation.from_pretrained(
            repo, torch_dtype=dtype
        ).to(self.cfg.device).eval()
        return self

    def __exit__(self, *args):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def predict(self, image_chw) -> "torch.Tensor":
        """Predict depth-like map.

        Args:
            image_chw: (3, H, W) float tensor in [0, 1], on any device.

        Returns:
            (H, W) float32 tensor on CPU — DAv2 raw output (relative disparity).
        """
        import torch as _torch
        from PIL import Image as PILImage
        img_np = (image_chw.cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = PILImage.fromarray(img_np)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        if self.cfg.fp16:
            inputs = {k: v.half() if v.dtype == _torch.float32 else v
                      for k, v in inputs.items()}

        with _torch.no_grad():
            outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # (1, H', W')

        # Sanitize before interpolation: inf/nan at any pixel spreads to neighbours
        # via bilinear interp.
        depth = _torch.nan_to_num(depth.float(), nan=0.0, posinf=0.0, neginf=0.0)
        H_orig, W_orig = image_chw.shape[1], image_chw.shape[2]
        depth = _torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(H_orig, W_orig),
            mode="bilinear",
            align_corners=False,
        ).squeeze().cpu()

        return depth


# ---------------------------------------------------------------------------
# DA3 context manager
# ---------------------------------------------------------------------------

class DepthAnything3Wrapper:
    """Loads Depth Anything 3 on __enter__, frees ALL GPU memory on __exit__.

    Same context-manager contract as DepthAnythingV2Wrapper; additionally
    accepts_camera=True — predict() takes an optional MEGS-2 Camera whose
    COLMAP pose/intrinsics are passed to DA3 as conditioning.

    Deferred (TODO sketches, do not build yet — see Plan 2 Milestone 4.5):
      - infer_gs=True feed-forward 3DGS head as an instant-preview path
        (render something seconds after the first photos, before MEGS-2
        refinement takes over);
      - DA3 pose estimation as an SfM-bootstrap-failure fallback (run
        inference() WITHOUT extrinsics over the accumulated images and use
        prediction.extrinsics as provisional poses);
      - DA3-Streaming variant for continuous capture.
    """

    accepts_camera = True

    def __init__(self, cfg: DA3Config):
        self.cfg = cfg
        self.model = None
        self._vram_baseline = 0

    def __enter__(self):
        from depth_anything_3.api import DepthAnything3
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            self._vram_baseline = torch.cuda.memory_allocated()
        self.model = DepthAnything3.from_pretrained(
            self.cfg.model_name).to(self.cfg.device).eval()
        return self

    def __exit__(self, *args):
        del self.model
        self.model = None
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            residue_mb = (torch.cuda.memory_allocated()
                          - self._vram_baseline) / (1024 ** 2)
            if residue_mb > 100:
                logger.warning(
                    f"[dense-init] DA3 leaked {residue_mb:.0f} MB VRAM after "
                    "__exit__ — check for lingering references")

    def predict(self, image_chw, camera=None) -> "torch.Tensor":
        """Predict a depth map for one image.

        Args:
            image_chw: (3, H, W) float tensor in [0, 1], on any device.
            camera: optional MEGS-2 Camera; when given (and
                cfg.conditioning), its COLMAP pose + intrinsics are passed
                to DA3 as conditioning.

        Returns:
            (H, W) float32 tensor on CPU — DA3 raw output. This is direct
            depth (larger = farther), unlike DAv2's disparity; the RANSAC
            alignment absorbs scale either way, but the fitted `a` must be
            positive (validate_alignment).
        """
        import torch as _torch
        img_np = (image_chw.cpu().float().permute(1, 2, 0).numpy()
                  * 255).astype(np.uint8)

        kwargs = {}
        if camera is not None and self.cfg.conditioning:
            ext, ixt = camera_to_da3_conditioning(camera)
            kwargs["intrinsics"] = ixt
            # Extrinsics only when explicitly enabled (multi-view path); a
            # single-image call makes DA3's Umeyama pose alignment degenerate.
            if self.cfg.condition_extrinsics:
                kwargs["extrinsics"] = ext
        if self.cfg.use_ray_pose:
            kwargs["use_ray_pose"] = True

        with _torch.no_grad():
            prediction = self.model.inference(
                [img_np], process_res=self.cfg.process_res, **kwargs)

        depth = _torch.as_tensor(
            np.asarray(prediction.depth[0]), dtype=_torch.float32)
        depth = _torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

        H_orig, W_orig = image_chw.shape[1], image_chw.shape[2]
        if depth.shape != (H_orig, W_orig):
            depth = _torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig), mode="bilinear", align_corners=False,
            ).squeeze()
        return depth.cpu()


    def predict_batch(self, images_chw, cameras):
        """One DA3 multi-view pass over a Plan 6a window. Anchor is index 0.

        Every depth map in the returned list shares one scale fit: they come
        from a single inference() call, so the per-pixel consistency check in
        cross_view_consistency() is comparing like with like. An image's depth
        must never be stitched from more than one call (Plan 6a §1.3).

        Args:
            images_chw: list of (3, H, W) float tensors in [0, 1]; index 0 is
                the anchor, whose depth map is the one we keep.
            cameras: matching list of MEGS-2 Cameras, same order.

        Returns:
            (depths, confs) — lists of (H_i, W_i) float32 CPU tensors, each
            resized back to its own input resolution and in COLMAP depth units.
            confs is a list of None when the model exposes no confidence head.
        """
        import torch as _torch

        if len(images_chw) != len(cameras):
            raise ValueError(
                f"predict_batch: {len(images_chw)} images vs "
                f"{len(cameras)} cameras")
        if len(images_chw) < 2:
            raise ValueError(
                "predict_batch needs >= 2 views; DA3's Umeyama pose alignment "
                "is rank-degenerate on a single view")

        imgs = [(im.cpu().float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for im in images_chw]
        ext = np.concatenate(
            [_get_w2c(c).astype(np.float64)[None] for c in cameras], axis=0)
        ixt = np.stack([_intrinsic_matrix(c) for c in cameras]).astype(np.float32)

        # §8: pre-normalise defensively against api.py:333's 0.1 clamp.
        scale = 1.0
        if self.cfg.prenormalize_extrinsics:
            ext, scale = normalize_window_extrinsics(ext)

        kwargs = {
            "intrinsics": ixt,
            "extrinsics": ext.astype(np.float32),
            "align_to_input_ext_scale": True,
            "ref_view_strategy": self.cfg.ref_view_strategy,
            "process_res": self.cfg.process_res,
        }
        if self.cfg.use_ray_pose:
            kwargs["use_ray_pose"] = True

        with _torch.no_grad():
            prediction = self.model.inference(imgs, **kwargs)

        raw_depth = np.asarray(prediction.depth)
        if len(raw_depth) != len(imgs):
            raise RuntimeError(
                f"DA3 returned {len(raw_depth)} depth maps for {len(imgs)} "
                "views — the window's index mapping is not identity")
        self._warn_if_views_reordered(prediction)

        raw_conf = None
        for attr in ("conf", "confidence", "conf_map"):
            if getattr(prediction, attr, None) is not None:
                raw_conf = np.asarray(getattr(prediction, attr))
                break

        depths, confs = [], []
        for k, image_chw in enumerate(images_chw):
            H, W = int(image_chw.shape[1]), int(image_chw.shape[2])
            d = _torch.as_tensor(raw_depth[k], dtype=_torch.float32)
            d = _torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
            depths.append(_resize_map(d, H, W) * float(scale))
            if raw_conf is None:
                confs.append(None)
            else:
                c = _torch.as_tensor(raw_conf[k], dtype=_torch.float32)
                c = _torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
                confs.append(_resize_map(c, H, W))
        return depths, confs

    def _warn_if_views_reordered(self, prediction):
        """Plan 6a §8 asks us to verify the identity mapping anyway."""
        for attr in ("view_order", "view_indices", "input_order"):
            order = getattr(prediction, attr, None)
            if order is None:
                continue
            order = [int(v) for v in np.asarray(order).reshape(-1)]
            if order != list(range(len(order))):
                logger.warning(
                    "[dense-init] DA3 reordered views (%s=%s) despite "
                    "ref_view_strategy=%r — depths[0] is NOT the anchor",
                    attr, order, self.cfg.ref_view_strategy)
            return
        ref = getattr(prediction, "ref_view_index", None)
        if ref is not None and int(ref) != 0:
            logger.warning(
                "[dense-init] DA3 chose reference view %d, not the anchor; "
                "ref_view_strategy=%r did not take", int(ref),
                self.cfg.ref_view_strategy)


def camera_to_da3_conditioning(camera):
    """Build DA3 conditioning arrays from a MEGS-2 Camera.

    Returns (extrinsics (1, 4, 4), intrinsics (1, 3, 3)) float32 numpy —
    OpenCV/COLMAP world-to-camera convention, which is what both MEGS-2
    (DL3) and DA3 use.
    """
    ext = _get_w2c(camera).astype(np.float32)[None]
    ixt = _intrinsic_matrix(camera).astype(np.float32)[None]
    return ext, ixt


def validate_alignment(backend: str, result: "AlignmentResult"):
    """Backend-specific sanity on the RANSAC fit. Returns (ok, reason).

    DAv2 predicts disparity, so a < 0 is the normal, expected fit.
    DA3 predicts depth directly, so a must be positive — a negative fit
    means the model output or the projection is wrong for this image and
    its dense points must not be trusted.
    """
    if backend == "da3" and result.a <= 0:
        return False, (f"DA3 fitted a={result.a:.4f} <= 0 but DA3 predicts "
                       "direct depth — rejecting this image's dense init")
    return True, ""


# ---------------------------------------------------------------------------
# Camera projection helpers
# ---------------------------------------------------------------------------

def _get_camera_intrinsics(camera) -> Tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) in pixels from a MEGS-2 Camera."""
    W = camera.image_width
    H = camera.image_height
    fx = W / (2.0 * math.tan(camera.FoVx / 2.0))
    fy = H / (2.0 * math.tan(camera.FoVy / 2.0))
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


def _get_w2c(camera) -> np.ndarray:
    """Return (4,4) world-to-camera matrix as float64 numpy array."""
    return camera.world_view_transform.T.cpu().double().numpy()


def project_world_to_image(
    xyz_world: np.ndarray,  # (N, 3)
    camera,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world-space points into camera image coords.

    Returns:
        u: (N,) float pixel x
        v: (N,) float pixel y
        z: (N,) depth in camera frame (positive = in front)
    """
    W2C = _get_w2c(camera)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    p_cam = (R @ xyz_world.T).T + t   # (N, 3)
    z = p_cam[:, 2]

    fx, fy, cx, cy = _get_camera_intrinsics(camera)
    safe_z = np.where(z > 0, z, 1.0)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    return u, v, z


def transform_to_camera_frame(xyz_world: np.ndarray, camera) -> np.ndarray:
    """Transform world-space points to camera frame. Returns (N, 3) array."""
    W2C = _get_w2c(camera)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    return (R @ xyz_world.T).T + t


# ---------------------------------------------------------------------------
# RANSAC depth-to-SfM alignment
# ---------------------------------------------------------------------------

def align_depth_to_sfm(
    depth_map,  # torch.Tensor          # (H, W) DAv2 raw output, CPU float32
    camera,                           # MEGS-2 Camera
    sfm_xyz_visible: np.ndarray,      # (N, 3) SfM points visible in camera
    cfg: RansacConfig,
    scene_scale: float = 1.0,
) -> AlignmentResult:
    """RANSAC-fit (a, b) s.t. aligned_depth = a * predicted + b ≈ true SfM depth.

    Raises AlignmentFailed if insufficient inliers.
    """
    H, W = depth_map.shape
    fx, fy, cx, cy = _get_camera_intrinsics(camera)

    u, v, z_true = project_world_to_image(sfm_xyz_visible, camera)

    # Filter: positive depth, inside image
    valid = (z_true > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if valid.sum() < 2:
        raise AlignmentFailed(f"Only {valid.sum()} valid SfM projections inside image")

    u_valid = u[valid].astype(int).clip(0, W - 1)
    v_valid = v[valid].astype(int).clip(0, H - 1)
    z_pred = depth_map[v_valid, u_valid].numpy().astype(np.float64)
    z_sfm = z_true[valid]

    # Remove non-finite predicted depths (zeros come from nan_to_num sanitization in predict())
    finite_mask = np.isfinite(z_pred) & (z_pred != 0)
    n_total, n_finite = len(z_pred), finite_mask.sum()
    z_pred = z_pred[finite_mask]
    z_sfm = z_sfm[finite_mask]
    if len(z_pred) < 2:
        raise AlignmentFailed(
            f"Insufficient finite predicted-depth samples "
            f"({n_finite}/{n_total} SfM-projection pixels have valid depth)"
        )

    inlier_thresh = cfg.inlier_threshold * scene_scale
    best_n_inliers = 0
    best_a, best_b = 1.0, 0.0
    best_inliers = np.zeros(len(z_pred), dtype=bool)
    rng = np.random.default_rng(42)

    for _ in range(cfg.iterations):
        idxs = rng.choice(len(z_pred), size=2, replace=False)
        p1, p2 = z_pred[idxs[0]], z_pred[idxs[1]]
        t1, t2 = z_sfm[idxs[0]], z_sfm[idxs[1]]
        if abs(p1 - p2) < 1e-12:
            continue
        a = (t1 - t2) / (p1 - p2)
        b = t1 - a * p1
        residuals = np.abs(a * z_pred + b - z_sfm)
        inliers = residuals < inlier_thresh
        n = inliers.sum()
        if n > best_n_inliers:
            best_n_inliers = n
            best_inliers = inliers
            best_a, best_b = a, b

    if best_n_inliers < cfg.min_inliers:
        raise AlignmentFailed(
            f"RANSAC found only {best_n_inliers} inliers (need {cfg.min_inliers})")

    # Refit on all inliers via least-squares
    p_in = z_pred[best_inliers]
    t_in = z_sfm[best_inliers]
    A = np.stack([p_in, np.ones_like(p_in)], axis=1)
    result_ls, _, _, _ = np.linalg.lstsq(A, t_in, rcond=None)
    a_fit, b_fit = float(result_ls[0]), float(result_ls[1])

    return AlignmentResult(
        a=a_fit,
        b=b_fit,
        inlier_indices=np.where(best_inliers)[0],
        n_inliers=int(best_n_inliers),
    )


# ---------------------------------------------------------------------------
# Back-project aligned depth to world-space points
# ---------------------------------------------------------------------------

def depth_to_points(
    depth_map,  # torch.Tensor          # (H, W) DAv2 raw output, CPU float32
    camera,
    a: float,
    b: float,
    target_n_points: int,
    sanity_threshold: float,          # absolute depth units
    sfm_xyz_visible: np.ndarray,      # (N, 3) for sanity filter
    image_rgb,  # torch.Tensor (3, H, W)          # (3, H, W) in [0, 1], CPU or CUDA
    max_rejected_fraction: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Back-project aligned depth map to world-space points with sanity filter.

    Returns (xyz, rgb) each (M, 3) numpy float32, or None if sanity filter rejects
    more than max_rejected_fraction of pixels.
    """
    H, W = depth_map.shape
    fx, fy, cx, cy = _get_camera_intrinsics(camera)

    aligned = (a * depth_map.float() + b)    # (H, W) float32

    # Project SfM points for sanity filter
    if len(sfm_xyz_visible) > 0:
        u_sfm, v_sfm, z_sfm = project_world_to_image(sfm_xyz_visible, camera)
        valid_sfm = (
            (z_sfm > 0.01) & (u_sfm >= 0) & (u_sfm < W) & (v_sfm >= 0) & (v_sfm < H)
        )
        u_sfm = u_sfm[valid_sfm].astype(int).clip(0, W - 1)
        v_sfm = v_sfm[valid_sfm].astype(int).clip(0, H - 1)
        z_sfm = z_sfm[valid_sfm]

        sfm_uv = np.stack([u_sfm, v_sfm], axis=1).astype(np.float32)
        sfm_depth_at_uv = z_sfm
    else:
        sfm_uv = None
        sfm_depth_at_uv = None

    aligned_np = aligned.numpy()

    # Build per-pixel (u, v) grid
    u_grid, v_grid = np.meshgrid(np.arange(W), np.arange(H))
    u_flat = u_grid.reshape(-1).astype(np.float32)
    v_flat = v_grid.reshape(-1).astype(np.float32)
    d_flat = aligned_np.reshape(-1)

    # Remove pixels with non-positive or non-finite aligned depth
    valid_depth = (d_flat > 0.01) & np.isfinite(d_flat)

    # Sanity filter: compare against nearest SfM point in (u,v) space
    rejected = np.zeros(H * W, dtype=bool)
    if sfm_uv is not None and len(sfm_uv) >= 3:
        px_tree = cKDTree(sfm_uv)
        all_uv = np.stack([u_flat, v_flat], axis=1)
        _, nearest = px_tree.query(all_uv, workers=-1)
        nearest_sfm_depth = sfm_depth_at_uv[nearest]
        rejected = np.abs(d_flat - nearest_sfm_depth) > sanity_threshold

    rejected_fraction = (rejected & valid_depth).sum() / max(valid_depth.sum(), 1)
    if rejected_fraction > max_rejected_fraction:
        import logging as _log
        _log.getLogger("dense-init").info(
            f"sanity filter rejected {rejected_fraction:.0%} of pixels "
            f"(threshold {max_rejected_fraction:.0%}); using surviving {(1-rejected_fraction):.0%}"
        )

    surviving = valid_depth & ~rejected
    n_survive = surviving.sum()
    if n_survive == 0:
        return None

    # Subsample
    survive_idx = np.where(surviving)[0]
    if n_survive > target_n_points:
        rng = np.random.default_rng(0)
        survive_idx = rng.choice(survive_idx, size=target_n_points, replace=False)

    u_sel = u_flat[survive_idx]
    v_sel = v_flat[survive_idx]
    d_sel = d_flat[survive_idx]

    # Back-project: pixel (u, v, d) → camera frame → world frame
    x_cam = (u_sel - cx) * d_sel / fx
    y_cam = (v_sel - cy) * d_sel / fy
    z_cam = d_sel
    p_cam = np.stack([x_cam, y_cam, z_cam], axis=1)   # (M, 3)

    W2C = _get_w2c(camera)
    R = W2C[:3, :3]
    t = W2C[:3, 3]
    # world = R^T @ (p_cam - t)
    p_world = (R.T @ (p_cam - t).T).T.astype(np.float32)

    # Sample RGB at surviving pixels
    img_np = image_rgb.cpu().float().permute(1, 2, 0).numpy()  # (H, W, 3)
    u_int = u_sel.astype(int).clip(0, W - 1)
    v_int = v_sel.astype(int).clip(0, H - 1)
    rgb_sel = img_np[v_int, u_int, :].astype(np.float32)   # (M, 3)

    return p_world, rgb_sel


# ---------------------------------------------------------------------------
# Multi-view windows (Plan 6a) — §6 consistency, §8 normalisation, §9 residual
#
# Everything below is pure numpy over data already in RAM and is testable
# without a GPU; only DepthAnything3Wrapper.predict_batch needs the model.
# ---------------------------------------------------------------------------

def _intrinsic_matrix(camera) -> np.ndarray:
    """(3, 3) float64 pinhole intrinsics for a MEGS-2 Camera."""
    fx, fy, cx, cy = _get_camera_intrinsics(camera)
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _resize_map(tensor_hw, H: int, W: int):
    """Bilinearly resize a (h, w) torch map to (H, W); no-op when it matches."""
    import torch as _torch
    if tuple(tensor_hw.shape) == (H, W):
        return tensor_hw.cpu()
    return _torch.nn.functional.interpolate(
        tensor_hw.unsqueeze(0).unsqueeze(0), size=(H, W),
        mode="bilinear", align_corners=False,
    ).squeeze(0).squeeze(0).cpu()


def camera_world_center(camera) -> np.ndarray:
    """World-space camera centre: c = -R^T t (DL3)."""
    w2c = _get_w2c(camera)
    return -w2c[:3, :3].T @ w2c[:3, 3]


def normalize_window_extrinsics(ext: np.ndarray) -> Tuple[np.ndarray, float]:
    """Plan 6a §8 — rescale a window's translations to unit median baseline.

    DA3's `_normalize_extrinsics` clamps the median camera distance at 0.1
    (api.py:333). COLMAP's scale is arbitrary, so a session whose inter-camera
    spacing falls under that clamp gets silently mis-normalised. We divide the
    world by the window's own median camera-centre distance before handing the
    poses over, and multiply the returned depth back by the same scalar.

    Args:
        ext: (N, 4, 4) world-to-camera matrices, anchor at index 0.

    Returns:
        (ext_scaled, scale) — `ext_scaled` has t/scale, R untouched; multiply
        DA3's returned depths by `scale` to get back to COLMAP units.
    """
    ext = np.asarray(ext, dtype=np.float64)
    if ext.ndim != 3 or ext.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) extrinsics, got {ext.shape}")

    centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in ext])
    dists = np.linalg.norm(centers[1:] - centers[0], axis=1)
    dists = dists[np.isfinite(dists) & (dists > 0)]
    scale = float(np.median(dists)) if len(dists) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        # Every camera at the same centre: a pure rotation. Leave the poses
        # alone — select_window should already have flagged this window
        # degenerate (§5.5), and a bogus scale would only hide it.
        return ext.copy(), 1.0

    scaled = ext.copy()
    scaled[:, :3, 3] /= scale
    return scaled, scale


def bilinear_sample(image_hw: np.ndarray, u: np.ndarray,
                    v: np.ndarray) -> np.ndarray:
    """Sample a (H, W) map at float pixel coords. Out of bounds -> NaN."""
    img = np.asarray(image_hw, dtype=np.float64)
    H, W = img.shape
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= W - 1) \
        & (v >= 0) & (v <= H - 1)
    us = np.where(inside, u, 0.0)
    vs = np.where(inside, v, 0.0)

    u0 = np.floor(us).astype(np.int64)
    v0 = np.floor(vs).astype(np.int64)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    du = us - u0
    dv = vs - v0

    out = (img[v0, u0] * (1 - du) * (1 - dv)
           + img[v0, u1] * du * (1 - dv)
           + img[v1, u0] * (1 - du) * dv
           + img[v1, u1] * du * dv)
    return np.where(inside, out, np.nan)


def cross_view_consistency(
    depths,
    cameras,
    reproj_px_threshold: float = 1.5,
    depth_rel_threshold: float = 0.01,
) -> np.ndarray:
    """Plan 6a §6 — how many neighbours agree with each anchor pixel.

    This is the frustum-intersection count the plan set out to compute, done
    per pixel by reprojection instead of analytically as polyhedra: it counts
    the cameras that contain the point *and* see it unoccluded, because an
    anchor pixel occluded in view j reprojects onto a different surface and
    fails the round trip.

    All depth maps must come from one DA3 call, so they share a scale.

    Args:
        depths: list of (H_i, W_i) depth maps (numpy or torch); index 0 is the
            anchor, whose grid the result is on.
        cameras: matching MEGS-2 Cameras, same order.
        reproj_px_threshold: step 5's ||p - p'|| limit, anchor pixels.
        depth_rel_threshold: step 5's |d_j - z_j| / z_j limit.

    Returns:
        (H, W) int32 count of agreeing neighbours; 0 wherever the anchor's own
        depth is non-positive or non-finite.
    """
    def _np(d):
        return np.asarray(d.cpu().numpy() if hasattr(d, "cpu") else d,
                          dtype=np.float64)

    anchor_depth = _np(depths[0])
    H, W = anchor_depth.shape
    n_consistent = np.zeros((H, W), dtype=np.int32)
    valid_anchor = np.isfinite(anchor_depth) & (anchor_depth > 0)
    if not valid_anchor.any() or len(depths) < 2:
        return n_consistent

    fx, fy, cx, cy = _get_camera_intrinsics(cameras[0])
    w2c_a = _get_w2c(cameras[0])
    R_a, t_a = w2c_a[:3, :3], w2c_a[:3, 3]

    u_grid, v_grid = np.meshgrid(np.arange(W, dtype=np.float64),
                                 np.arange(H, dtype=np.float64))
    u_flat, v_flat = u_grid.reshape(-1), v_grid.reshape(-1)
    d_flat = np.where(valid_anchor, anchor_depth, 1.0).reshape(-1)

    # 1. Back-project every anchor pixel to a world point.
    p_cam = np.stack([(u_flat - cx) * d_flat / fx,
                      (v_flat - cy) * d_flat / fy,
                      d_flat], axis=1)
    X = (R_a.T @ (p_cam - t_a).T).T

    agree_any = np.zeros(H * W, dtype=np.int32)
    for j in range(1, len(depths)):
        depth_j = _np(depths[j])
        H_j, W_j = depth_j.shape
        fxj, fyj, cxj, cyj = _get_camera_intrinsics(cameras[j])
        w2c_j = _get_w2c(cameras[j])
        R_j, t_j = w2c_j[:3, :3], w2c_j[:3, 3]

        # 2. Project into neighbour j -> pixel p_j, expected depth z_j.
        p_j = (R_j @ X.T).T + t_j
        z_j = p_j[:, 2]
        front = z_j > 1e-9
        safe_z = np.where(front, z_j, 1.0)
        u_j = fxj * p_j[:, 0] / safe_z + cxj
        v_j = fyj * p_j[:, 1] / safe_z + cyj

        # 3. Read j's own depth there.
        d_j = bilinear_sample(depth_j, u_j, v_j)
        good = front & np.isfinite(d_j) & (d_j > 0)
        d_safe = np.where(good, d_j, 1.0)

        # 4. Back-project p_j with d_j and reproject into the anchor.
        p_back = np.stack([(u_j - cxj) * d_safe / fxj,
                           (v_j - cyj) * d_safe / fyj,
                           d_safe], axis=1)
        X_back = (R_j.T @ (p_back - t_j).T).T
        p_a = (R_a @ X_back.T).T + t_a
        z_a = p_a[:, 2]
        front_a = z_a > 1e-9
        safe_za = np.where(front_a, z_a, 1.0)
        u_back = fx * p_a[:, 0] / safe_za + cx
        v_back = fy * p_a[:, 1] / safe_za + cy

        # 5. Agree iff the round trip lands back on the same pixel AND the
        #    neighbour's depth matches the depth we expected there.
        reproj_err = np.hypot(u_back - u_flat, v_back - v_flat)
        rel_depth_err = np.abs(d_safe - z_j) / np.maximum(np.abs(z_j), 1e-12)
        agree = (good & front_a
                 & (reproj_err < reproj_px_threshold)
                 & (rel_depth_err < depth_rel_threshold))
        agree_any += agree.astype(np.int32)

    n_consistent = agree_any.reshape(H, W)
    n_consistent[~valid_anchor] = 0
    return n_consistent


# ---------------------------------------------------------------------------
# §9 — per-window scale residual against the SfM points
# ---------------------------------------------------------------------------

@dataclass
class ScaleResidual:
    """Outcome of the §9 check. `action` is accept | rescale | reject."""

    r: float
    n_points: int
    action: str
    factor: float          # multiply the window's depth by this
    reason: str = ""


def window_scale_residual(
    depth_map,
    camera,
    sfm_xyz_visible: np.ndarray,
) -> Tuple[float, int]:
    """r = median(DA3_depth(p) / SfM_depth(p)) over SfM points seen by `camera`.

    Note this uses the SfM points as a residual check on *one scalar*, not as
    the alignment mechanism — the affine a*d+b fit is gone in the window path.

    Returns (r, n_points_used); r is NaN when nothing usable projected.
    """
    depth = np.asarray(depth_map.cpu().numpy()
                       if hasattr(depth_map, "cpu") else depth_map,
                       dtype=np.float64)
    H, W = depth.shape
    if sfm_xyz_visible is None or len(sfm_xyz_visible) == 0:
        return float("nan"), 0

    u, v, z_sfm = project_world_to_image(sfm_xyz_visible, camera)
    inside = (z_sfm > 1e-6) & (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    if not inside.any():
        return float("nan"), 0

    d_pred = bilinear_sample(depth, u[inside], v[inside])
    z = z_sfm[inside]
    ok = np.isfinite(d_pred) & (d_pred > 0)
    if not ok.any():
        return float("nan"), 0
    ratios = d_pred[ok] / z[ok]
    return float(np.median(ratios)), int(ok.sum())


def resolve_scale_residual(r: float, n_points: int, cfg) -> ScaleResidual:
    """Turn a residual into accept / rescale / reject (Plan 6a §9).

    `cfg` is a MultiViewConfig (residual_* fields).
    """
    if not np.isfinite(r) or n_points < cfg.residual_min_points:
        return ScaleResidual(
            r, n_points, "reject", 1.0,
            f"only {n_points} SfM points to check scale against "
            f"(need {cfg.residual_min_points})")
    if r < cfg.residual_reject_low or r > cfg.residual_reject_high:
        return ScaleResidual(
            r, n_points, "reject", 1.0,
            f"depth/SfM ratio {r:.3f} outside "
            f"[{cfg.residual_reject_low}, {cfg.residual_reject_high}]")
    if abs(r - 1.0) <= cfg.residual_accept_tolerance:
        return ScaleResidual(r, n_points, "accept", 1.0)
    return ScaleResidual(
        r, n_points, "rescale", 1.0 / r,
        f"depth/SfM ratio {r:.3f}; applying scalar {1.0 / r:.3f}")


# ---------------------------------------------------------------------------
# Back-projection with an explicit keep mask (window path)
# ---------------------------------------------------------------------------

def depth_to_points_masked(
    depth_map,
    camera,
    keep_mask: np.ndarray,
    image_rgb,
    target_n_points: int,
    weights: Optional[np.ndarray] = None,
    rng_seed: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Back-project the pixels `keep_mask` selects, no affine fit involved.

    The window path has already decided which pixels to trust (§6 consistency
    x DA3 confidence x the §9 scale check), so unlike depth_to_points this does
    no SfM-nearest-neighbour sanity filtering of its own.

    Returns (xyz, rgb, weight) each length M, or None if nothing survives.
    """
    depth = np.asarray(depth_map.cpu().numpy()
                       if hasattr(depth_map, "cpu") else depth_map,
                       dtype=np.float64)
    H, W = depth.shape
    fx, fy, cx, cy = _get_camera_intrinsics(camera)

    keep = np.asarray(keep_mask, dtype=bool).reshape(-1)
    d_flat = depth.reshape(-1)
    keep &= np.isfinite(d_flat) & (d_flat > 0.01)
    n_keep = int(keep.sum())
    if n_keep == 0:
        return None

    idx = np.where(keep)[0]
    if target_n_points > 0 and n_keep > target_n_points:
        rng = np.random.default_rng(rng_seed)
        idx = np.sort(rng.choice(idx, size=target_n_points, replace=False))

    u_sel = (idx % W).astype(np.float64)
    v_sel = (idx // W).astype(np.float64)
    d_sel = d_flat[idx]

    p_cam = np.stack([(u_sel - cx) * d_sel / fx,
                      (v_sel - cy) * d_sel / fy,
                      d_sel], axis=1)
    w2c = _get_w2c(camera)
    R, t = w2c[:3, :3], w2c[:3, 3]
    p_world = (R.T @ (p_cam - t).T).T.astype(np.float32)

    img_np = (image_rgb.cpu().float().permute(1, 2, 0).numpy()
              if hasattr(image_rgb, "cpu") else np.asarray(image_rgb))
    rgb_sel = img_np[v_sel.astype(int), u_sel.astype(int), :].astype(np.float32)

    if weights is None:
        w_sel = np.ones(len(idx), dtype=np.float32)
    else:
        w_sel = np.asarray(weights, dtype=np.float32).reshape(-1)[idx]

    return p_world, rgb_sel, w_sel


# ---------------------------------------------------------------------------
# Incremental alignment to the dense cloud already on the ground
#
# Plan 6a §9 corrects a window by one scalar fitted to the SfM points. Measured
# on sessions/30d37bfd that is the wrong target: the sparse, textured SfM points
# do not represent the dense surface, so r estimates a scale the dense pixels do
# not have — dividing it out barely helps and makes DA3-LARGE *worse*. What the
# windows actually disagree on is a per-window AFFINE: a scale plus an origin
# offset of 2-3% of depth, which reads as parallel sheets ("pancaking").
#
# So fit the new window against the geometry already reconstructed from the
# earlier images, down the new camera's own rays, and keep the SfM residual as a
# guard rather than the corrector. The first window has nothing to align to and
# falls back to §9, which is what sets absolute scale for everything after it.
# ---------------------------------------------------------------------------

@dataclass
class ReferenceFit:
    """Outcome of aligning one window's depth to the existing dense cloud."""

    scale: float = 1.0
    offset: float = 0.0
    n_matched: int = 0
    inlier_fraction: float = 0.0
    ok: bool = False
    reason: str = ""
    # True when the reference depths spanned too little range to separate
    # scale from offset, so only a scale was fitted.
    scale_only: bool = False
    depth_span: float = 0.0

    def apply(self, depth):
        """corrected = scale * depth + offset."""
        return depth * self.scale + self.offset


def reference_depth_buffer(camera, reference_xyz: np.ndarray, bin_px: int = 4):
    """Nearest-surface depth of `reference_xyz` per coarse pixel bin.

    Nearest, not median: a camera sees the closest surface down each ray, so
    taking the minimum is what makes this occlusion-aware. Coarse bins because
    the reference cloud is subsampled and will not hit every pixel.
    """
    xyz = np.asarray(reference_xyz, dtype=np.float64)
    if xyz.ndim != 2 or len(xyz) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)

    H, W = camera.image_height, camera.image_width
    u, v, z = project_world_to_image(xyz, camera)
    inside = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not inside.any():
        return np.zeros(0, dtype=np.int64), np.zeros(0)

    nb = int(math.ceil(W / bin_px))
    key = ((v[inside] // bin_px).astype(np.int64) * nb
           + (u[inside] // bin_px).astype(np.int64))
    z_in = z[inside]

    order = np.lexsort((z_in, key))          # by cell, then nearest first
    key, z_in = key[order], z_in[order]
    first = np.r_[True, key[1:] != key[:-1]]
    return key[first], z_in[first]


def _huber_affine(x: np.ndarray, y: np.ndarray, delta: float,
                  iters: int) -> Tuple[float, float, np.ndarray]:
    """Robust y ~ s*x + c by IRLS with Huber weights. Returns (s, c, inliers)."""
    s, c = 1.0, 0.0
    w = np.ones_like(x)
    for _ in range(max(1, iters)):
        sw = np.sqrt(w)
        A = np.stack([x * sw, sw], axis=1)
        sol, *_ = np.linalg.lstsq(A, y * sw, rcond=None)
        s, c = float(sol[0]), float(sol[1])
        resid = np.abs(y - (s * x + c))
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-12))
    return s, c, np.abs(y - (s * x + c)) <= delta


def _huber_scale(x: np.ndarray, y: np.ndarray, delta: float,
                 iters: int) -> Tuple[float, float, np.ndarray]:
    """Robust y ~ s*x with no offset. Same IRLS, one parameter."""
    s = 1.0
    w = np.ones_like(x)
    for _ in range(max(1, iters)):
        denom = float(np.sum(w * x * x))
        s = float(np.sum(w * x * y) / denom) if denom > 1e-12 else 1.0
        resid = np.abs(y - s * x)
        w = np.where(resid <= delta, 1.0, delta / np.maximum(resid, 1e-12))
    return s, 0.0, np.abs(y - s * x) <= delta


def fit_depth_to_reference(
    depth_map,
    camera,
    reference_xyz: Optional[np.ndarray],
    cfg,
    valid_mask: Optional[np.ndarray] = None,
) -> ReferenceFit:
    """Fit `scale`/`offset` putting this window's depth onto the existing cloud.

    Args:
        depth_map: (H, W) anchor depth, DA3 units.
        camera: the anchor camera.
        reference_xyz: (N, 3) dense points from the images ingested so far.
        cfg: MultiViewConfig (reference_* fields).
        valid_mask: optional (H, W) bool of pixels worth matching — pass the
            §6 consistency mask so the fit only sees depth we already trust.

    A failed fit is not an error: it means this window is the first, or does not
    overlap what is already there, and the caller falls back to §9.
    """
    fit = ReferenceFit()
    if reference_xyz is None or len(reference_xyz) < cfg.reference_min_correspondences:
        fit.reason = "no dense cloud to align to yet"
        return fit

    depth = np.asarray(depth_map.cpu().numpy() if hasattr(depth_map, "cpu")
                       else depth_map, dtype=np.float64)
    H, W = depth.shape
    bin_px = max(1, int(cfg.reference_bin_px))

    cells, z_ref = reference_depth_buffer(camera, reference_xyz, bin_px)
    if len(cells) < cfg.reference_min_correspondences:
        fit.reason = (f"only {len(cells)} reference cells project into "
                      f"{getattr(camera, 'image_name', 'this view')}")
        return fit

    # Sample this window's own depth at the centre of each occupied cell.
    nb = int(math.ceil(W / bin_px))
    cu = (cells % nb) * bin_px + bin_px / 2.0
    cv = (cells // nb) * bin_px + bin_px / 2.0
    z_pred = bilinear_sample(depth, cu, cv)

    good = np.isfinite(z_pred) & (z_pred > 0) & np.isfinite(z_ref) & (z_ref > 0)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        ui = np.clip(cu.astype(int), 0, W - 1)
        vi = np.clip(cv.astype(int), 0, H - 1)
        good &= vm[vi, ui]
    z_pred, z_ref = z_pred[good], z_ref[good]
    if len(z_pred) < cfg.reference_min_correspondences:
        fit.reason = (f"only {len(z_pred)} usable correspondences "
                      f"(need {cfg.reference_min_correspondences})")
        return fit

    median_depth = float(np.median(z_ref))
    delta = cfg.reference_huber_frac * median_depth

    # Scale and offset are only separable over a real depth span. Looking at a
    # fronto-parallel wall every z is the same, s*z+c is rank-deficient, and
    # the solver returns an arbitrary (s, c) on the line s*z+c=z — which fits
    # the observed depths perfectly and extrapolates disastrously outside them.
    # Under that span, fit the scale alone.
    span = float(z_ref.max() - z_ref.min())
    span_ok = span >= cfg.reference_min_depth_span_frac * max(median_depth, 1e-9)
    if span_ok:
        scale, offset, inliers = _huber_affine(
            z_pred, z_ref, delta, cfg.reference_iters)
    else:
        scale, offset, inliers = _huber_scale(
            z_pred, z_ref, delta, cfg.reference_iters)
    fit.scale_only = not span_ok
    fit.depth_span = span

    fit.scale, fit.offset = scale, offset
    fit.n_matched = int(len(z_pred))
    fit.inlier_fraction = float(inliers.mean())

    if not (np.isfinite(scale) and np.isfinite(offset)):
        fit.reason = "non-finite fit"
    elif abs(scale - 1.0) > cfg.reference_max_scale_dev:
        fit.reason = (f"scale {scale:.3f} deviates more than "
                      f"{cfg.reference_max_scale_dev:.2f} from 1")
    elif abs(offset) > cfg.reference_max_offset_frac * median_depth:
        fit.reason = (f"offset {offset:.3f} exceeds "
                      f"{cfg.reference_max_offset_frac:.0%} of median depth "
                      f"{median_depth:.2f}")
    elif fit.inlier_fraction < cfg.reference_min_inlier_fraction:
        fit.reason = (f"only {fit.inlier_fraction:.0%} of correspondences are "
                      f"inliers (need {cfg.reference_min_inlier_fraction:.0%})")
    else:
        fit.ok = True
    return fit
