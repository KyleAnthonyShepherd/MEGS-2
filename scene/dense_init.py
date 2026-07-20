"""Dense initialization via Depth Anything v2.

Three public entry points:
  DepthAnythingV2Wrapper  — context manager: loads DAv2 on __enter__, frees on __exit__
  align_depth_to_sfm      — RANSAC alignment of DAv2 depth to SfM ground-truth depths
  depth_to_points         — back-project aligned depth map to world-space point cloud

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


def camera_to_da3_conditioning(camera):
    """Build DA3 conditioning arrays from a MEGS-2 Camera.

    Returns (extrinsics (1, 4, 4), intrinsics (1, 3, 3)) float32 numpy —
    OpenCV/COLMAP world-to-camera convention, which is what both MEGS-2
    (DL3) and DA3 use.
    """
    ext = _get_w2c(camera).astype(np.float32)[None]
    fx, fy, cx, cy = _get_camera_intrinsics(camera)
    ixt = np.array([[fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0]], dtype=np.float32)[None]
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
