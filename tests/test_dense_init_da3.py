"""CPU tests for the Depth Anything 3 dense-init backend (Milestone 4).

Model inference needs a GPU + the depth_anything_3 package; these tests
cover everything around it: config plumbing, the conditioning-matrix math,
the direct-depth sanity gate, and the backend dispatch/fallback logic in
dense_init_for_new_images (driven with a fake depth model).
"""

import importlib.util
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_spec = importlib.util.spec_from_file_location(
    "dense_init",
    os.path.join(os.path.dirname(__file__), "..", "scene", "dense_init.py"),
)
_di = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_di)


class FakeCamera:
    def __init__(self, R_w2c=None, t_w2c=None, fovx_deg=60, fovy_deg=45,
                 W=640, H=480, name="fake"):
        R = np.eye(3) if R_w2c is None else R_w2c
        t = np.zeros(3) if t_w2c is None else t_w2c
        self.FoVx = math.radians(fovx_deg)
        self.FoVy = math.radians(fovy_deg)
        self.image_width = W
        self.image_height = H
        self.image_name = name
        self.colmap_id = 0
        self.original_image = torch.full((3, H, W), 0.5)
        Rt = np.eye(4)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        self.world_view_transform = torch.tensor(Rt, dtype=torch.float64).T


# ---------------------------------------------------------------------------
# Config + wrapper surface
# ---------------------------------------------------------------------------

def test_da3_config_defaults():
    cfg = _di.DA3Config()
    assert cfg.model_name == "depth-anything/DA3-SMALL"
    assert cfg.conditioning is True
    assert cfg.process_res == 504


def test_wrapper_constructor_does_not_load_model():
    w = _di.DepthAnything3Wrapper(_di.DA3Config())
    assert w.model is None
    assert w.accepts_camera is True
    # DAv2 wrapper must NOT advertise camera conditioning
    assert not getattr(_di.DepthAnythingV2Wrapper(_di.DAv2Config()),
                       "accepts_camera", False)


# ---------------------------------------------------------------------------
# Conditioning math
# ---------------------------------------------------------------------------

def test_conditioning_matrices():
    theta = math.radians(30)
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta), math.cos(theta), 0],
                  [0, 0, 1.0]])
    t = np.array([1.0, -2.0, 3.0])
    cam = FakeCamera(R_w2c=R, t_w2c=t)
    ext, ixt = _di.camera_to_da3_conditioning(cam)

    assert ext.shape == (1, 4, 4) and ixt.shape == (1, 3, 3)
    assert ext.dtype == np.float32 and ixt.dtype == np.float32
    np.testing.assert_allclose(ext[0, :3, :3], R, atol=1e-6)
    np.testing.assert_allclose(ext[0, :3, 3], t, atol=1e-6)
    np.testing.assert_allclose(ext[0, 3], [0, 0, 0, 1], atol=1e-6)

    fx_expected = 640 / (2 * math.tan(math.radians(60) / 2))
    fy_expected = 480 / (2 * math.tan(math.radians(45) / 2))
    np.testing.assert_allclose(ixt[0, 0, 0], fx_expected, rtol=1e-5)
    np.testing.assert_allclose(ixt[0, 1, 1], fy_expected, rtol=1e-5)
    np.testing.assert_allclose(ixt[0, 0, 2], 320, rtol=1e-5)
    np.testing.assert_allclose(ixt[0, 1, 2], 240, rtol=1e-5)


# ---------------------------------------------------------------------------
# Direct-depth sanity gate
# ---------------------------------------------------------------------------

def _result(a):
    return _di.AlignmentResult(a=a, b=0.1, inlier_indices=np.arange(3),
                               n_inliers=3)


def test_validate_alignment_da3_rejects_negative_a():
    ok, reason = _di.validate_alignment("da3", _result(-0.5))
    assert not ok and "direct depth" in reason


def test_validate_alignment_da3_accepts_positive_a():
    assert _di.validate_alignment("da3", _result(2.0)) == (True, "")


def test_validate_alignment_dav2_accepts_negative_a():
    """DAv2 predicts disparity — negative a is the normal fit."""
    assert _di.validate_alignment("dav2", _result(-0.5))[0]


# ---------------------------------------------------------------------------
# Config plumbing through continuous_train.load_config
# ---------------------------------------------------------------------------

def test_load_config_backend_and_da3(ct, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "dense_init:\n"
        "  backend: da3\n"
        "  da3_fallback_to_dav2: false\n"
        "  da3:\n"
        "    model_name: depth-anything/DA3-BASE\n"
        "    conditioning: false\n"
    )
    cfg = ct.load_config(str(cfg_path))
    assert cfg.dense_init.backend == "da3"
    assert cfg.dense_init.da3_fallback_to_dav2 is False
    assert cfg.dense_init.da3.model_name == "depth-anything/DA3-BASE"
    assert cfg.dense_init.da3.conditioning is False
    # Defaults preserved elsewhere
    assert cfg.dense_init.da3.process_res == 504
    assert cfg.dense_init.dav2.model_size == "small"


def test_load_config_default_backend_is_dav2(ct, tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("dense_init:\n  enabled: true\n")
    assert ct.load_config(str(cfg_path)).dense_init.backend == "dav2"


# ---------------------------------------------------------------------------
# Backend dispatch + DAv2 fallback in dense_init_for_new_images
# ---------------------------------------------------------------------------

def small_cam():
    """A small-resolution camera so per-pixel back-projection stays cheap."""
    return FakeCamera(W=64, H=48)


class SyntheticScene:
    """Fake ProgressiveScene: a fronto-parallel plane of SfM points."""

    def __init__(self, cam, depth=2.0, n=100):
        self.cameras_extent = 5.0
        rng = np.random.default_rng(0)
        # World == camera frame for the identity FakeCamera: points on a
        # z=depth plane, inside the frustum.
        x = rng.uniform(-0.5, 0.5, n)
        y = rng.uniform(-0.4, 0.4, n)
        z = np.full(n, depth) + rng.uniform(-0.5, 0.5, n)
        self._xyz = np.stack([x, y, z], axis=1)
        self._cam = cam

    def get_sfm_points_visible_to(self, cam):
        return self._xyz, np.ones(len(self._xyz), dtype=bool)


class FakeDepthModel:
    """Depth model returning a constant-plane 'depth' map.

    sign=+1 emulates DA3 (direct depth: raw correlates positively);
    sign=-1 emulates a broken/disparity-like output that fits with a < 0.
    """

    def __init__(self, sign=1.0, accepts_camera=False):
        self.sign = sign
        self.accepts_camera = accepts_camera
        self.calls = []

    def predict(self, image_chw, camera=None):
        self.calls.append(camera)
        H, W = image_chw.shape[1], image_chw.shape[2]
        # Raw values proportional to true depth (sign flips correlation);
        # add a gradient so RANSAC has slope to fit.
        v = torch.linspace(1.8, 2.7, H).unsqueeze(1).expand(H, W)
        return self.sign * v.clone()


class RecordingGaussians:
    def __init__(self):
        self.get_xyz = torch.zeros((0, 3))
        self.expanded = []
        self.marked = []

    def expand_from_pcd(self, pcd, mask, extent, birth_iter=0):
        self.expanded.append(len(pcd.points))
        self.get_xyz = torch.zeros((int(mask.sum()), 3))

    def mark_recently_added(self, sl, iteration, grace_iters):
        self.marked.append((sl, iteration, grace_iters))


def _dense_cfg(ct, backend="da3", fallback=True):
    cfg = ct.DenseInitConfig()
    cfg.backend = backend
    cfg.da3_fallback_to_dav2 = fallback
    cfg.min_sfm_points_for_alignment = 10
    cfg.min_sfm_depth_range_fraction = 0.01
    cfg.target_dense_points_per_image = 500
    cfg.max_rejected_fraction = 1.0
    return cfg


def test_da3_negative_a_falls_back_to_dav2(ct, monkeypatch):
    """When every DA3 fit violates the direct-depth gate, the same cameras
    must be retried through a DAv2 context."""
    cam = small_cam()
    scene = SyntheticScene(cam)
    gaussians = RecordingGaussians()

    da3_model = FakeDepthModel(sign=-1.0, accepts_camera=True)   # bad fits
    dav2_model = FakeDepthModel(sign=+1.0, accepts_camera=False)  # good fits

    class FakeCtx:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            return self.model

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ct, "DepthAnything3Wrapper", lambda cfg: FakeCtx(da3_model))
    monkeypatch.setattr(ct, "DepthAnythingV2Wrapper", lambda cfg: FakeCtx(dav2_model))

    ct.dense_init_for_new_images(
        gaussians, scene, [cam], _dense_cfg(ct, "da3", fallback=True),
        opt=None, current_iter=0, dav2_model=None, max_gaussians=0)

    # DA3 was tried with camera conditioning, then DAv2 retried without
    assert da3_model.calls == [cam]
    assert dav2_model.calls == [None]
    # DAv2 pass produced points
    assert gaussians.expanded and gaussians.marked


def test_da3_negative_a_no_fallback_skips_image(ct, monkeypatch):
    cam = small_cam()
    scene = SyntheticScene(cam)
    gaussians = RecordingGaussians()
    da3_model = FakeDepthModel(sign=-1.0, accepts_camera=True)
    dav2_used = []

    class FakeCtx:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            return self.model

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ct, "DepthAnything3Wrapper", lambda cfg: FakeCtx(da3_model))
    monkeypatch.setattr(ct, "DepthAnythingV2Wrapper",
                        lambda cfg: dav2_used.append(1) or FakeCtx(da3_model))

    ct.dense_init_for_new_images(
        gaussians, scene, [cam], _dense_cfg(ct, "da3", fallback=False),
        opt=None, current_iter=0, dav2_model=None, max_gaussians=0)

    assert not dav2_used
    assert not gaussians.expanded


def test_da3_positive_a_adds_points_directly(ct, monkeypatch):
    cam = small_cam()
    scene = SyntheticScene(cam)
    gaussians = RecordingGaussians()
    da3_model = FakeDepthModel(sign=+1.0, accepts_camera=True)

    class FakeCtx:
        def __enter__(self):
            return da3_model

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ct, "DepthAnything3Wrapper", lambda cfg: FakeCtx())

    ct.dense_init_for_new_images(
        gaussians, scene, [cam], _dense_cfg(ct, "da3"),
        opt=None, current_iter=7, dav2_model=None, max_gaussians=0)

    assert da3_model.calls == [cam]
    assert gaussians.expanded
    # Grace protection applied at the current iter
    assert gaussians.marked[0][1] == 7


def test_dav2_backend_unchanged(ct, monkeypatch):
    """backend=dav2 never touches the DA3 wrapper."""
    cam = small_cam()
    scene = SyntheticScene(cam)
    gaussians = RecordingGaussians()
    dav2_model = FakeDepthModel(sign=+1.0)
    da3_used = []

    class FakeCtx:
        def __enter__(self):
            return dav2_model

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ct, "DepthAnythingV2Wrapper", lambda cfg: FakeCtx())
    monkeypatch.setattr(ct, "DepthAnything3Wrapper",
                        lambda cfg: da3_used.append(1))

    ct.dense_init_for_new_images(
        gaussians, scene, [cam], _dense_cfg(ct, "dav2"),
        opt=None, current_iter=0, dav2_model=None, max_gaussians=0)

    assert not da3_used
    assert dav2_model.calls == [None]
    assert gaussians.expanded
