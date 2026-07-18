"""Import-level + save_train_state test for continuous_train.py.

Imports the real module with the CUDA-only dependencies (rasterizer,
simple_knn) stubbed, then round-trips save_train_state with a fake
gaussians object. Validates the checkpoint file layout the --resume path
reads, without needing a GPU.
"""

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ct():
    """Import continuous_train with CUDA-bound modules stubbed."""
    sys.path.insert(0, str(REPO_ROOT))

    def stub(name, **attrs):
        if name in sys.modules:
            return sys.modules[name]
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    stub("simple_knn")
    stub("simple_knn._C", distCUDA2=lambda *a, **k: None)
    stub("spherical_gaussian_renderer", render_imp=lambda *a, **k: {})
    stub("fused_ssim", fused_ssim=lambda *a, **k: None)
    # diff rasterizer pulled in transitively by the renderer package normally;
    # stub defensively in case scene modules import it.
    stub("diff_gaussian_rasterization_ms")

    import importlib
    try:
        mod = importlib.import_module("continuous_train")
    except Exception as e:
        pytest.skip(f"continuous_train not importable in this env: {e}")
    return mod


class FakeGaussians:
    def __init__(self):
        self._grace_records = [(0, 10, 55)]

    def capture(self):
        return {"xyz_cohorts": [torch.zeros(4, 3)], "note": "fake"}


class FakeCtrl:
    session_id = "s1"

    def ledger_state(self):
        return {"session_id": "s1", "seen_images": [["s1", "a.jpg", "rid1"]]}


def test_import_and_cli_flags(ct):
    """The module imports and exposes the resume/export CLI surface."""
    assert hasattr(ct, "save_train_state")
    assert ct.TRAIN_STATE_NAME == "train_state.pt"
    src = (REPO_ROOT / "continuous_train.py").read_text()
    assert "--resume" in src and "--export_dir" in src


def test_save_train_state_layout(ct, tmp_path):
    from tests._direct_import import load_scene_module
    convergence = load_scene_module("convergence")
    skipgs_mod = load_scene_module("skipgs")

    monitor = convergence.ConvergenceMonitor(loss_window=10)
    for _ in range(5):
        monitor.update_loss(1.0)
    gate = skipgs_mod.SkipGSGate()
    trainer_state = {"global_iter": 42, "ema_loss": 0.5,
                     "last_snapshot_dir": "/snap"}

    path = ct.save_train_state(str(tmp_path), FakeGaussians(), monitor,
                               gate, FakeCtrl(), trainer_state)
    assert Path(path).name == ct.TRAIN_STATE_NAME
    assert not (tmp_path / (ct.TRAIN_STATE_NAME + ".tmp")).exists()  # atomic

    ckpt = torch.load(path, weights_only=False)
    assert ckpt["version"] == 1
    assert ckpt["trainer"]["global_iter"] == 42
    assert ckpt["grace_records"] == [(0, 10, 55)]
    assert ckpt["ledger"]["session_id"] == "s1"
    assert ckpt["model"]["note"] == "fake"
    # Monitor state restores
    m2 = convergence.ConvergenceMonitor()
    m2.set_state(ckpt["monitor"])
    assert list(m2.loss_history) == [1.0] * 5


def test_save_train_state_none_skipgs(ct, tmp_path):
    from tests._direct_import import load_scene_module
    convergence = load_scene_module("convergence")
    path = ct.save_train_state(str(tmp_path), FakeGaussians(),
                               convergence.ConvergenceMonitor(), None,
                               FakeCtrl(), {"global_iter": 0})
    assert torch.load(path, weights_only=False)["skipgs"] is None
