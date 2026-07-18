"""Shared fixtures: import continuous_train with CUDA-only deps stubbed."""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def ct():
    """The real continuous_train module, importable without a GPU."""
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
    stub("diff_gaussian_rasterization_ms")

    import importlib
    try:
        mod = importlib.import_module("continuous_train")
    except Exception as e:
        pytest.skip(f"continuous_train not importable in this env: {e}")
    return mod
