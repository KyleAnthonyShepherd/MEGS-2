"""Tests for the viewer's lightweight dense point cloud export (dense.ply,
MEGS-2 ingest contract §8): the atomic PLY writer format + the accumulator
subsample cap.

The functions live in continuous_train.py; the `ct` fixture (tests/conftest.py)
imports it with the CUDA-only deps stubbed, so these run without a GPU (they
still need torch importable — skipped otherwise, like the other ct tests)."""

from pathlib import Path

import numpy as np
import pytest

plyfile = pytest.importorskip("plyfile")
from plyfile import PlyData


# ---------------------------------------------------------------------------
# subsample_points
# ---------------------------------------------------------------------------

def test_subsample_under_cap_is_identity(ct):
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    rgb = np.zeros((4, 3), dtype=np.float32)
    o_xyz, o_rgb = ct.subsample_points(xyz, rgb, max_points=10)
    assert o_xyz is xyz and o_rgb is rgb


def test_subsample_zero_cap_disables(ct):
    xyz = np.zeros((1000, 3), dtype=np.float32)
    rgb = np.zeros((1000, 3), dtype=np.float32)
    o_xyz, _ = ct.subsample_points(xyz, rgb, max_points=0)
    assert len(o_xyz) == 1000


def test_subsample_caps_and_stays_aligned(ct):
    n = 500
    xyz = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 3))
    rgb = xyz.copy()
    o_xyz, o_rgb = ct.subsample_points(xyz, rgb, max_points=50)
    assert len(o_xyz) == 50
    # rows stay xyz/rgb-aligned and in ascending (preserved) order
    np.testing.assert_array_equal(o_xyz, o_rgb)
    assert np.all(np.diff(o_xyz[:, 0]) > 0)


def test_subsample_deterministic(ct):
    xyz = np.random.RandomState(1).rand(300, 3).astype(np.float32)
    rgb = np.zeros((300, 3), dtype=np.float32)
    a, _ = ct.subsample_points(xyz, rgb, max_points=40)
    b, _ = ct.subsample_points(xyz, rgb, max_points=40)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# write_dense_cloud_atomic
# ---------------------------------------------------------------------------

def _read(path):
    return PlyData.read(path)["vertex"]


def test_dense_ply_format_and_color_scaling(ct, tmp_path):
    xyz = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    rgb01 = np.array([[1.0, 0.0, 0.5], [0, 1, 0], [0.2, 0.2, 0.2]], dtype=np.float32)
    path = ct.write_dense_cloud_atomic(str(tmp_path), "sess1", xyz, rgb01)

    assert Path(path).name == "dense.ply"
    assert not Path(path + ".tmp").exists()          # atomic
    el = _read(path)
    assert [p.name for p in el.properties] == ["x", "y", "z", "red", "green", "blue"]
    assert el.count == 3
    np.testing.assert_array_equal(list(el["x"]), [1, 4, 7])
    # [0,1] floats scale to 0..255 (0.5 -> 128, 0.2 -> 51)
    assert list(el["red"]) == [255, 0, 51]
    assert list(el["blue"]) == [128, 0, 51]


def test_dense_ply_accepts_0_255_rgb(ct, tmp_path):
    xyz = np.zeros((2, 3), dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.float32)
    el = _read(ct.write_dense_cloud_atomic(str(tmp_path), "s", xyz, rgb))
    assert list(el["red"]) == [255, 0]
    assert list(el["blue"]) == [0, 255]


def test_dense_ply_noop_without_export_or_points(ct, tmp_path):
    xyz = np.zeros((3, 3), dtype=np.float32)
    rgb = np.zeros((3, 3), dtype=np.float32)
    assert ct.write_dense_cloud_atomic(None, "s", xyz, rgb) is None
    assert ct.write_dense_cloud_atomic(str(tmp_path), None, xyz, rgb) is None
    empty = np.zeros((0, 3), dtype=np.float32)
    assert ct.write_dense_cloud_atomic(str(tmp_path), "s", empty, empty) is None


def test_dense_ply_overwrite_same_path(ct, tmp_path):
    xyz1 = np.zeros((2, 3), dtype=np.float32)
    xyz2 = np.ones((5, 3), dtype=np.float32)
    rgb1 = np.zeros((2, 3), dtype=np.float32)
    rgb2 = np.zeros((5, 3), dtype=np.float32)
    p1 = ct.write_dense_cloud_atomic(str(tmp_path), "s", xyz1, rgb1)
    p2 = ct.write_dense_cloud_atomic(str(tmp_path), "s", xyz2, rgb2)
    assert p1 == p2                    # same session path
    assert _read(p2).count == 5        # latest write wins
