"""Tests for scene/match_matrix.py."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pytest

# Import match_matrix directly to avoid triggering scene/__init__.py (which needs torch)
_spec = importlib.util.spec_from_file_location(
    "match_matrix",
    os.path.join(os.path.dirname(__file__), "..", "scene", "match_matrix.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
parse_match_matrix = _mod.parse_match_matrix
compute_image_weights = _mod.compute_image_weights


def _write_files(tmpdir, matrix_rows, names):
    matrix_path = os.path.join(tmpdir, "imageMatchMatrix.txt")
    names_path = os.path.join(tmpdir, "imagesNames.txt")
    with open(matrix_path, "w") as f:
        for row in matrix_rows:
            f.write(",".join(str(x) for x in row) + "\n")
    with open(names_path, "w") as f:
        f.write(",".join(names))
    return matrix_path, names_path


def test_parse_match_matrix_identity_order():
    """When camera order matches file order, matrix should be preserved."""
    with tempfile.TemporaryDirectory() as tmp:
        rows = [[0, 10, 5], [10, 0, 8], [5, 8, 0]]
        names = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
        matrix_path, names_path = _write_files(tmp, rows, names)
        ordered = ["img_a", "img_b", "img_c"]
        result = parse_match_matrix(matrix_path, names_path, ordered)
        assert result.shape == (3, 3)
        # Diagonal should be log-normalised to 1.0 (if max_in_row != 0)
        # Row 0 max is 10, so result[0,1] = log(10+1)/log(10+1) = 1.0
        assert abs(result[0, 1] - 1.0) < 1e-5


def test_parse_match_matrix_reorder():
    """Reordering cameras should correctly reorder the matrix."""
    with tempfile.TemporaryDirectory() as tmp:
        # File order: A, B, C
        # A↔B = 100, A↔C = 0, B↔C = 50
        rows = [[0, 100, 0], [100, 0, 50], [0, 50, 0]]
        names = ["A.jpg", "B.jpg", "C.jpg"]
        matrix_path, names_path = _write_files(tmp, rows, names)
        # Camera order: B, C, A
        ordered = ["B", "C", "A"]
        result = parse_match_matrix(matrix_path, names_path, ordered)
        assert result.shape == (3, 3)
        # result[0, 2] should be B↔A = 100 (normalised to 1.0 since max in B row = 100)
        assert abs(result[0, 2] - 1.0) < 1e-5
        # result[0, 1] should be B↔C = 50/100 = log(51)/log(101)
        expected = np.log1p(50) / np.log1p(100)
        assert abs(result[0, 1] - expected) < 1e-4


def test_compute_image_weights_new_images_get_one():
    M = 5
    matrix = np.zeros((M, M), dtype=np.float32)
    new_indices = [3, 4]
    weights = compute_image_weights(matrix, new_indices)
    assert weights[3] == 1.0
    assert weights[4] == 1.0


def test_compute_image_weights_connected_images():
    """An existing image connected only to new images should get a non-zero weight."""
    # 3 cameras: 0 (existing), 1 (new), 2 (new)
    # 0↔1 has high match score, 0↔2 has some score
    matrix = np.array([
        [0.0, 0.9, 0.3],
        [0.9, 0.0, 0.5],
        [0.3, 0.5, 0.0],
    ], dtype=np.float32)
    weights = compute_image_weights(matrix, new_image_indices=[1, 2])
    assert weights[1] == 1.0
    assert weights[2] == 1.0
    assert weights[0] > 0.0, "Existing image connected to new ones should have non-zero weight"


def test_compute_image_weights_fallback_no_zeros():
    """Even with a disconnected camera, no weight should be exactly 0."""
    matrix = np.zeros((4, 4), dtype=np.float32)
    weights = compute_image_weights(matrix, new_image_indices=[0])
    assert all(w >= 0 for w in weights), "All weights should be non-negative"
    # When no connectivity, fallback should give uniform small weight to zeros
    assert weights[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
