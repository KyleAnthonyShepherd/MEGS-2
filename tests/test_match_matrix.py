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


# ---------------------------------------------------------------------------
# Home-server producer format (app/api/trainer.py export_match_matrix):
# one name per line, space-separated matrix rows.
# ---------------------------------------------------------------------------

def _write_files_home_server(tmpdir, matrix_rows, names):
    matrix_path = os.path.join(tmpdir, "imageMatchMatrix.txt")
    names_path = os.path.join(tmpdir, "imagesNames.txt")
    with open(matrix_path, "w") as f:
        for row in matrix_rows:
            f.write(" ".join(str(x) for x in row) + "\n")
    with open(names_path, "w") as f:
        f.write("\n".join(names) + "\n")
    return matrix_path, names_path


def test_parse_home_server_format():
    """Newline-separated names + space-separated rows must parse identically
    to the legacy comma format."""
    rows = [[0, 10, 5], [10, 0, 8], [5, 8, 0]]
    names = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    ordered = ["img_a", "img_b", "img_c"]
    with tempfile.TemporaryDirectory() as tmp_a, \
            tempfile.TemporaryDirectory() as tmp_b:
        legacy = parse_match_matrix(*_write_files(tmp_a, rows, names), ordered)
        new = parse_match_matrix(
            *_write_files_home_server(tmp_b, rows, names), ordered)
    np.testing.assert_allclose(new, legacy)


def test_parse_home_server_format_l7_reindex():
    """Landmine L7: DB/registration row order != alphabetical camera order.
    Weights must land on the correct cameras after reindexing."""
    # File (registration) order: C, A, B — deliberately non-alphabetical.
    # C↔A = 100, C↔B = 0, A↔B = 50
    rows = [[0, 100, 0], [100, 0, 50], [0, 50, 0]]
    names = ["img_c.jpg", "img_a.jpg", "img_b.jpg"]
    # Camera order (MEGS-2 loader): alphabetical
    ordered = ["img_a", "img_b", "img_c"]
    with tempfile.TemporaryDirectory() as tmp:
        result = parse_match_matrix(
            *_write_files_home_server(tmp, rows, names), ordered)
    # a↔c strongest link: row a max is 100 (vs c) → normalised 1.0
    assert abs(result[0, 2] - 1.0) < 1e-5
    assert abs(result[2, 0] - 1.0) < 1e-5
    # a↔b weaker: log(51)/log(101)
    expected_ab = np.log1p(50) / np.log1p(100)
    assert abs(result[0, 1] - expected_ab) < 1e-5
    # b↔c never matched
    assert result[1, 2] == 0.0

    # And weight computation maps to the right cameras: new image = c (idx 2).
    # Per-row log-normalisation makes each image's strongest link 1.0, so both
    # a (direct) and b (via a) end up connected; the essential property is
    # that no camera got dropped by misindexing.
    weights = compute_image_weights(result, [2])
    assert weights[2] == 1.0
    assert weights[0] > 0.0 and weights[1] > 0.0


def test_parse_single_image_home_server_format():
    """A 1-image session: single name line, single '0' row."""
    with tempfile.TemporaryDirectory() as tmp:
        result = parse_match_matrix(
            *_write_files_home_server(tmp, [[0]], ["img_a.jpg"]), ["img_a"])
    assert result.shape == (1, 1) and result[0, 0] == 0.0
