"""Match-graph parsing and per-camera weight computation.

Ported from GS_On-The-Fly/ContinuosProgressiveTrain.py:
  GetImageMatchingMatrix / GetImagesWeightsFromMatrix
"""

import numpy as np
from typing import List


def parse_match_matrix(
    matrix_path: str,
    names_path: str,
    ordered_camera_names: List[str],
) -> np.ndarray:
    """Load and reorder the co-visibility feature-count matrix.

    The file rows/cols are in upstream-SfM registration order (from imagesNames.txt).
    We reorder both axes to match `ordered_camera_names` (MEGS-2 loader order).

    Returns an (M, M) float32 log-normalised matrix, where M = len(ordered_camera_names).
    Cameras absent from the file get row/col of zeros.

    Two producer formats are accepted (auto-detected):
      - GS_On-The-Fly: imagesNames.txt is one comma-separated line;
        matrix rows are comma-separated.
      - home-server (app/api/trainer.py export_match_matrix): one name per
        line; matrix rows are space-separated.
    """
    with open(names_path, "r") as f:
        raw = f.read().strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) > 1:
        file_names = lines
    else:
        file_names = [n.strip() for n in raw.split(",") if n.strip()]

    # Strip extensions so we match on stems only
    def stem(name: str) -> str:
        return name.rsplit(".", 1)[0]

    file_stems = [stem(n) for n in file_names]
    cam_stems = [stem(n) for n in ordered_camera_names]

    rows_raw = []
    with open(matrix_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sep = "," if "," in line else None  # None = any whitespace
            rows_raw.append([int(x) for x in line.split(sep) if x.strip()])

    n_file = len(file_stems)
    raw_matrix = np.zeros((n_file, n_file), dtype=np.float32)
    for i, row in enumerate(rows_raw):
        if i >= n_file:
            break
        for j, val in enumerate(row):
            if j >= n_file:
                break
            raw_matrix[i, j] = float(val)

    # Log-normalise each row by its own maximum
    row_max = raw_matrix.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1.0, row_max)
    log_matrix = np.log1p(raw_matrix) / np.log1p(row_max)

    # Build index mapping: file order → camera order
    file_stem_to_idx = {s: i for i, s in enumerate(file_stems)}

    M = len(cam_stems)
    result = np.zeros((M, M), dtype=np.float32)
    for i, si in enumerate(cam_stems):
        fi = file_stem_to_idx.get(si)
        if fi is None:
            continue
        for j, sj in enumerate(cam_stems):
            fj = file_stem_to_idx.get(sj)
            if fj is None:
                continue
            result[i, j] = log_matrix[fi, fj]

    return result


def compute_image_weights(
    match_matrix: np.ndarray,
    new_image_indices: List[int],
) -> np.ndarray:
    """Compute per-image training weights from the match matrix.

    New images get weight 1.0.  Existing images get the weighted average of
    their match scores against the new images (iterating up to 4 times to
    propagate to zero-weight images).

    Returns a (M,) float32 array.
    """
    M = match_matrix.shape[0]
    weights = np.zeros(M, dtype=np.float32)

    new_set = set(new_image_indices)
    for idx in new_image_indices:
        weights[idx] = 1.0

    for _ in range(4):
        for i in range(M):
            if i in new_set:
                continue
            scores = match_matrix[i]
            # Weighted sum using already-assigned weights of connected images
            w_sum = 0.0
            s_sum = 0.0
            for j in range(M):
                if j == i:
                    continue
                if scores[j] > 0 and weights[j] > 0:
                    w_sum += scores[j] * weights[j]
                    s_sum += scores[j]
            if s_sum > 0:
                weights[i] = w_sum / s_sum

    # Fallback: images that still have weight 0 get a small uniform weight
    zero_mask = weights == 0
    if zero_mask.any() and not zero_mask.all():
        weights[zero_mask] = weights[~zero_mask].min() * 0.1
    elif zero_mask.all():
        weights[:] = 1.0

    return weights
