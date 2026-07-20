"""Flat-index bookkeeping for cohort-based per-Gaussian arrays.

The model stores parameters as per-cohort tensors; the flat Gaussian order
is the concatenation of cohorts. `_append_to_cohort` inserts new rows at
the end of *each cohort* — i.e. mid-flat-order — while several call sites
historically assumed appended rows land at the flat end (a global
`torch.cat`). These helpers compute the correct order.

"endcat order" below means: all old rows in their old flat order, followed
by the appended chunks in cohort-ascending order — the order a naive
`torch.cat((old, new_c0, new_c1, ...))` produces.

Pure numpy; unit-tested on CPU (tests/test_index_remap.py).
"""

from typing import List, Sequence, Tuple

import numpy as np


def endcat_to_final_perm(sizes_before: Sequence[int],
                         appended: Sequence[int]) -> np.ndarray:
    """Permutation p (len n_final) with final_tensor = endcat_tensor[p].

    sizes_before[ci] = cohort ci's row count before the append;
    appended[ci] = rows appended to cohort ci (0 when none).
    """
    assert len(sizes_before) == len(appended)
    idx = np.empty(sum(sizes_before) + sum(appended), dtype=np.int64)
    pos = 0
    off_old = 0
    off_new = sum(sizes_before)
    for s, a in zip(sizes_before, appended):
        idx[pos:pos + s] = np.arange(off_old, off_old + s)
        pos += s
        idx[pos:pos + a] = np.arange(off_new, off_new + a)
        pos += a
        off_old += s
        off_new += a
    return idx


def old_to_new_after_append(sizes_before: Sequence[int],
                            appended: Sequence[int]) -> np.ndarray:
    """Map old flat index -> new flat index after per-cohort appends."""
    n_old = sum(sizes_before)
    out = np.empty(n_old, dtype=np.int64)
    off_old = 0
    off_final = 0
    for s, a in zip(sizes_before, appended):
        out[off_old:off_old + s] = np.arange(off_final, off_final + s)
        off_old += s
        off_final += s + a
    return out


def old_to_new_after_prune(keep_mask: np.ndarray) -> np.ndarray:
    """Map old flat index -> new flat index after pruning; -1 = removed."""
    keep_mask = np.asarray(keep_mask, dtype=bool)
    out = np.full(len(keep_mask), -1, dtype=np.int64)
    out[keep_mask] = np.arange(int(keep_mask.sum()))
    return out


def remap_grace_records(records: List[Tuple[int, int, int]],
                        old_to_new: np.ndarray) -> List[Tuple[int, int, int]]:
    """Translate grace records (start, stop, expires_iter) through an index
    map. A record whose covered rows scatter or partially survive becomes
    one record per surviving consecutive run; fully-pruned records drop.
    """
    n_old = len(old_to_new)
    out: List[Tuple[int, int, int]] = []
    for start, stop, expires in records:
        start = max(0, start)
        stop = min(stop, n_old)
        if start >= stop:
            continue
        new_pos = old_to_new[start:stop]
        new_pos = np.sort(new_pos[new_pos >= 0])
        if len(new_pos) == 0:
            continue
        run_start = prev = int(new_pos[0])
        for p in new_pos[1:]:
            p = int(p)
            if p != prev + 1:
                out.append((run_start, prev + 1, expires))
                run_start = p
            prev = p
        out.append((run_start, prev + 1, expires))
    return out
