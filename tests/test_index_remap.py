"""Tests for scene/index_remap.py — flat-index bookkeeping under
per-cohort appends and prunes.

These pin the fix for the end-append misalignment family: per-Gaussian
arrays tracked with a global `torch.cat` while _append_to_cohort inserts
rows mid-flat-order (end of each cohort). The same helpers keep grace
records valid across prunes.
"""

import numpy as np

from tests._direct_import import load_scene_module

ir = load_scene_module("index_remap")


# ---------------------------------------------------------------------------
# endcat_to_final_perm
# ---------------------------------------------------------------------------

def test_perm_identity_when_nothing_appended():
    p = ir.endcat_to_final_perm([3, 2], [0, 0])
    np.testing.assert_array_equal(p, np.arange(5))


def test_perm_append_to_last_cohort_is_identity():
    """Appending only to the last cohort IS an end-append — permutation
    must be identity (the old code was correct in this case)."""
    p = ir.endcat_to_final_perm([3, 2], [0, 2])
    np.testing.assert_array_equal(p, np.arange(7))


def test_perm_append_to_first_cohort_reorders():
    # cohorts: A(2 rows), B(2 rows); 1 row appended to A.
    # endcat order: A0 A1 B0 B1 | a0      final order: A0 A1 a0 B0 B1
    p = ir.endcat_to_final_perm([2, 2], [1, 0])
    np.testing.assert_array_equal(p, [0, 1, 4, 2, 3])


def test_perm_gathers_endcat_into_final_order():
    """Applying the permutation to data in endcat order must yield data in
    true flat (per-cohort) order — the exact way the model uses it."""
    sizes, appended = [2, 1, 3], [2, 0, 1]
    # Label rows: old rows "cX_Y", appended rows "aX_Y"
    endcat = (["c0_0", "c0_1", "c1_0", "c2_0", "c2_1", "c2_2"]
              + ["a0_0", "a0_1", "a2_0"])
    p = ir.endcat_to_final_perm(sizes, appended)
    final = [endcat[i] for i in p]
    assert final == ["c0_0", "c0_1", "a0_0", "a0_1",
                     "c1_0",
                     "c2_0", "c2_1", "c2_2", "a2_0"]


def test_perm_is_a_valid_permutation():
    p = ir.endcat_to_final_perm([5, 0, 3, 2], [2, 4, 0, 3])
    assert sorted(p.tolist()) == list(range(19))


# ---------------------------------------------------------------------------
# old_to_new maps
# ---------------------------------------------------------------------------

def test_old_to_new_after_append_shifts_by_prior_insertions():
    # A(2)+1 appended, B(2)+0: old B rows shift right by 1
    m = ir.old_to_new_after_append([2, 2], [1, 0])
    np.testing.assert_array_equal(m, [0, 1, 3, 4])


def test_old_to_new_after_append_consistent_with_perm():
    sizes, appended = [3, 1, 4], [2, 1, 0]
    p = ir.endcat_to_final_perm(sizes, appended)
    m = ir.old_to_new_after_append(sizes, appended)
    # old row j (endcat index j) must land at final position m[j]
    for j in range(sum(sizes)):
        assert p[m[j]] == j


def test_old_to_new_after_prune():
    keep = np.array([True, False, True, True, False])
    np.testing.assert_array_equal(
        ir.old_to_new_after_prune(keep), [0, -1, 1, 2, -1])


# ---------------------------------------------------------------------------
# remap_grace_records
# ---------------------------------------------------------------------------

def test_grace_records_shift_after_append():
    # record covers old rows 2..4 (cohort B), 2 rows appended to cohort A
    m = ir.old_to_new_after_append([2, 3], [2, 0])
    out = ir.remap_grace_records([(2, 5, 100)], m)
    assert out == [(4, 7, 100)]


def test_grace_records_survive_full_prune_of_others():
    # rows 0-1 pruned; record covered rows 2..4 → shifts to 0..2
    keep = np.array([False, False, True, True, True])
    out = ir.remap_grace_records(
        [(2, 5, 100)], ir.old_to_new_after_prune(keep))
    assert out == [(0, 3, 100)]


def test_grace_record_partially_pruned_splits_into_runs():
    # record covers 0..5; rows 2 and 3 pruned → survivors 0,1 (→0,1) and
    # 4,5 (→2,3): one merged run 0..4 since they're consecutive post-prune
    keep = np.array([True, True, False, False, True, True])
    out = ir.remap_grace_records(
        [(0, 6, 50)], ir.old_to_new_after_prune(keep))
    assert out == [(0, 4, 50)]


def test_grace_record_split_by_appended_rows():
    # record covers rows 0..4 spanning cohorts A(2) and B(3); 1 row appended
    # to A lands between them → two runs
    m = ir.old_to_new_after_append([2, 3], [1, 0])
    out = ir.remap_grace_records([(0, 5, 77)], m)
    assert out == [(0, 2, 77), (3, 6, 77)]


def test_grace_record_fully_pruned_drops():
    keep = np.array([True, False, False, True])
    out = ir.remap_grace_records(
        [(1, 3, 10)], ir.old_to_new_after_prune(keep))
    assert out == []


def test_grace_records_preserve_expiry_and_out_of_bounds_clamped():
    m = ir.old_to_new_after_prune(np.array([True, True]))
    out = ir.remap_grace_records([(0, 99, 42), (5, 9, 3)], m)
    assert out == [(0, 2, 42)]


def test_grace_records_returns_plain_int_tuples():
    """train_state.pt and JSON logs store these; keep them python ints."""
    m = ir.old_to_new_after_prune(np.array([True, True, True]))
    out = ir.remap_grace_records([(0, 2, 9)], m)
    (start, stop, exp), = out
    assert type(start) is int and type(stop) is int and type(exp) is int


# ---------------------------------------------------------------------------
# End-to-end simulation of the historical bug
# ---------------------------------------------------------------------------

def test_simulated_axis_count_stays_aligned_through_clone_and_prune():
    """Simulate two cohorts with distinct axis counts, densify-clone in the
    FIRST cohort, then prune — with the fix, axis counts must track their
    Gaussians; the old global-cat bookkeeping scrambles them."""
    rng = np.random.default_rng(1)
    sizes = [4, 3]
    axis = np.array([1, 1, 1, 1, 2, 2, 2])  # cohort0 → 1 axis, cohort1 → 2

    # Clone rows 1 and 3 of cohort 0 (appended to end of cohort 0)
    appended = [2, 0]
    cloned_axis = axis[[1, 3]]
    endcat = np.concatenate([axis, cloned_axis])
    p = ir.endcat_to_final_perm(sizes, appended)
    final_axis = endcat[p]
    # True flat order: cohort0 old(4) + clones(2), then cohort1(3)
    np.testing.assert_array_equal(final_axis, [1, 1, 1, 1, 1, 1, 2, 2, 2])

    # The OLD behaviour (plain cat) would leave cohort-1 axis counts at
    # positions 4..6 where cohort-0 clones actually live:
    assert not np.array_equal(endcat, final_axis)

    # Now prune one cohort-0 row and one cohort-1 row; axis counts follow
    keep = np.ones(9, dtype=bool)
    keep[0] = False
    keep[7] = False
    np.testing.assert_array_equal(
        final_axis[keep], [1, 1, 1, 1, 1, 2, 2])
