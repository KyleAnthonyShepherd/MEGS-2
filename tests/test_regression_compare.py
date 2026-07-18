"""Tests for tests/regression/compare.py golden comparison logic."""

from tests.regression.compare import (
    compare_trigger_sequence, compare_metrics, compare_runs,
)


def test_identical_sequences_match():
    seq = ["densify@stalled", "fast_prune@converged"]
    assert compare_trigger_sequence(seq, list(seq)) == []


def test_sequence_diff_reports_position():
    diffs = compare_trigger_sequence(
        ["densify@stalled", "fast_prune@stalled"],
        ["densify@stalled", "lightweight_prune@stalled"])
    assert len(diffs) == 1 and "trigger[1]" in diffs[0]


def test_sequence_length_mismatch():
    diffs = compare_trigger_sequence(["a@x"], ["a@x", "b@y"])
    assert any("<end>" in d for d in diffs)


def test_sequence_diff_output_capped():
    diffs = compare_trigger_sequence(["a@x"] * 30, ["b@y"] * 30)
    assert len(diffs) <= 11


def test_metrics_within_tolerance_pass():
    golden = {"final_splat_count": 100_000, "final_iter": 4000, "psnr": 28.0}
    actual = {"final_splat_count": 105_000, "final_iter": 4500, "psnr": 28.3}
    assert compare_metrics(golden, actual) == []


def test_metrics_out_of_tolerance_fail():
    golden = {"final_splat_count": 100_000, "psnr": 28.0}
    actual = {"final_splat_count": 150_000, "psnr": 26.0}
    diffs = compare_metrics(golden, actual)
    assert len(diffs) == 2


def test_missing_metric_reported_extra_ignored():
    diffs = compare_metrics({"psnr": 28.0}, {"ssim": 0.9, "extra": 1})
    assert diffs == ["metric psnr: missing from actual"]


def test_unknown_metric_requires_exact():
    assert compare_metrics({"n_ingests": 10}, {"n_ingests": 10}) == []
    assert len(compare_metrics({"n_ingests": 10}, {"n_ingests": 11})) == 1


def test_compare_runs_combines_both():
    golden = {"trigger_sequence": ["a@x"], "psnr": 28.0}
    actual = {"trigger_sequence": ["b@y"], "psnr": 20.0}
    assert len(compare_runs(golden, actual)) == 2
