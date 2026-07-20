"""Golden-file comparison for the continuous-training regression harness.

The trigger-firing sequence is the scheduler's behavioural fingerprint: an
identical sequence means the state machine made the same decisions in the
same order. Metrics (splat count, PSNR when available) are compared with
tolerances, since they legitimately wobble across GPU/driver versions.

CLI:  python -m tests.regression.compare golden.json actual.json
Exit code 0 = match, 1 = mismatch (differences printed).
"""

import json
import sys
from typing import List, Optional

DEFAULT_TOLERANCES = {
    "final_splat_count": 0.10,   # relative
    "final_iter": 0.25,          # relative
    "psnr": 0.5,                 # absolute dB
    "ssim": 0.02,                # absolute
}


def compare_trigger_sequence(golden: List[str], actual: List[str]) -> List[str]:
    """Return a list of human-readable differences (empty = identical)."""
    diffs = []
    if golden == actual:
        return diffs
    n = max(len(golden), len(actual))
    for i in range(n):
        g = golden[i] if i < len(golden) else "<end>"
        a = actual[i] if i < len(actual) else "<end>"
        if g != a:
            diffs.append(f"trigger[{i}]: golden={g!r} actual={a!r}")
            if len(diffs) >= 10:
                diffs.append(
                    f"... ({len(golden)} golden vs {len(actual)} actual firings)")
                break
    return diffs


def compare_metrics(golden: dict, actual: dict,
                    tolerances: Optional[dict] = None) -> List[str]:
    """Compare scalar metrics with per-key tolerances.

    Keys ending in _count/_iter use relative tolerance; psnr/ssim absolute.
    Keys present in golden but missing from actual are reported; extra keys
    in actual are ignored (new metrics don't break old goldens).
    """
    tol = dict(DEFAULT_TOLERANCES)
    if tolerances:
        tol.update(tolerances)
    diffs = []
    for key, gval in golden.items():
        if key == "trigger_sequence" or not isinstance(gval, (int, float)) \
                or isinstance(gval, bool):
            continue
        if key not in actual:
            diffs.append(f"metric {key}: missing from actual")
            continue
        aval = actual[key]
        t = tol.get(key)
        if t is None:
            # Unknown metric: require exact match
            if aval != gval:
                diffs.append(f"metric {key}: golden={gval} actual={aval} (exact)")
        elif key in ("psnr", "ssim"):
            if abs(aval - gval) > t:
                diffs.append(
                    f"metric {key}: golden={gval} actual={aval} (abs tol {t})")
        else:
            denom = max(abs(gval), 1e-9)
            if abs(aval - gval) / denom > t:
                diffs.append(
                    f"metric {key}: golden={gval} actual={aval} (rel tol {t})")
    return diffs


def compare_runs(golden: dict, actual: dict,
                 tolerances: Optional[dict] = None) -> List[str]:
    diffs = compare_trigger_sequence(
        golden.get("trigger_sequence", []),
        actual.get("trigger_sequence", []))
    diffs += compare_metrics(golden, actual, tolerances)
    return diffs


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    with open(argv[1]) as f:
        golden = json.load(f)
    with open(argv[2]) as f:
        actual = json.load(f)
    diffs = compare_runs(golden, actual)
    if diffs:
        print(f"REGRESSION: {len(diffs)} difference(s) vs golden:")
        for d in diffs:
            print("  " + d)
        return 1
    print("OK: run matches golden")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
