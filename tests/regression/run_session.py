"""Replay a recorded capture session through continuous_train.py and record
a metrics JSON for golden-file regression comparison.

Requires a CUDA machine (the trainer renders); the harness itself is pure
orchestration. Determinism: fixed PYTHONHASHSEED and --quiet trainer.

Session input, one of:
  - a directory of numbered cumulative snapshot dirs (1/, 2/, ... each with
    images/ + sparse/0/) as produced by tools/simulate_progressive.py — each
    is POSTed in order;
  - a single home-server session dir (images/ + sparse/0/) — POSTed once
    with the home-server payload shape.

Usage:
  python -m tests.regression.run_session --session-dir /path/to/session \
      --model-path /tmp/regress_out --out actual.json \
      [--golden tests/regression/golden_fixture.json]

Exit code: 0 on success (and golden match if --golden), 1 on mismatch,
2 on harness failure.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _post(url: str, payload: dict, timeout: float = 10.0):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode())


def _get(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_for(predicate, timeout: float, interval: float = 0.5, desc: str = ""):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {desc} ({timeout}s)")


def discover_snapshots(session_dir: Path):
    """Return an ordered list of snapshot dirs to POST."""
    numbered = sorted(
        (d for d in session_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name))
    if numbered:
        return numbered
    if (session_dir / "sparse" / "0").is_dir():
        return [session_dir]
    raise SystemExit(f"no snapshots found under {session_dir}")


def run(args) -> dict:
    base = f"http://127.0.0.1:{args.port}"
    model_path = Path(args.model_path)
    model_path.mkdir(parents=True, exist_ok=True)
    events_path = model_path / "events.jsonl"
    if events_path.exists():
        events_path.unlink()  # fresh trigger record for this run

    env = dict(os.environ, PYTHONHASHSEED="0")
    cmd = [
        sys.executable, str(REPO_ROOT / "continuous_train.py"),
        "--model_path", str(model_path),
        "--config", args.config,
        "--http_host", "127.0.0.1", "--http_port", str(args.port),
        "--quiet",
    ]
    if args.resume:
        cmd.append("--resume")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env)
    t_start = time.monotonic()

    try:
        wait_for(lambda: _get(base + "/health") is not None,
                 timeout=120, desc="trainer /health")

        snapshots = discover_snapshots(Path(args.session_dir))
        for i, snap in enumerate(snapshots):
            if snap == Path(args.session_dir):
                payload = {"session_id": "regress",
                           "image_name": f"snapshot_{i}",
                           "sparse_dir": str(snap / "sparse" / "0")}
            else:
                payload = {"snapshot_dir": str(snap),
                           "request_id": f"snap-{snap.name}"}
            status, body = _post(base + "/ingest", payload)
            if status not in (200, 202):
                raise SystemExit(f"ingest of {snap} failed: {status} {body}")
            rid = body["request_id"]
            wait_for(
                lambda: _get(base + f"/status/{rid}")["state"] == "training",
                timeout=args.snapshot_timeout, desc=f"integration of {snap.name}")
            if args.inter_snapshot_delay > 0:
                time.sleep(args.inter_snapshot_delay)

        def converged():
            s = _get(base + "/status")
            return s["monitor_state"] == "converged" and s["queue_depth"] == 0

        wait_for(converged, timeout=args.converge_timeout, interval=2.0,
                 desc="convergence")

        _post(base + "/checkpoint", {}, timeout=60)
        final = _get(base + "/status")
        wall_s = time.monotonic() - t_start
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from _direct_import import load_scene_module
    event_log = load_scene_module("event_log")
    events = event_log.read_events(str(events_path))

    return {
        "n_images": final["n_images"],
        "final_splat_count": final["gaussian_count"],
        "final_iter": final["iter"],
        "wall_s": round(wall_s, 1),
        "n_ingests": sum(1 for e in events if e["event"] == "ingest"),
        "trigger_sequence": event_log.trigger_sequence(events),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/continuous.yaml"))
    ap.add_argument("--port", type=int, default=18666)
    ap.add_argument("--out", default="actual.json")
    ap.add_argument("--golden", default=None,
                    help="golden metrics JSON to compare against")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--snapshot-timeout", type=float, default=600)
    ap.add_argument("--converge-timeout", type=float, default=3600)
    ap.add_argument("--inter-snapshot-delay", type=float, default=0.0)
    args = ap.parse_args()

    metrics = run(args)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {json.dumps(metrics)[:200]}...")

    if args.golden:
        from tests.regression.compare import compare_runs
        with open(args.golden) as f:
            golden = json.load(f)
        diffs = compare_runs(golden, metrics)
        if diffs:
            print(f"REGRESSION vs {args.golden}:")
            for d in diffs:
                print("  " + d)
            sys.exit(1)
        print("OK: matches golden")


if __name__ == "__main__":
    main()
