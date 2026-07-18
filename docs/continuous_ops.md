# Continuous trainer — operations: events, crash recovery, regression

## Structured event log

The trainer appends one JSON object per line to `<model_path>/events.jsonl`:
every ingest, trigger firing (`densify`, `fast_prune`, `lightweight_prune`,
`cull_sg_axes`), `converged` transition, `train_state` save, and `resume`.
Each record carries the iter, monitor state, and splat counts. This is the
first thing to read when a field session misbehaves, and the input to the
regression harness's trigger-sequence comparison.

## Crash resilience: train_state.pt + --resume

Every `training.train_state_interval_iters` optimizer iters (default 500;
0 = off) and at every converged-idle transition, the trainer atomically
writes `<model_path>/train_state.pt` containing:

- model tensors + Adam optimizer state (`capture()`/`restore()`),
- grace-protection records for recently added Gaussians,
- ConvergenceMonitor and SkipGS gate state,
- the ingest idempotency ledger (so an upstream retry of an
  already-integrated image is still a no-op after restart),
- trainer-loop counters (`global_iter`, EMA loss, anti-thrash floors,
  per-image error EMA, match-matrix weights, last snapshot dir).

Restart with:

```
python continuous_train.py --model_path <same path> --config ... --resume
```

Resume re-ingests the last snapshot directory (snapshots are cumulative, so
this rebuilds the full camera set and sparse cloud), then restores the
model and scheduler state on top. If `train_state.pt` is absent, `--resume`
logs a warning and starts fresh.

Not persisted (rebuilt or accepted as loss): the in-flight ingest queue
(the home-server retries, and the ledger dedups), `mask_blur`, and any
partial iteration.

## Always-on invariants

`check_invariants` runs after every ingest and trigger firing; violations
log an error and emit an `invariant_violation` event (never crash training):

- **ceiling_exceeded** — splat count above `training.num_max_ceiling`;
- **optimizer_binding** — Adam state keyed by a parameter object no longer
  bound in any param group, i.e. momentum was silently discarded (the T1
  bug pattern; `scene/optim_guard.py`).

Two more invariants are structural rather than checked: all triggers in an
iter receive the monitor state snapshotted at iter start (commit 6fbdf9a),
and prunes respect grace protection — `run_lightweight_prune` masks
grace-protected rows, and `opacity_size_prune` /
`densify_and_prune_split` take `grace_iter` to do the same. Grace records
are index ranges that previously went stale after any prune or mid-order
densify append; `scene/index_remap.py` now remaps them (and fixes the
`_sg_axis_count` / split-parent prune-mask end-append misalignment in the
densify paths — see tests/test_index_remap.py).

## Regression harness

GPU machine required. Replay a session and record metrics:

```
python -m tests.regression.run_session \
    --session-dir /path/to/snapshots --model-path /tmp/regress \
    --out actual.json [--golden tests/regression/golden_<name>.json]
```

`--session-dir` accepts either numbered cumulative snapshot dirs
(`tools/simulate_progressive.py` output) or a single home-server session
dir. The output JSON records final splat count / iter / image count, wall
time, and the ordered trigger-firing sequence (`densify@stalled`, ...) —
the scheduler's behavioural fingerprint. `--golden` compares against a
stored baseline: the trigger sequence must match exactly; scalar metrics
use tolerances (`tests/regression/compare.py`). To re-golden after an
intentional scheduler change, copy `actual.json` over the golden and state
the rationale in the commit message.

Kill/resume check: run once to convergence, kill the trainer mid-session,
rerun with `--resume` and compare final metrics against the uninterrupted
golden — quality should land within the same tolerances.
