# Trainer ↔ home-server integration contract

This documents the interface between `continuous_train.py` (this repo) and
the home-server SfM app (`app/api/trainer.py` on that side). Both sides must
change together; restate this file in session notes when either side moves.

## POST /ingest

Preferred payload (what the home-server sends after each sparse/0 promotion):

```json
{
  "session_id": "30d37bfd",
  "image_name": "img_0005_20260429_185437.jpg",
  "image_path": "/abs/path/sessions/30d37bfd/images/img_0005_....jpg",
  "sparse_dir": "/abs/path/sessions/30d37bfd/sparse/0"
}
```

- The snapshot root the trainer loads is derived as `sparse_dir/../..`
  (the session directory), which must contain `images/` and `sparse/0/`
  with `cameras.bin|txt`. Malformed or missing paths → `400`.
- **Idempotent** on `(session_id, image_name)`: a retried POST is a `200`
  no-op returning the original `request_id`. First-time accepts are `202`.
- One trainer process serves one session. The first session-tagged ingest
  pins `session_id`; a different session later → `409`.
- Legacy payload `{"snapshot_dir": ..., "request_id": ...}` is still
  accepted (used by `tools/simulate_progressive.py`); it is never deduped.

## POST /pause  (GPU serialization)

Empty JSON body. The home-server calls this **before every image's COLMAP
work** and blocks up to 30 s for the ack. The trainer stops at a safe point
(between optimizer steps / after the current ingest — never mid-backward),
moves **all** GPU state to CPU RAM — every cohort `nn.Parameter`, the buffers
(`max_radii2D`, `xyz_gradient_accum`, `denom`, `_sg_axis_count`), **and** the
Adam optimizer state (`exp_avg`/`exp_avg_sq`) — then `empty_cache()` +
`synchronize()` and replies:

```json
{"paused": true, "vram_mb": 320.5}
```

- The reply comes **only after** the move completes; `vram_mb` is the
  post-move `torch.cuda.memory_allocated()` (residual CUDA context only).
- **No disk writes** — the model stays in RAM (this is the whole point vs
  `/checkpoint`). ~1.5 GB at 800k splats, well inside the 16 GB budget.
- **Idempotent:** a second `/pause` while already paused is a fast no-op that
  returns the current `vram_mb` (the server re-pauses each job in a backlog).
- If the trainer can't ack within `http.pause_timeout` (default 30 s), the
  reply is `504 {"error": "pause timed out", "paused": false}`; the
  home-server proceeds with COLMAP regardless (pause degrades to a no-op).

The residual CUDA context (~0.3–0.6 GB) can't be freed without a process
exit; with the model on CPU, COLMAP gets ~5.5 GB of the 6 GB card.

## POST /resume

Empty JSON body. Called once the capture queue drains. Moves everything back
to the GPU (`to_cuda()`) and rebinds so training continues from the exact
paused state, then replies `200 {"paused": false}`. Idempotent.

**Adam binding invariant:** after `to_cuda()`, `optim_guard.optimizer_binding_ok()`
holds — the optimizer state stays keyed to the moved params (params are
mutated in place, never replaced), so momentum survives the round trip (the
T1 pattern). Device→device float transfer is bit-exact, so pause→resume is a
no-op on values.

## GET /health

```json
{"state": "improving", "iter": 1234, "splat_count": 240000,
 "queue_depth": 0, "last_ingest": 1784357180.28, "vram_mb": 1450.0,
 "session_id": "30d37bfd", "n_images": 13, "bootstrap_complete": true,
 "paused": false}
```

`state` is the convergence-monitor state
(`initializing | improving | wants_capacity | stalled | converged`).
`vram_mb` is `null` without CUDA. `paused` reflects the /pause–/resume state
so the server/overlays can show which side owns the GPU. The home-server
polls this (cached 3 s) and surfaces it in its session status.

## Splat export

When started with `--export_dir` (or `$TRAINER_EXPORT_DIR`), every snapshot
write also atomically mirrors to `<export_dir>/<session_id>/latest.ply` —
the path the home-server's `latest_splat_ply()` serves to the viewer.

**Early dense snapshot:** on each ingest, right after dense-init seeds the new
camera and *before* any training iterations, the trainer writes a snapshot.
So `latest.ply` reflects the newly added camera within a second of the
`/ingest` (viewer step 4), not only at convergence.

## Bootstrap & dense init

- `bootstrap.min_images` is **3**, kept equal to the home-server's mapper gate
  (`SFM_MIN_IMAGES_TO_MAP`, default 3). The trainer idles `initializing` until
  `/ingest` brings the registered-image count to this.
- `dense_init.backend` is **da3** (Depth Anything V3), pose-conditioned: the
  COLMAP extrinsics+intrinsics of each new camera are fed to DA3 as
  conditioning. Per-image DA3 failures fall back to DAv2
  (`da3_fallback_to_dav2`, models load sequentially — never co-resident).

## Match matrix (optional, removes the L8 uniform-weight fallback)

The home-server's `export_match_matrix` writes into the promoted
`sparse/0/`:

- `imagesNames.txt` — one image name per line; line i = matrix row i
  (DB image_id order, NOT registration or alphabetical order — L7).
- `imageMatchMatrix.txt` — N rows of N space-separated ints; entry (i, j) =
  verified inlier count between images i and j; symmetric; diagonal 0.

`scene/match_matrix.py` auto-detects this format alongside the legacy
GS_On-The-Fly comma format and reindexes rows to the trainer's camera order
by name stem. When the files are absent the trainer logs one WARNING and
falls back to uniform weights with new-camera bias.

## Ports

Default trainer bind is `127.0.0.1:8666`, matching the home-server's
`TRAINER_URL` default.
