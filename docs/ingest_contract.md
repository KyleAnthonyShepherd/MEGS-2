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

## GET /health

```json
{"state": "improving", "iter": 1234, "splat_count": 240000,
 "queue_depth": 0, "last_ingest": 1784357180.28, "vram_mb": 1450.0,
 "session_id": "30d37bfd", "n_images": 13, "bootstrap_complete": true}
```

`state` is the convergence-monitor state
(`initializing | improving | wants_capacity | stalled | converged`).
`vram_mb` is `null` without CUDA. The home-server polls this (cached 3 s)
and surfaces it in its session status.

## Splat export

When started with `--export_dir` (or `$TRAINER_EXPORT_DIR`), every snapshot
write also atomically mirrors to `<export_dir>/<session_id>/latest.ply` —
the path the home-server's `latest_splat_ply()` serves to the viewer.

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
