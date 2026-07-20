# Dense-init backends: DAv2 vs DA3

`dense_init.backend` in `configs/continuous.yaml` selects the monocular
depth model used to seed Gaussians for newly ingested images:

- `dav2` (default) — Depth Anything V2 via transformers. Predicts relative
  *disparity*; the RANSAC alignment fits `aligned = a*raw + b`, so `a < 0`
  is the normal fit (DL2).
- `da3` — Depth Anything 3 (`DepthAnything3Wrapper`). Predicts *depth*
  directly, so the fitted `a` must be positive; `validate_alignment`
  rejects images with `a <= 0` and (with `da3_fallback_to_dav2: true`)
  retries just those images through a DAv2 context. The two models are
  never co-resident on GPU — contexts run sequentially, same VRAM
  discipline as always (<100 MB residue warning on exit).

DA3 extras:

- **Pose conditioning** (`da3.conditioning: true`): each predict passes the
  camera's COLMAP world-to-camera extrinsics (4×4) and pinhole intrinsics
  (3×3) to `DepthAnything3.inference`, both in OpenCV/COLMAP convention.
- **Models** (`da3.model_name`): start with `depth-anything/DA3-SMALL` on
  the 6 GB target; `DA3-BASE` next. `DA3MONO-LARGE`/`DA3METRIC-LARGE`
  exist only as LARGE — check headroom before trying.
- **Install** (only needed for `backend: da3`):
  `pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3`
  (plus `xformers`). Not added to requirements.txt while dav2 is default.

## Choosing the default

Run the benchmark on the target machine:

```
python tools/bench_dense_init.py --session-dir sessions/<id> \
    --backends dav2,da3 --out bench_dense_init.md
```

It reports per-image aligned-depth RMSE vs SfM sparse depths, RANSAC
inliers, predict latency, model-load time, and peak VRAM. For the
downstream iterations-to-quality comparison, run the regression harness
twice with the backend toggled. Flip the yaml default only if DA3 wins on
quality at acceptable VRAM, and commit the results table alongside.

## Deferred (sketched in DepthAnything3Wrapper's docstring, do not build yet)

- `infer_gs=True` feed-forward 3DGS instant-preview path;
- DA3 pose estimation as an SfM-bootstrap-failure fallback;
- DA3-Streaming for continuous capture.
