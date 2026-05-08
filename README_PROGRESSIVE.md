# Progressive MEGS² — Incremental Gaussian Splatting with Dense Depth Init

This extends [MEGS²](https://github.com/IGL-HKUST/MEGS-2) with:

1. **Progressive snapshot ingestion** — train as new images arrive from an upstream SfM pipeline (e.g. GLOMAP).
2. **Match-graph weighted camera selection** — bias training iterations toward newly registered images and their closest neighbors.
3. **Dense initialization via Depth Anything v2** — pre-seed Gaussians from aligned monocular depth before each merge window, dramatically reducing the iteration budget needed to absorb new views.

All work is in Python.  MEGS-2's CUDA rasterizer, `SphericalGaussianModel`, and pruning machinery are unchanged.

---

## Requirements

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install transformers pyyaml scipy
# Build MEGS-2 CUDA submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

---

## Snapshot folder layout

Each numbered subdirectory is a complete COLMAP reconstruction at that point in the incremental SfM.  Folder N is a strict superset of folder N-1.

```
<source_path_dir>/
  16/
    images/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
      imageMatchMatrix.txt    ← co-visibility counts (optional; enables Phase 6 weighting)
      imagesNames.txt         ← comma-sep filenames in SfM-registration order (optional)
  17/
    images/     ← all images from 16/ plus new ones
    sparse/0/
      ...
  18/
    ...
```

`imageMatchMatrix.txt` format: one line per registered image, comma-separated integer feature-match counts.  
`imagesNames.txt` format: single line of comma-separated filenames matching matrix row order.

> **Note**: If your GLOMAP adapter does not write these files, match-graph weighting is skipped silently and new cameras get weight 1.0 with existing cameras at 0.5.

---

## Running progressive training

```bash
python progressive_train.py \
    --source_path_dir /path/to/snapshots \
    --model_path /path/to/output \
    --config configs/progressive.yaml \
    --imp_metric outdoor
```

To disable dense init at runtime (overrides YAML):
```bash
python progressive_train.py ... --dense_init_disabled
```

To enable dense init at runtime:
```bash
python progressive_train.py ... --dense_init_enabled
```

---

## YAML config reference (`configs/progressive.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `training.iter_initial` | 3000 | Iterations on snapshot 0 |
| `training.iter_per_merge` | 200 | Iterations **per new image** in a merge phase |
| `training.iter_final` | 4000 | Final refinement iterations |
| `training.num_max` | 800 000 | Gaussian count ceiling (6 GB GPU) |
| `training.prune_every_n_snapshots` | 5 | Lightweight prune interval during merge |
| `training.imp_metric` | `outdoor` | `outdoor` or `indoor` (importance scoring) |
| `dense_init.enabled` | true | Enable Depth Anything v2 dense seeding |
| `dense_init.skip_first_n_snapshots` | 3 | Skip dense init for first N snapshots |
| `dense_init.target_dense_points_per_image` | 50 000 | Points seeded per new image |
| `dense_init.grace_iters` | 200 | Protect dense Gaussians from pruning this long |
| `dense_init.dav2.model_size` | `small` | `small` \| `base` \| `large` |
| `dense_init.dav2.fp16` | true | Half-precision inference (~half VRAM) |
| `dense_init.ransac.min_inliers` | 8 | Minimum SfM-depth inliers for alignment |

---

## Output structure

```
<model_path>/
  point_cloud/
    iteration_0/point_cloud.ply      ← after initial training
    iteration_17/point_cloud.ply     ← after snapshot 17 merge
    iteration_final/point_cloud.ply  ← after final refinement
  progressive_diary.txt              ← per-iteration camera name + weight log
  run_config.yaml                    ← reproducibility record
```

---

## WebGL viewer

The per-snapshot `.ply` files are written by MEGS-2's `SphericalGaussianModel.save_ply` unchanged, so they are directly compatible with the existing viewer.

```bash
# 1. Copy the checkpoint
cp <model_path>/point_cloud/iteration_final/point_cloud.ply WebGL_viewer/

# 2. In WebGL_viewer/index.html, make sure:
#    <script src="main_sg.js"></script>   ← NOT main_sh.js

# 3. In main_sg.js, set the url to "point_cloud.ply"

# 4. Serve and open in Chrome
cd WebGL_viewer
npx http-server -p 8080
# → open http://localhost:8080
```

---

## Running tests

```bash
# Pure-Python tests (no GPU needed)
pytest tests/test_match_matrix.py tests/test_dense_init.py -v

# GPU tests (require CUDA)
pytest tests/test_expand_from_pcd.py -v

# Progressive scene integration (requires Pillow)
pytest tests/test_progressive_scene.py -v
```

---

## Dense init A/B comparison

```bash
# Without dense init
python progressive_train.py \
    --source_path_dir /data/snapshots \
    --model_path /out/no_dense \
    --config configs/progressive.yaml \
    --dense_init_disabled

# With dense init
python progressive_train.py \
    --source_path_dir /data/snapshots \
    --model_path /out/with_dense \
    --config configs/progressive.yaml \
    --dense_init_enabled
```

Compare PSNR-vs-iteration curves by inspecting the `progressive_diary.txt` logs alongside PSNR checkpoints.  Dense init should reach the no-dense-init final PSNR in ≥30% fewer merge-phase iterations on the demo data.

---

## VRAM budget (6 GB GPU)

| Phase | Expected peak |
|-------|--------------|
| Initial training | ~2–4 GB |
| Merge training | ~2–4 GB |
| DAv2 inference (small, fp16) | +1–2 GB (then freed) |
| Final refinement | ~3–5 GB |

Total peak is kept under 5.5 GB during training and under 6 GB during DAv2 inference by the strict context-manager discipline in `dense_init_for_new_images`.

---

## Known limitations / future work

See `NOTES.md` → *Future Work* section for the full list, including:
- Adaptive LR (`On_The_Fly_Update_Lr`)
- Async DAv2 inference
- Depth-supervised training loss
- MVS-based final refinement
