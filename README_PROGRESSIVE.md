# Progressive MEGS² — Incremental Gaussian Splatting with Dense Depth Init

This extends [MEGS²](https://github.com/IGL-HKUST/MEGS-2) with:

1. **Progressive snapshot ingestion** — train as new images arrive from an upstream SfM pipeline (e.g. GLOMAP).
2. **Match-graph weighted camera selection** — bias training iterations toward newly registered images and their closest neighbours.
3. **Dense initialization via Depth Anything v2** — pre-seed Gaussians from aligned monocular depth before each merge window, dramatically reducing the iteration budget needed to absorb new views.
4. **Convergence-driven scheduling** — all iteration caps are safety belts, not targets; evidence-based triggers (gradient density, loss slope, splat growth) drive densify/prune decisions.
5. **SkipGS view-adaptive backward gating** — during the final phase, per-view EMA baselines gate which cameras contribute to each backward pass, reducing compute by ~37% with negligible quality loss.

All work is in Python. MEGS-2's CUDA rasterizer, `SphericalGaussianModel`, and pruning machinery are unchanged.

---

## Requirements

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install transformers pyyaml scipy
# Build MEGS-2 CUDA submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

---

## Training phases

The pipeline has three phase types. Understanding them clarifies why they are separate and what each feature targets.

### 1 — Initial phase (`iter_initial` cap, convergence-driven)

Runs **once**, on the first snapshot after dense init.

**Purpose**: establish a baseline 3D scene before progressive ingestion begins. Densification, opacity prune, and importance prune are all active and convergence-gated. Splat count is held under `num_max_ceiling` via random-subsampling at dense-init time and importance-prune during stall.

**Why SkipGS is off here**: SkipGS needs per-view EMA loss baselines accumulated over many iterations to distinguish well-fit cameras from under-fit ones. In the initial phase the scene is still forming; baselines would be unreliable and the budget floor (`rho_lo`) would force almost all backwards anyway — net zero benefit with added overhead.

**Why SG axis culling is off here**: Sharpness differentiation across SG axes develops gradually as the scene trains. Culling low-sharpness axes before the scene has converged removes capacity that may still be useful.

### 2 — Merge phase (`iter_per_merge` cap, convergence-driven)

Runs **once per new snapshot**. Dense init seeds Gaussians for new cameras first, then this short training window absorbs them.

**Purpose**: integrate new cameras and SfM points without over-fitting to them. The cap is intentionally short; the final phase does the deep refinement. Match-graph weighting biases gradient contribution toward newly registered cameras.

### 3 — Final phase (`iter_final` cap, convergence-driven)

Runs **once**, after all snapshots have been ingested.

**Purpose**: comprehensive refinement of the complete scene. All cameras are registered; the scene is stable enough for:

- **SkipGS** — per-view EMA baselines are reliable; backward is skipped for views whose loss is already at or below baseline. Budget floor (`rho_lo`) prevents over-skipping.
- **SG axis culling** — fires once at 85%+ through the window; removes SG axes whose sharpness has never risen above threshold, freeing capacity without quality loss.
- **Aggressive importance pruning** — stall-triggered lightweight prune fires repeatedly, driving an oscillating prune-densify loop that keeps splat count bounded while quality improves.

After the final phase exits, `fast_final_compression` runs a single importance prune + axis cull pass to compact the model before saving.

### Why the separation matters

| Feature | Initial | Merge | Final |
|---------|---------|-------|-------|
| Dense init | ✓ | ✓ | — |
| Densification (evidence-gated) | ✓ | ✓ | ✓ |
| Opacity/size prune (every iter) | ✓ | ✓ | ✓ |
| Importance prune (stall-gated) | ✓ | ✓ | ✓ |
| SkipGS backward gating | — | — | ✓ |
| SG axis cull | — | — | ✓ (once at 85%+) |
| `fast_final_compression` | — | — | after exit |

---

## Snapshot folder layout

Each numbered subdirectory is a complete COLMAP reconstruction at that point in the incremental SfM. Folder N is a strict superset of folder N-1.

```
<source_path_dir>/
  1/
    images/
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
      imageMatchMatrix.txt    ← co-visibility counts (optional; enables Phase 6 weighting)
      imagesNames.txt         ← comma-sep filenames in SfM-registration order (optional)
  2/
    images/     ← all images from 1/ plus new ones
    sparse/0/
      ...
  3/
    ...
```

`imageMatchMatrix.txt` format: one line per registered image, comma-separated integer feature-match counts.
`imagesNames.txt` format: single line of comma-separated filenames matching matrix row order.

> **Note**: If your GLOMAP adapter does not write these files, match-graph weighting is skipped silently and new cameras get weight 1.0 with existing cameras at 0.5.

---

## Simulating progressive ingestion for performance testing

If you have a complete COLMAP reconstruction but want to test the progressive pipeline, `tools/simulate_progressive.py` slices it into the numbered snapshot format automatically.

```bash
# Start with 3 images, add 1 per snapshot
python tools/simulate_progressive.py \
    --source /path/to/colmap_dataset \
    --output /path/to/output_snapshots \
    --n-init 3 \
    --step 1

# Start with 5 images, add 2 per snapshot, stop after 10 snapshots
python tools/simulate_progressive.py \
    --source /path/to/colmap_dataset \
    --output /path/to/output_snapshots \
    --n-init 5 \
    --step 2 \
    --max-snapshots 10
```

`--source` can be the COLMAP dataset root (containing `sparse/0/`) or the `sparse/0/` directory directly. Images are ordered by filename natural sort by default (`--order filename`); use `--order image_id` to follow COLMAP registration order instead.

Each output snapshot contains only the cameras, images, and 3D points visible at that stage. Tracks in `points3D.bin` are filtered to only include observations from images present in that snapshot. A symlink `images/ → <source>/images/` is created so MEGS-2 can load image data without copying.

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

### training

| Key | Default | Description |
|-----|---------|-------------|
| `iter_initial` | 3000 | Iteration cap for the initial phase (convergence exits earlier) |
| `iter_per_merge` | 200 | Iteration cap per merge phase |
| `iter_final` | 4000 | Iteration cap for the final phase |
| `num_max_ceiling` | 800 000 | Hard Gaussian count ceiling (VRAM-bounded) |
| `prune_ratio1` | 0.05 | Fraction pruned per lightweight importance prune pass |
| `prune_ratio2` | 0.05 | Fraction pruned in fast_final_compression |
| `sharpness_threshold` | 1.0 | SG axis cull threshold |
| `imp_metric` | `outdoor` | `outdoor` or `indoor` (importance scoring) |
| `imp_score_camera_subsample` | 0 | Camera subsample in importance scoring (0 = all) |
| `accumulation_views` | 2 | Views accumulated per optimizer step (T6 multi-view grad) |
| `densify_min_obs` | 10 | Min gradient observations before densify fires; also sets post-densify settling window |
| `densify_candidate_fraction` | 0.005 | Fraction of splats needing grad signal to trigger densify |
| `fast_prune_dead_fraction` | 0.02 | Dead-splat fraction triggering opacity/size prune |
| `lightweight_prune_growth_threshold` | 0.10 | Growth fraction (since last prune) to trigger stall-prune |
| `sg_axis_cull_low_fraction` | 0.20 | Fraction of low-sharpness axes required to trigger SG cull |
| `fast_final_prune_ratio` | 0.10 | Fraction pruned by fast_final_compression |

### training.skipgs

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | true | Enable SkipGS during the final phase |
| `phase` | `final` | Phase in which SkipGS activates |
| `warmup` | 500 | Iterations before gating starts; used to calibrate `rho_min` |
| `beta` | 0.95 | EMA decay for per-view loss baseline |
| `eps` | 1e-8 | Numerical floor in deviation ratio |
| `rho_lo` | 0.5 | Lower bound on backward budget ratio |

### dense_init

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | true | Enable Depth Anything v2 dense seeding |
| `skip_first_n_snapshots` | 0 | Skip dense init for first N snapshots |
| `target_dense_points_per_image` | 100 000 | Points seeded per new image (capped at `num_max_ceiling`) |
| `novelty_distance_threshold` | 0.01 | Min distance from existing splats (in scene-scale units) |
| `grace_iters` | 200 | Protect dense Gaussians from importance-prune this long |
| `persist_model` | false | Hold DAv2 in VRAM across snapshots (saves ~30 s/snap, costs ~670 MB) |
| `dav2.model_size` | `large` | `small` \| `base` \| `large` |
| `dav2.fp16` | false | Half-precision inference (not reliable on all hardware) |
| `ransac.min_inliers` | 8 | Minimum SfM-depth inliers for depth alignment |

---

## Output structure

```
<model_path>/
  point_cloud/
    iteration_0/point_cloud.ply      ← after initial phase
    iteration_N/point_cloud.ply      ← after snapshot N merge
    iteration_final/point_cloud.ply  ← after fast_final_compression
  progressive_diary.txt              ← per-iteration camera name + weight log
  run_config.yaml                    ← reproducibility record
```

---

## WebGL viewer

The per-snapshot `.ply` files are compatible with the existing MEGS-2 WebGL viewer.

```bash
cp <model_path>/point_cloud/iteration_final/point_cloud.ply WebGL_viewer/
cd WebGL_viewer
npx http-server -p 8080
# → open http://localhost:8080
# Ensure main_sg.js is loaded, not main_sh.js
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

## VRAM budget (6 GB GPU, GTX 1660 Ti)

| Phase | Expected peak |
|-------|--------------|
| Initial / merge / final training | ~2–4 GB |
| DAv2-Large inference (float32) | ~2 GB (freed after each snapshot) |
| DAv2-Large persistent (T4) | ~670 MB held across snapshots |

Total peak is held under 5.5 GB during training and ~4.5 GB during DAv2 inference by the context-manager discipline in `dense_init_for_new_images` and the `num_max_ceiling` cap on dense init output.

---

## Known limitations / future work

- Async DAv2 inference (overlap depth estimation with previous snapshot training)
- Depth-supervised training loss
- Adaptive learning rate schedule (On_The_Fly_Update_Lr)
- MVS-based final refinement
- `--order timestamp` using EXIF data in `simulate_progressive.py`
