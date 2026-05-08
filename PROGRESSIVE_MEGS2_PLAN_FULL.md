# Implementation Plan: Progressive MEGS² for Incremental SfM with Dense Depth Initialization

**Hand-off document for a Claude Code instance.** You are a senior PyTorch/CUDA engineer. You will graft GS_On-The-Fly's progressive training scheduler on top of MEGS² (Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning) as the base, and add an optional dense-initialization step that uses Depth Anything v2 to seed Gaussians in genuinely new regions of the scene. The result must train Gaussian Splatting scenes incrementally as new images arrive from an upstream Structure-from-Motion pipeline, fit in 6 GB of GPU VRAM, and produce checkpoints viewable in MEGS²'s WebGL viewer.

---

## 1. Mission and success criteria

You are merging two research codebases and adding a third capability on top. Read both READMEs and at least the two main training scripts before writing any code.

- **Base repo (keep its rasterizer, Gaussian model, pruning):** https://github.com/IGL-HKUST/MEGS-2
- **Donor repo (port progressive scheduling FROM here):** https://github.com/xywjohn/GS_On-The-Fly
- **Dense init dependency:** Depth Anything v2, via Hugging Face `transformers`: `depth-anything/Depth-Anything-V2-Small-hf`

Build a new training entry point, `progressive_train.py`, that lives inside a fork of MEGS-2 and reuses MEGS-2's `SphericalGaussianModel`, `spherical_gaussian_renderer`, and `OptimizingSpa`/`OptimizingSpaSG` pruning machinery unchanged. Layer GS_On-The-Fly's snapshot-driven progressive training on top of it. Add an optional dense-initialization step driven by a YAML config file.

**Done means all of these are true:**

1. `progressive_train.py` runs against the GS_On-The-Fly demo data (`demo_data/data1` + `On-The-Fly/data1`) on a 6 GB GPU without OOM.
2. After each snapshot ingestion, a `.ply` checkpoint is written that the MEGS-2 `WebGL_viewer` can load and render correctly (use `main_sg.js`, since we are using Spherical Gaussians).
3. PSNR on the final scene is within 2 dB of MEGS-2's offline `train.py` baseline run on the same final dataset (i.e., we don't pay an enormous quality tax for going progressive).
4. Peak GPU memory during training stays under 5.5 GB on a representative 30-image scene. Peak GPU memory during dense-init inference stays under 6 GB.
5. The match-graph weighting from GS_On-The-Fly is *actually used* — verifiable from training logs (per-camera weight values printed) and from a quality test where ablating the weighting noticeably reduces PSNR on the most recently added image.
6. Dense initialization reaches the no-dense-init final PSNR in ≥30% fewer merge-phase iterations on the demo data when enabled. When disabled, the pipeline runs as if dense init didn't exist — no overhead, no behavior change.
7. DAv2 and GS training are NEVER co-resident on GPU. Verified by VRAM instrumentation showing post-DAv2-context VRAM returns to within 100 MB of pre-context state.

**Anti-goals (do not do these):**

- Do not port GS_On-The-Fly's rasterizer (`gaussian_renderer/`) or its `load_distribution`/`load_loss` machinery. Those depend on a customized `diff-gaussian-rasterization` that returns load stats; MEGS-2's `spherical_gaussian_renderer` does not, and reproducing it is out of scope.
- Do not port GS_On-The-Fly's `On_The_Fly_Update_Lr` adaptive LR scheduler in the first pass. Use a simpler bounded LR per snapshot (described in Phase 7). You may revisit this in a second pass if quality requires it.
- Do not modify MEGS-2's CUDA submodules (`spherical_gaussian_renderer`, `spherical_gaussian_renderer_light`, `submodules/diff-gaussian-rasterization*`). All work is in Python.
- Do not add depth-supervised losses or other dense-depth uses in v1. Depth Anything is an *initialization* mechanism only. Use the depth maps to seed Gaussians, then discard them. No supervision during training.
- Do not implement async DAv2 inference, importance-sampled depth subsampling, or MVS-based refinement. All deferred to future work.

---

## 2. Repository setup

```bash
mkdir progressive-megs2 && cd progressive-megs2
git clone --recursive https://github.com/IGL-HKUST/MEGS-2.git
git clone --recursive https://github.com/xywjohn/GS_On-The-Fly.git
cd MEGS-2
git checkout -b progressive
```

You will work in the `MEGS-2/` directory. Treat `GS_On-The-Fly/` as **read-only reference material** — port concepts and small code blocks from it, but never depend on it at runtime.

---

## 3. Phase 0 — Environment, baseline validation

**Goal:** Both repos run their stock examples on the target 6 GB GPU before you touch anything.

### 3.1 Environment

MEGS-2's official env is Python 3.7 + PyTorch 1.12.1+cu116. This is too old for many useful tools and conflicts with what GS_On-The-Fly assumes. **Bump it.** Target: Python 3.10, PyTorch 2.1.x with CUDA 11.8 or 12.1 matching the host's `nvidia-smi` driver.

```bash
conda create -n pmegs2 python=3.10 -y
conda activate pmegs2
# Pin PyTorch to whatever cu118/cu121 matches the host driver:
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r MEGS-2/requirements.txt
pip install transformers pyyaml  # for DAv2 + config parsing in Phase 5
```

You will likely have to:

- Edit `MEGS-2/submodules/*/setup.py` if any pin `torch<2`. Replace with `torch>=2.1`.
- Rebuild CUDA submodules: `pip install MEGS-2/submodules/spherical_gaussian_renderer` and similarly for any others MEGS-2 ships. Verify each builds against the new PyTorch.
- Install `fused_ssim` (used by both repos): `pip install git+https://github.com/rahul-goel/fused-ssim`.
- Install `pytorch3d` only if a feature actually needs it. GS_On-The-Fly imports it but in practice MEGS-2 does not. Skip until proven needed.

### 3.2 Baseline runs (mandatory before proceeding)

1. Download MEGS-2's expected datasets per its README. Use the smallest one (e.g., `truck` from Tanks & Temples).
2. Run `bash MEGS-2/eval.sh` or invoke `MEGS-2/train.py` directly on the small scene with `--imp_metric outdoor` (or `indoor` as appropriate). Confirm:
   - It runs to completion.
   - Peak VRAM stays under 6 GB. If it does not, lower `--num_max` until it does, and record that ceiling.
   - Final PSNR matches the paper's ballpark (>23 dB on truck).
3. Download GS_On-The-Fly's demo data per its README. Run `DatasetPrepare.py` and `ContinuosProgressiveTrain.py` against `data1`. Goal here is just to *see* the snapshot folder structure that gets produced — you will mirror this format for your input data later. **Do not** waste effort getting their training to converge; the diff-gaussian-rasterization submodule may not even build under PyTorch 2.x. If it doesn't build, that's fine — read their code as documentation.
4. Verify DAv2 loads and runs: `python -c "from transformers import AutoModelForDepthEstimation; m = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf').cuda(); print(m)"`. Note the VRAM cost — should be ~1–2 GB. If it doesn't load, that's a blocker for Phase 5.

**Phase 0 acceptance:** MEGS-2 stock training works, fits in 6 GB on the target hardware, and you have a documented `--num_max` ceiling for the test scene. You have looked at GS_On-The-Fly's snapshot output structure on disk. DAv2-small loads on the target GPU.

---

## 4. Phase 1 — Code reconnaissance

**Goal:** Build a mental model of all three components before writing anything. Write a short `NOTES.md` in your fork capturing what you find. This will save you from rediscovering things later.

Read these files and answer these questions in your notes:

### 4.1 `MEGS-2/train.py` (≈420 lines)

- What is the iteration phase structure? (Confirm: pre-`simp_iteration1` densification, `simp_iteration1` first prune by `prune_ratio1`, `optimizing_spa_start_iter` to `_stop_iter` Lagrangian pruning, `optimizing_spa_stop_iter` final prune by `prune_ratio2`, `optimizing_spa_sg_stop_iter` SG axis culling.)
- What does the `reinitial_pts` call at every 5000th iteration do? (It re-seeds Gaussians from rendered depth.) Does it destroy or preserve Gaussian count? Will progressive training tolerate it?
- What does `update_imp_score` need from the scene? (Answer: it iterates *all* training cameras every time it's called. This is expensive and has implications for our progressive loop — see Phase 7.)

### 4.2 `MEGS-2/scene/spherical_gaussian_model.py`

- Identify every public method. The key ones we will call:
  - `__init__(sg_degree)`, `training_setup(opt)`, `capture()`, `restore(...)`
  - `oneupSGdegree()`, `update_learning_rate(iter)`
  - `densify_and_prune_split(...)`, `prune_points(mask)`, `add_densification_stats(...)`
  - `reinitial_pts(xyz, rgb)`, `cull_low_sharpness_axes(threshold)`
  - `get_xyz`, `get_opacity`, `get_sg_axis_count`, `get_sg_directions`, `get_sg_rgb`, `get_sg_sharpness`, `_xyz`, `_rgb_base`, `max_radii2D`
- **Identify what GS_On-The-Fly's `expand_from_pcd(pcd, cameras_extent, mask)` does** (in `GS_On-The-Fly/scene/gaussian_model.py`). You will need to write the SG-flavored equivalent. The contract: take a `BasicPointCloud`, an extent scalar, and a boolean mask selecting which points are *new*; append fresh Gaussians for those points to the model and extend the optimizer parameter groups. Document the existing densification code path that handles parameter-group extension; reuse that machinery rather than re-deriving it.

### 4.3 `MEGS-2/scene/__init__.py` (the `Scene` class)

- How does `Scene` discover cameras? It reads from a COLMAP-format `sparse/0/` directory. Confirm the data reader path in `scene/dataset_readers.py`.
- How does `Scene.save(iteration)` write the .ply? Note the path so we can re-use it from `progressive_train.py`.
- Does the COLMAP loader expose `points3D.bin`'s `image_ids` track info per 3D point? You'll need this in Phase 5 to determine which SfM points are observed in which images (more accurate than geometric projection, which doesn't know about occlusion).

### 4.4 `GS_On-The-Fly/ContinuosProgressiveTrain.py` (≈965 lines, read it carefully)

- The key methods, in execution order:
  1. `Gaussian_On_The_Fly_Splatting.__init__` → `Initialize_Gaussians` → `Train_Gaussians("Initialization")`
  2. `On_The_Fly_Train_Gaussians`: outer loop over snapshot indices, calls per-iteration:
     - `ExpandingGS_From_SparsePCD()` — uses `cKDTree` to find new points, calls `gaussians.expand_from_pcd(...)`. **Port this verbatim conceptually but adapted to SG model.**
     - `GetImageMatchingMatrix()` — parses `imageMatchMatrix.txt` and `imagesNames.txt`, log-normalizes by max per row, reorders rows/cols to match camera registration order. **Port verbatim, it's pure Python.**
     - `GetImagesWeightsFromMatrix()` — computes per-image weights: 1.0 for newly added, weighted average of match scores for existing. Iterates up to 4 times to fill in zero-weight images. **Port verbatim.**
     - `ImagesAlreadyBeTrainedIterations_Set()` — equivalent training count for new cameras. Optional, skip in v1.
     - `Train_Gaussians("On_The_Fly")` — the per-snapshot training loop with custom camera selection (`GetTraining_Viewpoints` picks top-10 weighted + 10 random) and custom LR (`On_The_Fly_Update_Lr`). **Port the camera selection; skip the LR scheduler in v1.**
  3. `Final_Refinement()` — full-scene optimization at the end.

- The snapshot folder format produced by `DatasetPrepare.py`:
  ```
  Source_Path_Dir/
    16/
      images/
      sparse/0/{cameras.bin,images.bin,points3D.bin,imageMatchMatrix.txt,imagesNames.txt}
    17/
      ...
  ```
  Each integer-named directory is a **complete** COLMAP scene as known at that point. Folder N is a strict superset of folder N-1 (same images plus new ones, plus updated poses).

- `imageMatchMatrix.txt` format: one row per registered image, comma-separated integers (raw co-visible-feature counts), trailing newline. The matrix is log-normalized as `log(x+1)/log(max_in_row+1)` after loading.
- `imagesNames.txt` format: a single line of comma-separated image filenames, in the order matching the matrix rows.

### 4.5 Depth Anything v2

- Read the DAv2 model card on Hugging Face: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Confirm the output convention: is it depth or inverse depth? Run on a known image (a near-and-far scene) and inspect raw output values at near vs. far pixels. **Document this in NOTES.md.** This is Landmine #L4 below — getting this wrong silently corrupts everything downstream.
- Confirm the input preprocessing: DAv2 expects images normalized with specific mean/std and resized so dimensions are multiples of 14. The `transformers` `AutoImageProcessor` handles this. If you bypass it, replicate exactly.

**Phase 1 acceptance:** `NOTES.md` exists in your fork and answers all the questions above. You have read `expand_from_pcd`, `GetImageMatchingMatrix`, `GetImagesWeightsFromMatrix`, `GetTraining_Viewpoints`, and `ExpandingGS_From_SparsePCD` from GS_On-The-Fly start to finish. You have verified DAv2's depth-vs-inverse-depth convention by inspection.

---

## 5. Phase 2 — Snapshot loader for MEGS²

**Goal:** Add the ability to read a numbered-folder snapshot stream into MEGS-2's `Scene` infrastructure.

### 5.1 New file: `scene/progressive_scene.py`

Create a `ProgressiveScene` class that wraps but does not subclass MEGS-2's `Scene`. Responsibilities:

- Discover snapshot folders: `sorted([int(d) for d in os.listdir(source_path_dir) if d.isdigit()])`. Store as `self.snapshot_indices`.
- Load snapshot N by pointing the underlying `Scene` at `<source_path_dir>/<N>/` and re-running its COLMAP loader. Reuse `sceneLoadTypeCallbacks["Colmap"]` from `scene/dataset_readers.py`.
- Maintain four pieces of state across snapshots:
  - `self.train_cameras: list[Camera]` — append-only across snapshots; existing entries get their `R`/`T` (and FoV if changed) updated when a snapshot revises a pose.
  - `self.last_basic_pcd: BasicPointCloud` — the previous snapshot's sparse cloud, kept so we can diff for new points.
  - `self.image_match_matrix: np.ndarray` — log-normalized, reordered to match `train_cameras` order. Only valid after the first progressive snapshot.
  - `self.point_track_info: dict[int, list[int]]` — for each SfM point ID, the list of camera colmap_ids that observe it. Parsed from `points3D.bin`. Used in Phase 5 to determine SfM visibility per camera.
- Expose two methods:
  - `load_next_snapshot() -> tuple[list[Camera], np.ndarray, list[int]]` returning (newly added cameras, point-mask of which sparse points are new, list of new-camera indices into `train_cameras`).
  - `get_sfm_points_visible_to(camera: Camera) -> tuple[np.ndarray, np.ndarray]` returning `(sfm_xyz, visible_mask)` where `sfm_xyz` is `(N, 3)` and `visible_mask` is a `(N,)` bool array — True when the point is in this camera's track (preferred) OR projects geometrically into the image with positive depth (fallback). Used by Phase 5.

### 5.2 New file: `scene/match_matrix.py`

Port `GetImageMatchingMatrix` and `GetImagesWeightsFromMatrix` from GS_On-The-Fly verbatim. Two pure functions:

```python
def parse_match_matrix(matrix_path: str, names_path: str, ordered_camera_names: list[str]) -> np.ndarray
def compute_image_weights(match_matrix: np.ndarray, new_image_indices: list[int]) -> np.ndarray
```

### 5.3 Acceptance test for Phase 2

Write `tests/test_progressive_scene.py`:

1. Use the GS_On-The-Fly demo data (or a synthetic mock with 3 snapshots of 5/7/10 cameras each).
2. Construct a `ProgressiveScene`, call `load_next_snapshot()` three times.
3. Assert: camera count grows monotonically, the second call returns 2 new cameras, the third returns 3, the match matrix has the right shape, weights for newly added images equal 1.0, and `get_sfm_points_visible_to(cam)` returns a boolean mask whose `sum() > 0` for every camera in the snapshot.

**Phase 2 acceptance:** Test passes. No GPU work yet.

---

## 6. Phase 3 — Progressive driver shell

**Goal:** A new top-level script `progressive_train.py` modeled on `MEGS-2/train.py`, restructured around snapshots. No match weighting yet — every camera weighted equally. No dense init. Get this end-to-end first; we will layer in cleverness in later phases.

### 6.1 New file: `progressive_train.py`

Structure:

```python
def progressive_training(dataset, opt, pipe, args, config):
    gaussians = SphericalGaussianModel(dataset.sg_degree)
    progressive_scene = ProgressiveScene(dataset, gaussians)

    # Load snapshot 0 = the initial scene.
    progressive_scene.load_next_snapshot()
    gaussians.training_setup(opt)

    # ---- Initial training phase ----
    train_window(
        gaussians, progressive_scene, opt, pipe, args,
        n_iters=config.training.iter_initial,
        phase="initial",
    )
    save_checkpoint(progressive_scene, gaussians, snapshot_idx=0)

    # ---- Progressive ingestion loop ----
    while progressive_scene.has_more_snapshots():
        new_cams, new_point_mask, new_cam_indices = progressive_scene.load_next_snapshot()
        expand_gaussians_from_new_points(gaussians, progressive_scene, new_point_mask, opt)

        # Phase 5: dense init goes here, conditional on config
        # (added when Phase 5 is implemented; left out in Phase 3 shell)

        train_window(
            gaussians, progressive_scene, opt, pipe, args,
            n_iters=config.training.iter_per_merge * len(new_cams),
            phase="merge",
            new_cam_indices=new_cam_indices,
        )
        if args.progressive_output:
            save_checkpoint(progressive_scene, gaussians, snapshot_idx=progressive_scene.current_idx)

    # ---- Final refinement phase ----
    train_window(
        gaussians, progressive_scene, opt, pipe, args,
        n_iters=config.training.iter_final,
        phase="final",
    )
    save_checkpoint(progressive_scene, gaussians, snapshot_idx="final")
```

`train_window` is the per-iteration loop, structurally identical to MEGS-2's existing one (the body of `training()` in `train.py`). Lift the body into a helper. **Important:** keep MEGS-2's full pre-existing scheduling logic intact — `simp_iteration1`, `optimizing_spa_*_iter`, depth reinitialization, SG axis culling — but make all those iteration thresholds *relative to a per-phase iteration counter*, not absolute. The simplest model: each phase has its own independent budget; the final-refinement phase is the one where MEGS²'s heavy pruning and SG culling actually fire (initial and merge phases stay in the lightweight pre-`simp_iteration1` regime).

### 6.2 Argument plumbing and config file

Two layers of configuration:

- **CLI args** (kept minimal): paths, output dir, `--config` pointing at a YAML file, debug flags.
- **YAML config file** (where all tunables live): training schedule, dense-init parameters, pruning parameters.

Use `pyyaml` and dataclass-based config objects (or `omegaconf` if you prefer). Plumb every value through to the call site that uses it; no hardcoded magic numbers anywhere in the new code.

Default config at `configs/progressive.yaml`:

```yaml
training:
  iter_initial: 3000
  iter_per_merge: 200          # per new image
  iter_final: 4000
  num_max: 800000              # Gaussian count ceiling for 6 GB GPU
  prune_every_n_snapshots: 5

  # Inherited from stock MEGS-2 (apply to final-refinement phase, scaled to iter_final)
  simp_iteration1_frac: 0.40           # → simp_iteration1 = iter_final * 0.40
  optimizing_spa_start_iter_frac: 0.50
  optimizing_spa_stop_iter_frac: 0.80
  optimizing_spa_sg_stop_iter_frac: 0.95
  prune_ratio1: 0.05
  prune_ratio2: 0.05
  sharpness_threshold: 1.0
  optimizing_spa_interval: 100

dense_init:
  enabled: true
  skip_first_n_snapshots: 3              # let SfM stabilize before trusting alignment

  min_sfm_points_for_alignment: 10       # below this, skip dense init for the image
  min_sfm_depth_range_fraction: 0.10     # require visible SfM points to span >=10% of scene scale (avoids planar-scene rank deficiency)
  target_dense_points_per_image: 50000   # subsample target after filtering

  novelty_distance_threshold: 0.01       # in units of cameras_extent; reject dense points within this distance of an existing Gaussian
  depth_disagreement_threshold: 0.10     # fraction of scene scale; per-pixel sanity vs nearest SfM point
  max_rejected_fraction: 0.5             # if more than this fraction of pixels fail sanity, skip dense init for the image entirely
  grace_iters: 200                       # protect dense Gaussians from pruning for this many iterations after insertion

  dav2:
    model_size: small                    # small | base | large
    input_resolution: 518                # DAv2 expects multiples of 14
    device: cuda
    fp16: true                           # ~half the VRAM, no observable quality cost

  ransac:
    iterations: 200
    inlier_threshold: 0.05               # in normalized depth units after candidate (a, b) applied
    min_inliers: 8

snapshots:
  source_path_dir: null                  # set via CLI, not config
  snapshot_skip: 1                       # process every Nth snapshot if upstream is too fast
  progressive_output: true               # save per-snapshot checkpoints
```

### 6.3 Acceptance test for Phase 3

Run `progressive_train.py` against the GS_On-The-Fly demo data with `dense_init.enabled=false`. With small iteration counts, it must complete without errors. PSNR will be poor — that is fine. We're checking plumbing.

**Phase 3 acceptance:** End-to-end run finishes, all snapshot checkpoints are written, no OOM. Quality is unchecked.

---

## 7. Phase 4 — Gaussian expansion from new sparse points

**Goal:** When a snapshot adds new sparse points, append fresh Gaussians for them to the existing `SphericalGaussianModel`.

### 7.1 The challenge

MEGS-2's `SphericalGaussianModel` has more parameter tensors than vanilla 3DGS (SG axes, sharpness, axis count, etc.). Existing densification code (`densify_and_prune_split` and friends) handles extending all of these correctly when cloning/splitting. **Reuse that machinery** — don't write parallel code that risks getting one tensor's shape wrong.

### 7.2 Implementation

In `scene/spherical_gaussian_model.py`, add a method `expand_from_pcd(pcd, mask, spatial_lr_scale)`:

```python
def expand_from_pcd(self, pcd: BasicPointCloud, mask: np.ndarray, spatial_lr_scale: float):
    """
    Append fresh Gaussians for the points in pcd selected by mask.
    Mask is a boolean np.ndarray over pcd.points (True = new, append).
    """
    new_xyz = torch.from_numpy(pcd.points[mask]).float().cuda()
    new_rgb = torch.from_numpy(pcd.colors[mask]).float().cuda()
    # Build the *complete* set of new tensors with SG attributes initialized
    # using the same conventions as create_from_pcd / densify_and_split:
    #   _xyz, _rgb_base (or _features_dc/rest), _scaling, _rotation, _opacity,
    #   _sg_axes, _sg_sharpness, _sg_axis_count, max_radii2D, etc.
    # Use the densification-postprocess helper (cat_tensors_to_optimizer or
    # whatever it's called in MEGS-2) to extend each optimizer param group
    # AND each model tensor in lockstep.
    ...
```

Also add `mark_recently_added` for use by Phase 5 dense init:

```python
def mark_recently_added(self, idx_range: slice, iteration: int, grace_iters: int):
    """Records that Gaussians at idx_range were added at `iteration` and should
    be protected from pruning until iteration + grace_iters. Used by Phase 7's
    pruning logic to give freshly-seeded dense Gaussians time to gradient-train
    before being judged on their importance score."""
    # Simplest implementation: maintain self._grace_records: list[(start, end, expires_iter)]
    # Provide a method get_grace_protected_mask(current_iter) -> bool mask over all Gaussians
    # that returns True for any Gaussian still inside its grace window.
    ...
```

**Read `create_from_pcd` and the densification append/cat helpers in `spherical_gaussian_model.py` first.** The pattern of how new tensors are sliced into the optimizer state must match exactly or Adam moments will be misaligned and training will diverge.

### 7.3 The diff

Mirror the cKDTree logic from `GS_On-The-Fly/ContinuosProgressiveTrain.py:ExpandingGS_From_SparsePCD`. In `progressive_train.py`:

```python
def expand_gaussians_from_new_points(gaussians, prog_scene, new_point_mask, opt,
                                     distance_threshold=1.0, distance_buffer=1.5):
    new_pcd = prog_scene.current_basic_pcd
    # The mask coming in says "in this snapshot, which points were not in last snapshot";
    # use cKDTree to confirm spatial novelty against existing Gaussians' xyz to avoid
    # double-seeding in regions already well-modeled.
    ...
    gaussians.expand_from_pcd(new_pcd, refined_mask, prog_scene.cameras_extent)
    gaussians.training_setup(opt)  # rebuild optimizer if append helper didn't
```

### 7.4 Acceptance test for Phase 4

Add a unit test: load snapshot 0, train 500 iterations, save Gaussian count `G0`. Load snapshot 1 (which adds 2 images and ~K new points), call `expand_gaussians_from_new_points`. Assert `G1 > G0` and `G1 - G0` is on the order of K (not 0, not 100×K). Train another 500 iterations and verify no NaN losses, no optimizer-state shape errors.

Add a unit test for `mark_recently_added`: call it for indices 100–200 with `iteration=0, grace_iters=200`. Call `get_grace_protected_mask(50)` and assert indices 100–199 are True. Call `get_grace_protected_mask(250)` and assert all entries are False.

**Phase 4 acceptance:** Snapshot ingestion measurably grows the Gaussian count and training proceeds without errors after expansion. Grace-period mechanism behaves correctly.

---

## 8. Phase 5 — Dense initialization via monocular depth

**Goal:** After Phase 4 has seeded sparse Gaussians from new SfM points, optionally seed dense Gaussians from Depth Anything v2 in regions not already covered. Driven by the `dense_init` section of the config file. Strictly load → infer → unload to keep DAv2 and GS training off-GPU simultaneously.

**Rationale:** GS_On-The-Fly seeds new Gaussians only at sparse SfM tie points (typically <1000 per image). With short merge-phase iteration budgets (hundreds, not tens of thousands), MEGS²'s densification doesn't have time to fill in textureless or sparsely-matched regions. Pre-seeding tens of thousands of Gaussians per image from depth-aligned points lets each merge phase converge in fewer iterations. The CVPR 2025 finding that dense init "provides little final-quality benefit" applies to long offline runs, not to short progressive merge windows — different regime, different tradeoff.

### 8.1 New file: `scene/dense_init.py`

Three classes/functions, in the order they're used:

#### `DepthAnythingV2Wrapper` (context manager)

```python
class DepthAnythingV2Wrapper:
    """Loads DAv2 on __enter__, frees on __exit__. Use ONLY inside a `with` block."""
    def __init__(self, cfg: DAv2Config):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def __enter__(self):
        from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        repo = {"small": "depth-anything/Depth-Anything-V2-Small-hf",
                "base":  "depth-anything/Depth-Anything-V2-Base-hf",
                "large": "depth-anything/Depth-Anything-V2-Large-hf"}[self.cfg.model_size]
        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            repo, torch_dtype=torch.float16 if self.cfg.fp16 else torch.float32
        ).to(self.cfg.device).eval()
        return self

    def __exit__(self, *args):
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def predict(self, image_chw: torch.Tensor) -> torch.Tensor:
        """image_chw: (3, H, W) in [0, 1]. Returns (H, W) depth-like map in DAv2's native units."""
        ...
```

**Critical:** The `model` reference must not leak outside the `with` block. No module-level globals, no instance attributes that live past `__exit__`, no closures capturing the model. Verify with the VRAM swap test in §8.5.

#### `align_depth_to_sfm`

Pure function. RANSAC-fits `(a, b)` such that `aligned_depth = a * predicted + b` matches true SfM depth at the projected SfM point locations.

```python
@dataclass
class AlignmentResult:
    a: float
    b: float
    inlier_indices: np.ndarray
    n_inliers: int

class AlignmentFailed(Exception): pass

def align_depth_to_sfm(
    depth_map: torch.Tensor,            # (H, W) DAv2 raw output
    camera: Camera,                     # MEGS-2 Camera (must give us K and world->cam transform)
    sfm_xyz_visible: np.ndarray,        # (N, 3) SfM points known to project into this image
    cfg: RansacConfig,
) -> AlignmentResult:
    """Returns (a, b, inliers). Raises AlignmentFailed if fewer than min_inliers found
    OR if depth-range constraint not satisfied."""
    # 1. Project sfm_xyz_visible into the camera, get (u, v, true_depth) per point
    # 2. Sample depth_map at (u, v) for each, get predicted_depth per point
    # 3. Reject any point with non-finite predicted_depth (off-image after rounding, etc.)
    # 4. RANSAC: for cfg.iterations rounds, sample 2 points, solve a*p1+b=t1 and a*p2+b=t2,
    #    count inliers (those with |a*pi + b - ti| < cfg.inlier_threshold * scene_scale)
    # 5. Refit (a, b) on all inliers via least-squares
    # 6. Raise AlignmentFailed if n_inliers < cfg.min_inliers
```

The model `aligned = a * predicted + b` works for both "predicted is depth" and "predicted is inverse depth" conventions — but you MUST verify which DAv2 returns (see Landmine #L4) before back-projection.

#### `depth_to_points`

```python
def depth_to_points(
    depth_map: torch.Tensor,            # (H, W) raw DAv2 output
    camera: Camera,
    a: float, b: float,                 # alignment from align_depth_to_sfm
    target_n_points: int,
    sanity_threshold: float,            # absolute depth units
    sfm_xyz_visible: np.ndarray,        # for sanity filter
    image_rgb: torch.Tensor,            # (3, H, W) for coloring
    max_rejected_fraction: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Returns (xyz, rgb) of shape (M, 3), where M ≤ target_n_points; or None if more than
    max_rejected_fraction of pixels failed the sanity check (caller skips dense init for this image).
    """
    # 1. aligned_depth = a * depth_map + b  (still on GPU)
    # 2. Project sfm_xyz_visible into image to get sparse (u, v, true_depth) constraint set
    # 3. For each pixel, find nearest-neighbor projected SfM point in (u, v) space (cKDTree on CPU is fine)
    # 4. If |aligned_depth[v, u] - sfm_depth_of_nearest| > sanity_threshold, mark rejected
    # 5. If rejected_fraction > max_rejected_fraction: return None
    # 6. Subsample surviving pixels uniformly to target_n_points (random.choice without replacement)
    # 7. Back-project surviving (u, v, depth) to camera frame using K^-1, then to world frame using cam pose
    # 8. Read RGB at surviving (u, v) from image_rgb
    # 9. Return (xyz, rgb)
```

### 8.2 Integration into `progressive_train.py`

Replace the placeholder comment from Phase 3 with the actual call:

```python
while progressive_scene.has_more_snapshots():
    new_cams, new_point_mask, new_cam_indices = progressive_scene.load_next_snapshot()

    # Phase 4: seed Gaussians from new sparse SfM points
    expand_gaussians_from_new_points(gaussians, progressive_scene, new_point_mask, opt)

    # Phase 5: optionally seed dense Gaussians from monocular depth
    if (config.dense_init.enabled
            and progressive_scene.snapshot_idx >= config.dense_init.skip_first_n_snapshots):
        dense_init_for_new_images(
            gaussians, progressive_scene, new_cams, config.dense_init, opt,
            current_iter=global_iter,
        )

    train_window(...)
```

The driver function:

```python
def dense_init_for_new_images(gaussians, prog_scene, new_cams, dense_cfg, opt, current_iter):
    """
    Run DAv2 on each new image, align to SfM, filter, and append as Gaussians.
    DAv2 is loaded ONCE for the snapshot batch and freed before this returns.
    """
    accumulated_xyz = []
    accumulated_rgb = []
    scene_scale = prog_scene.cameras_extent

    pre_vram = torch.cuda.memory_allocated() / 1024**2
    logger.info(f"[dense-init] VRAM before: {pre_vram:.0f} MB")

    # ===== DAv2 LOAD =====
    with DepthAnythingV2Wrapper(dense_cfg.dav2) as depth_model:
        for cam in new_cams:
            sfm_xyz_all, visible_mask = prog_scene.get_sfm_points_visible_to(cam)
            sfm_xyz_visible = sfm_xyz_all[visible_mask]

            if len(sfm_xyz_visible) < dense_cfg.min_sfm_points_for_alignment:
                logger.info(f"[dense-init] skip {cam.image_name}: only {len(sfm_xyz_visible)} SfM points visible")
                continue

            # Depth-range check (avoids planar-scene rank deficiency in alignment)
            depths_in_cam = transform_to_camera_frame(sfm_xyz_visible, cam)[:, 2]
            depth_range = depths_in_cam.max() - depths_in_cam.min()
            if depth_range < dense_cfg.min_sfm_depth_range_fraction * scene_scale:
                logger.info(f"[dense-init] skip {cam.image_name}: SfM depth range too narrow")
                continue

            depth_map = depth_model.predict(cam.original_image)

            try:
                result = align_depth_to_sfm(depth_map, cam, sfm_xyz_visible, dense_cfg.ransac)
            except AlignmentFailed as e:
                logger.warning(f"[dense-init] alignment failed for {cam.image_name}: {e}")
                continue

            outcome = depth_to_points(
                depth_map, cam, result.a, result.b,
                target_n_points=dense_cfg.target_dense_points_per_image,
                sanity_threshold=dense_cfg.depth_disagreement_threshold * scene_scale,
                sfm_xyz_visible=sfm_xyz_visible,
                image_rgb=cam.original_image,
                max_rejected_fraction=dense_cfg.max_rejected_fraction,
            )
            if outcome is None:
                logger.warning(f"[dense-init] sanity filter rejected too much for {cam.image_name}, skipping")
                continue

            xyz, rgb = outcome
            accumulated_xyz.append(xyz)
            accumulated_rgb.append(rgb)
            logger.info(f"[dense-init] {cam.image_name}: kept {len(xyz)} dense points (a={result.a:.3f}, b={result.b:.3f}, inliers={result.n_inliers})")

            del depth_map  # bound peak VRAM during the loop
    # ===== DAv2 UNLOAD =====

    post_vram = torch.cuda.memory_allocated() / 1024**2
    logger.info(f"[dense-init] VRAM after: {post_vram:.0f} MB (delta {post_vram - pre_vram:+.0f})")
    assert post_vram - pre_vram < 100, f"DAv2 leaked VRAM: {post_vram - pre_vram:.0f} MB"

    if not accumulated_xyz:
        logger.info("[dense-init] no points produced for this snapshot")
        return

    all_xyz = np.concatenate(accumulated_xyz, axis=0)
    all_rgb = np.concatenate(accumulated_rgb, axis=0)
    del accumulated_xyz, accumulated_rgb

    # Novelty filter — only points genuinely far from existing Gaussians
    existing_xyz = gaussians.get_xyz.detach().cpu().numpy()
    tree = cKDTree(existing_xyz)
    threshold = dense_cfg.novelty_distance_threshold * scene_scale
    distances, _ = tree.query(all_xyz, distance_upper_bound=threshold)
    novel_mask = distances >= threshold
    n_novel = int(novel_mask.sum())

    if n_novel == 0:
        logger.info("[dense-init] all dense points covered by existing Gaussians — nothing added")
        return

    novel_pcd = BasicPointCloud(
        points=all_xyz[novel_mask],
        colors=all_rgb[novel_mask],
        normals=np.zeros_like(all_xyz[novel_mask]),
    )

    n_before = gaussians.get_xyz.shape[0]
    gaussians.expand_from_pcd(novel_pcd, np.ones(n_novel, dtype=bool), scene_scale)
    gaussians.training_setup(opt)
    n_after = gaussians.get_xyz.shape[0]
    gaussians.mark_recently_added(slice(n_before, n_after), iteration=current_iter,
                                  grace_iters=dense_cfg.grace_iters)

    logger.info(f"[dense-init] added {n_novel} dense Gaussians (rejected {len(all_xyz) - n_novel} as redundant)")
```

### 8.3 SfM visibility — implemented in `ProgressiveScene` per Phase 2

`prog_scene.get_sfm_points_visible_to(cam)` was specified in §5.1. The implementation: use COLMAP's `points3D.bin` track info when available (handles occlusion correctly), fall back to geometric projection.

### 8.4 VRAM swap discipline

The user's hard constraint: DAv2 and GS training must not be co-resident on GPU.

The pattern in §8.2 — DAv2 loaded inside a `with` block whose scope ends BEFORE `train_window` is called — accomplishes this. The `assert post_vram - pre_vram < 100` line in the driver enforces it at runtime; if the assertion fires, you have a leak and silent OOMs are coming a few snapshots later. Common culprits:

- DAv2 model not actually being deleted (Python reference still held somewhere — keep the model object scoped strictly inside the `with` block, no module-level globals).
- Cached output tensors. After last `predict()` call, explicitly `del depth` before context exit.
- `transformers` library caching on its end. After `del model`, call `torch.cuda.empty_cache()` AND `gc.collect()` AND `torch.cuda.synchronize()` (the wrapper's `__exit__` already does all three).

If you can't get clean VRAM behavior with `transformers`, fall back to loading DAv2 weights manually from the official repo and running raw `forward()` calls. More work but more controllable. Document which approach you ended up with in `NOTES.md`.

### 8.5 Acceptance criteria

1. **Alignment unit test.** Construct a synthetic depth map `D`, scramble as `D' = 0.3 * D + 1.7`, project ~200 fake SfM points using `D` to get true `(u, v, depth)` triples, run `align_depth_to_sfm(D', ...)`. Assert recovered `(a, b)` is within 1% of `(1/0.3, -1.7/0.3)` and `n_inliers >= 180`.

2. **Back-projection unit test.** Synthetic camera at known pose with known intrinsics, synthetic depth map, run `depth_to_points` with `(a=1, b=0)`, assert points land where expected in the world frame to within 1e-4.

3. **Sanity filter test.** Inject a localized depth error (set a 50×50 patch to 10× true depth), run `depth_to_points`, assert that patch is in the rejected set and the rest is kept.

4. **Skip-first-N test.** With `skip_first_n_snapshots: 3`, log scrape confirms no DAv2 inference for snapshots 0–2 and clean turn-on at snapshot 3.

5. **Planar-scene test.** Mock visible SfM points all on a plane (depth range = 0). Assert `dense_init_for_new_images` skips that camera with the depth-range-too-narrow message and does not attempt alignment.

6. **VRAM swap test.** With training in progress (~4 GB allocated), enter dense-init context, run inference on 5 mock images, exit. Assert peak VRAM during context ≤ 6 GB AND post-exit allocated within 100 MB of pre-entry. (The runtime assertion in the driver covers the second part automatically.)

7. **Pruning grace test.** Insert dense Gaussians via `dense_init_for_new_images`, then immediately run a Phase 7 prune pass. Assert all dense-init Gaussians inserted in the last `grace_iters` iterations survive.

8. **End-to-end speedup test.** On the demo data, run two configurations: `dense_init.enabled=false` and `dense_init.enabled=true`, holding all other params equal. Compare PSNR-vs-iteration curves. Dense init should reach the no-dense-init final PSNR in ≥30% fewer total merge-phase iterations. **If it doesn't, dense init isn't pulling its weight — debug before declaring done. Likely culprits: alignment producing wrong scale (Landmines L1, L4), sanity filter too aggressive, novelty filter too aggressive, or pruning culling dense Gaussians prematurely (Landmine L7).**

9. **Regression test on final quality.** Same comparison as #8 but at equal iteration count. Final PSNR with dense init should be within 0.5 dB of without (we don't expect a quality win, just no significant loss). If dense init is significantly *worse*, alignment is broken or the sanity filter isn't catching DAv2 hallucinations.

**Phase 5 acceptance:** All 9 tests pass.

---

## 9. Phase 6 — Match-graph parsing and per-camera weighting

**Goal:** Bias training toward the most recently added images and their match-graph neighbors.

### 9.1 Wire up the match matrix

In `progressive_train.py`, after each `load_next_snapshot()`, call:

```python
prog_scene.image_match_matrix = parse_match_matrix(
    matrix_path = prog_scene.snapshot_dir / "sparse/0/imageMatchMatrix.txt",
    names_path  = prog_scene.snapshot_dir / "sparse/0/imagesNames.txt",
    ordered_camera_names = [cam.image_name for cam in prog_scene.train_cameras],
)
prog_scene.image_weights = compute_image_weights(
    prog_scene.image_match_matrix, new_cam_indices,
)
```

### 9.2 Camera selection (port `GetTraining_Viewpoints`)

Within `train_window`, when `phase == "merge"`, build the per-iteration camera stack the way GS_On-The-Fly does: top-10 by weight + 10 random others (or all cameras if fewer than 20). When `phase` is `initial` or `final`, use all cameras uniformly (matching MEGS-2's existing behavior).

### 9.3 Loss weighting

Multiply the per-camera loss by that camera's weight before backprop. **Do this after combining L1 and SSIM, before optimizing-spa loss is added** (the spa-loss is a regularizer over Gaussians, not over cameras, and should not be scaled by per-camera weight):

```python
weight = prog_scene.image_weights[cam_index_in_train_cameras]
loss = weight * ((1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim_value))
if opt.optimizing_spa and ...:
    loss = optimizing_spa.append_spa_loss(loss)
```

### 9.4 Logging

For every iteration in the merge phase, log the chosen camera's name and weight to a `progressive_diary.txt` in the model output dir. We need to be able to confirm the weighting is doing something at debug time.

### 9.5 Acceptance test for Phase 6

Run on demo data with weighting on, then again with weighting off (uniform weights). After the same total iteration budget, the most-recently-added image's PSNR should be higher in the weighted run by ≥1 dB. If it's not, the weighting isn't reaching the gradient — debug.

**Phase 6 acceptance:** Per-camera weights are demonstrably affecting the gradient via the PSNR diff above.

---

## 10. Phase 7 — Pruning schedule for incremental training

**Goal:** Get MEGS-2's pruning to fire at the right times for the progressive setting, so that VRAM stays bounded across many snapshots, AND respect the dense-init grace period from Phase 5.

### 10.1 Three-phase iteration counter

Have `train_window` keep its own `phase_iter` counter and pass it to MEGS-2's existing iteration-comparison logic. Map phases to behaviors:

- **`phase="initial"`**: pre-`simp_iteration1` regime only. Allow `densify_and_prune_split` to fire on its normal schedule. **Do not** call `update_imp_score` or `OptimizingSpa` here. End of phase: scene is established, possibly with too many Gaussians.

- **`phase="merge"`**: lightweight regime. Allow densification on a slower schedule (every `MergeScene_Densification_Interval` ≈ 100 iters). **Disable opacity reset and disable depth-based `reinitial_pts`** in this phase — both are destructive to recently added Gaussians. **Do not** invoke MEGS-2's heavy pruning here either; we don't want to repeatedly compute `update_imp_score` over all cameras every K snapshots, that's O(snapshots × cameras) and will dominate runtime. Instead, every `prune_every_n_snapshots` (default 5) snapshots, run a *single* lightweight prune pass: call `update_imp_score` once and `gaussians.prune_points(mask)` with a low `prune_ratio` (try 0.05).

- **`phase="final"`**: full MEGS-2 pipeline. This is the one place we let `OptimizingSpa`, `OptimizingSpaSG`, the prune at `simp_iteration1`, the second prune at `optimizing_spa_stop_iter`, and the SG axis culling at `optimizing_spa_sg_stop_iter` all run as in stock MEGS-2. Use the user's full `iter_final` budget (default 4000) so these fire at sensible relative iterations. Use the `*_frac` config values to derive absolute thresholds: `simp_iteration1 = int(iter_final * simp_iteration1_frac)`, etc.

### 10.2 Grace-period protection (interaction with Phase 5)

**Critical:** Every prune mask construction — both the lightweight merge-phase prune AND the final-phase MEGS-2 prunes — must OR-in protection for Gaussians whose grace window is still open. After computing the candidate prune mask, do:

```python
grace_protected = gaussians.get_grace_protected_mask(current_iter)
prune_mask = prune_mask & ~grace_protected   # never prune protected Gaussians
```

This applies to:
- The lightweight `prune_every_n_snapshots` prune in merge phase.
- The first prune at `simp_iteration1` (uses `prune_ratio1`) in final phase.
- The second prune at `optimizing_spa_stop_iter` (uses `prune_ratio2`) in final phase.
- The SG axis culling at `optimizing_spa_sg_stop_iter` (this one only protects entire Gaussians, not individual axes — if the Gaussian is protected, leave all its axes alone).

Add a unit test (also listed under Phase 5 acceptance #7): insert dense Gaussians, immediately run a prune pass, confirm zero of them are pruned.

### 10.3 The `num_max` ceiling

MEGS-2's `--num_max` is an absolute Gaussian-count ceiling that gates densification. **Halve it** for progressive runs. Default 4.5M is way too high for 6 GB. Set the new default to whatever you discovered as the empirical ceiling in Phase 0; expect something like 800K–1.2M. Note that with dense init enabled, the ceiling can be hit faster — log when it's reached and reduce or skip dense init accordingly.

### 10.4 LR scheduler

For the simple-first-pass goal, do NOT port `On_The_Fly_Update_Lr`. Instead, in the merge phase, freeze the position LR at its `update_learning_rate(simp_iteration1 / 2)` value — i.e., somewhere in the middle of the warmup-decay curve, so newly added Gaussians can move but old ones aren't being yanked around. In the final phase, restart MEGS-2's LR schedule fresh from iteration 0 of that phase.

### 10.5 Acceptance test for Phase 7

Run the full progressive pipeline on a 30-image scene split into 5 snapshots of 6 images each. Track:
- Peak VRAM via `nvidia-smi` polling or `torch.cuda.max_memory_allocated()`. Must stay <5.5 GB during training, <6 GB during dense-init inference.
- Gaussian count over time. Should grow during merge phases and dense-init events, drop after each `prune_every_n_snapshots` event, and drop substantially during the final phase.
- Final PSNR within 2 dB of MEGS-2 stock training on the same final scene (not progressive).
- Grace test: insert dense Gaussians, force a prune to fire on the next iteration, confirm grace-protected ones survive.

**Phase 7 acceptance:** All four metrics pass on the test scene.

---

## 11. Phase 8 — WebGL viewer hookup

**Goal:** The output `.ply` files render correctly in `MEGS-2/WebGL_viewer/` using `main_sg.js`.

This is mostly free if you used MEGS-2's `Scene.save()` unchanged in `save_checkpoint`. Verify:

1. After a progressive run, copy the most recent `point_cloud.ply` from `<model_path>/point_cloud/iteration_*/` into `MEGS-2/WebGL_viewer/`.
2. In `WebGL_viewer/index.html`, ensure the `<script>` tag points to `main_sg.js` (not `main_sh.js`).
3. Update the `url` in `main_sg.js`'s `main()` function to point at your `.ply` filename.
4. From `WebGL_viewer/`, run `http-server` and open the page in Chrome. The scene should render.

If the .ply doesn't load, the most likely cause is a mismatch between SG attribute names in your saved .ply vs. what `main_sg.js` expects. Inspect MEGS-2's stock `point_cloud.ply` (from the Phase 0 baseline) with `head -c 4096 point_cloud.ply | xxd | head -100` to read the header, and diff against your progressive output.

**Phase 8 acceptance:** Browser renders a recognizable scene from a checkpoint produced by `progressive_train.py`.

---

## 12. Phase 9 — Full validation pass on target hardware

Run the full pipeline on the largest scene that fits, on the actual target hardware (6 GB VRAM, 16 GB RAM Ubuntu server). Stress-test:

- 50+ images split into 10+ snapshots.
- Run while monitoring `nvidia-smi -l 1` in another terminal.
- Confirm peak VRAM stays under 5.5 GB during training and under 6 GB during dense-init inference.
- Confirm wall-clock per-snapshot is reasonable. Target: under 60s per merge for 800×800 images on a 6 GB-class card; flag if it's much worse. Dense init adds maybe 2–10s per new image (DAv2 + alignment + back-projection); should be a small fraction of the merge-phase training time.
- Confirm system RAM (16 GB) doesn't blow up — `psutil` checkpoints in your training logs will help diagnose if it does. Cameras hold their decoded images on CPU; with too many snapshots this can balloon. Dense-init code keeps temporary arrays per-snapshot (~6 MB each) but should not accumulate them across snapshots — if it does, that's Landmine #L8 biting.
- Run the full pipeline with `dense_init.enabled=true` AND with `dense_init.enabled=false`, sequentially, on the same snapshot data. Confirm both code paths complete cleanly. The enabled run should reach equivalent PSNR in fewer total iterations (per Phase 5 acceptance #8); the disabled run should behave exactly as if Phase 5 didn't exist.

**Phase 9 acceptance:** Production-shape run completes within VRAM and RAM budgets, with reasonable wall-clock, in both dense-init-enabled and disabled modes.

---

## 13. Known landmines (read this section before coding)

Phase-tagged where applicable. In rough order of how likely each is to cost you a day:

**[Phase 4] L1. Optimizer parameter group extension is fragile.** When `expand_from_pcd` appends Gaussians, every Adam moment tensor must be extended in lockstep with its parameter, or training silently corrupts. Don't reinvent this — find and call MEGS-2's existing `cat_tensors_to_optimizer` (or equivalently named) helper. If you can't find one, derive it from the existing `densification_postfix` code path. **Test parameter group integrity** after expansion by checking each group's `params[0].shape` matches the corresponding model tensor.

**[Phases 5, 7] L2. `update_imp_score` iterates all training cameras.** It's expensive. Calling it every snapshot will slow you down dramatically. Only call during the final-refinement phase (and the every-N-snapshots lightweight prune in merge phase, if you do that).

**[Phase 7] L3. `reinitial_pts` is a giant scene reset.** It samples points from rendered depth and replaces the entire Gaussian cloud. NEVER call this during the merge phase — it will discard everything you just learned, including any dense-init seeded Gaussians. Confine it to the final-refinement phase only, and even then, consider gating it off entirely to save time, since progressive training already produces a reasonable cloud.

**[Phase 7] L4. The `simp_iteration1` and `optimizing_spa_*_iter` defaults are absolute.** Stock MEGS-2 uses 15000+ for these on a 40000-iteration run. If you naively pass these into a 4000-iter final phase, none of MEGS-2's pruning will ever fire and the scene will OOM. Scale these proportionally via the `*_frac` config values (Phase 7.1 and the config in Phase 3.2 tell you how).

**[Phase 0] L5. `fused_ssim` may not build under PyTorch 2.x without patches.** If it doesn't, fall back to MEGS-2's `utils.loss_utils.ssim` (the import in `train.py` already has a try/except for this).

**[Phase 2, 5] L6. Pose updates for previously-registered cameras.** GS_On-The-Fly handles this by overwriting `R`/`T` on the existing Camera objects when a snapshot revises them. Replicate this exactly. If you instead create new Camera objects, you'll detach them from any image-name-based bookkeeping (image weights, training counts, dense-init grace records) and bugs follow.

**[Phase 6] L7. `imageMatchMatrix.txt` ordering vs. registration ordering.** The text file's row order is the order images were registered in the upstream SfM. Your `train_cameras` order is whatever MEGS-2's COLMAP loader produced. These are NOT guaranteed to match. The reorder logic in `GetImageMatchingMatrix` (look up each `train_camera.image_name` in `imagesNames.txt` and reindex) is essential — port it precisely. Don't skip it.

**[General] L8. The user's incremental SfM is GLOMAP-based, not the original "On-The-Fly SfM."** They will produce the snapshot folders themselves. Confirm with them (or assume) that their adapter writes `imageMatchMatrix.txt` and `imagesNames.txt` in the format described in §4.4. If they don't, the integration won't work at runtime even if your code is perfect — flag this in your final handoff notes.

**[Phase 8] L9. MEGS-2's WebGL viewer expects a specific .ply attribute layout.** If you change the SG attribute set or naming, the viewer breaks. Don't modify `SphericalGaussianModel.save_ply` or its inverse.

**[Phase 0] L10. PyTorch3D import in GS_On-The-Fly.** GS_On-The-Fly imports pytorch3d at the top of `ContinuosProgressiveTrain.py` even though most of the file doesn't use it. **You are not running their script,** so don't install pytorch3d unless your ported code actually uses it (it shouldn't).

**[Phase 5] DL1. Camera intrinsics units.** MEGS-2's `Camera` class stores FoV (radians? degrees? check), not focal length in pixels. To project SfM points and back-project depth, you need `fx = W / (2 * tan(FoVx / 2))` and `fy = H / (2 * tan(FoVy / 2))`. Off-by-one or radian/degree confusion here produces points at roughly the right place but wrong scale, alignment LSQ "succeeds" with a compensating `(a, b)`, and the ONLY symptom is that PSNR is a few dB worse than baseline. Verify by projecting a known SfM point into a camera and checking the projected pixel matches what COLMAP recorded — do this BEFORE you trust the dense back-projection.

**[Phase 5] DL2. DAv2's depth convention.** DAv2 outputs *inverse depth* (or relative depth needing inversion — verify per the model card and by inspection). Either way, the linear model `aligned = a * raw + b` still works for fitting, but if you assume the raw output is depth and it's actually inverse depth, your back-projection scales nonlinearly with distance and your point cloud has the right topology but wrong geometry. Test: feed DAv2 an image with a known close object and a known far object, inspect raw values at those pixels, confirm sign convention before writing the alignment.

**[Phase 5] DL3. Coordinate frame conventions.** COLMAP/MEGS-2 typically uses `cam = R @ world + T` (world-to-camera). Some loaders invert this. Test: pick an SfM point, transform to camera frame using your code, verify positive Z (in front of camera) corresponds to "I should see this point" not "this point is behind me." Same axis flip ambiguity exists for the back-projection step.

**[Phase 5] DL4. DAv2 input preprocessing.** DAv2 expects inputs normalized with specific mean/std and resized so dimensions are multiples of 14. The `transformers` `AutoImageProcessor` handles this; raw use of the model does not. If you go raw, replicate the exact preprocessing or your depths will be subtly wrong in ways the alignment LSQ partially compensates for. Strong recommendation: use `transformers` and don't go raw unless the VRAM swap test fails.

**[Phase 5] DL5. Alignment LSQ rank deficiency on planar scenes.** If all visible SfM points lie on a plane (e.g., scanning a flat painting), alignment is rank-deficient — `(a, b)` becomes unstable. The `min_sfm_depth_range_fraction` config setting handles this by skipping such images. Verify the threshold isn't so high it skips legitimate scenes.

**[Phases 5, 7] DL6. Pruning culling fresh dense Gaussians.** The most-likely-to-bite issue at integration time. MEGS²'s `update_imp_score` ranks Gaussians by their accumulated rendering contribution. Dense-init Gaussians have zero accumulated contribution at insertion time, so any pruning pass that fires within a few iterations will rank them lowest and cull them — destroying the entire point of dense init. **Mitigation (must implement, not optional):** the `mark_recently_added` mechanism in Phase 4 records which Gaussian indices came from dense init. Modify the prune mask construction in Phase 7 to OR-in protection for Gaussians whose grace window is still open. Default grace = 200 iters. Phase 5 acceptance test #7 covers this directly — make sure it passes.

**[Phase 5] DL7. Memory blowup on accumulated arrays.** With 50K points/image × 10 new images per snapshot, `accumulated_xyz` is 500K × 3 × 4 bytes = 6 MB. Fine for one snapshot. But if you accidentally retain it across snapshots (e.g., as instance state on `ProgressiveScene`), you'll consume host RAM rapidly. Keep it as a local variable that goes out of scope each call. Verify with `psutil` checkpoints during the Phase 9 stress test.

**[Phase 5] DL8. `transformers` library version drift.** `AutoModelForDepthEstimation` API has been stable but watch for breaking changes if you bump versions during development. Pin the version in `requirements.txt` once you have a working setup.

**[Phase 5] DL9. Subsample bias.** Uniform stride-sampling oversamples flat regions where depth is least informative. Importance-sampling by depth gradient would be better but is engineering not in v1 scope. Document in `NOTES.md` future work.

---

## 14. Deliverables checklist

When you're done, the user should have:

- [ ] Forked `MEGS-2/` repo with a `progressive` branch.
- [ ] `progressive_train.py` at the root.
- [ ] `configs/progressive.yaml` with documented defaults.
- [ ] `scene/progressive_scene.py` (with `get_sfm_points_visible_to`).
- [ ] `scene/match_matrix.py`.
- [ ] `scene/dense_init.py` with `DepthAnythingV2Wrapper`, `align_depth_to_sfm`, `depth_to_points`, `AlignmentFailed`, `AlignmentResult`.
- [ ] `expand_from_pcd` and `mark_recently_added` / `get_grace_protected_mask` methods on `SphericalGaussianModel`.
- [ ] Phase 7 prune logic that respects the grace-period mechanism.
- [ ] Tests: `tests/test_progressive_scene.py`, `tests/test_expand_from_pcd.py`, `tests/test_dense_init.py` (covering the 9 acceptance tests in §8.5), plus a smoke test that runs a 3-snapshot end-to-end on tiny data.
- [ ] `NOTES.md` documenting architectural decisions, DAv2 depth convention verification, any deviations from this plan, and future-work items.
- [ ] `README_PROGRESSIVE.md` documenting how to invoke `progressive_train.py`, the YAML config schema, the expected snapshot folder layout, the WebGL viewer steps, and the dense-init A/B comparison.
- [ ] Working browser render of a progressive-trained scene checkpoint.
- [ ] A "future work" section in `NOTES.md` listing items deferred from this implementation:
  - Adaptive LR (`On_The_Fly_Update_Lr`)
  - Depth-based reinit during progressive phase
  - Equivalent-training-time tracking
  - Streaming/diff-based viewer updates rather than full reload
  - Async DAv2 inference
  - Importance-sampled depth subsampling
  - Larger DAv2 variants (base / large) if quality requires
  - MVS-based refinement for the final-refinement phase
  - Depth-supervised loss term during training (not just initialization)
  - DAv2 confidence-aware sanity filter using model intermediate features

---

## 15. If you get stuck

If a phase's acceptance criteria fail and you can't diagnose within ~4 hours of focused work, stop and write a `BLOCKED.md` documenting:
- Which phase
- What the failing test/symptom is
- What you've tried
- What you suspect
- What you'd try next given more context

Hand that back to the user rather than guessing forward. The phases are ordered so that earlier failures don't get masked by later complexity — fix Phase N before moving to Phase N+1.

Particularly for Phase 5 dense init: if the end-to-end speedup test (acceptance #8) fails, the four most likely causes are (a) DAv2 depth convention misread (DL2), (b) camera intrinsics misread (DL1), (c) coordinate frame flip (DL3), (d) pruning eating dense Gaussians before they can prove themselves (DL6). Test these in that order — they're easiest-to-hardest to diagnose.

Good luck.
