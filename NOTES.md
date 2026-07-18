# Implementation Notes: Progressive MEGS²

## Landmine → test coverage (Milestone 6 index)

Items below that are now enforced by tests are listed here and kept in the
prose only for background. When a landmine bites again, its test is the
first thing to re-read.

| Landmine | Enforced by |
|---|---|
| L7 — matrix row order ≠ camera order | `tests/test_match_matrix.py::test_parse_match_matrix_reorder`, `::test_parse_home_server_format_l7_reindex` |
| L8 — match-matrix files may not exist | uniform-fallback branch logs one WARNING (`continuous_train.py`); home-server now exports the files (`docs/ingest_contract.md`); format round-trip in `tests/test_match_matrix.py::test_parse_home_server_format` |
| DL2 — DAv2 outputs disparity, DA3 outputs depth | `tests/test_dense_init_da3.py::test_validate_alignment_*` (sign gate per backend) |
| T1 — Adam state lost on rebind after prune | `tests/test_optim_guard.py` (+ always-on `check_invariants`) |
| Grace ranges stale after prune / mid-order densify append | `tests/test_index_remap.py` (incl. simulated historical misalignment) |
| `_sg_axis_count` / split prune-mask end-append misalignment | `tests/test_index_remap.py::test_simulated_axis_count_stays_aligned_through_clone_and_prune`, `::test_perm_gathers_endcat_into_final_order` |
| Scheduler bugs: densify_saturation never aging out; stalled grinding forever | `tests/test_convergence.py::test_densify_saturation_ages_out_by_iter`, `::test_plateau_reads_stalled_then_promotes_to_converged` |
| Idle must not burn GPU when converged | converged-idle gate in `continuous_train.py` (structure); state reachability pinned in `tests/test_convergence.py` |
| Trigger require-state / anti-thrash floors | `tests/test_triggers.py` (every predicate branch) |
| SkipGS budget floor / warmup lifecycle | `tests/test_skipgs.py` |
| Ingest retries must be no-ops (incl. across restarts) | `tests/test_continuous_server.py::test_ingest_duplicate_is_200_noop`, `tests/test_train_state.py::test_ledger_round_trip_preserves_idempotency` |
| Gravity/camera-convention math (DL1/DL3) | `tests/test_dense_init.py` geometry tests; conditioning matrices in `tests/test_dense_init_da3.py::test_conditioning_matrices` |

## Codebase Reconnaissance (Phase 1 answers)

### MEGS-2 train.py iteration structure

1. **Pre-`simp_iteration1` (default 15 000)**: Normal densify-and-clone/split loop every `densification_interval` iters. Depth-based reinit fires every 5000 iters (replaces all Gaussians from rendered depth — destructive, must NOT run during merge phases). Opacity reset also fires.

2. **At `simp_iteration1`**: `update_imp_score` over all cameras → random-sample keep `(1-prune_ratio1)` by importance probability → `prune_points` → `reinitial_pts` (resets the entire cloud to the surviving positions). This is a **large-scale destructive reset**.

3. **`optimizing_spa_start_iter` → `_stop_iter`**: Lagrangian-based sparsity regulariser (`OptimizingSpa`) fires every `optimizing_spa_interval` iters. Also `OptimizingSpaSG` fires over the same range for SG axis sparsity.

4. **At `optimizing_spa_stop_iter`**: Second importance-based prune by `prune_ratio2`.

5. **At `optimizing_spa_sg_stop_iter`**: `cull_low_sharpness_axes` kills SG axes below sharpness threshold.

The **absolute** thresholds (15000, 25200, 35200, …) must be scaled proportionally when running a shorter `iter_final` budget. The `*_frac` config values handle this — see `configs/progressive.yaml`.

### `reinitial_pts` behaviour

`reinitial_pts` **replaces** the entire Gaussian cloud (all tensors). It destroys any accumulated gradient stats, Adam moments, and any dense-init Gaussians that were present. This is why it is **strictly confined to the final-refinement phase** in this implementation and explicitly skipped during merge phases.

### `update_imp_score` cost

It renders **all training cameras** once. For a 50-image scene this is 50 forward passes per call. In stock MEGS-2 it's called ~100 times during the Lagrangian window, which is fine over a 40 000-iter run. In a merge phase with ~200 iters calling it even once would be ~25% overhead. Progressive strategy: call it only at `prune_every_n_snapshots` events and at final-phase schedule points.

### `SphericalGaussianModel` key API

- `create_from_pcd(pcd, spatial_lr_scale)`: Full init from scratch.
- `expand_from_pcd(pcd, mask, spatial_lr_scale)`: **New** — appends new Gaussians using `cat_tensors_to_optimizer` (Adam moments extended to zero for new Gaussians, preserved for old ones).
- `mark_recently_added(idx_range, iteration, grace_iters)` / `get_grace_protected_mask(iter)`: **New** — grace-period protection for freshly inserted Gaussians.
- `densification_postfix(...)`: Calls `cat_tensors_to_optimizer` and **resets** `xyz_gradient_accum`, `denom`, `max_radii2D` for ALL Gaussians. Our `expand_from_pcd` does NOT call `densification_postfix` for this reason — it calls `cat_tensors_to_optimizer` directly and extends the accum tensors by *concatenation*, preserving existing stats.
- `cull_low_sharpness_axes(sharpness_threshold)`: Only valid in non-`variable_sg_bands` mode (i.e. after `create_from_pcd` which calls `_initialize_spherical_gaussians_unified`). After loading a .ply, the model may switch to `variable_sg_bands=True` list-based tensors — be careful.

### Scene / Camera conventions

- `Camera.R` is stored as `R = np.transpose(qvec2rotmat(qvec))` (R_stored = R_colmap.T).
- `Camera.world_view_transform` = `getWorld2View2(R, T).T` where `getWorld2View2` = `[[R_stored.T, T], [0,0,0,1]]` = `[[R_colmap, T], [0,0,0,1]]` (the actual W2C).
- So `world_view_transform.T` = actual 4×4 W2C matrix.
- `Camera.FoVx/FoVy` are in **radians**. `fx = W / (2 * tan(FoVx/2))`.
- To project: `p_cam = R_w2c @ p_world + t_w2c` where `R_w2c = world_view_transform.T[:3, :3]`, `t_w2c = world_view_transform.T[:3, 3]`. Depth = `p_cam[2]`; must be > 0 for visible points (Z forward in COLMAP).
- To back-project: `p_world = R_w2c.T @ (p_cam - t_w2c)` = `R_w2c.T @ p_cam - R_w2c.T @ t_w2c`. Camera center in world = `R_w2c.T @ (-t_w2c)` (confirmed by MEGS-2: `camera_center = world_view_transform.inverse()[3, :3]`).

### GS_On-The-Fly snapshot format

```
<source_path_dir>/
  16/
    images/         ← all images registered so far (cumulative)
    sparse/0/
      cameras.bin
      images.bin
      points3D.bin
      imageMatchMatrix.txt   ← co-visibility feature counts (row = image, col = image)
      imagesNames.txt        ← comma-sep list of filenames, matches matrix row order
  17/
    ...             ← superset of 16/
```

Each numbered folder is a **complete** COLMAP reconstruction at that point. Images appear in ALL subsequent folders once registered.

`imageMatchMatrix.txt`: one line per registered image, comma-separated int counts, trailing newline. Log-normalised per row as `log(x+1)/log(max_in_row+1)`.

`imagesNames.txt`: single line, comma-sep filenames in **registration order** (NOT alphabetical; must reindex to match `train_cameras` order — Landmine L7).

## DAv2 Depth Convention (DL2)

**Verified by inspection**: Depth Anything v2 outputs *relative inverse depth* (disparity). Larger raw values correspond to **closer** objects. However, since `align_depth_to_sfm` fits the linear model `aligned = a * raw + b` using RANSAC against true SfM depths (which are metric camera-frame depths), the sign and scale are absorbed into `(a, b)`. For typical scenes (a > 0), the raw output already correlates positively with depth after alignment.

**Action**: The implementation correctly uses `aligned = a * raw + b` throughout. Do NOT assume raw output is metric depth — always align first.

## VRAM Discipline (Phase 5)

`DepthAnythingV2Wrapper` is a context manager. The `with` block ends BEFORE `train_window` is called, so DAv2 and GS training are never co-resident on GPU. The runtime assertion `post_vram - pre_vram < 100` (MB) in `dense_init_for_new_images` enforces this at runtime.

If the assertion fires: check for module-level globals or closures that hold references to the model. The wrapper's `__exit__` calls `del self.model`, `gc.collect()`, `torch.cuda.empty_cache()`, and `torch.cuda.synchronize()` in that order.

## `imageMatchMatrix.txt` ordering vs. registration ordering (L7)

The matrix file's row order matches `imagesNames.txt` (SfM registration order). MEGS-2's COLMAP loader sorts cameras alphabetically by image name (`sorted(..., key=lambda x: x.image_name)`). These are NOT the same. `parse_match_matrix` in `scene/match_matrix.py` handles this by looking up each `train_camera.image_name` in `imagesNames.txt` and constructing the reordered matrix explicitly.

## `cat_tensors_to_optimizer` vs `densification_postfix`

`densification_postfix` calls `cat_tensors_to_optimizer` AND resets `xyz_gradient_accum`, `denom`, `max_radii2D` for ALL Gaussians. Our `expand_from_pcd` instead:
1. Calls `cat_tensors_to_optimizer` directly (extends Adam moments with zeros for new Gaussians, keeps old moments intact).
2. Concatenates `xyz_gradient_accum`, `denom`, `max_radii2D` with zeros for new Gaussians only.

This preserves accumulated gradient statistics for existing Gaussians across snapshot boundaries.

## Deviations from Plan

1. **`variable_sg_bands`**: MEGS-2's model can run in `variable_sg_bands=True` mode (list-based tensors per degree) or unified tensor mode (single tensor). `_initialize_spherical_gaussians_unified` always creates unified tensors (the training path). `expand_from_pcd` assumes unified tensors (same as `create_from_pcd`). If `load_ply` was called first (which switches to variable_sg_bands mode with lists), `expand_from_pcd` would need adaptation. For the progressive use case we always start from `create_from_pcd` so this is not an issue.

2. **Adapter writes (L8 — resolved)**: The home-server now exports
   `imageMatchMatrix.txt`/`imagesNames.txt` into the promoted `sparse/0`
   (newline/space format, auto-detected by `parse_match_matrix`; see
   `docs/ingest_contract.md`). When absent, the fallback to uniform
   weights with new-camera bias logs one clear WARNING instead of being
   silent.

3. **`cull_low_sharpness_axes` with unified tensors**: The model after `create_from_pcd` uses unified `(N, max_sg_degree, 3)` tensors, and `cull_low_sharpness_axes` accesses `self._sg_directions.shape` directly — this works for the unified path. If `variable_sg_bands=True` with list tensors, a different code path would be needed.

## Future Work

- **Adaptive LR** (`On_The_Fly_Update_Lr`): GS_On-The-Fly tracks equivalent training iterations per camera and adjusts LR accordingly. Deferred to v2.
- **Depth-based reinit during progressive phase**: `reinitial_pts` is destructive; deferred until a non-destructive depth-seeding approach is designed.
- **Equivalent-training-time tracking** (`ImagesAlreadyBeTrainedIterations_Set`): Tracks how many iterations each camera has effectively been trained. Would improve fairness of camera sampling. Deferred.
- **Streaming viewer updates**: Currently writes a full .ply per snapshot. A diff-based approach would be faster to load.
- **Async DAv2 inference**: Currently blocks training per snapshot. A producer-consumer design on CPU would hide latency.
- **Importance-sampled depth subsampling** (DL9): Uniform stride oversamples flat regions. Gradient-based sampling deferred.
- **Larger DAv2 variants**: `base` and `large` need more VRAM. Use only if quality requires.
- **MVS-based refinement** for final phase.
- **Depth-supervised loss** during training (not just init).
- **DAv2 confidence-aware sanity filter** using model intermediate features.
