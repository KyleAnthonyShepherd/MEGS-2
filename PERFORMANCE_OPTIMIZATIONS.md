# Performance Optimizations for Progressive MEGS-2 on 6 GB VRAM

**Target system**: GTX 1660 Ti (Turing SM 7.5), 6 GB VRAM, Linux, CUDA 12.1.
**Workload**: progressive ingestion of multiple snapshots, ~13 images / ~800k splats steady-state, ~1.5 GB VRAM utilised.
**Branches considered**: `incremental` (upstream MEGS-2 baseline), `claude/optimize-low-vram-performance-ECpjf` (current branch with `progressive_train.py` built atop incremental).
**Scope**: this document is the complete research output. Each "P*" entry below is a self-contained brief that a Sonnet instance can implement without reading the rest of the report.

---

## 1. Executive Summary

Two bottleneck families dominate iters/sec and quality/iter in `progressive_train.py`:

1. **Optimizer-state churn**. `training_setup(opt)` is called after every prune (`progressive_train.py:632, 686`) and after `reinitial_pts`. This re-creates the Adam optimiser from scratch — discarding the `exp_avg` / `exp_avg_sq` buffers that `_prune_optimizer` (`scene/spherical_gaussian_model.py:389-405`) had just carefully pruned-in-place. Every prune resets thousands of iterations of momentum.
2. **Single-view forward/backward**. Every iteration renders one camera, computes loss, backprops. With 800k Gaussians the per-call rasterizer kernel-launch overhead and the per-call `screenspace_points` allocation (~9.6 MB, `spherical_gaussian_renderer/__init__.py:72`) dominate at this scale on a 1660 Ti.

A third axis — initialization quality — is the cheapest way to **reduce iters needed**: DAv2-small at 50k points/image is conservative for 4.5 GB free VRAM. Going to DAv2-base typically halves the depth error and reduces required `iter_initial` and `iter_per_merge`.

Top recommended changes (high impact, low risk, all Python-only):

| # | Change | Iters/sec | Iters needed | Effort |
|---|--------|-----------|--------------|--------|
| **P1** | Stop rebuilding Adam after prune (preserve state) | +5-10% | -10-25% | S |
| **P3** | DAv2-base + 100k dense points/image, dense-init from snapshot 0 | -5% | -25-40% | S |
| **P12** | Convergence-driven scheduling: monitor + elastic budget + early stopping + fast final refinement | +10-30% | -20-50% | M |
| **P13** | Content-driven densify/prune trigger predicates | +5-15% | -5-15% | M |
| **P5** | Multi-view gradient accumulation (2-4 cams/step, pure PyTorch) | +10-25% | -10-30% | M |

Cumulative: realistic 2-3× wall-clock speedup to a comparable-quality scene. CUDA kernel-level work (P10, P11) is **not recommended** for this hardware — the user's bottleneck is Python, optimiser bookkeeping, and over-training, not raster throughput.

---

## 2. Profiled Findings (Static Analysis)

### 2.1 Per-iteration hot path
Single iteration cost breakdown for the merge / initial / final loop (`progressive_train.py:464-722`):

| Step | File:lines | Cost @ 800k Gaussians | Notes |
|------|------------|-----------------------|-------|
| Camera selection + image fetch | 480-493 | trivial | `cam.original_image.cuda()` is a no-op (image already on GPU per `scene/cameras.py:39`). The `.cuda()` call is harmless but misleading — flag for cleanup, not a perf bug. |
| `screenspace_points` allocation | renderer:72 | ~9.6 MB malloc/iter | Caching allocator usually absorbs, but it churns the pool. Pool reuse is easy. |
| `compute_colors_precomp` | model:228-295 | dominant Python-side | Pure-PyTorch SH-with-spherical-Gaussian colour evaluation. Per-call: `_rgb_base.clone()`, `view_dirs` (N×3) twice (lines 236/260 — duplicated), norm, mask scatter, exp. Several N-sized intermediates per iter. |
| Rasterizer forward | renderer:53-61 (and `_ms` variant) | dominant CUDA-side | `diff_gaussian_rasterization_ms` for `render_imp` / `render_depth`; standard for `render`. Single-camera. |
| Activation getters | model:130-160 | repeatedly allocated | `get_scaling`, `get_opacity`, `get_rotation`, `get_sg_*` re-apply `exp/sigmoid/normalize/abs` on every property access. Called 2-4× per iter (render, densify, score). |
| L1 + (fused) SSIM | progressive_train.py:505-509 | trivial | `fused_ssim` is auto-detected. |
| Backward | 533 | dominant CUDA-side | Single-view backward through CUDA rasterizer + Python SH eval. |
| Densify/prune (every 100 iter) | 547-604 | spike every 100 iter | Many `.repeat(N,1)` allocations in `densify_and_prune_split` (model:797-847). |

### 2.2 Periodic costs

| Operation | Trigger | Cost |
|-----------|---------|------|
| `update_imp_score` | every 100 iters during SPA window in final phase + at simp_iteration1 + at optimizing_spa_stop_iter; also each `lightweight_prune` | **N_cameras × full render**, no gradients but full importance forward pass. With 13 cameras × ~12 calls per phase = ~150 extra full-render passes per snapshot. |
| `lightweight_prune` | every `prune_every_n_snapshots` (default 5) snapshots during merge | calls `update_imp_score` over all cameras + a `cKDTree` rebuild |
| `expand_from_pcd` | each merge snapshot's dense-init phase | `cat_tensors_to_optimizer` rebuilds Adam state for each param group; dense kNN tree build on CPU |
| `training_setup(opt)` | **after every prune (final phase) and after reinitial_pts** (`progressive_train.py:632, 686`) | reallocates entire Adam state dict — discards momentum |
| `dense_init_for_new_images` | each merge snapshot (after `skip_first_n_snapshots`) | DAv2 model load (~10-30 s for small) + N images × DAv2 inference + N × CPU RANSAC + `cKDTree` |
| `parse_match_matrix` + `compute_image_weights` | each merge snapshot | O(M²) numpy, M = num cameras. Fine. |

### 2.3 Initialisation budget (iters-needed axis)

`configs/progressive.yaml`:
- `dav2.model_size: small` (~100 MB VRAM)
- `target_dense_points_per_image: 50000`
- `skip_first_n_snapshots: 3` — first 3 snapshots get NO dense init
- `iter_initial: 3000`, `iter_per_merge: 200`, `iter_final: 4000`

User VRAM budget: 1.5 GB used of 6 GB → **4.5 GB headroom**. DAv2-base is ~390 MB; even with the live training scene this fits comfortably during the brief dense-init context.

### 2.4 Things that look bad but aren't

- `cam.original_image.cuda()` every iter — harmless (already GPU); just confusing. Don't waste effort here.
- `screenspace_points` 9.6 MB churn — caching allocator reuses; pooling is a clean-up, not a real win. Listed as P9 only for completeness.
- `parse_match_matrix` O(M²) — only runs once per snapshot, M ≤ ~50; not hot.
- RANSAC on CPU — only inside dense-init context (per-snapshot), 200 iter × ~13 cams ≈ <1 s. Not hot.

---

## 3. Recommended Optimisations (Prioritised)

### Category A — Iters/sec (wall-clock per iteration)

| ID | Change | Risk | Effort | Expected gain |
|----|--------|------|--------|---------------|
| **P1** | Don't rebuild Adam after prune; reset only side buffers | low | S | +5-10% iters/sec **and** -10-25% iters needed (preserved momentum converges faster) |
| **P2** | De-dup view-dir computation in `compute_colors_precomp`; fuse training/non-training branches | low | S | +1-3% iters/sec |
| **P5** | Multi-view gradient accumulation (2-4 cams/step, pure PyTorch loss summation) | med | M | +10-25% iters/sec equivalent (work/sec); -10-30% iters needed |
| **P9** | Pool `screenspace_points` buffer | low | S | +0-2%, mostly reduces fragmentation |
| **P10** | True multi-view batched rasterizer (CUDA kernel work) | **high** | XL | +20-40% but rebuild of `diff_gaussian_rasterization_ms` required; **not recommended for SM 7.5 in this iteration** |
| **P11** | SkipGS-style selective backward | **high** | XL | +10-20% but kernel work; **defer** |

### Category B — Iters needed (quality per iteration)

| ID | Change | Risk | Effort | Expected gain |
|----|--------|------|--------|---------------|
| **P3** | Upgrade DAv2 small → base; raise `target_dense_points_per_image` 50k → 100k; lower `skip_first_n_snapshots` 3 → 0 | low | S | -25-40% iters needed for comparable quality |
| **P4** | Smarter prune cadence: aggressive early prune to keep N below 60% of `num_max`, soft-cap at 95% | med | M | iters/sec speedup proportional to fewer splats; can compress `iter_per_merge` |
| **P6** | Reduce `update_imp_score` cost: subsample cameras (top-K by weight) instead of all | low | S | +2-5% wall-clock; quality unchanged |
| **P7** | Clamp `iter_per_merge` based on actually new content; if no new dense points, fewer iters | low | S | -10% wall-clock per merge |
| **P8** | Persist DAv2 model across snapshots (singleton) | low | S | -10-30 s per snapshot (load cost) |

### Category C — Out of scope for this 6 GB system

- Grendel-GS (ICLR'25) sharded data parallel — single-GPU, not applicable.
- "Efficient multi-view training for 3DGS" (CVPR'25) — multi-view batched rasterizer; useful inspiration for **P5** (Python-side accumulation captures most of the gain) but full kernel rewrite is XL effort.
- Note: the user's cited arXiv id `2603.08997` parses as a 2026-03 paper — the actual SkipGS paper id is different. Do not chase that link without the user confirming the correct reference. The Python-side selective-backward we propose in P5 is independent of any specific kernel paper.

---

## 4. Implementation Briefs (Sonnet-ready)

Each brief is self-contained and assumes a working tree on `claude/optimize-low-vram-performance-ECpjf`. Implement in order P1, P3, P5, P6, P7, P8, P4, P2, P9.

---

### P1 — Preserve Adam state across prunes

**Goal**: stop discarding optimiser momentum on every prune.

**Files**:
- `scene/spherical_gaussian_model.py`
- `progressive_train.py`

**Background**:
`prune_points` (`spherical_gaussian_model.py:407-426`) already calls `_prune_optimizer` which correctly indexes `exp_avg` / `exp_avg_sq` by the keep-mask. Adam state is preserved through that call. But `progressive_train.py:686` and `:632` call `gaussians.training_setup(opt)` **immediately after** the prune, which reconstructs the Adam optimizer from scratch — zero-momentum state. Same pattern after `reinitial_pts` at line 631-632.

**Tasks**:
1. Add a method `reset_densification_buffers(self)` on `SphericalGaussianModel` that resets only:
   - `self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")`
   - `self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")`
   - `self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")`
   It must NOT touch `self.optimizer`.
2. In `progressive_train.py`:
   - Line 686 (after first final-phase prune): replace `gaussians.training_setup(opt)` with `gaussians.reset_densification_buffers()`.
   - Lines 631-632 (after `reinitial_pts`): keep `training_setup(opt)` here only because `reinitial_pts` constructs new parameter tensors (different identities). Verify that case explicitly: read `reinitial_pts` (`spherical_gaussian_model.py`, search) and confirm — if `reinitial_pts` does in-place `nn.Parameter` swaps that orphan the optimizer's `params` references, `training_setup` is required there. Otherwise replace it too.
   - Inside `train_window` for `phase=="final"`, after the OptimizingSpa stop-iter prune (line 706-710) — that block doesn't currently call `training_setup`, but it does reset `optimizingSpa = None`. Confirm Adam state survives: `prune_points` was called at line 706, so it does. No change needed here.
3. Search the file for every occurrence of `training_setup` and verify whether each is "true reinit" (need full setup) or "post-prune reset" (use new helper).

**Test plan**:
- After change, `len(gaussians.optimizer.state)` should equal len(param_groups) before AND after a prune (was being recreated empty before).
- Loss curve at iters [3500, 4000, 5000] should be **lower** post-fix on the same data and seed (better convergence from preserved momentum).
- Final PSNR should match or beat baseline.

**Risk**: Low. If a param-group's params identity is re-bound (e.g. by `reinitial_pts`), Adam state's keying breaks silently — guard with an assert: each group's `params[0]` id must match a key in `optimizer.state` after the call. Print a warning if not.

---

### P2 — De-duplicate view-dir math in `compute_colors_precomp`

**Goal**: trivial CPU cycles + GPU allocation savings, every iter.

**File**: `scene/spherical_gaussian_model.py:228-295`.

**Tasks**:
1. The training and non-training branches both compute `view_dirs = self._xyz - viewpoint_camera.camera_center` followed by an `F.normalize`. Hoist this above the branch.
2. Verify with `git blame` that both branches actually need separate masks (look for differences beyond `out=` and `requires_grad`). If the only branch difference is the in-place clamp, unify: compute once, in-place clamp regardless of training.
3. Cache `view_dirs` on the model with an iteration tag: `self._cached_view_dirs = (cam_id, normalized_dirs)`. Clear at end of iteration. (Skip caching unless multiple consumers in same iter — currently only one, so just hoist.)

**Test**: PSNR within 0.001 dB of baseline. Visual check on first PLY.

**Risk**: low; pure refactor.

---

### P3 — Upgrade dense init quality

**Goal**: better seed point cloud → fewer iters needed for the same final quality.

**Files**:
- `configs/progressive.yaml`
- `scene/dense_init.py` (verify model loading paths support all sizes)

**Tasks**:
1. In `configs/progressive.yaml`:
   ```yaml
   dense_init:
     skip_first_n_snapshots: 0       # was 3
     target_dense_points_per_image: 100000   # was 50000
     dav2:
       model_size: base              # was small
   ```
2. Read `scene/dense_init.py:75-112` (`DepthAnythingV2Wrapper`). Confirm the repo map supports `"base"` — it does per the agent's earlier read. If `base` isn't in the map, add `"base": "depth-anything/Depth-Anything-V2-Base-hf"`.
3. **Memory check**: at `dense_init_for_new_images`, log VRAM before and after DAv2 context (already logged at lines 276, 339). With base, expect peak ~+390 MB during the context. If `accumulated_xyz` ever blows VRAM with 100k × 13 images (still only ~16 MB float32), no concern.
4. The `novelty_distance_threshold: 0.01` and `max_rejected_fraction: 0.5` defaults should still hold. If after upgrade the rejection rate spikes (look for the "sanity filter rejected too much" warning), bump `depth_disagreement_threshold` from 0.10 to 0.15 — DAv2-base is more accurate, so loosening the relative threshold doesn't make sense; instead inspect first failure case and decide.
5. Lowering `skip_first_n_snapshots` to 0 means snapshot 1 (first merge) gets dense init. The initial snapshot is already dense-inited per the recent commit `968aadd`. Confirm the merge-phase dense-init path's `min_sfm_points_for_alignment: 10` adequately rejects unreliable cameras.

**Test plan**:
- Run end-to-end. Compare:
  - Total wall-clock (DAv2-base loads slower; dense-init takes longer per snapshot).
  - PSNR at fixed iter budget (3000+4000) — expect +1 to +3 dB.
  - PSNR at reduced iter budget (try `iter_initial: 2000`, `iter_per_merge: 150`, `iter_final: 3000`).
- If quality matches at reduced budget, ship the reduced budget too.

**Risk**: low. DAv2-base is a drop-in upgrade. Watch the `[dense-init] WARNING: DAv2 leaked` message in `progressive_train.py:344-348`; if it fires with base, the leak is preexisting (small had it too) — diagnose separately.

---

### P4 — Smarter prune cadence

**Goal**: keep N_gaussians lower for more of the run, since iters/sec scales inversely with N.

**File**: `progressive_train.py`

**Tasks**:
1. Add to config (`configs/progressive.yaml`):
   ```yaml
   training:
     soft_cap_fraction: 0.85        # trigger emergency prune above this
     soft_cap_prune_ratio: 0.15
     prune_every_n_snapshots: 3     # was 5
   ```
2. In the merge loop (`progressive_train.py:810-876`), after `expand_gaussians_from_new_points` and `dense_init_for_new_images`:
   - If `gaussians._xyz.shape[0] > training_cfg.num_max * training_cfg.soft_cap_fraction`, call `lightweight_prune` immediately, with `prune_ratio = soft_cap_prune_ratio` (override). Don't wait for the next `prune_every_n_snapshots` boundary.
3. In `lightweight_prune` (`progressive_train.py:732-751`), accept an optional `prune_ratio` arg defaulting to `config.training.prune_ratio1`.
4. Track `n_gaussians` in the diary (`progressive_train.py:765, 538`). Plot it after a run to verify the cap holds.

**Test**: VRAM peak should drop. Iters/sec averaged over a multi-snapshot run should rise. Final PSNR should hold (lightweight_prune removes lowest-importance Gaussians).

**Risk**: medium. Aggressive pruning can hurt quality if `prune_every_n_snapshots: 3` triggers before new Gaussians have fully integrated. Mitigate by keeping the grace mask (already implemented at `model:939-952` and respected in `lightweight_prune` at `progressive_train.py:746`).

---

### P5 — Multi-view gradient accumulation (Python-side)

**Goal**: amortise per-iter Python + kernel-launch overhead over multiple cameras; produce a more stable gradient signal per optimiser step.

**File**: `progressive_train.py:464-722` (the `train_window` function).

**Background**:
The Grendel-GS / "Efficient Multi-view 3DGS Training" line of work shows that summing the loss over K views before `loss.backward()` and then taking one optimiser step gives convergence comparable to K separate steps, at lower wall-clock. For single-GPU with 800k Gaussians and a 1660 Ti, K=2 to K=4 is realistic given 4.5 GB headroom.

**Tasks**:
1. Add to config:
   ```yaml
   training:
     accumulation_views: 2     # 1 = original behaviour, 2-4 recommended
   ```
2. In the per-iteration loop body (`progressive_train.py:464-533`), wrap the camera-selection + forward + loss into an inner sub-loop of K views. Accumulate `loss = sum_i loss_i` (do **not** mean — Adam handles scale via `lr`). Single `loss.backward()`, single `optimizer.step()`.
3. Densification stats (`add_densification_stats`, `max_radii2D`) must be updated **for each** sub-iteration's `viewspace_point_tensor` and `radii` / `visibility_filter`. Because each forward creates a fresh `screenspace_points`, you must keep a list and aggregate after backward (gradients on each are valid after `loss.backward()` because Python keeps refs).
4. Loss-scaling for the merge-phase weighted loss (`progressive_train.py:514-518`): apply per-sub-iter weight before summation.
5. The `optimizing_spa.append_spa_loss` regulariser should be applied **once** per outer step, not per sub-iter — add it to the summed loss before backward.
6. Densify-and-prune intervals are based on `phase_iter` (the outer counter). Don't accidentally trigger densify K times per outer iter.
7. Diary logging: emit one line per outer iter listing the K cam names.

**Test plan**:
- With K=1, output must be bit-equivalent to baseline (run with same seed; verify `loss.item()` series matches).
- With K=2: VRAM peak should rise by ~`(intermediate_size_per_view)`, expect 200-500 MB higher peak.
- Wall-clock per outer iter: <2× single-view (Python overhead + one densify check amortised).
- Effective iters needed: at K=2, outer iter count to comparable PSNR should drop by ~30-50%.

**Risk**: medium. Grad-accum semantics differ subtly from per-view step (Adam's per-step normalisation). Validate by running with `iter_initial=3000, K=2` vs `iter_initial=6000, K=1` and comparing PSNR — should be within noise.

**Exit criteria**: at K=2, PSNR matches baseline at half the `iter_*` budgets, and wall-clock is <60% of baseline.

---

### P6 — Subsample cameras in `update_imp_score`

**Goal**: reduce SPA-window cost in final phase.

**File**: `progressive_train.py:154-170`.

**Tasks**:
1. Add config: `training.imp_score_camera_subsample: 0` (0 = all; >0 = top-K cameras by `prog_scene.image_weights` if available, else random).
2. Modify `update_imp_score(cameras, gaussians, ...)` to accept an optional `subsample_n` and weights array. If `subsample_n > 0` and len(cameras) > subsample_n, pick top-K by weight (fall back random).
3. Multiply the result by `len(cameras) / subsample_n` to keep the importance score on the same scale (so prune thresholds don't shift).
4. Default `0` (i.e. preserve current behaviour); user opts in via config.

**Test**: with subsample=8 on a 13-camera scene, prune behavior should be near-identical (importance ranks rarely change with 60% sample); wall-clock for SPA window should drop ~40%.

**Risk**: low.

---

### P7 — Skip merge iterations when no new content

**Goal**: don't burn 200 iters per merge when a snapshot adds zero new SfM points and no dense-init points were kept.

**File**: `progressive_train.py:810-876`.

**Tasks**:
1. After `expand_gaussians_from_new_points` and `dense_init_for_new_images`, count Gaussians-added-this-snapshot (`n_after - n_before` around both calls).
2. If new == 0 and `len(new_cams) == 0` (snapshot is just a pose update), skip the merge window entirely.
3. If new == 0 but `len(new_cams) > 0` (new poses but no new geometry), use a small fixed `min_merge_iters` (e.g. 50) instead of `iter_per_merge * len(new_cams)`.

**Test**: log `[snap N] merge iters: M`. Baseline always uses `200 × len(new_cams)`; new behaviour scales with content.

**Risk**: low.

---

### P8 — Persist DAv2 across snapshots

**Goal**: stop reloading the model per snapshot.

**File**: `progressive_train.py:263-391` and `scene/dense_init.py:75-112`.

**Background**:
Currently `dense_init_for_new_images` opens `with DepthAnythingV2Wrapper(...)` (line 279), which loads weights and exits / frees on each call. With DAv2-base, the load is ~15-30 s. Across N merge snapshots that's a lot of dead time.

**Tasks**:
1. Lift the `with` context to `progressive_training` (around `progressive_train.py:777`). Keep the model loaded for the entire run if `config.dense_init.enabled` is true.
2. Pass the model handle into `dense_init_for_new_images` as a new arg (or via `prog_scene.dav2_model`).
3. Free the model **before** the final-phase training begins (depth-init is not used in final). Add `del dav2_model; gc.collect(); torch.cuda.empty_cache()` before `train_window(phase="final", ...)`.
4. Verify VRAM impact: holding DAv2-base for the merge phase costs ~390 MB. If user is hitting memory pressure during merge, this competes with grace-period training. Add a config flag `dense_init.persist_model: true` (default true; set false to revert).

**Test**: log per-snapshot times before / after.

**Risk**: low if VRAM holds; medium if user reports OOMs during merge — fall back to per-snapshot context.

---

### P9 — Pool screenspace_points buffer

**Goal**: micro-optimisation; reduces allocator churn.

**File**: `spherical_gaussian_renderer/__init__.py`.

**Tasks**:
1. Add a module-level dict `_screenspace_pool: dict[int, torch.Tensor]` keyed by `pc.get_xyz.shape[0]`.
2. In each render fn, replace `screenspace_points = torch.zeros_like(pc.get_xyz, ...)` with a pool lookup; if missing or wrong size, allocate. **Do not** zero in place — gradients are accumulated additively, so `.zero_()` per call is required.
3. **Caveat**: the pool tensor needs `requires_grad=True` and `retain_grad()`, which means the gradient is reattached every iter — verify torch supports this (it does for leaf tensors; the tensor identity must persist between forward and `screenspace_points.grad` access in the training loop). If the rasterizer reads `means2D.grad` after backward, ensure `.grad` is cleared (`.grad = None`) before reuse.

**Test**: `loss.item()` series identical to baseline; `nvidia-smi` shows lower allocator peak.

**Risk**: low to medium. PyTorch's autograd engine is sensitive to tensor identity across backward calls — if the pooled tensor's `.grad` isn't properly cleared, subsequent iterations accumulate gradients incorrectly. Validate carefully.

**Recommendation**: do this last, it's worth ~1% and risks correctness bugs.

---

### P10 — True multi-view batched rasterizer (CUDA) — DEFERRED

**Why deferred**: Requires modifying `submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/`. SM 7.5 is a tier-1 architecture but kernel changes risk correctness regressions across forward/backward. The Python-side multi-view accumulation in P5 captures most of the gain at <5% the effort.

**If revisited**: the upstream repo to study is the "Efficient Multi-view 3DGS Training" CVPR'25 codebase, which adds a batched forward over K cameras with shared sort and tile assignment. That paper reports 1.4-1.6× speedup at K=4. For 6 GB VRAM the K=4 working set may not fit; profile first.

---

### P11 — SkipGS-style selective backward — DEFERRED

**Why deferred**: requires CUDA-kernel modification and the cited arXiv link `2603.08997` is malformed (looks like a 2026-03 future date). Confirm the actual paper reference with the user before any work. Lower-priority than P5.

---

### P12 — Convergence-driven scheduling (replaces parts of P4)

**Goal**: replace fixed iteration counts with content-driven decisions. Train each phase only as long as it's useful; let the Gaussian budget grow only as the scene demands; replace the 4000-iter "final refinement" with a fast compression pass.

**Files**:
- `progressive_train.py` (new module: `scene/convergence.py`)
- `configs/progressive.yaml`

**Background**:
The current schedule hard-codes `iter_initial: 3000`, `iter_per_merge: 200 * len(new_cams)`, `iter_final: 4000`, and `num_max: 800000` as a flat ceiling. None of these adapt to scene complexity. A simple scene wastes thousands of iters at the cap; a complex scene that could use more capacity hits the wall early.

The 4000-iter final phase is mostly compression (see §destructive-reset analysis): a 5 % importance prune, a destructive `reinitial_pts` reset, 2400 reconverge iters, a second 5 % prune, then SG axis culling. Net file-size reduction is ~10 %; net quality gain is small. This can collapse to seconds.

**Tasks**:

1. **Add `scene/convergence.py`** with a `ConvergenceMonitor` class:
   ```python
   from collections import deque

   class ConvergenceMonitor:
       def __init__(self, loss_window=200, densify_window=10):
           self.loss_history = deque(maxlen=loss_window)
           self.densify_history = deque(maxlen=densify_window)

       def update_loss(self, ema_loss: float):
           self.loss_history.append(ema_loss)

       def update_densify(self, n_added: int, n_total: int):
           self.densify_history.append(n_added / max(n_total, 1))

       def relative_slope(self) -> float:
           if len(self.loss_history) < self.loss_history.maxlen:
               return float('-inf')  # not enough data → assume improving
           recent = list(self.loss_history)
           slope = (recent[-1] - recent[0]) / len(recent)
           return slope / max(recent[-1], 1e-8)

       def densify_saturation(self) -> float:
           if not self.densify_history:
               return 0.0
           return max(self.densify_history)

       def state(self,
                 converged_slope=-1e-4,
                 active_densify=0.05,
                 active_slope=-1e-3) -> str:
           s = self.relative_slope()
           d = self.densify_saturation()
           if s > converged_slope and d < 0.01:
               return "converged"
           if d > active_densify and s < active_slope:
               return "wants_capacity"
           if s < active_slope:
               return "improving"
           return "stalled"
   ```
   Thresholds are starting heuristics. Log `relative_slope` and `densify_saturation` to the diary every 100 iter for one full run; pick thresholds where the curves visibly knee.

2. **Wire monitor updates into `train_window`** (`progressive_train.py:464-722`):
   - Construct `monitor = ConvergenceMonitor()` at top of `train_window`.
   - After `ema_loss = ...` (line 541): `monitor.update_loss(ema_loss)`.
   - After every densify call (lines 569 and 597): `monitor.update_densify(n_after - n_before, n_after)` — track the count yourself by reading `gaussians._xyz.shape[0]` before and after.

3. **Add early stopping per phase**:
   ```python
   min_iters = max(n_iters // 4, 200)  # never bail in first quarter
   for phase_iter in range(1, n_iters + 1):
       ...
       if phase_iter > min_iters and monitor.state() == "converged":
           logger.info(f"[{phase}] early stop at iter {phase_iter}/{n_iters}")
           break
   ```
   `iter_initial` / `iter_per_merge` / `iter_final` become *caps*, not targets.

4. **Add elastic Gaussian budget**. New config:
   ```yaml
   training:
     num_max_floor: 200000
     num_max_ceiling: 800000        # was: num_max
     num_max_step: 50000
     num_max_unlock_cooldown: 500
   ```
   In `train_window`:
   ```python
   n_current_cap = config.training.num_max_floor
   last_unlock_iter = 0
   ...
   # Inside iter loop, after densification stats update:
   if (monitor.state() == "wants_capacity"
           and phase_iter - last_unlock_iter > config.training.num_max_unlock_cooldown
           and n_current_cap < config.training.num_max_ceiling):
       n_current_cap = min(n_current_cap + config.training.num_max_step,
                           config.training.num_max_ceiling)
       last_unlock_iter = phase_iter
       logger.info(f"[budget] unlocked → {n_current_cap}")
   ```
   Replace every read of `training_cfg.num_max` (lines 552, 595) with `n_current_cap`. Remove `num_max` from config; treat `num_max_ceiling` as the hard VRAM-bounded ceiling.

5. **Add fast final refinement mode**. New config:
   ```yaml
   training:
     final_refinement_mode: fast     # "fast" | "full"
     fast_final_prune_ratio: 0.10
   ```
   New function in `progressive_train.py`:
   ```python
   def fast_final_compression(gaussians, prog_scene, opt, pipe, config, global_iter):
       """Single-shot importance prune + SG axis culling. Replaces 4000-iter final phase."""
       background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
       imp_score = update_imp_score(
           prog_scene.train_cameras, gaussians, pipe, background,
           imp_metric=config.training.imp_metric,
       )
       grace_mask = gaussians.get_grace_protected_mask(global_iter)
       ratio = config.training.fast_final_prune_ratio
       threshold = int(ratio * imp_score.shape[0])
       imp_sorted, _ = torch.sort(imp_score, 0)
       cutoff = imp_sorted[max(threshold - 1, 0)]
       prune_mask = (imp_score <= cutoff).squeeze() & ~grace_mask
       n_before = gaussians._xyz.shape[0]
       gaussians.prune_points(prune_mask)
       gaussians.cull_low_sharpness_axes(
           sharpness_threshold=config.training.sharpness_threshold)
       torch.cuda.empty_cache()
       logger.info(f"[fast-final] pruned {n_before} → {gaussians._xyz.shape[0]}, axes culled")
   ```
   In `progressive_training` (line 879), branch:
   ```python
   if config.training.final_refinement_mode == "fast":
       fast_final_compression(gaussians, prog_scene, opt, pipe, config, global_iter)
   else:
       gaussians.update_learning_rate(0)
       global_iter += train_window(..., phase="final", ...)
   ```

**Test plan**:
- **Step 1 (observability only)**: monitor enabled, no behaviour changes (still use fixed iter counts and flat cap). Log monitor state every 100 iter. Eyeball the curve to confirm convergence detection makes sense on your data.
- **Step 2**: enable early stopping with `min_iters = n_iters // 2`. Confirm wall-clock drops, PSNR holds.
- **Step 3**: enable elastic budget. Track peak `n_current_cap` reached on a complex vs simple scene; should differ. Confirm iters/sec stays high through early ramp.
- **Step 4**: enable `final_refinement_mode: fast`. Compare PSNR / SSIM / file size against `full` baseline on at least two scenes. If `fast` is within 0.3 dB PSNR and file size is comparable, ship it as default.

**Risk**:
- Monitor thresholds need real-data tuning (medium); ship with sane defaults plus logging so users can see when state transitions happen.
- Early stopping that fires too aggressively will hurt quality (medium); the `n_iters // 4` floor is the safety belt.
- Elastic budget interacts with P4's prune cadence (low if both use the monitor; medium otherwise). When P12 lands, **simplify P4**: drop the soft-cap-fraction logic; let `num_max_ceiling` be the hard bound and the monitor drive everything else.
- Fast final refinement loses the `reinitial_pts` reset effect (low-to-medium). Ablate per-scene.

**Effect on other briefs**:
- **Replaces** the soft-cap parts of P4. P4 becomes "tune `min_opacity` and `densify_grad_threshold`" only.
- **Composes** with P13 (smart triggers): the monitor state feeds the trigger predicates.

---

### P13 — Content-driven trigger predicates (replaces fixed cadences)

**Goal**: replace `phase_iter % densification_interval == 0` and `prune_every_n_snapshots` with predicates evaluated against per-Gaussian evidence and the convergence monitor. No hard-coded behaviour intervals; only cost-bounded check intervals.

**Files**:
- `scene/triggers.py` (new)
- `progressive_train.py` (replace cadence checks in `train_window` and `progressive_training`)

**Background — what fires what today**:

| Event | Selection (per-Gaussian) | Firing trigger (today) |
|-------|--------------------------|------------------------|
| `densify_and_clone` | grad ≥ threshold AND scaling ≤ percent_dense × extent | flat 100-iter cadence |
| `densify_and_split_mask` | (grad ≥ threshold AND scaling > percent_dense × extent) **OR** `mask_blur` (Gaussian covered > image_area/5000 in any view since last densify) | flat 100-iter cadence (paired with clone) |
| Opacity/size prune (inside `densify_and_prune_split`) | opacity < 0.005 OR oversize | paired with densify (no independent cadence) |
| `lightweight_prune` | bottom prune_ratio1 by importance | every N snapshots |

The selection logic is already evidence-based per Gaussian. Only the firing cadence is dumb.

**Shared substrate**:

```python
# scene/triggers.py

def evidence_settled(gaussians, min_obs=10, min_seen_fraction=0.5):
    """True iff per-Gaussian gradients have enough observations to act on."""
    denom = gaussians.denom.squeeze()
    seen = denom > 0
    if seen.sum().item() < min_seen_fraction * len(denom):
        return False
    return denom[seen].min().item() >= min_obs


def should_densify(gaussians, opt, mask_blur,
                   last_densify_iter, current_iter,
                   min_settling=50,
                   candidate_fraction=0.005,
                   max_interval=500):
    """Fire densify when per-Gaussian gradient evidence shows enough candidates,
    or as a stale-flush after max_interval iters."""
    if current_iter - last_densify_iter < min_settling:
        return False
    if current_iter - last_densify_iter > max_interval:
        return True  # force-flush stale grad accum

    if not evidence_settled(gaussians):
        return False
    grads = gaussians.xyz_gradient_accum / gaussians.denom.clamp(min=1)
    n_clone = (grads.squeeze() >= opt.densify_grad_threshold).sum().item()
    n_split = mask_blur.sum().item()
    n_total = gaussians._xyz.shape[0]
    return (n_clone + n_split) / max(n_total, 1) > candidate_fraction


def should_fast_prune(gaussians, opt, dead_fraction=0.02):
    """Fire opacity/size prune when enough dead splats have accumulated.
    Cheap; can be checked every iter."""
    min_op = getattr(opt, 'min_opacity_threshold', 0.005)
    dead = (gaussians.get_opacity < min_op).sum().item()
    return dead / gaussians._xyz.shape[0] > dead_fraction


def should_lightweight_prune(gaussians, monitor, n_at_last_prune,
                              soft_cap, growth_threshold=0.10):
    """Fire importance-prune on cap breach OR stagnation-with-growth.
    Expensive predicate (caller gates frequency)."""
    n = gaussians._xyz.shape[0]
    if n > soft_cap:
        return True
    grew = (n - n_at_last_prune) / max(n_at_last_prune, 1) > growth_threshold
    if grew and monitor.state() == "stalled":
        return True
    return False


def should_cull_sg_axes(gaussians, sharpness_threshold,
                        fraction_low=0.20):
    """Fire SG axis culling once enough axes are below sharpness threshold."""
    if gaussians.max_sg_degree == 0:
        return False
    sharpness = gaussians.get_sg_sharpness  # (N, K, 1)
    low_fraction = (sharpness < sharpness_threshold).float().mean().item()
    return low_fraction > fraction_low
```

**Tasks**:

1. **Decouple opacity/size prune from densify**. Add `gaussians.opacity_size_prune(min_opacity, max_screen_size, extent)` that runs only the prune block from `densify_and_prune_split` (`spherical_gaussian_model.py:804-809`), no clone/split. Keep `densify_and_prune_split` working for backward compat; add the new method.

2. **Replace cadence checks in `train_window`** (`progressive_train.py:566-604`):
   ```python
   # State carried across iters
   last_densify_iter = 0
   last_fast_prune_check = 0
   last_lightweight_prune_iter = 0
   n_at_last_prune = gaussians._xyz.shape[0]

   # Inside iter loop:

   # Cheap: every iter, but only acts on real evidence
   if should_fast_prune(gaussians, opt):
       gaussians.opacity_size_prune(
           min_opacity=getattr(opt, 'min_opacity_threshold', 0.005),
           max_screen_size=20 if phase == "initial" else None,
           extent=prog_scene.cameras_extent,
       )

   # Medium-cost predicate: check every 50 iter
   if (phase_iter % 50 == 0
           and gaussians._xyz.shape[0] < n_current_cap):
       if should_densify(gaussians, opt, mask_blur,
                         last_densify_iter, phase_iter):
           n_before = gaussians._xyz.shape[0]
           gaussians.densify_and_prune_split(
               opt.densify_grad_threshold, 0.005,
               prog_scene.cameras_extent,
               20 if phase == "initial" else None,
               mask_blur[:gaussians.xyz_gradient_accum.shape[0]],
           )
           n_after = gaussians._xyz.shape[0]
           monitor.update_densify(n_after - n_before, n_after)
           mask_blur = torch.zeros(gaussians._xyz.shape[0], device="cuda")
           last_densify_iter = phase_iter

   # Expensive predicate: check every 200 iter
   if phase_iter % 200 == 0 and phase != "initial":
       soft_cap = int(config.training.num_max_ceiling * 0.85)
       if should_lightweight_prune(gaussians, monitor,
                                    n_at_last_prune, soft_cap):
           lightweight_prune(gaussians, prog_scene, opt, config, global_iter)
           n_at_last_prune = gaussians._xyz.shape[0]
           last_lightweight_prune_iter = phase_iter

   # Late-phase compression: once when triggered
   if (phase == "final" and phase_iter > n_iters * 0.85
           and not has_culled
           and should_cull_sg_axes(gaussians, config.training.sharpness_threshold)):
       gaussians.cull_low_sharpness_axes(
           sharpness_threshold=config.training.sharpness_threshold)
       has_culled = True
   ```

3. **Replace `prune_every_n_snapshots` in `progressive_training`** (`progressive_train.py:872-873`):
   ```python
   # Before:
   # if snapshot_count % config.training.prune_every_n_snapshots == 0:
   #     lightweight_prune(...)

   # After: rely on the in-loop predicate to fire as needed.
   # Keep an explicit post-snapshot soft-cap check as a safety valve:
   soft_cap = int(config.training.num_max_ceiling * 0.85)
   if gaussians._xyz.shape[0] > soft_cap:
       lightweight_prune(gaussians, prog_scene, opt, config, global_iter)
   ```

4. **Add config knobs**:
   ```yaml
   training:
     densify_min_obs: 10
     densify_min_settling: 50
     densify_max_interval: 500
     densify_candidate_fraction: 0.005
     fast_prune_dead_fraction: 0.02
     lightweight_prune_growth_threshold: 0.10
     sg_axis_cull_low_fraction: 0.20
   ```
   Wire each as the corresponding predicate's optional arg.

5. **Logging**. Each predicate fire should emit a debug log line with the reason: which condition tripped (`evidence_settled`, candidate fraction, max_interval flush, soft cap, etc.). Crucial for tuning.

**Test plan**:

- **Phase 1 — predicate observability only**. Add the predicates and log their decisions at the existing fixed cadence checkpoints. Run a normal training session. For each old fixed-cadence fire, log whether the predicate would have fired. Build histograms of "predicate would fire" times vs "old fire" times. Confirm the predicate is at least as frequent as the old cadence at the start (or deliberately less, if the scene is converging fast).

- **Phase 2 — predicate-driven, with safety nets**. Replace the old cadence checks with predicates BUT keep `densify_max_interval` as a hard upper bound on dormancy. Run end-to-end. Compare:
  - Wall-clock per snapshot (should be similar or slightly faster).
  - Final PSNR / SSIM (should match within noise).
  - Total number of densify calls (should be lower for simple scenes, similar for complex scenes).
  - Total number of prune calls (should track densify calls + cap pressure).

- **Phase 3 — relax safety nets**. Once Phase 2 passes, bump `densify_max_interval` to 2000 (effectively disabled). Confirm no quality regression.

**Risk**:
- Predicate thresholds need tuning per scene type. Ship with conservative defaults; expose all as config.
- The "settling period" before evaluating evidence (`min_settling`) interacts with new Gaussian addition: after `expand_from_pcd`, freshly-added Gaussians have `denom = 0`, which trips `evidence_settled`'s "too many fresh" guard. This is correct behaviour — wait until they've been observed — but means the first densify after a snapshot expansion may be delayed. Acceptable.
- Decoupling opacity/size prune from densify changes the loop semantics slightly: previously a Gaussian could be cloned and then immediately pruned in the same call (if the clone landed below opacity threshold). Now those run separately. Validate no quality regression; if there is one, run opacity_size_prune immediately after densify call as a hot-fix.

**Composition with other briefs**:
- **Requires P12** (`monitor` is a P13 input). Land P12 first.
- **Subsumes P4**'s soft-cap logic. With P13, P4 reduces to "tune `min_opacity` to 0.01 in config" — a one-line change.
- **Independent of** P1, P2, P3, P5, P6, P7, P8 (all orthogonal).

---

## 5. Suggested Execution Order

For a Sonnet instance picking this up:

1. **Day 1**: P1 (Adam preservation) and P2 (view-dir dedup) — both small, validate independently.
2. **Day 1**: P3 (config-only DAv2 upgrade) — baseline measurement.
3. **Day 2**: P12 (convergence monitor + elastic budget + early stopping + fast final refinement) — largest qualitative shift in how training is scheduled. Land in four sub-steps with monitor-only observability first, then incremental enables.
4. **Day 2-3**: P13 (content-driven trigger predicates) — depends on P12's monitor. Validate per Phase 1/2/3 plan in the brief.
5. **Day 3**: P5 (multi-view accumulation) — largest single iters/sec win; needs careful K=1 regression check.
6. **Day 3**: P6, P7, P8 — small wall-clock savings, easy.
7. **Day 4**: P4 (now reduced to: tune `min_opacity` to 0.01 once P12 + P13 land).
8. **Day 4**: P9 (screenspace pool) — only if profile shows allocator pressure.

**Note on P4**: once P12 and P13 land, P4's soft-cap logic is subsumed by the elastic budget and the cap-driven `should_lightweight_prune` predicate. P4 becomes a one-line config tweak.

After all of P1, P2, P3, P5, P6, P7, P8, P12, P13 land, expected cumulative improvement: 2-3× wall-clock to comparable-quality scenes, with quality-per-iter also improving from P3 (better init), P5 (better gradient signal), and P12 (no over-training).

## 6. Validation Harness

Before any of the above, add a small script `scripts/bench_progressive.py` (not present today):

- Runs `progressive_train.py` with `--quiet`, captures wall-clock per phase and final Gaussian count.
- Parses `progressive_diary.txt` for camera weights / iter counts.
- Computes PSNR vs ground-truth held-out cameras (re-use logic from `metrics.py:1-end`).
- Emits a JSON summary: `{phase: {wall_s, peak_vram_mb}, final: {n_gauss, psnr, ssim}}`.

Each P-change should be benched against a fixed seed and the same snapshot data; commit results to `bench/` for regression tracking.
