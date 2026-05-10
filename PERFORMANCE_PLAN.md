# Progressive MEGS-2 Performance Plan

**Target system**: GTX 1660 Ti (Turing SM 7.5), 6 GB VRAM, Linux, CUDA 12.1.
**Workload**: streaming progressive ingestion — one new phone-camera image every ~12 s; ~13 images / ~800k splats steady-state; ~1.5 GB VRAM utilised → ~4.5 GB headroom.
**Branch**: `claude/optimize-low-vram-performance-ECpjf` atop `incremental`.

This document is a clean, self-contained implementation roadmap. Each task brief stands alone; you do not need to read the rest of the document to implement any one of them.

---

## Two Bottleneck Families

1. **Optimizer-state churn**. `training_setup(opt)` is called after every prune (`progressive_train.py:632, 686`) and after `reinitial_pts`. This re-creates the Adam optimiser from scratch — discarding the `exp_avg` / `exp_avg_sq` buffers that `_prune_optimizer` (`scene/spherical_gaussian_model.py:389-405`) had just carefully pruned-in-place. Every prune resets thousands of iterations of momentum.

2. **Single-view forward/backward**. Every iteration renders one camera, computes loss, backprops. With 800k Gaussians the per-call rasterizer kernel-launch overhead and the per-call `screenspace_points` allocation (~9.6 MB, `spherical_gaussian_renderer/__init__.py:72`) dominate at this scale on a 1660 Ti.

A third axis — initialization quality — is the cheapest way to **reduce iters needed**: DAv2-small at 50k points/image is conservative for 4.5 GB free VRAM. Going larger typically halves depth error and reduces required `iter_initial` and `iter_per_merge`.

A fourth axis — the schedule itself — is the largest qualitative win: hard-coded iter counts and a flat splat ceiling cause both over-training on simple scenes and capacity starvation on complex ones.

---

## Implementation Order

1. **T1**: Preserve Adam state across prunes — 1 file edit, 1 line of risk.
2. **T2**: De-dup view-dir math in `compute_colors_precomp` — pure refactor.
3. **T3**: Upgrade DAv2 to `large`, dense init from snapshot 0.
4. **T4**: DAv2 persistence with explicit on/off flag (default OFF).
5. **T5**: Camera subsample in `update_imp_score` (opt-in, default off).
6. **T6**: Multi-view gradient accumulation (Python-side).
7. **T7**: **Convergence-driven scheduling** — replaces hard-coded iter counts and the flat splat cap with monitor + content-driven trigger predicates. Single VRAM-bounded ceiling; special cap-bound regime; fast-finish compression.
8. **T8**: SkipGS — view-adaptive backward gating in late-phase iterations (Python-only).

**Deferred**: T9 (CUDA-batched rasterizer) — revisit after T6 + T8 land and you can profile Python vs kernel overhead.

**Skipped**: see end of doc for what was dropped and why.

---

## T1 — Preserve Adam state across prunes

**Files**: `scene/spherical_gaussian_model.py`, `progressive_train.py`.

**Background**:
`prune_points` (`spherical_gaussian_model.py:407-426`) calls `_prune_optimizer` which correctly indexes `exp_avg` / `exp_avg_sq` by the keep-mask. Adam state survives that call. But `progressive_train.py:686` and `:632` call `gaussians.training_setup(opt)` **immediately after**, reconstructing the Adam optimizer from scratch with zero-momentum state. Same pattern after `reinitial_pts` at line 631-632.

**Tasks**:

1. Add a method `reset_densification_buffers(self)` on `SphericalGaussianModel` that resets only:
   - `self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")`
   - `self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")`
   - `self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")`

   It must NOT touch `self.optimizer`.

2. In `progressive_train.py`:
   - Line 686 (after first final-phase prune): replace `gaussians.training_setup(opt)` with `gaussians.reset_densification_buffers()`.
   - Lines 631-632 (after `reinitial_pts`): inspect `reinitial_pts` first. If it constructs new `nn.Parameter` tensors (different identities), `training_setup` is required there because Adam state's keying breaks with new tensor identities. If it does in-place updates only, replace with the helper.
   - Inside `train_window` for `phase=="final"`, the OptimizingSpa stop-iter prune block (line 706-710) doesn't currently call `training_setup`; verify Adam state survives — it does, `prune_points` was called at line 706. No change.

3. Search the file for every occurrence of `training_setup` and classify each: "true reinit" (need full setup) vs "post-prune reset" (use new helper).

4. Add a guard: after every `training_setup` call site that you keep, assert each param-group's `params[0]` `id()` matches a key in `optimizer.state`. Print a warning on mismatch.

**Test**:
- After change, `len(gaussians.optimizer.state)` should equal `len(param_groups)` before AND after a prune (was being recreated empty before).
- Loss curve at iters [3500, 4000, 5000] should be **lower** post-fix on the same data and seed.
- Final PSNR should match or beat baseline.

---

## T2 — De-duplicate view-dir math in `compute_colors_precomp`

**File**: `scene/spherical_gaussian_model.py:228-295`.

**Tasks**:

1. The training and non-training branches both compute `view_dirs = self._xyz - viewpoint_camera.camera_center` followed by an `F.normalize`. Hoist these two lines above the branch.
2. Verify with `git blame` that both branches actually need separate masks (look for differences beyond `out=` and `requires_grad`). If the only branch difference is the in-place clamp, unify: compute view_dirs once and let both paths read it.
3. Do **not** add caching across iterations — only one consumer per iter.

**Test**: PSNR within 0.001 dB of baseline on a fixed-seed run. Visual check on first PLY.

---

## T3 — Upgrade dense init quality

**Files**: `configs/progressive.yaml`, `scene/dense_init.py` (verify model loading paths support `large`).

**Tasks**:

1. In `configs/progressive.yaml`:
   ```yaml
   dense_init:
     skip_first_n_snapshots: 0           # was 3
     target_dense_points_per_image: 100000   # was 50000
     dav2:
       model_size: large                 # was small; user has 4.5 GB headroom
   ```

2. Read `scene/dense_init.py:75-112` (`DepthAnythingV2Wrapper`). Confirm the repo map supports `"large"`. If not, add: `"large": "depth-anything/Depth-Anything-V2-Large-hf"`.

3. **Memory check**: at `dense_init_for_new_images`, log VRAM before and after the DAv2 context (already logged at lines 276, 339). With `large`, expect peak ~+1.3 GB during the context. With training scene at ~1.5 GB resident, peak should land near 2.8 GB — well under the 6 GB cap.

4. The `novelty_distance_threshold: 0.01` and `max_rejected_fraction: 0.5` defaults should still hold. If the rejection rate spikes after upgrade (look for the "sanity filter rejected too much" warning), inspect the first failure case before tuning thresholds — `large` is more accurate, so the threshold likely doesn't need to move.

5. `skip_first_n_snapshots: 0` means snapshot 1 (first merge) gets dense init. The initial snapshot is already dense-inited per commit `968aadd`. Confirm the merge-phase dense-init path's `min_sfm_points_for_alignment: 10` adequately rejects unreliable cameras.

**Test**:
- Run end-to-end. Compare:
  - Total wall-clock (DAv2-large loads slower; dense-init takes longer per snapshot).
  - PSNR at fixed iter budget (3000+4000) — expect +1-3 dB.
  - PSNR at reduced iter budget (try `iter_initial: 2000`, `iter_per_merge: 150`, `iter_final: 3000`).
- If quality matches at the reduced budget, ship the reduced budget too.

**Watch out**: the `[dense-init] WARNING: DAv2 leaked` message in `progressive_train.py:344-348`. If it fires with `large`, the leak is preexisting (small had it too) — diagnose separately.

---

## T4 — DAv2 persistence (opt-in)

**Files**: `progressive_train.py:263-391`, `scene/dense_init.py:75-112`, `configs/progressive.yaml`.

**Background**:
Currently `dense_init_for_new_images` opens `with DepthAnythingV2Wrapper(...)` (line 279) per call, which loads weights and exits / frees on each call. With DAv2-large the load is ~30+ s. With user's streaming workflow (one new photo every ~12 s), persisting saves ~30 s per snapshot but holds ~1.3 GB of VRAM continuously between snapshots.

User wants this **off by default** and explicitly opt-in.

**Tasks**:

1. Add config:
   ```yaml
   dense_init:
     persist_model: false                # default off; set true to hold DAv2 across snapshots
   ```

2. In `progressive_training` (around `progressive_train.py:777`), branch:
   - If `config.dense_init.persist_model`: open `DepthAnythingV2Wrapper(...)` once, keep handle on `prog_scene.dav2_model` (or pass through), close before final phase begins.
   - Else: leave the per-call `with` context as-is.

3. When persisted, `dense_init_for_new_images` accepts the held handle as an arg instead of constructing its own context.

4. Free the model **before** final-phase training starts:
   ```python
   if config.dense_init.persist_model and prog_scene.dav2_model is not None:
       del prog_scene.dav2_model
       prog_scene.dav2_model = None
       gc.collect()
       torch.cuda.empty_cache()
   ```

5. Log per-snapshot dense-init time and VRAM peak in the diary so the user can compare modes empirically.

**Test**:
- `persist_model: false` (default): behaviour identical to current.
- `persist_model: true`: per-snapshot dense-init time drops by the model load cost; VRAM idle between snapshots is ~1.3 GB higher.
- Confirm no OOM at any point with `persist_model: true` on a long run.

---

## T5 — Camera subsample in `update_imp_score`

**File**: `progressive_train.py:154-170`.

**Tasks**:

1. Add config: `training.imp_score_camera_subsample: 0` (0 = all, current behaviour; >0 = top-K by `prog_scene.image_weights` if available, else random).

2. Modify `update_imp_score(cameras, gaussians, ...)` to accept an optional `subsample_n` and weights array. If `subsample_n > 0` and `len(cameras) > subsample_n`, pick top-K by weight (fall back to random).

3. Multiply the result by `len(cameras) / subsample_n` to keep the importance score on the same scale (so prune thresholds don't shift).

**Test**: with `subsample=8` on a 13-camera scene, prune behaviour should be near-identical (importance ranks rarely change at 60% sample); SPA-window wall-clock should drop ~40%.

**Note**: with 13 cameras the absolute saving is small. Worth it as a one-line opt-in; not worth fighting over.

---

## T6 — Multi-view gradient accumulation (Python-side)

**File**: `progressive_train.py:464-722` (the `train_window` function).

**Background**:
Summing the loss over K views before `loss.backward()` and taking one optimiser step gives convergence comparable to K separate steps at lower wall-clock (amortises Python overhead, kernel launch, and the densify-check across K renders). For 800k Gaussians on a 1660 Ti with 4.5 GB headroom, K=2 to K=4 is realistic.

**Tasks**:

1. Add config:
   ```yaml
   training:
     accumulation_views: 2     # 1 = original behaviour; 2-4 recommended
   ```

2. In the per-iter loop body (`progressive_train.py:464-533`), wrap camera-selection + forward + loss into an inner sub-loop of K views. Accumulate `loss = sum_i loss_i` (do **not** mean — Adam handles scale via `lr`). Single `loss.backward()`, single `optimizer.step()`.

3. Densification stats (`add_densification_stats`, `max_radii2D`) must be updated **for each** sub-iteration's `viewspace_point_tensor`, `radii`, `visibility_filter`. Each forward creates a fresh `screenspace_points`; keep refs in a list and aggregate after backward (gradients on each are valid because Python keeps refs).

4. Loss-scaling for the merge-phase weighted loss (`progressive_train.py:514-518`): apply per-sub-iter weight before summation.

5. The `optimizing_spa.append_spa_loss` regulariser is applied **once** per outer step, not per sub-iter — add it to the summed loss before backward.

6. Densify-and-prune intervals are based on `phase_iter` (the outer counter). Don't accidentally trigger densify K times per outer iter.

7. Diary logging: emit one line per outer iter listing the K cam names.

**Test**:
- With K=1, output must be bit-equivalent to baseline (run with same seed; `loss.item()` series matches).
- With K=2: VRAM peak should rise by ~`(intermediate_size_per_view)`, expect 200-500 MB higher peak.
- Wall-clock per outer iter: <2× single-view (Python overhead + one densify check amortised).
- Effective iters needed: at K=2, outer iter count to comparable PSNR should drop by ~30-50%.
- Validate by running `iter_initial=3000, K=2` vs `iter_initial=6000, K=1`; PSNR should be within noise.

**Exit criteria**: at K=2, PSNR matches baseline at half the `iter_*` budgets, and wall-clock is <60% of baseline.

---

## T7 — Convergence-driven scheduling

**Goal**: replace fixed iteration counts (`iter_initial`, `iter_per_merge`, `iter_final`) and the flat splat cap (`num_max`) with convergence signals and content-driven trigger predicates. One hard splat ceiling = the VRAM-bounded maximum. Special cap-bound regime when actually pressing against it. Otherwise everything is driven by per-Gaussian evidence and loss-slope state.

**Files**:
- New: `scene/convergence.py` (monitor)
- New: `scene/triggers.py` (trigger predicates)
- Modified: `progressive_train.py` (replace cadence checks, replace fixed phase lengths)
- Modified: `configs/progressive.yaml`

### Background — what fires what today

| Event | Selection (per-Gaussian) | Firing trigger (today) |
|-------|--------------------------|------------------------|
| `densify_and_clone` | grad ≥ threshold AND scaling ≤ percent_dense × extent | flat 100-iter cadence |
| `densify_and_split_mask` | (grad ≥ threshold AND scaling > percent_dense × extent) **OR** `mask_blur` (covered > image_area/5000 in any view since last densify) | flat 100-iter cadence (paired with clone) |
| Opacity/size prune | opacity < 0.005 OR oversize | paired with densify (no independent cadence) |
| `lightweight_prune` | bottom prune_ratio1 by importance | every N snapshots |
| Phase termination | n/a | hard-coded iter count |

The **selection logic is already evidence-based per Gaussian**. Only the firing cadence and the phase length are dumb. T7 fixes the cadence and the phase length.

### Design

- **One splat cap**: `num_max_ceiling` = the VRAM-bounded maximum. No floor, no step, no ramp. Below it, growth is free and driven by evidence. At/near it, you switch into a cap-bound regime.

- **Convergence monitor**: tracks loss slope and densify saturation. State machine returns one of `improving | wants_capacity | stalled | converged`.

- **Phase termination**: phase ends when monitor reports `converged`. `iter_initial` / `iter_per_merge` / `iter_final` become *caps*, not targets — a safety belt against pathological cases. There is no early-stop floor; convergence detection is responsible for not bailing too early (the loss-window length is the floor).

- **Trigger predicates** decide when densify, fast-prune, lightweight-prune, and SG axis cull fire. They are evaluated at cheap cadences; the actions run only when evidence justifies.

- **Cap-bound regime**: when `n_xyz > soft_cap` (e.g. 0.85 × ceiling):
  - Densify is throttled (gated on `n_xyz < ceiling`).
  - Prune is more aggressive (`should_lightweight_prune` fires on cap breach without waiting for stall).
  - This is the only place we keep "schedule" logic — everywhere else is convergence-driven.

- **Fast final compression**: once final phase converges, the historical 4000-iter "final refinement" was mostly compression (5% importance prune, destructive `reinitial_pts` reset, 2400 reconverge iters, second 5% prune, SG axis cull) for ~10% file-size shrink and small quality gain. Replace with a single-shot importance prune + axis cull.

### Implementation

**1. `scene/convergence.py`**:

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

Thresholds are starting heuristics. **Phase 1 of the test plan** (below) calibrates them.

**2. `scene/triggers.py`**:

```python
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

**3. Decouple opacity/size prune from densify**.

Add `gaussians.opacity_size_prune(min_opacity, max_screen_size, extent)` that runs only the prune block from `densify_and_prune_split` (`spherical_gaussian_model.py:804-809`) — no clone/split. Keep `densify_and_prune_split` working for backward compat; the new method runs the cheap prune independently.

**4. Replace cadence checks in `train_window`** (`progressive_train.py:566-604`):

```python
# State carried across iters (init at top of train_window):
monitor = ConvergenceMonitor()
last_densify_iter = 0
last_lightweight_prune_iter = 0
n_at_last_prune = gaussians._xyz.shape[0]
has_culled = False
soft_cap = int(config.training.num_max_ceiling * 0.85)

# Inside iter loop:

# 0. Update monitor
monitor.update_loss(ema_loss)

# 1. Cheap, every iter
if should_fast_prune(gaussians, opt):
    gaussians.opacity_size_prune(
        min_opacity=getattr(opt, 'min_opacity_threshold', 0.005),
        max_screen_size=20 if phase == "initial" else None,
        extent=prog_scene.cameras_extent,
    )

# 2. Medium-cost predicate, every 50 iter, gated by cap
if (phase_iter % 50 == 0
        and gaussians._xyz.shape[0] < config.training.num_max_ceiling):
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

# 3. Expensive predicate, every 200 iter
if phase_iter % 200 == 0 and phase != "initial":
    if should_lightweight_prune(gaussians, monitor,
                                n_at_last_prune, soft_cap):
        lightweight_prune(gaussians, prog_scene, opt, config, global_iter)
        n_at_last_prune = gaussians._xyz.shape[0]
        last_lightweight_prune_iter = phase_iter

# 4. Late-phase compression, once when triggered
if (phase == "final" and phase_iter > n_iters * 0.85
        and not has_culled
        and should_cull_sg_axes(gaussians, config.training.sharpness_threshold)):
    gaussians.cull_low_sharpness_axes(
        sharpness_threshold=config.training.sharpness_threshold)
    has_culled = True

# 5. Convergence-driven phase termination (no explicit floor;
#    monitor's loss_window length acts as the floor)
if monitor.state() == "converged":
    logger.info(f"[{phase}] converged at iter {phase_iter}/{n_iters}")
    break
```

**5. Replace `prune_every_n_snapshots` in `progressive_training`** (`progressive_train.py:872-873`):

```python
# Rely on the in-loop predicate during train_window. Keep an
# explicit post-snapshot cap-breach check as a safety valve:
if gaussians._xyz.shape[0] > soft_cap:
    lightweight_prune(gaussians, prog_scene, opt, config, global_iter)
```

**6. Fast final compression**. Add to `progressive_train.py`:

```python
def fast_final_compression(gaussians, prog_scene, opt, pipe, config, global_iter):
    """Single-shot importance prune + SG axis culling. Runs after the
    convergence-driven final phase exits."""
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
    logger.info(f"[fast-final] {n_before} → {gaussians._xyz.shape[0]}, axes culled")
```

The final phase still runs (convergence-driven) for the actual quality refinement; `fast_final_compression` runs **after** it returns, replacing the historical destructive-reset pass.

**7. Config changes**:

```yaml
training:
  # Single hard ceiling (VRAM-bounded). No floor/step/ramp.
  num_max_ceiling: 800000          # was: num_max

  # Iter caps — safety belt only; convergence drives termination.
  iter_initial: 3000               # cap, not target
  iter_per_merge: 200              # cap, not target
  iter_final: 4000                 # cap, not target

  # Trigger predicate knobs (defaults are sane; expose for tuning)
  densify_min_obs: 10
  densify_min_settling: 50
  densify_max_interval: 500
  densify_candidate_fraction: 0.005
  fast_prune_dead_fraction: 0.02
  lightweight_prune_growth_threshold: 0.10
  sg_axis_cull_low_fraction: 0.20

  # Fast final compression
  fast_final_prune_ratio: 0.10
```

Remove `num_max`, `prune_every_n_snapshots`, `densification_interval`. Replace every read of `training_cfg.num_max` (lines 552, 595) with `config.training.num_max_ceiling`.

### Test plan (phased rollout)

**Phase 1 — observability only**. Land monitor + predicates. Don't change behaviour: keep fixed iter counts, keep flat cadence. At each fixed-cadence fire, log whether the predicate would have fired and what `monitor.state()` is. Run a normal end-to-end. Build histograms of "predicate would-fire" times vs "old-fire" times. Calibrate thresholds where the curves visibly knee.

**Phase 2 — predicate-driven, with safety nets**. Replace fixed cadences with predicates. Keep `densify_max_interval=500` as a hard upper bound on dormancy. Keep `iter_*` as caps; phase termination still uses `phase_iter == n_iters` as a hard cap, but `monitor.state() == "converged"` ends earlier. Run end-to-end. Compare:
- Wall-clock per snapshot (similar or faster).
- Final PSNR / SSIM (matches within noise).
- Total densify/prune calls (lower for simple scenes; similar for complex).

**Phase 3 — relax safety nets**. Bump `densify_max_interval=2000` (effectively disabled). Ship `fast_final_compression` enabled by default. Confirm no quality regression on at least two scenes.

### Risk & gotchas

- **Threshold calibration** (medium). Phase 1 logging is the safety belt. Ship conservative defaults; expose all knobs.

- **`evidence_settled` after expansion** (low). Freshly-added Gaussians have `denom = 0`, so `evidence_settled`'s "too many fresh" guard trips → first densify after `expand_from_pcd` is delayed. This is correct behaviour but worth noting.

- **Decoupled opacity/size prune** (low). Previously a Gaussian could be cloned and immediately pruned in the same `densify_and_prune_split` call. Now those are separate. If you see a quality regression, run `opacity_size_prune` immediately after each densify call as a hot-fix.

- **Fast final compression** loses the `reinitial_pts` reset (low–medium). The reset was destructive and cost 2400 reconverge iters; the gain was small. Ablate per-scene if quality drops more than 0.3 dB.

- **Cap-bound regime correctness** (low). When `n_xyz` is just under the ceiling, densify is gated off entirely. If the scene still wants capacity, pruning has to fire to make room. The `should_lightweight_prune` "cap breach" branch handles this. Verify with a deliberately-low ceiling on a complex scene that the system finds an equilibrium rather than thrashing.

---

## T8 — SkipGS: view-adaptive backward gating

**Reference**: Li, Lee, Fan. *SkipGS: Post-Densification Backward Skipping for Efficient 3DGS Training.* arXiv:2603.08997, March 2026. The id parses as a 2026-03 paper because it is one — verified.

**Goal**: in late-phase iterations, when per-view loss has stabilised, skip the backward pass for views whose current loss is at or below their recent EMA baseline. Forward always runs; the EMA is always updated; only `loss.backward()` and the optimiser step are gated.

This is **Python-only** — a conditional around `loss.backward()`. No renderer/representation/loss changes. Plug-and-play with everything else in this plan.

**Files**:
- New: `scene/skipgs.py` (gate state + decision logic)
- Modified: `progressive_train.py` (wire the gate into the train loop)
- Modified: `configs/progressive.yaml`

### Mechanism (from paper, §4)

For each training view `v`, maintain an EMA of observed loss:
```
L̄_v(t) = β · L̄_v(t-1) + (1 - β) · L_v(t)
```
Decision: `g = 1[s > 1]` where `s = L_v(t) / (L̄_v(t-1) + ε)`. If `s > 1` the view's loss has risen above its baseline — backward fires. Otherwise the view is performing at-or-better-than baseline and backward is a candidate to skip.

Two safety mechanisms:

1. **Warmup**. For the first `W` iterations after the gate is enabled, backward always fires (`g=1`), but the gate's prospective decision is recorded. This populates per-view EMAs and produces calibration statistics.

2. **Budget floor**. After warmup, calibrate `ρ_min` once from the warmup-window prospective decisions: `ρ_min = ρ_lo + (1 - ρ_lo) · ρ̂_W` with `ρ_lo = 0.5`. During gating, track the cumulative backward ratio `ρ_cum`; if it falls below `ρ_min`, force `g = 1` for the next iteration regardless of deviation.

Paper's hyperparameters (used across all benchmarks; no per-scene tuning): `W = 500`, `β = 0.95`, `ε = 1e-8`, `ρ_lo = 0.5`.

### Activation regime in progressive MEGS-2

Vanilla 3DGS has a clean "post-densification" boundary at iter `T_d = 15k`. Progressive MEGS-2 doesn't — densification fires throughout under T7's predicates. The mapping that works:

- **Activate SkipGS only in `phase == "final"`**, starting after a phase-local warmup of 500 iters.
- **Do not activate during initial or merge phases**: those are the regimes where new geometry is being added and per-view losses are non-stationary; the EMA is misleading there.
- **Optional follow-up**: once T7 lands and reports `monitor.state()` reliably, also enable SkipGS during late-merge windows when `monitor.state() in {"stalled", "improving"}` for ≥100 iters and densify hasn't fired recently. Defer this until T8's main scope is shipped and validated.

### Interaction with T6 (multi-view accumulation)

When T6 is enabled (K > 1 views per outer step), the gate has to decide what "skipping" means. Use **per-view inclusion in the gradient sum**:

- Forward all K sampled views every step; update each view's EMA every step (unconditional).
- Compute `s_k` for each. Build the gradient-bearing loss as `loss = Σ_{k : s_k > 1} loss_k` (i.e., only views above their baseline contribute gradient).
- If the set is empty, this is a "fully-skipped step": no backward, no optimiser step.
- Budget control runs at the **outer step level**: `ρ_cum = (# steps that called backward) / (# outer steps)`. If `ρ_cum < ρ_min`, force backward on the next step using the full `Σ_k loss_k`.

If T6 is not enabled (K=1), the implementation degenerates to the paper's algorithm verbatim.

### Tasks

1. **`scene/skipgs.py`**:
   ```python
   class SkipGSGate:
       def __init__(self, warmup=500, beta=0.95, eps=1e-8, rho_lo=0.5):
           self.warmup = warmup
           self.beta = beta
           self.eps = eps
           self.rho_lo = rho_lo
           self._ema = {}                 # cam_id -> float EMA
           self._step = 0                 # outer steps since enable
           self._backward_count = 0       # outer steps where backward fired
           self._warmup_would_fire = 0    # warmup iters where s > 1 would have fired
           self._warmup_evaluable = 0     # warmup iters where view had EMA history
           self._rho_min = None           # set once at end of warmup

       def deviation(self, cam_id: int, loss_value: float) -> float:
           if cam_id not in self._ema:
               return float('inf')        # no history → treat as high deviation
           return loss_value / (self._ema[cam_id] + self.eps)

       def update_ema(self, cam_id: int, loss_value: float):
           prev = self._ema.get(cam_id)
           if prev is None:
               self._ema[cam_id] = loss_value
           else:
               self._ema[cam_id] = self.beta * prev + (1 - self.beta) * loss_value

       def decide(self, deviation_scores: list[float]) -> tuple[list[bool], bool]:
           """Returns (per_view_gate, force_backward).
           per_view_gate[k] = True means view k contributes to the gradient sum.
           force_backward = True overrides per_view_gate to all-True for budget."""
           self._step += 1
           proposals = [s > 1.0 for s in deviation_scores]

           # Warmup: backward always fires; record stats for calibration
           if self._step <= self.warmup:
               for s, p in zip(deviation_scores, proposals):
                   if s != float('inf'):
                       self._warmup_evaluable += 1
                       if p:
                           self._warmup_would_fire += 1
               return [True] * len(proposals), True

           # First post-warmup step: calibrate rho_min once
           if self._rho_min is None:
               rho_hat_w = (self._warmup_would_fire
                            / max(self._warmup_evaluable, 1))
               self._rho_min = self.rho_lo + (1 - self.rho_lo) * rho_hat_w

           # Budget check (use ratio BEFORE this step's decision)
           rho_cum = self._backward_count / max(self._step - 1, 1)
           if rho_cum < self._rho_min:
               return [True] * len(proposals), True

           # Normal gating: include views with s > 1
           gate = [p for p in proposals]
           return gate, any(gate)

       def record_backward(self, fired: bool):
           if fired:
               self._backward_count += 1
   ```

2. **Wire into `train_window`** (`progressive_train.py`):

   At top of `train_window`, after argument parsing:
   ```python
   skipgs = None
   if (config.training.skipgs.enabled
           and phase == config.training.skipgs.phase):  # "final" by default
       skipgs = SkipGSGate(
           warmup=config.training.skipgs.warmup,
           beta=config.training.skipgs.beta,
           eps=config.training.skipgs.eps,
           rho_lo=config.training.skipgs.rho_lo,
       )
   ```

   In the per-iter loop, replace the existing forward + loss + backward block with:
   ```python
   # Forward all K sub-iter views (or K=1 if T6 disabled)
   per_view_packs = []  # list of (loss_tensor, render_pkg, cam_id)
   for sub in range(K):
       cam = pick_camera(...)
       render_pkg = render(cam, gaussians, pipe, background)
       loss_k = compute_loss(render_pkg, cam, ...)  # existing logic
       per_view_packs.append((loss_k, render_pkg, cam.uid))

   # Always update EMA from forward observation
   if skipgs is not None:
       for loss_t, _, cam_id in per_view_packs:
           skipgs.update_ema(cam_id, float(loss_t.detach()))

   # Decide which views contribute to the gradient
   if skipgs is not None:
       devs = [skipgs.deviation(cam_id, float(loss_t.detach()))
               for loss_t, _, cam_id in per_view_packs]
       gate, will_backward = skipgs.decide(devs)
   else:
       gate = [True] * K
       will_backward = True

   contributing = [loss_t for (loss_t, _, _), g in zip(per_view_packs, gate) if g]

   if not contributing:
       # Fully skipped step
       if skipgs is not None:
           skipgs.record_backward(False)
       # Visibility-side densify stats can still run from forward
       update_densify_stats_visibility_only(per_view_packs, gaussians, mask_blur)
       continue

   loss = sum(contributing)
   if optimizing_spa is not None:
       loss = optimizing_spa.append_spa_loss(loss)  # once per outer step
   loss.backward()

   if skipgs is not None:
       skipgs.record_backward(True)

   # Densify stats: visibility-side for ALL K views; gradient-side
   # only for views that contributed (their viewspace_point.grad is populated)
   update_densify_stats_full(per_view_packs, gate, gaussians, mask_blur)

   optimizer.step()
   optimizer.zero_grad(set_to_none=True)
   ```

   **Important**: `add_densification_stats` (`spherical_gaussian_model.py`) reads `viewspace_point_tensor.grad`. For views that didn't contribute to the gradient sum, that `.grad` is `None`. Split the existing stat update into:
   - **`add_densification_stats_visibility(render_pkg)`**: updates `max_radii2D` and `mask_blur` from `render_pkg["radii"]` and `render_pkg["visibility_filter"]`. Always safe to call.
   - **`add_densification_stats_gradient(render_pkg)`**: updates `xyz_gradient_accum` and `denom`. Only safe to call when backward fired through that view's `viewspace_point_tensor`.

3. **Config**:
   ```yaml
   training:
     skipgs:
       enabled: true
       phase: final          # only activate in this phase
       warmup: 500
       beta: 0.95
       eps: 1.0e-8
       rho_lo: 0.5
   ```

4. **Logging**. Per-phase summary: log calibrated `ρ_min`, total outer steps, backward steps, final `ρ_cum`. This is the proof the gate did work.

### Test plan

- **Unit**: with `enabled: false`, output bit-identical to T6 baseline at fixed seed.
- **Phase 1 (observability only)**: `enabled: true`, `phase: final`, but force `g = 1` always (set `rho_lo = 1.0`). Log the prospective gate decisions and `s` distribution. Confirm the EMA is stable, deviation scores cluster around 1, and the prospective skip rate matches the paper's range (~30-50% in late-phase).
- **Phase 2 (default)**: `rho_lo = 0.5`. Run end-to-end. Compare:
  - Final-phase wall-clock (target: 30-40% reduction).
  - End-to-end wall-clock (target: 15-25% reduction depending on final-phase share).
  - PSNR / SSIM (should match within 0.05 dB).
- **Phase 3**: ablate budget control by setting `rho_lo = 0.0`. Per the paper's Table 4, this should give larger speedup but worse quality. Confirm the user sees the expected degradation; this builds confidence the budget mechanism is doing its job in your codebase.

### Risk

- **Densify-stat tracking with skipped views** (medium). `viewspace_point_tensor.grad` is `None` for views excluded from the gradient sum. Splitting `add_densification_stats` into visibility and gradient halves is the cleanest fix. Get this wrong and densify decisions silently drift.
- **Interaction with `optimizing_spa.append_spa_loss`** (low). The SPA regulariser must be added only when backward fires. Code above puts it inside the `if contributing` branch — verify this matches T6's semantics.
- **Per-view weight scaling in merge phase** (low; doesn't apply if SkipGS is final-only). When SkipGS activates only in `phase == "final"`, this isn't an issue because final phase doesn't use per-view weights. If you later extend to merge phases, the per-view weight applies to `loss_k` *before* the gate decides; the gate uses raw per-view loss, but the gradient sum uses weighted loss.

### Composition

- **Independent of** T1-T5, T9.
- **Composes with T6** as detailed above.
- **Composes with T7**: SkipGS activates only in the final phase; T7's convergence monitor decides when the final phase ends. Both fire on the same step (T7's monitor checks at end of iter; SkipGS gates the iter itself). After T8 is validated, consider extending SkipGS activation to late-merge windows gated by `monitor.state()`.

---

## T9 — CUDA-batched rasterizer (deferred)

Revisit only after T6 + T8 land and you've measured how much of the multi-view speedup is Python overhead vs kernel-launch overhead. If kernel-launch dominates at K=4, true batched forward (shared sort + tile assignment over K cameras) gives reported 1.4-1.6× over Python-side accumulation alone. Reference: "Efficient Multi-view 3DGS Training" (CVPR'25) codebase.

**Caveat for 6 GB VRAM**: the K=4 working set may not fit during peak intermediate states. Profile T6 at K=4 first; if VRAM is the constraint, K=2 batched is still meaningful but lower payoff.

**Effort**: XL. Rebuild of `submodules/diff-gaussian-rasterization_ms/cuda_rasterizer/` with new forward + backward signatures.

---

## Skipped / dropped

- **P4 (smarter prune cadence)**. Subsumed by T7's cap-bound regime + content-driven trigger predicates. The relevant residual is "tune `min_opacity` to 0.01 in config" — a one-line change you can do during T7 if you want.

- **P7 (skip merge iterations when no new content)**. Doesn't apply to the user's streaming workflow — one new image every ~12 s; zero-content snapshots don't occur.

- **P9 (pool screenspace_points buffer)**. Expected gain ~1%; real autograd-correctness risk (tensor identity + `.grad` reuse semantics across backward calls). With 4.5 GB headroom the allocator's pool reuse already absorbs the 9.6 MB churn; there is no measurable allocator pressure to relieve. Not worth the bug surface.

---

## Validation harness

Before any of the above, add `scripts/bench_progressive.py`:

- Runs `progressive_train.py` with `--quiet`; captures wall-clock per phase and final Gaussian count.
- Parses `progressive_diary.txt` for camera weights / iter counts.
- Computes PSNR vs ground-truth held-out cameras (re-use `metrics.py`).
- Emits JSON: `{phase: {wall_s, peak_vram_mb}, final: {n_gauss, psnr, ssim}}`.

Each task should be benched against a fixed seed and the same snapshot data; commit results to `bench/` for regression tracking.
