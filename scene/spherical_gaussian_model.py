from contextlib import contextmanager

import torch
import torch.nn as nn
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.sh_utils import RGB2SH, SH2RGB
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.index_remap import (
    endcat_to_final_perm, old_to_new_after_append, old_to_new_after_prune,
    remap_grace_records,
)


class SphericalGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, max_sg_degree=3, variable_sg_bands=True):
        self.active_sg_degree = 0
        self.max_sg_degree = max_sg_degree
        self.variable_sg_bands = variable_sg_bands

        # When set (by stable_views() context manager), the 8 cohort-cat
        # properties return cached tensors instead of re-concatenating every
        # access. Trainer enters the context around the K-view render loop
        # where the model is read-only; exit before any mutation.
        self._iter_views: dict = None

        # Per-cohort parameter lists — one entry per creation event.
        self._xyz_cohorts: list = []
        self._rgb_base_cohorts: list = []
        self._opacity_cohorts: list = []
        self._scaling_cohorts: list = []
        self._rotation_cohorts: list = []
        self._sg_directions_cohorts: list = []
        self._sg_sharpness_cohorts: list = []
        self._sg_rgb_cohorts: list = []

        # Per-cohort metadata.
        self.cohort_birth_iter: list = []
        self.cohort_lr_scale: list = []
        self.cohort_xyz_scheduler: list = []

        # Transient buffers — kept as single concatenated tensors; not Adam-tracked.
        self._sg_axis_count = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    # ------------------------------------------------------------------
    # Concatenated read-only views of per-cohort parameters.
    # All existing read sites (renderer, densify, triggers) use these.
    # ------------------------------------------------------------------

    def _cat_cohort(self, name, cohorts, empty_shape):
        if self._iter_views is not None and name in self._iter_views:
            return self._iter_views[name]
        if not cohorts:
            t = torch.empty(*empty_shape, device="cuda")
        elif len(cohorts) == 1:
            t = cohorts[0]
        else:
            t = torch.cat(cohorts, dim=0)
        if self._iter_views is not None:
            self._iter_views[name] = t
        return t

    @property
    def _xyz(self):
        return self._cat_cohort("_xyz", self._xyz_cohorts, (0, 3))

    @property
    def _rgb_base(self):
        return self._cat_cohort("_rgb_base", self._rgb_base_cohorts, (0, 3))

    @property
    def _opacity(self):
        return self._cat_cohort("_opacity", self._opacity_cohorts, (0, 1))

    @property
    def _scaling(self):
        return self._cat_cohort("_scaling", self._scaling_cohorts, (0, 3))

    @property
    def _rotation(self):
        return self._cat_cohort("_rotation", self._rotation_cohorts, (0, 4))

    @property
    def _sg_directions(self):
        return self._cat_cohort(
            "_sg_directions", self._sg_directions_cohorts, (0, self.max_sg_degree, 3))

    @property
    def _sg_sharpness(self):
        return self._cat_cohort(
            "_sg_sharpness", self._sg_sharpness_cohorts, (0, self.max_sg_degree, 1))

    @property
    def _sg_rgb(self):
        return self._cat_cohort(
            "_sg_rgb", self._sg_rgb_cohorts, (0, self.max_sg_degree, 3))

    @contextmanager
    def stable_views(self):
        """Cache the cohort-concatenated tensors for the lifetime of the block.

        Inside this context, repeated reads of _xyz / get_xyz / etc. return
        the same cached tensor instead of re-concatenating per call. The
        caller must NOT mutate any cohort tensors inside the block —
        renders, triggers that only read, and importance scoring are fine.
        """
        self._iter_views = {}
        try:
            yield
        finally:
            self._iter_views = None

    # ------------------------------------------------------------------
    # Standard model properties (unchanged API for callers)
    # ------------------------------------------------------------------

    @property
    def per_band_count(self):
        result = list()
        if self.variable_sg_bands:
            if self._sg_directions_cohorts:
                sg = self._sg_directions
                if sg.numel() > 0:
                    result.append(sg.shape[0])
                else:
                    result.append(0)
        return result

    @property
    def num_primitives(self):
        return sum(c.shape[0] for c in self._xyz_cohorts)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_rgb_base(self):
        return self._rgb_base

    @property
    def get_sg_directions(self):
        return torch.nn.functional.normalize(self._sg_directions, dim=2) if self._sg_directions.numel() > 0 else self._sg_directions

    @property
    def get_sg_sharpness(self):
        return torch.abs(self._sg_sharpness)

    @property
    def get_sg_rgb(self):
        return self._sg_rgb

    @property
    def get_sg_axis_count(self):
        return self._sg_axis_count

    def get_active_sg_degree(self):
        return self.active_sg_degree

    def get_covariance(self, scaling_modifier):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSGdegree(self):
        if self.active_sg_degree < self.max_sg_degree:
            self.active_sg_degree += 1

    # ------------------------------------------------------------------
    # Cohort size helper
    # ------------------------------------------------------------------

    def _cohort_sizes(self):
        return [c.shape[0] for c in self._xyz_cohorts]

    def _split_mask_by_cohort(self, mask):
        """Split a flat boolean mask (over all Gaussians) into per-cohort masks."""
        sizes = self._cohort_sizes()
        masks = []
        offset = 0
        for s in sizes:
            masks.append(mask[offset:offset + s])
            offset += s
        return masks

    # ------------------------------------------------------------------
    # cull_low_sharpness_axes — operates on concatenated tensors in place
    # via per-cohort .data assignment
    # ------------------------------------------------------------------

    def cull_low_sharpness_axes(self, sharpness_threshold=0.5):
        if self.max_sg_degree == 0 or self.num_primitives == 0:
            return

        with torch.no_grad():
            # Work on the concatenated views; write results back per-cohort.
            sg_dir = self._sg_directions
            sg_sharp = self._sg_sharpness
            sg_rgb_t = self._sg_rgb
            axis_count = self._sg_axis_count

            device = sg_dir.device
            _, max_axes, _ = sg_dir.shape

            arange_d = torch.arange(max_axes, device=device)
            active_axes_mask = arange_d[None, :] < axis_count[:, None]

            current_sharpness = self.get_sg_sharpness.squeeze(-1)
            is_low_sharpness = current_sharpness < sharpness_threshold

            prune_mask = active_axes_mask & is_low_sharpness
            keep_mask = active_axes_mask & ~is_low_sharpness

            rgb_from_pruned = sg_rgb_t * prune_mask.unsqueeze(-1)
            sharpness_from_pruned = self.get_sg_sharpness * prune_mask.unsqueeze(-1)

            eps = 1e-8
            safe_sharpness = torch.clamp(sharpness_from_pruned, min=eps)
            rgb_energy = rgb_from_pruned * ((1 - torch.exp(-2 * safe_sharpness)) / (2 * safe_sharpness))
            rgb_to_add = torch.sum(rgb_energy, dim=1)

            new_axis_count = keep_mask.sum(dim=1)

            kept_directions = sg_dir[keep_mask]
            kept_sharpness = sg_sharp[keep_mask]
            kept_rgb = sg_rgb_t[keep_mask]

            new_sg_directions = torch.zeros_like(sg_dir)
            new_sg_sharpness = torch.zeros_like(sg_sharp)
            new_sg_rgb = torch.zeros_like(sg_rgb_t)

            unpack_mask = arange_d[None, :] < new_axis_count[:, None]

            new_sg_directions[unpack_mask] = kept_directions
            new_sg_sharpness[unpack_mask] = kept_sharpness
            new_sg_rgb[unpack_mask] = kept_rgb

            # Write results back to each cohort's Parameter.data
            sizes = self._cohort_sizes()
            rgb_base_full = self._rgb_base  # concat; we need to split adds
            offset = 0
            for ci, s in enumerate(sizes):
                sl = slice(offset, offset + s)
                self._rgb_base_cohorts[ci].data.add_(rgb_to_add[sl])
                self._sg_directions_cohorts[ci].data.copy_(new_sg_directions[sl])
                self._sg_sharpness_cohorts[ci].data.copy_(new_sg_sharpness[sl])
                self._sg_rgb_cohorts[ci].data.copy_(new_sg_rgb[sl])
                offset += s

            self._sg_axis_count = new_axis_count

    def compute_colors_precomp(self, viewpoint_camera, is_training=False):
        colors = self.get_rgb_base.clone()

        if self.max_sg_degree > 0 and self.active_sg_degree > 0:
            campos = viewpoint_camera.camera_center.to(self._xyz.device)
            active_bases = self.get_active_sg_degree()

            view_dirs = campos.unsqueeze(0) - self._xyz
            view_dirs = view_dirs / (torch.norm(view_dirs, dim=1, keepdim=True) + 1e-8)

            device = self._xyz.device
            arange_d = torch.arange(self.max_sg_degree, device=device)
            valid_axis_mask = (arange_d[None, :] < self._sg_axis_count[:, None]) & (arange_d[None, :] < active_bases)

            if not valid_axis_mask.any():
                color_rgb = SH2RGB(colors)
                return torch.clamp(color_rgb, 0.0, 1.0)

            valid_directions_raw = self._sg_directions[valid_axis_mask]
            valid_sharpness_raw = self._sg_sharpness[valid_axis_mask]
            valid_rgb = self._sg_rgb[valid_axis_mask]
            gaussian_indices_for_valid_axes = torch.where(valid_axis_mask)[0]
            valid_view_dirs = view_dirs[gaussian_indices_for_valid_axes]

            cos_theta = torch.sum(torch.nn.functional.normalize(valid_directions_raw, dim=1) * valid_view_dirs, dim=1, keepdim=True)
            directional_scale = torch.exp(torch.abs(valid_sharpness_raw) * (cos_theta - 1.0))
            weighted_rgb = valid_rgb * directional_scale

            colors.scatter_add_(0, gaussian_indices_for_valid_axes.unsqueeze(1).expand(-1, 3), weighted_rgb)

        color_rgb = SH2RGB(colors)

        if is_training:
            return torch.clamp(color_rgb, 0.0, 1.0)
        else:
            torch.clamp(color_rgb, 0.0, 1.0, out=color_rgb)
            return color_rgb

    # ------------------------------------------------------------------
    # Save / restore (cohort-aware)
    # ------------------------------------------------------------------

    def capture(self):
        captured_data = {
            "active_sg_degree": self.active_sg_degree,
            "variable_sg_bands": self.variable_sg_bands,
            "xyz_cohorts": [p.data for p in self._xyz_cohorts],
            "rgb_base_cohorts": [p.data for p in self._rgb_base_cohorts],
            "opacity_cohorts": [p.data for p in self._opacity_cohorts],
            "scaling_cohorts": [p.data for p in self._scaling_cohorts],
            "rotation_cohorts": [p.data for p in self._rotation_cohorts],
            "sg_directions_cohorts": [p.data for p in self._sg_directions_cohorts],
            "sg_sharpness_cohorts": [p.data for p in self._sg_sharpness_cohorts],
            "sg_rgb_cohorts": [p.data for p in self._sg_rgb_cohorts],
            "sg_axis_count": self._sg_axis_count,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "cohort_birth_iter": list(self.cohort_birth_iter),
            "cohort_lr_scale": list(self.cohort_lr_scale),
        }
        return captured_data

    def restore(self, model_args, training_args):
        self.active_sg_degree = model_args["active_sg_degree"]
        if "variable_sg_bands" in model_args:
            self.variable_sg_bands = model_args["variable_sg_bands"]

        cohort_birth_iters = model_args["cohort_birth_iter"]
        cohort_lr_scales = model_args["cohort_lr_scale"]
        n_cohorts = len(model_args["xyz_cohorts"])

        # A train_state.pt written while the trainer was paused (home-server
        # pause/resume) holds CPU tensors; loaded on a fresh --resume with a
        # free GPU, normalise every restored tensor back to CUDA so the
        # optimizer state (cast to param device by load_state_dict) lands on
        # CUDA too. Tensors saved on GPU are already there — .to() is a no-op.
        _dev = "cuda" if torch.cuda.is_available() else "cpu"

        def _r(t):
            return t.to(_dev)

        self._xyz_cohorts = []
        self._rgb_base_cohorts = []
        self._opacity_cohorts = []
        self._scaling_cohorts = []
        self._rotation_cohorts = []
        self._sg_directions_cohorts = []
        self._sg_sharpness_cohorts = []
        self._sg_rgb_cohorts = []
        self.cohort_birth_iter = []
        self.cohort_lr_scale = []
        self.cohort_xyz_scheduler = []

        for ci in range(n_cohorts):
            self._xyz_cohorts.append(nn.Parameter(_r(model_args["xyz_cohorts"][ci]).requires_grad_(True)))
            self._rgb_base_cohorts.append(nn.Parameter(_r(model_args["rgb_base_cohorts"][ci]).requires_grad_(True)))
            self._opacity_cohorts.append(nn.Parameter(_r(model_args["opacity_cohorts"][ci]).requires_grad_(True)))
            self._scaling_cohorts.append(nn.Parameter(_r(model_args["scaling_cohorts"][ci]).requires_grad_(True)))
            self._rotation_cohorts.append(nn.Parameter(_r(model_args["rotation_cohorts"][ci]).requires_grad_(True)))
            if self.max_sg_degree > 0:
                self._sg_directions_cohorts.append(nn.Parameter(_r(model_args["sg_directions_cohorts"][ci]).requires_grad_(True)))
                self._sg_sharpness_cohorts.append(nn.Parameter(_r(model_args["sg_sharpness_cohorts"][ci]).requires_grad_(True)))
                self._sg_rgb_cohorts.append(nn.Parameter(_r(model_args["sg_rgb_cohorts"][ci]).requires_grad_(True)))
            else:
                n = model_args["xyz_cohorts"][ci].shape[0]
                self._sg_directions_cohorts.append(nn.Parameter(torch.empty((n, 0, 3), device="cuda").requires_grad_(True)))
                self._sg_sharpness_cohorts.append(nn.Parameter(torch.empty((n, 0, 1), device="cuda").requires_grad_(True)))
                self._sg_rgb_cohorts.append(nn.Parameter(torch.empty((n, 0, 3), device="cuda").requires_grad_(True)))
            self.cohort_birth_iter.append(cohort_birth_iters[ci])
            self.cohort_lr_scale.append(cohort_lr_scales[ci])
            self.cohort_xyz_scheduler.append(get_expon_lr_func(
                lr_init=training_args.position_lr_init * cohort_lr_scales[ci],
                lr_final=training_args.position_lr_final * cohort_lr_scales[ci],
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps,
            ))

        self._sg_axis_count = _r(model_args["sg_axis_count"])
        self.max_radii2D = _r(model_args["max_radii2D"])
        self.xyz_gradient_accum = _r(model_args["xyz_gradient_accum"])
        self.denom = _r(model_args["denom"])
        self.spatial_lr_scale = model_args["spatial_lr_scale"]

        self.training_setup(training_args)
        self.optimizer.load_state_dict(model_args["optimizer"])

    # ------------------------------------------------------------------
    # GPU serialization (home-server pause/resume — MEGS-2 ingest contract §2/§3)
    # ------------------------------------------------------------------

    def _all_cohort_lists(self):
        return (
            self._xyz_cohorts, self._rgb_base_cohorts, self._opacity_cohorts,
            self._scaling_cohorts, self._rotation_cohorts,
            self._sg_directions_cohorts, self._sg_sharpness_cohorts,
            self._sg_rgb_cohorts,
        )

    def _move_all(self, device):
        """Move every GPU-resident tensor to ``device`` IN PLACE.

        Covers every cohort ``nn.Parameter`` (xyz/rgb_base/opacity/scaling/
        rotation/sg_directions/sg_sharpness/sg_rgb), the non-Adam buffers
        (``max_radii2D``, ``xyz_gradient_accum``, ``denom``,
        ``_sg_axis_count``) and the Adam optimizer momentum state
        (``exp_avg``, ``exp_avg_sq``, and ``max_exp_avg_sq`` under amsgrad) for
        each param. The Adam ``step`` counter is left where Adam created it
        (CPU for the default non-capturable Adam) so the resumed state keeps a
        standard layout.

        Parameter objects are *mutated* (``p.data = ...``), never replaced, so
        the optimizer's state stays keyed to the same params —
        ``optim_guard.optimizer_binding_ok()`` continues to hold after a
        round trip (the T1 momentum-loss invariant). Device→device float
        transfer is bit-exact, so a to_cpu()→to_cuda() round trip is a no-op
        on values.
        """
        device = torch.device(device)
        # Single GPU: compare device *type* so an already-on-cuda tensor
        # (cuda:0) isn't seen as different from the index-less torch.device("cuda").
        dtype = device.type
        # Any cached cohort-cat views (stable_views) are device-stale; drop them.
        self._iter_views = None

        for cohorts in self._all_cohort_lists():
            for p in cohorts:
                if p.data.device.type != dtype:
                    p.data = p.data.to(device)
                if p.grad is not None and p.grad.device.type != dtype:
                    p.grad = p.grad.to(device)

        for name in ("_sg_axis_count", "max_radii2D", "xyz_gradient_accum", "denom"):
            t = getattr(self, name)
            if torch.is_tensor(t) and t.device.type != dtype:
                setattr(self, name, t.to(device))

        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if k == "step":
                        continue  # scalar counter; Adam keeps it on CPU
                    if torch.is_tensor(v) and v.device.type != dtype:
                        state[k] = v.to(device)

    def to_cpu(self):
        """Move model + Adam state to CPU RAM and free VRAM. No disk writes.

        Idempotent: a second call while already on CPU is a cheap no-op. The
        caller (trainer loop) follows this with empty_cache()/synchronize()."""
        self._move_all("cpu")

    def to_cuda(self):
        """Move model + Adam state back to GPU and rebind for training.

        Inverse of :meth:`to_cpu`; the optimizer binding is preserved so
        training resumes from the exact paused state. Idempotent."""
        self._move_all("cuda")

    # ------------------------------------------------------------------
    # Cohort creation
    # ------------------------------------------------------------------

    def _make_cohort_params(self, xyz_t, rgb_base_t, opacity_t, scaling_t, rotation_t,
                            sg_directions_t, sg_sharpness_t, sg_rgb_t):
        """Wrap raw tensors into nn.Parameters and append to cohort lists."""
        self._xyz_cohorts.append(nn.Parameter(xyz_t.requires_grad_(True)))
        self._rgb_base_cohorts.append(nn.Parameter(rgb_base_t.requires_grad_(True)))
        self._opacity_cohorts.append(nn.Parameter(opacity_t.requires_grad_(True)))
        self._scaling_cohorts.append(nn.Parameter(scaling_t.requires_grad_(True)))
        self._rotation_cohorts.append(nn.Parameter(rotation_t.requires_grad_(True)))
        self._sg_directions_cohorts.append(nn.Parameter(sg_directions_t.requires_grad_(True)))
        self._sg_sharpness_cohorts.append(nn.Parameter(sg_sharpness_t.requires_grad_(True)))
        self._sg_rgb_cohorts.append(nn.Parameter(sg_rgb_t.requires_grad_(True)))

    def _build_sg_tensors_for_n(self, n):
        if self.max_sg_degree > 0:
            dirs = torch.randn((n, self.max_sg_degree, 3), device="cuda")
            dirs = dirs / (torch.norm(dirs, dim=2, keepdim=True) + 1e-8)
            sharp = torch.ones((n, self.max_sg_degree, 1), device="cuda") * 0.1
            rgb = torch.randn((n, self.max_sg_degree, 3), device="cuda") * 0.1
        else:
            dirs = torch.empty((n, 0, 3), device="cuda")
            sharp = torch.empty((n, 0, 1), device="cuda")
            rgb = torch.empty((n, 0, 3), device="cuda")
        return dirs, sharp, rgb

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float, birth_iter: int = 0):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        n = fused_point_cloud.shape[0]

        print("Number of points at initialisation : ", n)

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((n, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((n, 1), dtype=torch.float, device="cuda"))

        dirs, sharp, rgb = self._build_sg_tensors_for_n(n)

        self._make_cohort_params(fused_point_cloud, fused_color, opacities, scales, rots, dirs, sharp, rgb)
        self.cohort_birth_iter.append(birth_iter)
        self.cohort_lr_scale.append(spatial_lr_scale)
        self.cohort_xyz_scheduler.append(None)  # filled in by training_setup

        self._sg_axis_count = torch.full((n,), self.max_sg_degree, device="cuda", dtype=torch.int)
        self.max_radii2D = torch.zeros(n, device="cuda")

    def expand_from_pcd(self, pcd: BasicPointCloud, mask: np.ndarray, spatial_lr_scale: float, birth_iter: int = 0):
        """Create a new cohort from selected points in pcd. Always creates cohort N+1."""
        new_xyz = torch.from_numpy(pcd.points[mask]).float().cuda()
        new_rgb = torch.from_numpy(pcd.colors[mask]).float().cuda()
        n_new = new_xyz.shape[0]
        if n_new == 0:
            return

        new_rgb_sh = RGB2SH(new_rgb)
        dist2 = torch.clamp_min(distCUDA2(new_xyz), 0.0000001)
        new_scaling = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        new_rotation = torch.zeros((n_new, 4), device="cuda")
        new_rotation[:, 0] = 1.0
        new_opacity = inverse_sigmoid(0.1 * torch.ones((n_new, 1), dtype=torch.float, device="cuda"))
        new_sg_axis_count = torch.full((n_new,), self.max_sg_degree, device="cuda", dtype=torch.int)

        dirs, sharp, rgb = self._build_sg_tensors_for_n(n_new)

        ci = len(self._xyz_cohorts)
        self._make_cohort_params(new_xyz, new_rgb_sh, new_opacity, new_scaling, new_rotation, dirs, sharp, rgb)
        self.cohort_birth_iter.append(birth_iter)
        self.cohort_lr_scale.append(spatial_lr_scale)
        # Scheduler built here so it's available even if training_setup was already called.
        # training_setup overwrites when called fresh, but for mid-run expand we
        # build the scheduler now and register param groups into the existing optimizer.
        sched = get_expon_lr_func(
            lr_init=self._training_args.position_lr_init * spatial_lr_scale if hasattr(self, '_training_args') else 0.0,
            lr_final=self._training_args.position_lr_final * spatial_lr_scale if hasattr(self, '_training_args') else 0.0,
            lr_delay_mult=self._training_args.position_lr_delay_mult if hasattr(self, '_training_args') else 0.01,
            max_steps=self._training_args.position_lr_max_steps if hasattr(self, '_training_args') else 30000,
        )
        self.cohort_xyz_scheduler.append(sched)

        self._sg_axis_count = torch.cat((self._sg_axis_count, new_sg_axis_count), dim=0)
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros(n_new, device="cuda")), dim=0)
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((n_new, 1), device="cuda")), dim=0)
        self.denom = torch.cat((self.denom, torch.zeros((n_new, 1), device="cuda")), dim=0)

        # Register new cohort's param groups into the existing optimizer.
        if self.optimizer is not None:
            new_groups = self._param_groups_for_cohort(ci)
            for g in new_groups:
                self.optimizer.add_param_group(g)

    # ------------------------------------------------------------------
    # Optimizer setup
    # ------------------------------------------------------------------

    def _param_groups_for_cohort(self, ci):
        """Return a list of optimizer param-group dicts for cohort ci."""
        ta = self._training_args
        groups = [
            {'params': [self._xyz_cohorts[ci]],
             'lr': ta.position_lr_init * self.cohort_lr_scale[ci],
             'name': f'xyz_c{ci}'},
            {'params': [self._rgb_base_cohorts[ci]],
             'lr': ta.feature_lr,
             'name': f'rgb_base_c{ci}'},
            {'params': [self._opacity_cohorts[ci]],
             'lr': ta.opacity_lr,
             'name': f'opacity_c{ci}'},
            {'params': [self._scaling_cohorts[ci]],
             'lr': ta.scaling_lr,
             'name': f'scaling_c{ci}'},
            {'params': [self._rotation_cohorts[ci]],
             'lr': ta.rotation_lr,
             'name': f'rotation_c{ci}'},
        ]
        if self.max_sg_degree > 0:
            groups += [
                {'params': [self._sg_directions_cohorts[ci]],
                 'lr': ta.feature_lr,
                 'name': f'sg_directions_c{ci}'},
                {'params': [self._sg_sharpness_cohorts[ci]],
                 'lr': ta.feature_lr * 4.0,
                 'name': f'sg_sharpness_c{ci}'},
                {'params': [self._sg_rgb_cohorts[ci]],
                 'lr': ta.feature_lr,
                 'name': f'sg_rgb_c{ci}'},
            ]
        return groups

    def training_setup(self, training_args):
        self._training_args = training_args
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        all_groups = []
        for ci in range(len(self._xyz_cohorts)):
            all_groups.extend(self._param_groups_for_cohort(ci))
            # Build xyz scheduler for each cohort keyed off iter_since_birth.
            self.cohort_xyz_scheduler[ci] = get_expon_lr_func(
                lr_init=training_args.position_lr_init * self.cohort_lr_scale[ci],
                lr_final=training_args.position_lr_final * self.cohort_lr_scale[ci],
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps,
            )

        self.optimizer = torch.optim.Adam(all_groups, lr=0.0, eps=1e-15)

    def reset_densification_buffers(self):
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def update_learning_rate(self, global_iter):
        # Each cohort's xyz LR is driven by iter_since_birth, not global_iter.
        for ci, sched in enumerate(self.cohort_xyz_scheduler):
            if sched is None:
                continue
            local_iter = global_iter - self.cohort_birth_iter[ci]
            lr = sched(local_iter)
            for group in self.optimizer.param_groups:
                if group['name'] == f'xyz_c{ci}':
                    group['lr'] = lr
                    break

    def rescale_lr_scale_to(self, new_extent: float, training_args):
        """Retrofit every cohort's xyz LR schedule to use new_extent as spatial_lr_scale."""
        if new_extent <= 0 or not self._xyz_cohorts:
            return
        self.spatial_lr_scale = new_extent
        for ci in range(len(self._xyz_cohorts)):
            self.cohort_lr_scale[ci] = new_extent
            self.cohort_xyz_scheduler[ci] = get_expon_lr_func(
                lr_init=training_args.position_lr_init * new_extent,
                lr_final=training_args.position_lr_final * new_extent,
                lr_delay_mult=training_args.position_lr_delay_mult,
                max_steps=training_args.position_lr_max_steps,
            )

    # ------------------------------------------------------------------
    # Optimizer state manipulation — per-cohort aware
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor, attr_base_name):
        """Replace a per-Gaussian tensor (e.g. 'opacity') across all cohorts.

        `tensor` is the full concatenated replacement; we split by cohort size
        and update each cohort's param group + Adam state.
        """
        sizes = self._cohort_sizes()
        cohort_list = {
            'opacity': self._opacity_cohorts,
            'rgb_base': self._rgb_base_cohorts,
            'xyz': self._xyz_cohorts,
            'scaling': self._scaling_cohorts,
            'rotation': self._rotation_cohorts,
            'sg_directions': self._sg_directions_cohorts,
            'sg_sharpness': self._sg_sharpness_cohorts,
            'sg_rgb': self._sg_rgb_cohorts,
        }[attr_base_name]

        offset = 0
        for ci, s in enumerate(sizes):
            t_ci = tensor[offset:offset + s]
            group_name = f'{attr_base_name}_c{ci}'
            for group in self.optimizer.param_groups:
                if group['name'] == group_name:
                    stored_state = self.optimizer.state.get(group['params'][0], None)
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.zeros_like(t_ci)
                        stored_state["exp_avg_sq"] = torch.zeros_like(t_ci)
                        del self.optimizer.state[group['params'][0]]
                        group["params"][0] = nn.Parameter(t_ci.requires_grad_(True))
                        self.optimizer.state[group['params'][0]] = stored_state
                    else:
                        group["params"][0] = nn.Parameter(t_ci.requires_grad_(True))
                    cohort_list[ci] = group["params"][0]
                    break
            offset += s

    def _prune_optimizer(self, valid_mask):
        """Prune all param groups using a flat valid-point mask.

        valid_mask[i] == True means point i is kept.
        Cohorts that become empty are dropped from both lists and optimizer.
        """
        per_cohort_masks = self._split_mask_by_cohort(valid_mask)

        new_xyz_cohorts = []
        new_rgb_base_cohorts = []
        new_opacity_cohorts = []
        new_scaling_cohorts = []
        new_rotation_cohorts = []
        new_sg_directions_cohorts = []
        new_sg_sharpness_cohorts = []
        new_sg_rgb_cohorts = []
        new_birth_iters = []
        new_lr_scales = []
        new_schedulers = []

        # Process each cohort; build updated param groups.
        per_cohort_groups: dict = {}  # ci -> list of groups
        for group in self.optimizer.param_groups:
            name = group['name']
            # Parse cohort index from name like 'xyz_c0', 'sg_rgb_c1', etc.
            ci = int(name.rsplit('_c', 1)[1])
            per_cohort_groups.setdefault(ci, []).append(group)

        for ci in range(len(self._xyz_cohorts)):
            m = per_cohort_masks[ci]
            if not m.any():
                # Entire cohort pruned — remove from optimizer state.
                for group in per_cohort_groups.get(ci, []):
                    p = group['params'][0]
                    if p in self.optimizer.state:
                        del self.optimizer.state[p]
                continue

            new_xyz_cohorts.append(self._prune_cohort_param(
                self._xyz_cohorts[ci], m, f'xyz_c{ci}', per_cohort_groups.get(ci, [])))
            new_rgb_base_cohorts.append(self._prune_cohort_param(
                self._rgb_base_cohorts[ci], m, f'rgb_base_c{ci}', per_cohort_groups.get(ci, [])))
            new_opacity_cohorts.append(self._prune_cohort_param(
                self._opacity_cohorts[ci], m, f'opacity_c{ci}', per_cohort_groups.get(ci, [])))
            new_scaling_cohorts.append(self._prune_cohort_param(
                self._scaling_cohorts[ci], m, f'scaling_c{ci}', per_cohort_groups.get(ci, [])))
            new_rotation_cohorts.append(self._prune_cohort_param(
                self._rotation_cohorts[ci], m, f'rotation_c{ci}', per_cohort_groups.get(ci, [])))
            new_sg_directions_cohorts.append(self._prune_cohort_param(
                self._sg_directions_cohorts[ci], m, f'sg_directions_c{ci}', per_cohort_groups.get(ci, [])))
            new_sg_sharpness_cohorts.append(self._prune_cohort_param(
                self._sg_sharpness_cohorts[ci], m, f'sg_sharpness_c{ci}', per_cohort_groups.get(ci, [])))
            new_sg_rgb_cohorts.append(self._prune_cohort_param(
                self._sg_rgb_cohorts[ci], m, f'sg_rgb_c{ci}', per_cohort_groups.get(ci, [])))

            new_birth_iters.append(self.cohort_birth_iter[ci])
            new_lr_scales.append(self.cohort_lr_scale[ci])
            new_schedulers.append(self.cohort_xyz_scheduler[ci])

        self._xyz_cohorts = new_xyz_cohorts
        self._rgb_base_cohorts = new_rgb_base_cohorts
        self._opacity_cohorts = new_opacity_cohorts
        self._scaling_cohorts = new_scaling_cohorts
        self._rotation_cohorts = new_rotation_cohorts
        self._sg_directions_cohorts = new_sg_directions_cohorts
        self._sg_sharpness_cohorts = new_sg_sharpness_cohorts
        self._sg_rgb_cohorts = new_sg_rgb_cohorts
        self.cohort_birth_iter = new_birth_iters
        self.cohort_lr_scale = new_lr_scales
        self.cohort_xyz_scheduler = new_schedulers

        # Rebuild optimizer param_groups via add_param_group so PyTorch fills
        # in every default the Adam step expects (maximize, foreach,
        # capturable, differentiable, fused, ...). A hand-rolled dict misses
        # these and step() raises KeyError on whichever it touches first.
        self.optimizer.param_groups = []
        ta = self._training_args
        for ci in range(len(self._xyz_cohorts)):
            self.optimizer.add_param_group(
                {'params': [self._xyz_cohorts[ci]], 'lr': 0.0, 'name': f'xyz_c{ci}'})
            self.optimizer.add_param_group(
                {'params': [self._rgb_base_cohorts[ci]], 'lr': ta.feature_lr, 'name': f'rgb_base_c{ci}'})
            self.optimizer.add_param_group(
                {'params': [self._opacity_cohorts[ci]], 'lr': ta.opacity_lr, 'name': f'opacity_c{ci}'})
            self.optimizer.add_param_group(
                {'params': [self._scaling_cohorts[ci]], 'lr': ta.scaling_lr, 'name': f'scaling_c{ci}'})
            self.optimizer.add_param_group(
                {'params': [self._rotation_cohorts[ci]], 'lr': ta.rotation_lr, 'name': f'rotation_c{ci}'})
            if self.max_sg_degree > 0:
                self.optimizer.add_param_group(
                    {'params': [self._sg_directions_cohorts[ci]], 'lr': ta.feature_lr, 'name': f'sg_directions_c{ci}'})
                self.optimizer.add_param_group(
                    {'params': [self._sg_sharpness_cohorts[ci]], 'lr': ta.feature_lr * 4.0, 'name': f'sg_sharpness_c{ci}'})
                self.optimizer.add_param_group(
                    {'params': [self._sg_rgb_cohorts[ci]], 'lr': ta.feature_lr, 'name': f'sg_rgb_c{ci}'})

    def _prune_cohort_param(self, param, mask, group_name, all_groups):
        """Prune a single cohort's Parameter in place; update optimizer state."""
        new_data = param[mask]
        for group in all_groups:
            if group['name'] == group_name:
                old_param = group['params'][0]
                stored_state = self.optimizer.state.get(old_param, None)
                new_param = nn.Parameter(new_data.requires_grad_(True))
                # Adam only allocates state after the first step() that sees a
                # non-None grad for this param; sg_* groups can stay state-less
                # while active_sg_degree==0. Guard both the delete and the move.
                if stored_state is not None:
                    del self.optimizer.state[old_param]
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                    self.optimizer.state[new_param] = stored_state
                return new_param
        # No matching group (e.g. max_sg_degree==0 for sg groups) — just create param.
        return nn.Parameter(new_data.requires_grad_(True))

    def prune_points(self, mask):
        """mask[i]==True means point i should be pruned."""
        valid_points_mask = ~mask
        self._prune_optimizer(valid_points_mask)

        self._sg_axis_count = self._sg_axis_count[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

        # Grace records store flat-index ranges; pruning shifts flat indices,
        # so translate the ranges or later grace masks protect the wrong rows.
        if getattr(self, '_grace_records', None):
            self._grace_records = remap_grace_records(
                self._grace_records,
                old_to_new_after_prune(valid_points_mask.detach().cpu().numpy()))

    def _finalize_append(self, sizes_before, appended, nsac_chunks):
        """Fix flat-order bookkeeping after per-cohort appends.

        _append_to_cohort inserts new rows at the end of each *cohort* —
        mid-flat-order — while naive `torch.cat((old, new))` bookkeeping
        assumes the flat end. Rebuild _sg_axis_count in true flat order,
        shift grace ranges, and return the endcat→final permutation (as a
        torch index tensor) for callers that built masks in endcat order.
        Returns None when nothing was appended.
        """
        if sum(appended) == 0:
            return None
        device = self._xyz_cohorts[0].device
        perm = endcat_to_final_perm(sizes_before, appended)
        perm_t = torch.as_tensor(perm, device=device, dtype=torch.long)
        if nsac_chunks:
            endcat = torch.cat([self._sg_axis_count, *nsac_chunks], dim=0)
            self._sg_axis_count = endcat[perm_t]
        if getattr(self, '_grace_records', None):
            self._grace_records = remap_grace_records(
                self._grace_records,
                old_to_new_after_append(sizes_before, appended))
        return perm_t

    # ------------------------------------------------------------------
    # Densification helpers
    # ------------------------------------------------------------------

    def _append_to_cohort(self, ci, new_xyz, new_rgb_base, new_opacity, new_scaling,
                          new_rotation, new_sg_directions, new_sg_sharpness, new_sg_rgb):
        """Append new points to cohort ci's tensors + optimizer state.

        Split children inherit the parent cohort so their LR schedule continues
        from the same birth_iter and lr_scale as the original Gaussians.
        """
        attrs = [
            (f'xyz_c{ci}', self._xyz_cohorts, new_xyz),
            (f'rgb_base_c{ci}', self._rgb_base_cohorts, new_rgb_base),
            (f'opacity_c{ci}', self._opacity_cohorts, new_opacity),
            (f'scaling_c{ci}', self._scaling_cohorts, new_scaling),
            (f'rotation_c{ci}', self._rotation_cohorts, new_rotation),
        ]
        if self.max_sg_degree > 0:
            attrs += [
                (f'sg_directions_c{ci}', self._sg_directions_cohorts, new_sg_directions),
                (f'sg_sharpness_c{ci}', self._sg_sharpness_cohorts, new_sg_sharpness),
                (f'sg_rgb_c{ci}', self._sg_rgb_cohorts, new_sg_rgb),
            ]

        for group_name, cohort_list, ext_tensor in attrs:
            for group in self.optimizer.param_groups:
                if group['name'] == group_name:
                    old_param = group['params'][0]
                    stored_state = self.optimizer.state.get(old_param, None)
                    new_data = torch.cat((old_param.data, ext_tensor), dim=0)
                    new_param = nn.Parameter(new_data.requires_grad_(True))
                    if stored_state is not None:
                        stored_state["exp_avg"] = torch.cat(
                            (stored_state["exp_avg"], torch.zeros_like(ext_tensor)), dim=0)
                        stored_state["exp_avg_sq"] = torch.cat(
                            (stored_state["exp_avg_sq"], torch.zeros_like(ext_tensor)), dim=0)
                        del self.optimizer.state[old_param]
                        self.optimizer.state[new_param] = stored_state
                    else:
                        if old_param in self.optimizer.state:
                            del self.optimizer.state[old_param]
                    group['params'][0] = new_param
                    cohort_list[ci] = new_param
                    break

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        sizes = self._cohort_sizes()
        n_new_total = 0
        new_xyz_list, new_rgb_list, new_opacity_list = [], [], []
        new_scaling_list, new_rotation_list = [], []
        new_sg_dir_list, new_sg_sharp_list, new_sg_rgb_list, new_sg_ac_list = [], [], [], []

        full_xyz = self._xyz
        full_rgb_base = self._rgb_base
        full_opacity = self._opacity
        full_rotation = self._rotation

        offset = 0
        for ci, s in enumerate(sizes):
            cm = selected_pts_mask[offset:offset + s]
            if cm.any():
                c_scaling = self.get_scaling[offset:offset + s][cm]
                c_rotation = full_rotation[offset:offset + s][cm]
                c_xyz = full_xyz[offset:offset + s][cm]
                c_rgb_base = full_rgb_base[offset:offset + s][cm]
                c_opacity = full_opacity[offset:offset + s][cm]

                stds = c_scaling.repeat(N, 1)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(c_rotation).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + c_xyz.repeat(N, 1)
                new_scaling = self.scaling_inverse_activation(c_scaling.repeat(N, 1) / (0.8 * N))
                new_rotation = c_rotation.repeat(N, 1)
                new_rgb_base = c_rgb_base.repeat(N, 1)
                new_opacity = c_opacity.repeat(N, 1)

                new_xyz_list.append((ci, new_xyz))
                new_rgb_list.append((ci, new_rgb_base))
                new_opacity_list.append((ci, new_opacity))
                new_scaling_list.append((ci, new_scaling))
                new_rotation_list.append((ci, new_rotation))
                n_new_total += new_xyz.shape[0]

                if self.max_sg_degree > 0:
                    full_sg_dir = self._sg_directions
                    full_sg_sharp = self._sg_sharpness
                    full_sg_rgb_t = self._sg_rgb
                    new_sg_dir_list.append((ci, full_sg_dir[offset:offset + s][cm].repeat(N, 1, 1)))
                    new_sg_sharp_list.append((ci, full_sg_sharp[offset:offset + s][cm].repeat(N, 1, 1)))
                    new_sg_rgb_list.append((ci, full_sg_rgb_t[offset:offset + s][cm].repeat(N, 1, 1)))
                    new_sg_ac_list.append((ci, self._sg_axis_count[offset:offset + s][cm].repeat(N)))
            offset += s

        appended = [0] * len(sizes)
        nsac_chunks = []
        for i, (ci, new_xyz) in enumerate(new_xyz_list):
            ci2, new_rgb_base = new_rgb_list[i]
            ci3, new_opacity = new_opacity_list[i]
            ci4, new_scaling = new_scaling_list[i]
            ci5, new_rotation = new_rotation_list[i]
            nsd = new_sg_dir_list[i][1] if new_sg_dir_list else None
            nss = new_sg_sharp_list[i][1] if new_sg_sharp_list else None
            nsr = new_sg_rgb_list[i][1] if new_sg_rgb_list else None
            nsac = new_sg_ac_list[i][1] if new_sg_ac_list else None

            self._append_to_cohort(ci, new_xyz, new_rgb_base, new_opacity, new_scaling, new_rotation, nsd, nss, nsr)
            if nsac is not None:
                nsac_chunks.append(nsac)
            appended[ci] += new_xyz.shape[0]

        perm_t = self._finalize_append(sizes, appended, nsac_chunks)

        n_total = sum(c.shape[0] for c in self._xyz_cohorts)
        self.xyz_gradient_accum = torch.zeros((n_total, 1), device="cuda")
        self.denom = torch.zeros((n_total, 1), device="cuda")
        self.max_radii2D = torch.zeros(n_total, device="cuda")

        # selected_pts_mask indexes the OLD flat order; appended rows sit
        # mid-flat-order (end of each cohort), so build the split-parent
        # prune mask in endcat order and permute it into true flat order.
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(n_new_total, device="cuda", dtype=bool)))
        if perm_t is not None:
            prune_filter = prune_filter[perm_t]
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        sizes = self._cohort_sizes()
        n_new_total = 0
        appended = [0] * len(sizes)
        nsac_chunks = []
        full_xyz = self._xyz
        full_rgb_base = self._rgb_base
        full_opacity = self._opacity
        full_scaling_raw = self._scaling
        full_rotation = self._rotation

        offset = 0
        for ci, s in enumerate(sizes):
            cm = selected_pts_mask[offset:offset + s]
            if cm.any():
                new_xyz = full_xyz[offset:offset + s][cm]
                new_rgb_base = full_rgb_base[offset:offset + s][cm]
                new_opacity = full_opacity[offset:offset + s][cm]
                new_scaling = full_scaling_raw[offset:offset + s][cm]
                new_rotation = full_rotation[offset:offset + s][cm]

                nsd = nss = nsr = None
                nsac = None
                if self.max_sg_degree > 0:
                    full_sg_dir = self._sg_directions
                    full_sg_sharp = self._sg_sharpness
                    full_sg_rgb_t = self._sg_rgb
                    nsd = full_sg_dir[offset:offset + s][cm]
                    nss = full_sg_sharp[offset:offset + s][cm]
                    nsr = full_sg_rgb_t[offset:offset + s][cm]
                    nsac = self._sg_axis_count[offset:offset + s][cm]

                self._append_to_cohort(ci, new_xyz, new_rgb_base, new_opacity, new_scaling, new_rotation, nsd, nss, nsr)
                if nsac is not None:
                    nsac_chunks.append(nsac)
                appended[ci] = new_xyz.shape[0]
                n_new_total += new_xyz.shape[0]
            offset += s

        self._finalize_append(sizes, appended, nsac_chunks)

        n_total = sum(c.shape[0] for c in self._xyz_cohorts)
        self.xyz_gradient_accum = torch.zeros((n_total, 1), device="cuda")
        self.denom = torch.zeros((n_total, 1), device="cuda")
        self.max_radii2D = torch.zeros(n_total, device="cuda")

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def opacity_size_prune(self, min_opacity, max_screen_size, extent,
                           grace_iter=None):
        """grace_iter: pass the current training iter to exempt
        grace-protected (recently added) Gaussians from the prune."""
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size is not None:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if grace_iter is not None:
            prune_mask = prune_mask & ~self.get_grace_protected_mask(grace_iter)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        self.replace_tensor_to_optimizer(opacities_new, "opacity")

    def densify_and_prune_split(self, max_grad, min_opacity, extent, max_screen_size, mask,
                                grace_iter=None):
        """grace_iter: pass the current training iter to exempt
        grace-protected Gaussians from the trailing opacity/size prune. The
        mask is computed AFTER clone/split, whose index shifts are tracked
        in the grace records by _finalize_append/prune_points."""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split_mask(grads, max_grad, extent, mask)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if grace_iter is not None:
            prune_mask = prune_mask & ~self.get_grace_protected_mask(grace_iter)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def densify_and_split_mask(self, grads, grad_threshold, scene_extent, mask, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        padded_mask = torch.zeros((n_init_points), dtype=torch.bool, device='cuda')
        padded_mask[:grads.shape[0]] = mask
        selected_pts_mask = torch.logical_or(selected_pts_mask, padded_mask)

        sizes = self._cohort_sizes()
        n_new_total = 0
        appended = [0] * len(sizes)
        nsac_chunks = []
        full_xyz = self._xyz
        full_rgb_base = self._rgb_base
        full_opacity = self._opacity
        full_rotation = self._rotation

        offset = 0
        for ci, s in enumerate(sizes):
            cm = selected_pts_mask[offset:offset + s]
            if cm.any():
                c_scaling = self.get_scaling[offset:offset + s][cm]
                c_rotation = full_rotation[offset:offset + s][cm]
                c_xyz = full_xyz[offset:offset + s][cm]
                c_rgb_base = full_rgb_base[offset:offset + s][cm]
                c_opacity = full_opacity[offset:offset + s][cm]

                stds = c_scaling.repeat(N, 1)
                means = torch.zeros((stds.size(0), 3), device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(c_rotation).repeat(N, 1, 1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + c_xyz.repeat(N, 1)
                new_scaling = self.scaling_inverse_activation(c_scaling.repeat(N, 1) / (0.8 * N))
                new_rotation = c_rotation.repeat(N, 1)
                new_rgb_base = c_rgb_base.repeat(N, 1)
                new_opacity = c_opacity.repeat(N, 1)

                nsd = nss = nsr = None
                nsac = None
                if self.max_sg_degree > 0:
                    full_sg_dir = self._sg_directions
                    full_sg_sharp = self._sg_sharpness
                    full_sg_rgb_t = self._sg_rgb
                    nsd = full_sg_dir[offset:offset + s][cm].repeat(N, 1, 1)
                    nss = full_sg_sharp[offset:offset + s][cm].repeat(N, 1, 1)
                    nsr = full_sg_rgb_t[offset:offset + s][cm].repeat(N, 1, 1)
                    nsac = self._sg_axis_count[offset:offset + s][cm].repeat(N)

                self._append_to_cohort(ci, new_xyz, new_rgb_base, new_opacity, new_scaling, new_rotation, nsd, nss, nsr)
                if nsac is not None:
                    nsac_chunks.append(nsac)
                appended[ci] = new_xyz.shape[0]
                n_new_total += new_xyz.shape[0]
            offset += s

        perm_t = self._finalize_append(sizes, appended, nsac_chunks)

        n_total = sum(c.shape[0] for c in self._xyz_cohorts)
        self.xyz_gradient_accum = torch.zeros((n_total, 1), device="cuda")
        self.denom = torch.zeros((n_total, 1), device="cuda")
        self.max_radii2D = torch.zeros(n_total, device="cuda")

        # See densify_and_split: mask built in endcat order, permuted to
        # true flat order before pruning the split parents.
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(n_new_total, device="cuda", dtype=bool)))
        if perm_t is not None:
            prune_filter = prune_filter[perm_t]
        self.prune_points(prune_filter)

    def reinitial_pts(self, pts, rgb):
        fused_point_cloud = pts
        fused_color = RGB2SH(rgb)
        n = fused_point_cloud.shape[0]

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((n, 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((n, 1), dtype=torch.float, device="cuda"))

        dirs, sharp, rgb_sg = self._build_sg_tensors_for_n(n)

        # Replace all cohorts with a single new one.
        self._xyz_cohorts = []
        self._rgb_base_cohorts = []
        self._opacity_cohorts = []
        self._scaling_cohorts = []
        self._rotation_cohorts = []
        self._sg_directions_cohorts = []
        self._sg_sharpness_cohorts = []
        self._sg_rgb_cohorts = []
        self.cohort_birth_iter = [0]
        self.cohort_lr_scale = [self.spatial_lr_scale]
        self.cohort_xyz_scheduler = [None]

        self._make_cohort_params(fused_point_cloud, fused_color, opacities, scales, rots, dirs, sharp, rgb_sg)
        self._sg_axis_count = torch.full((n,), self.max_sg_degree, device="cuda", dtype=torch.int)
        self.max_radii2D = torch.zeros(n, device="cuda")

    # ------------------------------------------------------------------
    # Grace-period tracking
    # ------------------------------------------------------------------

    def mark_recently_added(self, idx_range: slice, iteration: int, grace_iters: int):
        if not hasattr(self, '_grace_records') or not isinstance(self._grace_records, list):
            self._grace_records = []
        start = idx_range.start if idx_range.start is not None else 0
        stop = idx_range.stop if idx_range.stop is not None else sum(c.shape[0] for c in self._xyz_cohorts)
        self._grace_records.append((start, stop, iteration + grace_iters))

    def get_grace_protected_mask(self, current_iter: int) -> torch.Tensor:
        n = sum(c.shape[0] for c in self._xyz_cohorts)
        mask = torch.zeros(n, dtype=torch.bool, device="cuda")
        if not hasattr(self, '_grace_records') or not self._grace_records:
            return mask
        active = []
        for record in self._grace_records:
            start, stop, expires = record
            if current_iter < expires:
                stop = min(stop, n)
                if start < stop:
                    mask[start:stop] = True
                active.append(record)
        self._grace_records = active
        return mask

    # ------------------------------------------------------------------
    # PLY I/O — geometry save/load (not checkpoint; no optimizer state)
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f'rgb_base_{i}')
        l.append('sg_axis_count')
        for base_idx in range(self.max_sg_degree):
            for i in range(3):
                l.append(f'sg_dir_{base_idx}_{i}')
            l.append(f'sg_sharp_{base_idx}')
            for i in range(3):
                l.append(f'sg_rgb_{base_idx}_{i}')
        for i in range(3):
            l.append(f'scale_{i}')
        for i in range(4):
            l.append(f'rot_{i}')
        l.append('opacity')
        return l

    def construct_list_of_attributes_for_degree(self, sg_degree):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(3):
            l.append(f'rgb_base_{i}')
        l.append('sg_axis_count')
        for base_idx in range(sg_degree):
            for i in range(3):
                l.append(f'sg_dir_{base_idx}_{i}')
            l.append(f'sg_sharp_{base_idx}')
            for i in range(3):
                l.append(f'sg_rgb_{base_idx}_{i}')
        for i in range(3):
            l.append(f'scale_{i}')
        for i in range(4):
            l.append(f'rot_{i}')
        l.append('opacity')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self.get_rgb_base.detach().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        sg_axis_count = self.get_sg_axis_count.detach().cpu().numpy() if self.max_sg_degree > 0 else None

        sg_directions = self._sg_directions.detach().cpu().numpy() if self.max_sg_degree > 0 else None
        sg_sharpness = self._sg_sharpness.detach().cpu().numpy() if self.max_sg_degree > 0 else None
        sg_rgb = self._sg_rgb.detach().cpu().numpy() if self.max_sg_degree > 0 else None

        elements_list = []
        for sg_degree in range(self.max_sg_degree + 1):
            degrees_mask = (sg_axis_count == sg_degree).squeeze() if sg_axis_count is not None else np.ones(xyz.shape[0], dtype=bool)
            if not degrees_mask.any():
                continue

            xyz_degree = xyz[degrees_mask]
            normals_degree = normals[degrees_mask]
            f_dc_degree = f_dc[degrees_mask]
            opacities_degree = opacities[degrees_mask]
            scale_degree = scale[degrees_mask]
            rotation_degree = rotation[degrees_mask]
            axis_count_degree = sg_axis_count[degrees_mask] if sg_axis_count is not None else np.zeros((xyz_degree.shape[0], 1))

            attribute_list = [xyz_degree, normals_degree, f_dc_degree, axis_count_degree.reshape(-1, 1)]

            if sg_degree > 0 and sg_directions is not None:
                sg_directions_degree = sg_directions[degrees_mask, :sg_degree, :]
                sg_sharpness_degree = sg_sharpness[degrees_mask, :sg_degree, :]
                sg_rgb_degree = sg_rgb[degrees_mask, :sg_degree, :]
                for axis_idx in range(sg_degree):
                    attribute_list.append(sg_directions_degree[:, axis_idx, :])
                    attribute_list.append(sg_sharpness_degree[:, axis_idx, :])
                    attribute_list.append(sg_rgb_degree[:, axis_idx, :])

            attribute_list.extend([scale_degree, rotation_degree, opacities_degree])
            attributes = np.concatenate(attribute_list, axis=1)

            dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_for_degree(sg_degree)]
            elements = np.empty(xyz_degree.shape[0], dtype=dtype_full)
            for i, attribute_name in enumerate(self.construct_list_of_attributes_for_degree(sg_degree)):
                elements[attribute_name] = attributes[:, i]

            elements_list.append(PlyElement.describe(elements, f'vertex_{sg_degree}'))

        PlyData(elements_list).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz_list = []
        rgb_base_list = []
        axis_counts_list = []
        scales_list = []
        rots_list = []
        opacities_list = []
        sg_directions_list = []
        sg_sharpness_list = []
        sg_rgb_list = []

        for sg_degree in range(self.max_sg_degree + 1):
            element_name = f'vertex_{sg_degree}'
            if element_name not in [elem.name for elem in plydata.elements]:
                continue

            element = next(elem for elem in plydata.elements if elem.name == element_name)

            xyz = np.stack((np.asarray(element["x"]),
                            np.asarray(element["y"]),
                            np.asarray(element["z"])), axis=1)
            rgb_base = np.stack((np.asarray(element["rgb_base_0"]),
                                 np.asarray(element["rgb_base_1"]),
                                 np.asarray(element["rgb_base_2"])), axis=1)
            axis_counts = np.asarray(element["sg_axis_count"]).astype(int)
            scale_names = [f'scale_{i}' for i in range(3)]
            scales = np.stack([np.asarray(element[attr_name]) for attr_name in scale_names], axis=1)
            rot_names = [f'rot_{i}' for i in range(4)]
            rots = np.stack([np.asarray(element[attr_name]) for attr_name in rot_names], axis=1)
            opacities = np.asarray(element["opacity"])[..., None]

            xyz_list.append(xyz)
            rgb_base_list.append(rgb_base)
            axis_counts_list.append(axis_counts)
            scales_list.append(scales)
            rots_list.append(rots)
            opacities_list.append(opacities)

            num_points = xyz.shape[0]
            if sg_degree > 0:
                directions = np.zeros((num_points, sg_degree, 3))
                sharpness = np.zeros((num_points, sg_degree, 1))
                rgb = np.zeros((num_points, sg_degree, 3))
                for axis_idx in range(sg_degree):
                    directions[:, axis_idx, :] = np.stack((
                        np.asarray(element[f"sg_dir_{axis_idx}_0"]),
                        np.asarray(element[f"sg_dir_{axis_idx}_1"]),
                        np.asarray(element[f"sg_dir_{axis_idx}_2"])
                    ), axis=1)
                    sharpness[:, axis_idx, 0] = np.asarray(element[f"sg_sharp_{axis_idx}"])
                    rgb[:, axis_idx, :] = np.stack((
                        np.asarray(element[f"sg_rgb_{axis_idx}_0"]),
                        np.asarray(element[f"sg_rgb_{axis_idx}_1"]),
                        np.asarray(element[f"sg_rgb_{axis_idx}_2"])
                    ), axis=1)
                sg_directions_list.append(torch.tensor(directions, dtype=torch.float, device="cuda"))
                sg_sharpness_list.append(torch.tensor(sharpness, dtype=torch.float, device="cuda"))
                sg_rgb_list.append(torch.tensor(rgb, dtype=torch.float, device="cuda"))
            else:
                sg_directions_list.append(torch.empty((num_points, 0, 3), dtype=torch.float, device="cuda"))
                sg_sharpness_list.append(torch.empty((num_points, 0, 1), dtype=torch.float, device="cuda"))
                sg_rgb_list.append(torch.empty((num_points, 0, 3), dtype=torch.float, device="cuda"))

        xyz = np.concatenate(xyz_list, axis=0)
        rgb_base = np.concatenate(rgb_base_list, axis=0)
        axis_counts = np.concatenate(axis_counts_list, axis=0)
        scales = np.concatenate(scales_list, axis=0)
        rots = np.concatenate(rots_list, axis=0)
        opacities = np.concatenate(opacities_list, axis=0)
        sg_directions = torch.cat(sg_directions_list, dim=0) if sg_directions_list else torch.empty((0, self.max_sg_degree, 3), device="cuda")
        sg_sharpness = torch.cat(sg_sharpness_list, dim=0) if sg_sharpness_list else torch.empty((0, self.max_sg_degree, 1), device="cuda")
        sg_rgb = torch.cat(sg_rgb_list, dim=0) if sg_rgb_list else torch.empty((0, self.max_sg_degree, 3), device="cuda")

        # Load as a single cohort (birth_iter=0, lr_scale=spatial_lr_scale).
        self._xyz_cohorts = [nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))]
        self._rgb_base_cohorts = [nn.Parameter(torch.tensor(rgb_base, dtype=torch.float, device="cuda").requires_grad_(True))]
        self._opacity_cohorts = [nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))]
        self._scaling_cohorts = [nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))]
        self._rotation_cohorts = [nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))]
        self._sg_directions_cohorts = [nn.Parameter(sg_directions.requires_grad_(True))]
        self._sg_sharpness_cohorts = [nn.Parameter(sg_sharpness.requires_grad_(True))]
        self._sg_rgb_cohorts = [nn.Parameter(sg_rgb.requires_grad_(True))]
        self._sg_axis_count = torch.tensor(axis_counts, dtype=torch.int, device="cuda")
        self.cohort_birth_iter = [0]
        self.cohort_lr_scale = [self.spatial_lr_scale]
        self.cohort_xyz_scheduler = [None]
        self.active_sg_degree = self.max_sg_degree
