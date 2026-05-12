"""ProgressiveScene: wraps MEGS-2's COLMAP loader for snapshot-driven progressive training.

Each snapshot is a numbered subdirectory (e.g. 16/, 17/, ...) that is a complete
COLMAP scene as known at that point in the incremental SfM.  Folder N is a strict
superset of folder N-1 (same images plus new ones, with updated poses).

Expected layout inside each snapshot folder:
  <source_path_dir>/<N>/
    images/
    sparse/0/
      cameras.bin   (or .txt)
      images.bin    (or .txt)
      points3D.bin  (or .txt)
      imageMatchMatrix.txt   (optional; required for Phase 6 weighting)
      imagesNames.txt        (optional; required for Phase 6 weighting)
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from scene.dataset_readers import sceneLoadTypeCallbacks, readColmapCameras, getNerfppNorm
from scene.colmap_loader import (
    read_extrinsics_binary, read_intrinsics_binary,
    read_extrinsics_text, read_intrinsics_text,
    read_points3D_binary_with_tracks, qvec2rotmat,
)
from utils.camera_utils import cameraList_from_camInfos
from utils.graphics_utils import BasicPointCloud, fov2focal
from scene.dataset_readers import CameraInfo, SceneInfo


class ProgressiveScene:
    """Manages the stream of COLMAP snapshots for progressive Gaussian training."""

    def __init__(self, source_path_dir: Optional[str] = None,
                 model_args=None, images_subdir: str = "images"):
        self.source_path_dir = Path(source_path_dir) if source_path_dir else None
        self.model_args = model_args
        self.images_subdir = images_subdir

        # Snapshot discovery: only when source_path_dir is a pre-existing tree
        if source_path_dir is not None:
            self.snapshot_indices = sorted(
                [int(d) for d in os.listdir(source_path_dir)
                 if os.path.isdir(os.path.join(source_path_dir, str(d))) and d.isdigit()]
            )
        else:
            self.snapshot_indices = []

        self._current_snapshot_pos = 0
        self.current_idx: Optional[int] = None
        self.snapshot_dir: Optional[Path] = None

        # Cumulative state across snapshots
        self.train_cameras: List = []
        self._cam_name_to_idx: Dict[str, int] = {}

        # Previous snapshot's sparse cloud (for diffing new points)
        self.last_basic_pcd: Optional[BasicPointCloud] = None
        self.current_basic_pcd: Optional[BasicPointCloud] = None

        # Normalisation radius (cameras_extent), set on first snapshot
        self.cameras_extent: float = 1.0

        # Match-matrix state (set externally after load_next_snapshot / add_snapshot)
        self.image_match_matrix: Optional[np.ndarray] = None
        self.image_weights: Optional[np.ndarray] = None

        # Per-image EMA of recent rendering loss. Populated by the trainer
        # after each per-view loss is computed. Used to bias view selection
        # toward currently-poorly-rendered images (importance sampling).
        self.image_error_ema: Dict[str, float] = {}

        # Point track info: point_id (int) → list of colmap image_ids
        self._point_track_info: Dict[int, List[int]] = {}
        self._current_sfm_xyz: Optional[np.ndarray] = None
        self._current_sfm_point_ids: Optional[List[int]] = None

        # Mapping from COLMAP image_id → index in train_cameras
        self._colmap_id_to_cam_idx: Dict[int, int] = {}

        # Count of distinct ingested images (not snapshots)
        self._n_images: int = 0

    @property
    def n_images(self) -> int:
        return self._n_images

    # ------------------------------------------------------------------
    # Public API — batch mode (original)
    # ------------------------------------------------------------------

    def has_more_snapshots(self) -> bool:
        return self._current_snapshot_pos < len(self.snapshot_indices)

    def load_next_snapshot(self) -> Tuple[List, np.ndarray, List[int]]:
        """Load the next snapshot; update internal state.

        Returns:
            new_cams: list of Camera objects that are new in this snapshot
            new_point_mask: bool np.ndarray (N,) over current_basic_pcd.points,
                            True = this point did not appear in the previous snapshot
            new_cam_indices: list of int — indices into self.train_cameras for new cams
        """
        snap_num = self.snapshot_indices[self._current_snapshot_pos]
        self._current_snapshot_pos += 1
        self.current_idx = snap_num
        self.snapshot_dir = self.source_path_dir / str(snap_num)

        snap_path = str(self.snapshot_dir)

        # ---- Load COLMAP extrinsics / intrinsics ----
        try:
            cam_extrinsics = read_extrinsics_binary(
                os.path.join(snap_path, "sparse/0", "images.bin"))
            cam_intrinsics = read_intrinsics_binary(
                os.path.join(snap_path, "sparse/0", "cameras.bin"))
        except Exception:
            cam_extrinsics = read_extrinsics_text(
                os.path.join(snap_path, "sparse/0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(
                os.path.join(snap_path, "sparse/0", "cameras.txt"))

        images_folder = os.path.join(snap_path, self.images_subdir)
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=images_folder,
        )

        # ---- Load sparse points with track info ----
        bin_path = os.path.join(snap_path, "sparse/0", "points3D.bin")
        txt_path = os.path.join(snap_path, "sparse/0", "points3D.txt")
        try:
            xyzs, rgbs, _, point_ids, track_image_ids = \
                read_points3D_binary_with_tracks(bin_path)
        except FileNotFoundError:
            # Fall back: read txt (no track info)
            from scene.colmap_loader import read_points3D_text
            xyzs, rgbs, _ = read_points3D_text(txt_path)
            point_ids = list(range(len(xyzs)))
            track_image_ids = [[] for _ in xyzs]

        current_pcd = BasicPointCloud(
            points=xyzs,
            colors=rgbs / 255.0 if rgbs.max() > 1.0 else rgbs,
            normals=np.zeros_like(xyzs),
        )

        self._current_sfm_xyz = xyzs
        self._current_sfm_point_ids = point_ids
        # Rebuild track info dict (overwrite; latest snapshot is ground truth)
        self._point_track_info = {}
        for pid, img_ids in zip(point_ids, track_image_ids):
            self._point_track_info[pid] = img_ids

        # ---- Compute new-point mask ----
        if self.last_basic_pcd is None:
            new_point_mask = np.ones(len(xyzs), dtype=bool)
        else:
            prev_xyzs = self.last_basic_pcd.points
            # A point is "new" if it doesn't appear (by position) in the previous cloud.
            # Efficient check: use a set of rounded positions.
            prev_set = set(map(tuple, np.round(prev_xyzs, 4)))
            new_point_mask = np.array(
                [tuple(np.round(p, 4)) not in prev_set for p in xyzs], dtype=bool)

        self.last_basic_pcd = current_pcd
        self.current_basic_pcd = current_pcd

        # ---- Update cameras extent on first snapshot ----
        if self._current_snapshot_pos == 1:
            norm = getNerfppNorm(cam_infos)
            self.cameras_extent = norm["radius"]

        # ---- Merge camera list ----
        resolution_scale = getattr(self.model_args, 'resolution', -1)
        # Use cameraList_from_camInfos to build Camera objects
        new_cam_objects = cameraList_from_camInfos(cam_infos, 1.0, self.model_args)

        new_cams = []
        new_cam_indices = []

        for cam in new_cam_objects:
            name = cam.image_name
            if name in self._cam_name_to_idx:
                # Update existing camera pose (handles SfM pose refinement)
                idx = self._cam_name_to_idx[name]
                existing = self.train_cameras[idx]
                existing.R = cam.R
                existing.T = cam.T
                existing.FoVx = cam.FoVx
                existing.FoVy = cam.FoVy
                import torch
                from utils.graphics_utils import getWorld2View2, getProjectionMatrix
                existing.world_view_transform = torch.tensor(
                    getWorld2View2(cam.R, cam.T)).transpose(0, 1).cuda()
                existing.projection_matrix = getProjectionMatrix(
                    znear=existing.znear, zfar=existing.zfar,
                    fovX=cam.FoVx, fovY=cam.FoVy).transpose(0, 1).cuda()
                existing.full_proj_transform = (
                    existing.world_view_transform.unsqueeze(0).bmm(
                        existing.projection_matrix.unsqueeze(0))).squeeze(0)
                existing.camera_center = existing.world_view_transform.inverse()[3, :3]
            else:
                idx = len(self.train_cameras)
                self.train_cameras.append(cam)
                self._cam_name_to_idx[name] = idx
                self._colmap_id_to_cam_idx[cam.colmap_id] = idx
                new_cams.append(cam)
                new_cam_indices.append(idx)

        self._n_images = len(self.train_cameras)
        return new_cams, new_point_mask, new_cam_indices

    # ------------------------------------------------------------------
    # Public API — incremental / streaming mode
    # ------------------------------------------------------------------

    def add_snapshot(self, snapshot_dir: str) -> Tuple[List, np.ndarray, List[int]]:
        """Ingest a new COLMAP snapshot directory on-the-fly.

        Equivalent to load_next_snapshot() but takes an explicit path rather
        than iterating through a pre-discovered list.  Safe to call from the
        trainer loop while training is paused.

        Returns same (new_cams, new_point_mask, new_cam_indices) triple.
        """
        snap_path = str(snapshot_dir)
        self.snapshot_dir = Path(snap_path)

        try:
            cam_extrinsics = read_extrinsics_binary(
                os.path.join(snap_path, "sparse/0", "images.bin"))
            cam_intrinsics = read_intrinsics_binary(
                os.path.join(snap_path, "sparse/0", "cameras.bin"))
        except Exception:
            cam_extrinsics = read_extrinsics_text(
                os.path.join(snap_path, "sparse/0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(
                os.path.join(snap_path, "sparse/0", "cameras.txt"))

        images_folder = os.path.join(snap_path, self.images_subdir)
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=images_folder,
        )

        bin_path = os.path.join(snap_path, "sparse/0", "points3D.bin")
        txt_path = os.path.join(snap_path, "sparse/0", "points3D.txt")
        try:
            xyzs, rgbs, _, point_ids, track_image_ids = \
                read_points3D_binary_with_tracks(bin_path)
        except FileNotFoundError:
            from scene.colmap_loader import read_points3D_text
            xyzs, rgbs, _ = read_points3D_text(txt_path)
            point_ids = list(range(len(xyzs)))
            track_image_ids = [[] for _ in xyzs]

        current_pcd = BasicPointCloud(
            points=xyzs,
            colors=rgbs / 255.0 if rgbs.max() > 1.0 else rgbs,
            normals=np.zeros_like(xyzs),
        )

        self._current_sfm_xyz = xyzs
        self._current_sfm_point_ids = point_ids
        self._point_track_info = {}
        for pid, img_ids in zip(point_ids, track_image_ids):
            self._point_track_info[pid] = img_ids

        if self.last_basic_pcd is None:
            new_point_mask = np.ones(len(xyzs), dtype=bool)
        else:
            prev_set = set(map(tuple, np.round(self.last_basic_pcd.points, 4)))
            new_point_mask = np.array(
                [tuple(np.round(p, 4)) not in prev_set for p in xyzs], dtype=bool)

        self.last_basic_pcd = current_pcd
        self.current_basic_pcd = current_pcd

        new_cam_objects = cameraList_from_camInfos(cam_infos, 1.0, self.model_args)

        new_cams = []
        new_cam_indices = []

        for cam in new_cam_objects:
            name = cam.image_name
            if name in self._cam_name_to_idx:
                idx = self._cam_name_to_idx[name]
                existing = self.train_cameras[idx]
                existing.R = cam.R
                existing.T = cam.T
                existing.FoVx = cam.FoVx
                existing.FoVy = cam.FoVy
                import torch
                from utils.graphics_utils import getWorld2View2, getProjectionMatrix
                existing.world_view_transform = torch.tensor(
                    getWorld2View2(cam.R, cam.T)).transpose(0, 1).cuda()
                existing.projection_matrix = getProjectionMatrix(
                    znear=existing.znear, zfar=existing.zfar,
                    fovX=cam.FoVx, fovY=cam.FoVy).transpose(0, 1).cuda()
                existing.full_proj_transform = (
                    existing.world_view_transform.unsqueeze(0).bmm(
                        existing.projection_matrix.unsqueeze(0))).squeeze(0)
                existing.camera_center = existing.world_view_transform.inverse()[3, :3]
            else:
                idx = len(self.train_cameras)
                self.train_cameras.append(cam)
                self._cam_name_to_idx[name] = idx
                self._colmap_id_to_cam_idx[cam.colmap_id] = idx
                new_cams.append(cam)
                new_cam_indices.append(idx)

        self._n_images = len(self.train_cameras)

        # Recompute extent over the full current camera set. New ingests can
        # dramatically expand scene scale (operator walks 50m away), and stale
        # extent makes densify/prune thresholds and new-point spatial_lr_scale
        # wrong. Existing Gaussians keep the spatial_lr_scale they were born
        # with — fixing that requires a separate LR-warmup-after-reset pass.
        if self.train_cameras:
            self.cameras_extent = getNerfppNorm(self.train_cameras)["radius"]

        return new_cams, new_point_mask, new_cam_indices

    def get_sfm_points_visible_to(
        self, camera
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return SfM points and a visibility mask for the given camera.

        Preferred: use COLMAP track info (handles occlusion correctly).
        Fallback: geometric projection (positive depth + within image bounds).

        Returns:
            sfm_xyz: (N, 3) all current SfM points
            visible_mask: (N,) bool — True if point is visible from camera
        """
        if self._current_sfm_xyz is None:
            return np.empty((0, 3)), np.empty(0, dtype=bool)

        xyzs = self._current_sfm_xyz
        N = len(xyzs)
        visible_mask = np.zeros(N, dtype=bool)

        # Try track-based visibility first
        colmap_id = camera.colmap_id
        track_based = False
        if self._point_track_info:
            for i, pid in enumerate(self._current_sfm_point_ids):
                if colmap_id in self._point_track_info.get(pid, []):
                    visible_mask[i] = True
            if visible_mask.sum() > 0:
                track_based = True

        if not track_based:
            # Geometric fallback: project all points into camera and check
            import torch
            W2C = camera.world_view_transform.T.cpu().numpy()  # actual W2C
            R_w2c = W2C[:3, :3]
            t_w2c = W2C[:3, 3]

            import math
            H = camera.image_height
            W_img = camera.image_width
            fx = W_img / (2.0 * math.tan(camera.FoVx / 2.0))
            fy = H / (2.0 * math.tan(camera.FoVy / 2.0))
            cx = W_img / 2.0
            cy = H / 2.0

            p_cam = (R_w2c @ xyzs.T).T + t_w2c  # (N, 3)
            z = p_cam[:, 2]
            pos_z = z > 0.01
            u = np.where(pos_z, fx * p_cam[:, 0] / np.where(pos_z, z, 1) + cx, -1)
            v = np.where(pos_z, fy * p_cam[:, 1] / np.where(pos_z, z, 1) + cy, -1)
            in_img = (u >= 0) & (u < W_img) & (v >= 0) & (v < H)
            visible_mask = pos_z & in_img

        return xyzs, visible_mask
