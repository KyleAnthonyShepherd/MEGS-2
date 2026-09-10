"""Covisibility graph and multi-view window selection (Plan 6a).

Answers one question: which images go into each DA3 multi-view call.

The covisibility signal is COLMAP's *tracks*, not frustum geometry: each 3D
point in points3D.bin carries the list of images that actually observed it, so
occluders (walls, foliage, glass) are handled for free — the matcher already
failed to match through them.  ProgressiveScene loads this into
`_point_track_info` (progressive_scene.py:155); everything here is set
intersection plus numpy over data already in RAM.

Public surface:
  MultiViewConfig          — all Plan 6a tunables (selection + consistency)
  CameraRecord             — per-image geometry distilled from COLMAP
  build_camera_records     — cameras + tracks -> {cam_index: CameraRecord}
  build_covisibility_graph — records -> {(i, j): CovisEdge}
  select_window            — anchor -> WindowSelection (anchor FIRST, always)
  WindowRegistry           — incremental staleness bookkeeping (§10)

Landmine notes:
  CV1: windows are anchor-first by construction.  DA3's `ref_view_strategy`
       defaults to "saddle_balanced", which *reorders views*; the caller must
       pass ref_view_strategy="first" or index 0 is not the anchor.
  CV2: a window whose median triangulation angle is below
       `degenerate_angle_deg` is a pure-rotation capture.  No MVS method can
       triangulate from it — it is reported, not silently absorbed (§5.5).
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MultiViewConfig:
    """Plan 6a tunables.

    The MVSNet constants (theta0/sigma1/sigma2) started at that paper's DTU
    values (5/1/10) and were re-centred on measured phone capture. On
    `sessions/30d37bfd` (13 images of a room) the pairwise triangulation angle
    runs p25 14 deg, median 27 deg, max 57 deg — so a 5-8 deg peak made the
    score *prefer* the most near-collinear pair available: anchor img_0004 led
    its window with a 6 deg neighbour. Peaking at 20 deg demotes those to last
    without excluding them. Re-measure before trusting these on capture that
    is not a hand-held walk around a room.
    """

    # Master switch.  When false, dense-init keeps the single-image path.
    enabled: bool = False

    # --- window selection (§4, §5) ---
    window_k: int = 5                      # neighbours per anchor (window = K+1)
    min_shared_points: int = 30            # §5.1 gate: evidence of shared surface
    theta0_deg: float = 20.0               # §5.2 MVSNet score peak
    sigma1_deg: float = 6.0                # below theta0: punish hard (collinear)
    sigma2_deg: float = 15.0               # above theta0: punish gently
    scale_ratio_min: float = 0.6           # §5.3 median-depth similarity
    scale_ratio_max: float = 1.7
    degenerate_angle_deg: float = 1.0      # §5.5 pure-rotation detection
    min_window_members: int = 3            # anchor + 2; below this go monocular
    # Floor on the §5.4 diversity multiplier. Without it a camera track that
    # is exactly collinear (walking along a wall) drives every candidate after
    # the first to a diversity of 0, and the window collapses to two views.
    # Diversity should reorder candidates, never eliminate them.
    diversity_floor: float = 0.05

    # --- §5.1 badly-registered-image gate ---
    min_track_observations: int = 20       # points this image actually observes
    # ...and, relative to the session's own median observation count. This is
    # the gate that actually catches the motivating case: img_0008 in
    # sessions/30d37bfd (the one Plan 4 found placed wildly wrong in BOTH arms
    # of its A/B) sits at 0.29x the median, while the next-weakest image is at
    # 0.44x. Neither the absolute count nor the centre test separates them —
    # img_0008's centre is mid-pack among the 13. Calibrated on that one
    # session; 0 disables it.
    min_track_observations_frac: float = 0.35
    # COLMAP px, mean over this image's tracks. NOTE this gate is INERT on
    # GLOMAP output: the incremental-GLOMAP path writes 0.0 as every point's
    # error, so every image reads as perfect. It only bites on reconstructions
    # from COLMAP's own bundle adjustment. 0 disables it.
    max_mean_reproj_error: float = 4.0
    max_center_outlier_ratio: float = 6.0  # centre distance from median, in MADs

    # --- §6 per-pixel cross-view consistency ---
    reproj_px_threshold: float = 1.5       # step 5: ||p - p'||
    depth_rel_threshold: float = 0.01      # step 5: |d_j - z_j| / z_j
    n_consistent_min: int = 2              # keep pixel iff >= this many agree

    # --- §9 per-window scale residual check ---
    residual_accept_tolerance: float = 0.03   # |r - 1| within this -> accept
    residual_reject_low: float = 0.7          # r outside [low, high] -> reject
    residual_reject_high: float = 1.3
    residual_min_points: int = 8              # too few SfM points -> reject

    # --- incremental alignment onto the existing dense cloud ---
    # Measured on sessions/30d37bfd: windows disagree by a per-window AFFINE
    # (scale + an origin offset of 2-3% of depth), which reads as parallel
    # sheets. §9's SfM scalar does not fix it — the sparse textured SfM points
    # do not represent the dense surface, so it estimates the wrong scale. Fit
    # each new window against the geometry the earlier images already built,
    # down the new camera's own rays, and keep §9 as a guard.
    align_to_reference_cloud: bool = True
    reference_min_correspondences: int = 500
    reference_bin_px: int = 4              # cell size for the reference buffer
    reference_huber_frac: float = 0.02     # Huber delta, fraction of depth
    reference_iters: int = 4               # IRLS passes
    reference_min_inlier_fraction: float = 0.5
    reference_max_scale_dev: float = 0.25  # |s - 1| bound before rejecting
    reference_max_offset_frac: float = 0.15  # |c| / median depth bound
    # Below this depth span (as a fraction of median depth) scale and offset
    # are not separable — a fronto-parallel wall gives every ray the same
    # depth — so only a scale is fitted. Real windows on sessions/30d37bfd
    # span ~30% of median depth, well clear of this.
    reference_min_depth_span_frac: float = 0.05
    # Sequential alignment CHAINS error: each window inherits its
    # predecessor's, so absolute scale walks away from the camera poses even
    # while neighbouring windows agree beautifully with each other. Measured on
    # sessions/30d37bfd (13 images): -4.1% end-to-end with no tether. This is a
    # damped pull back to the SfM gauge, applied as (1/r)**tether. 0 = pure
    # chaining, 1 = full §9 correction every window (which reintroduces the
    # SfM scalar's own noise). 0.35 measured best: pancaking 0.30% -> 0.36%
    # while drift halves, -4.1% -> -2.2%. Matters more the longer the session.
    reference_sfm_tether: float = 0.35

    # --- §10 incremental refresh ---
    # Stale anchors re-run per ingest.  Defaults to 0: a refreshed anchor would
    # *append* a second copy of its points until Plan 6 A8 (replace-by-image)
    # exists.  Raise only once A8 lands.
    refresh_budget_per_ingest: int = 0
    # Camera-centre motion (fraction of scene scale) that invalidates a window.
    ba_motion_threshold: float = 0.01


# ---------------------------------------------------------------------------
# Records & edges
# ---------------------------------------------------------------------------

@dataclass
class CameraRecord:
    """Everything window selection needs about one registered image."""

    index: int                    # index into ProgressiveScene.train_cameras
    colmap_id: int
    name: str
    center: np.ndarray            # (3,) world-space camera centre, -R^T t
    point_rows: np.ndarray        # sorted int rows into sfm_xyz that it observes
    median_depth: float           # median camera-frame z over its tracked points
    mean_reproj_error: float      # mean COLMAP reprojection error over its tracks
    n_observations: int


@dataclass
class CovisEdge:
    n_shared: int
    theta_median_deg: float
    score: float


@dataclass
class WindowSelection:
    """Result of select_window.  `members[0]` is always the anchor (CV1)."""

    anchor: int
    members: List[int]
    median_theta_deg: float
    degenerate: bool = False
    reason: str = ""

    @property
    def neighbours(self) -> List[int]:
        return self.members[1:]

    @property
    def usable(self) -> bool:
        return not self.degenerate and len(self.members) >= 2


# ---------------------------------------------------------------------------
# Building records
# ---------------------------------------------------------------------------

def camera_center(camera) -> np.ndarray:
    """World-space camera centre from a MEGS-2 Camera: c = -R^T t (DL3)."""
    w2c = camera.world_view_transform.T.cpu().double().numpy()
    return -w2c[:3, :3].T @ w2c[:3, 3]


def _camera_frame_z(xyz_world: np.ndarray, camera) -> np.ndarray:
    w2c = camera.world_view_transform.T.cpu().double().numpy()
    return (w2c[:3, :3] @ xyz_world.T).T[:, 2] + w2c[2, 3]


def build_camera_records(
    cameras: Sequence,
    sfm_xyz: Optional[np.ndarray],
    point_ids: Optional[Sequence[int]],
    point_track_info: Optional[Dict[int, List[int]]],
    sfm_errors: Optional[np.ndarray] = None,
) -> Dict[int, CameraRecord]:
    """Distil each registered camera into a CameraRecord.

    Args:
        cameras: ProgressiveScene.train_cameras (list index == record index).
        sfm_xyz: (N, 3) current sparse points.
        point_ids: length-N COLMAP point3D ids, row-aligned with sfm_xyz.
        point_track_info: {point_id: [colmap_image_id, ...]} — the covisibility
            signal (progressive_scene.py:155).
        sfm_errors: optional (N,) or (N, 1) COLMAP reprojection errors.

    Cameras with no tracks still get a record (with an empty point set) so the
    §5.1 gate can reject them explicitly rather than silently dropping them.
    """
    records: Dict[int, CameraRecord] = {}
    if cameras is None or not len(cameras):
        return records

    n_points = 0 if sfm_xyz is None else len(sfm_xyz)
    errors = None
    if sfm_errors is not None and n_points:
        errors = np.asarray(sfm_errors, dtype=np.float64).reshape(-1)

    # Invert the track dict once: colmap_image_id -> list of point rows.
    rows_by_image: Dict[int, List[int]] = {}
    if n_points and point_ids is not None and point_track_info:
        for row, pid in enumerate(point_ids):
            for img_id in point_track_info.get(pid, ()):
                rows_by_image.setdefault(int(img_id), []).append(row)

    for idx, cam in enumerate(cameras):
        rows = np.asarray(
            sorted(rows_by_image.get(int(cam.colmap_id), [])), dtype=np.int64)
        if len(rows):
            z = _camera_frame_z(sfm_xyz[rows], cam)
            z = z[np.isfinite(z) & (z > 0)]
            median_depth = float(np.median(z)) if len(z) else 0.0
            mean_err = float(np.mean(errors[rows])) if errors is not None else 0.0
        else:
            median_depth = 0.0
            mean_err = 0.0
        records[idx] = CameraRecord(
            index=idx,
            colmap_id=int(cam.colmap_id),
            name=getattr(cam, "image_name", str(idx)),
            center=camera_center(cam),
            point_rows=rows,
            median_depth=median_depth,
            mean_reproj_error=mean_err,
            n_observations=int(len(rows)),
        )
    return records


def build_camera_records_from_scene(prog_scene) -> Dict[int, CameraRecord]:
    """build_camera_records() bound to a ProgressiveScene's current snapshot."""
    return build_camera_records(
        prog_scene.train_cameras,
        getattr(prog_scene, "_current_sfm_xyz", None),
        getattr(prog_scene, "_current_sfm_point_ids", None),
        getattr(prog_scene, "_point_track_info", None),
        getattr(prog_scene, "_current_sfm_errors", None),
    )


# ---------------------------------------------------------------------------
# §5.1 — is this image trustworthy at all?
# ---------------------------------------------------------------------------

def registration_gate(
    records: Dict[int, CameraRecord], cfg: MultiViewConfig
) -> Dict[int, str]:
    """Return {index: reason} for every camera that fails the §5.1 gate.

    Badly-registered images matter more than they look: DA3's Umeyama pose
    alignment only turns on RANSAC at >= 10 views (api.py:341), so in a 6-view
    window one wild pose is a plain least-squares outlier that drags the whole
    window's scale.
    """
    rejected: Dict[int, str] = {}
    if not records:
        return rejected

    order = sorted(records)
    centers = np.stack([records[i].center for i in order])
    dists = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
    mad = float(np.median(np.abs(dists - np.median(dists))))
    # A zero MAD (two cameras, or a perfectly regular rig) disables the test
    # rather than rejecting every camera.
    limit = (float(np.median(dists)) + cfg.max_center_outlier_ratio * mad
             if mad > 1e-12 else np.inf)

    obs = np.array([records[i].n_observations for i in order], dtype=np.float64)
    median_obs = float(np.median(obs)) if len(obs) else 0.0
    obs_floor = (cfg.min_track_observations_frac * median_obs
                 if cfg.min_track_observations_frac > 0 else 0.0)

    for idx, dist in zip(order, dists):
        rec = records[idx]
        if rec.n_observations < cfg.min_track_observations:
            rejected[idx] = (f"only {rec.n_observations} tracked points "
                             f"(need {cfg.min_track_observations})")
        elif rec.n_observations < obs_floor:
            rejected[idx] = (
                f"{rec.n_observations} tracked points is "
                f"{rec.n_observations / median_obs:.2f}x the session median "
                f"{median_obs:.0f} (need {cfg.min_track_observations_frac:.2f}x)"
                " — weakly registered")
        elif (cfg.max_mean_reproj_error > 0
                and rec.mean_reproj_error > cfg.max_mean_reproj_error):
            rejected[idx] = (f"mean reprojection error "
                             f"{rec.mean_reproj_error:.2f}px "
                             f"(limit {cfg.max_mean_reproj_error:.2f})")
        elif dist > limit:
            rejected[idx] = (f"camera centre {dist:.3f} from median "
                             f"(limit {limit:.3f}) — likely mis-registered")
        elif rec.median_depth <= 0.0:
            rejected[idx] = "no positive-depth tracked points"
    return rejected


# ---------------------------------------------------------------------------
# §5.2 — MVSNet view-selection score
# ---------------------------------------------------------------------------

def triangulation_angles_deg(
    points: np.ndarray, c_i: np.ndarray, c_j: np.ndarray
) -> np.ndarray:
    """Angle at each point P between (c_i - P) and (c_j - P), in degrees."""
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return np.zeros(0)
    v_i = np.asarray(c_i, dtype=np.float64)[None, :] - points
    v_j = np.asarray(c_j, dtype=np.float64)[None, :] - points
    n_i = np.linalg.norm(v_i, axis=1)
    n_j = np.linalg.norm(v_j, axis=1)
    ok = (n_i > 1e-12) & (n_j > 1e-12)
    cos = np.ones(len(points))
    cos[ok] = np.sum(v_i[ok] * v_j[ok], axis=1) / (n_i[ok] * n_j[ok])
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def mvsnet_piecewise_gaussian(theta_deg: np.ndarray,
                              cfg: MultiViewConfig) -> np.ndarray:
    """MVSNet's per-point view-selection weight (arXiv:1804.02505).

    Asymmetric on purpose: a too-small baseline (near-collinear cameras) is
    punished hard because depth is genuinely unconstrained there; a large
    baseline is punished gently because its real cost is appearance change,
    which degrades gradually.  This is Plan 6a §3's "co-linearity limit".
    """
    d = np.asarray(theta_deg, dtype=np.float64) - cfg.theta0_deg
    sigma = np.where(d <= 0, cfg.sigma1_deg, cfg.sigma2_deg)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def _scale_similar(rec_i: CameraRecord, rec_j: CameraRecord,
                   cfg: MultiViewConfig) -> bool:
    """§5.3 — reject pairs whose median depths differ too much, either way."""
    if rec_i.median_depth <= 0 or rec_j.median_depth <= 0:
        return False
    r = rec_j.median_depth / rec_i.median_depth
    return (cfg.scale_ratio_min <= r <= cfg.scale_ratio_max
            and cfg.scale_ratio_min <= 1.0 / r <= cfg.scale_ratio_max)


def build_covisibility_graph(
    records: Dict[int, CameraRecord],
    sfm_xyz: np.ndarray,
    cfg: MultiViewConfig,
    excluded: Optional[Dict[int, str]] = None,
) -> Dict[Tuple[int, int], CovisEdge]:
    """Pairwise covisibility over COLMAP tracks.  Keys are canonical (i < j).

    `sfm_xyz` must be the same cloud the records' `point_rows` index into.
    Pairs failing the §5.1 / §5.2 / §5.3 gates are simply absent from the graph.
    """
    if excluded is None:
        excluded = registration_gate(records, cfg)

    xyz = np.asarray(sfm_xyz, dtype=np.float64)
    usable = [i for i in sorted(records) if i not in excluded]
    graph: Dict[Tuple[int, int], CovisEdge] = {}

    for a, i in enumerate(usable):
        rec_i = records[i]
        if not len(rec_i.point_rows):
            continue
        for j in usable[a + 1:]:
            rec_j = records[j]
            shared = np.intersect1d(rec_i.point_rows, rec_j.point_rows,
                                    assume_unique=True)
            if len(shared) < cfg.min_shared_points:
                continue
            if not _scale_similar(rec_i, rec_j, cfg):
                continue
            theta = triangulation_angles_deg(xyz[shared], rec_i.center,
                                             rec_j.center)
            score = float(np.sum(mvsnet_piecewise_gaussian(theta, cfg)))
            if score <= 0.0:
                continue
            graph[(i, j)] = CovisEdge(
                n_shared=int(len(shared)),
                theta_median_deg=float(np.median(theta)),
                score=score,
            )

    logger.debug("[covis] %d edges over %d usable of %d cameras",
                 len(graph), len(usable), len(records))
    return graph


# ---------------------------------------------------------------------------
# §5.4 — angular diversity: pick the set, not the top K
# ---------------------------------------------------------------------------

def _baseline_diversity(c_anchor: np.ndarray, c_cand: np.ndarray,
                        selected_centers: List[np.ndarray]) -> float:
    """Min angle between the candidate's baseline and each selected baseline.

    Normalised to [0, 1] (1 = 180 deg apart).  An empty selection is maximally
    diverse, so the first pick is by score alone.
    """
    if not selected_centers:
        return 1.0
    v = c_cand - c_anchor
    n_v = np.linalg.norm(v)
    if n_v < 1e-12:
        return 0.0
    best = math.pi
    for c_s in selected_centers:
        w = c_s - c_anchor
        n_w = np.linalg.norm(w)
        if n_w < 1e-12:
            continue
        cos = float(np.dot(v, w) / (n_v * n_w))
        best = min(best, math.acos(max(-1.0, min(1.0, cos))))
    return best / math.pi


def select_window(
    anchor: int,
    records: Dict[int, CameraRecord],
    graph: Dict[Tuple[int, int], CovisEdge],
    cfg: MultiViewConfig,
) -> WindowSelection:
    """Greedy score x diversity selection of K neighbours for `anchor`.

    Do NOT just take the top K by score: they cluster on one side of the
    anchor, which is an ill-conditioned configuration (§5.4).
    """
    if anchor not in records:
        return WindowSelection(anchor, [anchor], 0.0, True,
                               "anchor has no camera record")

    candidates: Dict[int, CovisEdge] = {}
    for (i, j), edge in graph.items():
        if i == anchor:
            candidates[j] = edge
        elif j == anchor:
            candidates[i] = edge

    if not candidates:
        return WindowSelection(anchor, [anchor], 0.0, True,
                               "no covisible neighbour survived the gates")

    c_anchor = records[anchor].center
    selected: List[int] = []
    selected_centers: List[np.ndarray] = []
    remaining = dict(candidates)

    floor = cfg.diversity_floor
    while len(selected) < cfg.window_k and remaining:
        best_j, best_val = None, 0.0
        for j, edge in sorted(remaining.items()):
            diversity = _baseline_diversity(
                c_anchor, records[j].center, selected_centers)
            val = edge.score * (floor + (1.0 - floor) * diversity)
            if val > best_val:
                best_j, best_val = j, val
        if best_j is None:
            break
        selected.append(best_j)
        selected_centers.append(records[best_j].center)
        remaining.pop(best_j)

    if not selected:
        return WindowSelection(anchor, [anchor], 0.0, True,
                               "no neighbour scored above zero")

    median_theta = float(np.median(
        [candidates[j].theta_median_deg for j in selected]))
    members = [anchor] + selected          # CV1: anchor first, always

    if median_theta < cfg.degenerate_angle_deg:
        return WindowSelection(
            anchor, members, median_theta, True,
            f"median triangulation angle {median_theta:.2f} deg < "
            f"{cfg.degenerate_angle_deg} deg — the capture is effectively a "
            "pure rotation; no MVS method can triangulate from it")
    if len(members) < cfg.min_window_members:
        return WindowSelection(
            anchor, members, median_theta, True,
            f"window has {len(members)} views "
            f"(need {cfg.min_window_members})")

    return WindowSelection(anchor, members, median_theta)


def select_all_windows(
    anchors: Sequence[int],
    records: Dict[int, CameraRecord],
    graph: Dict[Tuple[int, int], CovisEdge],
    cfg: MultiViewConfig,
) -> Dict[int, WindowSelection]:
    return {a: select_window(a, records, graph, cfg) for a in anchors}


# ---------------------------------------------------------------------------
# §10 — incremental behaviour
# ---------------------------------------------------------------------------

def _key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


class WindowRegistry:
    """Tracks which anchors' windows are current and which need re-running.

    Steady state is 1 DA3 pass per new photo plus a bounded refresh queue: a
    new image can displace a neighbour in nearby anchors' windows, and bundle
    adjustment moving a camera invalidates every window containing it.
    """

    def __init__(self):
        self.windows: Dict[int, WindowSelection] = {}
        self._stale: List[int] = []
        self._centers: Dict[int, np.ndarray] = {}

    # -- bookkeeping -------------------------------------------------------

    def record(self, window: WindowSelection,
               records: Optional[Dict[int, CameraRecord]] = None) -> None:
        self.windows[window.anchor] = window
        if window.anchor in self._stale:
            self._stale.remove(window.anchor)
        if records is not None:
            for m in window.members:
                if m in records:
                    self._centers[m] = np.array(records[m].center)

    def mark_stale(self, anchor: int) -> None:
        if anchor in self.windows and anchor not in self._stale:
            self._stale.append(anchor)

    @property
    def stale(self) -> List[int]:
        return list(self._stale)

    def pop_stale(self, budget: int) -> List[int]:
        """Take up to `budget` stale anchors, oldest first."""
        if budget <= 0:
            return []
        taken, self._stale = self._stale[:budget], self._stale[budget:]
        return taken

    # -- staleness sources -------------------------------------------------

    def mark_stale_for_new_image(
        self,
        new_index: int,
        records: Dict[int, CameraRecord],
        graph: Dict[Tuple[int, int], CovisEdge],
        cfg: MultiViewConfig,
    ) -> List[int]:
        """Stale any anchor for which `new_index` now outranks a current member.

        Cheaper than reselecting every window: an anchor's set can only change
        if the newcomer beats its weakest current neighbour, or the window was
        below capacity to begin with.
        """
        newly: List[int] = []
        for anchor, window in self.windows.items():
            if anchor == new_index or new_index in window.members:
                continue
            new_edge = graph.get(_key(anchor, new_index))
            if new_edge is None:
                continue
            member_scores = [
                graph[_key(anchor, m)].score
                for m in window.neighbours if _key(anchor, m) in graph
            ]
            if len(window.neighbours) < cfg.window_k or (
                    member_scores and new_edge.score > min(member_scores)):
                self.mark_stale(anchor)
                if anchor not in newly:
                    newly.append(anchor)
        return newly

    def mark_stale_for_moved_cameras(
        self,
        records: Dict[int, CameraRecord],
        cfg: MultiViewConfig,
        scene_scale: float,
    ) -> List[int]:
        """Stale windows whose members bundle adjustment has since moved.

        Also refreshes the stored centres, so each camera's motion is measured
        against its last *seen* position rather than accumulating.
        """
        threshold = cfg.ba_motion_threshold * max(float(scene_scale), 1e-9)
        moved = set()
        for idx, rec in records.items():
            prev = self._centers.get(idx)
            if prev is not None and np.linalg.norm(rec.center - prev) > threshold:
                moved.add(idx)
            self._centers[idx] = np.array(rec.center)
        if not moved:
            return []
        newly: List[int] = []
        for anchor, window in self.windows.items():
            if moved.intersection(window.members):
                self.mark_stale(anchor)
                if anchor not in newly:
                    newly.append(anchor)
        return newly
