"""CPU tests for Plan 6a multi-view windows.

Covers work items 1, 2, 4 and 5 — the parts that are pure numpy over data
already in RAM, so they run without a GPU or the depth_anything_3 package.
Item 3 (DepthAnything3Wrapper.predict_batch) needs the model; only its
pose-normalisation helper is exercised here.

The geometry is synthetic and the right answer is analytic: a known plane, a
known step edge, a known occluder, and a pure-rotation rig.
"""

import importlib.util
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_HERE = os.path.dirname(__file__)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, "..", relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_di = _load("dense_init", os.path.join("scene", "dense_init.py"))
_cv = _load("covisibility", os.path.join("scene", "covisibility.py"))


# ---------------------------------------------------------------------------
# Synthetic rig
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def look_at_w2c(center, target, up_ref=(0.0, 0.0, 1.0)):
    """OpenCV/COLMAP world-to-camera: +z forward, +x right, +y down."""
    center = np.asarray(center, dtype=np.float64)
    z = _unit(np.asarray(target, dtype=np.float64) - center)
    x = _unit(np.cross(z, up_ref))
    y = np.cross(z, x)
    R_c2w = np.stack([x, y, z], axis=1)
    R_w2c = R_c2w.T
    w2c = np.eye(4)
    w2c[:3, :3] = R_w2c
    w2c[:3, 3] = -R_w2c @ center
    return w2c


class FakeCamera:
    """Minimal stand-in for a MEGS-2 Camera (DL3: world_view_transform = W2C.T)."""

    _next_id = 1

    def __init__(self, w2c, fx=100.0, W=80, H=60, name=None, colmap_id=None,
                 image=None):
        self.image_width = W
        self.image_height = H
        self.FoVx = 2.0 * math.atan(W / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(H / (2.0 * fx))
        if colmap_id is None:
            colmap_id = FakeCamera._next_id
            FakeCamera._next_id += 1
        self.colmap_id = int(colmap_id)
        self.image_name = name or f"img_{self.colmap_id:04d}"
        self.original_image = (torch.full((3, H, W), 0.5) if image is None
                               else image)
        self.world_view_transform = torch.tensor(np.asarray(w2c, np.float64)).T

    @property
    def center(self):
        return _cv.camera_center(self)


def make_camera(center, target=(0.0, 10.0, 0.0), **kw):
    return FakeCamera(look_at_w2c(center, target), **kw)


def plane(normal, offset, bounds=None):
    """Plane {X : n.X = offset}, optionally clipped to an axis-aligned box.

    bounds is ((xlo, xhi), (ylo, yhi), (zlo, zhi)); None on any axis is free.
    """
    return {"n": _unit(normal), "c": float(offset), "bounds": bounds}


def render_depth(camera, planes):
    """Analytic camera-frame depth of the nearest surface at each pixel.

    Pixels that hit nothing come back as 0, which the consistency code treats
    as invalid.
    """
    fx, fy, cx, cy = _di._get_camera_intrinsics(camera)
    W, H = camera.image_width, camera.image_height
    w2c = _di._get_w2c(camera)
    R_c2w = w2c[:3, :3].T
    center = -R_c2w @ w2c[:3, 3]

    u, v = np.meshgrid(np.arange(W, dtype=np.float64),
                       np.arange(H, dtype=np.float64))
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    dirs = rays @ R_c2w.T                       # (H, W, 3) world directions

    best = np.full((H, W), np.inf)
    for pl in planes:
        n, c = pl["n"], pl["c"]
        denom = dirs @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            # p_cam = z * ray, so n . (center + z * dir) = c solves for z
            # directly: the ray's third component is 1 by construction.
            z = (c - n @ center) / denom
        hit = np.isfinite(z) & (z > 1e-9)
        X = center + z[..., None] * dirs
        if pl["bounds"] is not None:
            for axis, lim in enumerate(pl["bounds"]):
                if lim is None:
                    continue
                lo, hi = lim
                hit &= (X[..., axis] >= lo) & (X[..., axis] <= hi)
        best = np.where(hit & (z < best), z, best)
    return np.where(np.isfinite(best), best, 0.0)


def make_records(cameras, points, tracks, errors=None):
    """Build CameraRecords from an explicit {cam index: [point rows]} map."""
    point_ids = list(range(len(points)))
    track_info = {pid: [] for pid in point_ids}
    for cam_idx, rows in tracks.items():
        for row in rows:
            track_info[row].append(cameras[cam_idx].colmap_id)
    if errors is None:
        errors = np.zeros(len(points))
    return _cv.build_camera_records(cameras, np.asarray(points, np.float64),
                                    point_ids, track_info, errors)


def grid_points(n_x=8, n_z=8, y=10.0, x_span=3.0, z_span=2.0):
    xs = np.linspace(-x_span, x_span, n_x)
    zs = np.linspace(-z_span, z_span, n_z)
    xx, zz = np.meshgrid(xs, zs)
    return np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], axis=1)


def cfg(**overrides):
    c = _cv.MultiViewConfig()
    for k, v in overrides.items():
        assert hasattr(c, k), k
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# §5.2 — the MVSNet view-selection score
# ---------------------------------------------------------------------------

def test_triangulation_angle_is_the_angle_at_the_point():
    # Two cameras 90 degrees apart as seen from the origin.
    theta = _cv.triangulation_angles_deg(
        np.zeros((1, 3)), np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0]))
    assert theta[0] == pytest.approx(90.0)


def test_mvsnet_score_peaks_at_theta0_and_is_asymmetric():
    c = cfg(theta0_deg=8.0, sigma1_deg=1.0, sigma2_deg=10.0)
    at_peak = _cv.mvsnet_piecewise_gaussian(np.array([8.0]), c)[0]
    assert at_peak == pytest.approx(1.0)

    # A near-collinear baseline is punished hard; a wide one gently. Both
    # sit 6 degrees from the peak.
    too_narrow = _cv.mvsnet_piecewise_gaussian(np.array([2.0]), c)[0]
    too_wide = _cv.mvsnet_piecewise_gaussian(np.array([14.0]), c)[0]
    assert too_narrow < too_wide
    assert too_narrow < 1e-6
    assert too_wide > 0.8


def test_shipped_score_defaults_prefer_a_real_baseline_over_a_collinear_one():
    """Regression on the re-centring: sessions/30d37bfd runs p25 14 deg,
    median 27 deg, max 57 deg, and the DTU defaults (5/1/10) made the score
    rank a 6 deg neighbour ABOVE a 28 deg one."""
    shipped = cfg()
    assert shipped.theta0_deg == 20.0

    def w(theta, c):
        return _cv.mvsnet_piecewise_gaussian(np.array([float(theta)]), c)[0]

    dtu = cfg(theta0_deg=5.0, sigma1_deg=1.0, sigma2_deg=10.0)
    assert w(6, dtu) > w(28, dtu), "the DTU peak preferred near-collinear"
    assert w(28, shipped) > w(6, shipped), "the shipped peak does not"

    # A wide baseline still beats a collinear one, and 6 deg is not excluded.
    assert w(57, shipped) > w(2, shipped)
    assert w(6, shipped) > 0.0


# ---------------------------------------------------------------------------
# §5.1 / §5.3 — gates
# ---------------------------------------------------------------------------

def test_registration_gate_rejects_a_wildly_placed_camera():
    points = grid_points()
    good = [make_camera((x, 0.0, 0.0)) for x in (-1.5, -0.5, 0.5, 1.5)]
    stray = make_camera((0.0, 0.0, 400.0))
    cams = good + [stray]
    tracks = {i: list(range(len(points))) for i in range(len(cams))}
    records = make_records(cams, points, tracks)

    rejected = _cv.registration_gate(records, cfg())
    assert 4 in rejected and "mis-registered" in rejected[4]
    assert not set(rejected) & {0, 1, 2, 3}


def test_registration_gate_rejects_thin_tracks_and_bad_reprojection():
    points = grid_points()
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.0, 0.0, 1.0)]
    tracks = {0: list(range(len(points))),
              1: list(range(5)),                    # too few observations
              2: list(range(len(points)))}
    errors = np.zeros(len(points))
    errors[:] = 0.5
    records = make_records(cams, points, tracks, errors)
    records[2].mean_reproj_error = 12.0             # blown reprojection error

    rejected = _cv.registration_gate(records, cfg())
    assert "tracked points" in rejected[1]
    assert "reprojection error" in rejected[2]
    assert 0 not in rejected


def test_relative_observation_gate_catches_the_weakly_registered_image():
    """The motivating case: img_0008 in sessions/30d37bfd sits at 0.29x the
    session's median observation count while the next-weakest is at 0.44x,
    and its camera centre is mid-pack, so only a relative count separates it.
    """
    points = grid_points(n_x=12, n_z=12)          # 144 points
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.5, -0.5, 0.5, 1.5, 0.0)]
    rows = list(range(len(points)))
    tracks = {0: rows, 1: rows, 2: rows, 3: rows[:64],   # 0.44x the median
              4: rows[:42]}                              # 0.29x the median
    records = make_records(cams, points, tracks)

    rejected = _cv.registration_gate(records, cfg())
    assert 4 in rejected and "weakly registered" in rejected[4]
    assert 3 not in rejected, "the next-weakest image must survive"
    # The absolute count alone would not have caught it.
    assert records[4].n_observations > cfg().min_track_observations
    # Nor would the centre test: it sits in the middle of the rig.
    assert "centre" not in rejected[4]


def test_relative_observation_gate_can_be_switched_off():
    points = grid_points(n_x=12, n_z=12)
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.5, -0.5, 0.5, 1.5, 0.0)]
    rows = list(range(len(points)))
    tracks = {0: rows, 1: rows, 2: rows, 3: rows, 4: rows[:42]}
    records = make_records(cams, points, tracks)
    assert 4 in _cv.registration_gate(records, cfg())
    assert 4 not in _cv.registration_gate(
        records, cfg(min_track_observations_frac=0.0))


def test_a_gated_camera_appears_in_nobody_s_window():
    points = grid_points(n_x=12, n_z=12)
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.5, -0.5, 0.5, 1.5, 0.0)]
    rows = list(range(len(points)))
    records = make_records(cams, points,
                           {0: rows, 1: rows, 2: rows, 3: rows, 4: rows[:42]})
    c = cfg(window_k=4)
    graph = _cv.build_covisibility_graph(records, points, c)

    assert not any(4 in pair for pair in graph)
    for anchor in (0, 1, 2, 3):
        assert 4 not in _cv.select_window(anchor, records, graph, c).members
    # ...and it cannot anchor a window of its own either, so it falls back.
    assert _cv.select_window(4, records, graph, c).degenerate


def test_min_shared_points_gate_drops_thin_pairs():
    points = grid_points()
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.0, 1.0)]
    shared = list(range(len(points)))
    records = make_records(cams, points, {0: shared, 1: shared})

    assert _cv.build_covisibility_graph(records, points, cfg(min_shared_points=30))
    assert not _cv.build_covisibility_graph(
        records, points, cfg(min_shared_points=len(points) + 1))


def test_scale_similarity_gate_drops_a_distant_view():
    points = grid_points()
    near = make_camera((0.0, 0.0, 0.0))
    far = make_camera((0.0, -40.0, 0.0))            # ~5x the median depth
    rows = list(range(len(points)))
    records = make_records([near, far], points, {0: rows, 1: rows})

    assert records[1].median_depth / records[0].median_depth > 2.0
    assert not _cv.build_covisibility_graph(records, points, cfg())


# ---------------------------------------------------------------------------
# §4 / §5.4 — window selection
# ---------------------------------------------------------------------------

def _ring_scene(n=7, radius=2.0):
    points = grid_points()
    cams = [make_camera((radius * math.cos(a), 0.0, radius * math.sin(a)))
            for a in np.linspace(-0.6, 0.6, n)]
    rows = list(range(len(points)))
    return cams, points, make_records(cams, points, {i: rows for i in range(n)})


def test_window_is_anchor_first_and_k_plus_one_long():
    cams, points, records = _ring_scene(n=7)
    c = cfg(window_k=4)
    graph = _cv.build_covisibility_graph(records, points, c)
    window = _cv.select_window(3, records, graph, c)

    assert window.members[0] == 3, "CV1: the anchor must be index 0"
    assert len(window.members) == 5
    assert len(set(window.members)) == 5
    assert window.usable and not window.degenerate


def test_window_shrinks_gracefully_when_few_neighbours_qualify():
    cams, points, records = _ring_scene(n=3)
    c = cfg(window_k=5, min_window_members=3)
    graph = _cv.build_covisibility_graph(records, points, c)
    window = _cv.select_window(0, records, graph, c)
    assert window.members[0] == 0
    assert len(window.members) == 3


def test_angular_diversity_beats_raw_score():
    """Top-K by score alone clusters on one side — an ill-conditioned window."""
    points = grid_points(n_x=10, n_z=10)
    rows = list(range(len(points)))
    anchor = make_camera((0.0, 0.0, 0.0))
    right = [make_camera((x, 0.0, 0.0)) for x in (2.0, 2.05, 2.10)]
    left = make_camera((-2.0, 0.0, 0.0))
    cams = [anchor] + right + [left]

    # The clustered right-hand views share every point; the lone left-hand
    # view shares only half, so it loses on score and can only be picked for
    # its baseline.
    tracks = {0: rows, 1: rows, 2: rows, 3: rows, 4: rows[: len(rows) // 2]}
    records = make_records(cams, points, tracks)
    # This rig is deliberately lopsided, which is exactly what the §5.1
    # centre-outlier gate exists to reject; switch it off to test selection.
    c = cfg(window_k=2, max_center_outlier_ratio=1e9)
    graph = _cv.build_covisibility_graph(records, points, c)

    by_score = sorted(
        ((graph[(0, j)].score, j) for j in (1, 2, 3, 4)), reverse=True)
    top_two = {j for _, j in by_score[:2]}
    assert top_two <= {1, 2, 3}, "top-2 by score all cluster on the right"

    window = _cv.select_window(0, records, graph, c)
    assert window.members[0] == 0
    assert 4 in window.members, "greedy must reach across to the other side"


def test_pure_rotation_rig_is_reported_not_absorbed():
    """§5.5 — the user stood still and panned. Nothing can triangulate."""
    points = grid_points()
    rows = list(range(len(points)))
    cams = [FakeCamera(look_at_w2c((0.0, 0.0, 0.0), t))
            for t in [(0.0, 10.0, 0.0), (2.0, 10.0, 0.0), (-2.0, 10.0, 0.0),
                      (0.0, 10.0, 2.0)]]
    records = make_records(cams, points, {i: rows for i in range(len(cams))})
    c = cfg(window_k=3)
    graph = _cv.build_covisibility_graph(records, points, c)

    window = _cv.select_window(0, records, graph, c)
    assert window.degenerate
    assert not window.usable
    assert window.median_theta_deg < c.degenerate_angle_deg
    assert "pure rotation" in window.reason


def test_anchor_with_no_covisible_neighbour_is_degenerate():
    points = grid_points()
    cams = [make_camera((x, 0.0, 0.0)) for x in (-1.0, 1.0)]
    records = make_records(cams, points, {0: list(range(len(points))), 1: []})
    c = cfg()
    graph = _cv.build_covisibility_graph(records, points, c)
    window = _cv.select_window(0, records, graph, c)
    assert window.degenerate and window.members == [0]


# ---------------------------------------------------------------------------
# §10 — incremental staleness
# ---------------------------------------------------------------------------

def test_new_image_stales_anchors_it_outranks_a_member_of():
    cams, points, records = _ring_scene(n=6)
    c = cfg(window_k=2)
    graph = _cv.build_covisibility_graph(records, points, c)

    reg = _cv.WindowRegistry()
    window = _cv.select_window(0, records, graph, c)
    reg.record(window, records)
    assert not reg.stale

    newcomer = [j for j in range(1, 6) if j not in window.members][0]
    weakest = min(graph[_cv._key(0, m)].score for m in window.neighbours)
    graph[_cv._key(0, newcomer)].score = weakest * 10.0

    assert reg.mark_stale_for_new_image(newcomer, records, graph, c) == [0]
    assert reg.stale == [0]


def test_a_worse_newcomer_leaves_a_full_window_alone():
    cams, points, records = _ring_scene(n=6)
    c = cfg(window_k=2)
    graph = _cv.build_covisibility_graph(records, points, c)

    reg = _cv.WindowRegistry()
    window = _cv.select_window(0, records, graph, c)
    reg.record(window, records)

    newcomer = [j for j in range(1, 6) if j not in window.members][0]
    weakest = min(graph[_cv._key(0, m)].score for m in window.neighbours)
    graph[_cv._key(0, newcomer)].score = weakest * 0.5

    assert reg.mark_stale_for_new_image(newcomer, records, graph, c) == []
    assert reg.stale == []


def test_bundle_adjustment_motion_invalidates_every_window_containing_it():
    cams, points, records = _ring_scene(n=6)
    c = cfg(window_k=2, ba_motion_threshold=0.01)
    graph = _cv.build_covisibility_graph(records, points, c)

    reg = _cv.WindowRegistry()
    for anchor in range(6):
        reg.record(_cv.select_window(anchor, records, graph, c), records)
    assert reg.mark_stale_for_moved_cameras(records, c, 10.0) == []

    moved_idx = 2
    records[moved_idx].center = records[moved_idx].center + np.array([5.0, 0, 0])
    staled = reg.mark_stale_for_moved_cameras(records, c, 10.0)
    assert moved_idx in staled
    for anchor in staled:
        assert moved_idx in reg.windows[anchor].members
    # Centres are re-baselined, so the same motion is not counted twice.
    assert reg.mark_stale_for_moved_cameras(records, c, 10.0) == []


def test_pop_stale_respects_the_refresh_budget():
    reg = _cv.WindowRegistry()
    for anchor in range(5):
        reg.record(_cv.WindowSelection(anchor, [anchor, anchor + 10], 10.0))
        reg.mark_stale(anchor)

    assert reg.pop_stale(0) == []
    assert reg.pop_stale(2) == [0, 1]
    assert reg.stale == [2, 3, 4]
    # Re-running an anchor clears it from the queue.
    reg.record(_cv.WindowSelection(3, [3, 13], 10.0))
    assert reg.stale == [2, 4]


# ---------------------------------------------------------------------------
# §8 — extrinsics pre-normalisation
# ---------------------------------------------------------------------------

def test_normalize_window_extrinsics_gives_unit_median_baseline():
    cams = [make_camera((x * 0.02, 0.0, 0.0)) for x in (0, 1, 2, 3)]
    ext = np.stack([_di._get_w2c(c) for c in cams])

    scaled, scale = _di.normalize_window_extrinsics(ext)
    assert scale == pytest.approx(0.04)             # median of 0.02/0.04/0.06

    # Rotations untouched; translations divided by the scale.
    np.testing.assert_allclose(scaled[:, :3, :3], ext[:, :3, :3])
    np.testing.assert_allclose(scaled[:, :3, 3], ext[:, :3, 3] / scale)

    centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in scaled])
    dists = np.linalg.norm(centers[1:] - centers[0], axis=1)
    assert np.median(dists) == pytest.approx(1.0)
    # The clamp at api.py:333 would have mangled the raw 0.04 spacing.
    assert np.median(dists) > 0.1


def test_normalize_window_extrinsics_is_a_noop_for_a_pure_rotation_rig():
    cams = [FakeCamera(look_at_w2c((0.0, 0.0, 0.0), t))
            for t in [(0.0, 10.0, 0.0), (2.0, 10.0, 0.0), (-2.0, 10.0, 0.0)]]
    ext = np.stack([_di._get_w2c(c) for c in cams])
    scaled, scale = _di.normalize_window_extrinsics(ext)
    assert scale == 1.0
    np.testing.assert_allclose(scaled, ext)


# ---------------------------------------------------------------------------
# §6 — per-pixel cross-view consistency
# ---------------------------------------------------------------------------

BACK_PLANE = plane((0.0, 1.0, 0.0), 10.0)


def _three_view_rig():
    anchor = make_camera((0.0, 0.0, 0.0))
    right = make_camera((3.0, 0.0, 0.0))
    left = make_camera((-3.0, 0.0, 0.0))
    return [anchor, right, left]


# The neighbours sit 3 units either side of the anchor, so a band a few pixels
# wide around the anchor's border falls outside their frusta. That is not
# noise — it is the coverage the count is there to measure — so the analytic
# assertions below are made on the interior.
CORE = (slice(4, -4), slice(4, -4))


def test_a_known_plane_is_consistent_in_every_view():
    cams = _three_view_rig()
    depths = [render_depth(c, [BACK_PLANE]) for c in cams]
    n = _di.cross_view_consistency(depths, cams)

    assert (depths[0] > 0).all(), "the plane fills every anchor pixel"
    assert (n[CORE] == 2).all()


def test_the_border_outside_a_neighbours_frustum_loses_that_vote():
    """n_consistent IS the frustum-intersection count (Plan 6a §6)."""
    cams = _three_view_rig()
    depths = [render_depth(c, [BACK_PLANE]) for c in cams]
    n = _di.cross_view_consistency(depths, cams)

    # Only the left neighbour can see what falls off the anchor's left edge,
    # and vice versa — so the outer columns drop to one vote, never to zero.
    assert (n[:, 0] == 1).all() and (n[:, -1] == 1).all()
    assert n.min() >= 1
    assert (n == 2).mean() > 0.85


def test_a_wrongly_scaled_neighbour_agrees_nowhere():
    cams = _three_view_rig()
    depths = [render_depth(c, [BACK_PLANE]) for c in cams]
    depths[1] = depths[1] * 1.2            # 20% off, far outside the 1% gate
    n = _di.cross_view_consistency(depths, cams)
    assert n.max() == 1, "only the untouched neighbour should ever agree"


def test_a_known_occluder_costs_exactly_the_blocked_neighbour_its_vote():
    """The 'blocking walls' case, solved by construction rather than modelled.

    A slab at y=3 spanning world x in [1.8, 2.4] sits on the right-hand view's
    sight line to the back-plane points with world x in [-1, 1]. It is nowhere
    near the anchor's own rays, and nowhere near the left-hand view's.
    """
    cams = _three_view_rig()
    slab = plane((0.0, 1.0, 0.0), 3.0,
                 bounds=((1.8, 2.4), None, None))
    scene = [BACK_PLANE, slab]
    depths = [render_depth(cams[0], [BACK_PLANE]),
              render_depth(cams[1], scene),
              render_depth(cams[2], scene)]

    # The anchor's own view is untouched by the slab.
    np.testing.assert_allclose(depths[0], render_depth(cams[0], scene))
    assert (depths[1] < 5).any(), "the slab must be visible to the right view"

    n = _di.cross_view_consistency(depths, cams)

    # world_x = 10 * (u - cx) / fx, so |world_x| < 1 is |u - cx| < 10.
    fx, _, cx, _ = _di._get_camera_intrinsics(cams[0])
    u = np.arange(cams[0].image_width)
    occluded = np.abs(u - cx) < 8                        # inside the shadow
    clear = (np.abs(u - cx) > 13) & (np.abs(u - cx) < 34)  # outside it

    rows = slice(4, -4)
    assert (n[rows, occluded] == 1).all(), "the blocked view loses its vote"
    assert (n[rows, clear] == 2).all(), "everything else still agrees"


def test_a_known_step_edge_stays_consistent_away_from_the_discontinuity():
    """A 0.5-unit step at 10 units: 5% deep, five times the 1% depth gate."""
    cams = _three_view_rig()
    near = plane((0.0, 1.0, 0.0), 9.5, bounds=((-100.0, 0.0), None, None))
    far = plane((0.0, 1.0, 0.0), 10.0, bounds=((0.0, 100.0), None, None))
    depths = [render_depth(c, [near, far]) for c in cams]
    n = _di.cross_view_consistency(depths, cams)

    fx, _, cx, _ = _di._get_camera_intrinsics(cams[0])
    u = np.arange(cams[0].image_width)

    # The step is real depth structure, not noise: both plateaus must survive
    # into the anchor's map exactly.
    assert np.isclose(depths[0][:, u < cx - 4], 9.5).all()
    assert np.isclose(depths[0][:, u > cx + 4], 10.0).all()

    # Away from the discontinuity every view still agrees. Near it, the near
    # plane's silhouette shadows a sliver of the far plane from the left view
    # and bilinear sampling straddles the jump, so both are excluded.
    rows = slice(4, -4)
    away = (np.abs(u - cx) > 4) & (np.abs(u - cx) < 34)
    assert (n[rows, away] == 2).all()


def test_consistency_ignores_pixels_with_no_anchor_depth():
    cams = _three_view_rig()
    depths = [render_depth(c, [BACK_PLANE]) for c in cams]
    depths[0] = depths[0].copy()
    depths[0][:10, :] = 0.0
    n = _di.cross_view_consistency(depths, cams)
    assert (n[:10, :] == 0).all()
    assert (n[10:-4, 4:-4] == 2).all()


# ---------------------------------------------------------------------------
# §9 — per-window scale residual
# ---------------------------------------------------------------------------

def _visible_sfm_points():
    return grid_points(n_x=6, n_z=6, y=10.0, x_span=2.0, z_span=1.5)


def test_residual_of_a_correct_window_is_one_and_accepted():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE])
    r, n = _di.window_scale_residual(depth, cam, _visible_sfm_points())

    assert r == pytest.approx(1.0, abs=1e-6)
    assert n == 36
    outcome = _di.resolve_scale_residual(r, n, cfg())
    assert outcome.action == "accept" and outcome.factor == 1.0


def test_a_few_percent_off_is_rescaled_by_one_over_r():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE]) * 1.10
    r, n = _di.window_scale_residual(depth, cam, _visible_sfm_points())

    assert r == pytest.approx(1.10, abs=1e-6)
    outcome = _di.resolve_scale_residual(r, n, cfg())
    assert outcome.action == "rescale"
    assert outcome.factor == pytest.approx(1.0 / 1.10)
    # Applying the scalar puts the window back on the SfM points.
    r2, _ = _di.window_scale_residual(depth * outcome.factor, cam,
                                      _visible_sfm_points())
    assert r2 == pytest.approx(1.0, abs=1e-6)


def test_a_wild_residual_rejects_the_window():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE]) * 2.0
    r, n = _di.window_scale_residual(depth, cam, _visible_sfm_points())
    outcome = _di.resolve_scale_residual(r, n, cfg())
    assert outcome.action == "reject" and "outside" in outcome.reason


def test_too_few_points_to_measure_rejects_rather_than_guesses():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE])
    r, n = _di.window_scale_residual(depth, cam, np.zeros((0, 3)))
    assert n == 0 and math.isnan(r)
    outcome = _di.resolve_scale_residual(r, n, cfg())
    assert outcome.action == "reject"

    r, n = _di.window_scale_residual(depth, cam, _visible_sfm_points()[:3])
    assert n == 3
    assert _di.resolve_scale_residual(r, n, cfg()).action == "reject"


def test_points_behind_the_camera_are_not_counted():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE])
    behind = np.array([[0.0, -5.0, 0.0]])
    pts = np.concatenate([_visible_sfm_points(), behind])
    r, n = _di.window_scale_residual(depth, cam, pts)
    assert n == 36 and r == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Back-projection of the surviving pixels
# ---------------------------------------------------------------------------

def test_masked_backprojection_lands_on_the_known_plane():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE])
    n = _di.cross_view_consistency(
        [depth] + [render_depth(c, [BACK_PLANE]) for c in _three_view_rig()[1:]],
        _three_view_rig())
    keep = n >= 2

    xyz, rgb, weight = _di.depth_to_points_masked(
        depth, cam, keep, cam.original_image, target_n_points=500,
        weights=n.astype(np.float32))

    assert len(xyz) == 500 and len(rgb) == 500 and len(weight) == 500
    np.testing.assert_allclose(xyz[:, 1], 10.0, atol=1e-4)
    assert (weight == 2).all(), "n_consistent rides along as the fusion weight"


def test_masked_backprojection_returns_none_when_nothing_survives():
    cam = make_camera((0.0, 0.0, 0.0))
    depth = render_depth(cam, [BACK_PLANE])
    assert _di.depth_to_points_masked(
        depth, cam, np.zeros_like(depth, dtype=bool), cam.original_image,
        target_n_points=100) is None


def test_bilinear_sample_is_exact_on_a_linear_ramp_and_nan_outside():
    img = np.arange(20, dtype=np.float64).reshape(4, 5)
    got = _di.bilinear_sample(img, np.array([0.0, 2.5, 4.0]),
                              np.array([0.0, 1.5, 3.0]))
    np.testing.assert_allclose(got, [0.0, 10.0, 19.0])

    outside = _di.bilinear_sample(img, np.array([-0.1, 5.0]),
                                 np.array([0.0, 0.0]))
    assert np.isnan(outside).all()


# ---------------------------------------------------------------------------
# Item 3 — what predict_batch actually hands DA3
# ---------------------------------------------------------------------------

class RecordingDA3:
    """Stands in for DepthAnything3, capturing inference()'s kwargs."""

    def __init__(self, depths=None, conf=None, extra=None):
        self.last_kwargs = None
        self.last_images = None
        self._depths = depths
        self._conf = conf
        self._extra = extra or {}

    def inference(self, images, **kwargs):
        self.last_images = images
        self.last_kwargs = kwargs
        depths = (self._depths if self._depths is not None
                  else np.ones((len(images), 6, 8), dtype=np.float32))
        pred = type("Pred", (), dict(self._extra))()
        pred.depth = depths
        if self._conf is not None:
            pred.conf = self._conf
        return pred


def _wrapper_with(model, **cfg_kw):
    w = _di.DepthAnything3Wrapper(_di.DA3Config(**cfg_kw))
    w.model = model
    return w


def _window_inputs(n=3, W=8, H=6):
    cams = [make_camera((x, 0.0, 0.0), W=W, H=H)
            for x in np.linspace(0.0, 2.0, n)]
    images = [torch.full((3, H, W), 0.5) for _ in cams]
    return images, cams


def test_predict_batch_keeps_the_anchor_as_the_reference_view():
    model = RecordingDA3()
    images, cams = _window_inputs()
    _wrapper_with(model).predict_batch(images, cams)

    kw = model.last_kwargs
    assert kw["ref_view_strategy"] == "first", (
        "DA3 defaults to saddle_balanced, which reorders views")
    assert kw["align_to_input_ext_scale"] is True
    assert kw["extrinsics"].shape == (3, 4, 4)
    assert kw["intrinsics"].shape == (3, 3, 3)
    assert len(model.last_images) == 3
    # The anchor's own pose must be the one at index 0.
    np.testing.assert_allclose(
        kw["extrinsics"][0][:3, :3], _di._get_w2c(cams[0])[:3, :3], atol=1e-5)


def test_predict_batch_undoes_its_own_pose_normalisation_on_the_depth():
    """§8 — translations go in divided by the median baseline, depth comes
    back multiplied by it, so the caller sees COLMAP units either way."""
    raw = np.full((3, 6, 8), 2.0, dtype=np.float32)
    model = RecordingDA3(depths=raw)
    images, cams = _window_inputs()
    depths, confs = _wrapper_with(model).predict_batch(images, cams)

    _, scale = _di.normalize_window_extrinsics(
        np.stack([_di._get_w2c(c) for c in cams]))
    assert scale == pytest.approx(1.5)   # median of the 1.0 and 2.0 baselines
    for d in depths:
        assert d.shape == (6, 8)
        np.testing.assert_allclose(d.numpy(), 2.0 * scale, rtol=1e-6)
    assert confs == [None, None, None]

    # A tight rig: the same depths must come back scaled by its baseline.
    images, tight = _window_inputs()
    tight = [make_camera((x, 0.0, 0.0), W=8, H=6)
             for x in np.linspace(0.0, 0.04, 3)]
    model2 = RecordingDA3(depths=raw)
    depths2, _ = _wrapper_with(model2).predict_batch(images, tight)
    sent = model2.last_kwargs["extrinsics"]
    centers = np.stack([-e[:3, :3].T @ e[:3, 3] for e in sent])
    assert np.median(np.linalg.norm(centers[1:] - centers[0], axis=1)) \
        == pytest.approx(1.0, rel=1e-5), "poses are handed over pre-normalised"
    np.testing.assert_allclose(depths2[0].numpy(), 2.0 * 0.03, rtol=1e-5)


def test_predict_batch_can_skip_pose_normalisation():
    model = RecordingDA3()
    images, cams = _window_inputs()
    _wrapper_with(model, prenormalize_extrinsics=False).predict_batch(images, cams)
    np.testing.assert_allclose(
        model.last_kwargs["extrinsics"],
        np.stack([_di._get_w2c(c) for c in cams]).astype(np.float32), atol=1e-5)


def test_predict_batch_returns_confidence_when_the_model_has_it():
    conf = np.full((3, 6, 8), 0.25, dtype=np.float32)
    model = RecordingDA3(conf=conf)
    images, cams = _window_inputs()
    _, confs = _wrapper_with(model).predict_batch(images, cams)
    assert all(c is not None for c in confs)
    np.testing.assert_allclose(confs[0].numpy(), 0.25)


def test_predict_batch_refuses_a_single_view():
    images, cams = _window_inputs(n=1)
    with pytest.raises(ValueError, match="rank-degenerate"):
        _wrapper_with(RecordingDA3()).predict_batch(images, cams)


def test_predict_batch_rejects_a_mismatched_view_count():
    model = RecordingDA3(depths=np.ones((2, 6, 8), dtype=np.float32))
    images, cams = _window_inputs(n=3)
    with pytest.raises(RuntimeError, match="index mapping is not identity"):
        _wrapper_with(model).predict_batch(images, cams)


def test_predict_batch_warns_if_da3_reordered_the_views(caplog):
    model = RecordingDA3(extra={"view_order": [1, 0, 2]})
    images, cams = _window_inputs()
    with caplog.at_level("WARNING"):
        _wrapper_with(model).predict_batch(images, cams)
    assert "reordered views" in caplog.text


def test_predict_batch_resizes_depth_back_to_the_input_resolution():
    model = RecordingDA3(depths=np.ones((3, 3, 4), dtype=np.float32))
    images, cams = _window_inputs(W=8, H=6)
    depths, _ = _wrapper_with(model).predict_batch(images, cams)
    assert all(tuple(d.shape) == (6, 8) for d in depths)


# ---------------------------------------------------------------------------
# Item 6 — the anchor loop inside dense_init_for_new_images
# ---------------------------------------------------------------------------

class WindowScene:
    """Fake ProgressiveScene with real tracks over a known plane."""

    def __init__(self, cameras, points):
        self.train_cameras = list(cameras)
        self.cameras_extent = 10.0
        self._current_sfm_xyz = np.asarray(points, dtype=np.float64)
        self._current_sfm_point_ids = list(range(len(points)))
        self._current_sfm_errors = np.zeros(len(points))
        self._point_track_info = {
            pid: [c.colmap_id for c in self.train_cameras]
            for pid in self._current_sfm_point_ids
        }
        self._cam_name_to_idx = {c.image_name: i
                                 for i, c in enumerate(self.train_cameras)}
        self._colmap_id_to_cam_idx = {c.colmap_id: i
                                      for i, c in enumerate(self.train_cameras)}

    def get_sfm_points_visible_to(self, cam):
        return self._current_sfm_xyz, np.ones(len(self._current_sfm_xyz), bool)


class PlaneDepthModel:
    """Fake DA3 that renders the true depth of a known plane per view."""

    accepts_camera = True

    def __init__(self, planes, depth_gain=1.0):
        self.planes = planes
        self.depth_gain = depth_gain
        self.batch_calls = []
        self.single_calls = []

    def predict_batch(self, images, cameras):
        self.batch_calls.append([c.image_name for c in cameras])
        depths = [torch.tensor(render_depth(c, self.planes) * self.depth_gain,
                               dtype=torch.float32) for c in cameras]
        return depths, [None] * len(cameras)

    def predict(self, image_chw, camera=None):
        self.single_calls.append(camera.image_name if camera else None)
        return torch.tensor(render_depth(camera, self.planes),
                            dtype=torch.float32)


class RecordingGaussians:
    def __init__(self):
        self.get_xyz = torch.zeros((0, 3))
        self.expanded = []
        self.marked = []

    def expand_from_pcd(self, pcd, mask, extent, birth_iter=0):
        self.expanded.append(np.asarray(pcd.points).copy())
        self.get_xyz = torch.zeros((int(mask.sum()), 3))

    def mark_recently_added(self, sl, iteration, grace_iters):
        self.marked.append((sl, iteration, grace_iters))


def _window_dense_cfg(ct, **mv_overrides):
    c = ct.DenseInitConfig()
    c.backend = "da3"
    c.da3_fallback_to_dav2 = False
    c.min_sfm_points_for_alignment = 10
    c.min_sfm_depth_range_fraction = 0.0
    c.target_dense_points_per_image = 400
    c.novelty_distance_threshold = 0.0
    c.max_rejected_fraction = 1.0
    c.multiview.enabled = True
    for k, v in mv_overrides.items():
        assert hasattr(c.multiview, k), k
        setattr(c.multiview, k, v)
    return c


def _patch_wrapper(ct, monkeypatch, model):
    class Ctx:
        def __enter__(self_inner):
            return model

        def __exit__(self_inner, *a):
            return False

    monkeypatch.setattr(ct, "DepthAnything3Wrapper", lambda cfg: Ctx())
    monkeypatch.setattr(ct, "DepthAnythingV2Wrapper", lambda cfg: Ctx())


def _arc_scene(n=5, radius=3.0):
    cams = [make_camera((radius * math.sin(a), 0.0, 0.0), W=48, H=36)
            for a in np.linspace(-0.5, 0.5, n)]
    return WindowScene(cams, grid_points(n_x=8, n_z=8))


def test_window_path_runs_one_da3_call_per_anchor(ct, monkeypatch):
    """Plan 6a §4/§7 — each image's depth comes from exactly one call, and
    only from the window it anchors."""
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE])
    _patch_wrapper(ct, monkeypatch, model)
    gaussians = RecordingGaussians()

    new_cams = scene.train_cameras[:2]
    ct.dense_init_for_new_images(
        gaussians, scene, new_cams, _window_dense_cfg(ct, window_k=3),
        opt=None, current_iter=0, max_gaussians=0)

    assert len(model.batch_calls) == 2, "one window per new image, no more"
    assert not model.single_calls, "no monocular fallback should be needed"
    for call, cam in zip(model.batch_calls, new_cams):
        assert call[0] == cam.image_name, "the anchor must lead its window"
        assert len(call) == 4                      # anchor + K
        assert len(set(call)) == 4

    # The surviving points must land on the plane the model rendered.
    assert gaussians.expanded
    seeded = np.concatenate(gaussians.expanded)
    np.testing.assert_allclose(seeded[:, 1], 10.0, atol=1e-3)


def test_window_registry_survives_across_ingests(ct, monkeypatch):
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE])
    _patch_wrapper(ct, monkeypatch, model)
    registry = _cv.WindowRegistry()
    cfg_obj = _window_dense_cfg(ct, window_k=2)

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[:1], cfg_obj,
        opt=None, current_iter=0, window_registry=registry)
    assert set(registry.windows) == {0}

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[1:2], cfg_obj,
        opt=None, current_iter=1, window_registry=registry)
    assert set(registry.windows) == {0, 1}
    assert registry.windows[0].members[0] == 0


def test_a_pure_rotation_capture_is_surfaced_and_falls_back(ct, monkeypatch):
    """§5.5 — silently producing garbage is worse than saying 'walk sideways'."""
    cams = [FakeCamera(look_at_w2c((0.0, 0.0, 0.0), t), W=48, H=36)
            for t in [(0.0, 10.0, 0.0), (1.5, 10.0, 0.0), (-1.5, 10.0, 0.0)]]
    scene = WindowScene(cams, grid_points(n_x=8, n_z=8))
    model = PlaneDepthModel([BACK_PLANE])
    _patch_wrapper(ct, monkeypatch, model)
    events = []

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, cams[:1], _window_dense_cfg(ct),
        opt=None, current_iter=0,
        event_sink=lambda ev, **f: events.append((ev, f)))

    assert not model.batch_calls, "no window should have been run"
    assert model.single_calls == [cams[0].image_name], "monocular fallback"
    assert [e for e, _ in events] == ["dense_window_degenerate"]
    assert "pure rotation" in events[0][1]["reason"]


def test_a_window_whose_scale_is_wild_falls_back_to_monocular(ct, monkeypatch):
    """§9 — the residual check is a gate, not a rescale-everything knob."""
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE], depth_gain=2.0)   # 2x off the SfM
    _patch_wrapper(ct, monkeypatch, model)

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[:1],
        _window_dense_cfg(ct, window_k=2), opt=None, current_iter=0)

    assert len(model.batch_calls) == 1, "the window ran"
    assert model.single_calls, "...and was rejected, so monocular took over"


def test_a_window_a_few_percent_off_is_rescaled_not_dropped(ct, monkeypatch):
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE], depth_gain=1.08)
    _patch_wrapper(ct, monkeypatch, model)
    gaussians = RecordingGaussians()

    ct.dense_init_for_new_images(
        gaussians, scene, scene.train_cameras[:1],
        _window_dense_cfg(ct, window_k=2), opt=None, current_iter=0)

    assert len(model.batch_calls) == 1
    assert not model.single_calls
    seeded = np.concatenate(gaussians.expanded)
    # The 8% error is corrected by the single scalar, putting the points back
    # on the plane rather than 0.8 units behind it.
    np.testing.assert_allclose(seeded[:, 1], 10.0, atol=2e-2)


def test_multiview_disabled_keeps_the_single_image_path(ct, monkeypatch):
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE])
    _patch_wrapper(ct, monkeypatch, model)
    cfg_obj = _window_dense_cfg(ct)
    cfg_obj.multiview.enabled = False

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[:1], cfg_obj,
        opt=None, current_iter=0)

    assert not model.batch_calls
    assert model.single_calls == [scene.train_cameras[0].image_name]


def test_cold_start_without_tracks_keeps_the_single_image_path(ct, monkeypatch):
    scene = _arc_scene(n=5)
    scene._point_track_info = {}
    model = PlaneDepthModel([BACK_PLANE])
    _patch_wrapper(ct, monkeypatch, model)

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[:1],
        _window_dense_cfg(ct), opt=None, current_iter=0)

    assert not model.batch_calls
    assert model.single_calls


def test_a_failing_window_does_not_kill_the_ingest(ct, monkeypatch):
    scene = _arc_scene(n=5)
    model = PlaneDepthModel([BACK_PLANE])

    def boom(images, cameras):
        raise RuntimeError("CUDA OOM")

    model.predict_batch = boom
    _patch_wrapper(ct, monkeypatch, model)

    ct.dense_init_for_new_images(
        RecordingGaussians(), scene, scene.train_cameras[:1],
        _window_dense_cfg(ct, window_k=2), opt=None, current_iter=0)

    assert model.single_calls, "the anchor falls back instead of raising"


# ---------------------------------------------------------------------------
# Incremental alignment onto the dense cloud already on the ground
# ---------------------------------------------------------------------------

# A plane tilted away from the image plane, so depth genuinely varies across
# the view. On a fronto-parallel plane every ray has the same depth and the
# affine (scale, offset) is rank-deficient — which the fitter now detects.
TILTED = plane((0.45, 1.0, 0.0), 10.0)


def _plane_cloud(cam, planes=None, step=3):
    """World points from a camera's own analytic depth — a stand-in for the
    dense cloud an earlier ingest would have left behind."""
    planes = planes or [TILTED]
    depth = render_depth(cam, planes)
    keep = np.zeros_like(depth, dtype=bool)
    keep[::step, ::step] = True
    out = _di.depth_to_points_masked(depth, cam, keep, cam.original_image,
                                     target_n_points=0)
    return out[0].astype(np.float64)


def test_reference_buffer_keeps_the_nearest_surface_not_the_average():
    """Occlusion-awareness: a camera sees the closest surface down each ray."""
    cam = make_camera((0.0, 0.0, 0.0))
    near = plane((0.0, 1.0, 0.0), 6.0)
    both = np.concatenate([_plane_cloud(cam, [BACK_PLANE]),
                           _plane_cloud(cam, [near])])
    _cells, z = _di.reference_depth_buffer(cam, both, bin_px=4)
    assert len(z) > 100
    np.testing.assert_allclose(z, 6.0, atol=1e-4), "must take the near plane"


def test_alignment_recovers_a_known_scale_and_offset():
    cam = make_camera((0.0, 0.0, 0.0))
    reference = _plane_cloud(cam)
    truth = render_depth(cam, [TILTED])
    assert truth.max() - truth.min() > 0.05 * np.median(truth),         "the fixture must have a real depth span or the fit is degenerate"
    # The window came out 8% small and 0.7 units short of where it belongs.
    warped = (truth - 0.7) / 1.08
    fit = _di.fit_depth_to_reference(warped, cam, reference,
                                     cfg(reference_min_correspondences=100))
    assert fit.ok, fit.reason
    assert not fit.scale_only
    # Scale and offset trade off along the fit line, so the corrected DEPTH is
    # the thing to pin down; the parameters only need to be in the region.
    # 0.5% covers the fixture's own resampling noise: the reference is a
    # step-3 subsample re-projected into 4px cells, then sampled at centres.
    np.testing.assert_allclose(fit.apply(warped), truth, rtol=5e-3)
    assert fit.scale == pytest.approx(1.08, rel=0.02)
    assert fit.offset == pytest.approx(0.7, abs=0.15)


def test_alignment_is_robust_to_a_contaminated_reference():
    """A quarter of the reference is a stray sheet; the Huber fit ignores it."""
    cam = make_camera((0.0, 0.0, 0.0))
    good = _plane_cloud(cam)
    # In FRONT of the surface: the nearest-surface reference buffer already
    # discards anything behind, so only a nearer stray actually reaches the
    # fit — that is the case the Huber weighting has to survive.
    stray = good.copy()
    stray[:, 1] -= 3.0
    rng = np.random.default_rng(0)
    stray = stray[rng.random(len(stray)) < 0.33]
    truth = render_depth(cam, [TILTED])
    warped = truth / 1.05
    fit = _di.fit_depth_to_reference(
        warped, cam, np.concatenate([good, stray]),
        cfg(reference_min_correspondences=100))
    assert fit.ok, fit.reason
    assert fit.scale == pytest.approx(1.05, rel=0.03)
    assert fit.inlier_fraction < 1.0, "the stray sheet must be down-weighted"
    # Without robustness a third of the samples pulled 3 units short would
    # drag the scale far below 1.05.
    assert abs(fit.scale - 1.05) < abs(1.05 - 0.9)


def test_the_first_window_has_nothing_to_align_to():
    cam = make_camera((0.0, 0.0, 0.0))
    fit = _di.fit_depth_to_reference(render_depth(cam, [TILTED]), cam,
                                     None, cfg())
    assert not fit.ok and "no dense cloud" in fit.reason
    # A no-op fit must still be safe to apply.
    np.testing.assert_allclose(fit.apply(np.full((2, 2), 5.0)), 5.0)


def test_a_non_overlapping_window_is_not_aligned():
    """Nothing in common with what is already built -> leave it to §9."""
    cam = make_camera((0.0, 0.0, 0.0))
    # A cloud sitting behind the camera: nothing of it projects into view.
    behind = _plane_cloud(cam).copy()
    behind[:, 1] = -20.0
    fit = _di.fit_depth_to_reference(render_depth(cam, [TILTED]), cam,
                                     behind, cfg())
    assert not fit.ok and fit.n_matched == 0


def test_a_wild_fit_is_rejected_rather_than_applied():
    cam = make_camera((0.0, 0.0, 0.0))
    reference = _plane_cloud(cam)
    c = cfg(reference_min_correspondences=100, reference_max_scale_dev=0.10)
    fit = _di.fit_depth_to_reference(render_depth(cam, [TILTED]) / 2.0,
                                     cam, reference, c)
    assert not fit.ok and "scale" in fit.reason


def test_the_valid_mask_confines_the_fit_to_trusted_pixels():
    cam = make_camera((0.0, 0.0, 0.0))
    reference = _plane_cloud(cam)
    truth = render_depth(cam, [TILTED])
    corrupt = truth.copy()
    corrupt[:, : cam.image_width // 2] = 1.0     # left half is garbage
    mask = np.ones_like(truth, dtype=bool)
    mask[:, : cam.image_width // 2] = False

    c = cfg(reference_min_correspondences=50)
    unmasked = _di.fit_depth_to_reference(corrupt, cam, reference, c)
    masked = _di.fit_depth_to_reference(corrupt, cam, reference, c,
                                        valid_mask=mask)
    assert masked.ok
    # Judge by the corrected depth over the trusted half, not the parameters.
    right = np.s_[:, cam.image_width // 2:]
    np.testing.assert_allclose(masked.apply(corrupt)[right], truth[right],
                               rtol=5e-3)
    err = lambda f: float(np.max(np.abs(f.apply(corrupt)[right] - truth[right])))
    assert err(masked) < err(unmasked), "the garbage half must not steer the fit"


def test_a_flat_wall_falls_back_to_scale_only():
    """Fronto-parallel: every ray the same depth, so (s, c) is rank-deficient.
    Fitting both would return an arbitrary pair that extrapolates wildly."""
    cam = make_camera((0.0, 0.0, 0.0))
    reference = _plane_cloud(cam, [BACK_PLANE])
    truth = render_depth(cam, [BACK_PLANE])
    assert truth.max() == pytest.approx(truth.min())    # genuinely constant

    fit = _di.fit_depth_to_reference(truth / 1.06, cam, reference,
                                     cfg(reference_min_correspondences=100))
    assert fit.ok and fit.scale_only
    assert fit.offset == 0.0, "no offset may be invented from a flat wall"
    assert fit.scale == pytest.approx(1.06, rel=1e-3)


def test_alignment_config_defaults_are_the_measured_ones():
    c = cfg()
    assert c.align_to_reference_cloud is True
    # 0.35 came off the tether sweep on sessions/30d37bfd, not out of the air.
    assert c.reference_sfm_tether == 0.35
    assert 0.0 < c.reference_sfm_tether < 1.0
