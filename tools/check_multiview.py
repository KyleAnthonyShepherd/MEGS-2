"""Run the Plan 6a multi-view window path over a solved COLMAP session.

A deploy smoke test: it exercises DA3 on the real GPU, the covisibility graph,
the per-pixel consistency filter and the incremental cloud alignment, then
writes a PLY you can open — all without shooting a capture.

    python tools/check_multiview.py --session /var/www/spacemapper-server/sessions/<id>/undistorted \\
        --out /tmp/mv.ply

The session directory is the UNDISTORTED snapshot (PINHOLE), i.e. the one the
trainer ingests: it must contain sparse/ and images/.

Deliberately standalone — it does not import scene/__init__.py, so it needs
neither the CUDA rasterizer submodules nor a running trainer.
"""

import argparse
import importlib.util
import json
import math
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# export_to_colmap is the only pycolmap user in DA3's api import chain and we
# never call it; its C++ backend does not load on every platform.
try:
    import pycolmap  # noqa: F401
except Exception:
    sys.modules.setdefault("pycolmap", types.ModuleType("pycolmap"))


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


di = _load("dense_init", "scene/dense_init.py")
cv = _load("covisibility", "scene/covisibility.py")
cl = _load("colmap_loader", "scene/colmap_loader.py")


class SessionCamera:
    """The subset of a MEGS-2 Camera the dense-init path actually reads."""

    def __init__(self, image_id, extr, intr, images_dir):
        import torch
        from PIL import Image

        w2c = np.eye(4)
        w2c[:3, :3] = cl.qvec2rotmat(extr.qvec)
        w2c[:3, 3] = extr.tvec
        self.world_view_transform = torch.tensor(w2c, dtype=torch.float64).T
        self.colmap_id = int(image_id)
        self.image_name = extr.name

        img = Image.open(images_dir / extr.name).convert("RGB")
        W, H = img.size
        self.image_width, self.image_height = W, H
        fx = intr.params[0]
        fy = intr.params[1] if len(intr.params) > 1 else intr.params[0]
        self.FoVx = 2.0 * math.atan(W / (2.0 * fx))
        self.FoVy = 2.0 * math.atan(H / (2.0 * fy))
        self.original_image = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


class SessionScene:
    """Enough ProgressiveScene for get_sfm_points_visible_to()."""

    def __init__(self, cameras, xyz, point_ids, tracks):
        self.train_cameras = cameras
        self._current_sfm_xyz = xyz
        self._current_sfm_point_ids = point_ids
        self._point_track_info = tracks
        centers = np.stack([cv.camera_center(c) for c in cameras])
        self.cameras_extent = float(
            np.linalg.norm(centers - centers.mean(axis=0), axis=1).max())
        self._rows = {}
        for i, pid in enumerate(point_ids):
            for img_id in tracks.get(pid, ()):
                self._rows.setdefault(int(img_id), []).append(i)

    def get_sfm_points_visible_to(self, cam):
        mask = np.zeros(len(self._current_sfm_xyz), dtype=bool)
        rows = self._rows.get(int(cam.colmap_id))
        if rows:
            mask[np.asarray(rows)] = True
        return self._current_sfm_xyz, mask


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\n"
                 "property uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def load_session(session: Path):
    sparse, images_dir = session / "sparse", session / "images"
    if not sparse.exists():
        sparse = session / "sparse" / "0"
    extr = cl.read_extrinsics_binary(str(sparse / "images.bin"))
    intr = cl.read_intrinsics_binary(str(sparse / "cameras.bin"))
    xyz, _rgb, _err, point_ids, track_image_ids = \
        cl.read_points3D_binary_with_tracks(str(sparse / "points3D.bin"))

    order = sorted(extr, key=lambda k: extr[k].name)
    cameras = [SessionCamera(k, extr[k], intr[extr[k].camera_id], images_dir)
               for k in order]
    tracks = {pid: ids for pid, ids in zip(point_ids, track_image_ids)}
    return cameras, np.asarray(xyz, dtype=np.float64), point_ids, tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="the UNDISTORTED snapshot dir (has sparse/ + images/)")
    ap.add_argument("--config", default=str(REPO / "configs/continuous.yaml"))
    ap.add_argument("--out", default=None, help="write a PLY here")
    ap.add_argument("--stats", default=None, help="write per-anchor JSON here")
    args = ap.parse_args()

    import yaml
    raw = yaml.safe_load(open(args.config))["dense_init"]
    da3 = di.DA3Config()
    for k, v in (raw.get("da3") or {}).items():
        if hasattr(da3, k):
            setattr(da3, k, v)
    mv = cv.MultiViewConfig()
    for k, v in (raw.get("multiview") or {}).items():
        if hasattr(mv, k):
            setattr(mv, k, v)
    target = int(raw.get("target_dense_points_per_image", 30000))

    cameras, xyz, point_ids, tracks = load_session(Path(args.session))
    scene = SessionScene(cameras, xyz, point_ids, tracks)
    print(f"{len(cameras)} images, {len(xyz)} sparse points, "
          f"extent {scene.cameras_extent:.2f}")
    print(f"model {da3.model_name}  process_res {da3.process_res}  "
          f"K {mv.window_k}  align {mv.align_to_reference_cloud} "
          f"tether {mv.reference_sfm_tether}")

    records = cv.build_camera_records(cameras, xyz, point_ids, tracks)
    excluded = cv.registration_gate(records, mv)
    for i, why in excluded.items():
        print(f"  GATED {records[i].name}: {why}")
    graph = cv.build_covisibility_graph(records, xyz, mv, excluded)
    print(f"  covisibility graph: {len(graph)} edges")
    if not graph:
        print("NO COVISIBILITY GRAPH — nothing to run"); return 1

    reference, all_xyz, all_rgb, rows = [], [], [], []
    t0 = time.monotonic()
    with di.DepthAnything3Wrapper(da3) as model:
        print(f"model loaded in {time.monotonic() - t0:.0f}s")
        for anchor in sorted(records):
            cam = cameras[anchor]
            row = {"image": cam.image_name}
            if anchor in excluded:
                rows.append({**row, "outcome": "gated"}); continue
            window = cv.select_window(anchor, records, graph, mv)
            if window.degenerate:
                print(f"  {cam.image_name}: DEGENERATE — {window.reason}")
                rows.append({**row, "outcome": "degenerate",
                             "reason": window.reason})
                continue

            wcams = [cameras[m] for m in window.members]
            imgs = [c.original_image for c in wcams]
            t1 = time.monotonic()
            depths, confs = model.predict_batch(imgs, wcams)
            secs = time.monotonic() - t1

            n_cons = di.cross_view_consistency(
                depths, wcams, mv.reproj_px_threshold, mv.depth_rel_threshold)
            keep = n_cons >= mv.n_consistent_min
            if not keep.any():
                print(f"  {cam.image_name}: no consistent pixels")
                rows.append({**row, "outcome": "no_consistent_pixels"}); continue

            depth = depths[0]
            fit = di.ReferenceFit()
            if mv.align_to_reference_cloud:
                ref = np.concatenate(reference) if reference else None
                fit = di.fit_depth_to_reference(depth, cam, ref, mv,
                                                valid_mask=keep)
                if fit.ok:
                    depth = fit.apply(depth)

            sfm_all, vis = scene.get_sfm_points_visible_to(cam)
            r, n_pts = di.window_scale_residual(depth, cam, sfm_all[vis])
            res = di.resolve_scale_residual(r, n_pts, mv)
            if res.action == "reject":
                print(f"  {cam.image_name}: REJECTED — {res.reason}")
                rows.append({**row, "outcome": "rejected",
                             "reason": res.reason})
                continue
            if fit.ok and mv.reference_sfm_tether > 0 and np.isfinite(r) and r > 0:
                depth = depth * (1.0 / r) ** mv.reference_sfm_tether
            elif res.action == "rescale":
                depth = depth * res.factor

            out = di.depth_to_points_masked(
                depth, cam, keep, imgs[0], target_n_points=target,
                weights=n_cons.astype(np.float32))
            if out is None:
                rows.append({**row, "outcome": "no_points"}); continue
            pxyz, prgb, _w = out
            all_xyz.append(pxyz); all_rgb.append(prgb)
            reference.append(pxyz.astype(np.float64))
            rows.append({**row, "outcome": "ok", "points": len(pxyz),
                         "views": len(wcams), "kept": round(float(keep.mean()), 4),
                         "theta": round(window.median_theta_deg, 2),
                         "r": round(float(r), 4), "aligned": bool(fit.ok),
                         "seconds": round(secs, 2)})
            print(f"  {cam.image_name}: {len(pxyz):6d} pts  views={len(wcams)} "
                  f"theta={window.median_theta_deg:5.1f} r={r:.3f} "
                  f"kept={keep.mean():.1%} {secs:.2f}s"
                  + (f"  ALIGN s={fit.scale:.4f} c={fit.offset:+.3f}"
                     if fit.ok else f"  [{fit.reason[:44]}]"))

    ok = [r for r in rows if r.get("outcome") == "ok"]
    if not ok:
        print("NO WINDOW PRODUCED POINTS"); return 1
    total = sum(r["points"] for r in ok)
    print(f"\n{len(ok)}/{len(rows)} windows OK, {total} points, "
          f"{time.monotonic() - t0:.0f}s total, "
          f"mean {sum(r['seconds'] for r in ok)/len(ok):.2f}s per window, "
          f"mean kept {100*sum(r['kept'] for r in ok)/len(ok):.1f}%")

    if args.out:
        write_ply(args.out, np.concatenate(all_xyz), np.concatenate(all_rgb))
        print(f"wrote {args.out} "
              f"({Path(args.out).stat().st_size/1e6:.1f} MB)")
    if args.stats:
        Path(args.stats).write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
