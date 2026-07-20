"""Benchmark dense-init backends (DAv2 vs DA3) on a recorded session.

GPU machine required. For each backend, runs monocular depth + RANSAC
alignment per camera and reports aligned-depth RMSE against SfM sparse
depths, inlier counts, per-image latency, and peak VRAM — as a markdown
table. The decision rule (Plan 2 Milestone 4.4): flip the default backend
in configs/continuous.yaml only if DA3 wins on quality at acceptable VRAM,
and commit this table with the flip.

Downstream iterations-to-PSNR comparison is covered by running the
regression harness twice with dense_init.backend toggled, not by this
script.

Usage:
  python tools/bench_dense_init.py --session-dir /path/to/session \
      [--backends dav2,da3] [--da3-model depth-anything/DA3-SMALL] \
      [--out bench_dense_init.md]

--session-dir needs the snapshot layout: images/ + sparse/0/{cameras,images,points3D}.bin
"""

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scene.dense_init import (  # noqa: E402
    DepthAnythingV2Wrapper, DepthAnything3Wrapper, DAv2Config, DA3Config,
    RansacConfig, align_depth_to_sfm, AlignmentFailed, validate_alignment,
    project_world_to_image,
)


def load_cameras_and_points(session_dir: Path):
    """Load MEGS-2 Camera objects + sparse points + per-point track info."""
    from scene.progressive_scene import ProgressiveScene
    model_args = Namespace(resolution=-1, data_device="cuda", images="images",
                           white_background=False, sg_degree=3, eval=False,
                           model_path="", source_path=str(session_dir))
    scene = ProgressiveScene(source_path_dir=None, model_args=model_args)
    scene.add_snapshot(str(session_dir))
    return scene


def aligned_rmse(depth_map, camera, sfm_xyz_visible, a, b):
    """RMSE of a*pred+b vs SfM depth over valid projected samples."""
    H, W = depth_map.shape
    u, v, z_true = project_world_to_image(sfm_xyz_visible, camera)
    valid = (z_true > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if valid.sum() == 0:
        return float("nan"), 0
    ui = u[valid].astype(int).clip(0, W - 1)
    vi = v[valid].astype(int).clip(0, H - 1)
    z_pred = depth_map[vi, ui].numpy().astype(np.float64)
    finite = np.isfinite(z_pred) & (z_pred != 0)
    if finite.sum() == 0:
        return float("nan"), 0
    resid = a * z_pred[finite] + b - z_true[valid][finite]
    return float(np.sqrt((resid ** 2).mean())), int(finite.sum())


def bench_backend(backend: str, scene, ransac_cfg, args):
    import torch
    rows = []
    scene_scale = scene.cameras_extent

    if backend == "dav2":
        wrapper = DepthAnythingV2Wrapper(DAv2Config(model_size=args.dav2_model))
    else:
        wrapper = DepthAnything3Wrapper(DA3Config(
            model_name=args.da3_model,
            conditioning=not args.no_conditioning))

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.monotonic()
    with wrapper as model:
        load_s = time.monotonic() - t_load0
        pass_camera = getattr(model, "accepts_camera", False)
        for cam in scene.train_cameras:
            sfm_xyz_all, visible = scene.get_sfm_points_visible_to(cam)
            sfm_xyz = sfm_xyz_all[visible]
            if len(sfm_xyz) < 10:
                rows.append((cam.image_name, "skip (few SfM pts)",
                             None, None, None, None))
                continue
            t0 = time.monotonic()
            if pass_camera:
                depth = model.predict(cam.original_image.cpu(), camera=cam)
            else:
                depth = model.predict(cam.original_image.cpu())
            predict_s = time.monotonic() - t0
            try:
                res = align_depth_to_sfm(depth, cam, sfm_xyz, ransac_cfg,
                                         scene_scale=scene_scale)
            except AlignmentFailed as e:
                rows.append((cam.image_name, f"align failed: {e}",
                             None, None, predict_s, None))
                continue
            ok, reason = validate_alignment(backend, res)
            status = "ok" if ok else f"REJECTED: {reason}"
            rmse, n_samples = aligned_rmse(depth, cam, sfm_xyz, res.a, res.b)
            rows.append((cam.image_name, status, rmse / scene_scale,
                         res.n_inliers, predict_s, res.a))
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {"rows": rows, "load_s": load_s, "peak_vram_mb": peak_vram_mb}


def to_markdown(results: dict, scene) -> str:
    lines = ["# Dense-init backend benchmark",
             f"\nScene: {scene.n_images} cameras, extent={scene.cameras_extent:.2f}\n"]
    summary = []
    for backend, r in results.items():
        lines.append(f"\n## {backend}\n")
        lines.append("| image | status | RMSE/extent | inliers | predict s | a |")
        lines.append("|---|---|---|---|---|---|")
        rmses, lats = [], []
        n_ok = 0
        for name, status, rmse, inl, lat, a in r["rows"]:
            fmt = lambda x, p: ("" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}")
            lines.append(f"| {name} | {status} | {fmt(rmse,4)} | "
                         f"{inl if inl is not None else ''} | {fmt(lat,2)} | {fmt(a,3)} |")
            if status == "ok" and rmse is not None and np.isfinite(rmse):
                rmses.append(rmse)
                n_ok += 1
            if lat is not None:
                lats.append(lat)
        summary.append((backend, n_ok, len(r["rows"]),
                        float(np.mean(rmses)) if rmses else float("nan"),
                        float(np.mean(lats)) if lats else float("nan"),
                        r["load_s"], r["peak_vram_mb"]))

    lines.append("\n## Summary\n")
    lines.append("| backend | ok/total | mean RMSE/extent | mean predict s | load s | peak VRAM MB |")
    lines.append("|---|---|---|---|---|---|")
    for b, n_ok, n, rmse, lat, load_s, vram in summary:
        lines.append(f"| {b} | {n_ok}/{n} | {rmse:.4f} | {lat:.2f} | "
                     f"{load_s:.1f} | {vram:.0f} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--backends", default="dav2,da3")
    ap.add_argument("--dav2-model", default="small")
    ap.add_argument("--da3-model", default="depth-anything/DA3-SMALL")
    ap.add_argument("--no-conditioning", action="store_true",
                    help="disable COLMAP pose/intrinsics conditioning for DA3")
    ap.add_argument("--out", default="bench_dense_init.md")
    args = ap.parse_args()

    scene = load_cameras_and_points(Path(args.session_dir))
    ransac_cfg = RansacConfig()
    results = {}
    for backend in args.backends.split(","):
        backend = backend.strip()
        print(f"[bench] running {backend} over {scene.n_images} cameras...")
        results[backend] = bench_backend(backend, scene, ransac_cfg, args)

    md = to_markdown(results, scene)
    Path(args.out).write_text(md)
    print(md)
    print(f"[bench] wrote {args.out}")


if __name__ == "__main__":
    main()
