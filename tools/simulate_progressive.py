#!/usr/bin/env python3
"""
simulate_progressive.py — Slice a full COLMAP reconstruction into progressive snapshots.

Given a complete COLMAP model (all images already reconstructed), creates numbered
snapshot directories that simulate the GS-On-The-Fly progressive ingestion format
expected by progressive_train.py.

Usage:
    python tools/simulate_progressive.py \\
        --source /path/to/colmap_root \\
        --output /path/to/output_snapshots \\
        --n-init 3 \\
        [--step 1] \\
        [--order filename|image_id] \\
        [--max-snapshots N]

    --source    Root of a COLMAP dataset.  Expected layout:
                  <source>/sparse/0/cameras.bin  (or .txt)
                  <source>/sparse/0/images.bin
                  <source>/sparse/0/points3D.bin
                  <source>/images/               (optional; symlinked into each snapshot)

    --output    Where to write the numbered snapshot directories.  Created if absent.

    --n-init    Number of images in the first snapshot (default: 3).

    --step      Images added per subsequent snapshot (default: 1).

    --order     How to order images before slicing:
                  filename   — natural sort on image filename (default; good for timestamps)
                  image_id   — COLMAP image ID ascending

    --max-snapshots  Stop after this many snapshots (default: all images).

Output layout (matches ProgressiveScene expectations):
    output/
        1/
            sparse/0/
                cameras.bin  — cameras used by the first n-init images
                images.bin   — first n-init images, with out-of-subset point3D IDs cleared
                points3D.bin — points observed by ≥1 image in this snapshot (tracks filtered)
            images -> <source>/images  (relative symlink; absent if source has no images/)
        2/
            ...

Each snapshot N is a strict cumulative superset of snapshot N-1.
"""

import argparse
import json
import os
import struct
import sys
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Binary I/O helpers
# ---------------------------------------------------------------------------

def _read_bytes(fid, n, fmt):
    data = fid.read(n)
    return struct.unpack("<" + fmt, data)


def _pack(fmt, *args):
    return struct.pack("<" + fmt, *args)


# ---------------------------------------------------------------------------
# Readers (full-fidelity — preserve all fields for faithful round-tripping)
# ---------------------------------------------------------------------------

def read_cameras(path):
    """Returns {camera_id: (model_id, width, height, params_tuple)}."""
    cameras = {}
    with open(path, "rb") as f:
        n, = _read_bytes(f, 8, "Q")
        for _ in range(n):
            cam_id, model_id, width, height = _read_bytes(f, 24, "iiQQ")
            n_params = _model_num_params(model_id)
            params = _read_bytes(f, 8 * n_params, "d" * n_params)
            cameras[cam_id] = (model_id, width, height, params)
    return cameras


def read_images(path):
    """Returns {image_id: dict} with keys id, qvec, tvec, camera_id, name, xys, point3D_ids."""
    images = {}
    with open(path, "rb") as f:
        n, = _read_bytes(f, 8, "Q")
        for _ in range(n):
            img_id, *pose_cam = _read_bytes(f, 64, "idddddddi")
            qvec = tuple(pose_cam[:4])
            tvec = tuple(pose_cam[4:7])
            camera_id = pose_cam[7]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            n2d, = _read_bytes(f, 8, "Q")
            pts2d = _read_bytes(f, 24 * n2d, "ddq" * n2d)
            xys = [(pts2d[i * 3], pts2d[i * 3 + 1]) for i in range(n2d)]
            p3d_ids = [pts2d[i * 3 + 2] for i in range(n2d)]
            images[img_id] = dict(
                id=img_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=name, xys=xys, point3D_ids=p3d_ids,
            )
    return images


def read_points3d(path):
    """Returns {point3d_id: (xyz, rgb, error, [(img_id, pt2d_idx), ...])}."""
    points = {}
    with open(path, "rb") as f:
        n, = _read_bytes(f, 8, "Q")
        for _ in range(n):
            pid, x, y, z, r, g, b, err = _read_bytes(f, 43, "QdddBBBd")
            track_len, = _read_bytes(f, 8, "Q")
            track = _read_bytes(f, 8 * track_len, "ii" * track_len)
            track_pairs = [(track[i * 2], track[i * 2 + 1]) for i in range(track_len)]
            points[pid] = (np.array([x, y, z]), np.array([r, g, b], dtype=np.uint8), err, track_pairs)
    return points


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_cameras(path, cameras_subset):
    """cameras_subset: {cam_id: (model_id, width, height, params_tuple)}"""
    with open(path, "wb") as f:
        f.write(_pack("Q", len(cameras_subset)))
        for cam_id, (model_id, width, height, params) in sorted(cameras_subset.items()):
            f.write(_pack("iiQQ", cam_id, model_id, width, height))
            f.write(_pack("d" * len(params), *params))


def write_images(path, images_subset, valid_point3d_ids):
    """
    images_subset: list of image dicts.
    valid_point3d_ids: set of point3D IDs present in this snapshot.
    Point3D IDs not in the set are written as -1 (unregistered observation).
    """
    with open(path, "wb") as f:
        f.write(_pack("Q", len(images_subset)))
        for img in images_subset:
            f.write(_pack("idddddddi",
                          img["id"],
                          *img["qvec"],
                          *img["tvec"],
                          img["camera_id"]))
            f.write(img["name"].encode("utf-8") + b"\x00")
            n2d = len(img["xys"])
            f.write(_pack("Q", n2d))
            for (x, y), pid in zip(img["xys"], img["point3D_ids"]):
                out_pid = pid if pid in valid_point3d_ids else -1
                f.write(_pack("ddq", x, y, out_pid))


def write_points3d(path, points_subset, image_id_set):
    """
    points_subset: {pid: (xyz, rgb, error, track_pairs)}.
    image_id_set: only track entries from these images are written.
    """
    with open(path, "wb") as f:
        f.write(_pack("Q", len(points_subset)))
        for pid, (xyz, rgb, error, track_pairs) in sorted(points_subset.items()):
            filtered_track = [(iid, p2idx) for iid, p2idx in track_pairs
                              if iid in image_id_set]
            f.write(_pack("QdddBBBd",
                          pid, xyz[0], xyz[1], xyz[2],
                          int(rgb[0]), int(rgb[1]), int(rgb[2]),
                          error))
            f.write(_pack("Q", len(filtered_track)))
            for iid, p2idx in filtered_track:
                f.write(_pack("ii", iid, p2idx))


# ---------------------------------------------------------------------------
# Camera model param counts (COLMAP model IDs)
# ---------------------------------------------------------------------------

_MODEL_PARAMS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}


def _model_num_params(model_id):
    if model_id not in _MODEL_PARAMS:
        raise ValueError(f"Unknown COLMAP camera model_id {model_id}")
    return _MODEL_PARAMS[model_id]


# ---------------------------------------------------------------------------
# Natural sort (handles numeric runs in filenames, e.g. img_10 > img_9)
# ---------------------------------------------------------------------------

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# Core slicing logic
# ---------------------------------------------------------------------------

_PINHOLE_MODEL_IDS = {0, 1}  # SIMPLE_PINHOLE, PINHOLE


def find_sparse_dir(source: Path) -> Path:
    # colmap image_undistorter writes to sparse/ (no 0/ subdir); handle both layouts
    for candidate in [source / "sparse" / "0", source / "sparse", source]:
        if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find cameras.bin/txt under {source}. "
        "Pass the COLMAP dataset root (containing sparse/0/) or the sparse/0/ dir directly."
    )


def needs_undistortion(cameras: dict) -> bool:
    return any(model_id not in _PINHOLE_MODEL_IDS for model_id, *_ in cameras.values())


def run_undistortion(source: Path, sparse_dir: Path, output: Path, max_image_size: int) -> Path:
    """Run colmap image_undistorter; return the undistorted dataset root."""
    import subprocess

    images_path = None
    for candidate in [source / "images", sparse_dir.parent.parent / "images",
                      sparse_dir.parent / "images"]:
        if candidate.is_dir():
            images_path = candidate
            break
    if images_path is None:
        sys.exit(
            "Cannot find images/ directory relative to source. "
            "Pass the COLMAP dataset root as --source."
        )

    undist_dir = output / "_undistorted"
    undist_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colmap", "image_undistorter",
        "--image_path", str(images_path),
        "--input_path", str(sparse_dir),
        "--output_path", str(undist_dir),
        "--output_type", "COLMAP",
        "--max_image_size", str(max_image_size),
    ]
    print(f"Running undistortion (this may take a minute) ...")
    print(f"  {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("'colmap' not found on PATH. Install COLMAP or run undistortion manually.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"colmap image_undistorter failed with exit code {e.returncode}")

    return undist_dir


def build_snapshots(source: Path, output: Path, n_init: int, step: int,
                    order: str, max_snapshots: int,
                    undistort: bool = False, max_image_size: int = 1920):
    # Pre-flight: verify we can actually write to the output location before
    # spending time reading the COLMAP model.
    try:
        output.mkdir(parents=True, exist_ok=True)
        probe = output / ".write_probe"
        probe.touch()
        probe.unlink()
    except PermissionError:
        sys.exit(
            f"Error: cannot write to {output}\n"
            f"Try a path you own, e.g. --output /tmp/progressive_sim or ~/progressive_sim"
        )

    sparse_dir = find_sparse_dir(source)

    # --- Load full model ---
    cam_bin = sparse_dir / "cameras.bin"
    img_bin = sparse_dir / "images.bin"
    pt3_bin = sparse_dir / "points3D.bin"

    if not cam_bin.exists():
        sys.exit(f"cameras.bin not found in {sparse_dir} (text format not yet supported by this tool)")

    print(f"Reading full COLMAP model from {sparse_dir} ...")
    all_cameras = read_cameras(str(cam_bin))
    all_images = read_images(str(img_bin))
    all_points = read_points3d(str(pt3_bin))
    print(f"  {len(all_cameras)} cameras, {len(all_images)} images, {len(all_points)} 3D points")

    # MEGS-2 only handles PINHOLE/SIMPLE_PINHOLE.  GLOMAP typically outputs
    # SIMPLE_RADIAL, which must be undistorted first.
    if needs_undistortion(all_cameras):
        model_names = {v[0] for v in all_cameras.values()}
        if not undistort:
            sys.exit(
                f"Camera model(s) {model_names} are not PINHOLE/SIMPLE_PINHOLE.\n"
                f"MEGS-2 requires undistorted cameras. Re-run with --undistort:\n\n"
                f"  python tools/simulate_progressive.py --undistort "
                f"--source {source} --output {output} ..."
            )
        undist_root = run_undistortion(source, sparse_dir, output, max_image_size)
        # Reload from the undistorted model (colmap writes to sparse/, not sparse/0/)
        sparse_dir = find_sparse_dir(undist_root)
        cam_bin = sparse_dir / "cameras.bin"
        img_bin = sparse_dir / "images.bin"
        pt3_bin = sparse_dir / "points3D.bin"
        print(f"Reloading undistorted model from {sparse_dir} ...")
        all_cameras = read_cameras(str(cam_bin))
        all_images = read_images(str(img_bin))
        all_points = read_points3d(str(pt3_bin))
        print(f"  {len(all_cameras)} cameras, {len(all_images)} images, {len(all_points)} 3D points")
        # Images are now in the undistorted directory
        source = undist_root

    # --- Order images ---
    if order == "filename":
        ordered_ids = sorted(all_images.keys(), key=lambda iid: _natural_key(all_images[iid]["name"]))
    elif order == "image_id":
        ordered_ids = sorted(all_images.keys())
    else:
        sys.exit(f"Unknown --order value: {order!r}")

    total = len(ordered_ids)
    if total < n_init:
        sys.exit(f"Only {total} images but --n-init={n_init}")

    # Build list of (snapshot_number, image_id_set) slices
    slices = []
    snap_num = 1
    count = n_init
    while count <= total:
        slices.append((snap_num, set(ordered_ids[:count])))
        snap_num += 1
        count += step
        if max_snapshots and len(slices) >= max_snapshots:
            break
    # Always include the final complete set if step doesn't land on it
    if ordered_ids and set(ordered_ids) not in [s for _, s in slices]:
        if not max_snapshots or len(slices) < max_snapshots:
            slices.append((snap_num, set(ordered_ids)))

    print(f"Creating {len(slices)} snapshots "
          f"({n_init} images in snap 1, +{step}/snap, up to {total} total)")

    # Resolve images/ source for symlinking
    images_src = None
    for candidate in [source / "images", sparse_dir.parent.parent / "images"]:
        if candidate.is_dir():
            images_src = candidate.resolve()
            break

    # --- Write each snapshot ---
    for snap_num, img_id_set in slices:
        snap_dir = output / str(snap_num)
        out_sparse = snap_dir / "sparse" / "0"
        out_sparse.mkdir(parents=True, exist_ok=True)

        images_in_snap = [all_images[iid] for iid in ordered_ids if iid in img_id_set]
        used_cam_ids = {img["camera_id"] for img in images_in_snap}
        cameras_in_snap = {cid: all_cameras[cid] for cid in used_cam_ids if cid in all_cameras}

        # Points observed by ≥1 image in this snapshot
        points_in_snap = {
            pid: pt
            for pid, pt in all_points.items()
            if any(iid in img_id_set for iid, _ in pt[3])
        }
        valid_p3d_ids = set(points_in_snap.keys())

        write_cameras(str(out_sparse / "cameras.bin"), cameras_in_snap)
        write_images(str(out_sparse / "images.bin"), images_in_snap, valid_p3d_ids)
        write_points3d(str(out_sparse / "points3D.bin"), points_in_snap, img_id_set)

        # Symlink images/ if source has one.
        # Use a relative path so the output tree is portable.
        link = snap_dir / "images"
        if images_src and not link.exists():
            try:
                rel = os.path.relpath(images_src, snap_dir)
                link.symlink_to(rel)
            except OSError as e:
                print(f"  [warn] cannot create images symlink in {snap_dir}: {e}")

        print(f"  snap {snap_num:3d}: {len(images_in_snap):4d} images, "
              f"{len(points_in_snap):6d} 3D points → {snap_dir}")

    print(f"\nDone. Start continuous training, then feed snapshots:")
    print(f"  python continuous_train.py \\")
    print(f"    --model_path /path/to/output \\")
    print(f"    --config configs/continuous.yaml")
    print()
    print(f"  # Then post each snapshot in order, e.g.:")
    print(f"  python tools/simulate_progressive.py --post-snapshots {output} --server http://127.0.0.1:8765")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def post_snapshots(snapshot_root: Path, server_url: str, delay: float):
    """POST each numbered snapshot directory to a running continuous_train server."""
    import urllib.request
    import time as _time

    snap_dirs = sorted(
        [snapshot_root / d for d in os.listdir(snapshot_root)
         if (snapshot_root / d).is_dir() and d.isdigit()],
        key=lambda p: int(p.name),
    )
    if not snap_dirs:
        sys.exit(f"No numbered snapshot directories found in {snapshot_root}")

    url = server_url.rstrip("/") + "/ingest"
    for snap_dir in snap_dirs:
        body = json.dumps({"snapshot_dir": str(snap_dir.resolve())}).encode()
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
            print(f"  POSTed {snap_dir.name} → request_id={result.get('request_id')}")
        except Exception as e:
            print(f"  ERROR posting {snap_dir.name}: {e}", file=sys.stderr)
        if delay > 0:
            _time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(
        description="Slice a COLMAP reconstruction into progressive training snapshots "
                    "or feed existing snapshots to a running continuous_train server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Slice mode
    parser.add_argument("--source", type=Path, default=None,
                        help="COLMAP dataset root (contains sparse/0/) or sparse/0/ directly")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for numbered snapshot folders")
    parser.add_argument("--n-init", type=int, default=3, metavar="N",
                        help="Images in the first snapshot (default: 3)")
    parser.add_argument("--step", type=int, default=1,
                        help="Images added per subsequent snapshot (default: 1)")
    parser.add_argument("--order", choices=["filename", "image_id"], default="filename",
                        help="Image ordering: 'filename' (natural sort) or 'image_id' (default: filename)")
    parser.add_argument("--max-snapshots", type=int, default=0, metavar="N",
                        help="Stop after N snapshots (default: all)")
    parser.add_argument("--undistort", action="store_true",
                        help="Run colmap image_undistorter if cameras are not PINHOLE.")
    parser.add_argument("--max-image-size", type=int, default=1920, metavar="PX",
                        help="Max image dimension for undistortion (default: 1920)")
    # Post mode
    parser.add_argument("--post-snapshots", type=Path, default=None, metavar="DIR",
                        help="Directory of numbered snapshots to POST to a running server")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8765",
                        help="continuous_train server URL (default: http://127.0.0.1:8765)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Seconds to wait between POST requests (default: 0)")
    args = parser.parse_args()

    if args.post_snapshots is not None:
        post_snapshots(args.post_snapshots, args.server, args.delay)
        return

    if args.source is None or args.output is None:
        parser.error("--source and --output are required for slice mode")

    args.output.mkdir(parents=True, exist_ok=True)
    build_snapshots(args.source, args.output, args.n_init, args.step,
                    args.order, args.max_snapshots,
                    undistort=args.undistort, max_image_size=args.max_image_size)


if __name__ == "__main__":
    main()
