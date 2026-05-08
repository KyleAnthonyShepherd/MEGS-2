# User Setup Session Notes
# Saved: 2026-05-08

## Hardware
GPU: NVIDIA GeForce GTX 1660 Ti
VRAM: 6144 MiB (6 GB)
Driver: 590.48.01
CUDA driver capability: 13.1 (can run any CUDA runtime <= 12.x)
GPU architecture: Turing (SM 7.5)

## Software
OS: Ubuntu (home server)
Python: 3.12.3 (system)
Service: /var/www/spacemapper-server (has its own venv)
MEGS-2 clone: /home/user/MEGS-2   (actual username: dragonsmith)
Branch: claude/implement-progressive-megs2-53Kuj

## Data
Session: 30d37bfd
Original sparse reconstruction: /var/www/spacemapper-server/sessions/30d37bfd/sparse/0/
  Files: cameras.bin, frames.bin, images.bin, points3D.bin, rigs.bin
  Note: frames.bin and rigs.bin are GLOMAP-specific extras; progressive_scene.py ignores them
Original images: /var/www/spacemapper-server/sessions/30d37bfd/images/
Undistorted images: /var/www/spacemapper-server/sessions/30d37bfd/undistorted/images/
Snapshot tree:
  /var/www/spacemapper-server/sessions/30d37bfd/snapshots/1/sparse/0/
    cameras.bin   (from undistorted/)
    images.bin    (from undistorted/)
    points3D.bin  (from undistorted/)
Model output: /var/www/spacemapper-server/sessions/30d37bfd/output/

## Environment decision
New venv at: /var/www/spacemapper-server/pmegs2-venv
PyTorch: 2.4.1 with cu121 wheels (Driver 590 supports CUDA 12.x; cu121 is stable)

---

## Environment Setup (working sequence)

### 1. System packages
```
sudo apt-get install -y ninja-build build-essential python3.12-dev
# CUDA toolkit (if nvcc missing):
sudo apt-get install -y cuda-toolkit-12-1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

### 2. Create venv
```
python3.12 -m venv /var/www/spacemapper-server/pmegs2-venv
source /var/www/spacemapper-server/pmegs2-venv/bin/activate
pip install --upgrade pip wheel
```

### 3. PyTorch (must come BEFORE submodule builds)
```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Python deps (skip submodule lines from requirements.txt)
```
pip install tqdm plyfile
pip install transformers>=4.40.0 pyyaml scipy
```

### 5. CUDA submodules (requires python3.12-dev for Python.h)
```
export TORCH_CUDA_ARCH_LIST="7.5"
export FORCE_CUDA=1
PYTHON=/var/www/spacemapper-server/pmegs2-venv/bin/python

$PYTHON -m pip install --no-build-isolation submodules/simple-knn
$PYTHON -m pip install --no-build-isolation submodules/diff-gaussian-rasterization
$PYTHON -m pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms
$PYTHON -m pip install --no-build-isolation submodules/diff-gaussian-rasterization_ms_light
```

### 6. Data preparation
GLOMAP outputs a distorted camera model (SIMPLE_RADIAL or similar).
MEGS-2 only handles PINHOLE. Must undistort first:
```
colmap image_undistorter \
  --image_path  /var/www/spacemapper-server/sessions/30d37bfd/images \
  --input_path  /var/www/spacemapper-server/sessions/30d37bfd/sparse/0 \
  --output_path /var/www/spacemapper-server/sessions/30d37bfd/undistorted \
  --output_type COLMAP \
  --max_image_size 1920
```
Then wrap in snapshot layout:
```
mkdir -p /var/www/spacemapper-server/sessions/30d37bfd/snapshots/1/sparse/0
cp /var/www/spacemapper-server/sessions/30d37bfd/undistorted/sparse/cameras.bin  \
   /var/www/spacemapper-server/sessions/30d37bfd/snapshots/1/sparse/0/
cp /var/www/spacemapper-server/sessions/30d37bfd/undistorted/sparse/images.bin   \
   /var/www/spacemapper-server/sessions/30d37bfd/snapshots/1/sparse/0/
cp /var/www/spacemapper-server/sessions/30d37bfd/undistorted/sparse/points3D.bin \
   /var/www/spacemapper-server/sessions/30d37bfd/snapshots/1/sparse/0/
```

### 7. Run training
```
cd /home/user/MEGS-2
source /var/www/spacemapper-server/pmegs2-venv/bin/activate

python progressive_train.py \
  --config configs/progressive.yaml \
  --source_path_dir /var/www/spacemapper-server/sessions/30d37bfd/snapshots \
  --source_path     /var/www/spacemapper-server/sessions/30d37bfd/snapshots \
  --images          /var/www/spacemapper-server/sessions/30d37bfd/undistorted/images \
  --model_path      /var/www/spacemapper-server/sessions/30d37bfd/output
```

---

## Bugs fixed during first run (all committed to branch)

### Bug 1: `pip install -r requirements.txt` fails with "No module named 'torch'"
**Cause:** requirements.txt lists CUDA submodules alongside plain packages. pip tries
to build them in an isolated subprocess before torch is installed.
**Fix:** Install torch first, then plain deps, then submodules separately with
`--no-build-isolation`.

### Bug 2: `python3.12-dev` not installed → `Python.h: No such file or directory`
**Cause:** CUDA extensions need Python C headers to build the pybind11 bindings.
**Fix:** `sudo apt-get install -y python3.12-dev`

### Bug 3: `--source_path` vs `--source_path_dir`
**Cause:** progressive_train.py registers `--source_path_dir` for ProgressiveScene;
the standard MEGS-2 `--source_path` (from ModelParams) is a separate argument.
**Fix:** Pass both flags pointing at the snapshot directory.

### Bug 4: GLOMAP camera model not PINHOLE
**Cause:** MEGS-2's readColmapCameras only handles PINHOLE/SIMPLE_PINHOLE.
GLOMAP outputs SIMPLE_RADIAL by default.
**Fix:** Run `colmap image_undistorter` to convert to PINHOLE before training.

### Bug 5: `save_checkpoint() got unexpected keyword argument 'snapshot_idx'`
**Cause:** Function declared `tag` parameter but all call sites used `snapshot_idx=`.
**Fix:** Changed signature to `save_checkpoint(..., snapshot_idx=None, tag=None)`.
Commit: d980ad9

### Bug 6: `RuntimeError: size of tensor a must match tensor b` in OptimizingSpaSG
**Cause:** After the 2nd prune (808k→160k Gaussians), `self.z` and `self.u` in
OptimizingSpaSG still held the old size. The prune shrinks `gaussians._sg_sharpness`
but doesn't notify the optimizer auxiliary tensors.
**Fix:** Set `optimizingSpa = None` and `optimizingSpaSg = None` immediately after
both prune events. The loss guard (`if optimizingSpaSg is not None`) prevents crashes.
Commit: caded82

### Bug 7: `AttributeError: 'NoneType' object has no attribute 'update'`
**Cause:** The periodic update block at lines 654-664 called `.update()` on both
objects unconditionally, hitting None after Bug 6's fix.
**Fix:** Added `and optimizingSpa is not None` / `and optimizingSpaSg is not None`
guards to both branches.
Commit: c3dbea6

---

## First successful full run
Date: 2026-05-08
Result: "Progressive training complete. Final Gaussian count: 777926"
Wall-clock: ~10 minutes (initial 3000 iters + final 4000 iters, single snapshot)
Output PLY: /var/www/spacemapper-server/sessions/30d37bfd/output/point_cloud/iteration_final/point_cloud.ply
