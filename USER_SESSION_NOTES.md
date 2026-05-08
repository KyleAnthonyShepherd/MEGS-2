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
MEGS-2 clone: /home/me/MEGS-2
Branch: claude/implement-progressive-megs2-53Kuj

## Data
Session: 30d37bfd
Sparse reconstruction: /var/www/spacemapper-server/sessions/30d37bfd/sparse/0/
  Files: cameras.bin, frames.bin, images.bin, points3D.bin, rigs.bin
  Note: frames.bin and rigs.bin are GLOMAP-specific extras; cameras.bin,
        images.bin, points3D.bin are standard COLMAP format
Images: /var/www/spacemapper-server/sessions/30d37bfd/images/

## Environment decision
New venv at: /var/www/spacemapper-server/pmegs2-venv
PyTorch: 2.4.x with cu121 wheels (Driver 590 supports CUDA 12.x; cu121 is stable)
