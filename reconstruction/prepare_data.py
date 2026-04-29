"""
reconstruction/prepare_data.py
--------------------------------
Copy images, run COLMAP SfM, and convert sparse output to a NeRF-standard
transforms.json ready for train_nerf.py.

Usage:
    python -m reconstruction.prepare_data --species rose --images_dir path/to/photos/
    python -m reconstruction.prepare_data --species rose --images_dir path/to/photos/ --skip_colmap

Requirements:
    COLMAP installed and on PATH.
    Windows installer: https://demuc.de/colmap/
    Installation steps:
        1. Download the Windows pre-built binary from https://demuc.de/colmap/
        2. Extract the ZIP to e.g. C:\\colmap
        3. Add C:\\colmap\\bin to your System PATH
        4. Restart your terminal and verify: colmap --version
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# COLMAP binary format constants
# ---------------------------------------------------------------------------

# Number of intrinsic parameters for each COLMAP camera model id
_COLMAP_MODEL_PARAMS: dict[int, int] = {
    0: 3,   # SIMPLE_PINHOLE  : f, cx, cy
    1: 4,   # PINHOLE         : fx, fy, cx, cy
    2: 4,   # SIMPLE_RADIAL   : f, cx, cy, k
    3: 5,   # RADIAL          : f, cx, cy, k1, k2
    4: 8,   # OPENCV          : fx, fy, cx, cy, k1, k2, p1, p2
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# COLMAP availability check
# ---------------------------------------------------------------------------

def _check_colmap() -> bool:
    """Return True if `colmap` is on PATH and executable."""
    try:
        result = subprocess.run(
            ["colmap", "--version"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _colmap_not_found_error() -> None:
    print("\n" + "=" * 65)
    print("  [ERROR] COLMAP is not installed or not found on PATH.")
    print("=" * 65)
    print()
    print("  Download COLMAP for Windows:")
    print("    https://demuc.de/colmap/")
    print()
    print("  Installation steps:")
    print("    1. Go to https://demuc.de/colmap/")
    print("    2. Download the latest Windows pre-built ZIP")
    print("    3. Extract to e.g.  C:\\colmap")
    print("    4. Add  C:\\colmap\\bin  to your System PATH:")
    print("         Start -> 'Edit the system environment variables'")
    print("         -> Environment Variables -> Path -> New -> C:\\colmap\\bin")
    print("    5. Restart your terminal")
    print("    6. Verify: colmap --version")
    print()
    print("  TIP: Run with --skip_colmap to test the pipeline")
    print("       using synthetic poses (no COLMAP required).")
    print("=" * 65 + "\n")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Image copying
# ---------------------------------------------------------------------------

def copy_images(src_dir: Path, dest_dir: Path) -> list[Path]:
    """Copy all image files from src_dir into dest_dir.

    Returns:
        Sorted list of destination image paths.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_images = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)

    if not src_images:
        raise ValueError(f"No images found in {src_dir}. "
                         f"Supported formats: {', '.join(_IMG_EXTS)}")

    copied: list[Path] = []
    for src in src_images:
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    print(f"  [*]  Copied {len(copied)} images -> {dest_dir}")
    return copied


# ---------------------------------------------------------------------------
# COLMAP runner
# ---------------------------------------------------------------------------

def run_colmap(scene_dir: Path) -> Path:
    """Run COLMAP feature_extractor → exhaustive_matcher → mapper.

    Args:
        scene_dir: Root scene directory containing an 'images/' subfolder.

    Returns:
        Path to the sparse/0 reconstruction directory.
    """
    db      = scene_dir / "colmap.db"
    images  = scene_dir / "images"
    sparse  = scene_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    steps = [
        {
            "name": "feature_extractor",
            "cmd": [
                "colmap", "feature_extractor",
                "--database_path", str(db),
                "--image_path",    str(images),
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu",    "0",
            ],
        },
        {
            "name": "exhaustive_matcher",
            "cmd": [
                "colmap", "exhaustive_matcher",
                "--database_path", str(db),
                "--SiftMatching.use_gpu", "0",
            ],
        },
        {
            "name": "mapper",
            "cmd": [
                "colmap", "mapper",
                "--database_path", str(db),
                "--image_path",    str(images),
                "--output_path",   str(sparse),
            ],
        },
    ]

    for step in steps:
        print(f"\n  [>>] COLMAP {step['name']} ...")
        result = subprocess.run(step["cmd"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"     stdout: {result.stdout[-800:]}")
            print(f"     stderr: {result.stderr[-800:]}")
            raise RuntimeError(f"COLMAP {step['name']} failed (exit {result.returncode}).")
        print(f"     [OK] {step['name']} done.")

    # COLMAP mapper writes sparse/0, sparse/1, … — use the first one
    candidates = sorted(sparse.iterdir())
    if not candidates:
        raise RuntimeError(
            "COLMAP mapper produced no output in sparse/. "
            "The images may lack sufficient overlap or texture."
        )
    recon_dir = candidates[0]
    print(f"\n  [*]  Using sparse reconstruction: {recon_dir}")
    return recon_dir


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

def _read_cameras_bin(path: Path) -> dict[int, dict[str, Any]]:
    """Parse cameras.bin and return a dict keyed by camera_id."""
    cameras: dict[int, dict[str, Any]] = {}
    with path.open("rb") as f:
        (num_cameras,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_cameras):
            camera_id, model_id = struct.unpack("<iI", f.read(8))
            # camera_id is int32, model_id is int32 — actually both uint32/int32
            # Re-read: camera_id uint32, model_id int32
            # Correction — COLMAP uses:
            #   camera_id: uint32 (4 bytes)
            #   model_id:  int32  (4 bytes)  ← already read above (8 bytes total, ok)
            width, height = struct.unpack("<QQ", f.read(16))
            n_params = _COLMAP_MODEL_PARAMS.get(model_id, 0)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width":    width,
                "height":   height,
                "params":   list(params),
            }
    return cameras


def _read_images_bin(path: Path) -> dict[int, dict[str, Any]]:
    """Parse images.bin and return a dict keyed by image_id."""
    images: dict[int, dict[str, Any]] = {}
    with path.open("rb") as f:
        (num_images,) = struct.unpack("<Q", f.read(8))
        for _ in range(num_images):
            image_id,        = struct.unpack("<I", f.read(4))
            qvec             = struct.unpack("<4d", f.read(32))   # qw,qx,qy,qz
            tvec             = struct.unpack("<3d", f.read(24))   # tx,ty,tz
            camera_id,       = struct.unpack("<I", f.read(4))

            # Read null-terminated name
            name_chars: list[bytes] = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")

            (num_p2d,) = struct.unpack("<Q", f.read(8))
            # Skip 2D point observations (x float64, y float64, point3D_id int64)
            f.read(num_p2d * (8 + 8 + 8))

            images[image_id] = {
                "name":      name,
                "qvec":      qvec,   # (qw, qx, qy, qz)
                "tvec":      tvec,   # (tx, ty, tz)
                "camera_id": camera_id,
            }
    return images


# ---------------------------------------------------------------------------
# Quaternion → rotation matrix
# ---------------------------------------------------------------------------

def _qvec_to_rotmat(qvec: tuple) -> np.ndarray:
    """Convert a (qw, qx, qy, qz) quaternion to a 3×3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,  2*qx*qy - 2*qz*qw,  2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,      1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,      2*qy*qz + 2*qx*qw,  1 - 2*qx**2 - 2*qy**2],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# COLMAP → transforms.json converter
# ---------------------------------------------------------------------------

def _camera_angle_x(camera: dict[str, Any]) -> float:
    """Compute horizontal field-of-view in radians from COLMAP camera params."""
    model  = camera["model_id"]
    params = camera["params"]
    width  = camera["width"]

    # fx is first param for PINHOLE, or the single f for SIMPLE_PINHOLE / SIMPLE_RADIAL
    fx = params[0]  # works for model 0,1,2,3,4,5,…
    return 2 * math.atan(width / (2 * fx))


def colmap_to_transforms(recon_dir: Path, images_dir: Path) -> dict:
    """Convert a COLMAP sparse/0 directory to a NeRF transforms.json dict.

    Coordinate-system conversion:
        COLMAP  : +X right, +Y down,  +Z forward (into scene)
        NeRF    : +X right, +Y up,    -Z forward (out of scene)
    The c2w matrix columns 1 and 2 are negated to perform this flip.

    Args:
        recon_dir:  Path to sparse/0 (or whichever COLMAP reconstruction).
        images_dir: Path to the images/ folder (for relative file_path).

    Returns:
        transforms.json compatible dict.
    """
    cameras_bin = recon_dir / "cameras.bin"
    images_bin  = recon_dir / "images.bin"

    if not cameras_bin.exists() or not images_bin.exists():
        raise FileNotFoundError(
            f"Expected cameras.bin and images.bin inside {recon_dir}. "
            "Did COLMAP mapper succeed?"
        )

    cameras = _read_cameras_bin(cameras_bin)
    images  = _read_images_bin(images_bin)

    if not cameras:
        raise RuntimeError("cameras.bin is empty — no cameras were reconstructed.")
    if not images:
        raise RuntimeError("images.bin is empty — no images were registered.")

    # Use the first camera for global FOV (single-camera assumption)
    first_cam = next(iter(cameras.values()))
    angle_x   = _camera_angle_x(first_cam)

    frames: list[dict] = []
    for img in images.values():
        cam = cameras.get(img["camera_id"], first_cam)

        R   = _qvec_to_rotmat(img["qvec"])   # world → camera rotation
        t   = np.array(img["tvec"])          # world → camera translation

        # Build camera-to-world (4×4)
        c2w      = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R.T
        c2w[:3,  3] = -(R.T @ t)

        # COLMAP → NeRF coordinate flip (negate Y and Z columns)
        c2w[:, 1] *= -1
        c2w[:, 2] *= -1

        # file_path relative to scene root (no extension — NeRF convention)
        stem      = Path(img["name"]).stem
        file_path = f"./images/{stem}"

        frames.append({
            "file_path":        file_path,
            "transform_matrix": c2w.tolist(),
            "w":  cam["width"],
            "h":  cam["height"],
        })

    # Sort frames by file_path for reproducibility
    frames.sort(key=lambda f: f["file_path"])

    return {
        "camera_angle_x": angle_x,
        "frames":         frames,
    }


# ---------------------------------------------------------------------------
# Fallback: synthetic transforms.json for testing without COLMAP
# ---------------------------------------------------------------------------

def _make_dummy_transforms(images_dir: Path, width: int = 800, height: int = 600) -> dict:
    """Generate a plausible transforms.json from images without COLMAP.

    Uses an inward-facing camera arrangement on a unit sphere.
    This is only useful for testing the downstream NeRF pipeline.
    """
    imgs  = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
    n     = len(imgs)
    angle_x = math.radians(60.0)  # 60° horizontal FOV

    # Try to read actual image dimensions
    try:
        sample = Image.open(imgs[0])
        width, height = sample.size
    except Exception:
        pass

    frames: list[dict] = []
    for i, img_path in enumerate(imgs):
        theta = 2 * math.pi * i / n     # azimuth
        phi   = math.pi / 4             # elevation (45°)
        radius = 4.0

        # Camera position on sphere
        cx =  radius * math.sin(phi) * math.cos(theta)
        cy =  radius * math.cos(phi)
        cz =  radius * math.sin(phi) * math.sin(theta)
        cam_pos = np.array([cx, cy, cz])

        # Look-at: camera points toward origin
        forward = -cam_pos / np.linalg.norm(cam_pos)
        world_up = np.array([0.0, 1.0, 0.0])
        right    = np.cross(forward, world_up)
        right   /= np.linalg.norm(right) + 1e-8
        up       = np.cross(right, forward)

        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward   # NeRF convention: camera looks down -Z
        c2w[:3, 3] = cam_pos

        frames.append({
            "file_path":        f"./images/{img_path.stem}",
            "transform_matrix": c2w.tolist(),
            "w": width,
            "h": height,
        })

    return {"camera_angle_x": angle_x, "frames": frames}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def prepare_data(species: str, images_dir: Path, skip_colmap: bool = False) -> Path:
    """Full prepare pipeline: copy → COLMAP → transforms.json.

    Args:
        species:     Plant species name (used as folder name).
        images_dir:  Directory containing raw source images.
        skip_colmap: If True, generate a dummy transforms.json (for testing).

    Returns:
        Path to the written transforms.json file.
    """
    scene_dir  = Path("data") / species
    dest_imgs  = scene_dir / "images"

    print(f"\n{'='*60}")
    print(f"  [>>] Preparing data for species: {species}")
    print(f"{'='*60}")

    # 1. Copy images
    copied = copy_images(images_dir, dest_imgs)

    if skip_colmap:
        print("\n  [!] --skip_colmap active -- generating synthetic camera poses.")
        transforms = _make_dummy_transforms(dest_imgs)
    else:
        # 2. Check COLMAP
        if not _check_colmap():
            _colmap_not_found_error()

        print(f"\n  [COLMAP] Running on {len(copied)} images ...")
        recon_dir = run_colmap(scene_dir)

        # 3. Convert to transforms.json
        print("\n  [>>] Converting COLMAP output -> transforms.json ...")
        transforms = colmap_to_transforms(recon_dir, dest_imgs)

    # 4. Write transforms.json
    out_json = scene_dir / "transforms.json"
    with out_json.open("w") as f:
        json.dump(transforms, f, indent=2)

    n_frames  = len(transforms["frames"])
    angle_deg = math.degrees(transforms["camera_angle_x"])

    print(f"\n{'='*60}")
    print(f"  [OK] prepare_data complete")
    print(f"  [*]  Images processed : {len(copied)}")
    print(f"  [*]  Cameras found    : {n_frames}")
    print(f"  [*]  Horiz. FOV       : {angle_deg:.1f} deg")
    print(f"  [*]  transforms.json  : {out_json.resolve()}")
    print(f"{'='*60}\n")

    return out_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare images + COLMAP poses for NeRF training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--species",     required=True,          help="Species / scene name")
    parser.add_argument("--images_dir",  required=True, type=Path, help="Source image directory")
    parser.add_argument("--skip_colmap", action="store_true",
                        help="Skip COLMAP; generate synthetic poses (for testing)")
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"[ERROR] images_dir does not exist: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    prepare_data(
        species=args.species,
        images_dir=args.images_dir,
        skip_colmap=args.skip_colmap,
    )
