"""
reconstruction/run_all.py
--------------------------
Orchestrator: run the full 3D reconstruction pipeline in sequence.

    prepare_data  ->  train_nerf  ->  export_mesh

Usage:
    python -m reconstruction.run_all --species rose --images_dir path/to/photos/
    python -m reconstruction.run_all --species rose --images_dir path/ --iterations 2000 --skip_colmap

[T]  Estimated total time on CPU: ~45–90 minutes
   (predominantly train_nerf; prepare_data ? 5?15 min with COLMAP)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Step runner helper
# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    bar = "=" * 62
    print(f"\n+{bar}+")
    print(f"?  {title:<58}  ?")
    print(f"+{bar}+")


def _step_summary(step: str, duration: float, ok: bool) -> None:
    status = "[OK]  DONE" if ok else "[ERROR]  FAILED"
    print(f"\n  {status}  {step}  ({duration:.1f}s / {duration/60:.1f}min)\n")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_all(
    species: str,
    images_dir: Path,
    skip_colmap: bool = False,
    iterations: int = 5000,
    img_wh: tuple[int, int] = (100, 100),
    resolution: int = 64,
    threshold: float = 10.0,
    near: float = 2.0,
    far: float = 6.0,
) -> None:
    """Run the complete reconstruction pipeline.

    Args:
        species:     Plant species name.
        images_dir:  Folder of source images.
        skip_colmap: Use synthetic poses instead of COLMAP (for testing).
        iterations:  NeRF training iterations.
        img_wh:      (width, height) to resize images during training.
        resolution:  Marching-cubes grid resolution (N -> N³).
        threshold:   Density iso-surface threshold.
        near, far:   NeRF scene depth bounds.
    """
    from reconstruction.prepare_data import prepare_data
    from reconstruction.train_nerf   import train_nerf
    from reconstruction.export_mesh  import export_mesh

    t_pipeline_start = time.time()

    # ------------------------------------------------------------------
    print(f"\n{'#'*62}")
    print(f"  [*]  Virtual Garden -- 3D Reconstruction Pipeline")
    print(f"{'#'*62}")
    print(f"  Species     : {species}")
    print(f"  Images from : {images_dir.resolve()}")
    print(f"  Skip COLMAP : {skip_colmap}")
    print(f"  NeRF iters  : {iterations}")
    print(f"  Mesh res    : {resolution}³")
    print(f"\n  [T]   Estimated total time on CPU: ~45–90 minutes")
    print(f"       (prepare?5?15 min + train?30?60 min + export?2?10 min)")
    print(f"{'-'*62}\n")

    results: dict[str, tuple[bool, float, str]] = {}  # step -> (ok, seconds, note)

    # ==================================================================
    # STEP 1 -- prepare_data
    # ==================================================================
    _banner("Step 1 / 3 -- prepare_data  (image copy + camera poses)")
    t0 = time.time()
    try:
        transforms_path = prepare_data(
            species=species,
            images_dir=images_dir,
            skip_colmap=skip_colmap,
        )
        dt = time.time() - t0
        _step_summary("prepare_data", dt, ok=True)
        results["prepare_data"] = (True, dt, str(transforms_path))
    except Exception as exc:
        dt = time.time() - t0
        _step_summary("prepare_data", dt, ok=False)
        print(f"  [ERROR] {exc}", file=sys.stderr)
        results["prepare_data"] = (False, dt, str(exc))
        _print_summary(results, time.time() - t_pipeline_start)
        sys.exit(1)

    # ==================================================================
    # STEP 2 -- train_nerf
    # ==================================================================
    _banner("Step 2 / 3 -- train_nerf  (Tiny NeRF, PyTorch CPU)")
    t0 = time.time()
    try:
        ckpt_path = train_nerf(
            species=species,
            iterations=iterations,
            img_w=img_wh[0],
            img_h=img_wh[1],
            near=near,
            far=far,
        )
        dt = time.time() - t0
        _step_summary("train_nerf", dt, ok=True)
        results["train_nerf"] = (True, dt, str(ckpt_path))
    except Exception as exc:
        dt = time.time() - t0
        _step_summary("train_nerf", dt, ok=False)
        print(f"  [ERROR] {exc}", file=sys.stderr)
        results["train_nerf"] = (False, dt, str(exc))
        _print_summary(results, time.time() - t_pipeline_start)
        sys.exit(1)

    # ==================================================================
    # STEP 3 -- export_mesh
    # ==================================================================
    _banner("Step 3 / 3 -- export_mesh  (marching cubes -> .glb)")
    t0 = time.time()
    try:
        glb_path = export_mesh(
            species=species,
            resolution=resolution,
            threshold=threshold,
        )
        dt = time.time() - t0
        _step_summary("export_mesh", dt, ok=True)
        results["export_mesh"] = (True, dt, str(glb_path))
    except Exception as exc:
        dt = time.time() - t0
        _step_summary("export_mesh", dt, ok=False)
        print(f"  [ERROR] {exc}", file=sys.stderr)
        results["export_mesh"] = (False, dt, str(exc))
        # Don't exit -- export failing is non-fatal to the summary

    _print_summary(results, time.time() - t_pipeline_start)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: dict, total_sec: float) -> None:
    print(f"\n{'='*62}")
    print(f"  [END]  Pipeline Summary")
    print(f"{'='*62}")
    for step, (ok, dt, note) in results.items():
        icon = "[OK]" if ok else "[ERROR]"
        print(f"  {icon}  {step:<18}  {dt:>7.1f}s  {note[:40] if not ok else ''}")
    print(f"{'-'*62}")
    print(f"  [T]   Total elapsed : {total_sec:.1f}s  ({total_sec/60:.1f} min)")
    all_ok = all(v[0] for v in results.values())
    if all_ok:
        print(f"  [!]  All steps succeeded!")
    else:
        print(f"  [!]   Some steps failed -- check stderr above.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full Virtual Garden 3D reconstruction pipeline (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--species",     required=True,           help="Plant species name")
    parser.add_argument("--images_dir",  required=True, type=Path, help="Source image directory")
    parser.add_argument("--skip_colmap", action="store_true",
                        help="Skip COLMAP; generate synthetic poses (for testing)")
    parser.add_argument("--iterations",  type=int,   default=5000,  help="NeRF training iterations")
    parser.add_argument("--img_wh",      type=int,   nargs=2,       default=[100, 100],
                        metavar=("W", "H"), help="Training image size (smaller = faster)")
    parser.add_argument("--resolution",  type=int,   default=64,    help="Marching-cubes grid resolution")
    parser.add_argument("--threshold",   type=float, default=10.0,  help="Iso-surface density threshold")
    parser.add_argument("--near",        type=float, default=2.0,   help="NeRF near plane")
    parser.add_argument("--far",         type=float, default=6.0,   help="NeRF far plane")
    args = parser.parse_args()

    if not args.images_dir.exists():
        print(f"[ERROR] images_dir not found: {args.images_dir}", file=sys.stderr)
        sys.exit(1)

    run_all(
        species=args.species,
        images_dir=args.images_dir,
        skip_colmap=args.skip_colmap,
        iterations=args.iterations,
        img_wh=tuple(args.img_wh),
        resolution=args.resolution,
        threshold=args.threshold,
        near=args.near,
        far=args.far,
    )
