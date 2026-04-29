"""
reconstruction/colmap_runner.py
---------------------------------
Wrapper around the COLMAP CLI to run sparse (SfM) reconstruction.

Requires COLMAP to be installed and available on PATH.
Windows installer: https://colmap.github.io/install.html

Usage:
    python reconstruction/colmap_runner.py --scene data/scenes/rose
"""

import argparse
import subprocess
from pathlib import Path


def run_colmap(scene_dir: str) -> None:
    """Run COLMAP feature extraction, matching, and mapper on a scene directory.

    Args:
        scene_dir: Path to scene folder containing an 'images/' subdirectory.
    """
    scene = Path(scene_dir)
    db = scene / "colmap.db"
    images = scene / "images"
    sparse = scene / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    steps = [
        # 1. Feature extraction
        ["colmap", "feature_extractor",
         "--database_path", str(db),
         "--image_path", str(images),
         "--ImageReader.single_camera", "1"],
        # 2. Exhaustive matcher
        ["colmap", "exhaustive_matcher",
         "--database_path", str(db)],
        # 3. Mapper (sparse reconstruction)
        ["colmap", "mapper",
         "--database_path", str(db),
         "--image_path", str(images),
         "--output_path", str(sparse)],
    ]

    for cmd in steps:
        print(f"\n▶ Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP step failed: {cmd[1]}")

    print(f"\n✅ COLMAP sparse reconstruction complete → {sparse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run COLMAP sparse reconstruction")
    parser.add_argument("--scene", required=True, help="Scene directory with images/ subfolder")
    args = parser.parse_args()

    run_colmap(args.scene)
