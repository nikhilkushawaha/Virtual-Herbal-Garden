"""
reconstruction/nerf_train.py
-----------------------------
Train a NeRF model on COLMAP output.

NOTE: CPU NeRF training takes 30–90 minutes per species depending on
      the number of input images and --max-steps setting.

Usage:
    python reconstruction/nerf_train.py --scene data/scenes/rose
    python reconstruction/nerf_train.py --scene data/scenes/rose --max-steps 5000
"""

import argparse
from pathlib import Path


def train_nerf(scene_dir: str, max_steps: int = 30_000) -> None:
    """Placeholder NeRF training entrypoint.

    Replace the body of this function with your preferred NeRF
    implementation (e.g., nerfstudio, instant-ngp, tiny-cuda-nn CPU fork).

    Args:
        scene_dir: Path to scene folder containing COLMAP sparse output.
        max_steps: Maximum number of training iterations.
    """
    scene = Path(scene_dir)
    if not (scene / "sparse").exists():
        raise FileNotFoundError(
            f"No COLMAP sparse output found in {scene / 'sparse'}. "
            "Run reconstruction/colmap_runner.py first."
        )

    print(f"🌿 Starting NeRF training for scene: {scene.name}")
    print(f"   Max steps : {max_steps}")
    print(f"   Device    : CPU")
    print(f"   ETA       : 30–90 minutes on a modern CPU\n")

    # TODO: Integrate your NeRF library here.
    # Example with nerfstudio:
    #   subprocess.run([
    #       "ns-train", "vanilla-nerf",
    #       "--data", str(scene),
    #       "--max-num-iterations", str(max_steps),
    #   ], check=True)

    raise NotImplementedError("NeRF training backend not yet integrated. See TODO above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeRF on COLMAP scene")
    parser.add_argument("--scene", required=True, help="Scene directory")
    parser.add_argument("--max-steps", type=int, default=30_000)
    args = parser.parse_args()

    train_nerf(args.scene, args.max_steps)
