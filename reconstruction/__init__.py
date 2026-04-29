"""
reconstruction/
---------------
CPU-only 3D plant reconstruction pipeline.

Pipeline:
    prepare_data  — Copy images, run COLMAP SfM, emit transforms.json
    train_nerf    — Train Tiny NeRF on transforms.json (pure PyTorch, CPU)
    export_mesh   — Marching cubes → Open3D clean → trimesh .glb export
    run_all       — Orchestrate all three steps end-to-end

Quick start:
    # With COLMAP installed:
    python -m reconstruction.run_all --species rose --images_dir path/to/images/

    # Without COLMAP (synthetic poses, for testing):
    python -m reconstruction.run_all --species rose --images_dir path/to/images/ --skip_colmap

    # Steps individually:
    python -m reconstruction.prepare_data --species rose --images_dir path/to/images/ --skip_colmap
    python -m reconstruction.train_nerf   --species rose --iterations 5000
    python -m reconstruction.export_mesh  --species rose
"""

__all__ = [
    "prepare_data",
    "train_nerf",
    "export_mesh",
    "run_all",
]
