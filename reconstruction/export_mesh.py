"""
reconstruction/export_mesh.py
------------------------------
Load a trained NeRF checkpoint, extract a mesh via marching cubes,
clean it with Open3D, and export it as a .glb file using trimesh.

Usage:
    python -m reconstruction.export_mesh --species rose
    python -m reconstruction.export_mesh --species rose --resolution 64 --threshold 10.0

[T]  Expect 2-10 min on CPU at resolution=64 (64³ = 262 144 query points).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MODELS_ROOT = Path("models") / "nerf_exports"


# ---------------------------------------------------------------------------
# Lazy import helpers (avoid hard crash if optional deps missing)
# ---------------------------------------------------------------------------

def _require(pkg: str, pip_name: str | None = None) -> None:
    import importlib
    try:
        importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        print(f"\n[ERROR] Missing package '{name}'. Install with:\n"
              f"  pip install {name}\n", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Density-field evaluator
# ---------------------------------------------------------------------------

@torch.no_grad()
def query_density_grid(
    model,                    # TinyNeRF instance
    resolution: int = 64,
    scene_min: float = -1.5,
    scene_max: float =  1.5,
    chunk: int = 8192,
) -> np.ndarray:
    """Evaluate the NeRF density field on a resolution³ voxel grid.

    Args:
        model:       Trained TinyNeRF model (in eval mode).
        resolution:  Number of voxels per axis.
        scene_min/max: World-space bounding box extent (assumes cube).
        chunk:       Points per forward-pass (lower = less RAM).

    Returns:
        sigma: (resolution, resolution, resolution) numpy float32 array.
    """
    # Build the grid
    lin    = np.linspace(scene_min, scene_max, resolution, dtype=np.float32)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")  # each (R,R,R)
    pts_np = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)   # (R³, 3)
    pts_t  = torch.from_numpy(pts_np)

    # Dummy view direction (pointing inward along -Z; density is dir-independent)
    dirs_t = torch.zeros_like(pts_t)
    dirs_t[:, 2] = -1.0

    sigmas: list[np.ndarray] = []
    total = pts_t.shape[0]

    print(f"  [?]  Evaluating density on {resolution}³ = {total:,} grid points ...", flush=True)
    for start in range(0, total, chunk):
        end   = min(start + chunk, total)
        _, sigma_out  = model(pts_t[start:end], dirs_t[start:end])
        sigmas.append(sigma_out.numpy())

    sigma_flat = np.concatenate(sigmas, axis=0)              # (R³,)
    return sigma_flat.reshape(resolution, resolution, resolution)


# ---------------------------------------------------------------------------
# Marching-cubes extraction
# ---------------------------------------------------------------------------

def extract_mesh_marching_cubes(
    sigma_grid: np.ndarray,
    scene_min: float,
    scene_max: float,
    threshold: float = 10.0,
):
    """Run marching cubes on the density volume.

    Args:
        sigma_grid: (R, R, R) density array.
        scene_min/max: Bounding box extents.
        threshold:  Iso-surface threshold (tune per scene; 10 is a good default).

    Returns:
        (vertices, faces) as numpy arrays.
    """
    from skimage.measure import marching_cubes  # type: ignore

    print(f"  ?  Running marching cubes (threshold={threshold}) ...")

    if sigma_grid.max() < threshold:
        print(f"  [!]  Max density ({sigma_grid.max():.2f}) < threshold ({threshold}).")
        print(f"       Try --threshold {sigma_grid.max() * 0.3:.1f}")

    verts, faces, *_ = marching_cubes(sigma_grid, level=threshold)

    # Map voxel indices -> world coordinates
    R    = sigma_grid.shape[0]
    verts = verts / (R - 1) * (scene_max - scene_min) + scene_min

    print(f"  [<>]  Raw mesh: {len(verts):,} vertices, {len(faces):,} faces")
    return verts.astype(np.float32), faces


# ---------------------------------------------------------------------------
# Open3D mesh cleaning
# ---------------------------------------------------------------------------

def clean_mesh_open3d(
    verts: np.ndarray,
    faces: np.ndarray,
    min_component_fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove small disconnected components and degenerate triangles.

    Args:
        verts, faces: Raw mesh from marching cubes.
        min_component_fraction: Remove components smaller than this fraction
                                of the largest component's face count.

    Returns:
        Cleaned (verts, faces).
    """
    import open3d as o3d   # type: ignore

    print("  [C]  Cleaning mesh with Open3D ...")

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # Identify connected components and keep the largest cluster
    triangle_clusters, cluster_n_tris, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_tris    = np.asarray(cluster_n_tris)

    if len(cluster_n_tris) == 0:
        return verts, faces  # nothing to clean

    max_tris        = cluster_n_tris.max()
    min_tris_keep   = max(1, int(max_tris * min_component_fraction))

    # Build mask: keep triangles whose cluster is large enough
    keep_mask = np.zeros(len(triangle_clusters), dtype=bool)
    for ci, n in enumerate(cluster_n_tris):
        if n >= min_tris_keep:
            keep_mask[triangle_clusters == ci] = True

    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()

    clean_verts = np.asarray(mesh.vertices,  dtype=np.float32)
    clean_faces = np.asarray(mesh.triangles, dtype=np.int32)

    removed_tri = len(faces) - len(clean_faces)
    print(f"  [OK]  Removed {removed_tri:,} small-component triangles")
    print(f"  [<>]  Clean mesh: {len(clean_verts):,} vertices, {len(clean_faces):,} faces")

    return clean_verts, clean_faces


# ---------------------------------------------------------------------------
# Trimesh GLB export
# ---------------------------------------------------------------------------

def export_glb(
    verts: np.ndarray,
    faces: np.ndarray,
    out_path: Path,
    color: tuple[int, int, int] = (120, 180, 80),  # leaf-green default
) -> None:
    """Export mesh to .glb using trimesh.

    Args:
        verts, faces: Cleaned mesh arrays.
        out_path:     Destination .glb file path.
        color:        Vertex colour (R, G, B) ? [0,255] applied uniformly.
    """
    import trimesh   # type: ignore

    print(f"  [P]  Exporting to GLB: {out_path} ...")

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        process=True,
    )
    # Uniform vertex color
    rgba = np.tile(np.array([*color, 255], dtype=np.uint8), (len(verts), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))

    size_kb = out_path.stat().st_size / 1024
    print(f"\n  {'='*50}")
    print(f"  [OK]  GLB export complete")
    print(f"  [%]  Vertices  : {len(verts):,}")
    print(f"  [%]  Faces     : {len(faces):,}")
    print(f"  [S]  File size : {size_kb:.1f} KB  ({out_path.name})")
    print(f"  [D]  Location  : {out_path.resolve()}")
    print(f"  {'='*50}\n")


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------

def export_mesh(
    species: str,
    resolution: int = 64,
    threshold: float = 10.0,
    scene_min: float = -1.5,
    scene_max: float =  1.5,
    chunk: int = 8192,
) -> Path:
    """Load NeRF checkpoint -> marching cubes -> open3d clean -> GLB.

    Args:
        species:    Must match --species used in train_nerf.
        resolution: Voxel grid resolution per axis (64 -> 262 144 points).
        threshold:  Marching-cubes iso-surface level.
        scene_min/max: World-space bounding box.
        chunk:      Points per forward-pass.

    Returns:
        Path to the exported .glb file.
    """
    # Check optional dependencies
    for pkg, pip in [("skimage", "scikit-image"), ("open3d", "open3d"), ("trimesh", "trimesh[easy]")]:
        _require(pkg, pip)

    ckpt_path = MODELS_ROOT / species / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run  python -m reconstruction.train_nerf --species {species}  first."
        )

    glb_path = MODELS_ROOT / species / f"{species}.glb"

    print(f"\n{'='*60}")
    print(f"  ?  Exporting mesh for species: {species}")
    print(f"  ??   resolution={resolution}  threshold={threshold}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    # Import TinyNeRF from this package
    from reconstruction.train_nerf import TinyNeRF

    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyNeRF(L_pos=6, L_dir=4, hidden=128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [M]  Loaded checkpoint (iter={ckpt.get('iterations', '?')})")

    # ------------------------------------------------------------------
    # 2. Query density grid
    # ------------------------------------------------------------------
    sigma_grid = query_density_grid(
        model, resolution=resolution,
        scene_min=scene_min, scene_max=scene_max,
        chunk=chunk,
    )
    print(f"  [^]  Density stats: min={sigma_grid.min():.2f}  "
          f"mean={sigma_grid.mean():.2f}  max={sigma_grid.max():.2f}")

    # Auto-adjust threshold if nothing would be extracted
    if sigma_grid.max() < threshold:
        threshold = float(sigma_grid.max() * 0.3)
        print(f"  ??   Auto-adjusted threshold -> {threshold:.2f}")

    # ------------------------------------------------------------------
    # 3. Marching cubes
    # ------------------------------------------------------------------
    verts, faces = extract_mesh_marching_cubes(
        sigma_grid, scene_min, scene_max, threshold=threshold
    )

    if len(faces) == 0:
        raise RuntimeError(
            "Marching cubes produced an empty mesh. "
            "Try lowering --threshold or training for more iterations."
        )

    # ------------------------------------------------------------------
    # 4. Open3D clean
    # ------------------------------------------------------------------
    verts, faces = clean_mesh_open3d(verts, faces)

    if len(faces) == 0:
        raise RuntimeError("Mesh is empty after cleaning. The scene may be too sparse.")

    # ------------------------------------------------------------------
    # 5. Export .glb
    # ------------------------------------------------------------------
    export_glb(verts, faces, glb_path)
    return glb_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export trained NeRF to a .glb mesh (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--species",    required=True,               help="Species name")
    parser.add_argument("--resolution", type=int,   default=64,      help="Voxel grid resolution (N -> N³ points)")
    parser.add_argument("--threshold",  type=float, default=10.0,    help="Marching-cubes iso-surface level")
    parser.add_argument("--scene_min",  type=float, default=-1.5,    help="World-space bounding box min")
    parser.add_argument("--scene_max",  type=float, default= 1.5,    help="World-space bounding box max")
    parser.add_argument("--chunk",      type=int,   default=8192,    help="Query points per forward pass")
    args = parser.parse_args()

    export_mesh(
        species=args.species,
        resolution=args.resolution,
        threshold=args.threshold,
        scene_min=args.scene_min,
        scene_max=args.scene_max,
        chunk=args.chunk,
    )
