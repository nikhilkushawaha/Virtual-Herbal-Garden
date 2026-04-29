"""
reconstruction/train_nerf.py
-----------------------------
Train a Tiny NeRF (pure PyTorch, CPU-only) on a transforms.json dataset
produced by prepare_data.py, then render 8 novel-view previews.

Architecture:
    Positional encoding (L=6 pos, L=4 dir) → 4-layer MLP (hidden=128) →
    density head + colour head.  Volume rendering via alpha-compositing.

Usage:
    python -m reconstruction.train_nerf --species rose
    python -m reconstruction.train_nerf --species rose --iterations 5000
    python -m reconstruction.train_nerf --species rose --iterations 1000 --img_wh 64 64

⏱  CPU timing: ~10-30 min for 5000 iters at 64×64 image resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path("data")
MODELS_ROOT = Path("models") / "nerf_exports"


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding γ(p) = (sin(2^k π p), cos(2^k π p)) for k=0..L-1."""

    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L
        # Precompute frequency bands: 2^0, 2^1, …, 2^(L-1)
        self.register_buffer("freqs", 2.0 ** torch.arange(L, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)  →  returns: (..., d + 2*L*d)  (include raw x)
        x_freq = x[..., None] * self.freqs   # (..., d, L)
        x_freq = x_freq.reshape(*x.shape[:-1], -1)   # (..., d*L)
        return torch.cat([x, torch.sin(x_freq), torch.cos(x_freq)], dim=-1)

    def out_dim(self, in_dim: int) -> int:
        return in_dim + 2 * self.L * in_dim


# ---------------------------------------------------------------------------
# Tiny NeRF model
# ---------------------------------------------------------------------------

class TinyNeRF(nn.Module):
    """Tiny NeRF MLP.  Input: (pos, dir) → Output: (rgb ∈ [0,1]³, σ ≥ 0)."""

    def __init__(self, L_pos: int = 6, L_dir: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.enc_pos = PositionalEncoding(L_pos)
        self.enc_dir = PositionalEncoding(L_dir)

        in_pos = self.enc_pos.out_dim(3)   # 3 + 2*6*3 = 39
        in_dir = self.enc_dir.out_dim(3)   # 3 + 2*4*3 = 27

        # Density network (position only)
        self.density_net = nn.Sequential(
            nn.Linear(in_pos, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )
        self.density_head = nn.Linear(hidden, 1)   # raw σ (before ReLU)

        # Colour network (density features + view direction)
        self.color_net = nn.Sequential(
            nn.Linear(hidden + in_dir, hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )

    def forward(
        self,
        pts: torch.Tensor,   # (N, 3) world positions
        dirs: torch.Tensor,  # (N, 3) unit ray directions
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rgb, sigma) for each point."""
        pos_enc = self.enc_pos(pts)     # (N, in_pos)
        dir_enc = self.enc_dir(dirs)    # (N, in_dir)

        h     = self.density_net(pos_enc)              # (N, hidden)
        sigma = F.relu(self.density_head(h))           # (N, 1) ≥ 0
        rgb   = self.color_net(torch.cat([h, dir_enc], dim=-1))  # (N, 3) ∈ [0,1]

        return rgb, sigma.squeeze(-1)   # (N,3), (N,)


# ---------------------------------------------------------------------------
# Volume rendering
# ---------------------------------------------------------------------------

def volume_render(
    model: TinyNeRF,
    rays_o: torch.Tensor,   # (N_rays, 3) ray origins
    rays_d: torch.Tensor,   # (N_rays, 3) ray directions (unit vectors)
    near: float = 2.0,
    far:  float = 6.0,
    N_samples: int = 64,
    perturb: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render RGB and depth for a batch of rays.

    Returns:
        rgb_map:   (N_rays, 3) rendered colour ∈ [0,1]
        depth_map: (N_rays,)  expected depth
    """
    N_rays = rays_o.shape[0]

    # Sample depths along each ray
    t_vals = torch.linspace(0.0, 1.0, N_samples)             # (N_samples,)
    z_vals = near * (1 - t_vals) + far * t_vals               # (N_samples,)
    z_vals = z_vals.expand(N_rays, N_samples)                 # (N_rays, N_samples)

    if perturb:
        mids  = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        z_vals = lower + (upper - lower) * torch.rand_like(z_vals)

    # 3D sample positions
    pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., None]  # (N_rays, N_s, 3)
    dirs = rays_d[:, None, :].expand_as(pts)                            # (N_rays, N_s, 3)

    pts_flat  = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    rgb_flat, sigma_flat = model(pts_flat, dirs_flat)
    rgb   = rgb_flat.reshape(N_rays, N_samples, 3)   # (N_rays, N_s, 3)
    sigma = sigma_flat.reshape(N_rays, N_samples)    # (N_rays, N_s)

    # Alpha compositing
    deltas = z_vals[..., 1:] - z_vals[..., :-1]                         # (N_rays, N_s-1)
    deltas = torch.cat([deltas, torch.full((N_rays, 1), 1e10)], dim=-1) # (N_rays, N_s)

    alpha   = 1.0 - torch.exp(-sigma * deltas)                          # (N_rays, N_s)
    weights = alpha * torch.cumprod(
        torch.cat([torch.ones(N_rays, 1), 1.0 - alpha + 1e-10], dim=-1), dim=-1
    )[:, :-1]                                                            # (N_rays, N_s)

    rgb_map   = (weights[..., None] * rgb).sum(dim=-2)   # (N_rays, 3)
    depth_map = (weights * z_vals).sum(dim=-1)            # (N_rays,)

    return rgb_map, depth_map


# ---------------------------------------------------------------------------
# Dataset: rays from transforms.json
# ---------------------------------------------------------------------------

def _load_transforms(species: str) -> dict:
    json_path = DATA_ROOT / species / "transforms.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"transforms.json not found: {json_path}\n"
            "Run  python -m reconstruction.prepare_data --species {species} --images_dir <path>  first."
        )
    with json_path.open() as f:
        return json.load(f)


def _get_rays(c2w: torch.Tensor, H: int, W: int, focal: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ray origins and directions for all pixels in an image.

    Args:
        c2w:   (4,4) camera-to-world transform
        H, W:  image height and width in pixels
        focal: focal length in pixels

    Returns:
        rays_o: (H*W, 3), rays_d: (H*W, 3) unit vectors
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="xy",
    )
    # Camera-space directions
    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,   # flip Y: image top → +Y in NeRF
        -torch.ones_like(i),       # camera looks down -Z
    ], dim=-1)                     # (H, W, 3)

    # Rotate to world space
    rays_d = (dirs[..., None, :] * c2w[:3, :3]).sum(dim=-1)  # (H, W, 3)
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:3, 3].expand_as(rays_d)                    # (H, W, 3)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def build_ray_dataset(
    transforms: dict,
    img_w: int,
    img_h: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all images and precompute all rays + target RGB values.

    Args:
        transforms: Parsed transforms.json dict.
        img_w, img_h: Target image size to resize to (for speed on CPU).

    Returns:
        all_rays_o: (N_total, 3)
        all_rays_d: (N_total, 3)
        all_rgb:    (N_total, 3) ∈ [0,1]
    """
    angle_x = transforms["camera_angle_x"]
    focal   = img_w / (2.0 * math.tan(angle_x / 2.0))

    all_o, all_d, all_rgb = [], [], []

    print(f"  [>>] Loading {len(transforms['frames'])} frames (resize -> {img_w}x{img_h}) ...")

    # Resolve images relative to the data/<species>/ directory
    first_fp = transforms["frames"][0]["file_path"]
    # file_path like "./images/stem" — the root is data/<species>/
    # We'll derive species from the path in DATA_ROOT

    for frame in transforms["frames"]:
        # Load image
        # file_path is like "./images/stem" (no extension)
        fp = Path(frame["file_path"])
        # Try to find the actual file in DATA_ROOT/**/images/
        candidates: list[Path] = []
        for ext in _IMG_EXTS_TUPLE:
            candidates += list(DATA_ROOT.rglob(f"{fp.name}{ext}"))
        if not candidates:
            # Try with directory structure
            for ext in _IMG_EXTS_TUPLE:
                p = DATA_ROOT / (str(fp).lstrip("./")) 
                full = p.with_suffix(ext)
                if full.exists():
                    candidates.append(full)

        if not candidates:
            # Skip missing images with a warning
            print(f"     [!]  Image not found for {fp}, skipping.")
            continue

        img_path = candidates[0]
        try:
            img = Image.open(img_path).convert("RGB").resize((img_w, img_h), Image.LANCZOS)
        except Exception as e:
            print(f"     [!]  Failed to load {img_path}: {e}")
            continue

        rgb = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0)  # (H, W, 3)

        # Camera matrix
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32)[:4, :4]

        rays_o, rays_d = _get_rays(c2w, img_h, img_w, focal)
        rgb_flat = rgb.reshape(-1, 3)

        all_o.append(rays_o)
        all_d.append(rays_d)
        all_rgb.append(rgb_flat)

    if not all_o:
        raise RuntimeError("No images could be loaded. Check your transforms.json and image paths.")

    return torch.cat(all_o), torch.cat(all_d), torch.cat(all_rgb)


_IMG_EXTS_TUPLE = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


# ---------------------------------------------------------------------------
# Novel-view poses (spiral path)
# ---------------------------------------------------------------------------

def _spiral_poses(n: int = 8, radius: float = 4.0, elevation: float = 30.0) -> list[torch.Tensor]:
    """Return n camera-to-world matrices on a circular path."""
    poses: list[torch.Tensor] = []
    el = math.radians(elevation)

    for i in range(n):
        theta = 2 * math.pi * i / n
        cx = radius * math.cos(el) * math.cos(theta)
        cy = radius * math.sin(el)
        cz = radius * math.cos(el) * math.sin(theta)
        cam_pos = np.array([cx, cy, cz])

        forward   = -cam_pos / (np.linalg.norm(cam_pos) + 1e-8)
        world_up  = np.array([0.0, 1.0, 0.0])
        right     = np.cross(forward, world_up)
        right    /= np.linalg.norm(right) + 1e-8
        up        = np.cross(right, forward)

        c2w        = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = cam_pos
        poses.append(torch.from_numpy(c2w))

    return poses


# ---------------------------------------------------------------------------
# Render a full image (chunked to avoid RAM exhaustion)
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_image(
    model: TinyNeRF,
    c2w: torch.Tensor,
    H: int, W: int, focal: float,
    near: float = 2.0, far: float = 6.0,
    chunk: int = 512,
) -> np.ndarray:
    """Render a full H×W image using chunked ray batches."""
    rays_o, rays_d = _get_rays(c2w, H, W, focal)
    rgb_chunks: list[torch.Tensor] = []

    for i in range(0, rays_o.shape[0], chunk):
        ro = rays_o[i:i + chunk]
        rd = rays_d[i:i + chunk]
        rgb, _ = volume_render(model, ro, rd, near=near, far=far, perturb=False)
        rgb_chunks.append(rgb)

    rgb_map = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3)
    return (rgb_map.clamp(0, 1).numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_nerf(
    species: str,
    iterations: int = 5000,
    img_w: int = 100,
    img_h: int = 100,
    batch_rays: int = 1024,
    lr: float = 5e-4,
    near: float = 2.0,
    far: float  = 6.0,
    log_every: int = 500,
    n_novel_views: int = 8,
) -> Path:
    """Train Tiny NeRF on the species dataset and save a checkpoint.

    Args:
        species:    Scene name (must match prepare_data --species).
        iterations: Number of gradient steps.
        img_w/h:    Resize images to this resolution for training speed.
        batch_rays: Rays per gradient step.
        lr:         Adam learning rate.
        near, far:  Near/far clipping distances in world units.
        log_every:  Log loss every N iterations.
        n_novel_views: Number of novel views rendered after training.

    Returns:
        Path to the saved checkpoint.pt file.
    """
    out_dir  = MODELS_ROOT / species
    ckpt_path = out_dir / "checkpoint.pt"
    views_dir = out_dir / "preview_views"
    out_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  Tiny NeRF Training  --  species: {species}")
    print(f"  iterations={iterations}  batch_rays={batch_rays}  img={img_w}x{img_h}")
    print(f"  device: CPU")
    print(f"{'='*62}")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    transforms = _load_transforms(species)
    angle_x    = transforms["camera_angle_x"]
    focal      = img_w / (2.0 * math.tan(angle_x / 2.0))

    all_rays_o, all_rays_d, all_rgb = build_ray_dataset(transforms, img_w, img_h)
    N_total = all_rays_o.shape[0]
    print(f"  [*]  Total rays: {N_total:,}")

    # ------------------------------------------------------------------
    # 2. Model + optimizer
    # ------------------------------------------------------------------
    model     = TinyNeRF(L_pos=6, L_dir=4, hidden=128)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9997)

    # ------------------------------------------------------------------
    # 3. Training loop
    # ------------------------------------------------------------------
    print(f"\n  [>>] Training ...\n")
    loss_hist: list[float] = []
    t_start   = time.time()
    t_last    = t_start

    for step in range(1, iterations + 1):
        # Random ray batch
        idx      = torch.randint(0, N_total, (batch_rays,))
        rays_o   = all_rays_o[idx]
        rays_d   = all_rays_d[idx]
        target   = all_rgb[idx]

        # Forward
        rgb_pred, _ = volume_render(model, rays_o, rays_d, near=near, far=far)
        loss = F.mse_loss(rgb_pred, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_hist.append(loss.item())

        if step % log_every == 0 or step == 1:
            elapsed   = time.time() - t_start
            iters_rem = iterations - step
            secs_per  = elapsed / step
            eta_sec   = iters_rem * secs_per
            eta_str   = f"{int(eta_sec//60)}m {int(eta_sec%60)}s"

            psnr = -10.0 * math.log10(max(loss.item(), 1e-10))
            print(
                f"  [{step:>5}/{iterations}]  "
                f"loss={loss.item():.5f}  PSNR={psnr:.2f}dB  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  ETA={eta_str}"
            )
            t_last = time.time()

    total_time = time.time() - t_start
    print(f"\n  [OK] Training complete in {total_time/60:.1f} min")

    # ------------------------------------------------------------------
    # 4. Save checkpoint
    # ------------------------------------------------------------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "iterations":       iterations,
        "angle_x":          angle_x,
        "near":             near,
        "far":              far,
        "img_w":            img_w,
        "img_h":            img_h,
        "loss_history":     loss_hist,
    }, ckpt_path)
    print(f"  [*]  Checkpoint saved: {ckpt_path.resolve()}")

    # ------------------------------------------------------------------
    # 5. Render novel views
    # ------------------------------------------------------------------
    print(f"\n  [>>] Rendering {n_novel_views} novel views ...")
    model.eval()
    spiral = _spiral_poses(n=n_novel_views, radius=(near + far) / 2)

    for i, c2w in enumerate(spiral):
        img_np  = render_image(model, c2w, img_h, img_w, focal, near=near, far=far)
        out_img = views_dir / f"view_{i:02d}.png"
        Image.fromarray(img_np).save(out_img)
        print(f"     [+]  Saved {out_img.name}")

    print(f"\n  [*]  All previews in: {views_dir.resolve()}")
    print(f"{'='*62}\n")

    return ckpt_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Tiny NeRF on a plant species dataset (CPU-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--species",    required=True,         help="Species name (must match prepare_data)")
    parser.add_argument("--iterations", type=int, default=5000, help="Training iterations")
    parser.add_argument("--img_wh",     type=int, nargs=2, default=[100, 100],
                        metavar=("W", "H"), help="Training image resolution (smaller = faster on CPU)")
    parser.add_argument("--batch_rays", type=int, default=1024, help="Rays per gradient step")
    parser.add_argument("--lr",         type=float, default=5e-4, help="Adam learning rate")
    parser.add_argument("--near",       type=float, default=2.0)
    parser.add_argument("--far",        type=float, default=6.0)
    parser.add_argument("--log_every",  type=int, default=500)
    args = parser.parse_args()

    train_nerf(
        species=args.species,
        iterations=args.iterations,
        img_w=args.img_wh[0],
        img_h=args.img_wh[1],
        batch_rays=args.batch_rays,
        lr=args.lr,
        near=args.near,
        far=args.far,
        log_every=args.log_every,
    )
