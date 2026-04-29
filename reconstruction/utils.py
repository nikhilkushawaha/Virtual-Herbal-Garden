"""
reconstruction/utils.py
------------------------
Shared helpers for the reconstruction pipeline.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_scene_images(scene_dir: str | Path) -> list[Path]:
    """Return sorted list of image files in <scene_dir>/images/."""
    images_dir = Path(scene_dir) / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)


def configure_logging(level: int = logging.INFO) -> None:
    """Set up basic logging for the reconstruction pipeline."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )
