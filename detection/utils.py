"""
detection/utils.py
------------------
Shared helpers for the detection package.

Public API:
    load_class_names(yaml_path)           → list[str]
    draw_bbox(image, bbox, label, conf)   → PIL.Image
    get_glb_path(species_name)            → Path
"""

from __future__ import annotations

import yaml
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Class name loader
# ---------------------------------------------------------------------------

def load_class_names(yaml_path: str | Path) -> list[str]:
    """Parse a YOLOv8 data.yaml and return the ordered list of class names.

    Args:
        yaml_path: Path to a YOLOv8-compatible data.yaml file.

    Returns:
        List of class name strings (index == class id).

    Raises:
        FileNotFoundError: If the YAML does not exist.
        KeyError: If 'names' key is missing from the YAML.
    """
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if "names" not in cfg:
        raise KeyError(f"'names' key missing from {p}")

    names = cfg["names"]
    # Roboflow YAMLs can store names as a dict {0: 'rose', 1: 'tulip', ...}
    # or as a plain list ['rose', 'tulip', ...]
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return list(names)


# ---------------------------------------------------------------------------
# Bounding-box drawing
# ---------------------------------------------------------------------------

# Colour palette — cycles through these for different class indices
_PALETTE = [
    "#2ecc71", "#3498db", "#e74c3c", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#e91e63",
]


def draw_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float] | list[float],
    label: str,
    confidence: float,
    color: str | None = None,
    line_width: int = 3,
    font_size: int = 18,
) -> Image.Image:
    """Draw a single bounding box with a label badge on a PIL image.

    Args:
        image:      Source PIL Image (will NOT be modified in-place).
        bbox:       (x1, y1, x2, y2) in pixel coordinates.
        label:      Class name string.
        confidence: Float 0–1 confidence score.
        color:      Hex colour string. Auto-selected from palette if None.
        line_width: Box border thickness in pixels.
        font_size:  Font size for the label badge.

    Returns:
        New PIL Image with the annotation drawn on it.
    """
    img     = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    x1, y1, x2, y2 = (float(v) for v in bbox)

    # Choose colour
    if color is None:
        # Use a deterministic index based on the label string
        color = _PALETTE[hash(label) % len(_PALETTE)]

    # --- Box ---
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

    # --- Label badge ---
    text = f"{label}  {confidence:.0%}"
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Measure text
    bbox_text = draw.textbbox((0, 0), text, font=font)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    pad = 4

    badge_y1 = max(y1 - th - pad * 2, 0)
    badge_y2 = badge_y1 + th + pad * 2
    badge_x2 = x1 + tw + pad * 2

    # Semi-transparent filled badge
    draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=color + "CC")
    draw.text((x1 + pad, badge_y1 + pad), text, fill="white", font=font)

    # Composite
    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


# ---------------------------------------------------------------------------
# GLB path helper
# ---------------------------------------------------------------------------

def get_glb_path(species_name: str) -> Path:
    """Return the expected .glb path for a given plant species.

    Convention:  models/nerf_exports/<species>/<species>.glb
    The file may not exist yet (reconstruction may not have run).

    Args:
        species_name: Class name string, e.g. "rose".

    Returns:
        Absolute pathlib.Path (Windows-safe) to the .glb file.
    """
    safe_name = species_name.strip().lower().replace(" ", "_")
    return Path("models") / "nerf_exports" / safe_name / f"{safe_name}.glb"
