"""
detection/validate.py
---------------------
Validate the trained YOLOv8 model against the val split and print a clean
metrics table.

Usage:
    python detection/validate.py
    python detection/validate.py --weights models/yolo/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from detection.utils import load_class_names

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = Path("models/yolo/best.pt")
DEFAULT_DATA    = Path("data/dataset")   # Roboflow download root


def _find_yaml(dataset_dir: Path) -> Path:
    """Return the data.yaml inside dataset_dir (or raise)."""
    yaml = dataset_dir / "data.yaml"
    if yaml.exists():
        return yaml
    candidates = list(dataset_dir.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No data YAML found in {dataset_dir}. "
            "Run `python detection/train.py` first to download the dataset."
        )
    return candidates[0]


def validate(weights: Path = DEFAULT_WEIGHTS, dataset_dir: Path = DEFAULT_DATA) -> None:
    """Load best.pt, run validation on the val split, and print metrics.

    Args:
        weights:     Path to trained .pt weights.
        dataset_dir: Root directory of the downloaded Roboflow dataset.
    """
    if not weights.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights}\n"
            "Run `python detection/train.py` first."
        )

    yaml_path = _find_yaml(dataset_dir)

    print(f"\n🔍  Loading weights : {weights.resolve()}")
    print(f"📂  Dataset YAML   : {yaml_path.resolve()}\n")

    model   = YOLO(str(weights))
    metrics = model.val(
        data=str(yaml_path),
        device="cpu",
        verbose=False,   # suppress per-batch noise; we print our own table
    )

    # -----------------------------------------------------------------------
    # Load class names for the pretty table
    # -----------------------------------------------------------------------
    class_names = load_class_names(yaml_path)

    # -----------------------------------------------------------------------
    # Overall metrics
    # -----------------------------------------------------------------------
    map50    = metrics.box.map50    # mAP@0.50
    map75    = metrics.box.map75    # mAP@0.75
    map5095  = metrics.box.map      # mAP@0.50:0.95

    # Per-class AP at IoU=0.50
    ap_per_class = metrics.box.ap50  # numpy array, one value per class

    # -----------------------------------------------------------------------
    # Pretty table
    # -----------------------------------------------------------------------
    col_w = max((len(n) for n in class_names), default=10) + 2
    header = f"{'Class':<{col_w}} {'AP@0.50':>10}"
    divider = "─" * len(header)

    print("┌" + "─" * (len(header) + 2) + "┐")
    print(f"│  {header}  │")
    print("├" + "─" * (len(header) + 2) + "┤")

    for i, name in enumerate(class_names):
        ap = ap_per_class[i] if i < len(ap_per_class) else float("nan")
        print(f"│  {name:<{col_w}} {ap:>10.4f}  │")

    print("├" + "─" * (len(header) + 2) + "┤")
    print(f"│  {'mAP@0.50':<{col_w}} {map50:>10.4f}  │")
    print(f"│  {'mAP@0.75':<{col_w}} {map75:>10.4f}  │")
    print(f"│  {'mAP@0.50:0.95':<{col_w}} {map5095:>10.4f}  │")
    print("└" + "─" * (len(header) + 2) + "┘")

    print("\n✅  Validation complete.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate YOLOv8 plant detector on the val split (CPU-only)"
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help=f"Path to .pt weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help=f"Dataset root directory (default: {DEFAULT_DATA})"
    )
    args = parser.parse_args()

    validate(weights=args.weights, dataset_dir=args.data)
