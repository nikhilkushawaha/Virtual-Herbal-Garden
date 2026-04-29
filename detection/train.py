"""
detection/train.py
------------------
Download the MedLeaf dataset from Roboflow and fine-tune YOLOv8n on CPU.

Usage:
    python detection/train.py
    python detection/train.py --epochs 100 --batch 8
"""

import argparse
import shutil
from pathlib import Path

from roboflow import Roboflow
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Roboflow dataset config
# ---------------------------------------------------------------------------
RF_API_KEY   = "QAtdVAv3A97k4BWSzPF7"
RF_WORKSPACE = "medicinal-plants-snwoh"
RF_PROJECT   = "medleaf"
RF_VERSION   = 7
RF_FORMAT    = "yolov8"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
MODELS_DIR   = Path("models/yolo")
BEST_PT_DEST = MODELS_DIR / "best.pt"


def download_dataset(download_dir: Path = Path("data/dataset")) -> Path:
    """Download the Roboflow dataset and return the path to its data.yaml."""
    print("\n📥  Downloading MedLeaf dataset from Roboflow…")
    rf      = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    dataset = project.version(RF_VERSION).download(RF_FORMAT, location=str(download_dir))
    yaml_path = Path(dataset.location) / "data.yaml"
    if not yaml_path.exists():
        # Some Roboflow exports use a different name
        candidates = list(Path(dataset.location).glob("*.yaml"))
        if not candidates:
            raise FileNotFoundError(f"No YAML found in {dataset.location}")
        yaml_path = candidates[0]
    print(f"✅  Dataset ready at: {yaml_path}")
    return yaml_path


def train(epochs: int = 100, batch: int = 8, imgsz: int = 640) -> Path:
    """Download dataset, train YOLOv8n, and copy best.pt to models/yolo/.

    Args:
        epochs: Training epochs.
        batch:  Batch size (keep ≤8 on CPU to avoid RAM exhaustion).
        imgsz:  Input image size in pixels.

    Returns:
        Path to the saved best.pt file.
    """
    yaml_path = download_dataset()

    print(f"\n🚀  Starting YOLOv8n training on CPU for {epochs} epochs…")
    model = YOLO("yolov8n.pt")  # nano – fastest on CPU
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=2,
        project="models/yolo/runs",
        name="medleaf",
        exist_ok=True,
        verbose=True,
    )

    # Locate the best weights produced by this run
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("models/yolo/runs/medleaf")
    trained_best = run_dir / "weights" / "best.pt"
    if not trained_best.exists():
        # Fallback: search common locations
        candidates = list(Path("models/yolo/runs").rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError("Could not locate best.pt after training.")
        trained_best = sorted(candidates)[-1]  # most recent

    # Copy to a stable, well-known path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(trained_best, BEST_PT_DEST)
    print(f"\n✅  best.pt saved to: {BEST_PT_DEST.resolve()}")
    return BEST_PT_DEST


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on the MedLeaf plant dataset (CPU-only)"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--batch",  type=int, default=8,   help="Batch size (default: 8)")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size (default: 640)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
