"""
detection/predict.py
--------------------
Run YOLOv8 inference on a single image, print JSON result, and save an
annotated image.

Usage:
    python detection/predict.py --image path/to/plant.jpg
    python detection/predict.py --image path/to/plant.jpg --conf 0.5 --weights models/yolo/best.pt

Output (stdout):
    {"species": "rose", "confidence": 0.87, "bbox": [x1, y1, x2, y2]}
    or
    {"species": "unknown", "confidence": 0.0, "bbox": []}

Annotated image saved to:
    models/yolo/predictions/<original_stem>_annotated.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so 'detection.utils' can be found when run directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PIL import Image
from ultralytics import YOLO

from detection.utils import draw_bbox, get_glb_path, load_class_names

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS  = Path("models/yolo/best.pt")
DEFAULT_CONF     = 0.5
PREDICTIONS_DIR  = Path("models/yolo/predictions")
DATASET_YAML     = Path("data/dataset/data.yaml")  # written after Roboflow download


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def predict_image(
    image_path: str | Path,
    weights: Path = DEFAULT_WEIGHTS,
    conf_threshold: float = DEFAULT_CONF,
) -> dict:
    """Run YOLOv8 inference on a single image.

    Args:
        image_path:      Path to the input image file.
        weights:         Path to trained .pt weights.
        conf_threshold:  Minimum confidence to accept a detection.

    Returns:
        Dict with keys: species, confidence, bbox ([x1,y1,x2,y2] or []).
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}\n"
            "Run `python detection/train.py` first."
        )

    model = YOLO(str(weights))

    # ------------------------------------------------------------------
    # Run inference (CPU-only, single image)
    # ------------------------------------------------------------------
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        device="cpu",
        verbose=False,
    )

    # results is always a list; we asked for one image
    result = results[0]
    boxes  = result.boxes

    # ------------------------------------------------------------------
    # Load class names (from the dataset YAML if available, else use the
    # names stored inside the model itself as a fallback)
    # ------------------------------------------------------------------
    if DATASET_YAML.exists():
        class_names = load_class_names(DATASET_YAML)
    elif hasattr(model, "names") and model.names:
        class_names = [model.names[i] for i in sorted(model.names.keys())]
    else:
        class_names = [str(i) for i in range(1000)]

    # ------------------------------------------------------------------
    # Build output dict
    # ------------------------------------------------------------------
    if boxes is None or len(boxes) == 0:
        output = {"species": "unknown", "confidence": 0.0, "bbox": []}
    else:
        # Take the detection with the highest confidence
        confs   = boxes.conf.cpu().numpy()
        best_i  = int(confs.argmax())
        best_conf  = float(confs[best_i])
        best_cls   = int(boxes.cls.cpu().numpy()[best_i])
        best_xyxy  = boxes.xyxy.cpu().numpy()[best_i].tolist()  # [x1,y1,x2,y2]

        if best_conf < conf_threshold:
            output = {"species": "unknown", "confidence": 0.0, "bbox": []}
        else:
            species = class_names[best_cls] if best_cls < len(class_names) else str(best_cls)
            output  = {
                "species":    species,
                "confidence": round(best_conf, 4),
                "bbox":       [round(v, 2) for v in best_xyxy],
            }

    # ------------------------------------------------------------------
    # Save annotated image
    # ------------------------------------------------------------------
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pil_img = Image.open(image_path).convert("RGB")

    if output["bbox"]:
        annotated = draw_bbox(
            pil_img,
            bbox=output["bbox"],
            label=output["species"],
            confidence=output["confidence"],
        )
    else:
        # No detection — still save the original with a "No detection" watermark
        from PIL import ImageDraw, ImageFont
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (IOError, OSError):
            font = ImageFont.load_default()
        draw.text((10, 10), "No detection above threshold", fill="red", font=font)

    out_path = PREDICTIONS_DIR / f"{image_path.stem}_annotated.jpg"
    annotated.save(out_path, quality=95)

    # ------------------------------------------------------------------
    # GLB path hint (informational — reconstruction may not have run yet)
    # ------------------------------------------------------------------
    if output["species"] != "unknown":
        glb = get_glb_path(output["species"])
        output["glb_path"] = str(glb)
        output["glb_exists"] = glb.exists()

    output["annotated_image"] = str(out_path)
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 plant detection — CPU inference"
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--weights", type=Path, default=DEFAULT_WEIGHTS,
        help=f"Path to .pt weights (default: {DEFAULT_WEIGHTS})"
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default: {DEFAULT_CONF})"
    )
    args = parser.parse_args()

    try:
        result = predict_image(
            image_path=args.image,
            weights=args.weights,
            conf_threshold=args.conf,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Print clean JSON to stdout
    print(json.dumps(result, indent=2))

    # Human-readable summary
    print(f"\n🌿  Species    : {result['species']}")
    print(f"🎯  Confidence : {result['confidence']:.0%}")
    if result.get("bbox"):
        x1, y1, x2, y2 = result["bbox"]
        print(f"📦  BBox       : x1={x1}  y1={y1}  x2={x2}  y2={y2}")
    if result.get("glb_path"):
        exists = "✅ exists" if result.get("glb_exists") else "⚠️  not yet reconstructed"
        print(f"🧊  GLB model  : {result['glb_path']}  ({exists})")
    print(f"🖼️   Annotated  : {result['annotated_image']}")
