"""
backend/model_loader.py
------------------------
YOLOv8 model singleton loaded once at FastAPI app startup.

Pattern:
    - app lifespan (async context manager) calls load_model() on startup.
    - All request handlers call get_model() to retrieve the cached instance.
    - Forces device="cpu" — no CUDA required.

Usage (in main.py):
    from contextlib import asynccontextmanager
    from backend.model_loader import load_model, get_model, is_model_loaded

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        load_model()
        yield

    app = FastAPI(lifespan=lifespan)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weights path
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = Path("models") / "yolo" / "best.pt"

# Module-level singleton — populated by load_model()
_model = None
_weights_path: Path | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(weights: Path | str | None = None) -> None:
    """Load YOLOv8 model weights into memory (CPU-only).

    Called once during app startup via FastAPI lifespan.
    If weights file is not found, logs a warning and leaves the model as None;
    the /detect endpoint will return a descriptive 503 error.

    Args:
        weights: Path to a .pt file.  Defaults to models/yolo/best.pt.
                 Falls back to the base yolov8n.pt if the custom file is absent.
    """
    global _model, _weights_path

    from ultralytics import YOLO  # import here to keep startup import-time fast

    target = Path(weights) if weights else _DEFAULT_WEIGHTS

    if target.exists():
        logger.info("Loading YOLOv8 weights from: %s", target.resolve())
        _model = YOLO(str(target))
        _model.to("cpu")
        _weights_path = target
        logger.info("Model loaded successfully (%d classes).", len(_model.names))
    else:
        # Fall back to pretrained nano weights so the API stays functional
        fallback = "yolov8n.pt"
        logger.warning(
            "Weights not found at %s. Falling back to pretrained '%s'.",
            target, fallback,
        )
        logger.warning(
            "Run  python -m detection.train  to train on the MedLeaf dataset."
        )
        try:
            _model = YOLO(fallback)
            _model.to("cpu")
            _weights_path = Path(fallback)
            logger.info("Fallback model loaded (%d classes).", len(_model.names))
        except Exception as exc:
            logger.error("Could not load any model weights: %s", exc)
            _model = None
            _weights_path = None


def get_model():
    """Return the cached YOLO model instance.

    Returns:
        Loaded YOLO instance, or None if load_model() was not called / failed.
    """
    return _model


def is_model_loaded() -> bool:
    """Return True if a model is currently loaded and ready for inference."""
    return _model is not None


def get_weights_path() -> Path | None:
    """Return the path of the currently loaded weights file."""
    return _weights_path
