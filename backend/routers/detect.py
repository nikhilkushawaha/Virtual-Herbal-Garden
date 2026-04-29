"""
backend/routers/detect.py
--------------------------
Detection API endpoints — accept an image, return YOLO detections.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil, tempfile

router = APIRouter()


@router.post("/")
async def detect_plant(file: UploadFile = File(...)) -> dict:
    """Upload a plant image and receive detection results.

    Returns:
        JSON with detected species and bounding boxes.
    """
    # TODO: call detection.predict.predict() here
    raise HTTPException(status_code=501, detail="Detection endpoint not yet implemented.")
