"""
backend/main.py
---------------
Virtual Garden AI — FastAPI application.

Endpoints:
    POST /detect              — Upload image, detect plant, return JSON + metadata
    GET  /models/{species}.glb — Serve a pre-built GLB mesh file
    GET  /species             — List all 22 supported medicinal plant species
    GET  /viewer              — Serve the Three.js 3D viewer (viewer/index.html)
    GET  /health              — Health check with model-loaded status

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

API docs:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from __future__ import annotations

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from backend.cors_config import configure_cors
from backend.model_loader import get_model, is_model_loaded, load_model
from backend.species_db import SPECIES_DB, get_species_info, list_all_species

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("virtual_garden")

# ---------------------------------------------------------------------------
# Paths (Windows-safe pathlib)
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parent.parent   # project root
_VIEWER_HTML  = _ROOT / "viewer" / "index.html"
_GLB_ROOT     = _ROOT / "models" / "nerf_exports"
_PREDICTIONS  = _ROOT / "models" / "yolo" / "predictions"
_CONF_THRESH  = 0.30   # confidence threshold for detection

# ---------------------------------------------------------------------------
# FastAPI lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load YOLOv8 on startup; clean up (if needed) on shutdown."""
    logger.info("=== Virtual Garden AI — startup ===")
    load_model()                     # CPU-only; falls back to yolov8n.pt
    logger.info("=== Startup complete ===")
    yield
    logger.info("=== Virtual Garden AI — shutdown ===")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Virtual Garden AI",
    description=(
        "Detect Indian medicinal plants via YOLOv8 and explore them as "
        "interactive 3D models reconstructed with Tiny NeRF."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

configure_cors(app)


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

@app.post(
    "/detect",
    summary="Detect plant species from an uploaded image",
    tags=["Detection"],
    response_class=JSONResponse,
)
async def detect_plant(file: UploadFile = File(..., description="Plant image (JPEG / PNG)")):
    """Upload a plant photo and receive:
    - Detected species slug & confidence score
    - Bounding box [x1, y1, x2, y2]
    - URL to the pre-built GLB 3D model
    - Full botanical metadata from the species database

    If confidence < 0.30 or no detection, returns species='unknown'.
    """
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "YOLOv8 model is not loaded. "
                "Run  python -m detection.train  to train and generate best.pt, "
                "then restart the server."
            ),
        )

    # ------------------------------------------------------------------
    # Validate content type
    # ------------------------------------------------------------------
    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Upload a JPEG or PNG image.",
        )

    # ------------------------------------------------------------------
    # Save uploaded bytes to a temporary file (Windows-safe)
    # ------------------------------------------------------------------
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    # ------------------------------------------------------------------
    # YOLOv8 inference (CPU)
    # ------------------------------------------------------------------
    try:
        results = model.predict(
            source=str(tmp_path),
            conf=_CONF_THRESH,
            device="cpu",
            verbose=False,
        )
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Parse results
    # ------------------------------------------------------------------
    result = results[0]
    boxes  = result.boxes

    # Class names: prefer dataset-trained names, fall back to model.names
    class_names: list[str] = []
    if hasattr(model, "names") and model.names:
        class_names = [model.names[i] for i in sorted(model.names.keys())]

    if boxes is None or len(boxes) == 0:
        species_slug = "unknown"
        confidence   = 0.0
        bbox: list[float] = []
    else:
        confs   = boxes.conf.cpu().numpy()
        best_i  = int(confs.argmax())
        confidence = float(round(float(confs[best_i]), 4))
        cls_id     = int(boxes.cls.cpu().numpy()[best_i])
        xyxy       = boxes.xyxy.cpu().numpy()[best_i].tolist()
        bbox       = [round(v, 2) for v in xyxy]

        if confidence < _CONF_THRESH:
            species_slug = "unknown"
            confidence   = 0.0
            bbox         = []
        else:
            raw_name     = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            species_slug = raw_name.lower().replace(" ", "_")

    # ------------------------------------------------------------------
    # Enrich with species metadata
    # ------------------------------------------------------------------
    info = get_species_info(species_slug) or {
        "common_name":     species_slug.replace("_", " ").title() if species_slug != "unknown" else "Unknown",
        "scientific_name": "N/A",
        "family":          "N/A",
        "uses":            "N/A — species not in medicinal plant database",
        "local_name":      "N/A",
        "description":     "This species was not found in the database.",
    }

    # ------------------------------------------------------------------
    # GLB availability
    # ------------------------------------------------------------------
    glb_file = _GLB_ROOT / species_slug / f"{species_slug}.glb"
    glb_url  = f"/models/{species_slug}.glb" if glb_file.exists() else None

    # ------------------------------------------------------------------
    # Save annotated image (best-effort; non-fatal if it fails)
    # ------------------------------------------------------------------
    annotated_url: str | None = None
    if bbox and species_slug != "unknown":
        try:
            from PIL import Image, ImageDraw, ImageFont
            _PREDICTIONS.mkdir(parents=True, exist_ok=True)
            stem        = Path(file.filename or "upload").stem
            out_path    = _PREDICTIONS / f"{stem}_annotated.jpg"
            img         = Image.open(tmp_path if tmp_path.exists() else raw_bytes)
            # Re-open from bytes since tmp was deleted
            import io
            img         = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            draw        = ImageDraw.Draw(img)
            x1, y1, x2, y2 = bbox
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
            label = f"{info['common_name']}  {confidence:.0%}"
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except (IOError, OSError):
                font = ImageFont.load_default()
            draw.text((x1 + 4, y1 + 4), label, fill="lime", font=font)
            img.save(out_path, quality=90)
            annotated_url = f"/static/predictions/{out_path.name}"
        except Exception as ann_exc:
            logger.warning("Could not save annotated image: %s", ann_exc)

    # ------------------------------------------------------------------
    # Build response
    # ------------------------------------------------------------------
    response: dict = {
        "species":    species_slug,
        "confidence": confidence,
        "bbox":       bbox,
        "glb_url":    glb_url,
        "info":       {
            "common_name":     info["common_name"],
            "scientific_name": info["scientific_name"],
            "family":          info["family"],
            "uses":            info["uses"],
            "local_name":      info.get("local_name", "N/A"),
            "description":     info.get("description", ""),
        },
    }
    if annotated_url:
        response["annotated_image_url"] = annotated_url

    return JSONResponse(content=response)


# ---------------------------------------------------------------------------
# GET /models/{species}.glb
# ---------------------------------------------------------------------------

@app.get(
    "/models/{species_glb}",
    summary="Download a pre-built GLB 3D mesh",
    tags=["Models"],
)
async def serve_glb(species_glb: str):
    """Serve the NeRF-reconstructed GLB mesh for a given species.

    Path format: /models/tulsi.glb

    Returns 404 with a helpful message if the mesh has not been generated yet.
    """
    if not species_glb.endswith(".glb"):
        raise HTTPException(status_code=400, detail="Path must end with .glb, e.g. /models/tulsi.glb")

    species = species_glb[:-4]  # strip ".glb"
    glb_path = _GLB_ROOT / species / f"{species}.glb"

    if not glb_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"GLB model for '{species}' has not been generated yet. "
                f"Run the reconstruction pipeline:\n"
                f"  python -m reconstruction.run_all "
                f"--species {species} --images_dir path/to/images/"
            ),
        )

    return FileResponse(
        path=str(glb_path),
        media_type="model/gltf-binary",
        filename=f"{species}.glb",
    )


# ---------------------------------------------------------------------------
# GET /species
# ---------------------------------------------------------------------------

@app.get(
    "/species",
    summary="List all supported medicinal plant species",
    tags=["Species"],
)
async def get_species_list():
    """Return metadata for all 22 supported medicinal plant species.

    Each entry includes: slug, common_name, scientific_name, family,
    uses, local_name, description, and glb_available flag.
    """
    entries = list_all_species()

    # Annotate each entry with whether its GLB exists on disk
    enriched: list[dict] = []
    for entry in entries:
        slug     = entry["slug"]
        glb_path = _GLB_ROOT / slug / f"{slug}.glb"
        enriched.append({
            **entry,
            "glb_available": glb_path.exists(),
            "glb_url":       f"/models/{slug}.glb" if glb_path.exists() else None,
        })

    return JSONResponse(content={
        "total":   len(enriched),
        "species": enriched,
    })


# ---------------------------------------------------------------------------
# GET /viewer
# ---------------------------------------------------------------------------

@app.get(
    "/viewer",
    summary="Serve the Three.js 3D plant viewer",
    tags=["Viewer"],
    response_class=HTMLResponse,
)
async def serve_viewer():
    """Return viewer/index.html as an HTML page."""
    if not _VIEWER_HTML.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "viewer/index.html not found. "
                "Ensure the viewer/ directory is present in the project root."
            ),
        )
    return HTMLResponse(content=_VIEWER_HTML.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    summary="Health check",
    tags=["Health"],
)
async def health_check():
    """Return API health status and whether the YOLOv8 model is loaded."""
    from backend.model_loader import get_weights_path

    weights = get_weights_path()
    return JSONResponse(content={
        "status":        "ok",
        "model_loaded":  is_model_loaded(),
        "weights_file":  str(weights) if weights else None,
        "species_count": len(SPECIES_DB),
    })


# ---------------------------------------------------------------------------
# POST /reconstruct
# ---------------------------------------------------------------------------

def run_reconstruction_pipeline(species: str, video_path: Path):
    """Background task to extract frames and run NeRF pipeline."""
    logger.info(f"Starting background reconstruction for {species} from {video_path}")
    try:
        from reconstruction.preprocess import extract_frames
        from reconstruction.run_all import run_all

        # Create scenes directory
        scene_dir = _ROOT / "data" / "scenes" / species
        scene_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract frames
        logger.info(f"[{species}] Extracting frames from video...")
        extract_frames(str(video_path), str(scene_dir), every_n=5)

        # 2. Run pipeline
        logger.info(f"[{species}] Running NeRF pipeline...")
        images_dir = scene_dir / "images"
        run_all(
            species=species,
            images_dir=images_dir,
            skip_colmap=False,
            iterations=5000,
        )
        logger.info(f"[{species}] Reconstruction completed successfully!")
    except Exception as e:
        logger.error(f"[{species}] Reconstruction failed: {e}")
    finally:
        # Cleanup temp video
        try:
            video_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Could not delete temp video: {e}")

@app.post(
    "/reconstruct",
    summary="Trigger 3D Reconstruction from Video",
    tags=["Reconstruction"],
)
async def reconstruct_model(
    background_tasks: BackgroundTasks,
    species: str,
    file: UploadFile = File(..., description="Plant video (MP4/MOV)")
):
    """Upload a video of a plant to trigger the 3D NeRF reconstruction pipeline.
    This runs in the background and takes 30-90 minutes."""
    
    # Restrict file size (~100MB max) - typically handled by server config, but we can check content length if provided
    # Save video to temp location
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while content := await file.read(1024 * 1024):  # read in 1MB chunks
            tmp.write(content)
        tmp_path = Path(tmp.name)
        
    # Queue background task
    background_tasks.add_task(run_reconstruction_pipeline, species, tmp_path)
    
    return JSONResponse(content={
        "status": "queued",
        "message": f"Reconstruction pipeline started for {species}. This may take 30-90 minutes.",
        "species": species
    })


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse(content={
        "service": "Virtual Garden AI",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
        "species": "/species",
        "viewer":  "/viewer",
    })
