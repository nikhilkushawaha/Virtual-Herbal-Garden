"""
backend/routers/models.py
--------------------------
Model serving endpoints — list available .glb meshes and serve them.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()

MODELS_DIR = Path("models/meshes")


@router.get("/")
async def list_models() -> dict:
    """Return a list of available .glb mesh files."""
    if not MODELS_DIR.exists():
        return {"models": []}
    glbs = [f.name for f in MODELS_DIR.glob("*.glb")]
    return {"models": glbs}


@router.get("/{name}")
async def get_model(name: str) -> FileResponse:
    """Download a .glb mesh by name."""
    path = MODELS_DIR / name
    if not path.exists() or path.suffix != ".glb":
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found.")
    return FileResponse(str(path), media_type="model/gltf-binary", filename=name)
