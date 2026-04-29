"""
backend/routers/reconstruct.py
-------------------------------
Reconstruction API endpoints — trigger NeRF pipeline for a given scene.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()


class ReconstructRequest(BaseModel):
    scene: str  # e.g. "data/scenes/rose"
    max_steps: int = 30_000


@router.post("/")
async def start_reconstruction(req: ReconstructRequest, background_tasks: BackgroundTasks) -> dict:
    """Kick off a NeRF reconstruction job in the background.

    Returns:
        Job ID and status.
    """
    # TODO: Enqueue reconstruction job and return job_id
    raise HTTPException(status_code=501, detail="Reconstruction endpoint not yet implemented.")


@router.get("/{job_id}")
async def get_reconstruction_status(job_id: str) -> dict:
    """Poll the status of a running reconstruction job."""
    raise HTTPException(status_code=501, detail="Job status endpoint not yet implemented.")
