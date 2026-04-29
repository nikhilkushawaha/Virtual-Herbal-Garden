"""
backend/__init__.py
-------------------
Virtual Garden AI — FastAPI backend package.

Quick start:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    POST /detect                  — YOLOv8 plant detection
    GET  /models/{species}.glb    — Serve NeRF-reconstructed 3D mesh
    GET  /species                 — List all 22 medicinal plant species
    GET  /viewer                  — Three.js 3D viewer
    GET  /health                  — Health check

Modules:
    main          — FastAPI application, all route handlers
    model_loader  — YOLOv8 singleton (load_model / get_model)
    species_db    — 22-species medicinal plant metadata dictionary
    cors_config   — CORS middleware setup
"""

__all__ = [
    "main",
    "model_loader",
    "species_db",
    "cors_config",
]
