"""
backend/cors_config.py
----------------------
CORS middleware configuration for the Virtual Garden AI backend.

Allowed origins cover local development servers (Live Server, Vite, direct).
Tighten this list before deploying to production.
"""

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

# Origins that are allowed to make cross-origin requests to this API.
ALLOWED_ORIGINS: list[str] = [
    "http://localhost:5500",      # VS Code Live Server default
    "http://127.0.0.1:5500",
    "http://localhost:3000",      # Viewer served via python -m http.server 3000
    "http://127.0.0.1:3000",
    "http://localhost:8000",      # Same-origin (e.g. Swagger UI)
    "http://127.0.0.1:8000",
    "http://localhost:5173",      # Vite dev server
    "http://127.0.0.1:5173",
]


def configure_cors(app: FastAPI) -> None:
    """Attach CORSMiddleware to a FastAPI application instance.

    Args:
        app: The FastAPI application to configure.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],  # needed for file downloads
    )
