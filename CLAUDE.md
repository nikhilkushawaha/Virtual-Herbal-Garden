# Medicinal Plant Detection & 3D Reconstruction

## Project Goal
Build an end-to-end AI pipeline that:
1. Detects medicinal plants from real garden images using YOLOv8
2. Reconstructs a 3D model of the detected plant using a CPU-compatible NeRF pipeline
3. Displays the labeled, interactive 3D model in a Three.js web viewer
4. Connects everything via a FastAPI backend

## Environment
- OS: Windows (native, not WSL)
- GPU: None — all code must run on CPU only
- Python: 3.10+
- Shell: PowerShell or CMD

## Architecture
- Detection: YOLOv8n (Ultralytics) trained on 22 medicinal species via Roboflow
- 3D Reconstruction: COLMAP (camera poses) + TinyNeRF or torch-ngp CPU mode
  → export final mesh as .glb using trimesh + open3d
- Viewer: Three.js (single HTML file) with OrbitControls, floating label, botanical info
- Backend: FastAPI serving detection results + .glb files + species metadata

## Dataset
- Source: Roboflow workspace "medicinal-plants-snwoh", project "medleaf", version 7
- Format: YOLOv8
- API Key: QAtdVAv3A97k4BWSzPF7
- Classes: 22 medicinal plant species

## Project Structure
medicinal-plant-3d/
├── CLAUDE.md
├── detection/
│   ├── train.py
│   ├── validate.py
│   ├── predict.py
│   └── utils.py
├── reconstruction/
│   ├── prepare_data.py
│   ├── train_nerf.py
│   ├── export_mesh.py
│   └── run_all.py
├── viewer/
│   └── index.html
├── backend/
│   ├── main.py
│   ├── species_db.py
│   ├── model_loader.py
│   └── cors_config.py
├── models/
│   ├── yolo/
│   └── nerf_exports/
├── data/
│   └── <species_name>/
│       ├── images/
│       └── colmap_output/
└── requirements.txt

## Critical Constraints
- NO CUDA, NO GPU — every script must have a CPU fallback
- Windows-safe paths only — use pathlib everywhere, never hardcode forward slashes
- NeRF training will be slow on CPU (~30–90 min per species) — that is acceptable
- Three.js viewer is a static single HTML file served by FastAPI at GET /viewer
- YOLOv8 inference must complete in under 10 seconds on CPU
- COLMAP must be installed separately by the user (provide install instructions)

## Code Rules
- Python 3.10+, pathlib for all paths, argparse for all CLIs
- logging module only (no bare print statements except notebooks)
- Docstrings on every function
- requirements.txt must pin all versions