"""
detection/
-----------
YOLOv8-based medicinal plant detection package.

Quick start:
    from detection.predict import predict_image
    from detection.utils   import load_class_names, draw_bbox, get_glb_path

Modules:
    train    — Download from Roboflow + fine-tune YOLOv8n (CPU)
    validate — Val-split metrics table (mAP50/75/50-95, per-class AP)
    predict  — Single-image inference → JSON result + annotated image
    utils    — load_class_names, draw_bbox, get_glb_path
"""

# Lazy imports — only pull in submodules when explicitly requested
# to avoid circular-import RuntimeWarnings when running submodules
# directly via `python -m detection.<module>`.

__all__ = [
    "predict_image",
    "draw_bbox",
    "get_glb_path",
    "load_class_names",
]


def __getattr__(name: str):  # PEP 562 lazy module attribute
    if name == "predict_image":
        from detection.predict import predict_image
        return predict_image
    if name in ("draw_bbox", "get_glb_path", "load_class_names"):
        import detection.utils as _utils
        return getattr(_utils, name)
    raise AttributeError(f"module 'detection' has no attribute {name!r}")
