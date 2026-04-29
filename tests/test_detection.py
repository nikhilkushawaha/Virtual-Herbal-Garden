"""
tests/test_detection.py
------------------------
Smoke tests for the detection package.
"""

import pytest
from pathlib import Path


def test_utils_ensure_dir(tmp_path):
    from detection.utils import ensure_dir
    target = tmp_path / "new" / "nested"
    result = ensure_dir(target)
    assert result.exists()
    assert result.is_dir()


def test_load_image_missing_raises():
    from detection.utils import load_image
    with pytest.raises(FileNotFoundError):
        load_image("non_existent_image.jpg")
