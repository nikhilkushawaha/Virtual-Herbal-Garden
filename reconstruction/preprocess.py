"""
reconstruction/preprocess.py
-----------------------------
Extract frames from a plant video and prepare them for COLMAP.

Usage:
    python reconstruction/preprocess.py --video path/to/plant.mp4 --scene data/scenes/rose
"""

import argparse
import cv2
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path: str, output_dir: str, every_n: int = 5) -> int:
    """Extract every N-th frame from a video file.

    Args:
        video_path: Path to input video.
        output_dir: Directory to write frame images.
        every_n:    Save one frame every N frames.

    Returns:
        Total number of frames saved.
    """
    out = Path(output_dir) / "images"
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0

    for i in tqdm(range(total), desc="Extracting frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if i % every_n == 0:
            cv2.imwrite(str(out / f"frame_{saved:05d}.jpg"), frame)
            saved += 1

    cap.release()
    print(f"Saved {saved} frames to {out}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from plant video")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--scene", required=True, help="Output scene directory, e.g. data/scenes/rose")
    parser.add_argument("--every-n", type=int, default=5, help="Extract every N-th frame")
    args = parser.parse_args()

    extract_frames(args.video, args.scene, args.every_n)
