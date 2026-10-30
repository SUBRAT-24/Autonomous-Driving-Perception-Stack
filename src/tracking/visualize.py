"""Motion trail visualizer and annotated tracking demo generator.

Draws:
- Bounding boxes colored by unique Track ID
- Label pills with ID, class name, and confidence score
- Smooth polyline motion trails trailing behind moving objects
- Informational HUD with real-time FPS and active track count

Usage::

    python -m src.tracking.visualize --input data/sample/clips/clip1_highway_day.mp4 --output docs/results/tracking_demo.mp4
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from src.tracking.tracker import Track, Tracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, bright BGR color from a track ID."""
    # Golden ratio color generation
    hue = int((track_id * 137.5) % 180)
    hsv_pixel = np.uint8([[[hue, 220, 240]]])
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_pixel[0]), int(bgr_pixel[1]), int(bgr_pixel[2])


def draw_motion_trail(
    frame: np.ndarray,
    trail: Deque[Tuple[int, int]],
    color: Tuple[int, int, int],
) -> None:
    """Draw smooth fading motion trail polyline."""
    if len(trail) < 2:
        return

    pts = list(trail)
    n = len(pts)

    for i in range(1, n):
        # Thickness and brightness increase towards the head
        alpha = i / n
        thickness = max(1, int(3 * alpha))
        c = (int(color[0] * alpha), int(color[1] * alpha), int(color[2] * alpha))
        cv2.line(frame, pts[i - 1], pts[i], c, thickness, lineType=cv2.LINE_AA)


def draw_track_box(
    frame: np.ndarray,
    track: Track,
    color: Tuple[int, int, int],
) -> None:
    """Draw bounding box and label pill with ID, class name, and confidence."""
    x1, y1, x2, y2 = [int(v) for v in track.bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1] - 1, x2), min(frame.shape[0] - 1, y2)

    # 1. Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    # 2. Text label
    label = f"#{track.id} {track.cls} {track.conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Label background pill
    pill_top = max(0, y1 - text_h - 8)
    pill_bottom = y1
    pill_right = min(frame.shape[1] - 1, x1 + text_w + 10)

    cv2.rectangle(frame, (x1, pill_top), (pill_right, pill_bottom), color, -1)
    # Contrasting text (dark text on light colors, white text on dark colors)
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

    cv2.putText(
        frame,
        label,
        (x1 + 4, pill_bottom - 4),
        font,
        font_scale,
        text_color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    tracker_name: str,
    frame_idx: int,
    total_frames: int,
    active_count: int,
    fps: float,
) -> None:
    """Render top-left telemetry HUD."""
    overlay = frame.copy()
    hud_w, hud_h = 320, 105
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + hud_w, 15 + hud_h), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        frame,
        f"TRACKER: {tracker_name.upper()}",
        (25, 38),
        font,
        0.55,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FRAME   : {frame_idx} / {total_frames}",
        (25, 60),
        font,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ACTIVE  : {active_count} confirmed tracks",
        (25, 82),
        font,
        0.48,
        (0, 255, 160),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"SPEED   : {fps:.1f} FPS",
        (25, 104),
        font,
        0.48,
        (255, 200, 50),
        1,
        cv2.LINE_AA,
    )


def generate_annotated_demo_clip(
    input_video: Path,
    output_video: Path,
    tracker_type: str = "bytetrack",
    weights_path: Optional[Path] = None,
    trail_length: int = 25,
) -> Path:
    """Generate an annotated video clip with bounding boxes and motion trails.

    Args:
        input_video: Path to input MP4 video.
        output_video: Destination MP4 video path.
        tracker_type: 'bytetrack' or 'deepsort'.
        weights_path: Path to detector weights.
        trail_length: Number of past positions in motion trail.

    Returns:
        Path to generated video file.
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {input_video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    tracker = Tracker(tracker_type=tracker_type, weights_path=weights_path)
    trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
        lambda: deque(maxlen=trail_length)
    )

    frame_idx = 0
    logger.info(
        "Generating annotated demo clip from %s -> %s", input_video, output_video
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        tracks = tracker.update(frame)

        # Update motion trails
        active_ids = set()
        for t in tracks:
            active_ids.add(t.id)
            cx, cy = int(t.center[0]), int(t.center[1])
            trails[t.id].append((cx, cy))

        # Render motion trails
        for t_id, trail in trails.items():
            col = get_color_for_id(t_id)
            draw_motion_trail(frame, trail, col)

        # Render bounding boxes and labels
        for t in tracks:
            col = get_color_for_id(t.id)
            draw_track_box(frame, t, col)

        # Telemetry HUD
        draw_hud(frame, tracker_type, frame_idx, total_frames, len(tracks), fps)

        writer.write(frame)

    cap.release()
    writer.release()
    logger.info("Saved annotated demo clip to %s (%d frames)", output_video, frame_idx)
    return output_video


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotated tracking demo video"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/sample/clips/clip1_highway_day.mp4")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/results/tracking_demo.mp4")
    )
    parser.add_argument(
        "--tracker", default="bytetrack", choices=["bytetrack", "deepsort"]
    )
    args = parser.parse_args()

    generate_annotated_demo_clip(args.input, args.output, tracker_type=args.tracker)


if __name__ == "__main__":
    main()
