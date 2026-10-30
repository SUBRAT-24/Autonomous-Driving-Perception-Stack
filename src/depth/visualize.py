"""Depth and Distance Visualization Utilities.

Renders:
- Pseudo-colored depth maps (Inferno/Magma colormaps)
- Object distance tags with calibrated metres
- Pinhole camera cross-check comparison pills
- Color-coded Time-to-Collision (TTC) alerts (CRITICAL, WARNING, SAFE)
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def colorize_depth_map(
    depth_map: np.ndarray, colormap: int = cv2.COLORMAP_INFERNO
) -> np.ndarray:
    """Convert a single-channel float depth map to pseudo-colored 3-channel BGR."""
    d_min = np.min(depth_map)
    d_max = np.max(depth_map)
    if d_max > d_min:
        d_norm = ((depth_map - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    else:
        d_norm = np.zeros_like(depth_map, dtype=np.uint8)

    return cv2.applyColorMap(d_norm, colormap)


def draw_distance_and_ttc_overlay(
    frame: np.ndarray,
    track_id: int,
    cls_name: str,
    bbox: Tuple[float, float, float, float],
    metric_distance_m: float,
    pinhole_distance_m: float,
    ttc_seconds: float,
    ttc_status: str,
) -> None:
    """Render bounding box, lower-third sampling highlight, distance, and TTC badge."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h_box = y2 - y1

    # Status color code: CRITICAL = Red, WARNING = Orange, SAFE/APPROACHING = Cyan/Green
    if ttc_status == "CRITICAL":
        color = (0, 0, 255)  # Red
    elif ttc_status == "WARNING":
        color = (0, 140, 255)  # Orange
    elif ttc_status == "APPROACHING":
        color = (0, 220, 255)  # Yellow
    else:
        color = (0, 255, 120)  # Green

    # 1. Main bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    # 2. Lower-third sampling highlight
    y_lower = int(y1 + (2.0 / 3.0) * h_box)
    cv2.rectangle(frame, (x1, y_lower), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    # 3. Text label
    ttc_str = f"TTC: {ttc_seconds:.1f}s" if ttc_seconds < 50.0 else "TTC: Safe"
    diff_m = abs(metric_distance_m - pinhole_distance_m)
    label_main = f"#{track_id} {cls_name} {metric_distance_m:.1f}m | {ttc_str}"
    label_sub = f"Pinhole: {pinhole_distance_m:.1f}m (d: {diff_m:.1f}m)"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    (tw, th), _ = cv2.getTextSize(label_main, font, font_scale, 1)

    pill_top = max(0, y1 - th * 2 - 12)
    pill_bottom = y1
    pill_right = min(frame.shape[1] - 1, x1 + tw + 12)

    cv2.rectangle(frame, (x1, pill_top), (pill_right, pill_bottom), color, -1)
    text_color = (
        (255, 255, 255) if ttc_status in ("CRITICAL", "WARNING") else (10, 10, 10)
    )

    cv2.putText(
        frame,
        label_main,
        (x1 + 4, pill_top + th + 4),
        font,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        label_sub,
        (x1 + 4, pill_bottom - 4),
        font,
        0.40,
        text_color,
        1,
        cv2.LINE_AA,
    )
