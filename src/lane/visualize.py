"""Lane detection visualizer and comparison grid generator.

Draws:
- Polylines and transparent drive-space polygon between ego-lanes
- Telemetry HUD with road curvature ($R$ in meters) and vehicle lateral offset
- Side-by-side comparison grids (Original, Classical BEV, U-Net Segmentation, Ground Truth)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.lane.detector import LaneResult
from src.lane.geometry import compute_perspective_matrices, render_lane_polygon

logger = logging.getLogger(__name__)


def draw_lane_overlay(
    frame: np.ndarray,
    result: LaneResult,
    detector_name: str = "LANE DETECTOR",
    fps: Optional[float] = None,
    color: Tuple[int, int, int] = (0, 230, 80),
    alpha: float = 0.35,
) -> np.ndarray:
    """Render full lane overlay including drive space polygon, polylines, and HUD."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # 1. Drive space polygon
    if result.has_ego_lanes:
        _, Minv = compute_perspective_matrices(img_size=(w, h))
        overlay = render_lane_polygon(
            overlay, result.left_poly, result.right_poly, Minv, color=color, alpha=alpha
        )

    # 2. Draw lane polylines
    for poly in result.polylines:
        if len(poly) > 1:
            pts = poly.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                overlay,
                [pts],
                isClosed=False,
                color=(0, 255, 255),
                thickness=4,
                lineType=cv2.LINE_AA,
            )

    # 3. Telemetry HUD
    hud_w, hud_h = 360, 110
    hud_bg = overlay.copy()
    cv2.rectangle(hud_bg, (15, 15), (15 + hud_w, 15 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.75, overlay, 0.25, 0, overlay)
    cv2.rectangle(overlay, (15, 15), (15 + hud_w, 15 + hud_h), (70, 70, 70), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        overlay,
        f"SYSTEM   : {detector_name.upper()}",
        (25, 38),
        font,
        0.52,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    curv_str = (
        f"{result.curvature_m:.0f} m"
        if result.curvature_m < 5000
        else "Straight (>5000m)"
    )
    cv2.putText(
        overlay,
        f"CURVATURE: {curv_str}",
        (25, 60),
        font,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    dir_str = "RIGHT" if result.offset_m > 0 else "LEFT"
    offset_str = (
        f"{abs(result.offset_m):.2f} m {dir_str}"
        if abs(result.offset_m) > 0.02
        else "CENTERED (0.00 m)"
    )
    cv2.putText(
        overlay,
        f"OFFSET   : {offset_str}",
        (25, 82),
        font,
        0.48,
        (0, 255, 160),
        1,
        cv2.LINE_AA,
    )

    fps_str = f"{fps:.1f} FPS" if fps is not None else "Real-time"
    cv2.putText(
        overlay,
        f"SPEED    : {fps_str}",
        (25, 104),
        font,
        0.48,
        (255, 200, 50),
        1,
        cv2.LINE_AA,
    )

    return overlay


def create_side_by_side_grid(
    original_frame: np.ndarray,
    classical_overlay: np.ndarray,
    unet_overlay: np.ndarray,
    gt_mask: np.ndarray,
    title: str = "Lane Detection Comparison",
) -> np.ndarray:
    """Create a 2x2 comparison grid:
    Top-Left: Original Camera View | Top-Right: Classical BEV Sliding Window
    Bottom-Left: U-Net Segmentation | Bottom-Right: Ground Truth Lane Mask
    """
    h, w = original_frame.shape[:2]
    half_w, half_h = w // 2, h // 2

    # Prepare 4 panels
    # Panel 1: Original
    p1 = cv2.resize(original_frame, (half_w, half_h))
    cv2.putText(
        p1,
        "1. Camera Input",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Panel 2: Classical BEV
    p2 = cv2.resize(classical_overlay, (half_w, half_h))
    cv2.putText(
        p2,
        "2. Classical BEV Polynomial",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Panel 3: U-Net
    p3 = cv2.resize(unet_overlay, (half_w, half_h))
    cv2.putText(
        p3,
        "3. U-Net Segmentation",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 100),
        2,
        cv2.LINE_AA,
    )

    # Panel 4: GT Mask colorized
    gt_color = np.zeros_like(original_frame)
    gt_color[gt_mask > 0] = (0, 255, 0)
    blended_gt = cv2.addWeighted(original_frame, 0.6, gt_color, 0.4, 0)
    p4 = cv2.resize(blended_gt, (half_w, half_h))
    cv2.putText(
        p4,
        "4. Ground Truth Mask",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # Assemble grid
    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bottom_row])

    # Add header banner
    header = np.zeros((50, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (30, 30, 30)
    cv2.putText(
        header,
        title.upper(),
        (30, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([header, grid])
