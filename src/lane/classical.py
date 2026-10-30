"""Classical Lane Detection implementations using OpenCV.

Provides:
1. ``HoughLaneDetector``: Baseline classical pipeline (Grayscale -> Gaussian Blur -> Canny -> ROI -> HoughLinesP).
2. ``SlidingWindowLaneDetector``: Upgraded advanced pipeline (HLS/Sobel thresholding -> BEV Inverse Perspective Warp -> Sliding-Window Polynomial Fitting -> Lane Polygon Overlay).

Both implement the unified ``LaneDetector`` interface:
    detect(frame) -> LaneResult(polylines, mask, left_poly, right_poly, curvature_m, offset_m)
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from src.lane.detector import LaneDetector, LaneResult
from src.lane.geometry import (
    calculate_curvature,
    calculate_vehicle_offset,
    compute_perspective_matrices,
    evaluate_poly,
    fit_sliding_windows,
    render_lane_polygon,
    unwarp_from_bev,
    warp_to_bev,
)

logger = logging.getLogger(__name__)


class HoughLaneDetector(LaneDetector):
    """Baseline Classical Lane Detector.

    Pipeline:
    1. Grayscale conversion
    2. Gaussian blur smoothing
    3. Canny edge detection
    4. Trapezoidal Region of Interest (ROI) mask
    5. Probabilistic Hough Transform (HoughLinesP)
    6. Slope filtering and line extrapolation
    """

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        hough_rho: int = 2,
        hough_theta: float = np.pi / 180,
        hough_thresh: int = 40,
        min_line_len: int = 40,
        max_line_gap: int = 25,
        roi_top_y_pct: float = 0.58,
        roi_bottom_y_pct: float = 0.98,
    ) -> None:
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_rho = hough_rho
        self.hough_theta = hough_theta
        self.hough_thresh = hough_thresh
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.roi_top_y_pct = roi_top_y_pct
        self.roi_bottom_y_pct = roi_bottom_y_pct

    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lanes using Hough transform."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Trapezoidal ROI mask
        mask_roi = np.zeros_like(edges)
        roi_vertices = np.array(
            [
                [
                    (int(w * 0.10), int(h * self.roi_bottom_y_pct)),
                    (int(w * 0.42), int(h * self.roi_top_y_pct)),
                    (int(w * 0.58), int(h * self.roi_top_y_pct)),
                    (int(w * 0.90), int(h * self.roi_bottom_y_pct)),
                ]
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(mask_roi, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask_roi)

        lines = cv2.HoughLinesP(
            masked_edges,
            rho=self.hough_rho,
            theta=self.hough_theta,
            threshold=self.hough_thresh,
            minLineLength=self.min_line_len,
            maxLineGap=self.max_line_gap,
        )

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        polylines: List[np.ndarray] = []

        if lines is None or len(lines) == 0:
            return LaneResult(polylines=polylines, mask=lane_mask)

        # Separate into left and right lines by slope
        # In image coords (y down): left lane slope < -0.3, right lane slope > 0.3
        left_lines: List[Tuple[float, float]] = []
        right_lines: List[Tuple[float, float]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter near-horizontal lines
            if abs(slope) < 0.3 or abs(slope) > 3.0:
                continue

            intercept = y1 - slope * x1
            if slope < 0 and x1 < w * 0.55 and x2 < w * 0.55:
                left_lines.append((slope, intercept))
            elif slope > 0 and x1 > w * 0.45 and x2 > w * 0.45:
                right_lines.append((slope, intercept))

        y_bottom = int(h * self.roi_bottom_y_pct)
        y_top = int(h * self.roi_top_y_pct)

        # Average left line
        if left_lines:
            avg_slope = np.mean([item[0] for item in left_lines])
            avg_intercept = np.mean([item[1] for item in left_lines])
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            pts_left = np.array([[x_bottom, y_bottom], [x_top, y_top]], dtype=np.int32)
            polylines.append(pts_left)
            cv2.line(lane_mask, (x_bottom, y_bottom), (x_top, y_top), 255, thickness=8)

        # Average right line
        if right_lines:
            avg_slope = np.mean([item[0] for item in right_lines])
            avg_intercept = np.mean([item[1] for item in right_lines])
            x_bottom = int((y_bottom - avg_intercept) / avg_slope)
            x_top = int((y_top - avg_intercept) / avg_slope)
            pts_right = np.array([[x_bottom, y_bottom], [x_top, y_top]], dtype=np.int32)
            polylines.append(pts_right)
            cv2.line(lane_mask, (x_bottom, y_bottom), (x_top, y_top), 255, thickness=8)

        return LaneResult(
            polylines=polylines,
            mask=lane_mask,
            curvature_m=1e4,  # Straight line assumption
            offset_m=0.0,
            confidence=(
                0.8 if len(polylines) == 2 else (0.5 if len(polylines) == 1 else 0.0)
            ),
        )


class SlidingWindowLaneDetector(LaneDetector):
    """Upgraded Advanced Classical Lane Detector (BEV Perspective Warp + Sliding Window).

    Pipeline:
    1. Multi-space color & edge thresholding (HLS S-channel, Grayscale, Sobel X)
    2. Bird's-Eye View (BEV) Inverse Perspective Mapping (IPM)
    3. Histogram-based lane base detection
    4. Vertical sliding window pixel clustering
    5. 2nd-order polynomial fitting ($x = Ay^2 + By + C$)
    6. Radius of curvature and vehicle lateral offset computation
    7. Lane polygon unwarping back to camera view
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (1280, 720),
        s_thresh: Tuple[int, int] = (100, 255),
        l_thresh: Tuple[int, int] = (150, 255),
        sobel_thresh: Tuple[int, int] = (30, 220),
        nwindows: int = 9,
        margin: int = 80,
        minpix: int = 40,
    ) -> None:
        self.img_w, self.img_h = img_size
        self.s_thresh = s_thresh
        self.l_thresh = l_thresh
        self.sobel_thresh = sobel_thresh
        self.nwindows = nwindows
        self.margin = margin
        self.minpix = minpix

        self.M, self.Minv = compute_perspective_matrices(img_size=img_size)

    def _threshold_image(self, frame: np.ndarray) -> np.ndarray:
        """Combine HLS color thresholding and Sobel x gradient."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sobel x on L channel
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        max_sobel = np.max(abs_sobelx)
        scaled_sobel = (
            np.uint8(255 * abs_sobelx / max_sobel)
            if max_sobel > 0
            else np.zeros_like(abs_sobelx, dtype=np.uint8)
        )

        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[
            (scaled_sobel >= self.sobel_thresh[0])
            & (scaled_sobel <= self.sobel_thresh[1])
        ] = 1

        # S-channel threshold
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= self.s_thresh[0]) & (s_channel <= self.s_thresh[1])] = 1

        # L-channel / White threshold
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= self.l_thresh[0]) & (l_channel <= self.l_thresh[1])] = 1

        # Gray threshold
        gray_binary = np.zeros_like(gray)
        gray_binary[gray >= 180] = 1

        # Combined binary
        combined = np.zeros_like(sobel_binary)
        combined[
            (sobel_binary == 1)
            | ((s_binary == 1) & (l_binary == 1))
            | (gray_binary == 1)
        ] = 1
        return combined

    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lanes using BEV sliding-window polynomial fitting."""
        h, w = frame.shape[:2]

        # 1. Threshold
        binary_thresh = self._threshold_image(frame)

        # 2. Bird's-Eye View Warp
        binary_warped = warp_to_bev(binary_thresh, self.M, (w, h))

        # 3. Sliding Window Polynomial Fitting
        left_fit, right_fit, _ = fit_sliding_windows(
            binary_warped,
            nwindows=self.nwindows,
            margin=self.margin,
            minpix=self.minpix,
        )

        polylines: List[np.ndarray] = []
        ploty = np.linspace(0, h - 1, h)

        # Canvas for mask in BEV space
        bev_mask = np.zeros((h, w), dtype=np.uint8)

        # Generate left lane polyline in camera view
        if left_fit is not None:
            left_fitx = evaluate_poly(left_fit, ploty)
            bev_left_pts = np.vstack([left_fitx, ploty]).T.astype(np.float32)
            # Draw on BEV mask
            cv2.polylines(
                bev_mask,
                [np.int32(bev_left_pts)],
                isClosed=False,
                color=255,
                thickness=12,
            )
            # Unwarp points back to camera view
            cam_left_pts = cv2.perspectiveTransform(
                bev_left_pts.reshape(-1, 1, 2), self.Minv
            ).reshape(-1, 2)
            polylines.append(cam_left_pts)

        # Generate right lane polyline in camera view
        if right_fit is not None:
            right_fitx = evaluate_poly(right_fit, ploty)
            bev_right_pts = np.vstack([right_fitx, ploty]).T.astype(np.float32)
            # Draw on BEV mask
            cv2.polylines(
                bev_mask,
                [np.int32(bev_right_pts)],
                isClosed=False,
                color=255,
                thickness=12,
            )
            # Unwarp points back to camera view
            cam_right_pts = cv2.perspectiveTransform(
                bev_right_pts.reshape(-1, 1, 2), self.Minv
            ).reshape(-1, 2)
            polylines.append(cam_right_pts)

        # Unwarp BEV mask to camera view
        lane_mask = unwarp_from_bev(bev_mask, self.Minv, (w, h))

        # 4. Curvature and lateral offset
        curv_left = (
            calculate_curvature(left_fit, y_eval=float(h))
            if left_fit is not None
            else 1e4
        )
        curv_right = (
            calculate_curvature(right_fit, y_eval=float(h))
            if right_fit is not None
            else 1e4
        )
        curvature = float(np.mean([curv_left, curv_right]))

        offset = calculate_vehicle_offset(
            left_fit, right_fit, y_eval=float(h), img_width=w
        )

        conf = (
            0.9
            if (left_fit is not None and right_fit is not None)
            else (0.5 if len(polylines) == 1 else 0.0)
        )

        return LaneResult(
            polylines=polylines,
            mask=lane_mask,
            left_poly=left_fit,
            right_poly=right_fit,
            curvature_m=curvature,
            offset_m=offset,
            confidence=conf,
        )

    def render_overlay(self, frame: np.ndarray, result: LaneResult) -> np.ndarray:
        """Render drive space polygon and HUD onto camera frame."""
        overlay = frame.copy()
        if result.has_ego_lanes:
            overlay = render_lane_polygon(
                overlay, result.left_poly, result.right_poly, self.Minv
            )
        return overlay
