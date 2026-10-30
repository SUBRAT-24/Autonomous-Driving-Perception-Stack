"""Lane geometry mathematics, perspective transforms, and metric utilities.

Includes:
1. Inverse Perspective Mapping (IPM / Bird's-Eye View): warp and unwarp matrices.
2. Sliding-window polynomial fitting on bird's-eye view binary images.
3. Radius of curvature estimation (standard automotive formulation).
4. Vehicle lateral offset from lane center.
5. Quantitative evaluation metrics: IoU, F1 score, Accuracy.
6. Lane polygon overlay projection back onto camera view.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# Standard calibration constants for forward-facing automotive camera (1280x720)
# Real-world lane width ~ 3.7 meters (~700 pixels in warped BEV space)
# Projected lookahead ~ 30 meters (~720 pixels in warped BEV space)
DEFAULT_YM_PER_PIX = 30.0 / 720.0
DEFAULT_XM_PER_PIX = 3.7 / 700.0


def get_default_perspective_points(
    img_w: int = 1280, img_h: int = 720
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute standard trapezoidal source and rectangular destination points for BEV.

    Source trapezoid covers the road surface in front of the vehicle up to the horizon.
    Destination rectangle maps this to a top-down orthogonal grid.
    """
    # Perspective trapezoid in camera view
    src = np.float32(
        [
            [img_w * 0.15, img_h * 0.95],  # Bottom left
            [img_w * 0.43, img_h * 0.58],  # Top left
            [img_w * 0.57, img_h * 0.58],  # Top right
            [img_w * 0.88, img_h * 0.95],  # Bottom right
        ]
    )

    # Orthogonal rectangle in BEV
    offset_x = img_w * 0.22
    dst = np.float32(
        [
            [offset_x, img_h],  # Bottom left
            [offset_x, 0.0],  # Top left
            [img_w - offset_x, 0.0],  # Top right
            [img_w - offset_x, img_h],  # Bottom right
        ]
    )

    return src, dst


def compute_perspective_matrices(
    src: Optional[np.ndarray] = None,
    dst: Optional[np.ndarray] = None,
    img_size: Tuple[int, int] = (1280, 720),
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward perspective warp matrix M and inverse unwarp matrix Minv.

    Returns:
        Tuple of (M, Minv), where M * Minv = Identity (3x3).
    """
    img_w, img_h = img_size
    if src is None or dst is None:
        src, dst = get_default_perspective_points(img_w, img_h)

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp_to_bev(
    image: np.ndarray, M: np.ndarray, dsize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Warp camera perspective image to top-down Bird's-Eye View."""
    h, w = image.shape[:2]
    out_size = dsize if dsize is not None else (w, h)
    return cv2.warpPerspective(image, M, out_size, flags=cv2.INTER_LINEAR)


def unwarp_from_bev(
    bev_image: np.ndarray, Minv: np.ndarray, dsize: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Unwarp top-down Bird's-Eye View image back to original camera perspective."""
    h, w = bev_image.shape[:2]
    out_size = dsize if dsize is not None else (w, h)
    return cv2.warpPerspective(bev_image, Minv, out_size, flags=cv2.INTER_LINEAR)


def fit_sliding_windows(
    binary_warped: np.ndarray,
    nwindows: int = 9,
    margin: int = 80,
    minpix: int = 40,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Fit 2nd order polynomials to left and right lanes using sliding windows.

    Args:
        binary_warped: Binary top-down BEV image (H, W), values {0, 1} or {0, 255}.
        nwindows: Number of vertical sliding windows.
        margin: Width of the window (+/- margin around center).
        minpix: Minimum number of pixels required to re-center window.

    Returns:
        Tuple of:
        - left_fit: [A, B, C] polynomial coefficients ($x = Ay^2 + By + C$) or None
        - right_fit: [A, B, C] polynomial coefficients or None
        - (leftx, lefty, rightx, righty): Raw pixel coordinate arrays
    """
    h, w = binary_warped.shape[:2]

    # Histogram of bottom half to find lane bases
    histogram = np.sum(binary_warped[h // 2 :, :], axis=0)
    midpoint = w // 2

    leftx_base = (
        int(np.argmax(histogram[:midpoint]))
        if np.max(histogram[:midpoint]) > 0
        else int(w * 0.25)
    )
    rightx_base = (
        int(np.argmax(histogram[midpoint:]) + midpoint)
        if np.max(histogram[midpoint:]) > 0
        else int(w * 0.75)
    )

    window_height = h // nwindows
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xleft_low)
            & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low)
            & (nonzeroy < win_y_high)
            & (nonzerox >= win_xright_low)
            & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = (
        np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    )
    right_lane_inds = (
        np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)
    )

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit: Optional[np.ndarray] = None
    right_fit: Optional[np.ndarray] = None

    if len(leftx) >= minpix:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) >= minpix:
        right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, (leftx, lefty, rightx, righty)


def evaluate_poly(poly: np.ndarray, y_vals: np.ndarray) -> np.ndarray:
    """Evaluate 2nd-order polynomial $x = Ay^2 + By + C$ at given y values."""
    return poly[0] * (y_vals**2) + poly[1] * y_vals + poly[2]


def calculate_curvature(
    poly: np.ndarray,
    y_eval: float = 720.0,
    ym_per_pix: float = DEFAULT_YM_PER_PIX,
    xm_per_pix: float = DEFAULT_XM_PER_PIX,
) -> float:
    """Calculate radius of curvature in meters using automotive formulation.

    For curve $x = A y^2 + B y + C$:
        R = (1 + (2 A y + B)^2)^(3/2) / |2 A|

    Args:
        poly: [A, B, C] polynomial in pixel units.
        y_eval: Vertical position at which to evaluate curvature (usually bottom of image).
        ym_per_pix: Meters per vertical pixel.
        xm_per_pix: Meters per horizontal pixel.

    Returns:
        Radius of curvature in meters. Returns 1e4 (effectively straight) if A is negligible.
    """
    if poly is None or len(poly) < 3:
        return 1e4

    # Convert polynomial coefficients to real-world meters:
    # x_m = A_m * y_m^2 + B_m * y_m + C_m
    # A_m = A * xm / ym^2, B_m = B * xm / ym
    A_m = poly[0] * xm_per_pix / (ym_per_pix**2)
    B_m = poly[1] * xm_per_pix / ym_per_pix

    y_eval_m = y_eval * ym_per_pix

    if abs(A_m) < 1e-7:
        return 1e4

    curvature = ((1.0 + (2.0 * A_m * y_eval_m + B_m) ** 2) ** 1.5) / abs(2.0 * A_m)
    return float(min(1e4, curvature))


def calculate_vehicle_offset(
    left_fit: Optional[np.ndarray],
    right_fit: Optional[np.ndarray],
    y_eval: float = 720.0,
    img_width: int = 1280,
    xm_per_pix: float = DEFAULT_XM_PER_PIX,
) -> float:
    """Calculate lateral offset of vehicle center from lane center in meters.

    Assumes the camera is mounted at vehicle centerline (img_width / 2).
    Offset > 0: vehicle is to the right of lane center.
    Offset < 0: vehicle is to the left of lane center.
    """
    center_vehicle = img_width / 2.0

    if left_fit is not None and right_fit is not None:
        left_x = evaluate_poly(left_fit, np.array([y_eval]))[0]
        right_x = evaluate_poly(right_fit, np.array([y_eval]))[0]
        lane_center = (left_x + right_x) / 2.0
        return float((center_vehicle - lane_center) * xm_per_pix)

    if left_fit is not None:
        left_x = evaluate_poly(left_fit, np.array([y_eval]))[0]
        # Assume lane is ~3.7m wide (700 px)
        lane_center = left_x + 350.0
        return float((center_vehicle - lane_center) * xm_per_pix)

    if right_fit is not None:
        right_x = evaluate_poly(right_fit, np.array([y_eval]))[0]
        lane_center = right_x - 350.0
        return float((center_vehicle - lane_center) * xm_per_pix)

    return 0.0


def render_lane_polygon(
    frame: np.ndarray,
    left_poly: Optional[np.ndarray],
    right_poly: Optional[np.ndarray],
    Minv: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 100),
    alpha: float = 0.35,
) -> np.ndarray:
    """Render transparent green lane drive space polygon onto the frame.

    Args:
        frame: Original camera view BGR frame (H, W, 3).
        left_poly: 2nd order polynomial for left lane in BEV space.
        right_poly: 2nd order polynomial for right lane in BEV space.
        Minv: Inverse perspective matrix (BEV -> camera view).
        color: Polygon fill BGR color.
        alpha: Blend transparency.

    Returns:
        Frame with lane polygon overlaid.
    """
    if left_poly is None or right_poly is None:
        return frame

    h, w = frame.shape[:2]
    ploty = np.linspace(0, h - 1, h)

    left_fitx = evaluate_poly(left_poly, ploty)
    right_fitx = evaluate_poly(right_poly, ploty)

    # BEV warp space canvas
    warp_zero = np.zeros((h, w, 3), dtype=np.uint8)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Fill polygon and draw boundary lines on BEV canvas
    cv2.fillPoly(warp_zero, np.int32([pts]), color)
    cv2.polylines(
        warp_zero,
        np.int32([pts_left]),
        isClosed=False,
        color=(255, 255, 255),
        thickness=16,
    )
    cv2.polylines(
        warp_zero,
        np.int32([pts_right]),
        isClosed=False,
        color=(255, 255, 255),
        thickness=16,
    )

    # Unwarp back to original camera view
    unwarped = unwarp_from_bev(warp_zero, Minv, (w, h))

    # Blend with original frame
    return cv2.addWeighted(frame, 1.0, unwarped, alpha, 0)


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) for binary masks."""
    p_bool = pred_mask > 0
    g_bool = gt_mask > 0

    intersection = np.logical_and(p_bool, g_bool).sum()
    union = np.logical_or(p_bool, g_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_mask_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute F1 score (Dice coefficient) for binary masks."""
    p_bool = pred_mask > 0
    g_bool = gt_mask > 0

    intersection = np.logical_and(p_bool, g_bool).sum()
    total = p_bool.sum() + g_bool.sum()

    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def compute_mask_accuracy(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-wise binary classification accuracy."""
    p_bool = pred_mask > 0
    g_bool = gt_mask > 0
    correct = (p_bool == g_bool).sum()
    return float(correct / pred_mask.size)
