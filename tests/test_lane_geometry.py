"""Unit tests for lane geometry mathematics, perspective transforms, and detector interfaces.

Tests:
1. Perspective matrix computation and inversion property (M * Minv == Identity).
2. BEV warping and unwarping geometric consistency.
3. Polynomial evaluation and sliding window fitting on synthetic BEV lines.
4. Road radius of curvature calculation ($R \\to \\infty$ for straight lines).
5. Vehicle lateral offset calculation (centered, left, and right).
6. Quantitative mask metrics: IoU, F1 score, and pixel accuracy.
7. HoughLaneDetector contract compliance (detect(frame) -> LaneResult).
8. SlidingWindowLaneDetector contract compliance (detect(frame) -> LaneResult).
9. UNetLaneDetector contract compliance (detect(frame) -> LaneResult).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.lane.classical import HoughLaneDetector, SlidingWindowLaneDetector
from src.lane.detector import LaneDetector, LaneResult
from src.lane.geometry import (
    calculate_curvature,
    calculate_vehicle_offset,
    compute_mask_accuracy,
    compute_mask_f1,
    compute_mask_iou,
    compute_perspective_matrices,
    evaluate_poly,
    fit_sliding_windows,
    unwarp_from_bev,
    warp_to_bev,
)
from src.lane.segmentation import UNetLaneDetector


# ============================================================================
# 1. Perspective Transformation Tests
# ============================================================================
def test_perspective_matrices_inversion():
    """Verify that M and Minv are true inverse matrices: M * Minv = Identity(3)."""
    M, Minv = compute_perspective_matrices(img_size=(1280, 720))

    assert M.shape == (3, 3)
    assert Minv.shape == (3, 3)

    # Inversion check
    prod = np.dot(M, Minv)
    identity = np.eye(3, dtype=np.float64)

    # Normalize homogeneous scale
    prod /= prod[2, 2]
    assert np.allclose(prod, identity, atol=1e-4)


def test_warp_and_unwarp_roundtrip():
    """Verify image warping to BEV and unwarping preserves original dimensions."""
    M, Minv = compute_perspective_matrices(img_size=(1280, 720))
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.circle(img, (640, 500), 40, (255, 255, 255), -1)

    warped = warp_to_bev(img, M, (1280, 720))
    assert warped.shape == (720, 1280, 3)

    unwarped = unwarp_from_bev(warped, Minv, (1280, 720))
    assert unwarped.shape == (720, 1280, 3)


# ============================================================================
# 2. Polynomial Fitting & Evaluation Tests
# ============================================================================
def test_evaluate_poly_math():
    """Verify polynomial evaluation: x = A*y^2 + B*y + C."""
    # Straight vertical line: x = 0*y^2 + 0*y + 400 = 400
    poly_straight = np.array([0.0, 0.0, 400.0])
    y_vals = np.array([0.0, 360.0, 720.0])
    x_vals = evaluate_poly(poly_straight, y_vals)
    assert np.allclose(x_vals, [400.0, 400.0, 400.0])

    # Quadratic curve: x = 0.001*y^2 + 0.1*y + 200
    poly_curve = np.array([0.001, 0.1, 200.0])
    y_test = np.array([10.0])
    expected = 0.001 * (10.0**2) + 0.1 * 10.0 + 200.0  # 0.1 + 1.0 + 200.0 = 201.1
    assert evaluate_poly(poly_curve, y_test)[0] == pytest.approx(expected)


def test_fit_sliding_windows_on_synthetic_lanes():
    """Verify sliding-window histogram algorithm fits polynomial on binary BEV lines."""
    h, w = 720, 1280
    bev_binary = np.zeros((h, w), dtype=np.uint8)

    # Draw vertical left lane at x=300 and right lane at x=980
    bev_binary[:, 296:304] = 1
    bev_binary[:, 976:984] = 1

    left_fit, right_fit, _ = fit_sliding_windows(
        bev_binary, nwindows=9, margin=60, minpix=20
    )

    assert left_fit is not None
    assert right_fit is not None

    # Both lines are vertical -> A and B close to 0, C close to 300 and 980
    assert abs(left_fit[0]) < 1e-3
    assert left_fit[2] == pytest.approx(300.0, abs=15.0)

    assert abs(right_fit[0]) < 1e-3
    assert right_fit[2] == pytest.approx(980.0, abs=15.0)


# ============================================================================
# 3. Radius of Curvature Tests
# ============================================================================
def test_curvature_straight_line():
    """Straight line (A=0) must have an effectively infinite curvature radius (>5000m)."""
    poly_straight = np.array([0.0, 0.0, 500.0])
    r = calculate_curvature(poly_straight, y_eval=720.0)
    assert r >= 5000.0


def test_curvature_curved_road():
    """Curved polynomial must return a finite positive radius in realistic highway range."""
    # Typical highway turn: A ~ 0.0003 in pixel space
    poly_curve = np.array([3e-4, -0.2, 400.0])
    r = calculate_curvature(poly_curve, y_eval=720.0)
    # Realistic road curves are between 100m and 3000m
    assert 50.0 < r < 5000.0


# ============================================================================
# 4. Vehicle Lateral Offset Tests
# ============================================================================
def test_vehicle_offset_centered():
    """Vehicle centered between lanes (lane center = 640 px = img_w / 2) has 0m offset."""
    left_fit = np.array([0.0, 0.0, 290.0])
    right_fit = np.array([0.0, 0.0, 990.0])  # midpoint = (290+990)/2 = 640

    offset = calculate_vehicle_offset(left_fit, right_fit, y_eval=720.0, img_width=1280)
    assert offset == pytest.approx(0.0, abs=0.01)


def test_vehicle_offset_shifted_right_and_left():
    """Vehicle shifted right gives offset > 0; vehicle shifted left gives offset < 0."""
    # Lane center shifted to left (e.g. 600 px) -> vehicle (at 640) is to the right
    left_shift_l = np.array([0.0, 0.0, 250.0])
    right_shift_l = np.array([0.0, 0.0, 950.0])  # center = 600
    offset_right = calculate_vehicle_offset(
        left_shift_l, right_shift_l, y_eval=720.0, img_width=1280
    )
    assert offset_right > 0.1  # Vehicle is right of lane center

    # Lane center shifted to right (e.g. 680 px) -> vehicle is to the left
    left_shift_r = np.array([0.0, 0.0, 330.0])
    right_shift_r = np.array([0.0, 0.0, 1030.0])  # center = 680
    offset_left = calculate_vehicle_offset(
        left_shift_r, right_shift_r, y_eval=720.0, img_width=1280
    )
    assert offset_left < -0.1  # Vehicle is left of lane center


# ============================================================================
# 5. Mask Metric Tests (IoU, F1, Accuracy)
# ============================================================================
def test_mask_metrics_exact_math():
    """Verify IoU and F1 metrics on known synthetic binary masks."""
    # 1. Perfect match
    m1 = np.zeros((100, 100), dtype=np.uint8)
    m1[20:50, 20:50] = 255
    assert compute_mask_iou(m1, m1) == pytest.approx(1.0)
    assert compute_mask_f1(m1, m1) == pytest.approx(1.0)
    assert compute_mask_accuracy(m1, m1) == pytest.approx(1.0)

    # 2. No overlap
    m2 = np.zeros((100, 100), dtype=np.uint8)
    m2[60:90, 60:90] = 255
    assert compute_mask_iou(m1, m2) == pytest.approx(0.0)
    assert compute_mask_f1(m1, m2) == pytest.approx(0.0)

    # 3. Exactly 50% intersection
    # Area m1 = 100 px, Area m3 = 100 px, intersection = 50 px, union = 150 px
    m_a = np.zeros((10, 20), dtype=np.uint8)
    m_b = np.zeros((10, 20), dtype=np.uint8)
    m_a[:, 0:10] = 255  # 100 px
    m_b[:, 5:15] = 255  # 100 px (overlap is cols 5:10 = 50 px)

    expected_iou = 50.0 / 150.0  # 1/3
    expected_f1 = (2.0 * 50.0) / (100.0 + 100.0)  # 0.5
    assert compute_mask_iou(m_a, m_b) == pytest.approx(expected_iou)
    assert compute_mask_f1(m_a, m_b) == pytest.approx(expected_f1)


# ============================================================================
# 6. Unified LaneDetector Interface Contract Tests
# ============================================================================
def test_hough_lane_detector_contract():
    """Verify HoughLaneDetector implements LaneDetector.detect(frame) -> LaneResult."""
    detector = HoughLaneDetector()
    assert isinstance(detector, LaneDetector)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw two diagonal lines
    cv2.line(frame, (200, 700), (550, 420), (255, 255, 255), 6)
    cv2.line(frame, (1080, 700), (730, 420), (255, 255, 255), 6)

    result = detector.detect(frame)
    assert isinstance(result, LaneResult)
    assert isinstance(result.polylines, list)
    assert isinstance(result.mask, np.ndarray)
    assert result.mask.shape == (720, 1280)


def test_sliding_window_detector_contract():
    """Verify SlidingWindowLaneDetector implements LaneDetector.detect(frame) -> LaneResult."""
    detector = SlidingWindowLaneDetector()
    assert isinstance(detector, LaneDetector)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.line(frame, (200, 700), (550, 420), (255, 255, 255), 8)
    cv2.line(frame, (1080, 700), (730, 420), (255, 255, 255), 8)

    result = detector.detect(frame)
    assert isinstance(result, LaneResult)
    assert isinstance(result.polylines, list)
    assert isinstance(result.mask, np.ndarray)
    assert hasattr(result, "curvature_m")
    assert hasattr(result, "offset_m")


def test_unet_lane_detector_contract():
    """Verify UNetLaneDetector implements LaneDetector.detect(frame) -> LaneResult."""
    detector = UNetLaneDetector(device="cpu")
    assert isinstance(detector, LaneDetector)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(frame)

    assert isinstance(result, LaneResult)
    assert isinstance(result.polylines, list)
    assert isinstance(result.mask, np.ndarray)
    assert result.mask.shape == (720, 1280)
