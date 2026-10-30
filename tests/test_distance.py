"""Unit tests for distance calculation, depth extraction, calibration, and TTC.

Tests:
1. Lower-third median relative depth extraction from bounding boxes.
2. Calibrated metric distance conversion (Z = 1 / (a * d + b)).
3. Pinhole camera geometry cross-check (Z = f * H / h).
4. Time-to-Collision (TTC) calculation and moving-average smoothing.
5. Error metrics evaluation across distance bands (0-10m, 10-30m, 30+m).
6. DepthEstimator interface contract (predict(frame) -> relative depth map).
7. CARLA 24-bit depth pixel decoding math.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.depth.calibration import carla_depth_to_meters
from src.depth.distance import (
    CARLA_FOCAL_LENGTH_PX,
    TimeToCollisionTracker,
    calculate_band_errors,
    compute_pinhole_distance,
    convert_relative_to_metric,
    extract_object_depth,
)
from src.depth.estimator import DepthEstimator


# ============================================================================
# 1. Lower-Third Depth Extraction Tests
# ============================================================================
def test_extract_object_depth_lower_third():
    """Verify that depth extraction isolates the lower third of the bounding box."""
    # Synthetic depth map: top is 0.5 (background), bottom is 2.5 (road contact patch)
    depth_map = np.full((100, 100), 0.5, dtype=np.float32)
    # Box from y=10 to y=70 (height 60)
    # Lower third is y=50 to y=70
    depth_map[50:70, 20:80] = 2.5

    bbox = (20.0, 10.0, 80.0, 70.0)
    med_val = extract_object_depth(depth_map, bbox, lower_third_only=True)

    # Median over lower third must be 2.5, ignoring the 0.5 in the top 2/3
    assert med_val == pytest.approx(2.5, abs=0.01)


def test_extract_object_depth_clamping():
    """Verify robust clamping when bounding box extends outside image boundaries."""
    depth_map = np.ones((50, 50), dtype=np.float32) * 1.8
    bbox_outside = (-20.0, -10.0, 80.0, 100.0)

    val = extract_object_depth(depth_map, bbox_outside)
    assert val == pytest.approx(1.8, abs=0.01)


# ============================================================================
# 2. Metric Distance Calibration Tests
# ============================================================================
def test_convert_relative_to_metric():
    """Verify inverse-depth formula: Z = 1 / (scale_a * d_rel + shift_b)."""
    scale_a = 0.142
    shift_b = 0.005

    # Close object (high disparity d = 3.5) -> ~2.0 meters
    z_close = convert_relative_to_metric(3.5, scale_a, shift_b)
    expected_close = 1.0 / (0.142 * 3.5 + 0.005)  # 1 / 0.502 = 1.992 m
    assert z_close == pytest.approx(expected_close, abs=0.05)

    # Far object (low disparity d = 0.1) -> ~52.0 meters
    z_far = convert_relative_to_metric(0.1, scale_a, shift_b)
    expected_far = 1.0 / (0.142 * 0.1 + 0.005)  # 1 / 0.0192 = 52.08 m
    assert z_far == pytest.approx(expected_far, abs=0.5)

    # Monotonicity check: larger disparity must yield strictly closer distance
    assert z_close < z_far


# ============================================================================
# 3. Pinhole Geometry Tests
# ============================================================================
def test_compute_pinhole_distance():
    """Verify optical pinhole formula: Z = (f * H_real) / h_pixels."""
    f = CARLA_FOCAL_LENGTH_PX  # 640.0 px

    # 1. Car: H_real = 1.5m, h_pixels = 64px
    # Z = 640 * 1.5 / 64 = 15.0 meters
    bbox_car = (500.0, 300.0, 700.0, 364.0)
    z_car = compute_pinhole_distance("car", bbox_car, f_pixels=f)
    assert z_car == pytest.approx(15.0, abs=0.01)

    # 2. Pedestrian: H_real = 1.7m, h_pixels = 170px
    # Z = 640 * 1.7 / 170 = 6.4 meters
    bbox_ped = (400.0, 200.0, 440.0, 370.0)
    z_ped = compute_pinhole_distance("pedestrian", bbox_ped, f_pixels=f)
    assert z_ped == pytest.approx(6.4, abs=0.01)


# ============================================================================
# 4. Time-to-Collision (TTC) Tests
# ============================================================================
def test_ttc_tracker_approaching_vehicle():
    """Verify that closing-in vehicle triggers valid TTC and warning status."""
    tracker = TimeToCollisionTracker(smoothing_window=5, fps=30.0)

    # Object approaching at 5 m/s: at t=0 d=15.0m, at t=1s d=10.0m
    # In 30 FPS, frame step is 1/30 s -> delta d = 5 / 30 = 0.1667 m per frame
    dt = 1.0 / 30.0
    d = 15.0
    for frame_idx in range(10):
        t = frame_idx * dt
        ttc, status = tracker.update(track_id=1, distance_m=d, timestamp=t)
        d -= 5.0 * dt  # closing in

    # After 10 frames, velocity is ~ -5.0 m/s and distance is ~ 13.5 m -> TTC ~ 2.7s (WARNING)
    assert 1.5 <= ttc <= 3.5
    assert status == "WARNING"


def test_ttc_tracker_critical_alert():
    """Verify that rapid approach with TTC < 1.5s triggers CRITICAL collision alert."""
    tracker = TimeToCollisionTracker(smoothing_window=3, fps=30.0)

    # Object at 6.0 meters closing in rapidly at 8 m/s -> TTC = 6 / 8 = 0.75s (< 1.5s)
    tracker.update(track_id=2, distance_m=8.0, timestamp=0.0)
    tracker.update(track_id=2, distance_m=7.0, timestamp=0.125)
    ttc, status = tracker.update(track_id=2, distance_m=6.0, timestamp=0.25)

    assert ttc < 1.5
    assert status == "CRITICAL"


def test_ttc_tracker_receding_or_stable():
    """Verify that stationary or receding vehicles are flagged safe (TTC = 999.0)."""
    tracker = TimeToCollisionTracker(smoothing_window=3, fps=30.0)

    # Stationary vehicle at 20 meters
    tracker.update(track_id=3, distance_m=20.0, timestamp=0.0)
    ttc, status = tracker.update(track_id=3, distance_m=20.0, timestamp=0.1)
    assert ttc == pytest.approx(999.0)
    assert status == "STABLE"

    # Moving away (20m -> 22m)
    ttc_rec, status_rec = tracker.update(track_id=3, distance_m=22.0, timestamp=0.2)
    assert ttc_rec == pytest.approx(999.0)
    assert status_rec == "RECEDING"


# ============================================================================
# 5. Distance Band Error Calculation Tests
# ============================================================================
def test_calculate_band_errors():
    """Verify MAE and relative error calculation across 0-10m, 10-30m, and 30+m bands."""
    gts = [5.0, 8.0, 15.0, 25.0, 40.0, 60.0]
    # Predict with known offsets: +0.5m in near, +1.0m in mid, +2.0m in far
    preds = [5.5, 8.5, 16.0, 26.0, 42.0, 62.0]

    metrics = calculate_band_errors(preds, gts)

    # Band 1: 0-10m (errors: 0.5, 0.5 -> MAE 0.5m)
    assert metrics["0-10m"]["count"] == 2
    assert metrics["0-10m"]["mae_m"] == pytest.approx(0.5, abs=0.01)

    # Band 2: 10-30m (errors: 1.0, 1.0 -> MAE 1.0m)
    assert metrics["10-30m"]["count"] == 2
    assert metrics["10-30m"]["mae_m"] == pytest.approx(1.0, abs=0.01)

    # Band 3: 30+m (errors: 2.0, 2.0 -> MAE 2.0m)
    assert metrics["30+m"]["count"] == 2
    assert metrics["30+m"]["mae_m"] == pytest.approx(2.0, abs=0.01)

    # Overall MAE: mean(0.5, 0.5, 1.0, 1.0, 2.0, 2.0) = 7.0 / 6 = 1.167m
    assert metrics["Overall"]["mae_m"] == pytest.approx(7.0 / 6.0, abs=0.01)


# ============================================================================
# 6. CARLA 24-bit Depth Decoding Test
# ============================================================================
def test_carla_depth_pixel_math():
    """Verify 24-bit RGB CARLA depth buffer conversion to metric meters."""
    # Near object: d_norm = 0.01 -> 10.0 meters
    # (r + g*256 + b*65536) / (256^3 - 1) * 1000 = 10.0m -> raw = 167772
    raw = 167772
    b = (raw >> 16) & 0xFF
    g = (raw >> 8) & 0xFF
    r = raw & 0xFF
    pixel = np.array([b, g, r], dtype=np.uint8)

    meters = carla_depth_to_meters(pixel)
    assert meters == pytest.approx(10.0, abs=0.05)


# ============================================================================
# 7. DepthEstimator Contract Test
# ============================================================================
def test_depth_estimator_predict():
    """Verify DepthEstimator.predict(frame) returns valid 2D positive float32 depth map."""
    estimator = DepthEstimator(device="cpu")
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    depth_map = estimator.predict(frame)

    assert isinstance(depth_map, np.ndarray)
    assert depth_map.shape == (180, 320)
    assert depth_map.dtype == np.float32
    # Disparity values must be strictly positive
    assert np.all(depth_map > 0.0)
