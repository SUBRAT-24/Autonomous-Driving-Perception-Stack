"""Unit tests for multi-object tracking glue, algorithms, and interfaces.

Tests:
1. Track dataclass integrity (attributes, properties: center, width, height).
2. DeepSORT Kalman Filter (predict, update, state dimension 8x1, cov 8x8).
3. DeepSORT Re-ID Feature Extractor (128-d normalized embeddings).
4. DeepTrack state transitions (Tentative, Confirmed, Deleted).
5. DeepSORTTracker multi-frame association and tracking.
6. Unified Tracker wrapper (Tracker.update(frame) -> List[Track]).
7. Visualization helpers (color mapping, motion trail rendering, HUD).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tracking.deepsort.kalman_filter import KalmanBoxTracker
from src.tracking.deepsort.reid_model import AppearanceExtractor
from src.tracking.deepsort.track import DeepTrack
from src.tracking.deepsort.tracker import DeepSORTTracker
from src.tracking.tracker import Track, Tracker
from src.tracking.visualize import (
    draw_hud,
    draw_motion_trail,
    draw_track_box,
    get_color_for_id,
)


# ============================================================================
# 1. Track Dataclass Tests
# ============================================================================
def test_track_dataclass_properties():
    """Verify Track dataclass fields and computed geometric properties."""
    bbox = (100.0, 150.0, 300.0, 450.0)
    track = Track(id=1, cls="car", cls_id=0, bbox=bbox, conf=0.88)

    assert track.id == 1
    assert track.cls == "car"
    assert track.cls_id == 0
    assert track.conf == pytest.approx(0.88)
    assert track.bbox == bbox

    # Center computation
    cx, cy = track.center
    assert cx == pytest.approx(200.0)
    assert cy == pytest.approx(300.0)

    # Dimensions
    assert track.width == pytest.approx(200.0)
    assert track.height == pytest.approx(300.0)


# ============================================================================
# 2. Kalman Filter Tests
# ============================================================================
def test_kalman_filter_initialization_and_step():
    """Verify 8D Kalman filter initialization, prediction, and measurement update."""
    kf = KalmanBoxTracker()
    measurement = np.array([100.0, 200.0, 1.5, 80.0], dtype=np.float32)

    mean, covariance = kf.initiate(measurement)
    assert mean.shape == (8,)
    assert covariance.shape == (8, 8)
    assert np.allclose(mean[:4], measurement)
    assert np.allclose(mean[4:], 0.0)  # zero initial velocities

    # Predict step
    mean_pred, cov_pred = kf.predict(mean, covariance)
    assert mean_pred.shape == (8,)
    assert cov_pred.shape == (8, 8)

    # Update step with a slightly shifted measurement
    new_measurement = np.array([105.0, 204.0, 1.5, 80.0], dtype=np.float32)
    mean_upd, cov_upd = kf.update(mean_pred, cov_pred, new_measurement)
    assert mean_upd.shape == (8,)
    assert cov_upd.shape == (8, 8)
    # Velocity x and y should now be positive
    assert mean_upd[4] > 0
    assert mean_upd[5] > 0


def test_kalman_filter_gating_distance():
    """Verify Mahalanobis distance calculation against measurement vector."""
    kf = KalmanBoxTracker()
    measurement = np.array([50.0, 50.0, 1.0, 40.0], dtype=np.float32)
    mean, covariance = kf.initiate(measurement)

    # Closely matching measurement should have small distance
    close_meas = np.array([[51.0, 50.0, 1.0, 40.0]], dtype=np.float32)
    dist = kf.gating_distance(mean, covariance, close_meas)
    assert dist[0] < 5.0

    # Distant measurement should have large distance
    far_meas = np.array([[500.0, 800.0, 5.0, 200.0]], dtype=np.float32)
    dist_far = kf.gating_distance(mean, covariance, far_meas)
    assert dist_far[0] > dist[0]


# ============================================================================
# 3. DeepSORT Re-ID Feature Extractor Tests
# ============================================================================
def test_reid_feature_extractor():
    """Verify CNN feature extractor produces normalized 128D embeddings."""
    fe = AppearanceExtractor(device="cpu", feature_dim=128)
    # Synthetic frame and crops
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    boxes = np.array(
        [
            [50.0, 50.0, 150.0, 200.0],
            [200.0, 100.0, 350.0, 300.0],
        ],
        dtype=np.float32,
    )

    features = fe.extract(frame, boxes)
    assert features.shape == (2, 128)
    # Check L2 unit normalization
    norms = np.linalg.norm(features, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)

    # Empty boxes case
    empty_features = fe.extract(frame, np.empty((0, 4), dtype=np.float32))
    assert empty_features.shape == (0, 128)


# ============================================================================
# 4. DeepTrack State Machine Tests
# ============================================================================
def test_deeptrack_lifecycle():
    """Verify DeepTrack state transitions: Tentative -> Confirmed -> Deleted."""
    kf = KalmanBoxTracker()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 1.0, 50.0], dtype=np.float32))
    feature = np.random.randn(128).astype(np.float32)
    feature /= np.linalg.norm(feature)

    track = DeepTrack(
        mean=mean,
        covariance=cov,
        track_id=1,
        n_init=3,
        max_age=5,
        feature=feature,
        cls_id=0,
        conf=0.9,
    )

    # Initially tentative
    assert track.is_tentative()
    assert not track.is_confirmed()
    assert not track.is_deleted()

    # Hit 2 (still tentative since n_init=3)
    track.predict(kf)
    track.update(
        kf,
        np.array([102.0, 102.0, 1.0, 50.0], dtype=np.float32),
        feature,
        conf=0.9,
        cls_id=0,
    )
    assert track.is_tentative()

    # Hit 3 (now confirmed)
    track.predict(kf)
    track.update(
        kf,
        np.array([104.0, 104.0, 1.0, 50.0], dtype=np.float32),
        feature,
        conf=0.9,
        cls_id=0,
    )
    assert track.is_confirmed()
    assert not track.is_tentative()

    # Miss consecutive frames until max_age exceeded
    for _ in range(6):
        track.predict(kf)
        track.mark_missed()

    assert track.is_deleted()


# ============================================================================
# 5. DeepSORTTracker Association & Tracking Tests
# ============================================================================
def test_deepsort_tracker_tracking():
    """Verify DeepSORTTracker tracks objects across multiple synthetic frames."""
    tracker = DeepSORTTracker(max_dist=0.3, min_confidence=0.3, max_age=5, n_init=2)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: 2 detections [x1, y1, x2, y2, conf, cls_id]
    dets1 = np.array(
        [
            [100.0, 100.0, 160.0, 220.0, 0.85, 0],
            [300.0, 200.0, 380.0, 320.0, 0.90, 3],
        ],
        dtype=np.float32,
    )
    tracks1 = tracker.update(frame, dets1)
    # Both tracks are tentative at first frame
    assert len(tracks1) == 0

    # Frame 2: moving detections
    dets2 = np.array(
        [
            [105.0, 102.0, 165.0, 222.0, 0.88, 0],
            [305.0, 202.0, 385.0, 322.0, 0.92, 3],
        ],
        dtype=np.float32,
    )
    tracks2 = tracker.update(frame, dets2)
    # With n_init=2, both tracks should be confirmed
    assert len(tracks2) == 2
    ids = {t.track_id for t in tracks2}
    assert len(ids) == 2

    # Frame 3: verify ID persistence
    dets3 = np.array(
        [
            [110.0, 104.0, 170.0, 224.0, 0.87, 0],
            [310.0, 204.0, 390.0, 324.0, 0.91, 3],
        ],
        dtype=np.float32,
    )
    tracks3 = tracker.update(frame, dets3)
    assert len(tracks3) == 2
    ids3 = {t.track_id for t in tracks3}
    assert ids == ids3, "Track IDs must persist across frames"


# ============================================================================
# 6. Unified Tracker Wrapper Tests
# ============================================================================
def test_unified_tracker_interface():
    """Verify Tracker unified wrapper matches ROS2 node integration specification:
    Tracker.update(frame) -> List[Track(id, cls, bbox, conf)]
    """
    weights_path = Path("runs/detect/bdd100k_yolov8s/weights/best.pt")
    if not weights_path.exists():
        weights_path = Path("yolov8s.pt")

    # Initialize ByteTrack backend
    tracker = Tracker(
        tracker_type="bytetrack",
        weights_path=weights_path,
        conf_thresh=0.1,
    )
    assert tracker.tracker_type == "bytetrack"

    # Synthetic frame with a drawn rectangle
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)

    tracks = tracker.update(frame)
    assert isinstance(tracks, list)
    for t in tracks:
        assert isinstance(t, Track)
        assert hasattr(t, "id")
        assert hasattr(t, "cls")
        assert hasattr(t, "bbox")
        assert hasattr(t, "conf")
        assert len(t.bbox) == 4

    # Reset test
    tracker.reset()
    assert tracker._frame_count == 0


def test_unified_tracker_deepsort_backend():
    """Verify Tracker unified wrapper with DeepSORT backend."""
    weights_path = Path("runs/detect/bdd100k_yolov8s/weights/best.pt")
    if not weights_path.exists():
        weights_path = Path("yolov8s.pt")

    tracker = Tracker(
        tracker_type="deepsort",
        weights_path=weights_path,
        conf_thresh=0.1,
    )
    assert tracker.tracker_type == "deepsort"

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    tracks = tracker.update(frame)
    assert isinstance(tracks, list)


# ============================================================================
# 7. Visualization Helpers Tests
# ============================================================================
def test_visualization_helpers():
    """Verify color hashing, motion trails, track box, and HUD rendering."""
    # Color mapping
    c1 = get_color_for_id(1)
    c2 = get_color_for_id(2)
    assert len(c1) == 3 and all(0 <= v <= 255 for v in c1)
    assert len(c2) == 3 and all(0 <= v <= 255 for v in c2)
    assert c1 != c2  # Distinct colors

    # Motion trail
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)
    trail = deque([(50, 50), (60, 60), (70, 70), (80, 80)], maxlen=20)
    draw_motion_trail(canvas, trail, (0, 255, 0))
    # Should draw non-zero pixels
    assert np.count_nonzero(canvas) > 0

    # Track box rendering
    track = Track(id=1, cls="car", cls_id=0, bbox=(20.0, 20.0, 100.0, 100.0), conf=0.95)
    draw_track_box(canvas, track, (255, 0, 0))

    # HUD rendering
    draw_hud(
        canvas, "bytetrack", frame_idx=10, total_frames=90, active_count=3, fps=45.2
    )
