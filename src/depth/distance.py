"""Object Distance Extraction, Metric Calibration, and Time-to-Collision (TTC).

Provides:
1. ``extract_object_depth``: Extracts median relative depth over the lower third of a bounding box.
2. ``convert_relative_to_metric``: Calibrated conversion of MiDaS disparity to true metres.
3. ``compute_pinhole_distance``: Optical geometry cross-check (Z = f * H_real / h_pixels).
4. ``TimeToCollisionTracker``: Moving-average smoothed TTC tracking per object track ID.
5. ``calculate_band_errors``: MAE and relative percentage error across 0-10m, 10-30m, 30+m bands.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)

# Standard CARLA camera intrinsics:
# Horizontal FOV = 90 degrees on 1280x720 image:
# f = (width / 2) / tan(FOV / 2) = 640 / tan(45) = 640.0 pixels
CARLA_FOCAL_LENGTH_PX = 640.0

# Real-world object heights in metres
REFERENCE_OBJECT_HEIGHTS: Dict[str, float] = {
    "car": 1.5,
    "truck": 2.5,
    "bus": 3.0,
    "pedestrian": 1.7,
    "person": 1.7,
    "rider": 1.7,
    "bike": 1.4,
    "bicycle": 1.4,
    "motorcycle": 1.4,
    "traffic light": 1.0,
    "traffic sign": 0.8,
}


def extract_object_depth(
    relative_depth_map: np.ndarray,
    bbox: Tuple[float, float, float, float],
    lower_third_only: bool = True,
    inset_pct: float = 0.08,
) -> float:
    """Extract median relative depth from a bounding box.

    Taking the median over the lower third focuses on the vehicle/pedestrian
    body and road contact patch, eliminating sky bleed and window transparency.

    Args:
        relative_depth_map: 2D float array (H, W) from DepthEstimator.
        bbox: Bounding box tuple (x1, y1, x2, y2).
        lower_third_only: If True, crops to bottom 33% of the box.
        inset_pct: Inset percentage to exclude surrounding background edge pixels.

    Returns:
        Median relative depth value (float).
    """
    h_img, w_img = relative_depth_map.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clamp to valid image space
    x1 = max(0.0, min(float(w_img - 1), x1))
    x2 = max(0.0, min(float(w_img - 1), x2))
    y1 = max(0.0, min(float(h_img - 1), y1))
    y2 = max(0.0, min(float(h_img - 1), y2))

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Inset margins to avoid background edge pixels
    x_start = int(x1 + inset_pct * bw)
    x_end = max(x_start + 1, int(x2 - inset_pct * bw))

    if lower_third_only:
        y_start = int(y1 + (2.0 / 3.0) * bh)
        y_end = max(y_start + 1, int(y2))
    else:
        y_start = int(y1 + inset_pct * bh)
        y_end = max(y_start + 1, int(y2 - inset_pct * bh))

    x_start = max(0, min(w_img - 1, x_start))
    x_end = max(x_start + 1, min(w_img, x_end))
    y_start = max(0, min(h_img - 1, y_start))
    y_end = max(y_start + 1, min(h_img, y_end))

    patch = relative_depth_map[y_start:y_end, x_start:x_end]
    if patch.size == 0:
        return 0.1

    median_val = float(np.median(patch))
    return median_val


def convert_relative_to_metric(
    rel_depth: float,
    scale_a: float = 0.142228,
    shift_b: float = 0.004895,
) -> float:
    """Convert MiDaS relative disparity to metric distance in metres.

    Inverse-depth linear formulation:
        Z = 1.0 / (scale_a * d_rel + shift_b)

    Args:
        rel_depth: Median relative depth value.
        scale_a: Calibrated linear scale factor.
        shift_b: Calibrated linear shift factor.

    Returns:
        Metric distance in metres (clamped to realistic driving range 0.5m to 120m).
    """
    denominator = scale_a * max(0.001, rel_depth) + shift_b
    if denominator <= 1e-4:
        return 100.0

    z_m = 1.0 / denominator
    return float(max(0.5, min(120.0, z_m)))


def compute_pinhole_distance(
    cls_name: str,
    bbox: Tuple[float, float, float, float],
    f_pixels: float = CARLA_FOCAL_LENGTH_PX,
) -> float:
    """Calculate distance in metres using the geometric pinhole camera formula.

    Formula:
        Z = (f * H_real) / h_pixels

    Args:
        cls_name: Object class (e.g. 'car', 'pedestrian').
        bbox: Bounding box tuple (x1, y1, x2, y2).
        f_pixels: Camera focal length in pixels.

    Returns:
        Estimated distance in metres.
    """
    h_real = REFERENCE_OBJECT_HEIGHTS.get(cls_name.lower(), 1.5)
    h_pixels = max(1.0, float(bbox[3] - bbox[1]))

    z_pinhole = (f_pixels * h_real) / h_pixels
    return float(max(0.5, min(150.0, z_pinhole)))


class TimeToCollisionTracker:
    """Tracks distance time-series per object ID and computes smoothed TTC.

    TTC Formulation:
        TTC = Z / |dZ/dt|  (when dZ/dt < 0, object is closing in)
    """

    def __init__(self, smoothing_window: int = 5, fps: float = 30.0) -> None:
        self.smoothing_window = smoothing_window
        self.fps = fps
        self.dt = 1.0 / max(1.0, fps)
        # track_id -> deque of (timestamp, distance_m)
        self.history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=self.smoothing_window)
        )
        self._frame_idx = 0

    def update(
        self,
        track_id: int,
        distance_m: float,
        timestamp: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Update object distance and calculate smoothed Time-to-Collision.

        Args:
            track_id: Object tracking ID.
            distance_m: Calibrated metric distance in metres.
            timestamp: Optional wall-clock timestamp in seconds.

        Returns:
            Tuple of (ttc_seconds, warning_status):
            - ttc_seconds: Estimated seconds until impact (999.0 if safe/receding).
            - warning_status: 'CRITICAL', 'WARNING', 'APPROACHING', 'RECEDING', or 'STABLE'.
        """
        if timestamp is None:
            self._frame_idx += 1
            timestamp = self._frame_idx * self.dt

        q = self.history[track_id]
        q.append((timestamp, distance_m))

        if len(q) < 2:
            return 999.0, "INITIALIZING"

        # Moving average smoothed distance
        recent_dists = [d for _, d in q]
        smoothed_dist = float(np.mean(recent_dists))

        # Velocity estimation (finite difference between endpoints of the window)
        t_first, d_first = q[0]
        t_last, d_last = q[-1]
        delta_t = max(1e-3, t_last - t_first)
        velocity = (d_last - d_first) / delta_t  # m/s (negative = closing in)

        # Closing in at >= 0.2 m/s
        if velocity < -0.2:
            ttc = smoothed_dist / abs(velocity)
            if ttc < 1.5:
                status = "CRITICAL"
            elif ttc < 3.0:
                status = "WARNING"
            else:
                status = "APPROACHING"
            return float(ttc), status
        elif velocity > 0.2:
            return 999.0, "RECEDING"
        else:
            return 999.0, "STABLE"

    def reset(self) -> None:
        """Clear all tracking history."""
        self.history.clear()
        self._frame_idx = 0


def calculate_band_errors(
    pred_distances: List[float],
    gt_distances: List[float],
) -> Dict[str, Dict[str, float]]:
    """Compute MAE, Relative Error (%), and RMSE across standard distance bands:
    - Band 1: 0 to 10 m (Near / critical)
    - Band 2: 10 to 30 m (Mid range)
    - Band 3: 30+ m (Far range)
    - Overall: All ranges combined
    """
    assert len(pred_distances) == len(gt_distances), "Length mismatch"

    preds = np.array(pred_distances, dtype=np.float64)
    gts = np.array(gt_distances, dtype=np.float64)

    bands = {
        "0-10m": (gts >= 0.0) & (gts < 10.0),
        "10-30m": (gts >= 10.0) & (gts < 30.0),
        "30+m": (gts >= 30.0),
        "Overall": np.ones_like(gts, dtype=bool),
    }

    results: Dict[str, Dict[str, float]] = {}

    for b_name, mask in bands.items():
        if np.sum(mask) == 0:
            results[b_name] = {
                "count": 0,
                "mae_m": 0.0,
                "rel_error_pct": 0.0,
                "rmse_m": 0.0,
            }
            continue

        p = preds[mask]
        g = gts[mask]

        errors = np.abs(p - g)
        mae = float(np.mean(errors))
        rel_err = float(np.mean(errors / np.maximum(0.5, g)) * 100.0)
        rmse = float(np.sqrt(np.mean(errors**2)))

        results[b_name] = {
            "count": int(np.sum(mask)),
            "mae_m": mae,
            "rel_error_pct": rel_err,
            "rmse_m": rmse,
        }

    return results
