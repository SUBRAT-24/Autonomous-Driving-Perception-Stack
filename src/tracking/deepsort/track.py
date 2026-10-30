"""Track state management for DeepSORT."""

from __future__ import annotations

from enum import IntEnum
from typing import List, Tuple

import numpy as np

from src.tracking.deepsort.kalman_filter import KalmanBoxTracker


class TrackState(IntEnum):
    """Lifecycle states of a track."""

    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class DeepTrack:
    """A single object track managed by DeepSORT."""

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: np.ndarray,
        cls_id: int = 0,
        conf: float = 1.0,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = TrackState.TENTATIVE
        self.features: List[np.ndarray] = [feature]
        self.cls_id = cls_id
        self.conf = conf

        self._n_init = n_init
        self._max_age = max_age

    def to_tlbr(self) -> Tuple[float, float, float, float]:
        """Convert current state [x, y, a, h] to bounding box [x1, y1, x2, y2]."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # w = a * h
        ret[:2] -= ret[2:] / 2  # x1 = x - w/2, y1 = y - h/2
        x1, y1 = ret[0], ret[1]
        x2, y2 = ret[0] + ret[2], ret[1] + ret[3]
        return float(x1), float(y1), float(x2), float(y2)

    def to_tlwh(self) -> Tuple[float, float, float, float]:
        """Convert current state [x, y, a, h] to [top_left_x, top_left_y, w, h]."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return float(ret[0]), float(ret[1]), float(ret[2]), float(ret[3])

    def predict(self, kf: KalmanBoxTracker) -> None:
        """Propagate state distribution using Kalman filter prediction."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanBoxTracker,
        measurement: np.ndarray,
        feature: np.ndarray,
        conf: float = 1.0,
        cls_id: int = 0,
    ) -> None:
        """Update track state with matched measurement and appearance feature."""
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)
        self.features.append(feature)
        self.conf = conf
        self.cls_id = cls_id

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """Mark track as missed in current frame."""
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
