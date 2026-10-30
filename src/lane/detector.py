"""Unified Lane Detection interface and result dataclass.

Defines the contract:
    LaneDetector.detect(frame: np.ndarray) -> LaneResult
Used identically by Classical (Hough, Sliding Window BEV) and Deep Learning (U-Net).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class LaneResult:
    """Standardized lane detection result.

    Attributes:
        polylines: List of detected lane polylines. Each is an (N, 2) array of [x, y] coordinates.
        mask: Binary uint8 mask (H, W) where 255 = lane markings, 0 = background.
        left_poly: Optional 2nd-order polynomial coefficients [A, B, C] for left ego-lane ($x = Ay^2 + By + C$).
        right_poly: Optional 2nd-order polynomial coefficients [A, B, C] for right ego-lane.
        curvature_m: Estimated radius of road curvature in meters ($R \\to \\infty$ for straight road).
        offset_m: Lateral vehicle offset from lane center in meters (+ = shifted right, - = shifted left).
        confidence: Detection confidence score in [0.0, 1.0].
    """

    polylines: List[np.ndarray] = field(default_factory=list)
    mask: np.ndarray = field(
        default_factory=lambda: np.zeros((720, 1280), dtype=np.uint8)
    )
    left_poly: Optional[np.ndarray] = None
    right_poly: Optional[np.ndarray] = None
    curvature_m: float = 0.0
    offset_m: float = 0.0
    confidence: float = 1.0

    @property
    def has_ego_lanes(self) -> bool:
        """Whether both left and right ego-lanes are detected."""
        return self.left_poly is not None and self.right_poly is not None


class LaneDetector(ABC):
    """Abstract base class for all lane detection models and algorithms."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lane markings in a BGR image frame.

        Args:
            frame: Input image array (H, W, 3) in BGR format.

        Returns:
            A :class:`LaneResult` instance containing polylines and binary lane mask.
        """
        pass
