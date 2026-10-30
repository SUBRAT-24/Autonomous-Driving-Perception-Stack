"""Deep Learning Lane Segmentation detector using trained U-Net.

Implements the unified ``LaneDetector`` interface:
    UNetLaneDetector.detect(frame) -> LaneResult(polylines, mask, left_poly, right_poly, curvature_m, offset_m)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from src.lane.detector import LaneDetector, LaneResult
from src.lane.geometry import (
    calculate_curvature,
    calculate_vehicle_offset,
    compute_perspective_matrices,
    evaluate_poly,
    fit_sliding_windows,
    warp_to_bev,
)
from src.lane.unet_model import get_lane_unet

logger = logging.getLogger(__name__)


class UNetLaneDetector(LaneDetector):
    """Deep learning lane segmentation detector using lightweight U-Net."""

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_size: Tuple[int, int] = (384, 640),  # (H, W)
        threshold: float = 0.5,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        self.threshold = threshold

        if weights_path is None:
            weights_path = Path("weights/lane_unet.pth")
        self.weights_path = Path(weights_path)

        self.model = get_lane_unet(in_channels=3, out_channels=1, device=device)

        if self.weights_path.exists():
            logger.info("Loading U-Net lane model weights from %s", self.weights_path)
            state_dict = torch.load(
                str(self.weights_path), map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
        else:
            logger.warning(
                "No checkpoint found at %s; using initialized weights",
                self.weights_path,
            )

        self.model.eval()

        # Camera perspective matrices for polynomial curvature & offset fitting
        self.M, self.Minv = compute_perspective_matrices()

    def _preprocess(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Resize and normalize BGR frame for neural network input."""
        orig_h, orig_w = frame.shape[:2]
        h_in, w_in = self.input_size

        resized = cv2.resize(frame, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (rgb - mean) / std

        tensor = (
            torch.from_numpy(norm.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        return tensor, (orig_h, orig_w)

    def detect(self, frame: np.ndarray) -> LaneResult:
        """Detect lane markings and ego-lane polylines from input frame."""
        tensor, (orig_h, orig_w) = self._preprocess(frame)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Threshold and resize back to original camera resolution
        prob_resized = cv2.resize(
            probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        binary_mask = (prob_resized >= self.threshold).astype(np.uint8) * 255

        # Extract lane polylines via BEV projection & polynomial fitting
        binary_warped = warp_to_bev(binary_mask, self.M, (orig_w, orig_h))
        left_fit, right_fit, _ = fit_sliding_windows(
            binary_warped, nwindows=9, margin=70, minpix=30
        )

        polylines: List[np.ndarray] = []
        ploty = np.linspace(0, orig_h - 1, orig_h)

        if left_fit is not None:
            left_fitx = evaluate_poly(left_fit, ploty)
            bev_pts = np.vstack([left_fitx, ploty]).T.astype(np.float32)
            cam_pts = cv2.perspectiveTransform(
                bev_pts.reshape(-1, 1, 2), self.Minv
            ).reshape(-1, 2)
            polylines.append(cam_pts)

        if right_fit is not None:
            right_fitx = evaluate_poly(right_fit, ploty)
            bev_pts = np.vstack([right_fitx, ploty]).T.astype(np.float32)
            cam_pts = cv2.perspectiveTransform(
                bev_pts.reshape(-1, 1, 2), self.Minv
            ).reshape(-1, 2)
            polylines.append(cam_pts)

        # Fallback to contour extraction if polynomial fitting had too few pixels
        if not polylines:
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if cv2.contourArea(cnt) > 200:
                    polylines.append(cnt.reshape(-1, 2))

        # Curvature and lateral offset
        curv_left = (
            calculate_curvature(left_fit, y_eval=float(orig_h))
            if left_fit is not None
            else 1e4
        )
        curv_right = (
            calculate_curvature(right_fit, y_eval=float(orig_h))
            if right_fit is not None
            else 1e4
        )
        curvature = float(np.mean([curv_left, curv_right]))

        offset = calculate_vehicle_offset(
            left_fit, right_fit, y_eval=float(orig_h), img_width=orig_w
        )

        conf = (
            0.95
            if (left_fit is not None and right_fit is not None)
            else (0.6 if len(polylines) > 0 else 0.0)
        )

        return LaneResult(
            polylines=polylines,
            mask=binary_mask,
            left_poly=left_fit,
            right_poly=right_fit,
            curvature_m=curvature,
            offset_m=offset,
            confidence=conf,
        )
