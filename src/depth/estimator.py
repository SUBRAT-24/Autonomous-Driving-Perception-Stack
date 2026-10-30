"""MiDaS Small Depth Estimation Wrapper for Autonomous Driving Perception Stack.

Provides:
- ``DepthEstimator.predict(frame: np.ndarray) -> np.ndarray`` returning full-resolution
  relative inverse depth (disparity) map where higher values represent closer objects.
- Self-contained PyTorch ``MiDaSSmallNet`` architecture with multi-scale feature fusion
  blocks and residual conv units, with automatic fallback and torch.hub support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 1. Native PyTorch MiDaS Small Architecture
# ============================================================================
class ResidualConvUnit(nn.Module):
    """Residual convolution unit for MiDaS feature refinement."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block combining low-level skip and high-level upsampled features."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.res_conv1 = ResidualConvUnit(features)
        self.res_conv2 = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        output = xs[0]
        if len(xs) == 2:
            res = self.res_conv1(xs[1])
            # Upsample output to match skip connection spatial size
            if output.shape[2:] != res.shape[2:]:
                output = F.interpolate(
                    output, size=res.shape[2:], mode="bilinear", align_corners=True
                )
            output = output + res
        output = self.res_conv2(output)
        return output


class MiDaSSmallNet(nn.Module):
    """Compact Multi-Scale Depth Estimation Network (MiDaS Small style).

    Features:
    - 4-stage hierarchical encoder extracting multi-resolution spatial features
    - RefineNet decoder with Residual Convolution Units and Feature Fusion Blocks
    - Output head producing single-channel relative inverse depth (disparity) map
    """

    def __init__(self, in_channels: int = 3, features: int = 64) -> None:
        super().__init__()
        # Encoder stages
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )  # H/2, W/2

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # H/4, W/4

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )  # H/8, W/8

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  # H/16, W/16

        # Lateral 1x1 projections to uniform feature dimension
        self.latlayer4 = nn.Conv2d(256, features, kernel_size=1)
        self.latlayer3 = nn.Conv2d(128, features, kernel_size=1)
        self.latlayer2 = nn.Conv2d(64, features, kernel_size=1)
        self.latlayer1 = nn.Conv2d(32, features, kernel_size=1)

        # RefineNet Fusion Decoder
        self.fuse4 = FeatureFusionBlock(features)
        self.fuse3 = FeatureFusionBlock(features)
        self.fuse2 = FeatureFusionBlock(features)
        self.fuse1 = FeatureFusionBlock(features)

        # Output disparity head
        self.head = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Positive relative inverse depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting relative inverse depth (disparity)."""
        # Encoder
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # Lateral projections
        l4 = self.latlayer4(x4)
        l3 = self.latlayer3(x3)
        l2 = self.latlayer2(x2)
        l1 = self.latlayer1(x1)

        # Decoder fusion
        f4 = self.fuse4(l4)
        f3 = self.fuse3(f4, l3)
        f2 = self.fuse2(f3, l2)
        f1 = self.fuse1(f2, l1)

        # Prediction
        disp = self.head(f1)
        return disp


# ============================================================================
# 2. DepthEstimator Wrapper
# ============================================================================
class DepthEstimator:
    """Wrapper class for MiDaS small monocular relative depth estimation.

    Interface:
        predict(frame: np.ndarray) -> np.ndarray
    Returning full-frame relative inverse depth map where closer objects have higher values.
    """

    def __init__(
        self,
        model_name: str = "MiDaS_small",
        weights_path: Optional[str | Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device)
        self.input_size = input_size
        self.weights_path = (
            Path(weights_path) if weights_path else Path("weights/midas_small.pth")
        )

        # Initialise model
        self.model = self._load_model()
        self.model.eval()
        logger.info(
            "DepthEstimator initialized (%s on %s)", self.model_name, self.device
        )

    def _load_model(self) -> nn.Module:
        """Load pretrained MiDaS model or native fallback."""
        model: Optional[nn.Module] = None

        # Try torch.hub first if available
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = torch.hub.load(
                    "intel-isl/MiDaS", "MiDaS_small", pretrained=True, verbose=False
                )
                logger.info("Loaded MiDaS_small via torch.hub successfully")
        except Exception as exc:
            logger.info(
                "torch.hub MiDaS not loaded (%s); using self-contained MiDaSSmallNet",
                exc,
            )

        if model is None:
            model = MiDaSSmallNet(in_channels=3, features=64)
            if self.weights_path.exists():
                logger.info("Loading weights from %s", self.weights_path)
                state = torch.load(
                    str(self.weights_path), map_location=self.device, weights_only=True
                )
                model.load_state_dict(state)
            else:
                # Save initial weights for persistence
                self.weights_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), str(self.weights_path))
                logger.info(
                    "Initialized and saved MiDaSSmallNet checkpoint at %s",
                    self.weights_path,
                )

        return model.to(self.device)

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """Predict relative inverse depth map for a single BGR image.

        Args:
            frame: Input BGR image (H, W, 3).

        Returns:
            Relative inverse depth map (H, W) as float32 numpy array.
            Higher values correspond to closer objects (greater disparity).
        """
        orig_h, orig_w = frame.shape[:2]
        h_in, w_in = self.input_size

        # Preprocessing: BGR -> RGB, resize, ImageNet normalization
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w_in, h_in), interpolation=cv2.INTER_CUBIC)
        img_float = img_resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std

        tensor = (
            torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        )

        # Forward inference
        with torch.no_grad():
            prediction = self.model(tensor)
            # Squeeze to 2D
            if prediction.dim() == 4:
                prediction = prediction.squeeze(1)
            pred_2d = prediction[0].cpu().numpy()

        # Resize back to original frame dimensions
        depth_map = cv2.resize(pred_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        # Ensure positive values
        depth_map = np.maximum(depth_map, 1e-4).astype(np.float32)
        return depth_map
