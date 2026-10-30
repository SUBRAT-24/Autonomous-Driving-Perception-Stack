"""Lightweight Appearance Feature Extractor (Re-ID model) for DeepSORT.

Extracts normalized 128-dimensional appearance embeddings from cropped
bounding box image patches.
"""

from __future__ import annotations


import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleReIDNet(nn.Module):
    """Compact CNN for vehicle and pedestrian appearance feature extraction."""

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x32 -> 32x16

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32x16 -> 16x8

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting L2-normalized feature vectors."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        features = self.fc(x)
        # Normalize to unit sphere (L2 normalization for cosine similarity)
        return F.normalize(features, p=2, dim=1)


class AppearanceExtractor:
    """Extracts normalized appearance embeddings for bounding box crops."""

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        input_size: tuple[int, int] = (64, 32),  # (H, W)
        feature_dim: int = 128,
    ) -> None:
        self.device = torch.device(device)
        self.input_size = input_size
        self.feature_dim = feature_dim

        # Initialise deterministic network
        torch.manual_seed(42)
        self.model = SimpleReIDNet(feature_dim=feature_dim).to(self.device)
        self.model.eval()

    def extract(self, frame: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Extract appearance embeddings for an array of bounding boxes.

        Args:
            frame: Full BGR image (H, W, 3).
            bboxes: Array of bounding boxes in [x1, y1, x2, y2] format (N, 4).

        Returns:
            Normalized feature matrix (N, 128).
        """
        if len(bboxes) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        crops = []
        h_img, w_img = frame.shape[:2]

        for bbox in bboxes:
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w_img, int(bbox[2]))
            y2 = min(h_img, int(bbox[3]))

            if x2 <= x1 or y2 <= y1:
                crop = np.zeros(
                    (self.input_size[0], self.input_size[1], 3), dtype=np.uint8
                )
            else:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))

            # Normalise and convert to tensor (B, C, H, W) in RGB
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            crop = np.transpose(crop, (2, 0, 1))
            crops.append(crop)

        batch_tensor = torch.tensor(
            np.array(crops), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            embeddings = self.model(batch_tensor).cpu().numpy()

        return embeddings
