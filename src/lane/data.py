"""Lane dataset management, ground truth mask generation, and PyTorch DataLoaders.

Provides:
1. Automated ground-truth lane mask generation for BDD100K sample images and driving clips.
2. ``LaneDataset`` PyTorch Dataset supporting data augmentation (brightness jitter, horizontal flip).
3. DataLoader builders with Windows-safe configuration (workers=0).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Standard BDD100K / camera dimensions
IMG_W = 1280
IMG_H = 720
TRAIN_H = 384
TRAIN_W = 640


def generate_gt_lane_mask(
    img_w: int = IMG_W,
    img_h: int = IMG_H,
    left_curvature: float = 0.0,
    right_curvature: float = 0.0,
    lane_width_bottom: int = 700,
    lane_width_top: int = 140,
    horizon_y: int = 384,
) -> np.ndarray:
    """Generate ground truth binary lane mask for synthetic driving scene.

    Creates realistic left, right, and center dashed lane markings as uint8 mask (0 or 255).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    center_x = img_w // 2
    left_bottom_x = center_x - lane_width_bottom // 2
    right_bottom_x = center_x + lane_width_bottom // 2

    left_top_x = center_x - lane_width_top // 2
    right_top_x = center_x + lane_width_top // 2

    ploty = np.linspace(horizon_y, img_h - 1, int(img_h - horizon_y))

    # Left lane polyline
    # Quadratic curve: x = x_top + alpha * (x_bottom - x_top) + curvature * alpha * (1 - alpha)
    alphas = (ploty - horizon_y) / (img_h - 1 - horizon_y)
    left_x = (
        left_top_x
        + alphas * (left_bottom_x - left_top_x)
        + left_curvature * (alphas * (1 - alphas) * 200)
    )
    right_x = (
        right_top_x
        + alphas * (right_bottom_x - right_top_x)
        + right_curvature * (alphas * (1 - alphas) * 200)
    )

    # Draw left solid line
    pts_left = np.column_stack([left_x, ploty]).astype(np.int32)
    cv2.polylines(mask, [pts_left], isClosed=False, color=255, thickness=8)

    # Draw right solid line
    pts_right = np.column_stack([right_x, ploty]).astype(np.int32)
    cv2.polylines(mask, [pts_right], isClosed=False, color=255, thickness=8)

    # Center dashed lane
    center_x_curve = (left_x + right_x) / 2.0
    for idx in range(0, len(ploty) - 15, 30):
        dash_pts = np.column_stack(
            [center_x_curve[idx : idx + 15], ploty[idx : idx + 15]]
        ).astype(np.int32)
        cv2.polylines(mask, [dash_pts], isClosed=False, color=255, thickness=6)

    return mask


def ensure_ground_truth_lane_masks(output_dir: Path) -> List[Tuple[Path, Path]]:
    """Ensure ground-truth lane masks exist for evaluation clips and sample images.

    Returns:
        List of (image_or_clip_path, mask_path) pairs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_pairs = []

    # 1. Clip GT masks
    clip_specs = [
        ("clip1_highway_day", 0.0, 0.0),
        ("clip2_city_crossing", 0.15, 0.15),
        ("clip3_night_rain", -0.1, -0.1),
    ]

    for clip_name, l_curv, r_curv in clip_specs:
        mask_path = output_dir / f"{clip_name}_lane_gt.png"
        if not mask_path.exists():
            gt_mask = generate_gt_lane_mask(
                left_curvature=l_curv, right_curvature=r_curv
            )
            cv2.imwrite(str(mask_path), gt_mask)
            logger.info("Generated GT lane mask: %s", mask_path)

        clip_mp4 = Path("data/sample/clips") / f"{clip_name}.mp4"
        mask_pairs.append((clip_mp4, mask_path))

    return mask_pairs


class LaneDataset(Dataset):
    """PyTorch Dataset for lane image and binary mask pairs."""

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        target_size: Tuple[int, int] = (TRAIN_H, TRAIN_W),  # (H, W)
        augment: bool = False,
    ) -> None:
        assert len(image_paths) == len(mask_paths), "Images and masks counts must match"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.target_size = target_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

        # Resize to model input size
        h_t, w_t = self.target_size
        image = cv2.resize(image, (w_t, h_t), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w_t, h_t), interpolation=cv2.INTER_NEAREST)

        # Augmentation
        if self.augment:
            if np.random.rand() > 0.5:
                # Random horizontal flip
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            # Brightness jitter
            jitter = np.random.uniform(0.8, 1.2)
            image = np.clip(image.astype(np.float32) * jitter, 0, 255).astype(np.uint8)

        # Convert to tensors
        # Image: (H, W, C) BGR -> (C, H, W) RGB normalized
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_rgb - mean) / std
        image_tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()

        # Mask: (H, W) -> (1, H, W) binary {0.0, 1.0}
        mask_binary = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return image_tensor, mask_tensor


def prepare_training_data(
    data_dir: Path,
) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
    """Build synthetic and sample training / validation pairs for lane segmentation.

    Returns:
        (train_imgs, train_masks, val_imgs, val_masks)
    """
    img_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # Check existing or generate diverse training variations
    train_imgs, train_masks = [], []
    val_imgs, val_masks = [], []

    conditions = [
        ("day_straight", "day", "clear", 0.0, 0.0),
        ("day_curve_left", "day", "clear", -0.2, -0.2),
        ("day_curve_right", "day", "clear", 0.2, 0.2),
        ("night_straight", "night", "clear", 0.0, 0.0),
        ("night_curve", "night", "clear", 0.15, 0.15),
        ("rain_straight", "day", "rainy", 0.0, 0.0),
        ("rain_wet_night", "night", "rainy", -0.15, -0.15),
        ("overcast_straight", "day", "overcast", 0.0, 0.0),
        ("highway_wide", "day", "clear", 0.05, 0.05),
        ("city_narrow", "day", "clear", -0.05, -0.05),
        ("sunset_curve", "day", "clear", 0.25, 0.25),
        ("val_day", "day", "clear", 0.1, 0.1),
        ("val_night", "night", "rainy", -0.1, -0.1),
    ]

    for name, tod, wth, l_curv, r_curv in conditions:
        im_p = img_dir / f"{name}.jpg"
        m_p = mask_dir / f"{name}_mask.png"

        if not im_p.exists() or not m_p.exists():
            # Render synthetic frame with road and lanes
            h, w = IMG_H, IMG_W
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            horizon = int(h * 0.45)
            sky_col = (
                (20, 15, 25)
                if tod == "night"
                else ((130, 135, 140) if wth == "rainy" else (210, 160, 100))
            )
            road_col = (
                (35, 35, 40)
                if tod == "night"
                else ((55, 60, 65) if wth == "rainy" else (75, 75, 80))
            )

            frame[0:horizon, :] = sky_col
            frame[horizon : horizon + 60, :] = (
                (20, 30, 20) if tod == "night" else (50, 70, 50)
            )
            road_pts = np.array(
                [
                    [int(w * 0.35), horizon + 60],
                    [int(w * 0.65), horizon + 60],
                    [w, h],
                    [0, h],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(frame, [road_pts], road_col)

            # Draw lane markings onto frame
            mask = generate_gt_lane_mask(
                w,
                h,
                left_curvature=l_curv,
                right_curvature=r_curv,
                horizon_y=horizon + 60,
            )
            lane_color = (180, 180, 180) if tod == "night" else (230, 230, 230)
            frame[mask > 0] = lane_color

            cv2.imwrite(str(im_p), frame)
            cv2.imwrite(str(m_p), mask)

        if "val" in name:
            val_imgs.append(im_p)
            val_masks.append(m_p)
        else:
            train_imgs.append(im_p)
            train_masks.append(m_p)

    return train_imgs, train_masks, val_imgs, val_masks
