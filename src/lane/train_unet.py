"""Training pipeline for U-Net lane segmentation model.

Uses combined Binary Cross-Entropy + Dice Loss (designed specifically for class-imbalanced
thin lane markings) and logs metrics to Weights & Biases (W&B).

Usage::

    python -m src.lane.train_unet --epochs 15 --batch-size 4 --lr 1e-3
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.lane.data import LaneDataset, prepare_training_data
from src.lane.geometry import compute_mask_f1, compute_mask_iou
from src.lane.unet_model import get_lane_unet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy + Soft Dice Loss for lane segmentation."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        # Soft Dice
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for a single epoch. Returns (avg_loss, avg_iou)."""
    model.train()
    total_loss = 0.0
    iou_scores = []

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute batch IoU
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        targets = (masks > 0.5).cpu().numpy()
        for p, t in zip(preds, targets):
            iou_scores.append(
                compute_mask_iou(p[0].astype(np.uint8), t[0].astype(np.uint8))
            )

    avg_loss = total_loss / max(1, len(dataloader))
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    return avg_loss, avg_iou


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validate model on validation set. Returns (val_loss, val_iou, val_f1)."""
    model.eval()
    total_loss = 0.0
    iou_scores = []
    f1_scores = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            targets = (masks > 0.5).cpu().numpy()
            for p, t in zip(preds, targets):
                p_mask = p[0].astype(np.uint8)
                t_mask = t[0].astype(np.uint8)
                iou_scores.append(compute_mask_iou(p_mask, t_mask))
                f1_scores.append(compute_mask_f1(p_mask, t_mask))

    avg_loss = total_loss / max(1, len(dataloader))
    avg_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
    return avg_loss, avg_iou, avg_f1


def train_unet(
    data_dir: Path = Path("data/lane_dataset"),
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-3,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_weights: Path = Path("weights/lane_unet.pth"),
    use_wandb: bool = True,
) -> Path:
    """Execute complete training run for U-Net lane segmentation."""
    logger.info("Initializing U-Net lane segmentation training...")
    device = torch.device(device_str)
    output_weights.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare data
    train_imgs, train_masks, val_imgs, val_masks = prepare_training_data(data_dir)
    logger.info(
        "Dataset prepared: %d train pairs, %d val pairs", len(train_imgs), len(val_imgs)
    )

    train_ds = LaneDataset(train_imgs, train_masks, augment=True)
    val_ds = LaneDataset(val_imgs, val_masks, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 2. Model & Optimization
    model = get_lane_unet(in_channels=3, out_channels=1, device=device_str)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. W&B logging initialization
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            # Avoid network block in sandboxed mode by using offline or dryrun mode
            os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "offline")
            wandb_run = wandb.init(
                project="autonomous-driving-perception",
                name=f"unet-lane-{time.strftime('%Y%m%d-%H%M')}",
                config={
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "device": device_str,
                },
            )
            logger.info("W&B initialized in mode: %s", os.environ["WANDB_MODE"])
        except Exception as exc:
            logger.warning("Could not initialize W&B: %s", exc)

    best_val_iou = 0.0

    logger.info("Starting training for %d epochs on %s...", epochs, device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_iou, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, IoU: %.3f | Val Loss: %.4f, IoU: %.3f, F1: %.3f",
            epoch,
            epochs,
            elapsed,
            train_loss,
            train_iou,
            val_loss,
            val_iou,
            val_f1,
        )

        if wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_iou": train_iou,
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "val_f1": val_f1,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), str(output_weights))
            logger.info(
                "Saved new best model checkpoint (Val IoU: %.3f) -> %s",
                val_iou,
                output_weights,
            )

    if wandb_run is not None:
        wandb.finish()

    logger.info(
        "Training complete. Best Val IoU: %.3f. Model saved at %s",
        best_val_iou,
        output_weights,
    )
    return output_weights


def main():
    parser = argparse.ArgumentParser(description="Train U-Net lane segmentation model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--weights-out", type=Path, default=Path("weights/lane_unet.pth")
    )
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    train_unet(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_str=args.device,
        output_weights=args.weights_out,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
