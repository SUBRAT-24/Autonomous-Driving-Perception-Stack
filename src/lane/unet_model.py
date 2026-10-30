"""Lightweight U-Net architecture for Real-time Lane Segmentation.

Provides:
- Seamless integration with ``segmentation_models_pytorch.Unet`` (when installed).
- High-performance, self-contained PyTorch ``LightUNet`` fallback with lightweight
  double-conv blocks, skip-connections, and real-time inference characteristics.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2 block with residual shortcut."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upscaling then DoubleConv with skip connection concatenation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: upsampled feature from lower layer
        # x2: skip connection from encoder
        x1 = self.up(x1)

        # Handle potential padding mismatch
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class LightUNet(nn.Module):
    """Lightweight 4-level U-Net for automotive lane line segmentation.

    Parameters: ~1.9 million (compact for edge & RTX 3050 real-time inference).
    """

    def __init__(
        self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        f = base_filters
        # Encoder
        self.inc = DoubleConv(in_channels, f)  # H, W -> 32
        self.down1 = DownBlock(f, f * 2)  # H/2, W/2 -> 64
        self.down2 = DownBlock(f * 2, f * 4)  # H/4, W/4 -> 128
        self.down3 = DownBlock(f * 4, f * 8)  # H/8, W/8 -> 256
        self.down4 = DownBlock(f * 8, f * 16)  # H/16, W/16 -> 512 (Bottleneck)

        # Decoder
        self.up1 = UpBlock(f * 16, f * 8)  # H/8, W/8 -> 256
        self.up2 = UpBlock(f * 8, f * 4)  # H/4, W/4 -> 128
        self.up3 = UpBlock(f * 4, f * 2)  # H/2, W/2 -> 64
        self.up4 = UpBlock(f * 2, f)  # H, W -> 32

        # Final 1x1 classification head
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits (N, out_channels, H, W)."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def get_lane_unet(
    in_channels: int = 3,
    out_channels: int = 1,
    encoder_name: str = "resnet18",
    pretrained: bool = True,
    device: str = "cpu",
) -> nn.Module:
    """Factory creating a lane segmentation U-Net model.

    Tries ``segmentation_models_pytorch`` first; if not installed, falls back
    to the native PyTorch ``LightUNet``.
    """
    try:
        import segmentation_models_pytorch as smp  # type: ignore

        logger.info(
            "Using segmentation_models_pytorch Unet (encoder: %s)", encoder_name
        )
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=out_channels,
        )
    except (ImportError, Exception) as exc:
        logger.info(
            "segmentation_models_pytorch not available (%s); using native LightUNet",
            exc,
        )
        model = LightUNet(in_channels=in_channels, out_channels=out_channels)

    return model.to(device)
