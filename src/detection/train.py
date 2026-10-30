"""YOLO fine-tuning script for BDD100K object detection.

Fine-tunes a COCO-pretrained YOLOv8s on the BDD100K 10K subset
using the Ultralytics YOLO API.  Configuration is loaded from a YAML
file (default: configs/train_config.yaml).

Usage::

    python -m src.detection.train --config configs/train_config.yaml
    python -m src.detection.train --resume runs/detect/bdd100k_yolov8s/weights/last.pt
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"
DEFAULT_DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "bdd100k" / "dataset.yaml"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the YAML training configuration.

    Args:
        config_path: Absolute or relative path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return cfg


def _setup_wandb(cfg: Dict[str, Any]) -> None:
    """Configure W&B environment variables from the config.

    Args:
        cfg: Full training configuration dictionary.
    """
    wandb_cfg = cfg.get("wandb", {})
    project = wandb_cfg.get("project", "perception-stack")
    entity = wandb_cfg.get("entity", "subratkumar")

    os.environ["WANDB_PROJECT"] = project
    os.environ["WANDB_ENTITY"] = entity
    logger.info("W&B configured — project=%s, entity=%s", project, entity)


def _resolve_dataset_yaml(cfg: Dict[str, Any]) -> Path:
    """Resolve the path to the YOLO dataset.yaml.

    Looks for ``dataset.yaml`` in the ``processed_dir`` specified by the data
    config, falling back to the default location.

    Args:
        cfg: Full training configuration dictionary.

    Returns:
        Resolved ``Path`` to ``dataset.yaml``.
    """
    dataset_cfg = cfg.get("dataset", {})
    processed_dir = dataset_cfg.get("processed_dir", None)
    if processed_dir:
        candidate = PROJECT_ROOT / processed_dir / "dataset.yaml"
        if candidate.exists():
            return candidate
    return DEFAULT_DATASET_YAML


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(
    cfg: Dict[str, Any],
    data_yaml: Path,
    resume_weights: Optional[Path] = None,
) -> None:
    """Run YOLO fine-tuning.

    Args:
        cfg: Parsed YAML config dictionary.
        data_yaml: Path to the YOLO ``dataset.yaml``.
        resume_weights: If provided, resume training from this checkpoint.
    """
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    # ---- Model initialisation ----
    if resume_weights and resume_weights.exists():
        logger.info("Resuming training from %s", resume_weights)
        model = YOLO(str(resume_weights))
        resume = True
    else:
        pretrained = model_cfg.get("pretrained", "yolov8s.pt")
        logger.info("Loading COCO-pretrained model: %s", pretrained)
        model = YOLO(pretrained)
        resume = False

    # ---- Resolve data path ----
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}.  " "Run the BDD100K converter first."
        )
    logger.info("Dataset YAML: %s", data_yaml)

    # ---- Build training kwargs ----
    train_kwargs: Dict[str, Any] = {
        # Data
        "data": str(data_yaml),
        # Hyper-parameters
        "epochs": train_cfg.get("epochs", 50),
        "imgsz": train_cfg.get("imgsz", 640),
        "batch": train_cfg.get("batch_size", 4),
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": train_cfg.get("lr0", 0.001),
        "lrf": train_cfg.get("lrf", 0.01),
        "warmup_epochs": train_cfg.get("warmup_epochs", 3),
        "weight_decay": train_cfg.get("weight_decay", 0.0005),
        "momentum": train_cfg.get("momentum", 0.937),
        # Augmentation
        "augment": train_cfg.get("augment", True),
        "hsv_h": train_cfg.get("hsv_h", 0.015),
        "hsv_s": train_cfg.get("hsv_s", 0.7),
        "hsv_v": train_cfg.get("hsv_v", 0.4),
        "degrees": train_cfg.get("degrees", 0.0),
        "translate": train_cfg.get("translate", 0.1),
        "scale": train_cfg.get("scale", 0.5),
        "flipud": train_cfg.get("flipud", 0.0),
        "fliplr": train_cfg.get("fliplr", 0.5),
        "mosaic": train_cfg.get("mosaic", 1.0),
        "mixup": train_cfg.get("mixup", 0.0),
        # Device / workers
        "device": train_cfg.get("device", "0"),
        "workers": train_cfg.get("workers", 8),
        "seed": train_cfg.get("seed", 42),
        # Saving
        "project": str(PROJECT_ROOT / train_cfg.get("project", "runs/detect")),
        "name": train_cfg.get("name", "bdd100k_yolov8s"),
        "save_period": train_cfg.get("save_period", 10),
        "patience": train_cfg.get("patience", 10),
        "exist_ok": True,
        # Resume
        "resume": resume,
    }

    logger.info(
        "Starting training — epochs=%d, batch=%d, imgsz=%d, device=%s",
        train_kwargs["epochs"],
        train_kwargs["batch"],
        train_kwargs["imgsz"],
        train_kwargs["device"],
    )

    # ---- Launch training ----
    results = model.train(**train_kwargs)

    # ---- Report completion ----
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    logger.info("Training complete.")
    logger.info("  Best weights : %s (exists=%s)", best_pt, best_pt.exists())
    logger.info("  Last weights : %s (exists=%s)", last_pt, last_pt.exists())
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on BDD100K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Override path to dataset.yaml.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to last.pt to resume interrupted training.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    cfg = _load_config(args.config)

    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size

    # W&B setup
    _setup_wandb(cfg)

    # Resolve dataset YAML
    data_yaml = args.data if args.data else _resolve_dataset_yaml(cfg)

    train(cfg, data_yaml, resume_weights=args.resume)


if __name__ == "__main__":
    main()
