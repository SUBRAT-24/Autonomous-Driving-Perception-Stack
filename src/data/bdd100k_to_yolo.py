"""BDD100K to YOLO format label converter.

Converts BDD100K detection labels (JSON) to YOLO format (.txt) and
organises images and labels into the standard YOLO directory structure.

BDD100K JSON format:
    [{"name": "img.jpg", "labels": [{"category": "car",
      "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}}]}]

YOLO format (per line in .txt):
    class_id center_x center_y width height   (all normalised 0-1)

Usage:
    python -m src.data.bdd100k_to_yolo --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# BDD100K category name → YOLO class ID
BDD_TO_YOLO_CLASS: Dict[str, int] = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "pedestrian": 3,  # mapped to "person"
    "rider": 4,
    "bicycle": 5,  # mapped to "bike"
    "motorcycle": 6,
    "traffic light": 7,
    "traffic sign": 8,
}

YOLO_CLASS_NAMES: List[str] = [
    "car",
    "truck",
    "bus",
    "person",
    "rider",
    "bike",
    "motorcycle",
    "traffic light",
    "traffic sign",
]

NUM_CLASSES: int = 9

# BDD100K standard image dimensions
DEFAULT_IMG_WIDTH: int = 1280
DEFAULT_IMG_HEIGHT: int = 720


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class YOLOBox:
    """A single YOLO-format bounding box."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_line(self) -> str:
        """Serialise to a single YOLO label line."""
        return (
            f"{self.class_id} "
            f"{self.center_x:.6f} "
            f"{self.center_y:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass
class ConversionStats:
    """Accumulates conversion statistics."""

    total_images: int = 0
    images_with_labels: int = 0
    total_objects: int = 0
    per_class_counts: Dict[int, int] = field(
        default_factory=lambda: {i: 0 for i in range(NUM_CLASSES)}
    )
    skipped_classes: Dict[str, int] = field(default_factory=dict)
    skipped_invalid_boxes: int = 0

    def log_summary(self) -> None:
        """Log a human-readable summary of conversion statistics."""
        logger.info("=" * 60)
        logger.info("Conversion Statistics")
        logger.info("=" * 60)
        logger.info("Total images processed: %d", self.total_images)
        logger.info("Images with labels:     %d", self.images_with_labels)
        logger.info("Total objects:          %d", self.total_objects)
        logger.info("-" * 40)
        logger.info("Per-class counts:")
        for cls_id in range(NUM_CLASSES):
            logger.info(
                "  [%d] %-15s : %d",
                cls_id,
                YOLO_CLASS_NAMES[cls_id],
                self.per_class_counts.get(cls_id, 0),
            )
        if self.skipped_classes:
            logger.info("-" * 40)
            logger.info("Skipped classes (not in target set):")
            for name, count in sorted(
                self.skipped_classes.items(), key=lambda x: -x[1]
            ):
                logger.info("  %-20s : %d", name, count)
        if self.skipped_invalid_boxes > 0:
            logger.info("Skipped invalid boxes:  %d", self.skipped_invalid_boxes)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Core conversion functions
# ---------------------------------------------------------------------------


def convert_box(
    box2d: Dict[str, float],
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert BDD100K absolute box2d coords to YOLO normalised format.

    Args:
        box2d: Dict with keys ``x1``, ``y1``, ``x2``, ``y2`` (pixel coords).
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of ``(center_x, center_y, width, height)`` normalised to [0, 1],
        or ``None`` if the resulting box has non-positive width or height.
    """
    x1 = float(box2d["x1"])
    y1 = float(box2d["y1"])
    x2 = float(box2d["x2"])
    y2 = float(box2d["y2"])

    center_x = ((x1 + x2) / 2.0) / img_width
    center_y = ((y1 + y2) / 2.0) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # Clamp to [0, 1]
    center_x = max(0.0, min(1.0, center_x))
    center_y = max(0.0, min(1.0, center_y))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    # Skip degenerate boxes
    if w <= 0 or h <= 0:
        return None

    return (center_x, center_y, w, h)


def convert_label(
    label: Dict[str, Any],
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
) -> Optional[YOLOBox]:
    """Convert a single BDD100K label dict to a :class:`YOLOBox`.

    Args:
        label: A BDD100K label dict with ``category`` and ``box2d`` keys.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        A :class:`YOLOBox` instance, or ``None`` if the category is not in the
        target class set or the box is invalid.
    """
    category: str = label.get("category", "")
    class_id = BDD_TO_YOLO_CLASS.get(category)
    if class_id is None:
        return None

    box2d = label.get("box2d")
    if box2d is None:
        return None

    result = convert_box(box2d, img_width, img_height)
    if result is None:
        return None

    cx, cy, w, h = result
    return YOLOBox(class_id=class_id, center_x=cx, center_y=cy, width=w, height=h)


def convert_image_labels(
    image_entry: Dict[str, Any],
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
) -> Tuple[List[YOLOBox], List[str]]:
    """Convert all labels for a single BDD100K image entry.

    Args:
        image_entry: A single image dict from the BDD100K JSON array.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        A tuple of ``(yolo_boxes, skipped_categories)``.
    """
    labels = image_entry.get("labels") or []
    yolo_boxes: List[YOLOBox] = []
    skipped: List[str] = []

    for label in labels:
        category = label.get("category", "")
        yolo_box = convert_label(label, img_width, img_height)
        if yolo_box is not None:
            yolo_boxes.append(yolo_box)
        elif category not in BDD_TO_YOLO_CLASS:
            skipped.append(category)

    return yolo_boxes, skipped


def write_yolo_labels(
    yolo_boxes: List[YOLOBox],
    output_path: Path,
) -> None:
    """Write YOLO label lines to a .txt file.

    Args:
        yolo_boxes: List of :class:`YOLOBox` instances.
        output_path: Destination .txt file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [box.to_line() for box in yolo_boxes]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def generate_dataset_yaml(
    processed_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate the YOLO ``dataset.yaml`` configuration file.

    Args:
        processed_dir: Root of the processed dataset (e.g. ``data/processed/bdd100k``).
        output_path: Where to write the YAML. Defaults to ``processed_dir / dataset.yaml``.

    Returns:
        Path to the written YAML file.
    """
    if output_path is None:
        output_path = processed_dir / "dataset.yaml"

    dataset_cfg: Dict[str, Any] = {
        "path": str(processed_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": NUM_CLASSES,
        "names": YOLO_CLASS_NAMES,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)

    logger.info("Dataset YAML written to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# High-level conversion pipeline
# ---------------------------------------------------------------------------


def convert_split(
    json_path: Path,
    images_src_dir: Path,
    images_dst_dir: Path,
    labels_dst_dir: Path,
    img_width: int = DEFAULT_IMG_WIDTH,
    img_height: int = DEFAULT_IMG_HEIGHT,
    copy_images: bool = True,
) -> ConversionStats:
    """Convert a single BDD100K split (train or val) to YOLO format.

    Args:
        json_path: Path to the BDD100K detection JSON (e.g. ``det_train.json``).
        images_src_dir: Directory containing original BDD100K images.
        images_dst_dir: Destination directory for images.
        labels_dst_dir: Destination directory for YOLO label .txt files.
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        copy_images: If True, copy images to the destination directory.

    Returns:
        :class:`ConversionStats` with summary statistics.
    """
    logger.info("Loading BDD100K labels from %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        bdd_data: List[Dict[str, Any]] = json.load(f)

    stats = ConversionStats()
    images_dst_dir.mkdir(parents=True, exist_ok=True)
    labels_dst_dir.mkdir(parents=True, exist_ok=True)

    for entry in bdd_data:
        image_name: str = entry.get("name", "")
        if not image_name:
            logger.warning("Skipping entry with missing 'name' field")
            continue

        stats.total_images += 1

        yolo_boxes, skipped_cats = convert_image_labels(entry, img_width, img_height)

        # Track skipped categories
        for cat in skipped_cats:
            stats.skipped_classes[cat] = stats.skipped_classes.get(cat, 0) + 1

        # Write label file (even if empty — YOLO expects a .txt per image)
        label_name = Path(image_name).stem + ".txt"
        label_path = labels_dst_dir / label_name
        write_yolo_labels(yolo_boxes, label_path)

        if yolo_boxes:
            stats.images_with_labels += 1
            stats.total_objects += len(yolo_boxes)
            for box in yolo_boxes:
                stats.per_class_counts[box.class_id] += 1

        # Copy image
        if copy_images:
            src_img = images_src_dir / image_name
            dst_img = images_dst_dir / image_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
            else:
                logger.warning("Image not found: %s", src_img)

    return stats


def run_conversion(config_path: Path) -> None:
    """Run the full BDD100K → YOLO conversion pipeline.

    Args:
        config_path: Path to the data configuration YAML file.
    """
    logger.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg["dataset"]
    image_cfg = cfg["image"]

    # Resolve paths relative to project root (parent of configs/)
    project_root = config_path.parent.parent
    raw_images_dir = project_root / dataset_cfg["raw_images_dir"]
    raw_labels_dir = project_root / dataset_cfg["raw_labels_dir"]
    processed_dir = project_root / dataset_cfg["processed_dir"]

    img_width = image_cfg["width"]
    img_height = image_cfg["height"]

    # Conversion splits
    splits = {
        "train": {
            "json": raw_labels_dir / "det_20" / "det_train.json",
            "images_src": raw_images_dir / "train",
            "images_dst": processed_dir / "images" / "train",
            "labels_dst": processed_dir / "labels" / "train",
        },
        "val": {
            "json": raw_labels_dir / "det_20" / "det_val.json",
            "images_src": raw_images_dir / "val",
            "images_dst": processed_dir / "images" / "val",
            "labels_dst": processed_dir / "labels" / "val",
        },
    }

    all_stats = ConversionStats()

    for split_name, paths in splits.items():
        json_path = paths["json"]
        if not json_path.exists():
            logger.warning(
                "Label file not found, skipping split '%s': %s", split_name, json_path
            )
            continue

        logger.info("Converting split: %s", split_name)
        split_stats = convert_split(
            json_path=json_path,
            images_src_dir=paths["images_src"],
            images_dst_dir=paths["images_dst"],
            labels_dst_dir=paths["labels_dst"],
            img_width=img_width,
            img_height=img_height,
            copy_images=True,
        )

        logger.info("Split '%s' stats:", split_name)
        split_stats.log_summary()

        # Merge into aggregate stats
        all_stats.total_images += split_stats.total_images
        all_stats.images_with_labels += split_stats.images_with_labels
        all_stats.total_objects += split_stats.total_objects
        for k, v in split_stats.per_class_counts.items():
            all_stats.per_class_counts[k] += v
        for k, v in split_stats.skipped_classes.items():
            all_stats.skipped_classes[k] = all_stats.skipped_classes.get(k, 0) + v
        all_stats.skipped_invalid_boxes += split_stats.skipped_invalid_boxes

    # Generate dataset.yaml
    generate_dataset_yaml(processed_dir)

    logger.info("Overall conversion stats:")
    all_stats.log_summary()
    logger.info("Conversion complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Convert BDD100K detection labels to YOLO format.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to data configuration YAML (default: configs/data_config.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_conversion(args.config)
