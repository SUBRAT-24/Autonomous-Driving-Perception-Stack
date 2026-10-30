"""YOLO inference script for BDD100K object detection.

Runs YOLOv8 detection on individual images or directories and saves
annotated results with bounding boxes, class labels, and confidence
scores.

Usage::

    # Single image
    python -m src.detection.predict \\
        --weights runs/detect/bdd100k_yolov8s/weights/best.pt \\
        --source path/to/image.jpg

    # Directory of images
    python -m src.detection.predict \\
        --weights runs/detect/bdd100k_yolov8s/weights/best.pt \\
        --source path/to/images/ \\
        --output docs/results/detection/predictions/

    # With custom confidence threshold
    python -m src.detection.predict \\
        --weights runs/detect/bdd100k_yolov8s/weights/best.pt \\
        --source path/to/images/ \\
        --conf-threshold 0.5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# BDD100K class names and corresponding colours for visualisation
CLASS_NAMES: List[str] = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
]

# BGR colours for each class (visually distinct palette)
CLASS_COLORS: List[Tuple[int, int, int]] = [
    (0, 0, 255),  # pedestrian — red
    (0, 128, 255),  # rider — orange
    (0, 255, 0),  # car — green
    (255, 128, 0),  # truck — blue
    (255, 0, 128),  # bus — purple
    (128, 255, 0),  # train — lime
    (0, 255, 255),  # motorcycle — yellow
    (255, 255, 0),  # bicycle — cyan
    (0, 200, 200),  # traffic light — dark yellow
    (200, 200, 0),  # traffic sign — dark cyan
]

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str],
    colors: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Draw bounding boxes with class labels and confidence on an image.

    Args:
        image: BGR image as a NumPy array (H, W, 3).
        boxes: Bounding boxes in xyxy format, shape (N, 4).
        classes: Class indices, shape (N,).
        confidences: Confidence scores, shape (N,).
        class_names: List of class name strings.
        colors: List of BGR colour tuples per class.

    Returns:
        Annotated image copy.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for box, cls_id, conf in zip(boxes, classes, confidences):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = [int(c) for c in box]

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        color = colors[cls_id % len(colors)]
        name = class_names[cls_id] if cls_id < len(class_names) else f"cls_{cls_id}"
        label = f"{name} {conf:.2f}"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_images(
    model: YOLO,
    source: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
) -> int:
    """Run detection on images and save annotated results.

    Args:
        model: Loaded YOLO model.
        source: Path to a single image or a directory of images.
        output_dir: Directory to save annotated images.
        conf_threshold: Minimum confidence threshold.
        imgsz: Inference image size.
        device: Device string (``"0"``, ``"cpu"``).

    Returns:
        Number of images processed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    source = Path(source).resolve()
    if source.is_file() and source.suffix.lower() in SUPPORTED_EXTENSIONS:
        image_paths = [source]
    elif source.is_dir():
        image_paths = sorted(
            p for p in source.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    else:
        raise ValueError(f"Source must be an image file or directory: {source}")

    if not image_paths:
        logger.warning("No images found in %s", source)
        return 0

    logger.info(
        "Running inference on %d image(s), conf=%.2f, imgsz=%d",
        len(image_paths),
        conf_threshold,
        imgsz,
    )

    # Run batch prediction
    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )

    processed = 0
    for result in results:
        img_path = Path(result.path)
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read image: %s", img_path)
            continue

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            num_dets = len(boxes)
        else:
            xyxy = np.empty((0, 4))
            cls_ids = np.empty((0,))
            confs = np.empty((0,))
            num_dets = 0

        annotated = draw_detections(
            image,
            xyxy,
            cls_ids,
            confs,
            CLASS_NAMES,
            CLASS_COLORS,
        )

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        logger.info(
            "  %s → %d detections → %s",
            img_path.name,
            num_dets,
            out_path,
        )
        processed += 1

    logger.info("Processed %d images. Results in %s", processed, output_dir)
    return processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run YOLO detection inference on images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained model weights (best.pt).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to image or directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "results" / "detection" / "predictions",
        help="Output directory for annotated images.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device for inference ('0' for GPU, 'cpu' for CPU).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the prediction script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    # Load model
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    logger.info("Loading model from %s", weights)
    model = YOLO(str(weights))

    # Run inference
    predict_images(
        model=model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf_threshold,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
