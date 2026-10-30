"""Validation and failure-case analysis for trained YOLO models.

Runs YOLO validation on the BDD100K val set, reports per-class AP,
confusion matrix, precision-recall curves, and performs failure-case
analysis for night, rain, and small-object conditions.

Usage::

    python -m src.detection.validate \\
        --weights runs/detect/bdd100k_yolov8s/weights/best.pt \\
        --data data/processed/bdd100k/dataset.yaml

    # With W&B logging
    python -m src.detection.validate \\
        --weights runs/detect/bdd100k_yolov8s/weights/best.pt \\
        --wandb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_config.yaml"
DEFAULT_DATASET_YAML = PROJECT_ROOT / "data" / "processed" / "bdd100k" / "dataset.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "results" / "detection"
RAW_LABELS_DIR = PROJECT_ROOT / "data" / "raw" / "bdd100k" / "labels"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed config dictionary.
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _setup_wandb(cfg: Dict[str, Any]) -> None:
    """Set W&B environment variables from config.

    Args:
        cfg: Full configuration dictionary.
    """
    wandb_cfg = cfg.get("wandb", {})
    os.environ["WANDB_PROJECT"] = wandb_cfg.get("project", "perception-stack")
    os.environ["WANDB_ENTITY"] = wandb_cfg.get("entity", "subratkumar")


# ---------------------------------------------------------------------------
# BDD100K attribute loader
# ---------------------------------------------------------------------------


def _load_bdd_attributes(
    labels_dir: Path, split: str = "val"
) -> Dict[str, Dict[str, Any]]:
    """Load BDD100K image attributes (timeofday, weather, etc.).

    Reads the original BDD100K JSON label files to extract per-image
    attributes like ``timeofday`` and ``weather``.

    Args:
        labels_dir: Root directory containing BDD100K JSON label files.
        split: Dataset split to load (``train`` or ``val``).

    Returns:
        Dictionary mapping image filename → attribute dict.
    """
    attributes: Dict[str, Dict[str, Any]] = {}

    # BDD100K labels can be at several standard paths
    candidates = [
        labels_dir / f"det_{split}.json",
        labels_dir / "det_20" / f"det_{split}.json",
        labels_dir / f"bdd100k_labels_images_{split}.json",
    ]
    json_file = None
    for cand in candidates:
        if cand.exists():
            json_file = cand
            break

    if json_file is None:
        logger.warning(
            "BDD100K label file not found in %s (checked %s) — "
            "failure-case analysis by attribute will be skipped.",
            labels_dir,
            [c.name for c in candidates],
        )
        return attributes

    logger.info("Loading BDD100K attributes from %s", json_file)
    with open(json_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    for entry in data:
        name = entry.get("name", "")
        attrs = entry.get("attributes", {})
        attributes[name] = attrs

    logger.info("Loaded attributes for %d images", len(attributes))
    return attributes


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------


def run_validation(
    model: YOLO,
    data_yaml: Path,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Any:
    """Run standard YOLO validation and save artefacts.

    Args:
        model: Loaded YOLO model.
        data_yaml: Path to dataset.yaml.
        cfg: Training/validation config.
        output_dir: Directory for saving results.

    Returns:
        Ultralytics validation ``Results`` object.
    """
    val_cfg = cfg.get("validation", {})
    train_cfg = cfg.get("training", {})

    results = model.val(
        data=str(data_yaml),
        imgsz=train_cfg.get("imgsz", 640),
        batch=train_cfg.get("batch_size", 4),
        conf=val_cfg.get("conf_threshold", 0.25),
        iou=val_cfg.get("iou_threshold", 0.7),
        device=train_cfg.get("device", "0"),
        workers=train_cfg.get("workers", 0),
        split=val_cfg.get("split", "val"),
        project=str(output_dir),
        name="val_results",
        exist_ok=True,
        plots=True,  # Generates confusion matrix, PR curves, etc.
    )

    return results


def _extract_metrics(results: Any, class_names: List[str]) -> pd.DataFrame:
    """Extract per-class AP metrics from validation results.

    Args:
        results: Ultralytics validation Results object.
        class_names: List of class names.

    Returns:
        DataFrame with per-class AP50, AP50-95, precision, and recall.
    """
    metrics = results.results_dict

    # Overall metrics
    logger.info("=== Overall Metrics ===")
    logger.info("  mAP50     : %.4f", metrics.get("metrics/mAP50(B)", 0.0))
    logger.info("  mAP50-95  : %.4f", metrics.get("metrics/mAP50-95(B)", 0.0))
    logger.info("  Precision : %.4f", metrics.get("metrics/precision(B)", 0.0))
    logger.info("  Recall    : %.4f", metrics.get("metrics/recall(B)", 0.0))

    # Per-class metrics
    box = results.box
    rows = []
    for i, name in enumerate(class_names):
        try:
            ap50 = float(box.ap50[i]) if i < len(box.ap50) else 0.0
            ap = float(box.ap[i]) if i < len(box.ap) else 0.0
            p = float(box.p[i]) if i < len(box.p) else 0.0
            r = float(box.r[i]) if i < len(box.r) else 0.0
        except (IndexError, AttributeError):
            ap50 = ap = p = r = 0.0
        rows.append(
            {
                "class": name,
                "AP50": round(ap50, 4),
                "AP50-95": round(ap, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("\n%s", df.to_string(index=False))
    return df


def _copy_plots(results: Any, output_dir: Path) -> None:
    """Copy generated plots (confusion matrix, PR curves) to output dir.

    Args:
        results: Ultralytics validation Results object.
        output_dir: Destination directory.
    """
    # Ultralytics saves plots in the save_dir
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else None
    if save_dir is None or not save_dir.exists():
        logger.warning("No save_dir found on results; skipping plot copy.")
        return

    plot_files = list(save_dir.glob("*.png")) + list(save_dir.glob("*.jpg"))
    for pf in plot_files:
        dest = output_dir / pf.name
        shutil.copy2(pf, dest)
        logger.info("Saved plot: %s", dest)


# ---------------------------------------------------------------------------
# Failure-case analysis
# ---------------------------------------------------------------------------


def _get_small_object_indices(
    labels_dir: Path,
    threshold: int = 32,
) -> List[str]:
    """Find images that contain at least one small object.

    A 'small object' has bounding-box area < threshold × threshold pixels
    (in 1280 × 720 original resolution).

    Args:
        labels_dir: Path to YOLO-format label directory (val split).
        threshold: Max side length to consider an object 'small'.

    Returns:
        List of image stems containing small objects.
    """
    small_stems: List[str] = []
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        logger.warning(
            "Labels dir %s not found; skipping small-object analysis.", labels_path
        )
        return small_stems

    img_w, img_h = 1280, 720  # BDD100K original resolution
    area_threshold = threshold * threshold

    for txt_file in labels_path.glob("*.txt"):
        with open(txt_file, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # YOLO format: class cx cy w h (normalised)
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                if bw * bh < area_threshold:
                    small_stems.append(txt_file.stem)
                    break  # one small object is enough

    logger.info(
        "Found %d images with small objects (area < %d²)", len(small_stems), threshold
    )
    return small_stems


def _filter_images_by_attribute(
    attributes: Dict[str, Dict[str, Any]],
    key: str,
    value: str,
) -> List[str]:
    """Filter BDD100K images by a specific attribute value.

    Args:
        attributes: Per-image attribute dictionary.
        key: Attribute key (e.g. ``timeofday``, ``weather``).
        value: Attribute value to filter on (e.g. ``night``, ``rainy``).

    Returns:
        List of matching image filenames.
    """
    matches = [
        name
        for name, attrs in attributes.items()
        if attrs.get(key, "").lower() == value.lower()
    ]
    logger.info("Filter %s=%s → %d images", key, value, len(matches))
    return matches


def _compute_subset_metrics(
    model: YOLO,
    data_yaml: Path,
    image_stems: List[str],
    subset_name: str,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Optional[Dict[str, float]]:
    """Compute validation metrics for a subset of images.

    Uses the full validation pipeline but filters to only the specified
    images by running inference on them individually.

    Args:
        model: Loaded YOLO model.
        data_yaml: Path to dataset.yaml.
        image_stems: List of image file stems to validate on.
        subset_name: Human-readable name for logging.
        cfg: Configuration dictionary.
        output_dir: Directory for saving results.

    Returns:
        Dictionary with mAP50 and mAP50-95, or None if subset is empty.
    """
    if not image_stems:
        logger.warning("No images for subset '%s'; skipping.", subset_name)
        return None

    # Load dataset yaml to find image directory
    with open(data_yaml, "r", encoding="utf-8") as fh:
        ds_cfg = yaml.safe_load(fh)

    val_img_dir = Path(ds_cfg.get("path", "")) / ds_cfg.get("val", "images/val")
    if not val_img_dir.exists():
        # Try relative to project root
        val_img_dir = PROJECT_ROOT / "data" / "processed" / "bdd100k" / "images" / "val"

    # Collect existing image paths
    image_paths: List[Path] = []
    for stem in image_stems[:200]:  # Limit to 200 for speed
        for ext in (".jpg", ".png", ".jpeg"):
            candidate = val_img_dir / f"{stem}{ext}"
            if not candidate.exists():
                # stem might already include extension
                candidate = val_img_dir / stem
            if candidate.exists():
                image_paths.append(candidate)
                break

    if not image_paths:
        logger.warning("No valid image files found for subset '%s'.", subset_name)
        return None

    logger.info(
        "Running inference on %d images for subset '%s'...",
        len(image_paths),
        subset_name,
    )

    train_cfg = cfg.get("training", {})
    val_cfg = cfg.get("validation", {})

    # Run predictions
    all_results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=train_cfg.get("imgsz", 640),
        conf=val_cfg.get("conf_threshold", 0.25),
        iou=val_cfg.get("iou_threshold", 0.7),
        device=train_cfg.get("device", "0"),
        verbose=False,
    )

    # Gather stats
    total_preds = 0
    low_conf_cases: List[Dict[str, Any]] = []

    for result in all_results:
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            total_preds += len(boxes)
            confs = boxes.conf.cpu().numpy()
            for j, conf_val in enumerate(confs):
                if conf_val < 0.5:  # Low confidence predictions
                    cls_id = int(boxes.cls[j].cpu())
                    low_conf_cases.append(
                        {
                            "image": Path(result.path).name,
                            "class_id": cls_id,
                            "confidence": round(float(conf_val), 4),
                            "box": boxes.xyxy[j].cpu().tolist(),
                        }
                    )

    # Sort by confidence (ascending) to get worst cases
    low_conf_cases.sort(key=lambda x: x["confidence"])
    worst_20 = low_conf_cases[:20]

    # Save failure cases table
    if worst_20:
        df_failures = pd.DataFrame(worst_20)
        failure_path = output_dir / f"failures_{subset_name}.csv"
        df_failures.to_csv(failure_path, index=False)
        logger.info("Saved %d worst failures to %s", len(worst_20), failure_path)

    metrics = {
        "subset": subset_name,
        "num_images": len(image_paths),
        "total_predictions": total_preds,
        "low_conf_predictions": len(low_conf_cases),
    }
    logger.info(
        "Subset '%s': %d images, %d predictions, %d low-confidence",
        subset_name,
        metrics["num_images"],
        metrics["total_predictions"],
        metrics["low_conf_predictions"],
    )
    return metrics


def failure_analysis(
    model: YOLO,
    data_yaml: Path,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run failure-case analysis across conditions.

    Analyses performance on:
    - Night images (BDD100K ``timeofday == night``)
    - Rain images (BDD100K ``weather == rainy``)
    - Small objects (box area < 32×32 px)

    Args:
        model: Loaded YOLO model.
        data_yaml: Path to dataset.yaml.
        cfg: Full configuration dictionary.
        output_dir: Directory for saving failure tables and plots.

    Returns:
        Dictionary with per-condition metrics.
    """
    results: Dict[str, Any] = {}

    # Load BDD100K attributes
    attributes = _load_bdd_attributes(RAW_LABELS_DIR, split="val")

    # 1. Night images
    if attributes:
        night_images = _filter_images_by_attribute(attributes, "timeofday", "night")
        night_stems = [Path(n).stem for n in night_images]
        night_metrics = _compute_subset_metrics(
            model,
            data_yaml,
            night_stems,
            "night",
            cfg,
            output_dir,
        )
        results["night"] = night_metrics

    # 2. Rain images
    if attributes:
        rain_images = _filter_images_by_attribute(attributes, "weather", "rainy")
        rain_stems = [Path(n).stem for n in rain_images]
        rain_metrics = _compute_subset_metrics(
            model,
            data_yaml,
            rain_stems,
            "rain",
            cfg,
            output_dir,
        )
        results["rain"] = rain_metrics

    # 3. Small objects
    with open(data_yaml, "r", encoding="utf-8") as fh:
        ds_cfg = yaml.safe_load(fh)
    val_labels_dir = Path(ds_cfg.get("path", "")) / "labels" / "val"
    if not val_labels_dir.exists():
        val_labels_dir = (
            PROJECT_ROOT / "data" / "processed" / "bdd100k" / "labels" / "val"
        )

    small_stems = _get_small_object_indices(val_labels_dir, threshold=32)
    small_metrics = _compute_subset_metrics(
        model,
        data_yaml,
        small_stems,
        "small_objects",
        cfg,
        output_dir,
    )
    results["small_objects"] = small_metrics

    # Save summary
    summary_path = output_dir / "failure_analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Failure analysis summary saved to %s", summary_path)

    return results


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------


def _log_to_wandb(
    metrics_df: pd.DataFrame,
    failure_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Log validation results to Weights & Biases.

    Args:
        metrics_df: Per-class metrics DataFrame.
        failure_results: Failure analysis results dictionary.
        output_dir: Directory containing saved plots.
    """
    try:
        import wandb
    except ImportError:
        logger.error("wandb not installed; skipping W&B logging.")
        return

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "perception-stack"),
        entity=os.environ.get("WANDB_ENTITY", "subratkumar"),
        job_type="validation",
        tags=["validation", "bdd100k"],
    )

    # Log per-class metrics table
    wandb.log({"per_class_metrics": wandb.Table(dataframe=metrics_df)})

    # Log failure analysis
    for condition, metrics in failure_results.items():
        if metrics:
            wandb.log({f"failure/{condition}": metrics})

    # Log plots as images
    for img_path in output_dir.glob("*.png"):
        wandb.log({img_path.stem: wandb.Image(str(img_path))})

    # Log failure CSVs as artifacts
    failure_csvs = list(output_dir.glob("failures_*.csv"))
    if failure_csvs:
        artifact = wandb.Artifact("failure_cases", type="analysis")
        for csv_path in failure_csvs:
            artifact.add_file(str(csv_path))
        run.log_artifact(artifact)

    wandb.finish()
    logger.info("W&B logging complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Validate trained YOLO model on BDD100K",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained model weights (best.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATASET_YAML,
        help="Path to dataset.yaml.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for saving results.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point for the validation script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    cfg = _load_config(args.config)

    # Create output directory
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B setup
    if args.wandb:
        _setup_wandb(cfg)

    # Load model
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    logger.info("Loading model from %s", weights)
    model = YOLO(str(weights))

    # Load class names from data config
    data_cfg = _load_config(PROJECT_ROOT / "configs" / "data_config.yaml")
    class_names: List[str] = data_cfg.get("classes", [])

    # --- Standard validation ---
    logger.info("=" * 60)
    logger.info("Running standard validation...")
    logger.info("=" * 60)
    val_results = run_validation(model, args.data, cfg, output_dir)

    # Extract and save per-class metrics
    metrics_df = _extract_metrics(val_results, class_names)
    metrics_csv = output_dir / "per_class_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info("Per-class metrics saved to %s", metrics_csv)

    # Copy plots
    _copy_plots(val_results, output_dir)

    # --- Failure-case analysis ---
    logger.info("=" * 60)
    logger.info("Running failure-case analysis...")
    logger.info("=" * 60)
    failure_results = failure_analysis(model, args.data, cfg, output_dir)

    # --- W&B logging ---
    if args.wandb:
        _log_to_wandb(metrics_df, failure_results, output_dir)

    logger.info("Validation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
