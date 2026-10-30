#!/usr/bin/env python
"""Exploratory Data Analysis for BDD100K detection labels.

Generates publication-quality plots summarising class distributions,
temporal/weather splits, bounding-box statistics, and class co-occurrence.
Optionally logs everything to Weights & Biases.

Usage::

    python -m src.data.eda --config configs/data_config.yaml
    python -m src.data.eda --config configs/data_config.yaml --wandb
    python -m src.data.eda --config configs/data_config.yaml --split val
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
DPI = 150
FIGSIZE_DEFAULT = (12, 7)
FIGSIZE_HEATMAP = (12, 10)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config.

    Returns:
        Parsed config dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return cfg


def load_labels(label_path: Path) -> List[Dict[str, Any]]:
    """Load BDD100K detection label JSON.

    Args:
        label_path: Path to the JSON label file.

    Returns:
        List of per-image annotation dicts.
    """
    with open(label_path, "r", encoding="utf-8") as fh:
        data: List[Dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d image entries from %s", len(data), label_path)
    return data


# ---------------------------------------------------------------------------
# Statistics extraction
# ---------------------------------------------------------------------------


def extract_statistics(
    data: List[Dict[str, Any]],
    class_names: List[str],
) -> Dict[str, Any]:
    """Extract all statistics needed for the EDA plots.

    Args:
        data: Parsed BDD100K label entries.
        class_names: Ordered list of target class names from config.

    Returns:
        Dictionary with computed statistics including class counts,
        per-image object counts, time-of-day and weather distributions,
        bounding-box dimensions, and co-occurrence matrix.
    """
    class_counter: Counter = Counter()
    objects_per_image: List[int] = []
    time_of_day_counter: Counter = Counter()
    weather_counter: Counter = Counter()
    scene_counter: Counter = Counter()

    # Box info: (width, height, category)
    box_dims: List[Tuple[float, float, str]] = []

    # Co-occurrence: image-level class presence
    cooccurrence: Dict[str, Counter] = defaultdict(Counter)

    for entry in data:
        labels = entry.get("labels") or []
        attrs = entry.get("attributes") or {}

        # Attribute distributions
        tod = attrs.get("timeofday", "unknown")
        weather = attrs.get("weather", "unknown")
        scene = attrs.get("scene", "unknown")
        time_of_day_counter[tod] += 1
        weather_counter[weather] += 1
        scene_counter[scene] += 1

        # Per-image counts and class stats
        n_objects = 0
        classes_in_image: set = set()

        for label in labels:
            cat = label.get("category", "unknown")
            box = label.get("box2d")
            if box is None:
                continue  # skip entries without box2d (e.g. drivable area)

            class_counter[cat] += 1
            n_objects += 1
            classes_in_image.add(cat)

            w = box["x2"] - box["x1"]
            h = box["y2"] - box["y1"]
            box_dims.append((w, h, cat))

        objects_per_image.append(n_objects)

        # Co-occurrence
        sorted_classes = sorted(classes_in_image)
        for i, c1 in enumerate(sorted_classes):
            for c2 in sorted_classes[i:]:
                cooccurrence[c1][c2] += 1
                if c1 != c2:
                    cooccurrence[c2][c1] += 1

    # Build ordered co-occurrence matrix using the class names found in data
    all_classes_found = sorted(class_counter.keys())
    n_cls = len(all_classes_found)
    cooc_matrix = np.zeros((n_cls, n_cls), dtype=np.int64)
    for i, c1 in enumerate(all_classes_found):
        for j, c2 in enumerate(all_classes_found):
            cooc_matrix[i, j] = cooccurrence[c1][c2]

    stats: Dict[str, Any] = {
        "total_images": len(data),
        "total_objects": sum(class_counter.values()),
        "class_counter": class_counter,
        "objects_per_image": objects_per_image,
        "time_of_day": time_of_day_counter,
        "weather": weather_counter,
        "scene": scene_counter,
        "box_dims": box_dims,
        "cooccurrence_matrix": cooc_matrix,
        "cooccurrence_classes": all_classes_found,
        "config_classes": class_names,
    }
    return stats


def log_summary(stats: Dict[str, Any]) -> None:
    """Log a human-readable summary of key statistics.

    Args:
        stats: Statistics dict produced by :func:`extract_statistics`.
    """
    logger.info("=" * 60)
    logger.info("EDA Summary")
    logger.info("  Total images : %d", stats["total_images"])
    logger.info("  Total objects: %d", stats["total_objects"])
    logger.info("-" * 60)

    logger.info("  Class distribution:")
    for cls, cnt in stats["class_counter"].most_common():
        logger.info("    %-20s  %6d", cls, cnt)

    logger.info("-" * 60)
    opi = stats["objects_per_image"]
    logger.info(
        "  Objects/image — min: %d  max: %d  mean: %.1f  median: %.1f",
        min(opi) if opi else 0,
        max(opi) if opi else 0,
        float(np.mean(opi)) if opi else 0.0,
        float(np.median(opi)) if opi else 0.0,
    )

    logger.info("-" * 60)
    logger.info("  Time of day:")
    for k, v in stats["time_of_day"].most_common():
        logger.info("    %-20s  %6d", k, v)

    logger.info("  Weather:")
    for k, v in stats["weather"].most_common():
        logger.info("    %-20s  %6d", k, v)

    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save_fig(
    fig: plt.Figure,
    output_dir: Path,
    name: str,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Save figure to disk and optionally to W&B.

    Args:
        fig: Matplotlib figure.
        output_dir: Directory to save into.
        name: File name (without extension).
        wandb_run: Active W&B run (``None`` to skip logging).

    Returns:
        Path of the saved PNG file.
    """
    out_path = output_dir / f"{name}.png"
    fig.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved plot: %s", out_path)

    if wandb_run is not None:
        import wandb  # noqa: F811 — conditional import

        wandb_run.log({name: wandb.Image(str(out_path))})

    return out_path


def plot_class_distribution(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Bar chart of object class counts.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    counter = stats["class_counter"]
    classes, counts = zip(*counter.most_common()) if counter else ([], [])

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    palette = sns.color_palette("viridis", len(classes))
    bars = ax.barh(
        list(reversed(classes)), list(reversed(counts)), color=list(reversed(palette))
    )
    ax.set_xlabel("Count")
    ax.set_title("BDD100K — Class Distribution")
    ax.bar_label(bars, padding=3, fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "class_distribution", wandb_run)


def plot_objects_per_image(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Histogram of objects per image.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    opi = stats["objects_per_image"]
    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.hist(
        opi,
        bins=range(0, max(opi) + 2),
        edgecolor="black",
        alpha=0.7,
        color="steelblue",
    )
    ax.set_xlabel("Number of Objects")
    ax.set_ylabel("Number of Images")
    ax.set_title("BDD100K — Objects per Image")
    ax.axvline(
        np.mean(opi), color="red", linestyle="--", label=f"Mean = {np.mean(opi):.1f}"
    )
    ax.axvline(
        np.median(opi),
        color="orange",
        linestyle="--",
        label=f"Median = {np.median(opi):.1f}",
    )
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, output_dir, "objects_per_image", wandb_run)


def plot_time_of_day(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Bar chart of time-of-day distribution.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    tod = stats["time_of_day"]
    labels, values = zip(*tod.most_common()) if tod else ([], [])

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("Set2", len(labels))
    bars = ax.bar(labels, values, color=palette, edgecolor="black")
    ax.set_ylabel("Number of Images")
    ax.set_title("BDD100K — Time of Day Distribution")
    ax.bar_label(bars, padding=3, fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "time_of_day", wandb_run)


def plot_weather(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Bar chart of weather condition distribution.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    weather = stats["weather"]
    labels, values = zip(*weather.most_common()) if weather else ([], [])

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    palette = sns.color_palette("coolwarm", len(labels))
    bars = ax.bar(labels, values, color=palette, edgecolor="black")
    ax.set_ylabel("Number of Images")
    ax.set_title("BDD100K — Weather Distribution")
    ax.bar_label(bars, padding=3, fontsize=9)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return _save_fig(fig, output_dir, "weather_distribution", wandb_run)


def plot_box_size_scatter(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Scatter plot of bounding-box width vs height, coloured by class.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    dims = stats["box_dims"]
    if not dims:
        logger.warning("  No bounding-box data — skipping scatter plot.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save_fig(fig, output_dir, "box_size_scatter", wandb_run)

    rng = np.random.RandomState(RANDOM_SEED)

    widths = np.array([d[0] for d in dims])
    heights = np.array([d[1] for d in dims])
    cats = [d[2] for d in dims]
    unique_cats = sorted(set(cats))
    palette = sns.color_palette("tab10", len(unique_cats))
    cat_to_color = {c: palette[i] for i, c in enumerate(unique_cats)}

    # Sub-sample for readability if too many points
    max_points = 10_000
    if len(dims) > max_points:
        idx = rng.choice(len(dims), max_points, replace=False)
        widths = widths[idx]
        heights = heights[idx]
        cats = [cats[i] for i in idx]
        logger.info("  Sub-sampled %d → %d points for scatter.", len(dims), max_points)

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    for cat in unique_cats:
        mask = [c == cat for c in cats]
        ax.scatter(
            widths[mask],
            heights[mask],
            c=[cat_to_color[cat]],
            label=cat,
            alpha=0.3,
            s=8,
            edgecolors="none",
        )
    ax.set_xlabel("Box Width (px)")
    ax.set_ylabel("Box Height (px)")
    ax.set_title("BDD100K — Bounding Box Sizes by Class")
    ax.legend(markerscale=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save_fig(fig, output_dir, "box_size_scatter", wandb_run)


def plot_box_area_histogram(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Histogram of bounding-box areas on a log scale.

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    dims = stats["box_dims"]
    if not dims:
        logger.warning("  No bounding-box data — skipping area histogram.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save_fig(fig, output_dir, "box_area_histogram", wandb_run)

    areas = np.array([d[0] * d[1] for d in dims])
    areas = areas[areas > 0]  # filter degenerate boxes

    fig, ax = plt.subplots(figsize=FIGSIZE_DEFAULT)
    ax.hist(areas, bins=100, edgecolor="black", alpha=0.7, color="teal", log=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Box Area (px²) — log scale")
    ax.set_ylabel("Count — log scale")
    ax.set_title("BDD100K — Bounding Box Area Distribution")
    fig.tight_layout()
    return _save_fig(fig, output_dir, "box_area_histogram", wandb_run)


def plot_cooccurrence_heatmap(
    stats: Dict[str, Any],
    output_dir: Path,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Heatmap of class co-occurrence (image-level).

    Args:
        stats: Statistics dict.
        output_dir: Save directory.
        wandb_run: Optional W&B run.

    Returns:
        Path of saved figure.
    """
    matrix = stats["cooccurrence_matrix"]
    classes = stats["cooccurrence_classes"]

    fig, ax = plt.subplots(figsize=FIGSIZE_HEATMAP)
    sns.heatmap(
        matrix,
        xticklabels=classes,
        yticklabels=classes,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("BDD100K — Class Co-occurrence (image-level)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    return _save_fig(fig, output_dir, "class_cooccurrence", wandb_run)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_eda(
    config_path: Path,
    split: str = "train",
    use_wandb: bool = False,
) -> None:
    """Run the full EDA pipeline.

    Args:
        config_path: Path to ``data_config.yaml``.
        split: Which split to analyse (``"train"`` or ``"val"``).
        use_wandb: Whether to log plots and stats to Weights & Biases.
    """
    np.random.seed(RANDOM_SEED)

    # ---- Config ----------------------------------------------------------
    cfg = load_config(config_path)
    repo_root = Path(config_path).resolve().parent.parent
    raw_labels_dir = repo_root / cfg["dataset"]["raw_labels_dir"]
    output_dir = repo_root / cfg["eda"]["output_dir"]
    class_names: List[str] = cfg["classes"]

    # Resolve label file
    label_file_map = {
        "train": "det_20/det_train.json",
        "val": "det_20/det_val.json",
    }
    if split not in label_file_map:
        raise ValueError(f"Unknown split '{split}'. Choose 'train' or 'val'.")

    label_path = raw_labels_dir / label_file_map[split]
    output_dir = output_dir / split
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ---- Load data -------------------------------------------------------
    data = load_labels(label_path)

    # ---- Compute stats ---------------------------------------------------
    stats = extract_statistics(data, class_names)
    log_summary(stats)

    # ---- W&B init --------------------------------------------------------
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="perception-stack",
            entity="subratkumar",
            name=f"eda-{split}",
            job_type="eda",
            config={
                "split": split,
                "total_images": stats["total_images"],
                "total_objects": stats["total_objects"],
            },
        )
        # Log scalar summary stats
        wandb_run.summary["total_images"] = stats["total_images"]
        wandb_run.summary["total_objects"] = stats["total_objects"]
        wandb_run.summary["mean_objects_per_image"] = float(
            np.mean(stats["objects_per_image"])
        )
        # Log class counts table
        wandb_table = wandb.Table(columns=["class", "count"])
        for cls, cnt in stats["class_counter"].most_common():
            wandb_table.add_data(cls, cnt)
        wandb_run.log({"class_counts": wandb_table})

    # ---- Generate plots --------------------------------------------------
    logger.info("Generating plots …")
    plot_class_distribution(stats, output_dir, wandb_run)
    plot_objects_per_image(stats, output_dir, wandb_run)
    plot_time_of_day(stats, output_dir, wandb_run)
    plot_weather(stats, output_dir, wandb_run)
    plot_box_size_scatter(stats, output_dir, wandb_run)
    plot_box_area_histogram(stats, output_dir, wandb_run)
    plot_cooccurrence_heatmap(stats, output_dir, wandb_run)

    logger.info("All plots saved to %s", output_dir)

    # ---- W&B finish ------------------------------------------------------
    if wandb_run is not None:
        wandb_run.finish()
        logger.info("W&B run finished.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="BDD100K detection label EDA — generate plots and statistics.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_config.yaml"),
        help="Path to data_config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default="train",
        help="Dataset split to analyse (default: %(default)s).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log plots and statistics to Weights & Biases.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_eda(
        config_path=args.config,
        split=args.split,
        use_wandb=args.wandb,
    )
