#!/usr/bin/env python
"""Verify BDD100K dataset integrity and placement.

Checks that the BDD100K 10K subset is correctly placed under ``data/raw/``
with the expected directory layout, readable images, and consistent
cross-references between label JSON entries and image files on disk.

Usage::

    python scripts/verify_data.py                       # defaults
    python scripts/verify_data.py --data-root data/raw/bdd100k
    python scripts/verify_data.py --sample-count 10     # check 10 images
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = Path("data/raw/bdd100k")
EXPECTED_SPLITS = ("train", "val")
LABEL_FILES: Dict[str, str] = {
    "train": "labels/det_20/det_train.json",
    "val": "labels/det_20/det_val.json",
}
RANDOM_SEED = 42
DEFAULT_SAMPLE_COUNT = 5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _check_directory(
    dir_path: Path,
    description: str,
    extension: str = ".jpg",
) -> List[Path]:
    """Return list of files with *extension* inside *dir_path*, logging issues.

    Args:
        dir_path: Directory to inspect.
        description: Human-readable label for log messages.
        extension: File extension to look for (default ``".jpg"``).

    Returns:
        Sorted list of matching file paths (empty if directory is missing).
    """
    if not dir_path.exists():
        logger.error("MISSING directory: %s  (%s)", dir_path, description)
        return []
    if not dir_path.is_dir():
        logger.error("NOT a directory: %s  (%s)", dir_path, description)
        return []

    files = sorted(dir_path.glob(f"*{extension}"))
    logger.info(
        "  %-30s  →  %d %s files found",
        description,
        len(files),
        extension,
    )
    if len(files) == 0:
        logger.warning("  No %s files in %s", extension, dir_path)
    return files


def _check_label_file(label_path: Path) -> List[dict]:
    """Load and return label JSON entries, logging errors on failure.

    Args:
        label_path: Path to a BDD100K detection JSON file.

    Returns:
        List of annotation dicts (empty on failure).
    """
    if not label_path.exists():
        logger.error("MISSING label file: %s", label_path)
        return []
    if not label_path.is_file():
        logger.error("NOT a file: %s", label_path)
        return []

    try:
        with open(label_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read %s: %s", label_path, exc)
        return []

    if not isinstance(data, list):
        logger.error("Label file %s does not contain a JSON list", label_path)
        return []

    logger.info("  %-30s  →  %d entries", label_path.name, len(data))
    return data


def _check_images_readable(
    image_files: List[Path],
    sample_count: int,
) -> int:
    """Attempt to load *sample_count* random images with OpenCV.

    Args:
        image_files: Pool of image paths to sample from.
        sample_count: Number of images to attempt loading.

    Returns:
        Number of images that failed to load.
    """
    if not image_files:
        logger.warning("  No images to sample for readability check.")
        return 0

    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(image_files, min(sample_count, len(image_files)))
    failures = 0

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.error("  UNREADABLE image: %s", img_path)
            failures += 1
        else:
            h, w, c = img.shape
            logger.debug("  OK %s  (%dx%dx%d)", img_path.name, w, h, c)

    if failures == 0:
        logger.info(
            "  Readability check passed (%d/%d images OK).",
            len(sample),
            len(sample),
        )
    else:
        logger.error(
            "  Readability check: %d/%d images FAILED.",
            failures,
            len(sample),
        )
    return failures


def _cross_check_labels(
    label_entries: List[dict],
    image_dir: Path,
    split_name: str,
) -> int:
    """Verify that image file names referenced in labels exist on disk.

    Args:
        label_entries: Parsed label JSON list.
        image_dir: Directory where corresponding images should reside.
        split_name: ``"train"`` or ``"val"`` for logging.

    Returns:
        Number of missing image files referenced in the labels.
    """
    if not label_entries:
        return 0

    existing_files = {p.name for p in image_dir.iterdir() if p.is_file()}
    missing = 0

    for entry in label_entries:
        img_name = entry.get("name", "")
        if img_name and img_name not in existing_files:
            missing += 1
            if missing <= 5:  # only log first few
                logger.error(
                    "  [%s] Label references missing image: %s",
                    split_name,
                    img_name,
                )

    if missing > 5:
        logger.error(
            "  [%s] ... and %d more missing images (showing first 5).",
            split_name,
            missing - 5,
        )

    if missing == 0:
        logger.info(
            "  [%s] Cross-check passed — all %d label entries have matching images.",
            split_name,
            len(label_entries),
        )
    else:
        logger.error(
            "  [%s] Cross-check FAILED — %d / %d images missing.",
            split_name,
            missing,
            len(label_entries),
        )
    return missing


# ---------------------------------------------------------------------------
# Main verification routine
# ---------------------------------------------------------------------------


def verify(data_root: Path, sample_count: int) -> bool:
    """Run all verification checks and return ``True`` if everything passes.

    Args:
        data_root: Root of the BDD100K raw data (e.g. ``data/raw/bdd100k``).
        sample_count: Number of images to load for readability testing.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    problems = 0
    logger.info("=" * 60)
    logger.info("BDD100K Data Verification")
    logger.info("  data_root   : %s", data_root.resolve())
    logger.info("  sample_count: %d", sample_count)
    logger.info("=" * 60)

    # 1. Check image directories -------------------------------------------
    logger.info("\n[1/4] Checking image directories …")
    split_images: Dict[str, List[Path]] = {}
    for split in EXPECTED_SPLITS:
        img_dir = data_root / "images" / "100k" / split
        files = _check_directory(img_dir, f"{split} images")
        split_images[split] = files
        if len(files) == 0:
            problems += 1

    # 2. Check label files -------------------------------------------------
    logger.info("\n[2/4] Checking label files …")
    split_labels: Dict[str, List[dict]] = {}
    for split in EXPECTED_SPLITS:
        label_path = data_root / LABEL_FILES[split]
        entries = _check_label_file(label_path)
        split_labels[split] = entries
        if len(entries) == 0:
            problems += 1

    # 3. Readability check -------------------------------------------------
    logger.info("\n[3/4] Checking image readability (sample) …")
    for split in EXPECTED_SPLITS:
        logger.info("  — %s split:", split)
        failures = _check_images_readable(split_images[split], sample_count)
        problems += failures

    # 4. Cross-check labels → images --------------------------------------
    logger.info("\n[4/4] Cross-checking labels against image files …")
    for split in EXPECTED_SPLITS:
        img_dir = data_root / "images" / "100k" / split
        missing = _cross_check_labels(split_labels[split], img_dir, split)
        if missing > 0:
            problems += 1

    # Summary --------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    if problems == 0:
        logger.info("✅  All checks PASSED — dataset is ready.")
    else:
        logger.error(
            "❌  %d problem(s) detected — please fix before proceeding.",
            problems,
        )
    logger.info("=" * 60)

    return problems == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify BDD100K dataset placement and integrity.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory for BDD100K raw data (default: %(default)s).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help="Number of images to load for readability check (default: %(default)s).",
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
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ok = verify(data_root=args.data_root, sample_count=args.sample_count)
    sys.exit(0 if ok else 1)
