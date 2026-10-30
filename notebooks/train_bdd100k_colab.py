# ruff: noqa: E402
# %% [markdown]
# # 🚗 BDD100K Object Detection — YOLOv8s Training
#
# **Autonomous Driving Perception Stack — Phase 1**
#
# This notebook trains a YOLOv8s model on the BDD100K 10K subset for
# multi-class object detection (9 driving-relevant classes).
#
# | Class ID | Class Name     |
# |----------|----------------|
# | 0        | car            |
# | 1        | truck          |
# | 2        | bus            |
# | 3        | person         |
# | 4        | rider          |
# | 5        | bike           |
# | 6        | motorcycle     |
# | 7        | traffic light  |
# | 8        | traffic sign   |
#
# **Hardware targets:**
# - Google Colab T4 (16 GB VRAM) → `batch=16`
# - Kaggle P100 (16 GB VRAM) → `batch=16`
# - Local RTX 3050 (4 GB VRAM) → `batch=4`
#
# > Set `RUNTIME` below to `"colab"`, `"kaggle"`, or `"local"`.

# %% [markdown]
# ## 1. Environment Setup

# %%
# ── Install dependencies ─────────────────────────────────────────────
# fmt: off
# !pip install ultralytics==8.2.* wandb pyyaml tqdm   # uncomment on Colab/Kaggle
# fmt: on

import logging
import os
import shutil
import json
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Detect runtime ───────────────────────────────────────────────────
RUNTIME: str = "colab"  # Change to "kaggle" or "local"

if RUNTIME == "colab":
    ROOT = Path("/content")
elif RUNTIME == "kaggle":
    ROOT = Path("/kaggle/working")
else:
    ROOT = Path(".")  # repo root when running locally

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "bdd100k"
PROCESSED_DIR = DATA_DIR / "processed" / "bdd100k"

logger.info("Runtime   : %s", RUNTIME)
logger.info("ROOT      : %s", ROOT)
logger.info("GPU avail : %s", torch.cuda.is_available())
if torch.cuda.is_available():
    logger.info("GPU name  : %s", torch.cuda.get_device_name(0))
    logger.info(
        "GPU mem   : %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9
    )

# %% [markdown]
# ## 2. Mount Drive / Set Kaggle Paths
#
# **Google Colab** — mounts Google Drive so we can read the dataset and
# persist trained weights across sessions.
#
# **Kaggle** — expects the dataset as a Kaggle dataset input.
#
# **Local** — no mounting needed; paths are relative to the repo root.

# %%
DRIVE_MOUNT = ROOT / "drive"

if RUNTIME == "colab":
    from google.colab import drive  # type: ignore[import-untyped]

    drive.mount(str(DRIVE_MOUNT))
    DRIVE_DATA = DRIVE_MOUNT / "MyDrive" / "datasets" / "bdd100k"
    logger.info("Drive dataset path: %s", DRIVE_DATA)
elif RUNTIME == "kaggle":
    DRIVE_DATA = Path("/kaggle/input/bdd100k")
    logger.info("Kaggle input path: %s", DRIVE_DATA)
else:
    DRIVE_DATA = None
    logger.info("Local mode — data expected at %s", RAW_DIR)

# %% [markdown]
# ## 3. Dataset Setup
#
# ### Expected raw layout
# ```
# bdd100k/
# ├── images/
# │   └── 100k/
# │       ├── train/   (7000 images for 10K subset)
# │       └── val/     (1000 images for 10K subset)
# └── labels/
#     └── det_20/
#         └── det_train.json
#         └── det_val.json
# ```
#
# **Option A** — Upload the BDD100K 10K subset to Google Drive at
# `My Drive/datasets/bdd100k/` with the layout above, then run the
# cell below to symlink it.
#
# **Option B** — If you host the zip somewhere, uncomment the wget lines.

# %%
# ── Link or copy raw data ────────────────────────────────────────────
if DRIVE_DATA is not None and DRIVE_DATA.exists():
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_DIR.exists():
        os.symlink(str(DRIVE_DATA), str(RAW_DIR))
        logger.info("Symlinked %s → %s", DRIVE_DATA, RAW_DIR)
    else:
        logger.info("Raw data already linked at %s", RAW_DIR)
else:
    # ── Option B: download via wget / gdown (edit URLs) ──────────
    # !wget -q -O /tmp/bdd100k_10k.zip "YOUR_HOSTED_URL"
    # !unzip -q /tmp/bdd100k_10k.zip -d {RAW_DIR.parent}
    logger.info("Skipping download — ensure data exists at %s", RAW_DIR)

# Verify raw data
for sub in ["images/100k/train", "images/100k/val"]:
    p = RAW_DIR / sub
    if p.exists():
        logger.info("✓ %s  (%d files)", sub, len(list(p.iterdir())))
    else:
        logger.warning("✗ %s  NOT FOUND", sub)

# %% [markdown]
# ## 4. BDD100K → YOLO Label Conversion
#
# Converts the BDD100K JSON annotations to YOLO `.txt` format:
# `<class_id> <x_center> <y_center> <width> <height>` (normalized).
#
# The 9-class mapping matches our training config.

# %%
# ── Class mapping (BDD100K category → YOLO class id) ────────────────
BDD_TO_YOLO: dict[str, int] = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "person": 3,
    "pedestrian": 3,  # BDD100K uses "pedestrian"; map to "person"
    "rider": 4,
    "bike": 5,
    "bicycle": 5,  # BDD100K uses "bicycle"; map to "bike"
    "motorcycle": 6,
    "motor": 6,  # alternate BDD100K name
    "traffic light": 7,
    "traffic sign": 8,
}

IMG_W, IMG_H = 1280, 720  # BDD100K image dimensions


def convert_bdd_to_yolo(
    json_path: Path,
    output_label_dir: Path,
    output_image_dir: Path,
    source_image_dir: Path,
) -> dict[str, int]:
    """Convert a single BDD100K JSON annotation file to YOLO format.

    Args:
        json_path: Path to det_train.json or det_val.json.
        output_label_dir: Directory to write YOLO .txt label files.
        output_image_dir: Directory to symlink/copy images into.
        source_image_dir: Directory containing the source images.

    Returns:
        Dictionary with conversion statistics.
    """
    output_label_dir.mkdir(parents=True, exist_ok=True)
    output_image_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        frames = json.load(f)

    stats: dict[str, int] = {
        "total_images": 0,
        "converted_images": 0,
        "total_boxes": 0,
        "skipped_boxes": 0,
    }

    for frame in frames:
        stats["total_images"] += 1
        img_name: str = frame["name"]
        stem = Path(img_name).stem

        # Symlink image
        src_img = source_image_dir / img_name
        dst_img = output_image_dir / img_name
        if src_img.exists() and not dst_img.exists():
            os.symlink(str(src_img.resolve()), str(dst_img))

        labels_for_image: list[str] = []
        for label in frame.get("labels", []):
            cat = label.get("category", "")
            if cat not in BDD_TO_YOLO:
                stats["skipped_boxes"] += 1
                continue

            box = label.get("box2d")
            if box is None:
                stats["skipped_boxes"] += 1
                continue

            class_id = BDD_TO_YOLO[cat]
            x1, y1 = float(box["x1"]), float(box["y1"])
            x2, y2 = float(box["x2"]), float(box["y2"])

            # Convert to YOLO normalized format
            x_center = ((x1 + x2) / 2.0) / IMG_W
            y_center = ((y1 + y2) / 2.0) / IMG_H
            w = (x2 - x1) / IMG_W
            h = (y2 - y1) / IMG_H

            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w > 0 and h > 0:
                labels_for_image.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                )
                stats["total_boxes"] += 1

        # Write label file (even if empty — YOLO expects the file to exist)
        label_path = output_label_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(labels_for_image))

        if labels_for_image:
            stats["converted_images"] += 1

    return stats


# ── Run conversion ───────────────────────────────────────────────────
for split in ["train", "val"]:
    json_file = RAW_DIR / "labels" / "det_20" / f"det_{split}.json"
    if not json_file.exists():
        logger.warning("Label file not found: %s", json_file)
        continue

    logger.info("Converting %s split...", split)
    result = convert_bdd_to_yolo(
        json_path=json_file,
        output_label_dir=PROCESSED_DIR / "labels" / split,
        output_image_dir=PROCESSED_DIR / "images" / split,
        source_image_dir=RAW_DIR / "images" / "100k" / split,
    )
    for k, v in result.items():
        logger.info("  %s: %d", k, v)

logger.info("✓ Conversion complete")

# %% [markdown]
# ## 5. Generate `dataset.yaml`

# %%
import yaml

DATASET_YAML = PROCESSED_DIR / "dataset.yaml"

dataset_config: dict = {
    "path": str(PROCESSED_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "nc": 9,
    "names": {
        0: "car",
        1: "truck",
        2: "bus",
        3: "person",
        4: "rider",
        5: "bike",
        6: "motorcycle",
        7: "traffic light",
        8: "traffic sign",
    },
}

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
with open(DATASET_YAML, "w") as f:
    yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
logger.info("✓ Wrote %s", DATASET_YAML)

# %% [markdown]
# ## 6. Train YOLOv8s
#
# Hyperparameters are tuned for a Colab T4 GPU (16 GB).
# Adjust `batch` if you're on a different GPU.

# %%
import wandb
from ultralytics import YOLO

# ── W&B login (set WANDB_API_KEY env var, or call interactively) ─────
wandb.login()

# ── Batch size per runtime ───────────────────────────────────────────
BATCH_SIZE = {
    "colab": 16,
    "kaggle": 16,
    "local": 4,
}[RUNTIME]

model = YOLO("yolov8s.pt")

results = model.train(
    data=str(DATASET_YAML),
    epochs=50,
    imgsz=640,
    batch=BATCH_SIZE,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3,
    weight_decay=0.0005,
    momentum=0.937,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
    device=0,
    workers=8 if RUNTIME != "local" else 4,
    seed=42,
    project="runs/detect",
    name=f"bdd100k_yolov8s_{RUNTIME}",
    save_period=10,
    patience=10,
)

logger.info("✓ Training complete")

# %% [markdown]
# ## 7. Validation & mAP Report

# %%
# ── Validate on val split ────────────────────────────────────────────
metrics = model.val(
    data=str(DATASET_YAML),
    split="val",
    conf=0.25,
    iou=0.7,
    device=0,
)

logger.info("─── Validation Results ───")
logger.info("  mAP50    : %.4f", metrics.box.map50)
logger.info("  mAP50-95 : %.4f", metrics.box.map)
logger.info("  Precision: %.4f", metrics.box.mp)
logger.info("  Recall   : %.4f", metrics.box.mr)

# Per-class AP
class_names = dataset_config["names"]
for i, ap in enumerate(metrics.box.ap50):
    logger.info("  AP50 %-15s : %.4f", class_names[i], ap)

# %% [markdown]
# ## 8. Save Weights to Drive

# %%
# ── Copy best weights ────────────────────────────────────────────────
TRAIN_DIR = Path("runs/detect") / f"bdd100k_yolov8s_{RUNTIME}"
BEST_PT = TRAIN_DIR / "weights" / "best.pt"
LAST_PT = TRAIN_DIR / "weights" / "last.pt"

if RUNTIME == "colab" and DRIVE_MOUNT.exists():
    SAVE_DIR = DRIVE_MOUNT / "MyDrive" / "perception-stack" / "weights"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, SAVE_DIR / "best.pt")
        logger.info("✓ Saved best.pt → %s", SAVE_DIR / "best.pt")
    if LAST_PT.exists():
        shutil.copy2(LAST_PT, SAVE_DIR / "last.pt")
        logger.info("✓ Saved last.pt → %s", SAVE_DIR / "last.pt")
elif RUNTIME == "kaggle":
    SAVE_DIR = Path("/kaggle/working/weights")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_PT.exists():
        shutil.copy2(BEST_PT, SAVE_DIR / "best.pt")
        logger.info("✓ Saved best.pt → %s", SAVE_DIR / "best.pt")
else:
    logger.info("Local mode — weights at %s", TRAIN_DIR / "weights")

# %% [markdown]
# ## 9. Sample Predictions

# %%
import matplotlib.pyplot as plt

# ── Run inference on a few val images ────────────────────────────────
val_imgs = sorted((PROCESSED_DIR / "images" / "val").glob("*.jpg"))[:8]

if not val_imgs:
    logger.warning("No validation images found for visualization")
else:
    # Load best model for inference
    best_model = YOLO(str(BEST_PT)) if BEST_PT.exists() else model

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, img_path in enumerate(val_imgs):
        preds = best_model.predict(
            source=str(img_path),
            conf=0.25,
            iou=0.7,
            device=0,
            verbose=False,
        )
        # Ultralytics returns annotated image via plot()
        annotated = preds[0].plot()  # BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB

        axes[idx].imshow(annotated_rgb)
        axes[idx].set_title(img_path.name, fontsize=8)
        axes[idx].axis("off")

    # Hide unused axes
    for idx in range(len(val_imgs), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("YOLOv8s — BDD100K Sample Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(TRAIN_DIR / "sample_predictions.png"), dpi=150)
    plt.show()
    logger.info("✓ Saved sample predictions plot")

# %% [markdown]
# ---
# **Next steps:**
# - Download `best.pt` from Drive / Kaggle output
# - Run full evaluation with `scripts/evaluate.py`
# - Proceed to Phase 2 (Lane Detection) or Phase 3 (Depth Estimation)
