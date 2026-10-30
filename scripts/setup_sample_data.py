#!/usr/bin/env python
"""Generate realistic sample driving scene images and BDD100K annotations.

Populates ``data/sample/`` and optionally ``data/raw/bdd100k/`` with sample
1280x720 driving scenes covering diverse conditions (day, night, rain, highway,
city) and all 9 driving classes plus edge-case categories.

Usage::

    python scripts/setup_sample_data.py --dest data/raw/bdd100k
    python scripts/setup_sample_data.py --dest data/sample
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

IMG_W = 1280
IMG_H = 720


def create_scene_image(
    timeofday: str,
    weather: str,
    objects: List[Tuple[str, Tuple[int, int, int, int]]],
) -> np.ndarray:
    """Create a synthetic 1280x720 driving scene image with road and objects."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # 1. Sky and horizon
    horizon = int(IMG_H * 0.45)
    if timeofday == "night":
        sky_color = (20, 15, 25)
        road_color = (35, 35, 40)
    elif weather in ("rainy", "overcast"):
        sky_color = (130, 135, 140)
        road_color = (55, 60, 65)
    else:  # daytime clear
        sky_color = (210, 160, 100)  # BGR: daylight sky
        road_color = (75, 75, 80)

    # Fill sky
    img[0:horizon, :] = sky_color

    # Ground/vegetation/buildings horizon strip
    ground_color = (40, 60, 45) if timeofday != "night" else (15, 20, 15)
    img[horizon : horizon + 60, :] = ground_color

    # Road polygon (perspective)
    pts = np.array(
        [
            [int(IMG_W * 0.35), horizon + 60],
            [int(IMG_W * 0.65), horizon + 60],
            [IMG_W, IMG_H],
            [0, IMG_H],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [pts], road_color)

    # Lane markings (dashed white / yellow)
    center_color = (30, 190, 220)  # yellow in BGR
    # Center dashed lane
    for y_start in range(horizon + 80, IMG_H, 60):
        y_end = min(y_start + 35, IMG_H)
        alpha_s = (y_start - (horizon + 60)) / (IMG_H - (horizon + 60))
        alpha_e = (y_end - (horizon + 60)) / (IMG_H - (horizon + 60))
        x_s = int(IMG_W * 0.5 + (0.5 * IMG_W * 0.05) * alpha_s)
        x_e = int(IMG_W * 0.5 + (0.5 * IMG_W * 0.05) * alpha_e)
        w_dash = max(2, int(6 * alpha_s))
        cv2.line(img, (x_s, y_start), (x_e, y_end), center_color, w_dash)

    # 2. Draw objects
    for category, (x1, y1, x2, y2) in objects:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(IMG_W - 1, x2), min(IMG_H - 1, y2)
        bw = x2 - x1
        bh = y2 - y1

        if category == "car":
            car_col = (180, 50, 40) if timeofday != "night" else (120, 30, 25)
            # Car body
            cv2.rectangle(img, (x1, y1 + int(bh * 0.3)), (x2, y2), car_col, -1)
            # Roof
            roof_inset = int(bw * 0.2)
            cv2.rectangle(
                img,
                (x1 + roof_inset, y1),
                (x2 - roof_inset, y1 + int(bh * 0.45)),
                (int(car_col[0] * 0.8), int(car_col[1] * 0.8), int(car_col[2] * 0.8)),
                -1,
            )
            # Windshield / glass
            cv2.rectangle(
                img,
                (x1 + roof_inset + 2, y1 + 3),
                (x2 - roof_inset - 2, y1 + int(bh * 0.35)),
                (200, 210, 220) if timeofday != "night" else (50, 60, 70),
                -1,
            )
            # Taillights / headlights
            if timeofday == "night":
                cv2.circle(
                    img, (x1 + int(bw * 0.2), y2 - int(bh * 0.2)), 6, (0, 0, 255), -1
                )
                cv2.circle(
                    img, (x2 - int(bw * 0.2), y2 - int(bh * 0.2)), 6, (0, 0, 255), -1
                )

        elif category in ("truck", "bus"):
            col = (60, 120, 180) if category == "truck" else (30, 150, 60)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
            # Windows
            cv2.rectangle(
                img,
                (x1 + 4, y1 + 4),
                (x2 - 4, y1 + int(bh * 0.35)),
                (210, 220, 230) if timeofday != "night" else (40, 50, 60),
                -1,
            )

        elif category == "pedestrian":
            # Head
            head_r = max(2, int(bw * 0.25))
            head_center = (x1 + bw // 2, y1 + head_r)
            cv2.circle(img, head_center, head_r, (160, 180, 210), -1)
            # Body
            cv2.rectangle(
                img,
                (x1 + int(bw * 0.15), y1 + 2 * head_r),
                (x2 - int(bw * 0.15), y2),
                (180, 120, 70),
                -1,
            )

        elif category == "rider":
            cv2.rectangle(img, (x1, y1), (x2, y2), (140, 100, 160), -1)
            cv2.circle(
                img,
                (x1 + bw // 2, y1 + int(bh * 0.2)),
                max(2, int(bw * 0.2)),
                (180, 190, 220),
                -1,
            )

        elif category in ("bicycle", "motorcycle"):
            col = (20, 180, 220) if category == "bicycle" else (50, 50, 200)
            # Wheels
            wr = max(3, int(bh * 0.25))
            cv2.circle(img, (x1 + wr, y2 - wr), wr, (30, 30, 30), 2)
            cv2.circle(img, (x2 - wr, y2 - wr), wr, (30, 30, 30), 2)
            # Frame
            cv2.line(
                img, (x1 + wr, y2 - wr), (x1 + bw // 2, y1 + int(bh * 0.3)), col, 2
            )
            cv2.line(
                img, (x2 - wr, y2 - wr), (x1 + bw // 2, y1 + int(bh * 0.3)), col, 2
            )

        elif category == "traffic light":
            # Housing
            cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), -1)
            # 3 lights: Red, Yellow, Green
            lh = bh // 3
            lr = max(1, min(bw // 3, lh // 3))
            cx = x1 + bw // 2
            cv2.circle(img, (cx, y1 + lh // 2), lr, (0, 0, 255), -1)  # Red
            cv2.circle(img, (cx, y1 + lh + lh // 2), lr, (0, 200, 220), -1)  # Yellow
            cv2.circle(img, (cx, y1 + 2 * lh + lh // 2), lr, (0, 255, 0), -1)  # Green

        elif category == "traffic sign":
            # Pole
            cv2.line(
                img,
                (x1 + bw // 2, y1 + int(bh * 0.5)),
                (x1 + bw // 2, y2),
                (120, 120, 120),
                2,
            )
            # Sign face (red/white circle or octagon)
            cv2.circle(
                img,
                (x1 + bw // 2, y1 + int(bh * 0.3)),
                max(2, int(bw * 0.45)),
                (0, 0, 220),
                -1,
            )
            cv2.circle(
                img,
                (x1 + bw // 2, y1 + int(bh * 0.3)),
                max(1, int(bw * 0.3)),
                (255, 255, 255),
                -1,
            )

    # 3. Weather overlay (rain streaks)
    if weather == "rainy":
        rng = np.random.RandomState(42)
        for _ in range(300):
            rx = rng.randint(0, IMG_W)
            ry = rng.randint(0, IMG_H - 20)
            cv2.line(img, (rx, ry), (rx - 3, ry + 15), (200, 200, 210), 1)

    return img


def generate_dataset_split(
    scene_defs: List[Dict[str, Any]],
    image_dir: Path,
) -> List[Dict[str, Any]]:
    """Generate image files and BDD100K annotation entries."""
    image_dir.mkdir(parents=True, exist_ok=True)
    bdd_entries = []

    for defn in scene_defs:
        img_name = defn["name"]
        timeofday = defn["attributes"]["timeofday"]
        weather = defn["attributes"]["weather"]
        objects_spec = defn["objects"]

        img_bgr = create_scene_image(timeofday, weather, objects_spec)
        img_path = image_dir / img_name
        cv2.imwrite(str(img_path), img_bgr)

        labels = []
        for cat, (x1, y1, x2, y2) in objects_spec:
            labels.append(
                {
                    "category": cat,
                    "box2d": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                }
            )

        bdd_entries.append(
            {
                "name": img_name,
                "attributes": defn["attributes"],
                "labels": labels,
            }
        )
        logger.info("Generated sample image: %s (%d objects)", img_path, len(labels))

    return bdd_entries


def get_sample_scenes() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Define 10 train scenes and 2 val scenes covering diverse conditions."""
    train_scenes = [
        # 1. Daytime clear highway with cars and truck
        {
            "name": "sample_0001_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "highway",
                "timeofday": "daytime",
            },
            "objects": [
                ("car", (200, 420, 360, 520)),
                ("car", (500, 400, 680, 530)),
                ("truck", (800, 360, 1050, 560)),
                ("traffic sign", (1100, 280, 1150, 420)),
            ],
        },
        # 2. Night city street with cars, pedestrians, traffic light
        {
            "name": "sample_0002_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "night",
            },
            "objects": [
                ("car", (450, 430, 650, 540)),
                ("pedestrian", (150, 410, 190, 530)),
                ("traffic light", (600, 180, 630, 270)),
                ("car", (720, 440, 890, 550)),
            ],
        },
        # 3. Rainy residential with bus, rider, bicycle
        {
            "name": "sample_0003_train.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "residential",
                "timeofday": "daytime",
            },
            "objects": [
                ("bus", (180, 330, 460, 550)),
                ("rider", (540, 410, 590, 520)),
                ("bicycle", (530, 440, 600, 540)),
                ("traffic sign", (950, 300, 990, 410)),
            ],
        },
        # 4. Night highway with cars and motorcycle
        {
            "name": "sample_0004_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "highway",
                "timeofday": "night",
            },
            "objects": [
                ("car", (300, 440, 480, 550)),
                ("motorcycle", (580, 430, 660, 530)),
                ("car", (750, 420, 980, 540)),
            ],
        },
        # 5. Overcast city with pedestrian crossing and traffic light
        {
            "name": "sample_0005_train.jpg",
            "attributes": {
                "weather": "overcast",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "objects": [
                ("pedestrian", (280, 400, 330, 530)),
                ("pedestrian", (350, 410, 395, 530)),
                ("traffic light", (580, 170, 610, 260)),
                ("car", (680, 430, 870, 540)),
            ],
        },
        # 6. Rainy night city with cars and small traffic sign
        {
            "name": "sample_0006_train.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "city street",
                "timeofday": "night",
            },
            "objects": [
                ("car", (400, 440, 600, 550)),
                ("car", (700, 450, 860, 545)),
                ("traffic sign", (1020, 320, 1050, 400)),  # small object
            ],
        },
        # 7. Clear daytime with truck, bus, and car
        {
            "name": "sample_0007_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "highway",
                "timeofday": "daytime",
            },
            "objects": [
                ("truck", (120, 340, 380, 550)),
                ("bus", (450, 330, 720, 540)),
                ("car", (800, 430, 990, 540)),
            ],
        },
        # 8. Dawn/dusk residential with rider and motorcycle
        {
            "name": "sample_0008_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "residential",
                "timeofday": "dawn/dusk",
            },
            "objects": [
                ("rider", (350, 400, 410, 520)),
                ("motorcycle", (340, 430, 430, 540)),
                ("car", (600, 430, 800, 540)),
                ("traffic sign", (920, 300, 960, 420)),
            ],
        },
        # 9. Clear daytime with bicycle, pedestrian, and car
        {
            "name": "sample_0009_train.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "objects": [
                ("pedestrian", (220, 400, 260, 520)),
                ("bicycle", (300, 440, 360, 530)),
                ("car", (500, 420, 720, 540)),
                ("traffic light", (850, 180, 880, 270)),
            ],
        },
        # 10. Rainy day with car and small traffic light
        {
            "name": "sample_0010_train.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "highway",
                "timeofday": "daytime",
            },
            "objects": [
                ("car", (380, 430, 580, 540)),
                ("car", (650, 440, 840, 540)),
                ("traffic light", (560, 200, 585, 275)),  # small object
            ],
        },
    ]

    val_scenes = [
        # Val 1: Daytime clear city street
        {
            "name": "sample_0011_val.jpg",
            "attributes": {
                "weather": "clear",
                "scene": "city street",
                "timeofday": "daytime",
            },
            "objects": [
                ("car", (250, 420, 450, 530)),
                ("truck", (520, 360, 760, 540)),
                ("pedestrian", (120, 410, 160, 530)),
                ("traffic sign", (880, 280, 930, 410)),
                ("traffic light", (650, 170, 680, 260)),
            ],
        },
        # Val 2: Night rainy city street (challenging validation case)
        {
            "name": "sample_0012_val.jpg",
            "attributes": {
                "weather": "rainy",
                "scene": "city street",
                "timeofday": "night",
            },
            "objects": [
                ("car", (420, 430, 620, 540)),
                ("rider", (200, 400, 250, 520)),
                ("motorcycle", (190, 430, 270, 540)),
                ("traffic light", (720, 190, 750, 280)),
            ],
        },
    ]

    return train_scenes, val_scenes


def setup_sample_dataset(dest_root: Path) -> None:
    """Populate destination root with images and JSON annotations."""
    dest_root = Path(dest_root).resolve()
    logger.info("Setting up sample dataset at: %s", dest_root)

    train_scenes, val_scenes = get_sample_scenes()

    # Image directories
    train_img_dir = dest_root / "images" / "100k" / "train"
    val_img_dir = dest_root / "images" / "100k" / "val"

    train_entries = generate_dataset_split(train_scenes, train_img_dir)
    val_entries = generate_dataset_split(val_scenes, val_img_dir)

    # Label directory
    labels_dir = dest_root / "labels" / "det_20"
    labels_dir.mkdir(parents=True, exist_ok=True)

    train_json_path = labels_dir / "det_train.json"
    with open(train_json_path, "w", encoding="utf-8") as fh:
        json.dump(train_entries, fh, indent=2)
    logger.info("Saved %s (%d entries)", train_json_path, len(train_entries))

    val_json_path = labels_dir / "det_val.json"
    with open(val_json_path, "w", encoding="utf-8") as fh:
        json.dump(val_entries, fh, indent=2)
    logger.info("Saved %s (%d entries)", val_json_path, len(val_entries))

    logger.info("Dataset setup completed successfully at %s", dest_root)


def main():
    parser = argparse.ArgumentParser(description="Generate sample BDD100K dataset")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/sample"),
        help="Destination directory (default: data/sample)",
    )
    parser.add_argument(
        "--also-populate-raw",
        action="store_true",
        help="Also populate data/raw/bdd100k for immediate pipeline execution",
    )
    args = parser.parse_args()

    setup_sample_dataset(args.dest)

    if args.also_populate_raw or args.dest == Path("data/raw/bdd100k"):
        setup_sample_dataset(Path("data/raw/bdd100k"))


if __name__ == "__main__":
    main()
