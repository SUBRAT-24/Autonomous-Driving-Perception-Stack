#!/usr/bin/env python
"""Generate synthetic driving clips and MOT ground-truth annotations.

Generates 3 realistic driving video clips (1280x720 @ 30 FPS):
1. ``highway_day.mp4`` — Daylight highway with 4 vehicles at varying speeds (overtaking/lane change)
2. ``city_crossing.mp4`` — Urban street with pedestrians crossing and turning traffic
3. ``night_rain.mp4`` — Night-time wet asphalt with headlights, oncoming car and motorcycle

Each clip is saved with corresponding MOT ground-truth annotations:
``<clip_name>_gt.txt`` in MOT format:
``<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,<class_id>,<visibility>``

Usage::

    python scripts/generate_driving_clips.py --output-dir data/sample/clips
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

IMG_W = 1280
IMG_H = 720
FPS = 30


@dataclass
class MovingObject:
    """Trajectory specification for a moving object in a clip."""

    id: int
    cls: str
    cls_id: int
    start_box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    end_box: Tuple[float, float, float, float]
    start_frame: int
    end_frame: int


def _interpolate_box(
    start_box: Tuple[float, float, float, float],
    end_box: Tuple[float, float, float, float],
    progress: float,
) -> Tuple[int, int, int, int]:
    """Linearly interpolate bounding box coordinates."""
    x1 = int(start_box[0] + progress * (end_box[0] - start_box[0]))
    y1 = int(start_box[1] + progress * (end_box[1] - start_box[1]))
    x2 = int(start_box[2] + progress * (end_box[2] - start_box[2]))
    y2 = int(start_box[3] + progress * (end_box[3] - start_box[3]))
    return x1, y1, x2, y2


def _draw_driving_frame(
    timeofday: str,
    weather: str,
    objects_in_frame: List[Tuple[int, str, Tuple[int, int, int, int]]],
    frame_idx: int,
) -> np.ndarray:
    """Render a synthetic 1280x720 driving scene for a single frame."""
    frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    horizon = int(IMG_H * 0.45)

    if timeofday == "night":
        sky_color = (20, 15, 25)
        road_color = (35, 35, 40)
    elif weather in ("rainy", "overcast"):
        sky_color = (130, 135, 140)
        road_color = (55, 60, 65)
    else:  # daytime clear
        sky_color = (210, 160, 100)
        road_color = (75, 75, 80)

    # Sky
    frame[0:horizon, :] = sky_color

    # Ground horizon strip
    ground_color = (40, 60, 45) if timeofday != "night" else (15, 20, 15)
    frame[horizon : horizon + 60, :] = ground_color

    # Road perspective
    pts = np.array(
        [
            [int(IMG_W * 0.35), horizon + 60],
            [int(IMG_W * 0.65), horizon + 60],
            [IMG_W, IMG_H],
            [0, IMG_H],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], road_color)

    # Road lane dashes (animated with frame_idx to create forward ego-motion illusion)
    center_color = (30, 190, 220)  # yellow
    offset = (frame_idx * 6) % 60

    for y_start in range(horizon + 60 + offset, IMG_H, 60):
        y_end = min(y_start + 35, IMG_H)
        alpha_s = max(0.0, (y_start - (horizon + 60)) / (IMG_H - (horizon + 60)))
        alpha_e = min(1.0, (y_end - (horizon + 60)) / (IMG_H - (horizon + 60)))
        x_s = int(IMG_W * 0.5 + (0.5 * IMG_W * 0.05) * alpha_s)
        x_e = int(IMG_W * 0.5 + (0.5 * IMG_W * 0.05) * alpha_e)
        w_dash = max(2, int(6 * alpha_s))
        cv2.line(frame, (x_s, y_start), (x_e, y_end), center_color, w_dash)

    # Draw moving objects
    for track_id, category, (x1, y1, x2, y2) in objects_in_frame:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(IMG_W - 1, x2), min(IMG_H - 1, y2)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if category == "car":
            # Color hash from ID
            car_col = (
                (track_id * 50) % 200 + 40,
                (track_id * 85) % 200 + 40,
                (track_id * 120) % 200 + 40,
            )
            cv2.rectangle(frame, (x1, y1 + int(bh * 0.3)), (x2, y2), car_col, -1)
            roof_inset = int(bw * 0.2)
            cv2.rectangle(
                frame,
                (x1 + roof_inset, y1),
                (x2 - roof_inset, y1 + int(bh * 0.45)),
                (int(car_col[0] * 0.8), int(car_col[1] * 0.8), int(car_col[2] * 0.8)),
                -1,
            )
            cv2.rectangle(
                frame,
                (x1 + roof_inset + 2, y1 + 3),
                (x2 - roof_inset - 2, y1 + int(bh * 0.35)),
                (200, 210, 220) if timeofday != "night" else (50, 60, 70),
                -1,
            )
            if timeofday == "night":
                cv2.circle(
                    frame, (x1 + int(bw * 0.2), y2 - int(bh * 0.2)), 5, (0, 0, 255), -1
                )
                cv2.circle(
                    frame, (x2 - int(bw * 0.2), y2 - int(bh * 0.2)), 5, (0, 0, 255), -1
                )

        elif category in ("truck", "bus"):
            col = (60, 120, 180) if category == "truck" else (30, 150, 60)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(
                frame,
                (x1 + 4, y1 + 4),
                (x2 - 4, y1 + int(bh * 0.35)),
                (210, 220, 230) if timeofday != "night" else (40, 50, 60),
                -1,
            )

        elif category == "pedestrian":
            head_r = max(2, int(bw * 0.25))
            head_center = (x1 + bw // 2, y1 + head_r)
            cv2.circle(frame, head_center, head_r, (160, 180, 210), -1)
            cv2.rectangle(
                frame,
                (x1 + int(bw * 0.15), y1 + 2 * head_r),
                (x2 - int(bw * 0.15), y2),
                (180, 120, 70),
                -1,
            )

        elif category in ("bicycle", "motorcycle"):
            col = (20, 180, 220) if category == "bicycle" else (50, 50, 200)
            wr = max(3, int(bh * 0.25))
            cv2.circle(frame, (x1 + wr, y2 - wr), wr, (30, 30, 30), 2)
            cv2.circle(frame, (x2 - wr, y2 - wr), wr, (30, 30, 30), 2)
            cv2.line(
                frame, (x1 + wr, y2 - wr), (x1 + bw // 2, y1 + int(bh * 0.3)), col, 2
            )
            cv2.line(
                frame, (x2 - wr, y2 - wr), (x1 + bw // 2, y1 + int(bh * 0.3)), col, 2
            )

        elif category == "traffic sign":
            cv2.line(
                frame,
                (x1 + bw // 2, y1 + int(bh * 0.5)),
                (x1 + bw // 2, y2),
                (120, 120, 120),
                2,
            )
            cv2.circle(
                frame,
                (x1 + bw // 2, y1 + int(bh * 0.3)),
                max(2, int(bw * 0.45)),
                (0, 0, 220),
                -1,
            )

    # Weather rain streaks
    if weather == "rainy":
        rng = np.random.RandomState(frame_idx)
        for _ in range(250):
            rx = rng.randint(0, IMG_W)
            ry = rng.randint(0, IMG_H - 20)
            cv2.line(frame, (rx, ry), (rx - 3, ry + 15), (200, 200, 210), 1)

    return frame


def generate_clip(
    name: str,
    timeofday: str,
    weather: str,
    num_frames: int,
    objects: List[MovingObject],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate and write a video clip and its MOT format ground-truth file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{name}.mp4"
    gt_path = output_dir / f"{name}_gt.txt"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, FPS, (IMG_W, IMG_H))

    gt_lines = []

    for f in range(1, num_frames + 1):
        objects_in_f = []

        for obj in objects:
            if obj.start_frame <= f <= obj.end_frame:
                duration = max(1, obj.end_frame - obj.start_frame)
                progress = (f - obj.start_frame) / duration
                x1, y1, x2, y2 = _interpolate_box(obj.start_box, obj.end_box, progress)
                objects_in_f.append((obj.id, obj.cls, (x1, y1, x2, y2)))

                # MOT format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>
                w = x2 - x1
                h = y2 - y1
                gt_lines.append(f"{f},{obj.id},{x1},{y1},{w},{h},1,{obj.cls_id},1\n")

        frame_img = _draw_driving_frame(timeofday, weather, objects_in_f, f)
        out.write(frame_img)

    out.release()
    gt_path.write_text("".join(gt_lines), encoding="utf-8")

    logger.info(
        "Generated clip: %s (%d frames) with GT %s", video_path, num_frames, gt_path
    )
    return video_path, gt_path


def create_all_clips(output_dir: Path) -> List[Tuple[Path, Path]]:
    """Create all 3 driving clips."""
    clips = []

    # 1. Highway Day: 4 vehicles, overtaking scenario
    highway_objects = [
        # Lead vehicle in ego lane
        MovingObject(
            id=1,
            cls="car",
            cls_id=0,
            start_box=(540, 420, 700, 530),
            end_box=(520, 390, 740, 560),
            start_frame=1,
            end_frame=90,
        ),
        # Faster vehicle overtaking from left
        MovingObject(
            id=2,
            cls="car",
            cls_id=0,
            start_box=(220, 430, 350, 510),
            end_box=(140, 410, 420, 580),
            start_frame=1,
            end_frame=90,
        ),
        # Truck on right lane moving steady
        MovingObject(
            id=3,
            cls="truck",
            cls_id=1,
            start_box=(820, 360, 1040, 540),
            end_box=(850, 370, 1060, 530),
            start_frame=1,
            end_frame=90,
        ),
        # Distant car appearing and approaching
        MovingObject(
            id=4,
            cls="car",
            cls_id=0,
            start_box=(580, 400, 640, 440),
            end_box=(560, 410, 680, 490),
            start_frame=15,
            end_frame=90,
        ),
    ]
    clips.append(
        generate_clip(
            name="clip1_highway_day",
            timeofday="daytime",
            weather="clear",
            num_frames=90,
            objects=highway_objects,
            output_dir=output_dir,
        )
    )

    # 2. City Crossing: pedestrian crossing, turning car, bicycle
    city_objects = [
        # Pedestrian crossing from left to right
        MovingObject(
            id=1,
            cls="pedestrian",
            cls_id=3,
            start_box=(160, 410, 205, 530),
            end_box=(550, 410, 595, 530),
            start_frame=1,
            end_frame=90,
        ),
        # Waiting car yielding to pedestrian
        MovingObject(
            id=2,
            cls="car",
            cls_id=0,
            start_box=(620, 420, 800, 540),
            end_box=(610, 420, 810, 540),
            start_frame=1,
            end_frame=90,
        ),
        # Bicycle along right curb
        MovingObject(
            id=3,
            cls="bicycle",
            cls_id=5,
            start_box=(880, 430, 950, 530),
            end_box=(890, 420, 980, 550),
            start_frame=10,
            end_frame=90,
        ),
        # Second pedestrian following first
        MovingObject(
            id=4,
            cls="pedestrian",
            cls_id=3,
            start_box=(100, 415, 145, 530),
            end_box=(460, 415, 505, 530),
            start_frame=20,
            end_frame=90,
        ),
    ]
    clips.append(
        generate_clip(
            name="clip2_city_crossing",
            timeofday="daytime",
            weather="clear",
            num_frames=90,
            objects=city_objects,
            output_dir=output_dir,
        )
    )

    # 3. Night Rain: oncoming car, motorcycle, traffic sign
    night_objects = [
        # Lead car ahead with red taillights
        MovingObject(
            id=1,
            cls="car",
            cls_id=0,
            start_box=(480, 430, 680, 550),
            end_box=(460, 410, 720, 570),
            start_frame=1,
            end_frame=90,
        ),
        # Motorcycle in left lane accelerating
        MovingObject(
            id=2,
            cls="motorcycle",
            cls_id=6,
            start_box=(310, 440, 390, 530),
            end_box=(220, 430, 330, 560),
            start_frame=1,
            end_frame=90,
        ),
        # Approaching car in distant lane
        MovingObject(
            id=3,
            cls="car",
            cls_id=0,
            start_box=(720, 440, 880, 540),
            end_box=(750, 430, 950, 560),
            start_frame=10,
            end_frame=90,
        ),
    ]
    clips.append(
        generate_clip(
            name="clip3_night_rain",
            timeofday="night",
            weather="rainy",
            num_frames=90,
            objects=night_objects,
            output_dir=output_dir,
        )
    )

    return clips


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3 driving clips for tracking evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sample/clips"),
        help="Destination directory (default: data/sample/clips)",
    )
    args = parser.parse_args()
    create_all_clips(args.output_dir)


if __name__ == "__main__":
    main()
