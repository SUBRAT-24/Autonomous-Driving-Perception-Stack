"""ByteTrack tracking runner using Ultralytics.

Runs ByteTrack on video sequences and outputs MOT-compliant trajectory files
and visualised tracking overlays.

Usage::

    python -m src.tracking.bytetrack_runner --config configs/tracker_config.yaml
    python -m src.tracking.bytetrack_runner --video data/sample/clips/clip1_highway_day.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import yaml
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_bytetrack_on_video(
    video_path: Path,
    model: YOLO,
    conf_thresh: float = 0.15,
    iou_thresh: float = 0.5,
    device: str = "0",
    output_mot_file: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run ByteTrack on a video file.

    Args:
        video_path: Path to input MP4 video.
        model: Ultralytics YOLO model instance.
        conf_thresh: Detection confidence threshold.
        iou_thresh: NMS IoU threshold.
        device: '0' for GPU, 'cpu' for CPU.
        output_mot_file: Where to save MOT predictions file.

    Returns:
        Tuple of (all_frame_tracks, average_fps).
    """
    logger.info("Running ByteTrack on %s", video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    cap.release()

    start_time = time.perf_counter()
    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        stream=True,
        verbose=False,
    )

    all_frame_tracks: List[Dict[str, Any]] = []
    mot_lines = []
    frame_idx = 1

    for r in results:
        boxes = r.boxes
        frame_tracks = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

            for i in range(len(boxes)):
                t_id = int(ids[i]) if ids is not None else (i + 1)
                x1, y1, x2, y2 = xyxy[i]
                w = x2 - x1
                h = y2 - y1
                conf = float(confs[i])
                cls_id = int(classes[i])

                frame_tracks.append(
                    {
                        "frame": frame_idx,
                        "id": t_id,
                        "bbox": (float(x1), float(y1), float(x2), float(y2)),
                        "conf": conf,
                        "cls_id": cls_id,
                    }
                )

                # MOT format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class_id>,<visibility>
                mot_lines.append(
                    f"{frame_idx},{t_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls_id},1\n"
                )

        all_frame_tracks.append({"frame": frame_idx, "tracks": frame_tracks})
        frame_idx += 1

    elapsed = max(1e-4, time.perf_counter() - start_time)
    fps = frame_idx / elapsed
    logger.info(
        "ByteTrack completed %d frames in %.2f s (%.1f FPS)",
        frame_idx - 1,
        elapsed,
        fps,
    )

    if output_mot_file:
        output_mot_file.parent.mkdir(parents=True, exist_ok=True)
        output_mot_file.write_text("".join(mot_lines), encoding="utf-8")
        logger.info("Saved MOT tracking output to %s", output_mot_file)

    return all_frame_tracks, fps


def main():
    parser = argparse.ArgumentParser(description="Run ByteTrack on driving clips")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tracker_config.yaml")
    )
    parser.add_argument(
        "--video", type=Path, default=None, help="Optional single video path"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/bdd100k_yolov8s/weights/best.pt"),
    )
    args = parser.parse_args()

    model = YOLO(str(args.weights) if args.weights.exists() else "yolov8s.pt")

    if args.video:
        video_paths = [args.video]
    else:
        with open(args.config, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        video_paths = [Path(clip["video_path"]) for clip in cfg.get("clips", [])]

    out_dir = Path("docs/results/tracking/bytetrack")
    for vpath in video_paths:
        mot_out = out_dir / f"{vpath.stem}_bytetrack_mot.txt"
        run_bytetrack_on_video(vpath, model, output_mot_file=mot_out)


if __name__ == "__main__":
    main()
