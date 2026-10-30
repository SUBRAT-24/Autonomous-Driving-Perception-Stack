"""DeepSORT tracking runner fed with YOLO detections.

Runs DeepSORT on video sequences and outputs MOT-compliant trajectory files.

Usage::

    python -m src.tracking.deepsort_runner --config configs/tracker_config.yaml
    python -m src.tracking.deepsort_runner --video data/sample/clips/clip1_highway_day.mp4
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.tracking.deepsort.tracker import DeepSORTTracker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_deepsort_on_video(
    video_path: Path,
    model: YOLO,
    conf_thresh: float = 0.15,
    iou_thresh: float = 0.5,
    device: str = "0",
    output_mot_file: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Run DeepSORT tracking on a video file using YOLO detections.

    Args:
        video_path: Path to input MP4 video.
        model: Ultralytics YOLO model instance.
        conf_thresh: Minimum detection confidence.
        iou_thresh: NMS IoU threshold.
        device: '0' for GPU, 'cpu' for CPU.
        output_mot_file: Where to save MOT predictions file.

    Returns:
        Tuple of (all_frame_tracks, average_fps).
    """
    logger.info("Running DeepSORT on %s", video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    tracker = DeepSORTTracker(min_confidence=conf_thresh)
    all_frame_tracks: List[Dict[str, Any]] = []
    mot_lines = []
    frame_idx = 1

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run YOLO detections
        det_results = model.predict(
            source=frame,
            conf=conf_thresh,
            iou=iou_thresh,
            device=device,
            verbose=False,
        )

        dets_array = np.empty((0, 6), dtype=np.float32)
        if det_results and len(det_results) > 0:
            boxes = det_results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                dets_list = []
                for i in range(len(boxes)):
                    dets_list.append(
                        [
                            xyxy[i][0],
                            xyxy[i][1],
                            xyxy[i][2],
                            xyxy[i][3],
                            confs[i],
                            classes[i],
                        ]
                    )
                dets_array = np.array(dets_list, dtype=np.float32)

        # 2. Feed to DeepSORT
        active_tracks = tracker.update(frame, dets_array)

        frame_tracks = []
        for trk in active_tracks:
            x1, y1, x2, y2 = trk.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            frame_tracks.append(
                {
                    "frame": frame_idx,
                    "id": trk.track_id,
                    "bbox": (x1, y1, x2, y2),
                    "conf": trk.conf,
                    "cls_id": trk.cls_id,
                }
            )

            mot_lines.append(
                f"{frame_idx},{trk.track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{trk.conf:.4f},{trk.cls_id},1\n"
            )

        all_frame_tracks.append({"frame": frame_idx, "tracks": frame_tracks})
        frame_idx += 1

    cap.release()
    elapsed = max(1e-4, time.perf_counter() - start_time)
    fps = (frame_idx - 1) / elapsed
    logger.info(
        "DeepSORT completed %d frames in %.2f s (%.1f FPS)", frame_idx - 1, elapsed, fps
    )

    if output_mot_file:
        output_mot_file.parent.mkdir(parents=True, exist_ok=True)
        output_mot_file.write_text("".join(mot_lines), encoding="utf-8")
        logger.info("Saved MOT tracking output to %s", output_mot_file)

    return all_frame_tracks, fps


def main():
    parser = argparse.ArgumentParser(description="Run DeepSORT on driving clips")
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

    out_dir = Path("docs/results/tracking/deepsort")
    for vpath in video_paths:
        mot_out = out_dir / f"{vpath.stem}_deepsort_mot.txt"
        run_deepsort_on_video(vpath, model, output_mot_file=mot_out)


if __name__ == "__main__":
    main()
