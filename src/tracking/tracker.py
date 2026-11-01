"""Unified Tracker wrapper class for Autonomous Driving Perception Stack.

Provides a clean, framework-agnostic interface:
    Tracker.update(frame) -> List[Track(id, cls, bbox, conf)]
Ready for direct zero-copy integration into the ROS2 perception node.
"""

from __future__ import annotations
import os

os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs")

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from ultralytics import YOLO

from src.tracking.deepsort.tracker import DeepSORTTracker

logger = logging.getLogger(__name__)

# Class ID to class name mapping for BDD100K 9 classes
BDD100K_CLASS_NAMES: Dict[int, str] = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "person",
    4: "rider",
    5: "bike",
    6: "motorcycle",
    7: "traffic light",
    8: "traffic sign",
}


@dataclass
class Track:
    """Standardized object track representation.

    Attributes:
        id: Unique object tracking ID.
        cls: Human-readable class name (e.g. 'car', 'person').
        cls_id: Integer class ID (0-8).
        bbox: Bounding box tuple (x1, y1, x2, y2) in pixel coordinates.
        conf: Detection confidence score (0.0 to 1.0).
    """

    id: int
    cls: str
    cls_id: int
    bbox: Tuple[float, float, float, float]
    conf: float

    @property
    def center(self) -> Tuple[float, float]:
        """Compute bounding box center (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class Tracker:
    """Unified object detection and multi-object tracking interface.

    Wraps ByteTrack (primary / recommended) and DeepSORT, updating on raw BGR
    frames and outputting standardized :class:`Track` instances.

    Example::

        tracker = Tracker(tracker_type="bytetrack")
        tracks = tracker.update(frame_bgr)
        for t in tracks:
            print(f"Object #{t.id}: {t.cls} at {t.bbox} ({t.conf:.2f})")
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        weights_path: Optional[str | Path] = None,
        config_path: Optional[str | Path] = None,
        conf_thresh: float = 0.15,
        iou_thresh: float = 0.5,
        device: str = "0",
    ) -> None:
        """Initialise detection model and chosen tracker backend.

        Args:
            tracker_type: Tracking algorithm ('bytetrack' or 'deepsort').
            weights_path: Path to YOLO weights (.pt). Defaults to best trained weights.
            config_path: Path to tracker_config.yaml.
            conf_thresh: Detection confidence threshold.
            iou_thresh: NMS IoU threshold.
            device: Compute device ('0' for GPU, 'cpu' for CPU).
        """
        self.tracker_type = tracker_type.lower()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

        # Load weights
        if weights_path is None:
            default_weights = Path("runs/detect/bdd100k_yolov8s/weights/best.pt")
            if default_weights.exists():
                weights_path = default_weights
            else:
                weights_path = Path("yolov8s.pt")

        self.weights_path = Path(weights_path)
        logger.info("Tracker loading YOLO detector weights from %s", self.weights_path)
        self.model = YOLO(str(self.weights_path))

        # Load tracker config if available
        self.config: Dict[str, Any] = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                self.config = yaml.safe_load(fh)

        # Initialise backend
        if self.tracker_type == "deepsort":
            ds_cfg = self.config.get("deepsort", {})
            self._deepsort = DeepSORTTracker(
                max_dist=ds_cfg.get("max_dist", 0.2),
                min_confidence=self.conf_thresh,
                max_iou_distance=ds_cfg.get("max_iou_distance", 0.7),
                max_age=ds_cfg.get("max_age", 30),
                n_init=ds_cfg.get("n_init", 3),
            )
        else:
            self.tracker_type = "bytetrack"
            self._deepsort = None

        self._frame_count = 0
        try:
            import numpy as np

            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.track(
                source=dummy,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
                device=self.device,
            )
        except Exception:
            pass
        logger.info(
            "Tracker initialized successfully with backend: %s", self.tracker_type
        )

    def update(self, frame: np.ndarray) -> List[Track]:
        """Process a single BGR image frame and return active tracks.

        Args:
            frame: OpenCV BGR image (H, W, 3).

        Returns:
            List of confirmed, active :class:`Track` objects.
        """
        self._frame_count += 1
        tracks: List[Track] = []

        if self.tracker_type == "bytetrack":
            # Run Ultralytics ByteTrack
            results = self.model.track(
                source=frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
                imgsz=640,
                verbose=False,
            )

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)

                    # When track IDs are confirmed by ByteTrack
                    ids = (
                        boxes.id.cpu().numpy().astype(int)
                        if boxes.id is not None
                        else None
                    )

                    for i in range(len(boxes)):
                        t_id = int(ids[i]) if ids is not None else (i + 1)
                        cls_id = int(classes[i])
                        cls_name = BDD100K_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                        bbox = (
                            float(xyxy[i][0]),
                            float(xyxy[i][1]),
                            float(xyxy[i][2]),
                            float(xyxy[i][3]),
                        )
                        tracks.append(
                            Track(
                                id=t_id,
                                cls=cls_name,
                                cls_id=cls_id,
                                bbox=bbox,
                                conf=float(confs[i]),
                            )
                        )

        elif self.tracker_type == "deepsort":
            # Run YOLO detector first
            det_results = self.model.predict(
                source=frame,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self.device,
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

            # Feed to DeepSORT
            deep_tracks = self._deepsort.update(frame, dets_array)

            for dt in deep_tracks:
                x1, y1, x2, y2 = dt.to_tlbr()
                cls_name = BDD100K_CLASS_NAMES.get(dt.cls_id, f"class_{dt.cls_id}")
                tracks.append(
                    Track(
                        id=dt.track_id,
                        cls=cls_name,
                        cls_id=dt.cls_id,
                        bbox=(x1, y1, x2, y2),
                        conf=dt.conf,
                    )
                )

        return tracks

    def reset(self) -> None:
        """Reset internal tracking state."""
        self._frame_count = 0
        if self._deepsort is not None:
            self._deepsort = DeepSORTTracker(min_confidence=self.conf_thresh)
        logger.info("Tracker state reset.")
