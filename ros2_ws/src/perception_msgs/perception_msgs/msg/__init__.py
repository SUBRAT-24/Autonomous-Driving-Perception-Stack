"""
Message definitions for perception_msgs.
Provides native Python dataclasses that mirror ROS2 IDL for seamless operation
both inside standard ROS2 Humble environments and in pure-Python testing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List


@dataclass
class Header:
    """ROS2 standard Header representation."""

    stamp_sec: int = 0
    stamp_nanosec: int = 0
    frame_id: str = "camera_optical_frame"

    @classmethod
    def now(cls, frame_id: str = "camera_optical_frame") -> Header:
        now = time.time()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        return cls(stamp_sec=sec, stamp_nanosec=nanosec, frame_id=frame_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stamp_sec": self.stamp_sec,
            "stamp_nanosec": self.stamp_nanosec,
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Header:
        return cls(
            stamp_sec=int(data.get("stamp_sec", 0)),
            stamp_nanosec=int(data.get("stamp_nanosec", 0)),
            frame_id=str(data.get("frame_id", "camera_optical_frame")),
        )


@dataclass
class TrackedObject:
    """
    Tracked object message.
    id: Unique tracking ID across temporal frames.
    class_name: Detected semantic class (e.g. 'car', 'person', 'truck').
    confidence: Detection confidence [0.0, 1.0].
    bbox: Bounding box coordinates [x1, y1, x2, y2] in pixels.
    distance_m: Calibrated metric distance to the object in metres.
    ttc_s: Estimated Time-to-Collision in seconds (inf if receding or stopped).
    """

    id: int = 0
    class_name: str = ""
    confidence: float = 0.0
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    distance_m: float = 0.0
    ttc_s: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "class_name": str(self.class_name),
            "confidence": float(self.confidence),
            "bbox": [float(c) for c in self.bbox],
            "distance_m": float(self.distance_m),
            "ttc_s": float(self.ttc_s),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackedObject:
        return cls(
            id=int(data.get("id", 0)),
            class_name=str(data.get("class_name", "")),
            confidence=float(data.get("confidence", 0.0)),
            bbox=list(data.get("bbox", [0.0, 0.0, 0.0, 0.0])),
            distance_m=float(data.get("distance_m", 0.0)),
            ttc_s=float(data.get("ttc_s", float("inf"))),
        )


@dataclass
class TrackedObjectArray:
    """Array of TrackedObject instances with timestamp header."""

    header: Header = field(default_factory=Header.now)
    objects: List[TrackedObject] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackedObjectArray:
        header = Header.from_dict(data.get("header", {}))
        objs = [TrackedObject.from_dict(o) for o in data.get("objects", [])]
        return cls(header=header, objects=objs)


__all__ = ["Header", "TrackedObject", "TrackedObjectArray"]
