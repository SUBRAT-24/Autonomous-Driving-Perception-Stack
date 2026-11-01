"""
Lightweight Rosbag Recorder and Player.
Records and replays ROS2 topic messages across scenarios.
"""

from __future__ import annotations
import json
import os
import time
from typing import Any, Callable, Dict, Iterator, List, Tuple
from src.ros_compat.bus import MessageBus
from src.ros_compat.image import Image
from perception_msgs.msg import TrackedObjectArray


class RosbagRecorder:
    """Records published messages on specified topics into a structured bag file."""

    def __init__(self, output_path: str, topics: List[str]) -> None:
        self.output_path = output_path
        self.topics = topics
        self.bus = MessageBus.get_instance()
        self._file = None
        self._callbacks: Dict[str, Callable[[Any], None]] = {}
        self.count = 0

    def start(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        self.count = 0

        for topic in self.topics:

            def make_cb(t: str):
                return lambda msg: self._record_message(t, msg)

            cb = make_cb(topic)
            self._callbacks[topic] = cb
            self.bus.subscribe(topic, cb)

    def _record_message(self, topic: str, msg: Any) -> None:
        if self._file is None:
            return

        ts = time.time()
        msg_type = type(msg).__name__
        if hasattr(msg, "to_dict"):
            payload = msg.to_dict()
        elif isinstance(msg, dict):
            payload = msg
        else:
            payload = str(msg)

        record = {
            "timestamp": ts,
            "topic": topic,
            "type": msg_type,
            "data": payload,
        }
        self._file.write(json.dumps(record) + "\n")
        self.count += 1
        if self.count % 25 == 0:
            self._file.flush()

    def stop(self) -> None:
        for topic, cb in self._callbacks.items():
            self.bus.unsubscribe(topic, cb)
        self._callbacks.clear()
        if self._file is not None:
            self._file.close()
            self._file = None


class RosbagPlayer:
    """Reads recorded bag files and replays messages onto the bus or yields them."""

    def __init__(self, bag_path: str) -> None:
        self.bag_path = bag_path
        self.bus = MessageBus.get_instance()

    def read_messages(
        self, topics: List[str] | None = None
    ) -> Iterator[Tuple[str, Any, float]]:
        if not os.path.exists(self.bag_path):
            raise FileNotFoundError(f"Rosbag file not found: {self.bag_path}")

        with open(self.bag_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                topic = record["topic"]
                if topics is not None and topic not in topics:
                    continue

                mtype = record["type"]
                data = record["data"]
                ts = record["timestamp"]

                # Deserialize
                if mtype == "Image":
                    msg = Image.from_dict(data)
                elif mtype == "TrackedObjectArray":
                    msg = TrackedObjectArray.from_dict(data)
                else:
                    msg = data

                yield topic, msg, ts

    def play(self, rate_hz: float = 20.0, max_messages: int | None = None) -> int:
        count = 0
        interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
        for topic, msg, _ in self.read_messages():
            self.bus.publish(topic, msg)
            count += 1
            if interval > 0:
                time.sleep(interval)
            if max_messages is not None and count >= max_messages:
                break
        return count


__all__ = ["RosbagRecorder", "RosbagPlayer"]
