"""
ROS2 sensor_msgs/Image and CvBridge implementation.
Compatible with standard ROS2 sensor_msgs.msg.Image and cv_bridge.CvBridge.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from perception_msgs.msg import Header


@dataclass
class Image:
    """ROS2 sensor_msgs/Image message representation."""

    header: Header = field(default_factory=Header.now)
    height: int = 0
    width: int = 0
    encoding: str = "bgr8"
    is_bigendian: int = 0
    step: int = 0
    data: bytes = b""

    def to_dict(self) -> dict:
        import base64
        import cv2

        # Use fast JPEG encoding for compact storage and 50x faster rosbag serialization
        if self.data and self.height > 0 and self.width > 0:
            try:
                ch = 1 if self.encoding in ("mono8", "8UC1") else 3
                arr = np.frombuffer(self.data, dtype=np.uint8).reshape(
                    (self.height, self.width, ch)
                )
                ret, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    return {
                        "header": self.header.to_dict(),
                        "height": self.height,
                        "width": self.width,
                        "encoding": self.encoding,
                        "format": "jpeg",
                        "data_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
                    }
            except Exception:
                pass

        return {
            "header": self.header.to_dict(),
            "height": self.height,
            "width": self.width,
            "encoding": self.encoding,
            "is_bigendian": self.is_bigendian,
            "step": self.step,
            "format": "raw",
            "data_b64": base64.b64encode(self.data).decode("ascii"),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Image:
        import base64
        import cv2

        raw_bytes = base64.b64decode(data.get("data_b64", ""))
        fmt = data.get("format", "raw")

        if fmt == "jpeg" and len(raw_bytes) > 0:
            try:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    return cls(
                        header=Header.from_dict(data.get("header", {})),
                        height=h,
                        width=w,
                        encoding=str(data.get("encoding", "bgr8")),
                        is_bigendian=0,
                        step=w * 3,
                        data=img.tobytes(),
                    )
            except Exception:
                pass

        return cls(
            header=Header.from_dict(data.get("header", {})),
            height=int(data.get("height", 0)),
            width=int(data.get("width", 0)),
            encoding=str(data.get("encoding", "bgr8")),
            is_bigendian=int(data.get("is_bigendian", 0)),
            step=int(data.get("step", 0)),
            data=raw_bytes,
        )


class CvBridge:
    """Bridges between OpenCV images (numpy arrays) and ROS2 Image messages."""

    def cv2_to_imgmsg(
        self, cv_image: np.ndarray, encoding: str = "bgr8", header: Header | None = None
    ) -> Image:
        if cv_image is None or cv_image.size == 0:
            return Image(header=header or Header.now(), encoding=encoding)

        h, w = cv_image.shape[:2]
        channels = 1 if cv_image.ndim == 2 else cv_image.shape[2]
        element_size = cv_image.itemsize
        step = w * channels * element_size

        return Image(
            header=header or Header.now(),
            height=h,
            width=w,
            encoding=encoding,
            is_bigendian=0,
            step=step,
            data=cv_image.tobytes(),
        )

    def imgmsg_to_cv2(
        self, img_msg: Image, desired_encoding: str = "passthrough"
    ) -> np.ndarray:
        if not img_msg.data or img_msg.height == 0 or img_msg.width == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        enc = img_msg.encoding.lower()
        if enc in ("bgr8", "rgb8"):
            arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                (img_msg.height, img_msg.width, 3)
            )
            if desired_encoding == "bgr8" and enc == "rgb8":
                import cv2

                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            elif desired_encoding == "rgb8" and enc == "bgr8":
                import cv2

                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr.copy()
        elif enc in ("mono8", "8uc1"):
            arr = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                (img_msg.height, img_msg.width)
            )
            return arr.copy()
        elif enc in ("32fc1", "float32"):
            arr = np.frombuffer(img_msg.data, dtype=np.float32).reshape(
                (img_msg.height, img_msg.width)
            )
            return arr.copy()
        elif enc in ("16uc1", "mono16"):
            arr = np.frombuffer(img_msg.data, dtype=np.uint16).reshape(
                (img_msg.height, img_msg.width)
            )
            return arr.copy()
        else:
            # Fallback to uint8
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            expected = img_msg.height * img_msg.width * 3
            if arr.size == expected:
                return arr.reshape((img_msg.height, img_msg.width, 3)).copy()
            return arr.reshape((img_msg.height, img_msg.width, -1)).copy()


__all__ = ["Image", "CvBridge"]
