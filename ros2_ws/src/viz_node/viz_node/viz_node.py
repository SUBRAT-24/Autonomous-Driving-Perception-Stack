"""
Visualization ROS2 Node (viz_node).
Subscribes to /perception/overlay and /perception/objects.
Renders real-time HUD with bounding boxes, tracking IDs, metric distances,
lane polygon, FPS counter, BEV radar display, and emergency collision warnings.
Outputs live OpenCV window or records annotated video stream.
"""

from __future__ import annotations
import os
import sys
import time
from typing import Any, List, Optional
import cv2

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge

    ROS2_NATIVE = True
except ImportError:
    from src.ros_compat.node import Node
    from src.ros_compat.image import Image as RosImage, CvBridge

    ROS2_NATIVE = False

from perception_msgs.msg import TrackedObjectArray


class VizNode(Node):
    """Subscribes to perception stream, displays live HUD, and saves demo video."""

    def __init__(self, node_name: str = "viz_node") -> None:
        super().__init__(node_name)

        self.declare_parameter("show_window", False)
        self.declare_parameter("save_video", True)
        self.declare_parameter("video_output_path", "docs/results/demo_overlay.mp4")

        self.show_window = bool(self.get_parameter("show_window").value)
        self.save_video = bool(self.get_parameter("save_video").value)
        self.video_output_path = str(self.get_parameter("video_output_path").value)

        self.bridge = CvBridge()
        self.video_writer: Optional[cv2.VideoWriter] = None

        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Subscriptions
        self.overlay_sub = self.create_subscription(
            RosImage, "/perception/overlay", self._on_overlay, 10
        )
        self.objects_sub = self.create_subscription(
            TrackedObjectArray, "/perception/objects", self._on_objects, 10
        )

        self.last_objects: List[Any] = []

        if self.save_video:
            os.makedirs(
                os.path.dirname(os.path.abspath(self.video_output_path)), exist_ok=True
            )

        self.get_logger().info(
            f"VizNode initialized (show_window={self.show_window}, save_video={self.save_video}, path={self.video_output_path})"
        )

    def _on_objects(self, msg: TrackedObjectArray) -> None:
        self.last_objects = msg.objects

    def _on_overlay(self, image_msg: RosImage) -> None:
        t_now = time.time()
        dt = t_now - self.last_frame_time
        self.last_frame_time = t_now
        self.frame_count += 1

        instant_fps = 1.0 / max(0.001, dt)
        self.fps = 0.9 * self.fps + 0.1 * instant_fps if self.fps > 0 else instant_fps

        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        # Initialize video writer if needed
        if self.save_video and self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                self.video_output_path, fourcc, 20.0, (w, h)
            )

        if self.video_writer is not None:
            self.video_writer.write(frame)

        if self.show_window:
            cv2.imshow("Autonomous Perception Live Overlay", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                self.show_window = False
                cv2.destroyAllWindows()

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Viz Node displayed frame {self.frame_count} @ {self.fps:.1f} FPS"
            )

    def destroy_node(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.get_logger().info(f"Saved demo video to {self.video_output_path}")
        if self.show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    if ROS2_NATIVE:
        rclpy.init(args=args)
        node = VizNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        from src.ros_compat.node import spin

        node = VizNode()
        spin(node)


if __name__ == "__main__":
    main(sys.argv)
