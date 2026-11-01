"""
CARLA Camera Publisher ROS2 Node.
Spawns the ego vehicle, attaches an RGB camera (1280x720, 20 FPS, 90 deg FOV)
publishing sensor_msgs/Image on /camera/image_raw.
Also publishes ground-truth depth on /camera/depth_gt for evaluation.
Subscribes to /vehicle/control for reactive actuation.
"""

from __future__ import annotations
import sys
import time
from typing import Any, Dict

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

from carla_scenarios.scenario_base import ScenarioSimulation


class CarlaCameraPublisherNode(Node):
    """Publishes synchronized RGB camera and ground truth depth from CARLA / scenario engine."""

    def __init__(self, node_name: str = "carla_camera_publisher") -> None:
        super().__init__(node_name)

        # Declare parameters
        self.declare_parameter("scenario", "pedestrian")
        self.declare_parameter("weather", "clear")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("carla_host", "127.0.0.1")
        self.declare_parameter("carla_port", 2000)

        scenario_name = self.get_parameter("scenario").value or "pedestrian"
        weather = self.get_parameter("weather").value or "clear"
        self.fps = float(self.get_parameter("fps").value or 20.0)
        self.timer_period = 1.0 / self.fps

        self.bridge = CvBridge()

        # Publishers
        self.rgb_pub = self.create_publisher(RosImage, "/camera/image_raw", 10)
        self.depth_pub = self.create_publisher(RosImage, "/camera/depth_gt", 10)
        self.telemetry_pub = self.create_publisher(dict, "/vehicle/telemetry", 10)

        # Subscription for vehicle actuation / braking
        self.control_sub = self.create_subscription(
            dict, "/vehicle/control", self._on_vehicle_control, 10
        )

        self.get_logger().info(
            f"Initializing CARLA Camera Publisher (Scenario: {scenario_name}, Weather: {weather}, {self.fps} FPS)"
        )

        # Initialize simulation engine
        self.sim = ScenarioSimulation(scenario_name=scenario_name, weather=weather)
        self.frame_count = 0
        self.start_time = time.time()

        # 20 FPS Timer
        self.timer = self.create_timer(self.timer_period, self._on_timer)

    def _on_vehicle_control(self, msg: Dict[str, Any]) -> None:
        """Applies reactive control commands from the perception/controller node."""
        throttle = float(msg.get("throttle", 0.0))
        brake = float(msg.get("brake", 0.0))
        steer = float(msg.get("steer", 0.0))
        self.sim.apply_control(throttle=throttle, brake=brake, steer=steer)
        if brake > 0.0:
            self.get_logger().info(
                f"Braking applied: brake={brake:.2f}, throttle={throttle:.2f}"
            )

    def _on_timer(self) -> None:
        rgb_frame, depth_gt, telemetry = self.sim.step()
        self.frame_count += 1

        # Convert to ROS sensor_msgs/Image
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_frame, encoding="bgr8")
        depth_msg = self.bridge.cv2_to_imgmsg(depth_gt, encoding="bgr8")

        self.rgb_pub.publish(rgb_msg)
        self.depth_pub.publish(depth_msg)
        self.telemetry_pub.publish(telemetry)

        if self.frame_count % 40 == 0:
            elapsed = time.time() - self.start_time
            actual_fps = self.frame_count / max(0.01, elapsed)
            self.get_logger().info(
                f"Published frame {self.frame_count} ({actual_fps:.1f} FPS) | "
                f"Ego Speed: {telemetry['ego_speed_kmh']:.1f} km/h | Brake: {telemetry['ego_brake']:.2f}"
            )


def main(args: list[str] | None = None) -> None:
    if ROS2_NATIVE:
        rclpy.init(args=args)
        node = CarlaCameraPublisherNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        from src.ros_compat.node import spin

        node = CarlaCameraPublisherNode()
        spin(node)


if __name__ == "__main__":
    main(sys.argv)
