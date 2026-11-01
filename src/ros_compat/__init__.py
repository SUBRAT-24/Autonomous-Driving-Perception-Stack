"""
ROS2 Compatibility Layer for Autonomous Driving Perception Stack.
Provides a unified ROS2 API interface that automatically binds to native `rclpy`
when installed, or falls back to a high-performance in-process pub/sub bus
with full sensor_msgs, cv_bridge, parameter declaration, and rosbag support.
"""

from src.ros_compat.node import Node, spin, spin_once, Publisher, Subscription
from src.ros_compat.image import Image, CvBridge
from src.ros_compat.bag import RosbagRecorder, RosbagPlayer

__all__ = [
    "Node",
    "spin",
    "spin_once",
    "Publisher",
    "Subscription",
    "Image",
    "CvBridge",
    "RosbagRecorder",
    "RosbagPlayer",
]
