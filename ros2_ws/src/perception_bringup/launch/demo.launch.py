"""
ROS2 Launch File: demo.launch.py
Launches the full Autonomous Driving Perception Stack:
1. carla_camera_publisher (ego vehicle + RGB camera + Depth GT)
2. perception_node (YOLO detector + ByteTrack + LaneDetector + MiDaS depth + Reactive controller)
3. viz_node (live HUD overlay, BEV panel, TTC warning banner)

Usage:
  ros2 launch perception_bringup demo.launch.py scenario:=pedestrian weather:=rain
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # 1. Declare Launch Arguments
    scenario_arg = DeclareLaunchArgument(
        "scenario",
        default_value="pedestrian",
        description="Scenario to run: pedestrian, lead_car, night, rain",
    )
    weather_arg = DeclareLaunchArgument(
        "weather",
        default_value="rain",
        description="Weather condition: clear, rain, night",
    )
    backend_arg = DeclareLaunchArgument(
        "model_backend",
        default_value="pytorch",
        description="Model backend: pytorch, onnx, tensorrt",
    )
    show_window_arg = DeclareLaunchArgument(
        "show_window",
        default_value="false",
        description="Show live OpenCV display window",
    )

    scenario = LaunchConfiguration("scenario")
    weather = LaunchConfiguration("weather")
    backend = LaunchConfiguration("model_backend")
    show_window = LaunchConfiguration("show_window")

    # 2. Define ROS2 Nodes
    camera_publisher_node = Node(
        package="carla_camera_publisher",
        executable="publisher_node",
        name="carla_camera_publisher",
        output="screen",
        parameters=[
            {"scenario": scenario},
            {"weather": weather},
            {"fps": 20.0},
        ],
    )

    perception_node = Node(
        package="perception_node",
        executable="perception_node",
        name="perception_node",
        output="screen",
        parameters=[
            {"model_backend": backend},
            {"conf_threshold": 0.35},
            {"lane_method": "classical"},
            {"ttc_threshold_s": 2.5},
            {"brake_distance_m": 8.0},
            {"corridor_width_m": 1.4},
        ],
    )

    viz_node = Node(
        package="viz_node",
        executable="viz_node",
        name="viz_node",
        output="screen",
        parameters=[
            {"show_window": show_window},
            {"save_video": True},
            {"video_output_path": "docs/results/demo_overlay.mp4"},
        ],
    )

    return LaunchDescription(
        [
            scenario_arg,
            weather_arg,
            backend_arg,
            show_window_arg,
            camera_publisher_node,
            perception_node,
            viz_node,
        ]
    )
