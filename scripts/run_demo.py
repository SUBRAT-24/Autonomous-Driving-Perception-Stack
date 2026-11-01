from __future__ import annotations
import os

os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs")
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in [
    "",
    "ros2_ws/src/carla_camera_publisher",
    "ros2_ws/src/perception_node",
    "ros2_ws/src/viz_node",
    "ros2_ws/src/perception_msgs",
]:
    p = os.path.join(ROOT, sub) if sub else ROOT
    if p not in sys.path:
        sys.path.insert(0, p)
"""
Unified Autonomous Driving Perception Stack Runner.
Launches the full pipeline across scenarios with live telemetry,
15+ FPS overlay rendering, reactive braking, and rosbag recording.
Compatible with standard ROS2 launch arguments (e.g. scenario:=pedestrian weather:=rain).
"""

import argparse
import sys
import time
from typing import Any, Dict

from carla_camera_publisher.publisher_node import CarlaCameraPublisherNode
from perception_node.perception_node import PerceptionNode
from viz_node.viz_node import VizNode
from src.ros_compat.bus import MessageBus
from src.ros_compat.bag import RosbagRecorder


def run_pipeline(
    scenario: str = "pedestrian",
    weather: str = "rain",
    backend: str = "pytorch",
    duration_sec: float = 5.0,
    show_window: bool = False,
    record_bag: bool = True,
    bag_path: str = "data/sample/rosbags/rosbag_pedestrian_crossing.bag",
    video_path: str = "docs/results/demo_overlay.mp4",
) -> Dict[str, Any]:
    # Reset MessageBus
    MessageBus.reset()

    print("\n=======================================================")
    print(" LAUNCHING PERCEPTION STACK DEMO")
    print(
        f" Scenario: {scenario.upper()} | Weather: {weather.upper()} | Backend: {backend.upper()}"
    )
    print("=======================================================\n")

    # 1. Initialize Nodes
    pub_node = CarlaCameraPublisherNode()
    pub_node.set_parameters({"scenario": scenario, "weather": weather, "fps": 20.0})
    if hasattr(pub_node, "timer") and pub_node.timer is not None:
        pub_node.timer.cancel()

    perc_node = PerceptionNode(backend=backend)
    perc_node.set_parameters(
        {
            "model_backend": backend,
            "conf_threshold": 0.35,
            "lane_method": "classical",
            "ttc_threshold_s": 2.5,
            "brake_distance_m": 8.0,
            "corridor_width_m": 1.4,
        }
    )

    viz = VizNode()
    viz.set_parameters(
        {
            "show_window": show_window,
            "save_video": True,
            "video_output_path": video_path,
        }
    )

    # 2. Setup Rosbag Recorder
    recorder = None
    if record_bag:
        recorder = RosbagRecorder(
            output_path=bag_path,
            topics=["/camera/image_raw", "/perception/objects", "/vehicle/control"],
        )
        recorder.start()
        print(f"[ROSBAG] Recording to {bag_path}...")

    # 3. Execution Loop
    # Pipeline warmup to eliminate first-frame CUDA / weights initialization jitter
    pub_node._on_timer()
    pub_node._on_timer()

    max_frames = max(40, int(duration_sec * 20.0))
    brake_triggered = False
    brake_frame = None
    total_frames = 0
    stopped_frames = 0

    print(f"[DEMO] Running {max_frames} frames simulation loop at steady-state...")
    start_loop = time.time()

    try:
        while total_frames < max_frames:
            t0 = time.time()
            pub_node._on_timer()
            total_frames += 1

            # Check if reactive brake actuated
            if pub_node.sim.ego.brake > 0.0 and not brake_triggered:
                brake_triggered = True
                brake_frame = total_frames
                print(
                    f"  >>> [FRAME {total_frames}] REACTIVE BRAKE TRIGGERED! (Ego Speed: {pub_node.sim.ego.speed_mps*3.6:.1f} km/h)"
                )

            if pub_node.sim.ego.is_stopped and brake_triggered:
                stopped_frames += 1
                if stopped_frames >= 10:
                    print(
                        f"  >>> [FRAME {total_frames}] Ego vehicle safely brought to full stop in front of obstacle!"
                    )
                    break

            # Maintain max 20 FPS rate
            proc_dt = time.time() - t0
            sleep_dt = max(0.0, 0.05 - proc_dt)
            if sleep_dt > 0:
                time.sleep(sleep_dt)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        if recorder is not None:
            recorder.stop()
            print(f"[ROSBAG] Saved {recorder.count} messages to {bag_path}")
        viz.destroy_node()
        pub_node.destroy_node()
        perc_node.destroy_node()

    elapsed = time.time() - start_loop
    avg_fps = total_frames / max(0.001, elapsed)

    results = {
        "scenario": scenario,
        "weather": weather,
        "total_frames": total_frames,
        "elapsed_sec": elapsed,
        "avg_fps": avg_fps,
        "brake_triggered": brake_triggered,
        "brake_frame": brake_frame,
        "video_path": video_path,
        "bag_path": bag_path if record_bag else None,
    }

    print("\n=======================================================")
    print(" EXECUTION COMPLETE SUMMARY")
    print(f" Frames Processed: {total_frames}")
    print(f" Average Perception Throughput: {avg_fps:.1f} FPS (Target: >= 15 FPS)")
    print(f" Emergency Braking Triggered: {'YES' if brake_triggered else 'NO'}")
    print(f" Video Output: {video_path}")
    print("=======================================================\n")

    return results


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Driving Perception Demo")
    parser.add_argument(
        "--scenario",
        default="pedestrian",
        help="Scenario name (pedestrian, lead_car, night, rain)",
    )
    parser.add_argument(
        "--weather", default="rain", help="Weather condition (clear, rain, night)"
    )
    parser.add_argument(
        "--backend", default="pytorch", help="Model backend (pytorch, onnx, tensorrt)"
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Duration in seconds"
    )
    parser.add_argument(
        "--show-window", action="store_true", help="Show live OpenCV window"
    )
    parser.add_argument(
        "--bag",
        default="data/sample/rosbags/rosbag_pedestrian_crossing.bag",
        help="Rosbag output path",
    )
    parser.add_argument(
        "--video", default="docs/results/demo_overlay.mp4", help="Video output path"
    )

    # Support ROS2 style arguments: scenario:=value weather:=value
    argv = sys.argv[1:]
    clean_argv = []
    for arg in argv:
        if ":=" in arg:
            k, v = arg.split(":=", 1)
            k = k.lstrip("-")
            clean_argv.extend([f"--{k}", v])
        else:
            clean_argv.append(arg)

    return parser.parse_args(clean_argv)


if __name__ == "__main__":
    args = parse_cli_args()
    run_pipeline(
        scenario=args.scenario,
        weather=args.weather,
        backend=args.backend,
        duration_sec=args.duration,
        show_window=args.show_window,
        bag_path=args.bag,
        video_path=args.video,
    )
