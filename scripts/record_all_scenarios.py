from __future__ import annotations
import os

os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs")
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
"""
Scenario Rosbag Recorder.
Records one comprehensive rosbag per scenario:
1. pedestrian crossing (rosbag_pedestrian_crossing.bag)
2. lead car braking (rosbag_lead_car_braking.bag)
3. night driving (rosbag_night.bag)
4. rain driving (rosbag_rain.bag)
"""

import os
import sys

from scripts.run_demo import run_pipeline


def record_all_scenarios() -> None:
    bag_dir = "data/sample/rosbags"
    os.makedirs(bag_dir, exist_ok=True)

    scenarios = [
        ("pedestrian", "rain", os.path.join(bag_dir, "rosbag_pedestrian_crossing.bag")),
        ("lead_car", "clear", os.path.join(bag_dir, "rosbag_lead_car_braking.bag")),
        ("night", "night", os.path.join(bag_dir, "rosbag_night.bag")),
        ("rain", "rain", os.path.join(bag_dir, "rosbag_rain.bag")),
    ]

    for sc_name, weather, bag_path in scenarios:
        print("\n=======================================================")
        print(f" RECORDING SCENARIO: {sc_name.upper()} -> {bag_path}")
        print("=======================================================")
        video_out = f"docs/results/demo_{sc_name}.mp4"
        run_pipeline(
            scenario=sc_name,
            weather=weather,
            backend="pytorch",
            duration_sec=3.5,
            show_window=False,
            record_bag=True,
            bag_path=bag_path,
            video_path=video_out,
        )

    print("\n[COMPLETE] All 4 scenario rosbags recorded successfully!")


if __name__ == "__main__":
    record_all_scenarios()
