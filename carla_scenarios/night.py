"""
Scenario: Night Driving.
Evaluates perception pipeline performance under reduced nighttime illumination with headlights.
"""

from __future__ import annotations
import argparse
from carla_scenarios.scenario_base import ScenarioSimulation


def run_scenario(duration_sec: float = 5.0, fps: float = 20.0) -> ScenarioSimulation:
    sim = ScenarioSimulation(scenario_name="night", weather="night")
    total_frames = int(duration_sec * fps)
    print(f"[SCENARIO] Starting Night Driving Scenario ({total_frames} frames)...")

    for i in range(total_frames):
        rgb, depth_gt, telemetry = sim.step()
        for actor in telemetry["actors"]:
            dist = actor["z"]
            speed = telemetry["ego_speed_kmh"] / 3.6
            ttc = dist / max(0.1, speed) if speed > 0.1 else float("inf")
            if abs(actor["x"]) < 1.5 and (ttc < 2.5 or dist < 8.0):
                sim.apply_control(brake=1.0, throttle=0.0)
                print(
                    f"  [FRAME {i}] ⚠️ Emergency Brake Activated! Obstacle at {dist:.2f}m, TTC={ttc:.2f}s"
                )
                break

        if telemetry["ego_stopped"]:
            print(f"  [FRAME {i}] Ego vehicle stopped at safe distance.")
            break

    return sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Night Driving Scenario")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    run_scenario(duration_sec=args.duration)
