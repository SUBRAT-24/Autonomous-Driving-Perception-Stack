"""
Scenario: Lead Car Braking.
A lead vehicle cruising ahead suddenly applies hard brakes.
Ego perception tracks distance drop and TTC, triggering emergency braking.
"""

from __future__ import annotations
import argparse
from carla_scenarios.scenario_base import ScenarioSimulation


def run_scenario(duration_sec: float = 6.0, fps: float = 20.0) -> ScenarioSimulation:
    sim = ScenarioSimulation(scenario_name="lead_car", weather="clear")
    total_frames = int(duration_sec * fps)
    print(f"[SCENARIO] Starting Lead Car Braking Scenario ({total_frames} frames)...")

    for i in range(total_frames):
        rgb, depth_gt, telemetry = sim.step()
        for actor in telemetry["actors"]:
            if actor["class"] == "car":
                dist = actor["z"]
                speed = telemetry["ego_speed_kmh"] / 3.6
                ttc = dist / max(0.1, speed) if speed > 0.1 else float("inf")
                if abs(actor["x"]) < 1.5 and (ttc < 2.5 or dist < 8.0):
                    sim.apply_control(brake=1.0, throttle=0.0)
                    print(
                        f"  [FRAME {i}] ⚠️ Emergency Brake Activated! Lead car at {dist:.2f}m, TTC={ttc:.2f}s"
                    )
                    break

        if telemetry["ego_stopped"]:
            print(
                f"  [FRAME {i}] Ego vehicle safely stopped. Distance to lead car: {sim.actors[0].z:.2f}m"
            )
            break

    return sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lead Car Braking Scenario")
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    run_scenario(duration_sec=args.duration)
