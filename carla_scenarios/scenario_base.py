"""
CARLA Scenario Base & Physics Engine.
Defines scenario parameters, ego vehicle kinematics, actor trajectories,
and frame generation for both live CARLA simulator and standalone simulation.
"""

from __future__ import annotations
from dataclasses import dataclass
import os
import numpy as np
import cv2
from typing import Any, Dict, List, Tuple


@dataclass
class ActorState:
    """State of an actor (vehicle, pedestrian) in camera coordinates (X: right, Y: down, Z: forward)."""

    actor_id: int
    class_name: str
    x: float  # metres (lateral offset from camera optical axis)
    y: float  # metres (vertical offset, 0 is camera height ~1.6m above road)
    z: float  # metres (longitudinal distance forward)
    width: float  # physical width in metres
    height: float  # physical height in metres
    length: float  # physical length in metres
    vx: float = 0.0  # lateral velocity m/s
    vz: float = 0.0  # longitudinal velocity m/s

    def get_bbox_2d(
        self, f: float = 640.0, cx: float = 640.0, cy: float = 360.0
    ) -> List[float]:
        """Projects 3D bounding box to 2D image coordinates [x1, y1, x2, y2]."""
        if self.z <= 0.1:
            return [0.0, 0.0, 0.0, 0.0]

        # Center in image
        u_c = cx + (self.x / self.z) * f
        v_c = cy + (self.y / self.z) * f

        # Pixel dimensions
        w_px = (self.width / self.z) * f
        h_px = (self.height / self.z) * f

        x1 = max(0.0, u_c - w_px / 2.0)
        y1 = max(0.0, v_c - h_px / 2.0)
        x2 = min(1280.0, u_c + w_px / 2.0)
        y2 = min(720.0, v_c + h_px / 2.0)

        return [float(x1), float(y1), float(x2), float(y2)]


@dataclass
class EgoVehicleKinematics:
    """Ego vehicle longitudinal kinematics with throttle, braking, and physics."""

    speed_mps: float = 8.33  # ~30 km/h initial speed
    position_m: float = 0.0
    throttle: float = 0.3
    brake: float = 0.0
    steer: float = 0.0
    max_accel: float = 2.5  # m/s^2
    max_brake_decel: float = 6.5  # m/s^2
    is_stopped: bool = False

    def update(self, dt: float = 0.05) -> None:
        if self.brake > 0.0:
            # Active braking
            decel = self.brake * self.max_brake_decel
            self.speed_mps = max(0.0, self.speed_mps - decel * dt)
            if self.speed_mps < 0.1:
                self.speed_mps = 0.0
                self.is_stopped = True
        else:
            # Cruising with throttle
            target_speed = 8.33 * max(0.1, self.throttle / 0.3)
            if self.speed_mps < target_speed:
                self.speed_mps = min(target_speed, self.speed_mps + self.max_accel * dt)

        self.position_m += self.speed_mps * dt


class ScenarioSimulation:
    """
    Simulates a dynamic driving scenario generating synchronized RGB + 24-bit Depth frames
    at 20 FPS with active ego vehicle reaction.
    """

    def __init__(
        self, scenario_name: str = "pedestrian", weather: str = "clear"
    ) -> None:
        self.scenario_name = scenario_name.lower()
        self.weather = weather.lower()
        self.fps = 20.0
        self.dt = 1.0 / self.fps
        self.frame_idx = 0
        self.sim_time = 0.0
        self.ego = EgoVehicleKinematics()
        self.actors: List[ActorState] = []

        # Precompute static road depth template
        h, w = 720, 1280
        self._road_depth = np.full((h, w), 80.0, dtype=np.float32)
        horizon = 360
        y_coords = np.arange(horizon, h, dtype=np.float32)
        dists = np.clip((1.6 * 640.0) / (y_coords - horizon + 1e-3), 2.0, 80.0)
        self._road_depth[horizon:, :] = dists[:, None]

        # Asset memory cache to eliminate per-frame disk I/O
        self._assets = {}
        for key in ("pedestrian", "car"):
            p = f"data/sample/assets/{key}.png"
            if os.path.exists(p):
                self._assets[key] = cv2.imread(p)

        # Precompute rain streak overlays (10 cyclic patterns)
        self._rain_patterns = []
        for seed in range(10):
            np.random.seed(seed)
            r_layer = np.zeros((h, w, 3), dtype=np.uint8)
            num_streaks = 120
            rx = np.random.randint(0, w, num_streaks)
            ry = np.random.randint(0, h - 20, num_streaks)
            for x, y in zip(rx, ry):
                cv2.line(r_layer, (x, y), (x + 2, y + 15), (200, 200, 220), 1)
            self._rain_patterns.append(r_layer)

        self._init_scenario()

    def _init_scenario(self) -> None:
        self.ego = EgoVehicleKinematics(speed_mps=8.5, throttle=0.3)
        self.actors.clear()

        if "pedestrian" in self.scenario_name:
            # Pedestrian crossing from right sidewalk (x=+3.5m) to left (x=-3.5m) at distance z=26m
            # Walking speed: -1.2 m/s lateral
            ped = ActorState(
                actor_id=101,
                class_name="person",
                x=3.2,
                y=0.2,  # centered near eye height
                z=25.0,
                width=0.6,
                height=1.75,
                length=0.4,
                vx=-1.1,  # crossing towards ego lane center (x=0)
                vz=0.0,
            )
            self.actors.append(ped)

        elif "lead_car" in self.scenario_name or "lead" in self.scenario_name:
            # Lead vehicle 22m ahead traveling at same initial speed, then brakes hard at t=1.0s
            lead = ActorState(
                actor_id=202,
                class_name="car",
                x=0.0,  # directly in ego lane
                y=0.2,
                z=22.0,
                width=1.9,
                height=1.5,
                length=4.5,
                vx=0.0,
                vz=8.0,  # initially cruising at 8 m/s
            )
            self.actors.append(lead)

        elif "night" in self.scenario_name:
            # Night scenario: lead vehicle at 28m and oncoming traffic at x=-3.5m
            lead = ActorState(
                actor_id=303,
                class_name="car",
                x=0.0,
                y=0.2,
                z=28.0,
                width=1.9,
                height=1.5,
                length=4.5,
                vx=0.0,
                vz=7.5,
            )
            self.actors.append(lead)
            self.weather = "night"

        elif "rain" in self.scenario_name:
            # Rain scenario: pedestrian on shoulder and car ahead
            car = ActorState(
                actor_id=404,
                class_name="car",
                x=0.1,
                y=0.2,
                z=24.0,
                width=1.9,
                height=1.5,
                length=4.5,
                vx=0.0,
                vz=6.5,
            )
            self.actors.append(car)
            self.weather = "rain"

        else:
            # Generic highway scenario
            car = ActorState(
                actor_id=505,
                class_name="car",
                x=0.0,
                y=0.2,
                z=26.0,
                width=1.9,
                height=1.5,
                length=4.5,
            )
            self.actors.append(car)

    def apply_control(
        self, throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0
    ) -> None:
        self.ego.throttle = float(throttle)
        self.ego.brake = float(brake)
        self.ego.steer = float(steer)

    def step(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Advances simulation by 1 tick (dt=0.05s).
        Returns:
            rgb_frame: np.ndarray (720, 1280, 3) BGR8
            depth_gt: np.ndarray (720, 1280, 3) CARLA 24-bit encoded depth
            telemetry: Dict with ego speed, braking, actor states
        """
        old_ego_pos = self.ego.position_m
        self.ego.update(self.dt)
        ego_delta = self.ego.position_m - old_ego_pos
        self.sim_time += self.dt
        self.frame_idx += 1

        # Update actors relative to ego vehicle
        for actor in self.actors:
            # Longitudinal distance decreases as ego advances
            actor.z = max(1.5, actor.z + (actor.vz * self.dt) - ego_delta)
            # Lateral position changes by actor vx
            actor.x += actor.vx * self.dt

            # Scenario specific logic
            if "lead" in self.scenario_name and self.sim_time > 1.2:
                # Lead car applies emergency brakes at t=1.2s
                actor.vz = max(0.0, actor.vz - 5.5 * self.dt)

        # Generate frames
        rgb_frame, depth_gt = self._render_frame()

        telemetry = {
            "frame_idx": self.frame_idx,
            "sim_time": self.sim_time,
            "ego_speed_kmh": self.ego.speed_mps * 3.6,
            "ego_brake": self.ego.brake,
            "ego_throttle": self.ego.throttle,
            "ego_stopped": self.ego.is_stopped,
            "actors": [
                {
                    "id": a.actor_id,
                    "class": a.class_name,
                    "x": a.x,
                    "z": a.z,
                    "bbox": a.get_bbox_2d(),
                }
                for a in self.actors
            ],
        }

        return rgb_frame, depth_gt, telemetry

    def _render_frame(self) -> Tuple[np.ndarray, np.ndarray]:
        """Renders synthetic driving scene matching CARLA camera intrinsics & depth encoding."""
        h, w = 720, 1280
        bg_path = "data/sample/images/100k/train/sample_0001_train.jpg"
        if not hasattr(self, "_base_bg") or self._base_bg is None:
            if os.path.exists(bg_path):
                self._base_bg = cv2.imread(bg_path)
            else:
                self._base_bg = np.zeros((h, w, 3), dtype=np.uint8)
                self._base_bg[:360, :] = [185, 140, 100]
                self._base_bg[360:, :] = [65, 65, 65]

        rgb = self._base_bg.copy()
        depth_m = self._road_depth.copy()

        horizon = 360
        if "night" in self.weather:
            rgb = cv2.convertScaleAbs(rgb, alpha=0.35, beta=-20)
        elif "rain" in self.weather:
            rgb = cv2.convertScaleAbs(rgb, alpha=0.85, beta=10)

        # 2. Lane markings (left and right)
        lane_pts_left = []
        lane_pts_right = []
        for y in range(horizon + 20, h, 20):
            z = (1.6 * 640.0) / (y - horizon)
            # Left lane at x = -1.8m, right lane at x = +1.8m
            u_l = int(640 + (-1.8 / z) * 640.0)
            u_r = int(640 + (1.8 / z) * 640.0)
            lane_pts_left.append((u_l, y))
            lane_pts_right.append((u_r, y))

        for (ul, y), (ur, _) in zip(lane_pts_left, lane_pts_right):
            if 0 <= ul < w:
                cv2.circle(rgb, (ul, y), 3, (255, 255, 255), -1)
            if 0 <= ur < w:
                cv2.circle(rgb, (ur, y), 3, (255, 255, 255), -1)

        # 3. Render Actors (Pedestrians, Vehicles)
        for actor in sorted(self.actors, key=lambda a: -a.z):
            bbox = actor.get_bbox_2d()
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if x2 > x1 and y2 > y1 and actor.z > 0.5:
                # Update depth map
                depth_m[y1:y2, x1:x2] = np.minimum(depth_m[y1:y2, x1:x2], actor.z)

                # Composite real photorealistic asset if available
                aw = max(10, x2 - x1)
                ah = max(20, y2 - y1)
                asset_key = "pedestrian" if actor.class_name == "person" else "car"
                asset_img = self._assets.get(asset_key)

                if asset_img is not None:
                    asset_res = cv2.resize(asset_img, (aw, ah))
                    rgb[y1:y2, x1:x2] = asset_res
                else:
                    col = (
                        (50, 60, 140) if actor.class_name == "person" else (170, 40, 40)
                    )
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), col, -1)

        # 4. Weather Overlays (Night headlights / Rain streaks)
        if "night" in self.weather:
            # Ego headlights cone
            mask = np.zeros((h, w), dtype=np.uint8)
            cone_pts = np.array(
                [[500, 360], [780, 360], [1280, 720], [0, 720]], dtype=np.int32
            )
            cv2.fillPoly(mask, [cone_pts], 255)
            # Additive illumination
            rgb = cv2.addWeighted(
                rgb, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0
            )

        elif "rain" in self.weather:
            # Fast vectorized rain overlay
            pat = self._rain_patterns[self.frame_idx % len(self._rain_patterns)]
            rgb = cv2.add(rgb, pat)

        # 5. Encode CARLA 24-bit Depth
        # Formula: d_norm = Z / 1000.0
        # R = floor(d_norm * 255)
        # G = floor((d_norm * 255 * 256) % 256)
        # B = floor((d_norm * 255 * 256 * 256) % 256)
        # In CARLA, Normalized Depth is encoded as:
        # normalized = (R + G*256 + B*65536) / (256^3 - 1)
        # depth_meters = normalized * 1000
        d_norm = np.clip(depth_m / 1000.0, 0.0, 1.0)
        int_depth = ((256**3 - 1) * d_norm).astype(np.uint32)
        depth_carla_encoded = int_depth.view(np.uint8).reshape((h, w, 4))[:, :, :3]

        return rgb, depth_carla_encoded


__all__ = ["ActorState", "EgoVehicleKinematics", "ScenarioSimulation"]
