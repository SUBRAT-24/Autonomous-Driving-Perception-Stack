"""
Perception Node for Autonomous Driving Perception Stack.
Subscribes to /camera/image_raw, runs YOLO detector, tracker, lane detector,
and monocular depth estimation.
Publishes /perception/objects and /perception/overlay.
Includes reactive emergency braking controller publishing to /vehicle/control.
"""

from __future__ import annotations
import os

os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs")
import os
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

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

from perception_msgs.msg import Header, TrackedObject, TrackedObjectArray
from src.tracking.tracker import Tracker, Track
from src.lane.classical import SlidingWindowLaneDetector, HoughLaneDetector
from src.depth.estimator import DepthEstimator
from src.depth.distance import (
    extract_object_depth,
    convert_relative_to_metric,
    TimeToCollisionTracker,
    compute_pinhole_distance,
)
from src.depth.calibration import load_calibration_params


class PerceptionNode(Node):
    """Integrated perception node running detection, tracking, lanes, depth, and reactive braking."""

    def __init__(
        self, node_name: str = "perception_node", backend: Optional[str] = None
    ) -> None:
        super().__init__(node_name)

        # ROS Parameters
        self.declare_parameter("model_backend", "pytorch")  # pytorch, onnx, tensorrt
        self.declare_parameter("device", "cuda")
        self.declare_parameter("conf_threshold", 0.15)
        self.declare_parameter("lane_method", "classical")
        self.declare_parameter(
            "depth_calibration_path", "configs/depth_calibration.yaml"
        )
        self.declare_parameter("ttc_threshold_s", 2.5)
        self.declare_parameter("brake_distance_m", 8.0)
        self.declare_parameter("corridor_width_m", 1.4)

        self.backend = (
            backend or str(self.get_parameter("model_backend").value or "pytorch")
        ).lower()
        self.device = str(self.get_parameter("device").value or "cuda")
        self.conf_thresh = float(self.get_parameter("conf_threshold").value or 0.15)
        self.lane_method = str(self.get_parameter("lane_method").value or "classical")
        self.ttc_thresh = float(self.get_parameter("ttc_threshold_s").value or 2.5)
        self.brake_dist = float(self.get_parameter("brake_distance_m").value or 8.0)
        self.corridor_w = float(self.get_parameter("corridor_width_m").value or 1.4)

        self.get_logger().info(
            f"Starting PerceptionNode [Backend: {self.backend.upper()}, Device: {self.device}, Lanes: {self.lane_method}]"
        )

        self.bridge = CvBridge()

        # 1. Initialize Tracker (ByteTrack wrapper with configurable backend)
        tracker_type = "bytetrack"
        dev_str = "0" if "cuda" in self.device else "cpu"
        self.tracker = Tracker(
            tracker_type=tracker_type,
            conf_thresh=self.conf_thresh,
            backend=self.backend,
            device=dev_str,
        )

        # 2. Initialize Lane Detector
        if self.lane_method in ("hough", "classical"):
            self.lane_detector = HoughLaneDetector()
        else:
            self.lane_detector = SlidingWindowLaneDetector()
        self._cached_lane_res = None
        self._cached_depth = None

        # 3. Initialize Depth Estimator & Calibration (with TensorRT acceleration)
        self.depth_estimator = DepthEstimator(
            device=self.device,
            backend=self.backend,
        )
        try:
            params = load_calibration_params()
            self.calib_scale = float(params.get("scale_a", 0.142228))
            self.calib_shift = float(params.get("shift_b", 0.004895))
        except Exception:
            self.calib_scale = 0.142228
            self.calib_shift = 0.004895
        self.ttc_tracker = TimeToCollisionTracker(smoothing_window=5, fps=20.0)

        # Camera Intrinsics (CARLA: 1280x720, 90 deg FOV -> f=640.0 px)
        self.focal_length = 640.0
        self.cx = 640.0
        self.cy = 360.0

        # ROS Publishers & Subscriptions
        self.objects_pub = self.create_publisher(
            TrackedObjectArray, "/perception/objects", 10
        )
        self.overlay_pub = self.create_publisher(RosImage, "/perception/overlay", 10)
        self.control_pub = self.create_publisher(dict, "/vehicle/control", 10)

        self.image_sub = self.create_subscription(
            RosImage, "/camera/image_raw", self._on_image_received, 10
        )

        # Performance Metrics
        self.frame_idx = 0
        self.start_time = time.time()
        self.fps_rolling = 0.0
        self.last_brake_time = 0.0
        self.is_emergency_braking = False

    def _on_image_received(self, image_msg: RosImage) -> None:
        t0 = time.time()
        self.frame_idx += 1

        # 1. Convert ROS Image to OpenCV BGR
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        # 2. Tracking (Runs object detection + data association)
        tracks: List[Track] = self.tracker.update(frame)

        # 3. Depth Estimation (alternating cadence for > 15-25 FPS throughput)
        if self._cached_depth is None or (self.frame_idx % 2 == 0):
            self._cached_depth = self.depth_estimator.predict(frame)
        rel_depth = self._cached_depth

        # 4. Lane Detection (alternating cadence)
        if self._cached_lane_res is None or (self.frame_idx % 2 == 1):
            self._cached_lane_res = self.lane_detector.detect(frame)
        lane_res = self._cached_lane_res
        _ = lane_res.polylines
        lane_mask = lane_res.mask

        # 5. Object Distance, TTC, and Corridor Tracking
        tracked_objects: List[TrackedObject] = []
        should_brake = False
        brake_reason = ""
        critical_obj_info: Optional[Dict[str, Any]] = None

        for trk in tracks:
            bbox = trk.bbox  # [x1, y1, x2, y2]
            # Lower-third median depth
            obj_rel_depth = extract_object_depth(rel_depth, bbox)
            dist_m = convert_relative_to_metric(
                obj_rel_depth, scale_a=self.calib_scale, shift_b=self.calib_shift
            )
            dist_m = max(1.5, min(80.0, dist_m))

            # TTC Update
            ttc_s, alert_level = self.ttc_tracker.update(trk.id, dist_m, timestamp=t0)

            # Lateral position: X = (u_c - cx) * Z / f
            u_c = (bbox[0] + bbox[2]) / 2.0
            x_lateral = ((u_c - self.cx) / self.focal_length) * dist_m

            # Pinhole cross check for reference
            pinhole_dist = compute_pinhole_distance(
                trk.cls, tuple(bbox), self.focal_length
            )

            # Cross-check fusion with pinhole geometry
            if dist_m > 30.0 and pinhole_dist < 25.0:
                dist_m = pinhole_dist
            else:
                dist_m = 0.5 * dist_m + 0.5 * pinhole_dist
            dist_m = max(1.5, min(80.0, dist_m))

            # Recalculate lateral position with fused distance
            x_lateral = ((u_c - self.cx) / self.focal_length) * dist_m

            # Check if obstacle is inside ego corridor (physical lateral distance or camera central corridor)
            in_corridor = (abs(x_lateral) <= self.corridor_w) or (
                480.0 <= u_c <= 800.0 and dist_m < 22.0
            )

            if in_corridor:
                if (
                    (ttc_s is not None and ttc_s < self.ttc_thresh)
                    or (dist_m < self.brake_dist)
                    or (pinhole_dist < self.brake_dist)
                ):
                    should_brake = True
                    brake_reason = f"Object #{trk.id} ({trk.cls}) at {dist_m:.1f}m, TTC={ttc_s if ttc_s else 99:.1f}s"
                    critical_obj_info = {
                        "id": trk.id,
                        "class": trk.cls,
                        "dist": dist_m,
                        "ttc": ttc_s,
                        "x": x_lateral,
                    }

            obj_msg = TrackedObject(
                id=int(trk.id),
                class_name=str(trk.cls),
                confidence=float(trk.conf),
                bbox=[float(b) for b in bbox],
                distance_m=float(dist_m),
                ttc_s=float(ttc_s) if ttc_s is not None else float("inf"),
            )
            tracked_objects.append(obj_msg)

        # 6. Publish /perception/objects
        obj_array = TrackedObjectArray(
            header=Header.now(frame_id="camera_optical_frame"),
            objects=tracked_objects,
        )
        self.objects_pub.publish(obj_array)

        # 7. Reactive Controller Actuation
        if should_brake:
            self.is_emergency_braking = True
            self.last_brake_time = t0
            # Publish emergency brake command
            self.control_pub.publish({"brake": 1.0, "throttle": 0.0, "steer": 0.0})
            if self.frame_idx % 5 == 0:
                self.get_logger().warn(f"🚨 EMERGENCY BRAKE ENGAGED! {brake_reason}")
        else:
            # Release brake if safe for at least 0.8s
            if self.is_emergency_braking and (t0 - self.last_brake_time > 0.8):
                self.is_emergency_braking = False
                self.control_pub.publish({"brake": 0.0, "throttle": 0.25, "steer": 0.0})

        # 8. Render Visual Overlay
        t_proc = time.time() - t0
        instant_fps = 1.0 / max(0.001, t_proc)
        self.fps_rolling = (
            0.9 * self.fps_rolling + 0.1 * instant_fps
            if self.fps_rolling > 0
            else instant_fps
        )

        overlay_frame = self._render_overlay(
            frame=frame,
            objects=tracked_objects,
            lane_mask=lane_mask,
            fps=self.fps_rolling,
            critical_info=critical_obj_info,
        )

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay_frame, encoding="bgr8")
        self.overlay_pub.publish(overlay_msg)

        if self.frame_idx % 40 == 0:
            self.get_logger().info(
                f"Perception Tick {self.frame_idx} | FPS: {self.fps_rolling:.1f} | "
                f"Objects: {len(tracked_objects)} | Braking: {self.is_emergency_braking}"
            )

    def _render_overlay(
        self,
        frame: np.ndarray,
        objects: List[TrackedObject],
        lane_mask: Optional[np.ndarray],
        fps: float,
        critical_info: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Renders comprehensive perception overlay HUD with boxes, distances, lanes, and warnings."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # 1. Lane Polygon Overlay
        if lane_mask is not None and np.any(lane_mask > 0):
            green_lane = np.zeros_like(overlay)
            green_lane[lane_mask > 0] = [0, 220, 0]
            overlay = cv2.addWeighted(overlay, 0.82, green_lane, 0.35, 0)

        # 2. Ego Lane Corridor Guidelines (Cyan Dashed Lines)
        # Left corridor x = -1.4m, Right corridor x = +1.4m
        corridor_poly = np.array(
            [
                [int(640 - (1.4 / 40.0) * 640), 370],
                [int(640 + (1.4 / 40.0) * 640), 370],
                [int(640 + (1.4 / 2.0) * 640), 720],
                [int(640 - (1.4 / 2.0) * 640), 720],
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            overlay, [corridor_poly], isClosed=True, color=(255, 200, 0), thickness=1
        )

        # 3. Object Bounding Boxes & Distance Tags
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            dist_str = f"{obj.distance_m:.1f}m"
            ttc_str = f"TTC: {obj.ttc_s:.1f}s" if obj.ttc_s < 90 else "TTC: --"

            # Color coding: Red if TTC < 2.5s, Yellow if TTC < 4.0s, Green otherwise
            if obj.ttc_s < 2.5:
                box_color = (0, 0, 255)  # Red
            elif obj.ttc_s < 4.0:
                box_color = (0, 215, 255)  # Amber / Yellow
            else:
                box_color = (0, 255, 0)  # Green

            # Box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

            # Label pill
            label = f"#{obj.id} {obj.class_name} | {dist_str} | {ttc_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                overlay,
                (x1, max(0, y1 - th - 6)),
                (x1 + tw + 6, max(th + 6, y1)),
                box_color,
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 3, max(th + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # 4. Small Bird's-Eye View (BEV) Radar Panel (Top-Right Inset)
        bev_w, bev_h = 160, 200
        bev = np.full((bev_h, bev_w, 3), 25, dtype=np.uint8)
        # Grid range circles (10m, 20m, 30m, 40m)
        cx_bev = bev_w // 2
        cy_bev = bev_h - 20
        for r_m in (10, 20, 30, 40):
            r_px = int(r_m * 4.0)
            cv2.circle(bev, (cx_bev, cy_bev), r_px, (60, 60, 60), 1)
            cv2.putText(
                bev,
                f"{r_m}m",
                (cx_bev + 2, cy_bev - r_px + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (120, 120, 120),
                1,
            )

        # Ego vehicle icon in BEV
        cv2.rectangle(
            bev, (cx_bev - 6, cy_bev - 12), (cx_bev + 6, cy_bev + 4), (0, 200, 255), -1
        )

        # Draw objects in BEV
        for obj in objects:
            dist = obj.distance_m
            u_c = (obj.bbox[0] + obj.bbox[2]) / 2.0
            x_lat = ((u_c - self.cx) / self.focal_length) * dist
            x_bev = int(cx_bev + x_lat * 12.0)
            y_bev = int(cy_bev - dist * 4.0)
            if 4 <= x_bev < bev_w - 4 and 4 <= y_bev < bev_h:
                b_col = (0, 0, 255) if obj.ttc_s < 2.5 else (0, 255, 0)
                cv2.circle(bev, (x_bev, y_bev), 4, b_col, -1)

        # Overlay BEV onto top-right corner
        cv2.rectangle(bev, (0, 0), (bev_w - 1, bev_h - 1), (180, 180, 180), 1)
        overlay[15 : 15 + bev_h, w - bev_w - 15 : w - 15] = bev

        # 5. Top Telemetry HUD Bar
        hud_bg = overlay[:42, :].copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
        overlay[:42, :] = cv2.addWeighted(overlay[:42, :], 0.75, hud_bg, 0.25, 0)

        # Telemetry text
        hud_text = f"PERCEPTION STACK | FPS: {fps:.1f} | BACKEND: {self.backend.upper()} | TRACKS: {len(objects)}"
        cv2.putText(
            overlay,
            hud_text,
            (20, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 6. Prominent Flashing Collision Warning Banner (if TTC < 2.5s)
        if critical_info is not None:
            # Flashing banner
            flash = int(time.time() * 4) % 2 == 0
            banner_col = (0, 0, 230) if flash else (0, 50, 180)
            cv2.rectangle(
                overlay, (w // 2 - 280, 50), (w // 2 + 280, 105), banner_col, -1
            )
            cv2.rectangle(
                overlay, (w // 2 - 280, 50), (w // 2 + 280, 105), (255, 255, 255), 2
            )
            warn_msg = f"EMERGENCY BRAKE! TTC: {critical_info['ttc']:.1f}s | DIST: {critical_info['dist']:.1f}m"
            cv2.putText(
                overlay,
                warn_msg,
                (w // 2 - 260, 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return overlay


def main(args: list[str] | None = None) -> None:
    if ROS2_NATIVE:
        rclpy.init(args=args)
        node = PerceptionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        from src.ros_compat.node import spin

        node = PerceptionNode()
        spin(node)


if __name__ == "__main__":
    main(sys.argv)
