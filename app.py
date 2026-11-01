"""Autonomous Driving Perception Stack — Interactive Gradio Web Demo.

Ready for local execution and one-click deployment to Hugging Face Spaces.
Accepts driving videos (or built-in sample clips) and generates an annotated
perception stream with bounding boxes, persistent IDs, metric distances,
lane overlays, bird's-eye radar view, and emergency collision warnings.
"""

from __future__ import annotations
import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Ensure root is on python path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.depth.calibration import load_calibration_params
from src.depth.distance import TTCTracker, calculate_object_distance
from src.depth.estimator import DepthEstimator
from src.lane.detector import LaneDetector
from src.tracking.tracker import Track, Tracker

SAMPLE_DIR = ROOT / "data" / "sample" / "videos"
SAMPLE_CLIPS = [
    str(SAMPLE_DIR / "clip1_highway_day.mp4"),
    str(SAMPLE_DIR / "clip2_city_crossing.mp4"),
    str(SAMPLE_DIR / "clip3_night_rain.mp4"),
]
# Filter to existing clips
EXISTING_SAMPLES = [[p] for p in SAMPLE_CLIPS if os.path.exists(p)]


def draw_hud(
    frame: np.ndarray,
    tracks: List[Track],
    lane_res,
    distances: dict[int, float],
    ttcs: dict[int, float],
    fps: float,
    is_braking: bool,
) -> np.ndarray:
    """Renders real-time autonomous driving perception HUD overlay."""
    h, w, _ = frame.shape
    vis = frame.copy()

    # 1. Draw Lane Polygon Overlay if detected
    if lane_res and lane_res.mask is not None:
        lane_overlay = np.zeros_like(vis)
        lane_overlay[lane_res.mask > 0] = (0, 200, 100)
        vis = cv2.addWeighted(vis, 0.85, lane_overlay, 0.35, 0)

    # 2. Draw Tracked Bounding Boxes with Metric Distance & TTC
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        dist = distances.get(t.id, 99.0)
        ttc = ttcs.get(t.id, 999.0)

        # Danger color if TTC < 2.5s or distance < 8m in ego lane
        is_threat = ttc < 2.5 or dist < 8.0
        box_color = (0, 0, 255) if is_threat else (0, 255, 128)

        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        # Tag text: #ID Class | 14.2m | TTC: 2.1s
        tag = f"#{t.id} {t.cls} | {dist:.1f}m"
        if ttc < 90.0:
            tag += f" | TTC: {ttc:.1f}s"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            vis,
            (x1, max(0, y1 - 18)),
            (x1 + tw + 6, max(18, y1)),
            box_color,
            -1,
        )
        cv2.putText(
            vis,
            tag,
            (x1 + 3, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # 3. Bird's-Eye View (BEV) Mini-Radar Map (Top-Right)
    bev_w, bev_h = 180, 180
    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)
    cx, cy = bev_w // 2, bev_h - 20
    scale = 3.5  # pixels per meter

    # Radar range rings (10m, 20m, 30m, 40m)
    for r in [10, 20, 30, 40]:
        r_px = int(r * scale)
        cv2.circle(bev, (cx, cy), r_px, (40, 50, 60), 1)

    # Ego vehicle safety corridor (-1.4m to +1.4m)
    cv2.rectangle(
        bev,
        (cx - int(1.4 * scale), cy - int(40 * scale)),
        (cx + int(1.4 * scale), cy),
        (0, 60, 30),
        -1,
    )
    cv2.rectangle(
        bev,
        (cx - int(1.4 * scale), cy - int(40 * scale)),
        (cx + int(1.4 * scale), cy),
        (0, 180, 80),
        1,
    )

    # Ego vehicle icon
    cv2.rectangle(bev, (cx - 6, cy - 10), (cx + 6, cy + 6), (255, 180, 50), -1)

    # Project actors onto BEV
    for t in tracks:
        dist = distances.get(t.id, 99.0)
        if 0.5 < dist <= 45.0:
            box_cx = (t.bbox[0] + t.bbox[2]) / 2.0
            # Optical center x=640, fx=640 => X = (x - cx) * Z / fx
            x_m = (box_cx - (w / 2.0)) * dist / 640.0
            ax = int(cx + (x_m * scale))
            ay = int(cy - (dist * scale))
            if 0 <= ax < bev_w and 0 <= ay < bev_h:
                color = (
                    (0, 0, 255)
                    if (ttcs.get(t.id, 999.0) < 2.5 or dist < 8.0)
                    else (0, 255, 100)
                )
                cv2.circle(bev, (ax, ay), 4, color, -1)

    # Blend BEV into top-right corner
    vis[15 : 15 + bev_h, w - bev_w - 15 : w - 15] = bev

    # 4. Emergency Braking Banner if Active
    if is_braking:
        cv2.rectangle(vis, (w // 2 - 250, 20), (w // 2 + 250, 70), (0, 0, 220), -1)
        cv2.putText(
            vis,
            "EMERGENCY BRAKE ENGAGED",
            (w // 2 - 220, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 5. Telemetry Banner (Bottom Left)
    fps_text = f"PERCEPTION: {fps:.1f} FPS | TRACKS: {len(tracks)}"
    cv2.putText(
        vis,
        fps_text,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def process_video(
    video_path: str,
    tracker_type: str = "bytetrack",
    lane_method: str = "classical",
    conf_thresh: float = 0.20,
    max_frames: int = 150,
    progress=None,
) -> Tuple[str, str]:
    """Processes an uploaded or sample driving video through the perception stack."""
    if not video_path or not os.path.exists(video_path):
        return None, "Please upload a valid driving video or select a sample clip."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Failed to open video: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames

    # Initialize Modules (CPU fallback if CUDA not available)
    import torch

    device = "0" if torch.cuda.is_available() else "cpu"
    dev_str = "cuda" if torch.cuda.is_available() else "cpu"

    tracker = Tracker(
        tracker_type=tracker_type.lower(),
        conf_thresh=float(conf_thresh),
        backend="pytorch",
        device=device,
    )
    lane_detector = LaneDetector(method=lane_method.lower())
    depth_estimator = DepthEstimator(device=dev_str, backend="pytorch")
    ttc_tracker = TTCTracker(window_size=5, alpha=0.7)

    params = load_calibration_params()
    scale_a = float(params.get("scale_a", 0.142228))
    shift_b = float(params.get("shift_b", 0.051412))

    # Output video file
    out_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps_in, (w, h))

    frame_idx = 0
    t_start = time.time()
    cached_lane = None
    brake_count = 0
    total_objects_tracked = set()
    closest_distance_seen = 999.0

    while cap.isOpened() and frame_idx < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t_frame_start = time.time()

        # 1. Tracking
        tracks = tracker.update(frame)
        for t in tracks:
            total_objects_tracked.add(t.id)

        # 2. Lanes (every 2nd frame)
        if cached_lane is None or (frame_idx % 2 == 1):
            cached_lane = lane_detector.detect(frame)

        # 3. Depth
        rel_depth = depth_estimator.predict(frame)

        # 4. Metric Distances & TTC
        distances = {}
        ttcs = {}
        is_braking = False

        for t in tracks:
            d = calculate_object_distance(
                depth_map=rel_depth,
                bbox=t.bbox,
                scale_a=scale_a,
                shift_b=shift_b,
            )
            distances[t.id] = d
            if d < closest_distance_seen:
                closest_distance_seen = d

            ttc = ttc_tracker.update(t.id, d, 1.0 / max(1.0, fps_in))
            ttcs[t.id] = ttc

            # Safety corridor logic (|x| <= 1.4m)
            box_cx = (t.bbox[0] + t.bbox[2]) / 2.0
            x_m = (box_cx - (w / 2.0)) * d / 640.0
            if abs(x_m) <= 1.4 and d < 35.0:
                if ttc < 2.5 or d < 8.0:
                    is_braking = True

        if is_braking:
            brake_count += 1

        dt_frame = max(1e-4, time.time() - t_frame_start)
        inst_fps = 1.0 / dt_frame

        # Render HUD
        hud_frame = draw_hud(
            frame, tracks, cached_lane, distances, ttcs, inst_fps, is_braking
        )
        out_writer.write(hud_frame)

    cap.release()
    out_writer.release()

    total_time = max(0.1, time.time() - t_start)
    avg_fps = frame_idx / total_time

    summary = f"""### 📊 Perception Execution Summary
- **Frames Processed**: {frame_idx}
- **Average Perception Throughput**: **{avg_fps:.1f} FPS** (Device: `{dev_str.upper()}`)
- **Unique Tracked Obstacles**: {len(total_objects_tracked)} objects
- **Closest Track Distance**: **{closest_distance_seen:.1f} m**
- **Emergency Braking Events**: {brake_count} frames triggered
- **Tracking Algorithm**: `{tracker_type.upper()}` | **Lanes**: `{lane_method.upper()}`
"""
    return out_path, summary


def build_app():
    """Builds the Gradio user interface."""
    import gradio as gr

    title = "🚗 Autonomous Driving Perception Stack"
    description = """
Interactive perception pipeline for self-driving vehicles running YOLOv8s object detection, ByteTrack / DeepSORT tracking, 
sliding-window / U-Net lane detection, and MiDaS metric depth estimation with autonomous emergency braking (AEB) alerts.
"""

    with gr.Blocks(title="Autonomous Perception Stack") as demo:
        gr.Markdown(f"# {title}\n{description}")

        with gr.Row():
            with gr.Column(scale=1):
                input_video = gr.Video(
                    label="Input Driving Video (or select sample below)"
                )

                with gr.Accordion("Perception Parameters", open=True):
                    tracker_choice = gr.Radio(
                        ["bytetrack", "deepsort"],
                        value="bytetrack",
                        label="Object Tracker",
                    )
                    lane_choice = gr.Radio(
                        ["classical", "unet"],
                        value="classical",
                        label="Lane Detection Method",
                    )
                    conf_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.8,
                        value=0.20,
                        step=0.05,
                        label="Detection Confidence Threshold",
                    )
                    max_frames_slider = gr.Slider(
                        minimum=30,
                        maximum=300,
                        value=120,
                        step=30,
                        label="Max Frames to Process",
                    )

                btn_run = gr.Button("🚀 Run Perception Stack", variant="primary")

                if EXISTING_SAMPLES:
                    sample_paths = [s[0] for s in EXISTING_SAMPLES]
                    sample_dropdown = gr.Dropdown(
                        choices=sample_paths,
                        value=sample_paths[0] if sample_paths else None,
                        label="Select Built-in Sample Clip",
                    )

                    def select_sample(path: str) -> str:
                        return path

                    sample_dropdown.change(
                        select_sample, inputs=sample_dropdown, outputs=input_video
                    )

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Annotated Perception Output (HUD + BEV Radar)"
                )
                output_summary = gr.Markdown(label="Execution Telemetry")

        btn_run.click(
            fn=process_video,
            inputs=[
                input_video,
                tracker_choice,
                lane_choice,
                conf_slider,
                max_frames_slider,
            ],
            outputs=[output_video, output_summary],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
