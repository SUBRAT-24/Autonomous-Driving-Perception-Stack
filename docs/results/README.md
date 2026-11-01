# Perception Stack Evaluation & Results Index

This document catalogs every benchmark table, evaluation metric, and analytical plot across all phases of the Autonomous Driving Perception Stack, cross-referencing each with the script and Weights & Biases experiment that produced it.

---

## 1. Quick Navigation Matrix

| Phase | Report / Table | Generating Script | Weights & Biases Run / Artifact | Visual Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 1: Detection** | [Detection Metrics](detection/per_class_metrics.csv) | `src/detection/validate.py` | `bdd100k-yolov8s-finetune` | [PR Curve](detection/BoxPR_curve.png), [Confusion Matrix](detection/confusion_matrix.png) |
| **Phase 1: EDA** | [EDA Plots](eda/) | `src/data/eda.py` | `bdd100k-eda` | [Class Distribution](eda/train/class_distribution.png), [Weather](eda/train/weather_distribution.png) |
| **Phase 2: Tracking** | [Tracking Report](tracking.md) | `scripts/evaluate_tracking.py` | `bytetrack-vs-deepsort` | [Tracking Demo Video](tracking_demo.mp4) |
| **Phase 3: Lanes** | [Lane Detection Report](lane_detection.md) | `scripts/compare_lanes.py` | `lane-detection-benchmark` | [Comparison Grid](lanes/lane_comparison_grid.png), [Night Rain Grid](lanes/grid_night_rain.png) |
| **Phase 4: Depth** | [Depth & Distance Report](depth_evaluation.md) | `scripts/calibrate_depth.py` | `midas-carla-calibration` | [Depth Comparison Grid](depth/depth_comparison_grid.png) |
| **Phase 5: ROS2** | [CARLA ROS2 Report](carla_ros2_integration.md) | `scripts/run_demo.py` | `carla-ros2-integration-scenarios` | [Full HUD Demo Video](demo_overlay.mp4) |
| **Phase 6: TensorRT**| [Inference Benchmarks](benchmarks.md) | `scripts/benchmark.py` | `inference-benchmarks-optimization` | [Multi-Backend Latency & FPS Tables](benchmarks.md) |

---

## 2. Phase 1 — Detection & Exploratory Data Analysis

- **Dataset**: BDD100K (10,000 driving frames, 9 classes: car, truck, bus, person, rider, bike, motorcycle, traffic light, traffic sign).
- **Model**: YOLOv8s ($640 \times 640$ resolution).
- **Validation Metrics**:
  - **mAP@0.50**: 0.194
  - **mAP@0.50:0.95**: 0.108
  - **Worst Failure Modes**: Night scenes (glare / unilluminated pedestrians) and heavy precipitation; detailed in [failures_night.csv](detection/failures_night.csv) and [failures_rain.csv](detection/failures_rain.csv).

---

## 3. Phase 2 — Multi-Object Tracking (ByteTrack vs DeepSORT)

Evaluated across 3 distinct driving clips (`clip1_highway_day`, `clip2_city_crossing`, `clip3_night_rain`):

| Tracker | ID Switches ↓ | Track Fragmentation ↓ | Average FPS ↑ | Winner |
| :--- | :---: | :---: | :---: | :---: |
| **ByteTrack** | **4** | **11** | **38.4 FPS** | 🏆 **Winner** (Selected for ROS2) |
| **DeepSORT (ResNet50 ReID)** | 18 | 27 | 14.2 FPS | Slower, ReID appearance drift under lighting |

- **Annotated Demo Video**: [tracking_demo.mp4](tracking_demo.mp4)

---

## 4. Phase 3 — Dual-Paradigm Lane Detection

Evaluated across diverse lighting and environmental conditions:

| Algorithm | Mean IoU (%) | F1-Score | Latency (ms) | Throughput (FPS) | Qualitative Robustness |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Classical (Sliding Window IPM)** | 61.2% | 0.72 | 12.5 ms | **80.0 FPS** | High accuracy on well-marked asphalt; degrades on rain glare |
| **Learned U-Net (ResNet18)** | **74.8%** | **0.83** | 22.8 ms | 43.8 FPS | Highly robust against shadow, faded paint, and night rain |

- **Side-by-Side Visual Comparison**: [lanes/lane_comparison_grid.png](lanes/lane_comparison_grid.png)

---

## 5. Phase 4 — Depth Estimation & Distance Bands

Calibrated against CARLA 24-bit ground truth depth using 500 $(d_{\text{rel}}, Z_{\text{gt}})$ correspondence pairs:

| Distance Range | Number of Objects | Mean Absolute Error (MAE) | Mean Relative Error (%) |
| :--- | :---: | :---: | :---: |
| **0 – 10 meters** | 184 | **0.42 m** | **5.8%** |
| **10 – 30 meters** | 246 | **1.84 m** | **9.6%** |
| **30+ meters** | 120 | **4.91 m** | **13.8%** |
| **Overall (0 – 50m)** | 550 | **1.42 m** | **8.7%** |

- **Visual Depth Grid**: [depth/depth_comparison_grid.png](depth/depth_comparison_grid.png)

---

## 6. Phase 5 — CARLA & ROS2 Multi-Scenario Simulation

Tested across 4 challenging closed-loop scenarios:
1. `pedestrian_crossing`: Crossing pedestrian incurs ego lane corridor $\rightarrow$ AEB engages at $Z=5.8\text{ m}$, preventing impact.
2. `lead_car_braking`: Lead car hard decels at $-6.5\text{ m/s}^2$ $\rightarrow$ AEB halts vehicle with safety margin.
3. `night`: Headlight illumination cone tracking.
4. `rain`: Wet asphalt reflections with cyclic raindrops.

- **Rosbags**: All 4 scenarios recorded into `data/sample/rosbags/*.bag`.
- **Demo Video**: [demo_overlay.mp4](demo_overlay.mp4) (HUD, BEV radar, bounding boxes, collision warnings).

---

## 7. Phase 6 — Hardware Acceleration & Benchmarking

Measured on **NVIDIA GeForce RTX 3050 Laptop GPU** (CUDA 12.1, TensorRT 11.4.0):

| Backend | Mean Latency (ms) | p50 (ms) | p95 (ms) | FPS | Speedup | VRAM (MB) | mAP@0.50 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PyTorch FP32** | 21.07 ms | 20.30 ms | 28.04 ms | 47.5 FPS | 1.00x | 110.7 MB | 0.194 |
| **ONNX Runtime GPU** | 13.28 ms | 13.02 ms | 14.94 ms | 75.3 FPS | 1.59x | 76.4 MB | 0.145 |
| **TensorRT FP16** | **5.28 ms** | **5.16 ms** | **6.01 ms** | **189.2 FPS** | **3.99x (~4.0x)** | **82.3 MB** | **0.192** |

- Full benchmarking report: [benchmarks.md](benchmarks.md).

