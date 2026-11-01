# 10-Minute Technical Presentation Outline: Autonomous Driving Perception Stack

**Target Audience**: Autonomous Vehicle Engineers, Technical Reviewers, Capstone Examination Committee.  
**Presentation Time**: 10 Minutes (~1 minute per slide) + 5 Minutes Q&A.

---

## Slide 1: Title & Executive Summary (0:00 - 1:00)
- **Title**: Real-Time Multi-Task Autonomous Driving Perception Stack
- **Subtitle**: Vision-Based 2D Detection, Tracking, Dual-Paradigm Lane Extraction, Monocular Metric Depth, and Safety-Critical Control in CARLA & ROS2
- **Speaker**: Subrat
- **Key Highlight**: A production-grade perception pipeline achieving **189 FPS (TensorRT FP16)** with integrated Automatic Emergency Braking (AEB) preventing collisions under severe weather and low-light conditions.

---

## Slide 2: Problem Statement & Motivation (1:00 - 2:00)
- **The Challenge**: Autonomous vehicles operate in complex, unpredictable dynamic environments. Processing high-resolution video streams through disjoint neural networks introduces latency bottlenecks ($> 100\text{ ms}$), catastrophic tracking ID switches, and depth ambiguity from monocular cameras.
- **Goals**:
  1. Multi-object detection & tracking across diverse driving classes (BDD100K).
  2. Redundant lane detection capable of handling curve geometry and degraded paint.
  3. Absolute metric distance and Time-to-Collision (TTC) from standard monocular RGB.
  4. Real-time inference ($> 30\text{ FPS}$) on edge compute platforms (RTX 3050 Laptop GPU).
  5. Closed-loop reactive control stopping the vehicle before impact.

---

## Slide 3: End-to-End System Architecture (2:00 - 3:00)
- **System Diagram**:
  - CARLA Simulator front bumper camera ($1280 \times 720$ @ 20 FPS, $90^\circ$ FOV).
  - ROS2 Node Pipeline (`carla_camera_publisher` $\rightarrow$ `perception_node` $\rightarrow$ `viz_node`).
  - Alternating execution cadence (20 Hz Detection/Tracking, 10 Hz Lanes, 10-20 Hz Depth) maximizing GPU efficiency.
  - Custom ROS2 interfaces (`TrackedObject.msg`, `TrackedObjectArray.msg`).
  - Safety Corridor Logic: Monitoring vehicle bounds ($\pm 1.4\text{ m}$) for proactive collision intervention.

---

## Slide 4: Real-Time Object Detection & Tracking (3:00 - 4:15)
- **Detection**: YOLOv8s fine-tuned on BDD100K across 9 autonomous driving classes.
  - Evaluation: mAP@0.50 of 0.194, mAP@0.50:0.95 of 0.108.
- **Tracking Algorithm Faceoff**: **ByteTrack vs. DeepSORT**:
  - *DeepSORT*: Uses appearance ReID (ResNet50). Suffers from appearance drift under headlight glare and heavy rain ($18$ ID switches, $14.2\text{ FPS}$).
  - *ByteTrack*: Low-confidence detection association salvaging occluded targets. Achieves only **4 ID switches** and **38.4 FPS** (**2.7x faster**).
  - *Decision*: ByteTrack selected as the primary ROS2 tracking engine.

---

## Slide 5: Dual-Paradigm Lane Detection (4:15 - 5:30)
- **Classical OpenCV Pipeline**:
  - Color filtering, Gaussian blur, Canny edges, Inverse Perspective Mapping (IPM).
  - 9-window sliding window polynomial fitting ($x = Ay^2 + By + C$).
  - Mathematical derivation of road curvature radius $R = \frac{(1 + (2Ay + B)^2)^{3/2}}{|2A|}$ and vehicle lane center offset.
  - *Performance*: **80.0 FPS** (12.5 ms latency), 61.2% IoU.
- **Learned Segmentation (U-Net ResNet18)**:
  - Deep feature segmentation trained on drivable lane masks.
  - *Performance*: **74.8% IoU** (0.83 F1-score), highly resistant to rain puddles and shadow artifacts.
- **Architectural Synergy**: Unified `LaneDetector` interface allows dynamic switching based on compute budget.

---

## Slide 6: Monocular Metric Depth & Time-to-Collision (5:30 - 6:45)
- **Relative to Metric Calibration**:
  - Monocular MiDaS yields scale- and shift-invariant relative disparity $d_{\text{rel}}$.
  - Collected 500 correspondence pairs against CARLA 24-bit ground truth depth across diverse Town10 scenes.
  - Inverse-depth linear fitting: $Z_{\text{metric}} = \frac{0.1422}{d_{\text{rel}}} + 0.0514$.
- **Object ROI Extraction**:
  - Median depth sampled exclusively across the **lower third** of object bounding boxes, isolating the road-tire contact patch.
  - Cross-validated against pinhole camera geometry: $Z = \frac{f_y \cdot H_{\text{real}}}{h_{\text{pixels}}}$.
- **Accuracy by Distance Band**:
  - $0 - 10\text{ m}$: **0.42 m MAE (5.8% error)** $\rightarrow$ Critical braking zone.
  - $10 - 30\text{ m}$: **1.84 m MAE (9.6% error)** $\rightarrow$ Adaptive cruise control zone.
- **TTC Calculation**: Moving-average smoothed range rate: $\text{TTC} = \frac{Z}{\dot{Z}}$.

---

## Slide 7: Safety-Critical Reactive Braking in CARLA (6:45 - 8:00)
- **Closed-Loop Safety Validation**:
  - Automated Emergency Braking (AEB) rule: If an obstacle enters the ego corridor ($|X| \le 1.4\text{ m}$) with $\text{TTC} < 2.5\text{ s}$ or $\text{Distance} < 8.0\text{ m}$, full braking is commanded ($\text{brake} = 1.0$, $\text{throttle} = 0.0$).
- **Scenario Testing**:
  1. *Pedestrian Crossing*: Vehicle traveling at $30.6\text{ km/h}$ detects pedestrian entering lane; brakes to a complete halt at $Z=5.8\text{ m}$ with zero impact.
  2. *Lead Vehicle Hard Braking*: Immediate detection of deceleration ($-6.5\text{ m/s}^2$); vehicle halts safely.
  3. *Adverse Conditions*: Verified in Night and Rain scenarios.
- **Rosbag Verification**: Replay integration tests verify 100% telemetry capture and message deserialization.

---

## Slide 8: Hardware Acceleration with TensorRT 11 (8:00 - 9:00)
- **Optimization Pipeline**:
  - PyTorch $\rightarrow$ ONNX (opset 17) with strict numerical parity check ($\Delta < 0.03\%$).
  - ONNX $\rightarrow$ TensorRT 11 FP16 Engine with explicit Strong Typing.
- **Benchmark Results (RTX 3050 Laptop GPU, 640x640)**:
  - **PyTorch FP32**: $21.07\text{ ms}$ (47.5 FPS) | $1.00\times$
  - **ONNX Runtime GPU**: $13.28\text{ ms}$ (75.3 FPS) | $1.59\times$
  - **TensorRT FP16**: **$5.28\text{ ms}$ (189.2 FPS) | $3.99\times$ (~4.0x Speedup)**
- **VRAM Footprint**: Reduced from 110.7 MB to 82.3 MB.
- **Parity**: mAP@0.50 preserved within $0.002$ (0.192 vs 0.194).

---

## Slide 9: Limitations & Edge Case Analysis (9:00 - 9:30)
- **Monocular Scale Ambiguity**: Extreme distances ($> 40\text{ m}$) exhibit error growth up to $13.8\%$ due to finite pixel resolution.
- **Weather Degradation**: Unlit night scenes without street lamps reduce YOLO detection recall on non-reflective objects.
- **Severe Pitch/Roll Dynamics**: Aggressive speed bumps or hill crests alter IPM homography assumptions for classical lane detection (mitigated by U-Net).

---

## Slide 10: Future Roadmap & Summary (9:30 - 10:00)
- **Deliverables Completed**:
  - Full modular ROS2 pipeline + CARLA simulation suite.
  - Interactive Web Dashboard & Hugging Face Gradio space.
  - 77 unit & integration tests passing with 100% clean linting.
- **Future Enhancements**:
  1. Multimodal sensor fusion (LiDAR point clouds + Radar Doppler velocity).
  2. 3D bounding box regression directly in vehicle coordinates.
  3. ModelOpt INT8 quantization for edge deployment on NVIDIA Jetson Orin.

