# Autonomous Driving Perception Stack — Demo Video Shot List

**Target Video Duration**: 2 Minutes 30 Seconds (150 Seconds).  
**Audio / Voiceover**: Technical narration or clear background audio with prominent sound effects for emergency alerts.  
**Video Resolution**: 1080p (1920x1080) @ 60 FPS / 30 FPS.

---

## Shot List & Storyboard

### Scene 1: Title & Project Overview (0:00 – 0:15)
- **Visual**: Full-screen title card with animated badges: "Autonomous Driving Perception Stack — Real-Time Vision, Tracking & Safety in CARLA/ROS2".
- **Overlay Text**: "Built on YOLOv8, ByteTrack, MiDaS Depth, IPM Lane Polynomials, and TensorRT 11".
- **Transition**: Smooth fade into the CARLA Town10 driving environment.

### Scene 2: Daytime Highway Driving & Multi-Task HUD (0:15 – 0:40)
- **Visual**: Primary front camera view driving on highway (`clip1_highway_day.mp4`).
- **Features Highlighted**:
  - Green lane corridor overlay matching road curvature ($R \approx 850\text{ m}$).
  - Bounding boxes tracking surrounding vehicles with green tags (`#102 car | 28.4m | TTC: --`).
  - Top-right Bird's-Eye View (BEV) radar map showing real-time actor positions relative to distance rings (10m, 20m, 30m, 40m).
- **On-Screen Callout**: "Real-time Multi-Task Fusion: Object Tracking + BEV Lane Curvature + Calibrated Metric Depth".

### Scene 3: Pedestrian Crossing & Reactive Emergency Braking (0:40 – 1:10)
- **Visual**: City intersection scenario (`scenario:=pedestrian`).
- **Sequence**:
  1. Vehicle cruising at $30.6\text{ km/h}$ towards crosswalk.
  2. Pedestrian steps off the curb into the roadway.
  3. Bounding box transitions from Green $\rightarrow$ Yellow $\rightarrow$ Flashing Red as TTC drops below $2.5\text{ s}$.
  4. Flashing red emergency banner appears: **"⚠️ EMERGENCY BRAKE ENGAGED | TTC: 1.4s | Distance: 5.8m"**.
  5. Ego vehicle violently decelerates ($-6.5\text{ m/s}^2$) and comes to a complete safe stop in front of the pedestrian.
- **On-Screen Callout**: "Closed-Loop Safety: Automatic Emergency Braking (AEB) prevents collision with 5.8m safety margin".

### Scene 4: Adverse Weather & Low-Light Night Scenarios (1:10 – 1:35)
- **Visual**: Split screen comparing **Night Driving** (headlight cone tracking) vs. **Heavy Rain** (wet road glare reflections).
- **Features Highlighted**:
  - Night: Headlight illumination tracking pedestrians and taillights.
  - Rain: Sliding window vs. U-Net segmentation preserving lane continuity through puddles.
- **On-Screen Callout**: "Dual-Paradigm Lane Detection: U-Net ResNet18 maintains 74.8% IoU under severe rain reflections".

### Scene 5: Hardware Acceleration & TensorRT 11 Benchmark (1:35 – 2:00)
- **Visual**: Live terminal side-by-side or graphical split comparing PyTorch FP32 vs. TensorRT FP16.
- **Metrics Displayed**:
  - PyTorch: $21.07\text{ ms}$ (47.5 FPS).
  - TensorRT FP16: **$5.28\text{ ms}$ (189.2 FPS)** $\rightarrow$ **3.99x (~4.0x) Speedup**.
  - Detection mAP@0.50 comparison showing zero accuracy loss ($0.192$ vs $0.194$).
- **On-Screen Callout**: "TensorRT 11 Optimization: 4.0x Acceleration, Zero-Copy CUDA Buffers, Sub-6ms Latency".

### Scene 6: Interactive Web & Gradio Interfaces (2:00 – 2:20)
- **Visual**: Screen capture of the interactive Web Dashboard (`web/index.html`) and Hugging Face Gradio app (`app.py`).
- **Demonstration**:
  - Selecting different scenarios from the dropdown.
  - Dragging the ego cruise speed slider.
  - Live BEV radar responding in real time.
- **On-Screen Callout**: "Interactive Web Dashboard & Gradio Spaces: Full telemetry, BEV radar, and cloud deployability".

### Scene 7: Summary & GitHub Repository Link (2:20 – 2:30)
- **Visual**: End screen featuring GitHub repository link, test pass badge (77/77 tests passed), and project credits.
- **Text**: `github.com/SUBRAT-24/Autonomous-Driving-Perception-Stack`.
