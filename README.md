# 🚗 Autonomous Driving Perception Stack

> Real-time vision pipeline for self-driving: object detection, tracking, lane detection, and depth estimation — running live in CARLA through ROS2.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO](https://img.shields.io/badge/YOLO-v8-green.svg)](https://docs.ultralytics.com/)
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange.svg)](https://carla.org/)

## 📋 Overview

This project implements a complete perception stack for autonomous driving as a B.Tech capstone project. It processes RGB camera images from the CARLA simulator through a modular ROS2 pipeline that performs:

1. **Object Detection** — YOLOv8m fine-tuned on BDD100K (vehicles, pedestrians, cyclists, traffic signs)
2. **Object Tracking** — ByteTrack (primary) with DeepSORT comparison (persistent IDs across frames)
3. **Lane Detection** — Classical OpenCV pipeline + learned segmentation model (with comparison)
4. **Depth Estimation** — MiDaS monocular depth → metric distance per tracked object + TTC warnings
5. **TensorRT Optimization** — ONNX export → TensorRT FP16 for real-time inference

## 🏗️ Architecture

```
CARLA RGB Camera
       │
       ▼
carla_camera_publisher ──► /camera/image_raw
       │
       ▼
perception_node
  ├── YOLO (TRT) → ByteTrack → Track IDs + Classes + BBoxes
  ├── Lane Detection (Classical + Segmentation)
  └── MiDaS Depth → Metric Distance + TTC
       │
       ├──► /perception/objects (custom msg)
       └──► /perception/overlay (annotated image)
              │
              ▼
         viz_node (FPS, lanes, collision warnings)
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/SUBRAT-24/Autonomous-Driving-Perception-Stack.git
cd Autonomous-Driving-Perception-Stack

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/WSL
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Download BDD100K dataset (see HUMAN_TODO.md for instructions)

# Run EDA
make eda

# Train YOLO
make train

# Launch full pipeline (requires CARLA + ROS2)
make demo
```

## 📊 Results

> _Results will be populated after each phase. All metrics are from actual runs logged in W&B._

| Metric | Value | Notes |
|--------|-------|-------|
| Detection mAP50 | — | YOLOv8m on BDD100K val |
| Detection mAP50-95 | — | |
| ByteTrack MOTA | — | |
| DeepSORT MOTA | — | |
| Lane IoU (Classical) | — | |
| Lane IoU (Segmentation) | — | |
| Depth MAE (0-30m) | — | Meters |
| FPS (PyTorch) | — | |
| FPS (TensorRT FP16) | — | |

## 📁 Repository Structure

```
├── configs/          # YAML configuration files
├── data/             # Datasets (gitignored)
├── src/
│   ├── data/         # EDA and label conversion
│   ├── detection/    # YOLO training and inference
│   ├── tracking/     # ByteTrack and DeepSORT
│   ├── lane/         # Classical and segmentation lane detection
│   ├── depth/        # MiDaS depth estimation
│   └── optimize/     # ONNX/TensorRT export and benchmarking
├── ros2_ws/          # ROS2 workspace
├── carla_scenarios/  # Scripted CARLA test scenarios
├── scripts/          # One-command entry points
├── tests/            # pytest unit tests
└── docs/             # Architecture, API docs, results
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection | YOLOv8m (Ultralytics) |
| Tracking | ByteTrack, DeepSORT |
| Lane Detection | OpenCV, Segmentation Model |
| Depth | MiDaS Small |
| Optimization | ONNX, TensorRT FP16 |
| Simulator | CARLA 0.9.15 |
| Middleware | ROS2 Humble |
| Experiment Tracking | Weights & Biases |
| Dataset | BDD100K |

## 📹 Demo Video

> _Link will be added after Phase 8._

## 📄 Documentation

- [Architecture](docs/architecture.md)
- [API Documentation](docs/api.md)
- [PLAN.md](PLAN.md) — Project milestones
- [DECISIONS.md](DECISIONS.md) — Design decisions with rationale
- [LEARNING_NOTES.md](LEARNING_NOTES.md) — Component explanations for presentation

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

## 👤 Author

**Subrat** — B.Tech Computer Science, Final Year
