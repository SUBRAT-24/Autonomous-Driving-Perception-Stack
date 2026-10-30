# PLAN.md — Autonomous Driving Perception Stack

## Assumptions (correct me if wrong)
| Item | Assumed | Update here |
|------|---------|-------------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) | ✅ Confirmed |
| GPU | NVIDIA RTX 3050 (4 GB VRAM) | ⚠️ Tight — see GPU notes below |
| CARLA | Not yet installed | Will install in Phase 7 |
| W&B username | TBD — creating account | See HUMAN_TODO.md |
| ROS2 | Humble (will install in WSL2) | |

> **⚠️ GPU Notes (RTX 3050 — 4 GB VRAM)**
> - Training: batch_size=4, may need Colab/Kaggle for full runs
> - Model: Start with YOLOv8s (small), try v8m if it fits
> - TensorRT FP16 is critical — halves memory, doubles throughput
> - CARLA + perception simultaneously is tight — may use rosbag replay fallback
> - MiDaS small should fit fine (~200 MB)

> **If any assumption is wrong, tell me and I'll adjust the plan.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CARLA Simulator                              │
│   (scripted scenarios: crossing, braking, rain, night)              │
│         │ RGB camera sensor attached to ego vehicle                  │
│         ▼                                                           │
│   carla_camera_publisher (ROS2 node)                                │
│         │  /camera/image_raw  (sensor_msgs/Image)                   │
└─────────┼───────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    perception_node (ROS2)                            │
│                                                                     │
│  ┌──────────────┐   ┌───────────┐   ┌───────────────┐              │
│  │ YOLO (TRT)   │──▶│ ByteTrack │──▶│ Track IDs +   │              │
│  │ detect objs  │   │ / DeepSORT│   │ classes+bboxes│              │
│  └──────────────┘   └───────────┘   └───────┬───────┘              │
│                                             │                       │
│  ┌──────────────┐                  ┌────────▼────────┐             │
│  │ Lane Detect  │                  │ MiDaS depth →   │             │
│  │ classical +  │                  │ metric distance  │             │
│  │ segmentation │                  │ per object       │             │
│  └──────┬───────┘                  └────────┬────────┘             │
│         │                                   │                       │
│         ▼                                   ▼                       │
│  /perception/lanes    /perception/objects (custom msg)               │
│  /perception/overlay  (id, class, bbox, dist_m, ttc_s)              │
└─────────────────────────────────────────────────────────────────────┘
          │                         │
          ▼                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      viz_node (ROS2)                                │
│  overlay image + FPS + lane polygon + collision warning             │
│         │                                                           │
│         ▼                                                           │
│  Screen / recorded demo video + W&B logging                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Milestone Schedule

We have **22 days** (Oct 29 → Nov 20, 2026). Below is a tight but achievable plan.

### Phase 0 — Scaffold & Environment (Day 1: Oct 29)
- [x] Init repo, .gitignore, foundational docs
- [ ] Create full directory structure with placeholder `__init__.py`
- [ ] `requirements.txt` (pinned), `pyproject.toml`
- [ ] Verify Python 3.10+, CUDA, PyTorch install

### Phase 1 — Data Exploration (Days 2–3: Oct 30–31)
- [ ] Download BDD100K 10K subset (or KITTI fallback)
- [ ] EDA notebook/script: class distribution, image stats, sample visualizations
- [ ] Convert BDD100K labels → YOLO format
- [ ] Train/val split with config YAML
- [ ] Unit tests for label conversion
- **Deliverable**: `data_config.yaml`, EDA plots, conversion script, tests passing

### Phase 2 — Object Detection (Days 4–7: Nov 1–4)
- [ ] Fine-tune YOLOv8m (or v11m) on BDD100K
- [ ] Validate: mAP50, mAP50-95, per-class AP
- [ ] W&B logging of training curves
- [ ] Predict on sample images/video, save visualizations
- [ ] Unit tests for prediction pipeline
- **Deliverable**: Trained weights, W&B run link, results table

### Phase 3 — Object Tracking (Days 8–9: Nov 5–6)
- [ ] Integrate ByteTrack with YOLO detections
- [ ] Integrate DeepSORT for comparison
- [ ] Evaluate on BDD100K validation: MOTA, IDF1, ID switches
- [ ] Side-by-side comparison table
- **Deliverable**: Tracker configs, comparison table, tracked video

### Phase 4 — Lane Detection (Days 10–12: Nov 7–9)
- [ ] Classical pipeline: grayscale → Canny → Hough → fit lines → ROI
- [ ] Segmentation model (lightweight, e.g., SCNN or ENet-SAD or YOLO-seg)
- [ ] Evaluate both: IoU on lane masks, qualitative comparison
- [ ] Unit tests for lane geometry math
- **Deliverable**: Comparison table, overlay videos, test results

### Phase 5 — Depth Estimation (Days 13–14: Nov 10–11)
- [ ] MiDaS small inference on BDD100K samples
- [ ] Calibration: relative depth → metric distance (using known object sizes or CARLA ground truth)
- [ ] Per-object distance extraction from bounding boxes
- [ ] TTC (time-to-collision) estimation
- [ ] Unit tests for distance math
- **Deliverable**: Distance overlay images, error-by-distance-band table

### Phase 6 — ONNX & TensorRT Optimization (Days 15–16: Nov 12–13)
- [ ] Export YOLO to ONNX, validate output parity
- [ ] Build TensorRT engine (FP16)
- [ ] Export MiDaS to ONNX → TensorRT
- [ ] Benchmark: PyTorch vs ONNX vs TRT (latency, FPS, mAP)
- [ ] Log all results to W&B
- **Deliverable**: Benchmark table, optimization scripts

### Phase 7 — CARLA + ROS2 Integration (Days 17–19: Nov 14–16)
- [ ] Install CARLA 0.9.15 (WSL2 or native Linux)
- [ ] Install ROS2 Humble
- [ ] `carla_camera_publisher` node: attach camera to ego vehicle, publish to `/camera/image_raw`
- [ ] `perception_node`: subscribe, run full pipeline, publish `/perception/objects` and `/perception/overlay`
- [ ] `viz_node`: visualization with overlays, FPS counter, collision warnings
- [ ] Custom `PerceptionObject.msg` and `PerceptionArray.msg`
- [ ] Launch file: one-command bringup
- [ ] Scripted CARLA scenarios (pedestrian crossing, braking, rain, night)
- **Deliverable**: Working ROS2 pipeline, launch file, scenario scripts

### Phase 8 — Demo, Docs & Polish (Days 20–22: Nov 17–20)
- [ ] Record demo video (screen capture of full pipeline in CARLA)
- [ ] Complete README with setup, architecture diagram, results tables, demo link
- [ ] `docs/architecture.md` and `docs/api.md`
- [ ] Final W&B dashboard with all experiments
- [ ] All tests passing, fresh-clone verification
- [ ] Presentation slides (outline + key figures)
- [ ] Push final version
- **Deliverable**: README, docs, demo video, presentation outline, clean repo

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CARLA won't run (GPU/driver issues) | Medium | High | Fallback: replay rosbag through ROS2 nodes |
| Training too slow on local GPU | Medium | Medium | Use Colab/Kaggle; BDD100K 10K subset |
| TensorRT build fails | Low | Medium | Keep ONNX Runtime as fallback; document in DECISIONS.md |
| BDD100K download blocked | Low | Medium | KITTI fallback; small validation subset committed |
| WSL2 GPU passthrough issues | Medium | High | Native Ubuntu dual-boot or cloud VM |

---

## Directory Tree

```
Autonomous-Driving-Perception-Stack/
├── CLAUDE.md
├── PLAN.md                    ← you are here
├── PROGRESS.md
├── DECISIONS.md
├── LEARNING_NOTES.md
├── HUMAN_TODO.md
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── Makefile
│
├── configs/
│   ├── data_config.yaml       # dataset paths, classes, splits
│   ├── train_config.yaml      # YOLO training hyperparams
│   ├── tracker_config.yaml    # ByteTrack / DeepSORT params
│   ├── lane_config.yaml       # Canny thresholds, ROI, model choice
│   ├── depth_config.yaml      # MiDaS model, calibration params
│   └── benchmark_config.yaml  # batch sizes, precision, warmup
│
├── data/
│   ├── raw/                   # (gitignored) BDD100K / KITTI
│   ├── processed/             # (gitignored) YOLO-formatted labels
│   └── sample/                # ~10 images for testing without full dataset
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── eda.py             # exploratory data analysis
│   │   ├── bdd100k_to_yolo.py # label format conversion
│   │   └── split.py           # train/val split utility
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── train.py           # YOLO training script
│   │   ├── validate.py        # mAP evaluation
│   │   └── predict.py         # inference on images/video
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── bytetrack_runner.py
│   │   ├── deepsort_runner.py
│   │   └── evaluate.py        # MOTA, IDF1, ID switches
│   │
│   ├── lane/
│   │   ├── __init__.py
│   │   ├── classical.py       # Canny + Hough + polyfit
│   │   ├── segmentation.py    # learned lane segmentation
│   │   └── evaluate.py        # IoU, qualitative comparison
│   │
│   ├── depth/
│   │   ├── __init__.py
│   │   ├── midas_runner.py    # MiDaS inference
│   │   ├── calibration.py     # relative → metric depth
│   │   └── distance.py        # per-object distance + TTC
│   │
│   └── optimize/
│       ├── __init__.py
│       ├── export_onnx.py     # YOLO & MiDaS → ONNX
│       ├── build_trt.py       # ONNX → TensorRT engine
│       └── benchmark.py       # latency/FPS/accuracy comparison
│
├── ros2_ws/
│   └── src/
│       ├── perception_msgs/           # custom .msg definitions
│       ├── carla_camera_publisher/    # CARLA → ROS2 image
│       ├── perception_node/           # full perception pipeline
│       ├── viz_node/                  # visualization overlay
│       └── perception_bringup/        # launch files
│
├── carla_scenarios/
│   ├── pedestrian_crossing.py
│   ├── braking_lead_car.py
│   ├── rain_weather.py
│   └── night_driving.py
│
├── scripts/
│   ├── download_data.sh       # one-command dataset download
│   ├── train.sh               # one-command training
│   ├── benchmark.sh           # one-command benchmarking
│   └── demo.sh                # one-command demo launch
│
├── tests/
│   ├── __init__.py
│   ├── test_label_conversion.py
│   ├── test_distance_math.py
│   ├── test_lane_geometry.py
│   └── test_tracker_glue.py
│
└── docs/
    ├── architecture.md
    ├── api.md
    └── results/               # tables, plots, video links
```

---

## Success Criteria Checklist
- [ ] `ros2 launch perception_bringup full_pipeline.launch.py` starts everything
- [ ] Detection: mAP50, mAP50-95, per-class AP logged
- [ ] Tracking: ByteTrack vs DeepSORT comparison (MOTA, IDF1, ID switches)
- [ ] Lanes: classical vs segmentation comparison (IoU, visual)
- [ ] Depth: error by distance band (0–10m, 10–30m, 30–50m, 50m+)
- [ ] Optimization: PyTorch vs ONNX vs TRT (latency, FPS, mAP retention)
- [ ] README: setup, architecture, results tables, demo video link
- [ ] All pytest tests pass
- [ ] Fresh clone → working system from README instructions alone
