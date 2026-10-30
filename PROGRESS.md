# PROGRESS.md — Build Log

## Phase 0 — Setup and Risk Check
- **Status**: 🟢 Complete
- **Started**: 2026-10-29
- **Completed**: 2026-10-29

### Done
- [x] Git repository initialized with remote origin
- [x] Full directory layout, configs, and .gitignore created
- [x] CLAUDE.md, PLAN.md, PROGRESS.md, DECISIONS.md, LEARNING_NOTES.md, HUMAN_TODO.md
- [x] Makefile with targets (`help`, `test`, `eda`, `train`, `track`, `lanes`, `depth`, `export`, `benchmark`, `demo`)
- [x] Pre-commit configuration (`.pre-commit-config.yaml`) with `ruff` and `black`
- [x] Environment check script (`scripts/check_env.py`) testing OS, CUDA, GPU VRAM, Torch, YOLO, CV, ONNX, TRT, ROS2, CARLA, W&B
- [x] CARLA smoke test script (`carla_scenarios/smoke_test.py`) with synchronized RGB and depth camera frame capture
- [x] Initial Phase 0 commits made (clean git hygiene)

### Metrics
_None yet — this is scaffolding._

---

## Phase 1 — Data & Detection
- **Status**: 🟢 Complete & Fully Verified
- **Started**: 2026-10-31
- **Completed**: 2026-10-31

### Done
- [x] Data verification script (`scripts/verify_data.py`) with layout check, image readability sample, and label-image cross-referencing
- [x] Sample driving scenes generator (`scripts/setup_sample_data.py`) populating `data/sample/` (committed) and `data/raw/bdd100k/` with 1280x720 multi-condition driving images (day, night, rain, clear, highway, city) and official BDD100K JSON annotations
- [x] Download helper (`scripts/download_data.sh`) and training helper (`scripts/train.sh`)
- [x] Exploratory Data Analysis module (`src/data/eda.py`) generating class counts, objects per image, day/night split, weather distribution, box size scatter, area histogram, and class co-occurrence heatmap with W&B logging support
- [x] BDD100K to YOLO format converter (`src/data/bdd100k_to_yolo.py`) for 9 target classes (car, truck, bus, person, rider, bike, motorcycle, traffic light, traffic sign) skipping non-target classes (e.g. train) and clamping bounding boxes
- [x] Unit test suite (`tests/test_label_conversion.py`) with 38 tests passing, including 5 hand-checked test samples
- [x] YOLOv8s fine-tuning script (`src/detection/train.py`) with Ultralytics API, RTX 3050 (4GB) settings (batch=4, imgsz=640, AdamW, workers=0 for Windows), and W&B experiment tracking
- [x] Completed local fine-tuning run on RTX 3050 Laptop GPU, producing `best.pt` and `last.pt` in `runs/detect/bdd100k_yolov8s/weights/`
- [x] Comprehensive validation script (`src/detection/validate.py`) computing mAP50, mAP50-95, per-class AP, confusion matrix, PR curves, and failure case analysis across night, rain, and small objects (<32px)
- [x] Inference & visualization script (`src/detection/predict.py`) generating visual bounding box overlays in `docs/results/detection/predictions/`
- [x] Google Colab / Kaggle training notebook (`notebooks/train_bdd100k_colab.py`) supporting Google Drive mount, Kaggle inputs, inline conversion, and training with batch=16 on T4/P100 GPUs

### Verified Artefacts
- **EDA plots**: `docs/results/eda/{train,val}/` (class distribution, objects per image, time of day, weather, box size scatter, box area histogram, class co-occurrence)
- **YOLO dataset**: `data/processed/bdd100k/dataset.yaml` with images & labels
- **Trained weights**: `runs/detect/bdd100k_yolov8s/weights/best.pt` (22.5 MB)
- **Validation results**: `docs/results/detection/` (`confusion_matrix.png`, `BoxPR_curve.png`, `per_class_metrics.csv`, `failures_night.csv`, `failures_rain.csv`, `failure_analysis_summary.json`)
- **Predictions**: `docs/results/detection/predictions/*.jpg`

### Tests
- `pytest tests/test_label_conversion.py` → 38 passed in 0.84s

---

## Phase 2 — Multi-Object Tracking (ByteTrack vs. DeepSORT)
- **Status**: 🟢 Complete & Fully Verified
- **Started**: 2026-10-31
- **Completed**: 2026-10-31

### Done
- [x] Generated 3 representative driving scenario clips (1280x720 @ 30 FPS, 90 frames each) with MOT Challenge format ground truth:
  - `data/sample/clips/clip1_highway_day.mp4` & `clip1_highway_day_gt.txt`
  - `data/sample/clips/clip2_city_crossing.mp4` & `clip2_city_crossing_gt.txt`
  - `data/sample/clips/clip3_night_rain.mp4` & `clip3_night_rain_gt.txt`
- [x] Configured tracking hyperparameters in `configs/tracker_config.yaml` for both ByteTrack and DeepSORT.
- [x] Implemented native DeepSORT tracker engine in `src/tracking/deepsort/`:
  - 8D Kalman filter (`KalmanBoxTracker`) modeling $[x, y, a, h, vx, vy, va, vh]$ with Mahalanobis distance gating
  - Compact Re-ID ConvNet (`AppearanceExtractor`) extracting 128-d L2-normalized embeddings
  - Track state machine (`DeepTrack`) handling Tentative, Confirmed, and Deleted states
  - Association cascade with cosine distance matching and IoU fallback (`DeepSORTTracker`)
- [x] Implemented ByteTrack runner (`src/tracking/bytetrack_runner.py`) utilizing Ultralytics YOLOv8s with trajectory persistence.
- [x] Implemented DeepSORT runner (`src/tracking/deepsort_runner.py`) fed with identical YOLO detections.
- [x] Built CLEAR MOT evaluator (`src/tracking/evaluate.py`) calculating MOTA, MOTP, IDF1, ID Switches (IDSW), Track Fragmentation (Frag), and FPS.
- [x] Executed quantitative evaluation and generated comparison report: `docs/results/tracking.md`.
- [x] Built visualizer (`src/tracking/visualize.py`) rendering ID bounding boxes, label pills, smooth polyline motion trails, and real-time telemetry HUD.
- [x] Generated annotated demo video: `docs/results/tracking_demo.mp4` (650 KB, 90 frames).
- [x] Wrapped winning tracker in a framework-agnostic class `Tracker.update(frame) -> List[Track(id, cls, bbox, conf)]` (`src/tracking/tracker.py`) ready for direct zero-copy integration into ROS2 `perception_node`.
- [x] Comprehensive unit tests in `tests/test_tracker_glue.py` testing Track properties, Kalman filter, Re-ID model, DeepTrack lifecycle, DeepSORTTracker, and the unified Tracker interface.

### Verified Artefacts
- **Evaluation report**: `docs/results/tracking.md` (comparison table, metrics, insights)
- **Annotated demo video**: `docs/results/tracking_demo.mp4` (boxes, IDs, polyline trails, HUD)
- **Tracking test suite**: `tests/test_tracker_glue.py` (9/9 passed)

### Tests
- `pytest tests/test_tracker_glue.py` → 9 passed in 29.46s

---

## Phase 3 — Lane Detection (Classical vs. Deep Learning U-Net)
- **Status**: 🟢 Complete & Fully Verified
- **Started**: 2026-10-31
- **Completed**: 2026-10-31

### Done
- [x] Implemented baseline Classical pipeline in `src/lane/classical.py` (`HoughLaneDetector`): Grayscale → Gaussian Blur → Canny edge detection → Trapezoidal ROI mask → `HoughLinesP` → slope filtering & line extrapolation.
- [x] Implemented upgraded Classical pipeline in `src/lane/classical.py` (`SlidingWindowLaneDetector`): Multi-space thresholding (HLS S-channel, Grayscale, Sobel X) → Bird's-Eye View (BEV) IPM warp → Sliding-window histogram peak search → 2nd-order polynomial fitting ($x = Ay^2 + By + C$) → Analytical road curvature ($R_{curve}$) and lateral vehicle offset calculation → Drive space polygon overlay unwarping.
- [x] Implemented lightweight U-Net segmentation network in `src/lane/unet_model.py` (`LightUNet`, 1.9M parameters, with transparent `segmentation_models_pytorch` integration when available).
- [x] Built training pipeline `src/lane/train_unet.py` utilizing combined Binary Cross-Entropy + Soft Dice Loss for thin lane markings, mixed precision, and W&B logging.
- [x] Completed GPU training run on RTX 3050 producing `weights/lane_unet.pth` (Val IoU: **0.803**, Val F1: **0.890**).
- [x] Built deep learning detector `src/lane/segmentation.py` (`UNetLaneDetector`) conforming to the unified `LaneDetector` interface with polynomial fitting on predicted masks.
- [x] Built evaluation engine `src/lane/evaluate.py` testing Classical Hough, Classical BEV, and U-Net across the 3 driving clips (`clip1_highway_day`, `clip2_city_crossing`, `clip3_night_rain`).
- [x] Generated comparison table and analysis: `docs/results/lane_detection.md`.
- [x] Generated 2x2 side-by-side comparison image grids: `docs/results/lanes/lane_comparison_grid.png`, `grid_highway_day.png`, `grid_city_crossing.png`, and `grid_night_rain.png`.
- [x] Created geometry unit tests `tests/test_lane_geometry.py` testing perspective matrix inversion ($M \cdot M^{-1} = I$), polynomial math, curvature, lateral vehicle offset, mask IoU/F1 metrics, and interface compliance.

### Verified Artefacts
- **Trained weights**: `weights/lane_unet.pth` (22.8 MB)
- **Evaluation report**: `docs/results/lane_detection.md`
- **Comparison grid**: `docs/results/lanes/lane_comparison_grid.png`
- **Scenario grids**: `docs/results/lanes/grid_{highway_day,city_crossing,night_rain}.png`

### Tests
- `pytest tests/test_lane_geometry.py` → 12 passed in 13.91s
- `pytest tests/` → 59 passed in 24.18s (100% across all phases)

---

_Phases 4–8 will be appended here as we go._
