# DECISIONS.md — Architecture & Design Decisions

Each entry records: **what** was decided, **alternatives** considered, and **why** this choice was made.

---

## DEC-001: YOLOv8m as primary detector
- **Date**: 2026-10-29
- **Decision**: Start with YOLOv8m (medium) from Ultralytics.
- **Alternatives**: YOLOv8n (nano — faster, less accurate), YOLOv8l/x (larger — better accuracy but slower), YOLOv11 (newest but less community support), Faster R-CNN (two-stage, heavier).
- **Rationale**: YOLOv8m balances speed and accuracy well for real-time driving. Ultralytics has excellent ONNX/TRT export support. If accuracy is insufficient, we can upgrade to v8l. If speed is the bottleneck, we can drop to v8s. YOLOv11 is available as a stretch goal.
- **Update (2026-10-29)**: Switched default to **YOLOv8s** due to RTX 3050 (4 GB VRAM). v8m may OOM at batch_size=4. Will try v8m on Colab T4 if v8s accuracy is insufficient.

## DEC-007: RTX 3050 (4 GB) — GPU strategy
- **Date**: 2026-10-29
- **Decision**: Local training with batch_size=4 on RTX 3050. Colab/Kaggle T4 as fallback for larger batches or bigger models.
- **Alternatives**: Only use cloud GPUs (slower iteration), buy more VRAM (not realistic for a student).
- **Rationale**: 4 GB is tight but workable with YOLOv8s at batch=4, imgsz=640. TensorRT FP16 halves memory at inference. CARLA may need to run on Windows-side while perception runs in WSL2. If CARLA + pipeline doesn't fit in 4 GB simultaneously, we fall back to rosbag replay.

## DEC-002: BDD100K 10K subset as primary dataset
- **Date**: 2026-10-29
- **Decision**: Use BDD100K 10K-image subset for training/validation.
- **Alternatives**: Full BDD100K (100K images — slow download, long training), KITTI (smaller, different label format, less diverse), nuScenes (3D focus, overkill), COCO (not driving-specific).
- **Rationale**: BDD100K is the gold standard for driving perception: diverse weather, lighting, and scenes. The 10K subset is manageable on a single GPU in ~2 hours. KITTI is our fallback.

## DEC-003: ByteTrack as primary tracker, DeepSORT for comparison
- **Date**: 2026-10-29
- **Decision**: ByteTrack primary, DeepSORT comparison.
- **Alternatives**: SORT (no re-ID), StrongSORT (heavier), OC-SORT, BoT-SORT.
- **Rationale**: ByteTrack matches unmatched low-confidence detections in a second round, reducing ID switches. It's lightweight (no re-ID network needed unlike DeepSORT). Comparing with DeepSORT shows we understand the trade-off between appearance features and motion-only tracking.

## DEC-004: MiDaS small for monocular depth
- **Date**: 2026-10-29
- **Decision**: MiDaS v2.1 small (or DPT-Small via MiDaS v3).
- **Alternatives**: MiDaS large/hybrid (more accurate, slower), Depth Anything (newer, larger), ZoeDepth (metric-aware but heavier), stereo camera in CARLA (changes sensor setup).
- **Rationale**: MiDaS small runs at ~30+ FPS on a mid-range GPU, fast enough for real-time. It produces relative depth which we calibrate to metric using known object sizes or CARLA ground truth. The "small" variant keeps the pipeline lightweight.

## DEC-005: Classical + segmentation dual approach for lanes
- **Date**: 2026-10-29
- **Decision**: Build both a classical OpenCV pipeline and a learned segmentation model, then compare.
- **Alternatives**: Only classical (fragile), only learned (black box), LaneATT/CLRNet (SOTA but complex).
- **Rationale**: The brief requires comparing classical and ML approaches. The classical pipeline (Canny → Hough → RANSAC/polyfit) is interpretable and works in clear conditions. A segmentation model handles curves, occlusions, and varied lighting better. Showing both demonstrates understanding of the trade-offs.

## DEC-006: Project structure — monorepo with ROS2 workspace
- **Date**: 2026-10-29
- **Decision**: Single repo with `src/` for Python ML code and `ros2_ws/` for ROS2 packages.
- **Alternatives**: Separate repos for ML and ROS2, using git submodules.
- **Rationale**: Simpler for a capstone project. One clone, one set of docs, clear dependency chain. The ROS2 workspace is nested but self-contained.
