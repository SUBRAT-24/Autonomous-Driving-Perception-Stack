# API Documentation

> _Detailed API reference for all modules. Will be populated as each module is implemented._

## Modules

### `src.data`
- `bdd100k_to_yolo.py` — Convert BDD100K JSON labels to YOLO format
- `eda.py` — Exploratory data analysis and visualization
- `split.py` — Train/val split utility

### `src.detection`
- `train.py` — YOLO fine-tuning on BDD100K
- `validate.py` — mAP evaluation
- `predict.py` — Inference on images/video

### `src.tracking`
- `bytetrack_runner.py` — ByteTrack integration
- `deepsort_runner.py` — DeepSORT integration
- `evaluate.py` — MOT metrics evaluation

### `src.lane`
- `classical.py` — Classical OpenCV lane detection
- `segmentation.py` — Learned lane segmentation
- `evaluate.py` — Lane detection evaluation

### `src.depth`
- `midas_runner.py` — MiDaS inference
- `calibration.py` — Depth calibration (relative → metric)
- `distance.py` — Per-object distance and TTC

### `src.optimize`
- `export_onnx.py` — ONNX export
- `build_trt.py` — TensorRT engine build
- `benchmark.py` — Performance benchmarking

## ROS2 Nodes

### `carla_camera_publisher`
_Publishes:_ `/camera/image_raw` (sensor_msgs/Image)

### `perception_node`
_Subscribes:_ `/camera/image_raw`
_Publishes:_ `/perception/objects` (PerceptionArray), `/perception/overlay` (sensor_msgs/Image)

### `viz_node`
_Subscribes:_ `/perception/objects`, `/perception/overlay`

---

_Detailed function signatures and parameters will be added as modules are implemented._
