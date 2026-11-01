# API Reference — Autonomous Driving Perception Stack

This document provides a comprehensive specification of ROS2 topics, custom message schemas, node parameters, configuration files, and Python library APIs.

---

## 1. ROS2 Topics

| Topic | Message Type | Rate | Description |
| :--- | :--- | :---: | :--- |
| `/camera/image_raw` | `sensor_msgs/Image` | 20 Hz | Uncompressed 1280x720 RGB camera feed from front bumper. |
| `/camera/depth_gt` | `sensor_msgs/Image` | 20 Hz | CARLA 24-bit logarithmic ground truth depth (evaluation only). |
| `/perception/objects` | `perception_msgs/TrackedObjectArray` | 20 Hz | Array of active tracks with 2D bboxes, metric distance, and TTC. |
| `/perception/overlay` | `sensor_msgs/Image` | 20 Hz | Rendered perception HUD overlay (boxes, IDs, distances, lane mask). |
| `/vehicle/control` | `perception_msgs/VehicleControl` | 20 Hz | Actuation command (throttle, brake, steer, emergency AEB flag). |

---

## 2. ROS2 Message Definitions (`perception_msgs`)

### `TrackedObject.msg`
Single tracked obstacle detected in vehicle coordinate frame:
```ros2
uint32 id              # Persistent track identifier
string class_name      # Class label ("car", "pedestrian", "truck", etc.)
float32 confidence     # Detection confidence score [0.0, 1.0]
int32[4] bbox          # 2D bounding box [x1, y1, x2, y2] in pixels
float32 distance_m     # Estimated distance in meters along z-axis
float32 ttc_s          # Estimated Time-To-Collision in seconds (999.0 if stable/receding)
```

### `TrackedObjectArray.msg`
Header and container for all active tracked objects:
```ros2
std_msgs/Header header
TrackedObject[] objects
float32 fps
```

---

## 3. ROS2 Parameters

The perception nodes expose configurable parameters through ROS2 parameter management:

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `model_backend` | `string` | `"pytorch"` | Inference backend: `"pytorch"`, `"onnx"`, or `"tensorrt"`. |
| `device` | `string` | `"cuda"` | Compute target: `"cuda"`, `"cuda:0"`, or `"cpu"`. |
| `conf_threshold` | `float` | `0.15` | Minimum detection confidence threshold for YOLOv8. |
| `lane_method` | `string` | `"classical"` | Lane detection algorithm: `"classical"` (sliding window) or `"unet"`. |
| `ttc_threshold_s` | `float` | `2.5` | Critical Time-to-Collision threshold triggering emergency braking. |
| `brake_distance_m`| `float` | `8.0` | Absolute distance threshold triggering emergency braking. |
| `corridor_width_m`| `float` | `1.4` | Half-width of safety corridor centered along vehicle longitudinal axis. |

---

## 4. Configuration Files (`configs/`)

### `configs/camera_intrinsics.json`
Specifies optical and geometric parameters for CARLA and BDD100K camera models:
```json
{
  "width": 1280,
  "height": 720,
  "fov_deg": 90.0,
  "fx": 640.0,
  "fy": 640.0,
  "cx": 640.0,
  "cy": 360.0
}
```

### `configs/distance_calibration.json`
Inverse-depth linear calibration parameters relating MiDaS relative output to metric meters:
```json
{
  "scale_a": 0.142228,
  "shift_b": 0.051412,
  "method": "inverse_linear",
  "mae_overall_m": 1.42,
  "dataset": "carla_town10_varied"
}
```

### `configs/yolo_bdd100k.yaml`
Dataset mapping for YOLOv8s training and inference on BDD100K:
```yaml
path: data/bdd100k
train: images/train
val: images/val
names:
  0: car
  1: truck
  2: bus
  3: person
  4: rider
  5: bike
  6: motorcycle
  7: traffic light
  8: traffic sign
```

---

## 5. Python Core Library API

### `src.tracking.tracker.Tracker`
Unified object tracking interface supporting both ByteTrack and DeepSORT:
```python
from src.tracking.tracker import Tracker, Track

tracker = Tracker(
    tracker_type="bytetrack",  # "bytetrack" or "deepsort"
    conf_thresh=0.15,
    backend="tensorrt",         # "tensorrt", "onnx", or "pytorch"
    device="0"                  # "0" for GPU, "cpu" for CPU
)

tracks: list[Track] = tracker.update(frame)
# Returns list of Track(id=int, cls=str, bbox=tuple[int, int, int, int], conf=float)
```

### `src.lanes.detector.LaneDetector`
Unified lane boundary detector returning bird's-eye polynomial overlays:
```python
from src.lanes.detector import LaneDetector, LaneResult

detector = LaneDetector(method="classical")  # "classical" or "unet"
result: LaneResult = detector.detect(frame)
# Properties:
# result.polylines: dict[str, np.ndarray] (left and right lane coordinates)
# result.mask: np.ndarray (binary lane mask)
# result.curvature_m: float (radius of road curvature in meters)
# result.offset_m: float (vehicle lateral offset from lane center in meters)
```

### `src.depth.estimator.DepthEstimator`
Monocular relative depth inference with GPU TensorRT acceleration:
```python
from src.depth.estimator import DepthEstimator

estimator = DepthEstimator(device="cuda", backend="tensorrt")
relative_depth_map: np.ndarray = estimator.predict(frame)
# Returns 2D float32 depth map matching input height/width
```

### `src.depth.distance.calculate_object_distance`
Calculates calibrated metric distance from relative depth over lower-third object ROI:
```python
from src.depth.distance import calculate_object_distance

distance_m: float = calculate_object_distance(
    depth_map=relative_depth_map,
    bbox=(x1, y1, x2, y2),
    scale_a=0.142228,
    shift_b=0.051412
)
```
