# Architecture — Autonomous Driving Perception Stack

> _Detailed architecture documentation. Diagrams and component details will be expanded as each phase is completed._

## System Overview

The perception stack processes RGB camera frames from the CARLA simulator through a modular ROS2 pipeline. Each processing stage is a separate module that can be tested independently.

## Data Flow

```
[CARLA Simulator]
       │
       │ CARLA Python API (sensor callback)
       ▼
[carla_camera_publisher]  ──────────────────────────────────────┐
       │                                                         │
       │  sensor_msgs/Image @ 30 Hz                             │
       │  Topic: /camera/image_raw                              │
       ▼                                                         │
[perception_node]                                                │
       │                                                         │
       ├─► YOLO Detection (TensorRT FP16)                       │
       │      └─► ByteTrack Tracking                            │
       │              └─► Track IDs + Classes + BBoxes           │
       │                                                         │
       ├─► Lane Detection                                        │
       │      ├─► Classical (Canny + Hough + Polyfit)           │
       │      └─► Segmentation Model                            │
       │              └─► Lane Mask + Center Line               │
       │                                                         │
       ├─► MiDaS Depth Estimation                               │
       │      └─► Per-Object Distance (meters)                  │
       │              └─► TTC Calculation                       │
       │                                                         │
       ├──► /perception/objects (PerceptionArray.msg)            │
       └──► /perception/overlay (sensor_msgs/Image)             │
              │                                                  │
              ▼                                                  │
[viz_node]                                                       │
       │                                                         │
       ├─► Annotated Display (OpenCV window)                    │
       ├─► FPS Counter                                          │
       ├─► Collision Warning Overlay                            │
       └─► W&B Logging                                          │
                                                                 │
[CARLA Scenario Controller] ◄────────────────────────────────────┘
       │
       └─► Spawns traffic, pedestrians, weather changes
```

## Component Details

_Will be expanded in each phase._

## Message Types

### PerceptionObject.msg
```
uint32 id              # Track ID
string class_name      # e.g., "car", "pedestrian"
uint8 class_id         # Numeric class ID
float32 confidence     # Detection confidence
int32[4] bbox          # [x1, y1, x2, y2] in pixels
float32 distance_m     # Estimated distance in meters
float32 ttc_s          # Time-to-collision in seconds (-1 if N/A)
```

### PerceptionArray.msg
```
std_msgs/Header header
PerceptionObject[] objects
float32 fps
```
