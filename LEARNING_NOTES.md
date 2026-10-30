# LEARNING_NOTES.md — Concepts Explained for Presentation

> Plain-language explanations of every component in the pipeline.
> Use these to prepare for the 10-minute presentation and Q&A.

---

## 1. The Big Picture: What Is a Perception Stack?

A self-driving car needs to **see** the world before it can **decide** what to do. The perception stack is the "eyes" — it takes raw camera images and produces structured understanding:
- **What** objects are around? (detection)
- **Which** object is which across time? (tracking)
- **Where** are the lane lines? (lane detection)
- **How far** away is each object? (depth estimation)

Our pipeline chains these together: Camera → Detect → Track → Estimate Distance → Output structured data.

---

## 2. YOLO — You Only Look Once

**What it does**: Takes an image, outputs bounding boxes with class labels and confidence scores — all in one forward pass.

**Why "one look"**: Older detectors (R-CNN family) first propose regions, then classify each one (two stages). YOLO treats detection as a single regression problem: divide the image into a grid, and for each cell predict boxes + classes simultaneously. This makes it very fast.

**YOLOv8 specifics**:
- **Backbone**: CSPDarknet — extracts features from the image
- **Neck**: PANet (Path Aggregation) — combines features at different scales so both large and small objects are detected
- **Head**: Decoupled head — separate branches for classification and box regression (better than coupled in practice)
- **Anchor-free**: Unlike earlier YOLO versions, v8 doesn't use pre-defined anchor boxes. It directly predicts box center offsets and dimensions.

**Fine-tuning**: We start from COCO-pretrained weights and train on BDD100K driving images. This is transfer learning — the model already knows "what edges and shapes look like" from COCO, we just teach it "what cars and pedestrians look like in driving scenes."

**Key metrics**:
- **mAP50**: Mean Average Precision at IoU ≥ 0.50 (how often the box overlaps at least 50% with ground truth)
- **mAP50-95**: Average mAP across IoU thresholds from 0.50 to 0.95 (stricter, penalizes imprecise boxes)
- **Per-class AP**: How well we detect each category (cars vs pedestrians vs trucks, etc.)

---

## 3. Object Tracking — ByteTrack vs DeepSORT

**Why tracking?** Detection gives you boxes per frame, but you don't know if the car in frame 1 is the same car in frame 2. Tracking assigns persistent IDs.

### ByteTrack
- Uses **only motion** (Kalman filter predicts where each object will be in the next frame)
- Key innovation: **two-round matching**. First, match high-confidence detections to existing tracks. Then, match remaining low-confidence detections to unmatched tracks. This recovers partially occluded objects that other trackers miss.
- No re-identification network needed → very fast

### DeepSORT
- Uses **motion + appearance**. Each tracked object gets a feature vector from a small CNN (re-ID network).
- Matching considers both predicted position AND visual similarity.
- Better at re-identifying objects after full occlusion, but slower due to the extra CNN.

### Evaluation Metrics
- **MOTA** (Multiple Object Tracking Accuracy): 1 − (FN + FP + ID_switches) / GT. Higher is better.
- **IDF1**: How well predicted and ground-truth IDs match. Focuses on identity preservation.
- **ID Switches**: How often the tracker changes the ID assigned to a ground-truth object.

---

## 4. Lane Detection

### Classical Approach (OpenCV)
1. Convert to grayscale
2. Gaussian blur to reduce noise
3. Canny edge detection (finds intensity gradients)
4. Region of Interest (ROI) mask — only look at the road area
5. Hough Line Transform — finds straight lines in edge map
6. Fit polynomial or average to get lane lines
7. Overlay on original image

**Strengths**: Interpretable, no training needed, works well in clear daylight.
**Weaknesses**: Fails on curves, shadows, worn markings, rain, night.

### Segmentation Approach
A neural network labels each pixel as "lane" or "not lane." We use a lightweight architecture (e.g., ENet or a YOLO-seg variant).

**Strengths**: Handles curves, shadows, complex scenes.
**Weaknesses**: Needs training data, less interpretable, slower.

### Comparison
We evaluate both on the same images and compare IoU (Intersection over Union) of the detected lane region vs ground truth. The segmentation model should win on difficult scenes; classical should be competitive on clear highway.

---

## 5. Monocular Depth Estimation — MiDaS

**The problem**: A single camera gives a 2D image. We need to estimate how far away things are (3D) from this 2D input.

**How MiDaS works**: It's a neural network trained on millions of images with depth labels. It learns depth cues: perspective convergence, relative size, texture gradients, occlusion patterns. Given an image, it outputs a **relative depth map** — each pixel gets a value indicating depth ordering (closer = smaller value).

**Relative → Metric**: MiDaS outputs relative depth, not absolute meters. We calibrate:
- In CARLA: We have ground-truth depth from the simulator's depth sensor
- In real images: We use known object sizes (average car width ≈ 1.8m) as a reference

**Distance per object**: For each tracked bounding box, we extract the median depth value in that region and convert to meters.

**Time-to-Collision (TTC)**: If we know the distance `d` to an object and its rate of change `Δd/Δt` (from consecutive frames), then TTC ≈ d / (Δd/Δt). This triggers collision warnings.

---

## 6. ONNX & TensorRT Optimization

**Why optimize?** PyTorch is great for training but carries overhead (dynamic graph, Python GIL). For real-time inference at 30+ FPS, we need optimization.

**ONNX** (Open Neural Network Exchange):
- A standard format for neural networks
- Export from PyTorch → ONNX: captures the computation graph as a static protobuf file
- Can run via ONNX Runtime with hardware-specific optimizations

**TensorRT**:
- NVIDIA's inference optimizer for their GPUs
- Takes an ONNX model and builds an **engine** optimized for the specific GPU
- Optimizations: layer fusion (combine Conv+BN+ReLU into one kernel), FP16 precision (half the memory, nearly same accuracy), kernel auto-tuning
- Typically 2–5× faster than PyTorch

**Our benchmark**: We measure latency (ms per frame), throughput (FPS), and verify that accuracy (mAP) is preserved after optimization.

---

## 7. CARLA Simulator

**What it is**: An open-source driving simulator built on Unreal Engine. Provides photorealistic urban environments with controllable weather, traffic, and pedestrians.

**Why we use it**: Testing a perception system on a real car is dangerous and expensive. CARLA provides:
- Controllable ground truth (exact positions, depths, segmentation masks)
- Repeatable scenarios (same pedestrian crossing, same rain intensity)
- Safe failure analysis (what happens when the system misses a pedestrian?)

**Our setup**: We attach an RGB camera sensor to the ego vehicle and stream frames via a Python client to ROS2.

---

## 8. ROS2 (Robot Operating System 2)

**What it is**: Not actually an OS — it's a middleware framework for robotics. It provides:
- **Nodes**: Independent processes that do one thing (e.g., one node reads the camera, another runs detection)
- **Topics**: Named channels for message passing (e.g., `/camera/image_raw`)
- **Messages**: Typed data structures (e.g., `sensor_msgs/Image`, our custom `PerceptionObject`)
- **Launch files**: Scripts that start multiple nodes together

**Why ROS2 for this project**: It's the industry standard for autonomous vehicles (used by Autoware, Apollo-adjacent projects). It makes our pipeline modular — we can swap out the detector or tracker without touching other nodes.

**Our nodes**:
1. `carla_camera_publisher` — reads from CARLA, publishes images
2. `perception_node` — subscribes to images, runs full pipeline, publishes results
3. `viz_node` — subscribes to results, draws overlays, shows visualization

---

## 9. Weights & Biases (W&B)

**What it is**: An experiment tracking platform. Every training run, evaluation, and benchmark gets logged with:
- Hyperparameters (learning rate, batch size, epochs)
- Metrics over time (loss curves, mAP progression)
- System info (GPU utilization, memory)
- Artifacts (model weights, config files)

**Why we use it**: Reproducibility and comparison. When I say "model A got 0.72 mAP," you can click the W&B link and see the exact config, training curve, and even the specific git commit.

---

_More notes will be added as we build each component._
