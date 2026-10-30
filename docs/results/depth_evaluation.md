# Monocular Depth Estimation & Distance Calibration Benchmark

**Autonomous Driving Perception Stack — Phase 4**

This benchmark evaluates **MiDaS Small** relative depth estimation calibrated to metric distance using the CARLA depth camera sensor formulation ($Z_{true} = d_{norm} \times 1000$ m) across standard automotive distance bands.

---

## 1. Metric Depth Calibration Formulation

MiDaS estimates relative inverse depth (disparity) $d_{rel}$, which is proportional to $1 / Z$. Metric conversion is performed via an inverse-depth linear formulation:

$$Z_{metric} = \frac{1.0}{a \cdot d_{rel} + b}$$

| Parameter | Calibrated Value | Physical Meaning |
|---|---|---|
| **$a$ (scale)** | `0.142228` | Focal length & baseline scaling factor |
| **$b$ (shift)** | `0.004895` | Camera sensor disparity offset |
| **Fitted $R^2$** | `0.9979` | Goodness of fit across 650 calibration pairs |
| **Calibration Config** | [`configs/depth_calibration.yaml`](file:///d:/Autonomous-Driving-Perception-Stack/configs/depth_calibration.yaml) | Persistent project calibration file |

---

## 2. Quantitative Error Analysis by Distance Band

Cross-checking **Calibrated MiDaS** and **Pinhole Camera Geometry** ($Z = f \cdot H_{real} / h_{pixels}$) against true ground truth:

| Distance Band | Evaluation Range | Sample Count | Calibrated MiDaS MAE | Calibrated MiDaS Rel. Error | Pinhole Formula MAE | Pinhole Rel. Error |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Near Range (Critical)** | **0 – 10 m** | 210 | **0.14 m** | **2.2%** | 0.05 m | 0.8% |
| **Mid Range (Following)** | **10 – 30 m** | 240 | **0.42 m** | **2.1%** | 0.54 m | 2.5% |
| **Far Range (Lookahead)** | **30+ m** | 150 | **1.32 m** | **2.3%** | 4.40 m | 7.3% |
| **Overall** | **2 – 80 m** | 600 | **0.55 m** | **2.2%** | 1.33 m | 3.1% |

---

## 3. Engineering Insights & Robustness

1. **Lower-Third Sampling Rationale:**
   - Instead of averaging over the entire bounding box, taking the **median over the lower third** isolates the object's contact zone with the road surface.
   - This eliminates background sky bleed and window transparency artifacts that otherwise skew depth estimates for sedans and buses.
2. **Near-Range Safety Precision:**
   - In the safety-critical **0 – 10 m** zone, calibrated MiDaS achieves **0.14 m MAE** and **2.2% relative error**, providing reliable inputs for Autonomous Emergency Braking (AEB).
3. **Pinhole vs. Depth Fusion:**
   - The pinhole formula provides an independent geometric cross-check when camera intrinsics ($f = 640$ px) and standard vehicle heights ($H_{real} = 1.5$ m) are known.
   - Dual estimation allows cross-validation: if $|Z_{midas} - Z_{pinhole}| > 3.0$ m, the system can flag occluded or truncated bounding boxes.
4. **Time-to-Collision (TTC) Filtering:**
   - Raw frame-to-frame velocity derivatives are noisy due to discretization. Smoothing distances over a moving average window ($K=5$) stabilizes approach speed $\dot{Z}$ and eliminates false positive collision warnings.

---

## 4. Visual Comparison Artifacts

The side-by-side visualization displaying **Object Distance & TTC Tracking** alongside the **MiDaS Relative Depth Map** is saved at:
[`docs/results/depth/depth_comparison_grid.png`](file:///d:/Autonomous-Driving-Perception-Stack/docs/results/depth/depth_comparison_grid.png)

Calibration configuration saved at:
[`configs/depth_calibration.yaml`](file:///d:/Autonomous-Driving-Perception-Stack/configs/depth_calibration.yaml)
