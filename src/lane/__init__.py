"""Lane detection package."""

from src.lane.classical import HoughLaneDetector, SlidingWindowLaneDetector
from src.lane.detector import LaneDetector, LaneResult
from src.lane.geometry import (
    calculate_curvature,
    calculate_vehicle_offset,
    compute_mask_accuracy,
    compute_mask_f1,
    compute_mask_iou,
    compute_perspective_matrices,
    evaluate_poly,
    fit_sliding_windows,
    render_lane_polygon,
    unwarp_from_bev,
    warp_to_bev,
)
from src.lane.segmentation import UNetLaneDetector
from src.lane.unet_model import LightUNet, get_lane_unet

__all__ = [
    "LaneDetector",
    "LaneResult",
    "HoughLaneDetector",
    "SlidingWindowLaneDetector",
    "UNetLaneDetector",
    "LightUNet",
    "get_lane_unet",
    "compute_perspective_matrices",
    "warp_to_bev",
    "unwarp_from_bev",
    "fit_sliding_windows",
    "calculate_curvature",
    "calculate_vehicle_offset",
    "render_lane_polygon",
    "compute_mask_iou",
    "compute_mask_f1",
    "compute_mask_accuracy",
    "evaluate_poly",
]
