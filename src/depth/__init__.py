"""Monocular depth estimation and metric distance calculation."""

from src.depth.calibration import (
    CalibrationResult,
    carla_depth_to_meters,
    fit_inverse_depth_linear,
    fit_inverse_depth_polynomial,
    generate_carla_depth_pairs,
    load_calibration_params,
    run_depth_calibration,
)
from src.depth.distance import (
    CARLA_FOCAL_LENGTH_PX,
    REFERENCE_OBJECT_HEIGHTS,
    TimeToCollisionTracker,
    calculate_band_errors,
    compute_pinhole_distance,
    convert_relative_to_metric,
    extract_object_depth,
)
from src.depth.estimator import DepthEstimator, MiDaSSmallNet
from src.depth.visualize import colorize_depth_map, draw_distance_and_ttc_overlay

__all__ = [
    "DepthEstimator",
    "MiDaSSmallNet",
    "extract_object_depth",
    "convert_relative_to_metric",
    "compute_pinhole_distance",
    "TimeToCollisionTracker",
    "calculate_band_errors",
    "CARLA_FOCAL_LENGTH_PX",
    "REFERENCE_OBJECT_HEIGHTS",
    "CalibrationResult",
    "carla_depth_to_meters",
    "fit_inverse_depth_linear",
    "fit_inverse_depth_polynomial",
    "generate_carla_depth_pairs",
    "load_calibration_params",
    "run_depth_calibration",
    "colorize_depth_map",
    "draw_distance_and_ttc_overlay",
]
