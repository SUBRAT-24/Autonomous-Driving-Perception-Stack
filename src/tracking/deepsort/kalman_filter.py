"""Kalman Filter for bounding box tracking in image space.

State vector (8-dimensional):
    [x, y, a, h, vx, vy, va, vh]
where:
    - (x, y): center coordinates of bounding box
    - a: aspect ratio (w / h)
    - h: height of bounding box
    - (vx, vy, va, vh): respective velocities

Measurement vector (4-dimensional):
    [x, y, a, h]
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.linalg


class KalmanBoxTracker:
    """A Kalman filter for tracking bounding boxes in image space."""

    # Chi-square 0.95 quantile threshold table for gating
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
    }

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition matrix F (8x8)
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # Measurement projection matrix H (4x8)
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Standard deviations for position and velocity noise
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track state and covariance from initial measurement [x, y, a, h].

        Args:
            measurement: Bounding box center and aspect ratio [x, y, a, h].

        Returns:
            Initial state vector (8,) and covariance matrix (8, 8).
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman prediction step.

        Args:
            mean: Current state vector (8,).
            covariance: Current state covariance (8, 8).

        Returns:
            Predicted state vector (8,) and covariance (8, 8).
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project state distribution into measurement space.

        Args:
            mean: State vector (8,).
            covariance: State covariance (8, 8).

        Returns:
            Projected mean (4,) and covariance (4, 4).
        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
            + innovation_cov
        )
        return mean, covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman measurement update step.

        Args:
            mean: Predicted state vector (8,).
            covariance: Predicted covariance (8, 8).
            measurement: Measurement vector (4,) [x, y, a, h].

        Returns:
            Updated state vector (8,) and covariance (8, 8).
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Compute squared Mahalanobis distance between state and measurements.

        Args:
            mean: State vector (8,).
            covariance: State covariance (8, 8).
            measurements: Array of measurements (N, 4).
            only_position: If True, compute distance in 2D position space only.

        Returns:
            Array of squared Mahalanobis distances (N,).
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean = projected_mean[:2]
            projected_cov = projected_cov[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
