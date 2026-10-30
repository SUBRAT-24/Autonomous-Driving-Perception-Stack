"""DeepSORT Multi-Object Tracking implementation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.tracking.deepsort.kalman_filter import KalmanBoxTracker
from src.tracking.deepsort.reid_model import AppearanceExtractor
from src.tracking.deepsort.track import DeepTrack


def bbox_to_z(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to Kalman measurement [center_x, center_y, aspect_ratio, height]."""
    x1, y1, x2, y2 = bbox
    w = max(1e-4, x2 - x1)
    h = max(1e-4, y2 - y1)
    x = x1 + w / 2.0
    y = y1 + h / 2.0
    a = w / h
    return np.array([x, y, a, h], dtype=np.float32)


def compute_iou_matrix(
    tracks: List[DeepTrack],
    detections_tlbr: np.ndarray,
) -> np.ndarray:
    """Compute IoU cost matrix (1 - IoU) between tracks and detections."""
    cost_matrix = np.ones((len(tracks), len(detections_tlbr)), dtype=np.float32)
    if len(tracks) == 0 or len(detections_tlbr) == 0:
        return cost_matrix

    for i, trk in enumerate(tracks):
        tx1, ty1, tx2, ty2 = trk.to_tlbr()
        t_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)

        for j, det in enumerate(detections_tlbr):
            dx1, dy1, dx2, dy2 = det[:4]
            d_area = max(0.0, dx2 - dx1) * max(0.0, dy2 - dy1)

            # Intersection
            ix1 = max(tx1, dx1)
            iy1 = max(ty1, dy1)
            ix2 = min(tx2, dx2)
            iy2 = min(ty2, dy2)
            iw = max(0.0, ix2 - ix1)
            ih = max(0.0, iy2 - iy1)
            intersection = iw * ih

            union = t_area + d_area - intersection
            iou = intersection / union if union > 0 else 0.0
            cost_matrix[i, j] = 1.0 - iou

    return cost_matrix


def compute_cosine_distance_matrix(
    tracks: List[DeepTrack],
    features: np.ndarray,
    nn_budget: int = 100,
) -> np.ndarray:
    """Compute minimum cosine distance between detection features and track gallery."""
    cost_matrix = np.ones((len(tracks), len(features)), dtype=np.float32)
    if len(tracks) == 0 or len(features) == 0:
        return cost_matrix

    for i, trk in enumerate(tracks):
        trk_feats = np.array(trk.features[-nn_budget:], dtype=np.float32)  # (M, 128)
        # Cosine distance = 1 - dot product (since features are L2-normalized)
        dot_products = np.dot(trk_feats, features.T)  # (M, N)
        cos_dists = 1.0 - dot_products
        min_dists = np.min(cos_dists, axis=0)  # (N,)
        cost_matrix[i, :] = min_dists

    return cost_matrix


class DeepSORTTracker:
    """DeepSORT Tracker using appearance embeddings and Kalman filtering."""

    def __init__(
        self,
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        nn_budget: int = 100,
        device: str = "cuda" if np.show_config is None else "cpu",  # auto
    ) -> None:
        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget

        self.kf = KalmanBoxTracker()
        self.extractor = AppearanceExtractor()
        self.tracks: List[DeepTrack] = []
        self._next_id: int = 1

    def predict(self) -> None:
        """Predict forward all existing tracks."""
        for trk in self.tracks:
            trk.predict(self.kf)

    def update(
        self,
        frame: np.ndarray,
        detections: np.ndarray,  # (N, 6): [x1, y1, x2, y2, conf, cls_id]
    ) -> List[DeepTrack]:
        """Process a new frame and set of detections, returning confirmed tracks.

        Args:
            frame: Full BGR frame (H, W, 3).
            detections: Array of detections in [x1, y1, x2, y2, conf, cls_id].

        Returns:
            List of confirmed active tracks.
        """
        # 1. Filter detections by confidence
        if len(detections) > 0:
            mask = detections[:, 4] >= self.min_confidence
            valid_dets = detections[mask]
        else:
            valid_dets = np.empty((0, 6), dtype=np.float32)

        # 2. Kalman prediction
        self.predict()

        # 3. Extract appearance features for detections
        if len(valid_dets) > 0:
            features = self.extractor.extract(frame, valid_dets[:, :4])
        else:
            features = np.empty((0, 128), dtype=np.float32)

        # 4. Association matching
        matches, unmatched_tracks, unmatched_dets = self._match(valid_dets, features)

        # 5. Update matched tracks
        for trk_idx, det_idx in matches:
            meas = bbox_to_z(tuple(valid_dets[det_idx, :4]))
            self.tracks[trk_idx].update(
                self.kf,
                meas,
                features[det_idx],
                conf=float(valid_dets[det_idx, 4]),
                cls_id=int(valid_dets[det_idx, 5]),
            )

        # 6. Mark unmatched tracks as missed
        for trk_idx in unmatched_tracks:
            self.tracks[trk_idx].mark_missed()

        # 7. Initialise new tentative tracks for unmatched detections
        for det_idx in unmatched_dets:
            meas = bbox_to_z(tuple(valid_dets[det_idx, :4]))
            mean, cov = self.kf.initiate(meas)
            new_trk = DeepTrack(
                mean=mean,
                covariance=cov,
                track_id=self._next_id,
                n_init=self.n_init,
                max_age=self.max_age,
                feature=features[det_idx],
                cls_id=int(valid_dets[det_idx, 5]),
                conf=float(valid_dets[det_idx, 4]),
            )
            self.tracks.append(new_trk)
            self._next_id += 1

        # 8. Filter out deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Return confirmed active tracks
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update == 0]

    def _match(
        self,
        detections: np.ndarray,
        features: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Run cascade cosine matching followed by IoU matching."""
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_indices = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Stage 1: Cascade matching on confirmed tracks
        matches_a: List[Tuple[int, int]] = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks_a: List[int] = []

        if confirmed_indices and len(detections) > 0:
            confirmed_tracks = [self.tracks[i] for i in confirmed_indices]
            cost_matrix = compute_cosine_distance_matrix(
                confirmed_tracks, features, self.nn_budget
            )

            # Apply Mahalanobis gating
            for row, trk in enumerate(confirmed_tracks):
                measurements = np.array([bbox_to_z(tuple(d[:4])) for d in detections])
                gating_dists = self.kf.gating_distance(
                    trk.mean, trk.covariance, measurements
                )
                cost_matrix[row, gating_dists > KalmanBoxTracker.chi2inv95[4]] = 1e5

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_det_set = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= self.max_dist:
                    matches_a.append((confirmed_indices[r], c))
                    matched_det_set.add(c)
                else:
                    unmatched_tracks_a.append(confirmed_indices[r])

            for r, idx in enumerate(confirmed_indices):
                if r not in row_ind:
                    unmatched_tracks_a.append(idx)

            unmatched_dets = [
                c for c in range(len(detections)) if c not in matched_det_set
            ]
        else:
            unmatched_tracks_a = confirmed_indices

        # Stage 2: IoU matching for unconfirmed tracks and remaining confirmed tracks
        tracks_stage2_indices = unmatched_tracks_a + unconfirmed_indices
        matches_b: List[Tuple[int, int]] = []
        unmatched_tracks_b: List[int] = []

        if tracks_stage2_indices and unmatched_dets:
            tracks_stage2 = [self.tracks[i] for i in tracks_stage2_indices]
            sub_dets = detections[unmatched_dets, :4]
            iou_cost = compute_iou_matrix(tracks_stage2, sub_dets)

            row_ind, col_ind = linear_sum_assignment(iou_cost)
            matched_dets_b_set = set()

            for r, c in zip(row_ind, col_ind):
                if iou_cost[r, c] <= self.max_iou_distance:
                    matches_b.append((tracks_stage2_indices[r], unmatched_dets[c]))
                    matched_dets_b_set.add(unmatched_dets[c])
                else:
                    unmatched_tracks_b.append(tracks_stage2_indices[r])

            for r, idx in enumerate(tracks_stage2_indices):
                if r not in row_ind:
                    unmatched_tracks_b.append(idx)

            final_unmatched_dets = [
                d for d in unmatched_dets if d not in matched_dets_b_set
            ]
        else:
            unmatched_tracks_b = tracks_stage2_indices
            final_unmatched_dets = unmatched_dets

        matches = matches_a + matches_b
        return matches, unmatched_tracks_b, final_unmatched_dets
