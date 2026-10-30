"""Unit tests for BDD100K → YOLO label conversion.

Tests cover:
- Single box coordinate conversion
- Class mapping (all 9 target classes + skip of unknown)
- Edge cases (image boundaries, degenerate boxes)
- Full image entry conversion (end-to-end)
- Output file format validation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.data.bdd100k_to_yolo import (
    BDD_TO_YOLO_CLASS,
    NUM_CLASSES,
    YOLO_CLASS_NAMES,
    YOLOBox,
    convert_box,
    convert_image_labels,
    convert_label,
    write_yolo_labels,
)

# ---------------------------------------------------------------------------
# Constants for BDD100K standard image size
# ---------------------------------------------------------------------------
IMG_W = 1280
IMG_H = 720


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_car_label() -> Dict[str, Any]:
    """Sample 1: Single car, normal position."""
    return {
        "category": "car",
        "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
    }


@pytest.fixture()
def pedestrian_edge_label() -> Dict[str, Any]:
    """Sample 2: Pedestrian at left edge of image."""
    return {
        "category": "pedestrian",
        "box2d": {"x1": 0.0, "y1": 100.0, "x2": 50.0, "y2": 400.0},
    }


@pytest.fixture()
def traffic_light_label() -> Dict[str, Any]:
    """Sample 3: Small traffic light."""
    return {
        "category": "traffic light",
        "box2d": {"x1": 600.0, "y1": 50.0, "x2": 620.0, "y2": 100.0},
    }


@pytest.fixture()
def train_label() -> Dict[str, Any]:
    """Sample 4: Unknown class 'train' — should be skipped."""
    return {
        "category": "train",
        "box2d": {"x1": 0.0, "y1": 0.0, "x2": 1280.0, "y2": 720.0},
    }


@pytest.fixture()
def multi_object_image_entry() -> Dict[str, Any]:
    """Sample 5: Image with car + motorcycle."""
    return {
        "name": "multi_object.jpg",
        "labels": [
            {
                "category": "car",
                "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
            },
            {
                "category": "motorcycle",
                "box2d": {"x1": 500.0, "y1": 300.0, "x2": 600.0, "y2": 450.0},
            },
        ],
    }


@pytest.fixture()
def full_bdd_image_entry() -> Dict[str, Any]:
    """A full BDD100K image entry with mixed categories."""
    return {
        "name": "test_image.jpg",
        "attributes": {"weather": "clear", "scene": "highway", "timeofday": "daytime"},
        "labels": [
            {
                "category": "car",
                "box2d": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0},
            },
            {
                "category": "pedestrian",
                "box2d": {"x1": 0.0, "y1": 100.0, "x2": 50.0, "y2": 400.0},
            },
            {
                "category": "train",
                "box2d": {"x1": 0.0, "y1": 0.0, "x2": 1280.0, "y2": 720.0},
            },
            {
                "category": "traffic light",
                "box2d": {"x1": 600.0, "y1": 50.0, "x2": 620.0, "y2": 100.0},
            },
        ],
    }


# ---------------------------------------------------------------------------
# test_single_box_conversion — tests convert_box directly
# ---------------------------------------------------------------------------


class TestSingleBoxConversion:
    """Tests for the low-level ``convert_box`` function."""

    def test_sample1_car(self) -> None:
        """Sample 1: Car at (100,200)-(300,400) on 1280×720."""
        box2d = {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx == pytest.approx(0.15625, abs=1e-5)
        assert cy == pytest.approx(0.41667, abs=1e-4)
        assert w == pytest.approx(0.15625, abs=1e-5)
        assert h == pytest.approx(0.27778, abs=1e-4)

    def test_sample2_pedestrian_edge(self) -> None:
        """Sample 2: Pedestrian at (0,100)-(50,400) on 1280×720."""
        box2d = {"x1": 0.0, "y1": 100.0, "x2": 50.0, "y2": 400.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx == pytest.approx(0.01953, abs=1e-4)
        assert cy == pytest.approx(0.34722, abs=1e-4)
        assert w == pytest.approx(0.03906, abs=1e-4)
        assert h == pytest.approx(0.41667, abs=1e-4)

    def test_sample3_traffic_light_small(self) -> None:
        """Sample 3: Traffic light at (600,50)-(620,100) on 1280×720."""
        box2d = {"x1": 600.0, "y1": 50.0, "x2": 620.0, "y2": 100.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx == pytest.approx(0.47656, abs=1e-4)
        assert cy == pytest.approx(0.10417, abs=1e-4)
        assert w == pytest.approx(0.01563, abs=1e-4)
        assert h == pytest.approx(0.06944, abs=1e-4)

    def test_full_image_box(self) -> None:
        """Box covering the entire image → cx=0.5, cy=0.5, w=1.0, h=1.0."""
        box2d = {"x1": 0.0, "y1": 0.0, "x2": 1280.0, "y2": 720.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx == pytest.approx(0.5, abs=1e-5)
        assert cy == pytest.approx(0.5, abs=1e-5)
        assert w == pytest.approx(1.0, abs=1e-5)
        assert h == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# test_class_mapping — tests all 9 mappings + skip
# ---------------------------------------------------------------------------


class TestClassMapping:
    """Tests for BDD100K → YOLO class ID mapping."""

    @pytest.mark.parametrize(
        "bdd_category, expected_id",
        [
            ("car", 0),
            ("truck", 1),
            ("bus", 2),
            ("pedestrian", 3),
            ("rider", 4),
            ("bicycle", 5),
            ("motorcycle", 6),
            ("traffic light", 7),
            ("traffic sign", 8),
        ],
    )
    def test_valid_class_mappings(self, bdd_category: str, expected_id: int) -> None:
        """Each of the 9 target BDD100K categories maps to the correct YOLO ID."""
        assert BDD_TO_YOLO_CLASS[bdd_category] == expected_id

    def test_total_classes(self) -> None:
        """There are exactly 9 target classes."""
        assert len(BDD_TO_YOLO_CLASS) == NUM_CLASSES
        assert NUM_CLASSES == 9

    def test_yolo_class_names_length(self) -> None:
        """YOLO_CLASS_NAMES has exactly NUM_CLASSES entries."""
        assert len(YOLO_CLASS_NAMES) == NUM_CLASSES

    def test_unknown_class_skipped(self, train_label: Dict[str, Any]) -> None:
        """Category 'train' is not in the mapping and should be skipped."""
        assert "train" not in BDD_TO_YOLO_CLASS
        result = convert_label(train_label, IMG_W, IMG_H)
        assert result is None

    def test_empty_category_skipped(self) -> None:
        """Empty category string is skipped."""
        label = {"category": "", "box2d": {"x1": 0, "y1": 0, "x2": 100, "y2": 100}}
        result = convert_label(label, IMG_W, IMG_H)
        assert result is None

    def test_convert_label_returns_correct_class_id(
        self, single_car_label: Dict[str, Any]
    ) -> None:
        """convert_label assigns the correct class_id for 'car'."""
        result = convert_label(single_car_label, IMG_W, IMG_H)
        assert result is not None
        assert result.class_id == 0

    def test_convert_label_pedestrian(
        self, pedestrian_edge_label: Dict[str, Any]
    ) -> None:
        """convert_label assigns class_id=3 for 'pedestrian'."""
        result = convert_label(pedestrian_edge_label, IMG_W, IMG_H)
        assert result is not None
        assert result.class_id == 3

    def test_convert_label_traffic_light(
        self, traffic_light_label: Dict[str, Any]
    ) -> None:
        """convert_label assigns class_id=7 for 'traffic light'."""
        result = convert_label(traffic_light_label, IMG_W, IMG_H)
        assert result is not None
        assert result.class_id == 7


# ---------------------------------------------------------------------------
# test_edge_cases — boundary conditions and degenerate boxes
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases in box conversion."""

    def test_box_at_origin(self) -> None:
        """Box starting at (0, 0)."""
        box2d = {"x1": 0.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx >= 0.0
        assert cy >= 0.0
        assert w > 0.0
        assert h > 0.0

    def test_box_at_bottom_right_corner(self) -> None:
        """Box touching the bottom-right corner."""
        box2d = {"x1": 1180.0, "y1": 620.0, "x2": 1280.0, "y2": 720.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert cx <= 1.0
        assert cy <= 1.0
        assert w <= 1.0
        assert h <= 1.0

    def test_zero_width_box_skipped(self) -> None:
        """Box with x1 == x2 should be skipped (zero width)."""
        box2d = {"x1": 100.0, "y1": 100.0, "x2": 100.0, "y2": 200.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is None

    def test_zero_height_box_skipped(self) -> None:
        """Box with y1 == y2 should be skipped (zero height)."""
        box2d = {"x1": 100.0, "y1": 200.0, "x2": 200.0, "y2": 200.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is None

    def test_negative_width_box_skipped(self) -> None:
        """Box with x2 < x1 (inverted) should be skipped."""
        box2d = {"x1": 300.0, "y1": 100.0, "x2": 100.0, "y2": 200.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is None

    def test_values_clamped_to_unit_range(self) -> None:
        """Box coordinates extending beyond image boundaries are clamped."""
        box2d = {"x1": -50.0, "y1": -50.0, "x2": 1330.0, "y2": 770.0}
        result = convert_box(box2d, IMG_W, IMG_H)
        assert result is not None
        cx, cy, w, h = result
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0

    def test_missing_box2d_returns_none(self) -> None:
        """Label without box2d key should return None."""
        label = {"category": "car"}
        result = convert_label(label, IMG_W, IMG_H)
        assert result is None

    def test_image_with_no_labels(self) -> None:
        """Image entry with empty labels list produces empty result."""
        entry = {"name": "empty.jpg", "labels": []}
        boxes, skipped = convert_image_labels(entry, IMG_W, IMG_H)
        assert boxes == []
        assert skipped == []

    def test_image_with_none_labels(self) -> None:
        """Image entry with labels=None produces empty result."""
        entry = {"name": "none_labels.jpg", "labels": None}
        boxes, skipped = convert_image_labels(entry, IMG_W, IMG_H)
        assert boxes == []
        assert skipped == []


# ---------------------------------------------------------------------------
# test_full_image_conversion — end-to-end of convert_image_labels
# ---------------------------------------------------------------------------


class TestFullImageConversion:
    """End-to-end tests converting a complete BDD100K image entry."""

    def test_sample5_multi_object(
        self, multi_object_image_entry: Dict[str, Any]
    ) -> None:
        """Sample 5: Image with car + motorcycle → two YOLO lines."""
        boxes, skipped = convert_image_labels(multi_object_image_entry, IMG_W, IMG_H)
        assert len(boxes) == 2
        assert skipped == []
        # First box is car (class 0)
        assert boxes[0].class_id == 0
        # Second box is motorcycle (class 6)
        assert boxes[1].class_id == 6

    def test_mixed_valid_and_skipped(
        self, full_bdd_image_entry: Dict[str, Any]
    ) -> None:
        """Image with car, pedestrian, train (skip), traffic light."""
        boxes, skipped = convert_image_labels(full_bdd_image_entry, IMG_W, IMG_H)
        # 3 valid labels: car, pedestrian, traffic light
        assert len(boxes) == 3
        # 1 skipped: train
        assert skipped == ["train"]

        class_ids = [b.class_id for b in boxes]
        assert 0 in class_ids  # car
        assert 3 in class_ids  # pedestrian
        assert 7 in class_ids  # traffic light

    def test_conversion_values_are_correct(
        self, single_car_label: Dict[str, Any]
    ) -> None:
        """Verify exact numeric output for sample 1 car through convert_label."""
        entry = {
            "name": "sample1.jpg",
            "labels": [single_car_label],
        }
        boxes, _ = convert_image_labels(entry, IMG_W, IMG_H)
        assert len(boxes) == 1
        box = boxes[0]
        assert box.class_id == 0
        assert box.center_x == pytest.approx(0.15625, abs=1e-5)
        assert box.center_y == pytest.approx(0.41667, abs=1e-4)
        assert box.width == pytest.approx(0.15625, abs=1e-5)
        assert box.height == pytest.approx(0.27778, abs=1e-4)


# ---------------------------------------------------------------------------
# test_output_file_format — verify .txt output format
# ---------------------------------------------------------------------------


class TestOutputFileFormat:
    """Tests for YOLO .txt label file output format."""

    def test_single_box_file(self, tmp_path: Path) -> None:
        """Single box writes one line with correct format."""
        box = YOLOBox(
            class_id=0,
            center_x=0.15625,
            center_y=0.416667,
            width=0.15625,
            height=0.277778,
        )
        output_path = tmp_path / "labels" / "test.txt"
        write_yolo_labels([box], output_path)

        assert output_path.exists()
        content = output_path.read_text().strip()
        parts = content.split()
        assert len(parts) == 5
        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.15625, abs=1e-5)
        assert float(parts[2]) == pytest.approx(0.41667, abs=1e-4)
        assert float(parts[3]) == pytest.approx(0.15625, abs=1e-5)
        assert float(parts[4]) == pytest.approx(0.27778, abs=1e-4)

    def test_multi_box_file(self, tmp_path: Path) -> None:
        """Multiple boxes write multiple lines."""
        boxes = [
            YOLOBox(class_id=0, center_x=0.15, center_y=0.42, width=0.16, height=0.28),
            YOLOBox(class_id=6, center_x=0.43, center_y=0.52, width=0.08, height=0.21),
        ]
        output_path = tmp_path / "labels" / "multi.txt"
        write_yolo_labels(boxes, output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Verify first line
        p1 = lines[0].split()
        assert p1[0] == "0"

        # Verify second line
        p2 = lines[1].split()
        assert p2[0] == "6"

    def test_empty_boxes_file(self, tmp_path: Path) -> None:
        """Empty box list creates an empty file."""
        output_path = tmp_path / "labels" / "empty.txt"
        write_yolo_labels([], output_path)

        assert output_path.exists()
        assert output_path.read_text() == ""

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        """write_yolo_labels creates parent directories if they don't exist."""
        output_path = tmp_path / "deep" / "nested" / "dir" / "test.txt"
        box = YOLOBox(class_id=1, center_x=0.5, center_y=0.5, width=0.1, height=0.1)
        write_yolo_labels([box], output_path)
        assert output_path.exists()

    def test_yolo_line_format(self) -> None:
        """YOLOBox.to_line produces space-separated values with 6 decimal places."""
        box = YOLOBox(
            class_id=7,
            center_x=0.476563,
            center_y=0.104167,
            width=0.015625,
            height=0.069444,
        )
        line = box.to_line()
        parts = line.split()
        assert len(parts) == 5
        assert parts[0] == "7"
        # All float values have 6 decimal places
        for p in parts[1:]:
            assert "." in p
            decimal_part = p.split(".")[1]
            assert len(decimal_part) == 6

    def test_end_to_end_file_from_bdd_entry(
        self, tmp_path: Path, full_bdd_image_entry: Dict[str, Any]
    ) -> None:
        """Full pipeline: BDD100K entry → convert → write → read back & validate."""
        boxes, _ = convert_image_labels(full_bdd_image_entry, IMG_W, IMG_H)
        output_path = tmp_path / "labels" / "test_image.txt"
        write_yolo_labels(boxes, output_path)

        assert output_path.exists()
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 3  # car, pedestrian, traffic light

        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            class_id = int(parts[0])
            assert 0 <= class_id < NUM_CLASSES
            for val_str in parts[1:]:
                val = float(val_str)
                assert 0.0 <= val <= 1.0
