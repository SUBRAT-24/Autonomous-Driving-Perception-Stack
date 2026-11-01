"""Integration tests for Rosbag recording and replay.

Verifies:
1. Rosbags for all 4 scenarios exist and contain valid recorded data.
2. RosbagPlayer successfully deserializes and yields recorded messages.
3. /perception/objects messages contain valid TrackedObject detections with metric distance and TTC.
4. Reactive emergency braking was actuated in the pedestrian crossing scenario.
5. End-to-end replay via MessageBus correctly triggers subscribed callbacks.
"""

from __future__ import annotations
from pathlib import Path
from typing import List

import pytest

from src.ros_compat.bag import RosbagPlayer
from src.ros_compat.bus import MessageBus
from perception_msgs.msg import TrackedObject, TrackedObjectArray


BAG_DIR = Path("data/sample/rosbags")
SCENARIOS = [
    "rosbag_pedestrian_crossing.bag",
    "rosbag_lead_car_braking.bag",
    "rosbag_night.bag",
    "rosbag_rain.bag",
]
PEDESTRIAN_BAG = BAG_DIR / "rosbag_pedestrian_crossing.bag"


@pytest.fixture(autouse=True)
def clean_bus():
    """Reset MessageBus singleton state between tests."""
    bus = MessageBus.get_instance()
    bus.reset()
    yield
    bus.reset()


@pytest.mark.parametrize("bag_name", SCENARIOS)
def test_all_scenario_bags_exist(bag_name: str):
    """Verify that rosbags exist for all scenarios and are non-empty."""
    path = BAG_DIR / bag_name
    assert path.exists(), f"Rosbag file missing: {path}"
    assert path.stat().st_size > 100, f"Rosbag file too small: {path}"


def test_rosbag_player_deserialization():
    """Test reading and deserializing recorded messages from rosbag."""
    player = RosbagPlayer(str(PEDESTRIAN_BAG))
    messages = list(player.read_messages())
    assert len(messages) > 0, "Rosbag contained zero messages"

    topics_found = {topic for topic, _, _ in messages}
    assert "/camera/image_raw" in topics_found, "Missing /camera/image_raw topic"
    assert "/perception/objects" in topics_found, "Missing /perception/objects topic"


def test_perception_objects_validity():
    """Verify that /perception/objects messages contain valid TrackedObject instances."""
    player = RosbagPlayer(str(PEDESTRIAN_BAG))
    object_arrays: List[TrackedObjectArray] = [
        msg
        for topic, msg, _ in player.read_messages(topics=["/perception/objects"])
        if isinstance(msg, TrackedObjectArray)
    ]

    assert len(object_arrays) > 0, "No TrackedObjectArray messages found in rosbag"

    total_objects = 0
    has_valid_distance = False
    has_valid_ttc = False

    for arr in object_arrays:
        for obj in arr.objects:
            total_objects += 1
            assert isinstance(obj, TrackedObject)
            assert obj.id >= 0
            assert len(obj.bbox) == 4
            assert obj.confidence >= 0.0
            assert obj.class_name != ""

            if obj.distance_m > 0.0:
                has_valid_distance = True
            if obj.ttc_s is not None:
                has_valid_ttc = True

    assert (
        total_objects > 0
    ), "Rosbag /perception/objects messages had 0 total tracked objects"
    assert (
        has_valid_distance
    ), "At least one tracked object must have a positive metric distance"
    assert has_valid_ttc, "At least one tracked object must have a valid TTC computed"


def test_emergency_braking_in_rosbag():
    """Verify that reactive emergency braking was triggered in the recording."""
    player = RosbagPlayer(str(PEDESTRIAN_BAG))
    control_messages = [
        msg for topic, msg, _ in player.read_messages(topics=["/vehicle/control"])
    ]

    assert len(control_messages) > 0, "No /vehicle/control messages found in rosbag"
    braking_events = [
        ctrl
        for ctrl in control_messages
        if isinstance(ctrl, dict) and ctrl.get("brake", 0.0) >= 0.99
    ]
    assert (
        len(braking_events) > 0
    ), "Reactive emergency brake was not triggered in pedestrian scenario"


def test_message_bus_replay():
    """Verify end-to-end replay onto MessageBus dispatches to subscriber callbacks."""
    received_objects: List[TrackedObjectArray] = []

    bus = MessageBus.get_instance()
    bus.subscribe("/perception/objects", lambda msg: received_objects.append(msg))

    player = RosbagPlayer(str(PEDESTRIAN_BAG))
    published_count = player.play(rate_hz=0.0, max_messages=30)

    assert published_count > 0
    assert (
        len(received_objects) > 0
    ), "Subscriber did not receive replayed /perception/objects messages"
    assert isinstance(received_objects[0], TrackedObjectArray)
