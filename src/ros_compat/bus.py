"""
In-process Pub/Sub Message Bus.
Connects ROS2-compatible nodes when native rclpy / CycloneDDS is not initialized.
"""

from __future__ import annotations
import collections
import logging
from typing import Any, Callable, Dict, List


class MessageBus:
    """Thread-safe pub/sub broker for local topic routing."""

    _instance: MessageBus | None = None

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Any], None]]] = (
            collections.defaultdict(list)
        )
        self._last_messages: Dict[str, Any] = {}
        self.logger = logging.getLogger("ROS_BUS")

    @classmethod
    def get_instance(cls) -> MessageBus:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is not None:
            cls._instance._subscribers.clear()
            cls._instance._last_messages.clear()

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        if callback not in self._subscribers[topic]:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        if callback in self._subscribers[topic]:
            self._subscribers[topic].remove(callback)

    def publish(self, topic: str, msg: Any) -> None:
        self._last_messages[topic] = msg
        subs = list(self._subscribers.get(topic, []))
        for cb in subs:
            try:
                cb(msg)
            except Exception as e:
                self.logger.error(
                    f"Error in subscriber callback on {topic}: {e}", exc_info=True
                )

    def get_last_message(self, topic: str) -> Any | None:
        return self._last_messages.get(topic, None)


__all__ = ["MessageBus"]
