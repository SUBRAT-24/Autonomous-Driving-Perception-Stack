"""
ROS2 Node and Lifecycle abstractions.
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Type
from src.ros_compat.bus import MessageBus


class Parameter:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


class Publisher:
    def __init__(self, topic: str, msg_type: Type[Any]) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.bus = MessageBus.get_instance()

    def publish(self, msg: Any) -> None:
        self.bus.publish(self.topic, msg)


class Subscription:
    def __init__(
        self, topic: str, msg_type: Type[Any], callback: Callable[[Any], None]
    ) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        self.bus = MessageBus.get_instance()
        self.bus.subscribe(topic, callback)

    def destroy(self) -> None:
        self.bus.unsubscribe(self.topic, self.callback)


class Timer:
    def __init__(self, period_sec: float, callback: Callable[[], None]) -> None:
        self.period_sec = period_sec
        self.callback = callback
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while self.running:
            time.sleep(self.period_sec)
            if self.running:
                try:
                    self.callback()
                except Exception as e:
                    logging.getLogger("Timer").error(f"Timer callback error: {e}")

    def cancel(self) -> None:
        self.running = False


class Node:
    """ROS2-compatible Node base class."""

    def __init__(self, node_name: str) -> None:
        self.node_name = node_name
        self.logger = logging.getLogger(node_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(f"[{node_name}] %(levelname)s: %(message)s")
            )
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self._parameters: Dict[str, Any] = {}
        self._publishers: List[Publisher] = []
        self._subscriptions: List[Subscription] = []
        self._timers: List[Timer] = []

    def declare_parameter(self, name: str, default_value: Any) -> Parameter:
        self._parameters[name] = default_value
        return Parameter(name, default_value)

    def get_parameter(self, name: str) -> Parameter:
        val = self._parameters.get(name, None)
        return Parameter(name, val)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        self._parameters.update(params)

    def get_logger(self) -> logging.Logger:
        return self.logger

    def create_publisher(
        self, msg_type: Type[Any], topic: str, qos_profile: int = 10
    ) -> Publisher:
        pub = Publisher(topic, msg_type)
        self._publishers.append(pub)
        return pub

    def create_subscription(
        self,
        msg_type: Type[Any],
        topic: str,
        callback: Callable[[Any], None],
        qos_profile: int = 10,
    ) -> Subscription:
        sub = Subscription(topic, msg_type, callback)
        self._subscriptions.append(sub)
        return sub

    def create_timer(
        self, timer_period_sec: float, callback: Callable[[], None]
    ) -> Timer:
        timer = Timer(timer_period_sec, callback)
        self._timers.append(timer)
        return timer

    def destroy_node(self) -> None:
        for timer in self._timers:
            timer.cancel()
        for sub in self._subscriptions:
            sub.destroy()
        self._timers.clear()
        self._subscriptions.clear()
        self._publishers.clear()


def spin_once(node: Node, timeout_sec: float = 0.01) -> None:
    time.sleep(timeout_sec)


def spin(node: Node) -> None:
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        node.destroy_node()


__all__ = [
    "Node",
    "Publisher",
    "Subscription",
    "Timer",
    "Parameter",
    "spin",
    "spin_once",
]
