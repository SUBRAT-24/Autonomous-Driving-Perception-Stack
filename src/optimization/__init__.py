"""Optimization package for ONNX and TensorRT compilation and acceleration."""

from src.optimization.build_engine import build_tensorrt_engine

__all__ = ["build_tensorrt_engine"]
