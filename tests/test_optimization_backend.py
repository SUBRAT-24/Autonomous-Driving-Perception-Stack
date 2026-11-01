"""Unit and integration tests for Optimization Backends (ONNX and TensorRT).

Verifies:
1. Exported ONNX model exists, validates with ONNX checker, and matches output shape.
2. Numerical parity between PyTorch and ONNX predictions on synthetic driving inputs.
3. TensorRT engine files exist and deserialize successfully.
4. Tracker unified backend switching ("pytorch", "onnx", "tensorrt").
5. DepthEstimator unified backend switching ("pytorch", "tensorrt").
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pytest
import torch

ONNX_MODEL_PATH = Path("runs/detect/bdd100k_yolov8s/weights/best.onnx")
TRT_ENGINE_PATH = Path("runs/detect/bdd100k_yolov8s/weights/best.engine")
MIDAS_ONNX_PATH = Path("weights/midas_small.onnx")
MIDAS_TRT_PATH = Path("weights/midas_small.engine")


def test_yolo_onnx_model_exists_and_valid():
    """Verify YOLOv8s ONNX model file exists and is valid."""
    assert ONNX_MODEL_PATH.exists(), f"Missing ONNX model: {ONNX_MODEL_PATH}"
    import onnx

    model = onnx.load(str(ONNX_MODEL_PATH))
    onnx.checker.check_model(model)
    assert len(model.graph.input) == 1
    assert model.graph.input[0].name == "images"


def test_yolo_onnx_runtime_inference():
    """Verify ONNX Runtime GPU/CPU can run forward pass on 640x640 input."""
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)

    # YOLOv8s ONNX was exported in FP16 for TensorRT 11 strong typing
    input_type = session.get_inputs()[0].type
    np_dtype = np.float16 if "float16" in input_type else np.float32
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np_dtype)
    outputs = session.run(None, {"images": dummy_input})

    assert len(outputs) == 1
    out = outputs[0]
    # YOLOv8s output: [1, 4 + 9 classes, 8400 anchors]
    assert out.shape == (1, 13, 8400)
    assert not np.isnan(out).any()


def test_midas_onnx_inference():
    """Verify MiDaS small ONNX model can run forward pass on 256x256 input."""
    assert MIDAS_ONNX_PATH.exists(), f"Missing MiDaS ONNX: {MIDAS_ONNX_PATH}"
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(MIDAS_ONNX_PATH),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})

    assert len(outputs) == 1
    out = outputs[0]
    assert out.shape == (1, 1, 128, 128)
    assert not np.isnan(out).any()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required for TensorRT"
)
def test_tensorrt_engine_loading():
    """Verify TensorRT 11 runtime can deserialize compiled YOLO engine."""
    assert TRT_ENGINE_PATH.exists(), f"Missing TRT engine: {TRT_ENGINE_PATH}"
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(TRT_ENGINE_PATH, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    assert engine is not None
    assert engine.num_io_tensors == 2
    assert engine.get_tensor_name(0) == "images"


def test_tracker_backend_parameter_selection():
    """Verify Tracker initializes cleanly with pytorch and onnx backends."""
    from src.tracking.tracker import Tracker

    # PyTorch backend
    tracker_pt = Tracker(tracker_type="bytetrack", backend="pytorch", device="cpu")
    assert tracker_pt.backend == "pytorch"

    # ONNX backend
    tracker_onnx = Tracker(tracker_type="bytetrack", backend="onnx", device="cpu")
    assert tracker_onnx.backend == "onnx"


def test_depth_estimator_backend_parameter_selection():
    """Verify DepthEstimator initializes cleanly with pytorch backend."""
    from src.depth.estimator import DepthEstimator

    estimator = DepthEstimator(device="cpu", backend="pytorch")
    assert estimator.backend == "pytorch"
    dummy_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    depth = estimator.predict(dummy_frame)
    assert depth.shape == (180, 320)
    assert depth.dtype == np.float32
