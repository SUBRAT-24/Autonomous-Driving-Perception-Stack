"""TensorRT Engine Builder for Autonomous Driving Perception Stack.

Compiles ONNX models to optimized TensorRT runtime engines:
- YOLOv8s detector (FP16 / INT8)
- MiDaS small depth estimator (FP16)
Includes INT8 calibrator supporting BDD100K calibration subsets.
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tensorrt as trt

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class BDD100KInt8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 Entropy Calibrator reading calibration images from disk."""

    def __init__(
        self,
        image_paths: List[str],
        cache_file: str,
        input_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
    ) -> None:
        super().__init__()
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        self.image_paths = image_paths
        self.cache_file = cache_file
        self.input_shape = input_shape
        self.batch_size = input_shape[0]
        self.current_idx = 0

        # Allocate host and device memory for one batch
        self.n_bytes = int(np.prod(input_shape) * 4)
        self.device_input = cuda.mem_alloc(self.n_bytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self.current_idx >= len(self.image_paths):
            return None

        import pycuda.driver as cuda

        batch_imgs = []
        for _ in range(self.batch_size):
            if self.current_idx < len(self.image_paths):
                img = cv2.imread(self.image_paths[self.current_idx])
                if img is None:
                    img = np.zeros(
                        (self.input_shape[2], self.input_shape[3], 3), dtype=np.uint8
                    )
                else:
                    img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
                # Preprocess: BGR -> RGB -> Float32 [0, 1] -> NCHW
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                batch_imgs.append(img)
                self.current_idx += 1

        batch_arr = np.ascontiguousarray(np.stack(batch_imgs, axis=0), dtype=np.float32)
        cuda.memcpy_htod(self.device_input, batch_arr)
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes:
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return b""

    def write_calibration_cache(self, cache: bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_file)), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    fp16: bool = True,
    int8: bool = False,
    calib_images: Optional[List[str]] = None,
    workspace_gb: float = 2.0,
) -> Path:
    """Builds and serializes a TensorRT engine from an ONNX file.

    Args:
        onnx_path: Path to input .onnx file.
        engine_path: Path to output .engine file.
        fp16: Enable FP16 precision.
        int8: Enable INT8 precision (requires calibrator).
        calib_images: Optional list of calibration image paths for INT8.
        workspace_gb: Maximum GPU workspace memory in gigabytes.

    Returns:
        Path to compiled serialized engine.
    """
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    logger.info("Initializing TensorRT builder for %s -> %s", onnx_path, engine_path)
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(error))
            raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

    config = builder.create_builder_config()
    # Workspace limit
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3))
    )

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 precision mode")

    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("Enabled INT8 precision mode")
        if calib_images:
            cache_file = str(engine_path.with_suffix(".calib_cache"))
            calibrator = BDD100KInt8Calibrator(
                image_paths=calib_images,
                cache_file=cache_file,
            )
            config.int8_calibrator = calibrator

    logger.info("Building serialized TensorRT engine (this may take 1-2 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logger.info(
        "Successfully built and saved TensorRT engine to %s (%d MB)",
        engine_path,
        len(serialized_engine) // (1024 * 1024),
    )
    return engine_path


def main() -> None:
    # 1. Build YOLOv8s TensorRT FP16 engine
    yolo_onnx = Path("runs/detect/bdd100k_yolov8s/weights/best.onnx")
    yolo_engine = Path("runs/detect/bdd100k_yolov8s/weights/best.engine")

    if yolo_onnx.exists():
        build_tensorrt_engine(
            onnx_path=yolo_onnx,
            engine_path=yolo_engine,
            fp16=True,
            int8=False,
        )

    # 2. Build MiDaS Small TensorRT FP16 engine
    midas_onnx = Path("weights/midas_small.onnx")
    midas_engine = Path("weights/midas_small.engine")

    if midas_onnx.exists():
        try:
            build_tensorrt_engine(
                onnx_path=midas_onnx,
                engine_path=midas_engine,
                fp16=True,
                int8=False,
            )
        except Exception as e:
            logger.warning("MiDaS TensorRT compilation note: %s", e)


if __name__ == "__main__":
    main()
