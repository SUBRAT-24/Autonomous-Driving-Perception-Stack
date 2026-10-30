#!/usr/bin/env python3
"""
Environment and hardware check script for Autonomous Driving Perception Stack.
Checks:
- OS & Architecture
- Python version
- NVIDIA Driver & CUDA
- GPU model & VRAM
- PyTorch & CUDA availability
- Ultralytics (YOLO)
- OpenCV
- ONNX & ONNXRuntime (GPU vs CPU)
- TensorRT
- ROS2 distribution (if in ROS2 environment)
- CARLA Python client API
- Weights & Biases (wandb) login status
"""

import os
import platform
import shutil
import subprocess
import sys


def print_header(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


def check_os():
    print_header("1. Operating System & Platform")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    is_wsl = "microsoft" in platform.uname().release.lower()
    print(f"Running inside WSL: {'YES (WSL2)' if is_wsl else 'NO (Native)'}")


def check_python():
    print_header("2. Python Environment")
    print(f"Python Executable: {sys.executable}")
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 10:
        print("[PASS] Python version >= 3.10")
    else:
        print("[WARN] Python >= 3.10 is recommended.")


def check_system_cuda():
    print_header("3. System NVIDIA Driver & CUDA")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                encoding="utf-8",
            ).strip()
            print("[PASS] GPU detected via nvidia-smi:")
            for line in output.split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    print(f"  - Device: {parts[0]}")
                    print(f"  - Total VRAM: {parts[1]}")
                    print(f"  - Driver Version: {parts[2]}")
                else:
                    print(f"  - {line}")
        except Exception as e:
            print(f"[WARN] Failed to query nvidia-smi: {e}")
    else:
        print("[FAIL] nvidia-smi not found in PATH.")

    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output(["nvcc", "--version"], encoding="utf-8")
            print("[PASS] nvcc (CUDA Toolkit) found:")
            for line in out.split("\n"):
                if "release" in line.lower():
                    print(f"  {line.strip()}")
        except Exception as e:
            print(f"[WARN] nvcc execution error: {e}")
    else:
        print(
            "[INFO] nvcc not found in PATH (CUDA toolkit compiler optional if PyTorch wheel has prebuilt CUDA)."
        )


def check_pytorch():
    print_header("4. PyTorch & GPU Acceleration")
    try:
        import torch

        print(f"[PASS] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"torch.cuda.is_available(): {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Version in Torch: {torch.version.cuda}")
            print(f"Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024**3)
                print(f"  - GPU {i}: {props.name} ({vram_gb:.2f} GB VRAM)")
        else:
            print("[FAIL] PyTorch is NOT using CUDA / GPU!")
    except ImportError:
        print("[FAIL] PyTorch is not installed in current environment.")


def check_vision_and_ml():
    print_header("5. Vision & ML Libraries")

    # Ultralytics
    try:
        import ultralytics

        print(f"[PASS] Ultralytics (YOLO) Version: {ultralytics.__version__}")
    except ImportError:
        print("[FAIL] Ultralytics is not installed.")

    # OpenCV
    try:
        import cv2

        print(f"[PASS] OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("[FAIL] OpenCV is not installed.")

    # ONNX
    try:
        import onnx

        print(f"[PASS] ONNX Version: {onnx.__version__}")
    except ImportError:
        print("[FAIL] ONNX is not installed.")

    # ONNXRuntime
    try:
        import onnxruntime as ort

        print(f"[PASS] ONNXRuntime Version: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Available Providers: {providers}")
        if "CUDAExecutionProvider" in providers:
            print("  [PASS] CUDAExecutionProvider is enabled.")
        else:
            print("  [WARN] CUDAExecutionProvider is NOT enabled (CPU only).")
    except ImportError:
        print("[FAIL] ONNXRuntime is not installed.")

    # TensorRT
    try:
        import tensorrt as trt

        print(f"[PASS] TensorRT Version: {trt.__version__}")
    except ImportError:
        print("[FAIL] TensorRT python module is not installed.")


def check_ros2():
    print_header("6. ROS2 Environment")
    ros_distro = os.environ.get("ROS_DISTRO")
    if ros_distro:
        print(f"[PASS] ROS2 Distro: {ros_distro}")
        ros_version = os.environ.get("ROS_VERSION")
        print(f"ROS Version: {ros_version}")
    else:
        print(
            "[INFO] ROS_DISTRO environment variable not set (Not sourced or not in ROS2 shell)."
        )

    ros2_cli = shutil.which("ros2")
    if ros2_cli:
        print(f"[PASS] ros2 CLI located at: {ros2_cli}")
    else:
        print("[INFO] ros2 command line tool not found in current PATH.")


def check_carla():
    print_header("7. CARLA Client API")
    try:
        import carla

        # Query client version / egg info
        version = getattr(carla, "__version__", "Available (version attribute missing)")
        print(f"[PASS] CARLA Python API is installed: {version}")
    except ImportError:
        print("[FAIL] carla package not installed in current Python environment.")


def check_wandb():
    print_header("8. Weights & Biases (W&B)")
    try:
        import wandb

        print(f"[PASS] wandb library Version: {wandb.__version__}")
        try:
            if wandb.login(anonymous="never", timeout=2):
                api = wandb.Api()
                user = api.viewer
                print(
                    f"[PASS] W&B Login Status: Logged in as '{user.entity}' ({user.username})"
                )
            else:
                print(
                    "[WARN] W&B is installed but NOT logged in. Run: .venv\\Scripts\\wandb.exe login"
                )
        except Exception:
            print(
                "[WARN] W&B is installed but NOT logged in. Run: .venv\\Scripts\\wandb.exe login"
            )
    except ImportError:
        print("[FAIL] wandb library is not installed.")


def main():
    print("=" * 50)
    print(" AUTONOMOUS DRIVING PERCEPTION STACK - ENV CHECK")
    print("=" * 50)
    check_os()
    check_python()
    check_system_cuda()
    check_pytorch()
    check_vision_and_ml()
    check_ros2()
    check_carla()
    check_wandb()
    print("\n" + "=" * 50)
    print(" CHECK COMPLETED")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
