#!/usr/bin/env bash
# ==============================================================================
# Autonomous Driving Perception Stack — Linux / Ubuntu / WSL 1-Click Setup
# ==============================================================================

set -e

echo "=== [SETUP] Autonomous Driving Perception Stack Environment Setup ==="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found. Please install Python 3.10+ (sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

echo "--> Creating Python Virtual Environment (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
else
    echo "--> .venv already exists."
fi

source .venv/bin/activate

echo "--> Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "--> Installing dependencies from requirements.txt..."
pip install -r requirements.txt
pip install gradio

echo "--> Verifying PyTorch and CUDA support..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

echo "=== [SUCCESS] Environment successfully initialized! ==="
echo "To run the ROS2 perception demo:"
echo "    source .venv/bin/activate"
echo "    python scripts/run_demo.py scenario:=pedestrian weather:=rain"
echo "To run the Gradio web interface:"
echo "    python app.py"
