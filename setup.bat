@echo off
REM ==============================================================================
REM Autonomous Driving Perception Stack — Windows 1-Click Environment Setup
REM ==============================================================================

echo [SETUP] Initializing Autonomous Driving Perception Stack Setup...

REM Check Python installation
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH! Please install Python 3.10 from python.org.
    exit /b 1
)

echo [SETUP] Creating Python Virtual Environment (.venv)...
if not exist .venv (
    python -m venv .venv
) else (
    echo [SETUP] Virtual environment .venv already exists.
)

call .venv\Scripts\activate.bat

echo [SETUP] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo [SETUP] Installing core dependencies from requirements.txt...
pip install -r requirements.txt
pip install gradio

echo [SETUP] Verifying PyTorch CUDA & Environment...
python -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"

echo [SETUP] Setup completed successfully!
echo [USAGE] To launch the perception demo:
echo     python scripts\run_demo.py scenario:=pedestrian weather:=rain
echo [USAGE] To launch the interactive Gradio UI:
echo     python app.py

