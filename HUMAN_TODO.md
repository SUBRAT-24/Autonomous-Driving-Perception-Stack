# HUMAN_TODO.md — Actions Requiring Subrat's Hands

> Items the AI cannot do and needs you (Subrat) to handle.
> Check items off as you complete them.

---

## 🔴 Blocking (do these first)

### ENV-01: Confirm system details
Update the assumptions table in PLAN.md with your actual:
- [ ] OS (Windows 11 native? WSL2? Ubuntu?)
- [ ] GPU model and VRAM
- [ ] Whether CARLA is installed
- [ ] Your W&B username

### ENV-02: Python & CUDA environment
- [ ] Ensure Python 3.10+ is installed (`python --version`)
- [ ] Ensure NVIDIA driver + CUDA toolkit is installed (`nvidia-smi`)
- [ ] Create a virtual environment: `python -m venv .venv` and activate it

### DATA-01: Download BDD100K dataset
- [ ] Go to https://bdd-data.berkeley.edu/
- [ ] Register / log in
- [ ] Download "100K Images" (or the 10K subset if available)
- [ ] Download "Detection 2020 Labels" (JSON format)
- [ ] Place in `data/raw/bdd100k/images/` and `data/raw/bdd100k/labels/`

### WANDB-01: Weights & Biases setup
- [ ] Create account at https://wandb.ai if you don't have one
- [ ] Run `wandb login` and paste your API key
- [ ] Tell me your W&B username so I can configure the project

---

## 🟡 Soon (needed before Phase 7)

### CARLA-01: Install CARLA Simulator
- [ ] Download CARLA 0.9.15+ from https://github.com/carla-simulator/carla/releases
- [ ] Follow install instructions for your OS
- [ ] Verify it runs: `./CarlaUE4.sh` (Linux) or `CarlaUE4.exe` (Windows)
- [ ] Install the Python API: `pip install carla`

### ROS2-01: Install ROS2 Humble
- [ ] If on Ubuntu 22.04 / WSL2: follow https://docs.ros.org/en/humble/Installation.html
- [ ] Source the setup: `source /opt/ros/humble/setup.bash`
- [ ] Verify: `ros2 --version`

---

## 🟢 Nice to have (Phase 8)

### DEMO-01: Record demo video
- [ ] Screen-record the full pipeline running in CARLA (OBS or similar)
- [ ] Save as MP4, upload to YouTube/Google Drive
- [ ] Share the link for README

### PRES-01: Prepare presentation slides
- [ ] Use the outline from docs/ to build 10-minute slides
- [ ] Include: problem statement, architecture diagram, key results, challenges, next steps

### REFLECT-01: Write individual reflection
- [ ] This is your own writing — no AI help needed
- [ ] Reflect on what you learned, what was hard, what you'd do differently

---

## 🔵 Git Authentication

### GIT-01: Push to GitHub
When I say "push to origin," you may need to authenticate:
- [ ] Ensure you have push access to https://github.com/SUBRAT-24/Autonomous-Driving-Perception-Stack.git
- [ ] If using HTTPS: `git remote set-url origin https://<your-token>@github.com/SUBRAT-24/Autonomous-Driving-Perception-Stack.git`
- [ ] Or use SSH: `git remote set-url origin git@github.com:SUBRAT-24/Autonomous-Driving-Perception-Stack.git`
- [ ] Test: `git push -u origin main`
