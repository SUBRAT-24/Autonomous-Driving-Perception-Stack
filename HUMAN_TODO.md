# HUMAN_TODO.md — Actions Requiring Subrat's Hands

> Items the AI cannot do and needs you (Subrat) to handle.
> Check items off as you complete them.

---

## 🔴 Blocking (do these first)

### ENV-01: Confirm system details
- [x] OS: Windows 11 + WSL2 (Ubuntu 22.04)
- [x] GPU: NVIDIA RTX 3050 (4 GB)
- [x] CARLA status: Not yet installed
- [x] W&B username: subratkumar

### ENV-02: Python & CUDA environment
- [ ] Ensure Python 3.10+ is installed (`python --version`)
- [ ] Ensure NVIDIA driver + CUDA toolkit is installed (`nvidia-smi`)
- [ ] Create a virtual environment: `python -m venv .venv` and activate it

### DATA-01: Download BDD100K dataset
- [ ] Go to https://bdd-data.berkeley.edu/
- [ ] Register / log in
- [ ] Download:
  - **10K Images** (or 100K Images): `10k_images_train.zip` and `10k_images_val.zip` (or 100K images split)
  - **Detection 2020 Labels**: `bdd100k_det_20_labels_trainval.zip`
- [ ] Unpack the files so the structure matches exactly:
  ```text
  data/raw/bdd100k/
  ├── images/
  │   └── 100k/
  │       ├── train/    # ~7,000 .jpg images (e.g. 0000f77c-62c2a249.jpg)
  │       └── val/      # ~1,000 .jpg images
  └── labels/
      └── det_20/
          ├── det_train.json
          └── det_val.json
  ```
- [ ] Verify the placement by running:
  ```bash
  python scripts/verify_data.py
  ```
  (Must report `All checks PASSED — dataset is ready.` with exit code 0)

### WANDB-01: Weights & Biases setup
- [x] Account created: `subratkumar`
- [ ] Run `wandb login` in terminal and paste your API key

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
