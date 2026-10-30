# PROGRESS.md — Build Log

## Phase 0 — Setup and Risk Check
- **Status**: 🟢 Complete
- **Started**: 2026-10-29
- **Completed**: 2026-10-29

### Done
- [x] Git repository initialized with remote origin
- [x] Full directory layout, configs, and .gitignore created
- [x] CLAUDE.md, PLAN.md, PROGRESS.md, DECISIONS.md, LEARNING_NOTES.md, HUMAN_TODO.md
- [x] Makefile with targets (`help`, `test`, `eda`, `train`, `track`, `lanes`, `depth`, `export`, `benchmark`, `demo`)
- [x] Pre-commit configuration (`.pre-commit-config.yaml`) with `ruff` and `black`
- [x] Environment check script (`scripts/check_env.py`) testing OS, CUDA, GPU VRAM, Torch, YOLO, CV, ONNX, TRT, ROS2, CARLA, W&B
- [x] CARLA smoke test script (`carla_scenarios/smoke_test.py`) with synchronized RGB and depth camera frame capture
- [x] Initial Phase 0 commits made (clean git hygiene)

### Metrics
_None yet — this is scaffolding._

---

_Phases 1–8 will be appended here as we go._
