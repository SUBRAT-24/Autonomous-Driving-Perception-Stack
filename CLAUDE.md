# CLAUDE.md — Agent Context for This Repo

## Project
Autonomous Driving Perception Stack — B.Tech capstone project by Subrat.

## Tech Stack
- **Detection**: YOLOv8/v11 (Ultralytics) fine-tuned on BDD100K
- **Tracking**: ByteTrack (primary), DeepSORT (comparison)
- **Lane Detection**: Classical OpenCV + segmentation model
- **Depth**: MiDaS small (monocular depth estimation)
- **Optimization**: ONNX export → TensorRT
- **Sim & Middleware**: CARLA 0.9.15+ → ROS2 Humble
- **Logging**: Weights & Biases
- **Testing**: pytest

## Working Rules Summary
1. One phase at a time; wait for human OK before next phase.
2. Never fabricate metrics — every number must trace to a script run or W&B log.
3. Pinned deps, YAML configs, fixed seeds, one-command entry points.
4. Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `perf:`).
5. Never commit datasets, weights, `.engine` files, or videos.
6. Type hints, docstrings, small functions, logging (no print).
7. Anything needing human action → HUMAN_TODO.md.
8. Ask before destructive/system-wide actions.
9. Log decisions in DECISIONS.md, explanations in LEARNING_NOTES.md.
10. Simplest-first, then depth (comparisons, robustness, failure analysis).
11. Two failures on same error → stop, report, propose two alternatives.

## Repo Layout
See PLAN.md for full directory tree and milestone schedule.
