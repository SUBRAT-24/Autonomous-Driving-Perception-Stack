# Autonomous Driving Perception Stack — Makefile
# Usage: make <target>
# On Windows without make, run the commands directly from scripts/

.PHONY: help install test lint eda train track lanes depth export benchmark demo clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run all pytest tests
	pytest tests/ -v --tb=short

lint: ## Run basic linting
	python -m py_compile src/data/bdd100k_to_yolo.py
	python -m py_compile src/detection/train.py
	python -m py_compile src/depth/distance.py

eda: ## Run exploratory data analysis
	python -m src.data.eda --config configs/data_config.yaml

train: ## Fine-tune YOLO on BDD100K
	python -m src.detection.train --config configs/train_config.yaml

track: ## Run ByteTrack on validation set
	python -m src.tracking.bytetrack_runner --config configs/tracker_config.yaml

lanes: ## Run lane detection comparison
	python -m src.lane.evaluate --config configs/lane_config.yaml

depth: ## Run MiDaS depth estimation
	python -m src.depth.midas_runner --config configs/depth_config.yaml

export: ## Export models to ONNX
	python -m src.optimize.export_onnx --config configs/benchmark_config.yaml

benchmark: ## Benchmark PyTorch vs ONNX vs TensorRT
	python -m src.optimize.benchmark --config configs/benchmark_config.yaml

demo: ## Launch full CARLA + ROS2 demo (requires ROS2 sourced)
	ros2 launch perception_bringup full_pipeline.launch.py

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ dist/ build/ *.egg-info/
