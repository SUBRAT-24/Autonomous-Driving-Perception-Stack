#!/usr/bin/env python3
"""
CARLA Smoke Test Scenario.
Connects to a running CARLA Simulator (default localhost:2000),
spawns an ego vehicle, enables autopilot, attaches an RGB camera and a Depth camera,
and records 10 synchronized frames of each to disk in data/sample/smoke_test/.
"""

import argparse
import os
import queue
import sys
import time

try:
    import carla
except ImportError:
    print("[ERROR] 'carla' module not found. Please install the CARLA Python API.")
    print("Refer to HUMAN_TODO.md or run 'pip install carla'.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="CARLA Perception Smoke Test")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA RPC port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Connection timeout in seconds")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames to save")
    parser.add_argument("--output-dir", default="data/sample/smoke_test", help="Directory to save images")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.output_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "depth"), exist_ok=True)

    print(f"Connecting to CARLA simulator at {args.host}:{args.port}...")
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        world = client.get_world()
    except Exception as e:
        print(f"[ERROR] Could not connect to CARLA at {args.host}:{args.port}: {e}")
        print("Please ensure CARLA Simulator is running (e.g. CarlaUE4.exe or ./CarlaUE4.sh).")
        sys.exit(1)

    print(f"Connected to CARLA! Map: {world.get_map().name}")

    actor_list = []
    original_settings = world.get_settings()

    try:
        # Enable synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS simulation rate
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # 1. Spawn Ego Vehicle
        vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)
        print(f"Spawned ego vehicle: {vehicle.type_id} at {spawn_point.location}")

        # Queues for sensor synchronization
        rgb_queue = queue.Queue()
        depth_queue = queue.Queue()

        # 2. Attach RGB Camera
        # Location: x=1.6 (hood/windshield), z=1.7 (driver eye level)
        cam_transform = carla.Transform(carla.Location(x=1.6, z=1.7))

        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", "1280")
        rgb_bp.set_attribute("image_size_y", "720")
        rgb_bp.set_attribute("fov", "90")
        rgb_camera = world.spawn_actor(rgb_bp, cam_transform, attach_to=vehicle)
        actor_list.append(rgb_camera)
        rgb_camera.listen(rgb_queue.put)

        # 3. Attach Depth Camera (aligned exactly with RGB camera)
        depth_bp = blueprint_library.find("sensor.camera.depth")
        depth_bp.set_attribute("image_size_x", "1280")
        depth_bp.set_attribute("image_size_y", "720")
        depth_bp.set_attribute("fov", "90")
        depth_camera = world.spawn_actor(depth_bp, cam_transform, attach_to=vehicle)
        actor_list.append(depth_camera)
        depth_camera.listen(depth_queue.put)

        print(f"Sensors attached. Capturing {args.num_frames} frames...")

        # Warmup a few frames
        for _ in range(10):
            world.tick()

        # Capture requested frames
        for i in range(args.num_frames):
            world.tick()
            rgb_img = rgb_queue.get(timeout=2.0)
            depth_img = depth_queue.get(timeout=2.0)

            rgb_path = os.path.join(args.output_dir, "rgb", f"frame_{i:04d}.png")
            depth_path = os.path.join(args.output_dir, "depth", f"frame_{i:04d}.png")

            # Save raw RGB image
            rgb_img.save_to_disk(rgb_path)

            # Save depth image converted to logarithmic depth representation for visual inspection
            depth_img.save_to_disk(depth_path, carla.ColorConverter.LogarithmicDepth)

            print(f"  Frame {i+1}/{args.num_frames} saved -> {rgb_path}")

        print(f"\n[SUCCESS] Smoke test completed! Images stored in: {args.output_dir}")

    finally:
        print("\nCleaning up actors and restoring world settings...")
        world.apply_settings(original_settings)
        for actor in reversed(actor_list):
            if actor.is_alive:
                actor.destroy()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
