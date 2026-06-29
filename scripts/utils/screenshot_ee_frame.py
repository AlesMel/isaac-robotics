# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the lift scene at t=0 (robot in default pose) with the ee_frame TCP
marker enabled, from several camera angles, so the TCP offset between the
gripper jaws can be inspected visually.

Usage:
    python scripts/utils/screenshot_ee_frame.py \
        --task Isaac-Lift-Cube-UR3e-2F85-Play-v0 --out /tmp/ee_frame
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Screenshot the ee_frame TCP marker at t=0.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-Play-v0")
parser.add_argument("--out", type=str, default="/tmp/ee_frame")
parser.add_argument("--cam_dist", type=float, default=0.30, help="camera distance from TCP (m)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# screenshots require offscreen rendering
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below runs after the Kit app is up ----
import os

import numpy as np
import torch
from PIL import Image

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401  # registers the isaac_robots gym envs
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    # turn ON the TCP frame marker (config ships with it off)
    env_cfg.scene.ee_frame.debug_vis = True
    env_cfg.viewer.resolution = (1280, 720)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped

    # reset -> robot goes to its default joint pose (t=0)
    env.reset()
    # a couple of zero-action steps so sensors + debug markers populate;
    # with use_default_offset the arm holds the default pose.
    zero = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
    for _ in range(3):
        env.step(zero)

    # TCP (end-effector frame) world position = where the marker sits
    tcp = base.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    print(f"[INFO] TCP marker world position: {tcp}")

    os.makedirs(args_cli.out, exist_ok=True)
    d = args_cli.cam_dist
    views = {
        "front_x": tcp + np.array([d, 0.0, 0.03]),
        "side_y": tcp + np.array([0.0, d, 0.03]),
        "side_ny": tcp + np.array([0.0, -d, 0.03]),
        "iso": tcp + np.array([d * 0.7, d * 0.7, d * 0.7]),
    }

    for name, eye in views.items():
        base.sim.set_camera_view(eye=tuple(float(v) for v in eye), target=tuple(float(v) for v in tcp))
        # warm up the renderer (first frames come back empty)
        rgb = None
        for _ in range(12):
            rgb = env.render()
        if rgb is None or rgb.size == 0:
            print(f"[WARN] empty frame for view '{name}'")
            continue
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        path = os.path.join(args_cli.out, f"ee_frame_{name}.png")
        Image.fromarray(rgb.astype(np.uint8)).save(path)
        print(f"[INFO] saved {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
