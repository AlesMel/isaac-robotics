# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stress-test the articulation with random actions and report the peak
per-joint |velocity|. If specific joints run away to huge values, the sim is
numerically unstable there. Headless, no rendering."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-v0")
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=64)
    env = gym.make(args_cli.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    names = robot.data.joint_names

    env.reset()
    n_act = base.action_manager.total_action_dim
    peak = torch.zeros(len(names), device=base.device)
    first_blow_step = -1

    for t in range(args_cli.steps):
        act = torch.randn((base.num_envs, n_act), device=base.device)  # policy-like noise
        env.step(act)
        v = robot.data.joint_vel.abs().amax(dim=0)  # max over envs, per joint
        peak = torch.maximum(peak, v)
        if first_blow_step < 0 and v.max().item() > 1e3:
            first_blow_step = t

    order = torch.argsort(peak, descending=True)
    lines = [f"steps={args_cli.steps}  first |vel|>1e3 at step={first_blow_step}",
             "peak |joint_vel| (rad/s or m/s), worst first:"]
    for i in order.tolist():
        lines.append(f"  {names[i]:34s} {peak[i].item():.4g}")

    with open("/tmp/jointvel_result.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
