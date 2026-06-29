# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke-test the grasp-gated lift rewards: build env, list active reward terms,
step a few times, and print per-term values. Headless, no rendering."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-v0")
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
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=4)
    env = gym.make(args_cli.task, cfg=cfg)
    base = env.unwrapped

    lines = ["========== ACTIVE REWARD TERMS ==========", str(base.reward_manager.active_terms)]

    env.reset()
    zero = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
    rew = None
    for _ in range(5):
        _, rew, _, _, _ = env.step(zero)

    lines += ["", "========== STEP OK ==========", f"total reward (env 0..3): {rew.detach().cpu().numpy()}"]
    lines += ["", "========== PER-TERM (env 0) =========="]
    for name, val in base.reward_manager.get_active_iterable_terms(env_idx=0):
        lines.append(f"  {name:34s} {val}")

    with open("/tmp/smoke_result.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
