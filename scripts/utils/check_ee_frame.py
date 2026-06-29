# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Print world positions of wrist_3_link, the ee_frame TCP, and the gripper
finger bodies at the default pose, so the TCP offset can be checked numerically.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    env.reset()
    zero = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)
    for _ in range(3):
        env.step(zero)

    robot = base.scene["robot"]
    names = robot.data.body_names
    pos = robot.data.body_pos_w[0].cpu().numpy()  # (num_bodies, 3)

    tcp = base.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()

    def p(name_substr):
        idx = [i for i, n in enumerate(names) if name_substr.lower() in n.lower()]
        return idx

    print("\n================ BODY POSITIONS (env 0, default pose) ================")
    for i, n in enumerate(names):
        print(f"  {n:40s} {np.round(pos[i], 4)}")

    wi = p("wrist_3")
    wrist = pos[wi[0]] if wi else None
    print("\n================ KEY POINTS ================")
    if wrist is not None:
        print(f"  wrist_3_link world : {np.round(wrist, 4)}")
    print(f"  TCP (ee_frame)     : {np.round(tcp, 4)}")
    if wrist is not None:
        print(f"  TCP - wrist (world delta) : {np.round(tcp - wrist, 4)}  |len|={np.linalg.norm(tcp-wrist):.4f}")

    # try to locate fingertip / pad bodies
    pad_idx = []
    for key in ("inner_finger", "finger_pad", "pad", "fingertip", "tip"):
        pad_idx = p(key)
        if pad_idx:
            print(f"\n  finger bodies matched on '{key}':")
            for i in pad_idx:
                print(f"    {names[i]:40s} {np.round(pos[i],4)}  (dist to TCP {np.linalg.norm(pos[i]-tcp):.4f})")
            break

    if len(pad_idx) >= 2:
        mid = pos[pad_idx].mean(axis=0)
        print(f"\n  fingertip-body midpoint  : {np.round(mid, 4)}")
        print(f"  TCP - fingertip midpoint : {np.round(tcp - mid, 4)}  |len|={np.linalg.norm(tcp-mid):.4f}")
        if wrist is not None:
            print(f"  wrist -> fingertip mid   : |len|={np.linalg.norm(mid-wrist):.4f}  (this is the offset you want)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
