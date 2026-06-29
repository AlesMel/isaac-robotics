# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""1-minute test: does the 2F-85 PhysX mimic actually CLOSE the gripper linkage?

Holds the arm at its default ready pose and commands the gripper both ways
(+1 and -1), so we don't depend on the BinaryJointAction sign convention.
For each phase it prints `finger_joint` (the only driven joint) and the 5
mimic-coupled joints.

Interpretation:
  * The phase that drives finger_joint toward ~0.82 rad (47 deg) is CLOSE.
  * In that phase the 5 mimic joints SHOULD also move (gearing +-1).
  * If finger_joint closes but the mimic joints stay ~0 -> mimic is INERT.
    That is the "scrape from underneath / pin against body / never close the
    pads" bug: the linkage never forms a grip, so the policy avoids grasping.

Run (watch in the GUI):
  conda activate env_isaaclab
  python scripts/utils/test_gripper_mimic.py
Headless (numbers only, faster):
  python scripts/utils/test_gripper_mimic.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-v0")
parser.add_argument("--steps", type=int, default=120, help="env steps to hold each command")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

ARM = ("shoulder", "elbow", "wrist")


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]

    names = list(robot.joint_names)
    grip = [i for i, n in enumerate(names) if not any(a in n for a in ARM)]
    finger = names.index("finger_joint") if "finger_joint" in names else grip[0]
    mimic = [i for i in grip if i != finger]

    act = torch.zeros((base.num_envs, base.action_manager.total_action_dim), device=base.device)

    def phase(gripper_val):
        env.reset()
        act[:] = 0.0
        act[:, -1] = gripper_val  # last action dim is the gripper (arm held at default)
        for _ in range(args_cli.steps):
            env.step(act)
        q = robot.data.joint_pos[0]
        print(f"\n--- gripper action = {gripper_val:+.1f} ---")
        print(f"  finger_joint = {q[finger].item():+.4f} rad ({torch.rad2deg(q[finger]).item():+.1f} deg)")
        for i in mimic:
            print(f"  {names[i]:36s} = {q[i].item():+.4f} rad")
        return q[finger].item(), max(abs(q[i].item()) for i in mimic)

    print("\ngripper joints found:", [names[i] for i in grip])
    f_pos, m_pos = phase(+1.0)
    f_neg, m_neg = phase(-1.0)

    close_f, close_m = (f_pos, m_pos) if abs(f_pos) > abs(f_neg) else (f_neg, m_neg)
    print("\n================ VERDICT ================")
    print(f"CLOSE drives finger_joint to {close_f:+.3f} rad; max mimic-joint motion = {close_m:.3f} rad")
    if abs(close_f) < 0.1:
        print("  finger_joint barely moved -> gripper not actually being driven (check action wiring / sign).")
    elif close_m < 0.02:
        print("  >>> MIMIC INERT: finger_joint closes but the 5 linkage joints DON'T follow.")
        print("      Pads never converge -> policy scoops. Fix: re-export USD so the mimic's")
        print("      referenceJoint paths survive, OR drive all 6 gripper joints with a")
        print("      coordinated open/close action (gearing +-1) to bypass the mimic.")
    else:
        print("  >>> MIMIC OK: the linkage joints follow finger_joint. If it still scoops,")
        print("      grip is too weak / cube too big -> raise gripper effort_limit_sim & stiffness,")
        print("      bump pad friction, and/or shrink cube scale 0.8 -> 0.5.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
