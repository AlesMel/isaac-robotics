# Copyright (c) 2026.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print action, joint, and scene diagnostics for UR3e stack tasks.

Run this with Isaac Lab's Python launcher, for example:

    ./isaaclab.sh -p scripts/ur3e/diagnose_stack_env.py \
        --task Isaac-Robots-Stack-Cube-UR3e-Robotiq-2F85-Joint-Pos-v0 \
        --num_envs 1 --headless
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Diagnose UR3e stack environment action and joint setup.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create.")
parser.add_argument("--task", type=str, required=True, help="Name of the Isaac Lab task to diagnose.")
parser.add_argument(
    "--broad-arm-pattern",
    action="append",
    default=[".*_joint"],
    help="Regex to test as the broad/legacy arm pattern. Can be passed multiple times.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, slice):
        return [str(value)]
    if isinstance(value, (str, bytes)):
        return [value.decode() if isinstance(value, bytes) else value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _regex_matches(patterns: list[str], names: list[str]) -> list[str]:
    return [name for name in names if any(re.fullmatch(pattern, name) for pattern in patterns)]


def _print_action_manager(env) -> None:
    action_manager = env.action_manager
    print("Action manager:", action_manager)
    print("Action dim:", getattr(action_manager, "total_action_dim", None))

    terms = getattr(action_manager, "_terms", {})
    for name, term in terms.items():
        print(name, "action_dim=", getattr(term, "action_dim", None), term)
        for attr in ["joint_names", "_joint_names", "_joint_ids", "asset"]:
            if hasattr(term, attr):
                print(" ", attr, getattr(term, attr))


def _print_scene_diagnostics(env) -> None:
    scene = env.scene
    scene_keys = set()
    if hasattr(scene, "keys"):
        try:
            scene_keys.update(scene.keys())
        except TypeError:
            pass
    for attr in [
        "_entities",
        "_articulations",
        "_rigid_objects",
        "_rigid_object_collections",
        "_sensors",
        "_extras",
    ]:
        mapping = getattr(scene, attr, {})
        if isinstance(mapping, dict):
            scene_keys.update(mapping.keys())

    print("Scene has surface_gripper:", "surface_gripper" in scene_keys)
    print("Scene keys:", sorted(scene_keys))
    if "ee_frame" in scene_keys:
        print("ee_frame debug_vis:", getattr(scene["ee_frame"].cfg, "debug_vis", None))


def _print_robot_joint_diagnostics(env) -> None:
    from isaac_robots.tasks.manager_based.stack.config.ur3e_gripper.stack_joint_pos_env_cfg import (
        UR3E_ARM_JOINT_NAMES,
        UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES,
    )

    robot = env.scene["robot"]
    all_joints = list(robot.data.joint_names)
    print("All robot joints:")
    for index, name in enumerate(all_joints):
        print(index, name)

    broad_matches = _regex_matches(args_cli.broad_arm_pattern, all_joints)
    arm_matches = _regex_matches(UR3E_ARM_JOINT_NAMES, all_joints)
    gripper_matches = _regex_matches(UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES, all_joints)

    print(f"Matches for broad arm patterns {args_cli.broad_arm_pattern}:")
    print(broad_matches)
    print(f"Matches for UR3E_ARM_JOINT_NAMES {UR3E_ARM_JOINT_NAMES}:")
    print(arm_matches)
    print(f"Matches for UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES {UR3E_ROBOTIQ_GRIPPER_JOINT_NAMES}:")
    print(gripper_matches)

    arm_term = getattr(env.action_manager, "_terms", {}).get("arm_action")
    if arm_term is not None:
        matched_arm_names = [str(name) for name in _as_list(getattr(arm_term, "_joint_names", []))]
        overlap = sorted(set(matched_arm_names).intersection(gripper_matches))
        print("arm_action matched gripper joints:", overlap)
        if overlap:
            print("WARNING: arm_action is controlling gripper joints. Narrow the arm joint regex immediately.")


def main() -> None:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    import isaac_robots.tasks  # noqa: F401

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.unwrapped.reset()
        _print_action_manager(env.unwrapped)
        _print_robot_joint_diagnostics(env.unwrapped)
        _print_scene_diagnostics(env.unwrapped)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
