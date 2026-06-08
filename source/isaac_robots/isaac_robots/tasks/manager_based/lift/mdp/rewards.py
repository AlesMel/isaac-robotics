# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward helpers for the UR3e + Hand-E lift task.

The single helper here replaces upstream's tanh-based fine-grained goal
tracking with a Gaussian kernel. The motivation (verified empirically in
run 2026-06-06_22-24-24): tanh has nonzero gradient at d=0, so even when
the cube is exactly at the goal the policy feels a strong corrective
"pull" (gradient = -1/std = -20 at d=0 for std=0.05). Cube weight perturbs
the arm by ~1 mm/step -> tanh reward drops -> policy overshoots correcting
-> limit cycle.

Gaussian kernel ``exp(-(d/std)^2)`` has the same maximum (1.0 at d=0) but
**zero gradient** at d=0, so no corrective force when the cube is at the
goal. Limit cycle dissolves at the source.

Use only on the fine-grained term; keep upstream tanh on the coarse
(std=0.3) term so its strong far-field gradient still drives approach.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_goal_distance_gaussian(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward tracking the goal pose with a Gaussian kernel (smooth peak at d=0).

    Kept for reference. Gaussian kernel removed the limit cycle but caused
    late-stage policy drift (no centering gradient at d=0). See README
    section 6.1 and use ``object_goal_distance_velocity_gated`` instead.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    lifted = object.data.root_pos_w[:, 2] > minimal_height
    return lifted.float() * torch.exp(-((distance / std) ** 2))


def object_goal_distance_velocity_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    velocity_thresh: float = 0.5,
    neighborhood: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """tanh tracking reward gated by joint velocity inside the goal neighborhood.

    Solves the limit-cycle vs centering-pull trade-off:
      * Outside neighborhood (d > neighborhood): full tanh tracking reward
        applies. Strong gradient at all distances -> reach is undisturbed.
      * Inside neighborhood (d <= neighborhood): tracking is multiplied by
        a linear velocity gate. ``vel_factor = clip(1 - |q_dot|/thresh, 0, 1)``.
        Joints still -> full reward; joints moving fast -> reward suppressed.

    Net effect: at the goal pose, the policy only collects full reward when
    the arm is held still. Any oscillation costs reward. Yet the tanh
    centering pull is preserved (gradient at d=0 stays nonzero), so the
    policy is not free to drift away post-convergence the way the Gaussian
    kernel allowed.

    Pass ``robot_cfg=SceneEntityCfg("robot", joint_names=[...])`` to filter
    which joints contribute to the velocity magnitude (e.g. arm only,
    excluding gripper sliders).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    lifted = object.data.root_pos_w[:, 2] > minimal_height

    tracking = 1.0 - torch.tanh(distance / std)

    joint_vels = robot.data.joint_vel[:, robot_cfg.joint_ids]
    joint_vel_mag = torch.norm(joint_vels, dim=1)
    vel_factor = torch.clamp(1.0 - joint_vel_mag / velocity_thresh, min=0.0, max=1.0)

    inside = (distance < neighborhood).float()
    gate = inside * vel_factor + (1.0 - inside) * 1.0

    return lifted.float() * tracking * gate
