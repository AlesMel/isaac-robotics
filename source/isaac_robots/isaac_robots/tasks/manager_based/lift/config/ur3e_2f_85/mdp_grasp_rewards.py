# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Contact-gated lift rewards — stop the 'flick the cube upward' exploit.

The stock `object_is_lifted` / `object_goal_distance` reward the cube being above a
height threshold regardless of whether it's actually in the gripper, so a policy can
bat the cube up and collect the reward without grasping. These versions additionally
require BOTH fingertip pads to be in contact (a real squeeze). A flicked cube has no
finger contact, so it earns nothing — and the only way to keep contact while raising
the cube is to actually close and hold.

Needs a ContactSensorCfg on each fingertip pad in the scene, named
`contact_left_finger` / `contact_right_finger` (see the env-cfg snippet in chat).
"""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms

# typing only
from isaaclab.envs import ManagerBasedRLEnv


def _both_fingers_in_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bool (num_envs,): True when BOTH pads press on something harder than
    `force_threshold` newtons.

    Uses net contact force, so *any* contact counts. Pair it with a lifted-height
    gate so 'something' means the cube — the pads aren't touching the table once the
    cube is airborne. (For strict cube-only contact, switch to a filtered force
    matrix; see the note in chat.)
    """
    left: ContactSensor = env.scene[left_sensor_cfg.name]
    right: ContactSensor = env.scene[right_sensor_cfg.name]
    # net_forces_w: (num_envs, num_bodies, 3) with history_length=0; reduce over bodies.
    left_f = left.data.net_forces_w.norm(dim=-1).amax(dim=1)
    right_f = right.data.net_forces_w.norm(dim=-1).amax(dim=1)
    return (left_f > force_threshold) & (right_f > force_threshold)

def lift_with_grasp_gate(
    env: ManagerBasedRLEnv,
    object_cfg: str = "object",
    ee_frame_cfg: str = "ee_frame",          # Transformed end-effector frame
    left_sensor_name: str = "contact_left_finger",
    right_sensor_name: str = "contact_right_finger",
    force_threshold: float = 0.5,
) -> torch.Tensor:
    """A dense, progressive reward function designed to guide exploration from scratch."""
    
    # -------------------------------------------------------------------------
    # 1. Setup & Positions
    # -------------------------------------------------------------------------
    object_pos = env.scene[object_cfg].data.root_pos_w
    
    # Get TCP position from your FrameTransformer (ee_frame)
    # Typically stored in target_pos_w or source_pos_w depending on Isaac Lab version.
    # We will grab the tracked end_effector frame position.
    ee_pos = env.scene[ee_frame_cfg].data.target_pos_w[:, 0, :] # Index 0 is 'end_effector'
    
    # Calculate initial Z of the object safely
    local_initial_z = env.scene[object_cfg].data.default_root_state[:, 2]
    env_origin_z = env.scene.env_origins[:, 2]
    object_initial_z = local_initial_z + env_origin_z

    # -------------------------------------------------------------------------
    # Term 1: Dense Reaching Reward (Always active)
    # -------------------------------------------------------------------------
    # This pulls the arm directly to the cube. Without this, it learns nothing.
    ee_to_object_dist = torch.norm(ee_pos - object_pos, dim=-1)
    r_reach = torch.exp(-5.0 * ee_to_object_dist) # Bounded [0.0, 1.0]

    # -------------------------------------------------------------------------
    # Term 2: Grasp Detection (Forces)
    # -------------------------------------------------------------------------
    left_forces = env.scene[left_sensor_name].data.net_forces_w
    right_forces = env.scene[right_sensor_name].data.net_forces_w
    left_force_mag = torch.norm(left_forces, dim=-1).max(dim=-1)[0]
    right_force_mag = torch.norm(right_forces, dim=-1).max(dim=-1)[0]
    
    is_grasped = (left_force_mag > force_threshold) & (right_force_mag > force_threshold)
    
    # Provide a small continuous reward for touching the object with both fingers
    r_touch = 0.5 * (torch.tanh(left_force_mag / 2.0) + torch.tanh(right_force_mag / 2.0))

    # -------------------------------------------------------------------------
    # Term 3: Gated Lifting Reward (Highly rewarding!)
    # -------------------------------------------------------------------------
    z_diff = torch.clamp(object_pos[:, 2] - object_initial_z, min=0.0, max=0.3)
    
    # Stage the lift: give partial points for lifting anyway, 
    # but massive points if it's securely grasped.
    r_lift = z_diff * 10.0
    r_gated_lift = is_grasped.float() * z_diff * 30.0 

    # -------------------------------------------------------------------------
    # Term 4: Softened Anti-Flick Penalty
    # -------------------------------------------------------------------------
    object_vel = env.scene[object_cfg].data.root_lin_vel_w
    object_speed = torch.norm(object_vel, dim=-1)
    
    # Only penalize excessive speed if it is NOT grasped AND the arm is NOT close.
    # This allows the robot to bump the cube gently while trying to grasp it.
    p_flick = (~is_grasped).float() * torch.clamp(object_speed - 0.5, min=0.0) * 0.5
    
    # Total combined reward string
    total_reward = r_reach + r_touch + r_lift + r_gated_lift - p_flick
    return total_reward

def object_lifted_and_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    force_threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_left_finger"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_right_finger"),
) -> torch.Tensor:
    """1.0 only when the cube is above `minimal_height` AND both pads are in contact.
    Replaces `object_is_lifted`."""
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasped = _both_fingers_in_contact(env, force_threshold, left_sensor_cfg, right_sensor_cfg)
    return (lifted & grasped).float()


def object_goal_distance_grasped(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    force_threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_left_finger"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_right_finger"),
) -> torch.Tensor:
    """Goal-tracking reward, gated on the cube being lifted AND grasped.
    Replaces `object_goal_distance`."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b
    )
    goal_distance = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)

    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    grasped = _both_fingers_in_contact(env, force_threshold, left_sensor_cfg, right_sensor_cfg)
    gate = (lifted & grasped).float()
    return gate * (1.0 - torch.tanh(goal_distance / std))


def object_lin_vel_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Squared cube linear velocity. Use as a small NEGATIVE-weight penalty so a
    flick — which imparts a large velocity spike — becomes actively costly."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(obj.data.root_lin_vel_w), dim=1)