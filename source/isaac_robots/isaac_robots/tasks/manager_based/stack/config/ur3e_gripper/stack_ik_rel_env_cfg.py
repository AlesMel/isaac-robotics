# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg


UR3E_IK_ACTION_SCALE = (0.05, 0.05, 0.05, 0.15, 0.15, 0.15)
UR3E_REVOLUTE_TARGET_LIMIT = 2.0 * math.pi


class SafeDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK action that keeps PhysX revolute targets in a valid range."""

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()

        soft_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        physx_lower = torch.full_like(joint_pos_des, -UR3E_REVOLUTE_TARGET_LIMIT)
        physx_upper = torch.full_like(joint_pos_des, UR3E_REVOLUTE_TARGET_LIMIT)
        joint_pos_des = torch.clamp(
            joint_pos_des,
            min=torch.maximum(soft_limits[..., 0], physx_lower),
            max=torch.minimum(soft_limits[..., 1], physx_upper),
        )
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class SafeDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Differential IK config that clamps final UR3e joint targets before PhysX."""

    class_type: type = SafeDifferentialInverseKinematicsAction


@configclass
class UR3eRobotiq2F85CubeStackEnvCfg(stack_joint_pos_env_cfg.UR3eRobotiq2F85CubeStackEnvCfg):
    """Configuration for the UR3e Robotiq 2F-85 Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR3e + Robotiq 2F-85)
        self.actions.arm_action = SafeDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=stack_joint_pos_env_cfg.UR3E_ARM_JOINT_NAMES,
            body_name=stack_joint_pos_env_cfg.UR3E_ROBOTIQ_EE_BODY_NAME,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=UR3E_IK_ACTION_SCALE,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=stack_joint_pos_env_cfg.UR3E_ROBOTIQ_EE_OFFSET
            ),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class UR3eLongSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR3eLongSuctionCubeStackEnvCfg):
    """Configuration for the UR3e Long Suction Cube Stack Environment."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR3e LONG SUCTION)
        self.actions.arm_action = SafeDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=stack_joint_pos_env_cfg.UR3E_ARM_JOINT_NAMES,
            body_name="tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=UR3E_IK_ACTION_SCALE,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.22]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


@configclass
class UR3eShortSuctionCubeStackEnvCfg(stack_joint_pos_env_cfg.UR3eShortSuctionCubeStackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (UR3e SHORT SUCTION)
        self.actions.arm_action = SafeDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=stack_joint_pos_env_cfg.UR3E_ARM_JOINT_NAMES,
            body_name="tool0",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=UR3E_IK_ACTION_SCALE,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, -0.159]),
        )

        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.02,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )
