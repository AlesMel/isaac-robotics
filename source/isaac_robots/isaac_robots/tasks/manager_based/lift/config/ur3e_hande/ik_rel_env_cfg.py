# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg
from .joint_pos_env_cfg import UR3E_ARM_JOINT_NAMES, UR3E_HANDE_EE_BODY_NAME, UR3E_HANDE_EE_OFFSET


@configclass
class UR3eHandECubeLiftEnvCfg(joint_pos_env_cfg.UR3eHandECubeLiftEnvCfg):
    """UR3e + Robotiq Hand-E lift-cube task with relative-pose differential IK control."""

    def __post_init__(self):
        super().__post_init__()

        # Swap joint-position arm action for a relative-pose DLS IK action.
        # UR3E_ROBOTIQ_HANDE_CFG already uses Franka-high-PD-class stiffness
        # (1320 / 600 / 216), so we don't need a separate "high PD" robot variant.
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=UR3E_ARM_JOINT_NAMES,
            body_name=UR3E_HANDE_EE_BODY_NAME,
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=list(UR3E_HANDE_EE_OFFSET)),
        )


@configclass
class UR3eHandECubeLiftEnvCfg_PLAY(UR3eHandECubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
