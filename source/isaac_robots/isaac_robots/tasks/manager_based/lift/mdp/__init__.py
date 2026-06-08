# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export upstream Isaac Lab lift MDP helpers.

This pulls in all of ``isaaclab.envs.mdp`` (joint_pos_rel, last_action,
reset_root_state_uniform, action_rate_l2, time_out, root_height_below_minimum,
modify_reward_weight, JointPositionActionCfg, BinaryJointPositionActionCfg,
UniformPoseCommandCfg, generated_commands, ...) plus the lift-specific extras
(object_is_lifted, object_ee_distance, object_goal_distance,
object_position_in_robot_root_frame, object_reached_goal).
"""

from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403

# Local overrides / additions (must come AFTER the wildcard import so they win).
from .rewards import object_goal_distance_gaussian, object_goal_distance_velocity_gated  # noqa: F401
