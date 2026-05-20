from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _object_pos(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_pos_w


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_pos_w[:, 0, :]


def reaching_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    std: float = 0.12,
) -> torch.Tensor:
    """Reward the TCP for moving close to an object."""
    distance = torch.linalg.vector_norm(_object_pos(env, object_cfg) - _ee_pos(env, ee_frame_cfg), dim=1)
    return torch.exp(-(distance / std) ** 2)


def lifting_object(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_height: float = 0.0203,
    lift_height: float = 0.08,
) -> torch.Tensor:
    """Reward lifting an object above its table-resting height."""
    object_height = _object_pos(env, object_cfg)[:, 2]
    return torch.clamp((object_height - table_height) / lift_height, min=0.0, max=1.0)


def aligning_object_over_object(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_std: float = 0.08,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Reward horizontal alignment while the upper object is lifted over the lower object."""
    upper_pos = _object_pos(env, upper_object_cfg)
    lower_pos = _object_pos(env, lower_object_cfg)
    xy_dist = torch.linalg.vector_norm(upper_pos[:, :2] - lower_pos[:, :2], dim=1)
    lifted_gate = torch.clamp((upper_pos[:, 2] - lower_pos[:, 2]) / height_diff, min=0.0, max=1.0)
    return torch.exp(-(xy_dist / xy_std) ** 2) * lifted_gate


def object_stacked_without_gripper_check(
    env: ManagerBasedRLEnv,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.01,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Reward a pair of cubes being stacked, without assuming finger joints exist."""
    upper_pos = _object_pos(env, upper_object_cfg)
    lower_pos = _object_pos(env, lower_object_cfg)
    pos_diff = upper_pos - lower_pos
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)
    z_error = torch.abs(pos_diff[:, 2] - height_diff)
    return torch.logical_and(xy_dist < xy_threshold, z_error < height_threshold).float()


def three_cube_stack_success(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    xy_threshold: float = 0.05,
    height_threshold: float = 0.012,
    height_diff: float = 0.0468,
) -> torch.Tensor:
    """Check the full three-cube stack without assuming finger joints exist."""
    cube_2_on_1 = object_stacked_without_gripper_check(
        env,
        upper_object_cfg=cube_2_cfg,
        lower_object_cfg=cube_1_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        height_diff=height_diff,
    )
    cube_3_on_2 = object_stacked_without_gripper_check(
        env,
        upper_object_cfg=cube_3_cfg,
        lower_object_cfg=cube_2_cfg,
        xy_threshold=xy_threshold,
        height_threshold=height_threshold,
        height_diff=height_diff,
    )
    return torch.logical_and(cube_2_on_1.bool(), cube_3_on_2.bool())
