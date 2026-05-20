from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_positions_from_pose_provider(
    env: ManagerBasedRLEnv,
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("cube_2"),
    cube_3_cfg: SceneEntityCfg = SceneEntityCfg("cube_3"),
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    dropout_value: float = 0.0,
) -> torch.Tensor:
    """Return cube XYZ positions as if they came from an external pose provider.

    During initial RL training this uses privileged sim state. Later, a camera
    model can replace this term's source while preserving the policy contract:
    ``[cube_1_xyz, cube_2_xyz, cube_3_xyz]`` in the environment-local frame.
    """
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]
    cube_3: RigidObject = env.scene[cube_3_cfg.name]

    positions = torch.cat(
        (
            cube_1.data.root_pos_w - env.scene.env_origins,
            cube_2.data.root_pos_w - env.scene.env_origins,
            cube_3.data.root_pos_w - env.scene.env_origins,
        ),
        dim=1,
    )

    if noise_std > 0.0:
        positions = positions + torch.randn_like(positions) * noise_std

    if dropout_prob > 0.0:
        cube_positions = positions.view(env.num_envs, 3, 3)
        dropout_mask = torch.rand(env.num_envs, 3, 1, device=positions.device) < dropout_prob
        positions = torch.where(dropout_mask, torch.full_like(cube_positions, dropout_value), cube_positions).view(
            env.num_envs, 9
        )

    return positions
