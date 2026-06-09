# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Reward kernels under comparison in the velocity-gated reward benchmark.

All three kernels share the same signature and the same lifted-gate
condition. They differ ONLY in how they shape the reward as a function of
goal distance (and, for the velocity-gated kernel, joint velocity).

Kernel definitions
------------------

Let ``d`` be the Euclidean distance from the manipulated object to the
goal pose (both in world frame), ``σ`` the kernel scale parameter, and
``lifted`` the gating condition that the object is above a minimum height.

1. **tanh** (upstream Isaac Lab default):

       R(d) = lifted * (1 - tanh(d / σ))

   Maximum 1.0 at d=0. **Nonzero gradient at d=0**: ∂R/∂d|_{d=0} = -1/σ.
   This is the root cause of the limit cycle: any micro-perturbation
   from d=0 produces a strong corrective signal, which the policy
   overshoots, producing a visible oscillation.

2. **gaussian**:

       R(d) = lifted * exp(-(d / σ)²)

   Same maximum 1.0 at d=0. **Zero gradient at d=0**: smooth peak.
   Eliminates the limit cycle but introduces a different failure mode:
   without any centering pull, the policy slowly drifts away from the
   optimum under PPO noise after convergence (observed -20% reward over
   100k steps post-peak in our preliminary work).

3. **velocity_gated_tanh** (this work):

       R(d, q̇) = lifted * (1 - tanh(d / σ)) * G(d, q̇)
       G(d, q̇) = inside_neighborhood(d) * vel_factor(q̇) + (1 - inside_neighborhood(d))
       inside_neighborhood(d) = 1[d < r_neighborhood]
       vel_factor(q̇) = clip(1 - ‖q̇‖_arm / v_thresh, 0, 1)

   Outside the goal neighborhood, identical to vanilla tanh.
   Inside, the reward is multiplied by a linear "velocity gate" that drops
   from 1.0 (joints still) to 0.0 (joints at threshold velocity). The
   centering pull of tanh is preserved (no drift), and the policy now has
   a direct economic incentive against its own corrective oscillation (no
   limit cycle).

Ablation parameters
-------------------

For each kernel, the following parameters can be swept to study sensitivity:

- ``std``: kernel scale, common to tanh and gaussian.
- ``velocity_thresh`` (vel-gated only): rad/s magnitude across the gated
  joint set above which the gate fully closes.
- ``neighborhood`` (vel-gated only): meters; the radius around the goal
  within which the velocity gate is active. Beyond this radius the kernel
  reduces exactly to vanilla tanh.

API
---

Each kernel function has the signature expected by Isaac Lab's
``RewardManager``:

    func(env, std, minimal_height, command_name, **kwargs) -> Tensor

so it can be plugged directly into a ``RewTerm.func`` slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Shared helper: distance and lifted gate (factored out to keep the three
# kernel implementations parallel and obviously equivalent except in shape).
# -----------------------------------------------------------------------------


def _distance_and_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (distance_to_goal_world, lifted_mask)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    lifted = obj.data.root_pos_w[:, 2] > minimal_height
    return distance, lifted


# -----------------------------------------------------------------------------
# Kernel 1: vanilla tanh (upstream baseline)
# -----------------------------------------------------------------------------


def tanh_kernel(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 - tanh(d / σ), gated by lifted. Upstream Isaac Lab default."""
    distance, lifted = _distance_and_lifted(
        env, minimal_height, command_name, robot_cfg, object_cfg
    )
    return lifted.float() * (1.0 - torch.tanh(distance / std))


# -----------------------------------------------------------------------------
# Kernel 2: Gaussian
# -----------------------------------------------------------------------------


def gaussian_kernel(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """exp(-(d / σ)²), gated by lifted. Smooth peak at d=0 (zero gradient)."""
    distance, lifted = _distance_and_lifted(
        env, minimal_height, command_name, robot_cfg, object_cfg
    )
    return lifted.float() * torch.exp(-((distance / std) ** 2))


# -----------------------------------------------------------------------------
# Kernel 3: velocity-gated tanh (this work)
# -----------------------------------------------------------------------------


def velocity_gated_tanh_kernel(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    command_name: str,
    velocity_thresh: float = 0.5,
    neighborhood: float = 0.10,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """tanh tracking gated by joint velocity inside the goal neighborhood.

    Outside the neighborhood: identical to vanilla tanh.
    Inside: reward × linear velocity gate, dropping from 1.0 (still) to 0.0
    (above velocity_thresh).

    Use ``robot_cfg=SceneEntityCfg("robot", joint_names=[...])`` to filter
    which joints contribute to the velocity magnitude (e.g. arm joints only,
    excluding gripper sliders that would otherwise spuriously trigger the
    gate during open/close events).
    """
    distance, lifted = _distance_and_lifted(
        env, minimal_height, command_name, robot_cfg, object_cfg
    )

    tracking = 1.0 - torch.tanh(distance / std)

    robot: Articulation = env.scene[robot_cfg.name]
    joint_vels = robot.data.joint_vel[:, robot_cfg.joint_ids]
    joint_vel_mag = torch.norm(joint_vels, dim=1)
    vel_factor = torch.clamp(1.0 - joint_vel_mag / velocity_thresh, min=0.0, max=1.0)

    inside = (distance < neighborhood).float()
    gate = inside * vel_factor + (1.0 - inside) * 1.0

    return lifted.float() * tracking * gate


# -----------------------------------------------------------------------------
# Registry: name -> (function, default params override)
# -----------------------------------------------------------------------------


KERNELS = {
    "tanh": (tanh_kernel, {}),
    "gaussian": (gaussian_kernel, {}),
    "velocity_gated_tanh": (
        velocity_gated_tanh_kernel,
        {"velocity_thresh": 0.5, "neighborhood": 0.10},
    ),
}
"""Map from kernel name -> (function, default kwargs to inject into RewTerm.params)."""


def get_kernel(name: str):
    """Return (function, default_kwargs) for ``name``. Raises KeyError if unknown."""
    if name not in KERNELS:
        raise KeyError(
            f"Unknown kernel '{name}'. Available: {sorted(KERNELS.keys())}"
        )
    return KERNELS[name]
