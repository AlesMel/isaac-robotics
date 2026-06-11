# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda + DexCube lift task with swappable fine-grained reward kernel.

Wraps the upstream
``isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg.FrankaCubeLiftEnvCfg``
and overrides the ``object_goal_tracking_fine_grained`` reward term to use one
of the kernels in :mod:`velgate_bench.kernels`.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

from velgate_bench.kernels import get_kernel

##
# Per-kernel arm-joint name patterns for velocity-gate filtering.
##
FRANKA_ARM_JOINT_NAMES = ["panda_joint.*"]

##
# Gym ID template
##
_GYM_ID_TEMPLATE = "Velgate-Bench-Franka-Lift-{kernel_display}-v0"

_KERNEL_DISPLAY = {
    "tanh": "Tanh",
    "gaussian": "Gaussian",
    "velocity_gated_tanh": "VelocityGatedTanh",
    "tanh_additive_velpen": "TanhAdditiveVelPen",
    "velocity_gated_tanh_smooth": "VelocityGatedTanhSmooth",
}


def _make_franka_lift_cfg(kernel_name: str):
    """Factory that returns a configclass subclass of FrankaCubeLiftEnvCfg
    with the requested fine-grained kernel installed."""

    kernel_func, kernel_default_kwargs = get_kernel(kernel_name)

    @configclass
    class _Cfg(FrankaCubeLiftEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            # Override the fine-grained term's func and inject kernel kwargs.
            # Standard params (std, minimal_height, command_name) stay the
            # same; only the function and any kernel-specific extras change.
            self.rewards.object_goal_tracking_fine_grained.func = kernel_func
            new_params = {
                "std": 0.05,
                "minimal_height": 0.04,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot", joint_names=FRANKA_ARM_JOINT_NAMES),
                "object_cfg": SceneEntityCfg("object"),
            }
            new_params.update(kernel_default_kwargs)
            self.rewards.object_goal_tracking_fine_grained.params = new_params

    _Cfg.__name__ = f"FrankaLift{_KERNEL_DISPLAY[kernel_name]}EnvCfg"
    _Cfg.__qualname__ = _Cfg.__name__
    return _Cfg


def register_franka_lift_variants() -> None:
    """Register Velgate-Bench-Franka-Lift-{Tanh,Gaussian,VelocityGatedTanh}-v0."""
    for kernel_name, kernel_display in _KERNEL_DISPLAY.items():
        gym_id = _GYM_ID_TEMPLATE.format(kernel_display=kernel_display)
        if gym_id in gym.registry:
            continue
        cfg_cls = _make_franka_lift_cfg(kernel_name)
        # Store the cfg class on the module so the entry point string resolves.
        globals()[cfg_cls.__name__] = cfg_cls
        gym.register(
            id=gym_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{cfg_cls.__name__}",
                "skrl_cfg_entry_point": (
                    "isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents:skrl_ppo_cfg.yaml"
                ),
            },
            disable_env_checker=True,
        )
