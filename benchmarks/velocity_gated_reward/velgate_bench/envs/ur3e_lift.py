# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""UR3e + Robotiq Hand-E lift task with swappable fine-grained reward kernel.

Wraps ``isaac_robots.tasks.manager_based.lift.config.ur3e_hande.joint_pos_env_cfg.UR3eHandECubeLiftEnvCfg``
(the project's manager-based env) and overrides the fine-grained tracking
reward kernel.
"""

from __future__ import annotations

import gymnasium as gym

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaac_robots.tasks.manager_based.lift.config.ur3e_hande.joint_pos_env_cfg import (
    UR3E_ARM_JOINT_NAMES,
    UR3eHandECubeLiftEnvCfg,
)

from velgate_bench.kernels import get_kernel

_GYM_ID_TEMPLATE = "Velgate-Bench-UR3e-HandE-Lift-{kernel_display}-v0"

_KERNEL_DISPLAY = {
    "tanh": "Tanh",
    "gaussian": "Gaussian",
    "velocity_gated_tanh": "VelocityGatedTanh",
    "tanh_additive_velpen": "TanhAdditiveVelPen",
    "velocity_gated_tanh_smooth": "VelocityGatedTanhSmooth",
}


def _make_ur3e_lift_cfg(kernel_name: str):
    """Factory returning a configclass subclass of UR3eHandECubeLiftEnvCfg
    with the requested fine-grained kernel installed.
    """

    kernel_func, kernel_default_kwargs = get_kernel(kernel_name)

    @configclass
    class _Cfg(UR3eHandECubeLiftEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.rewards.object_goal_tracking_fine_grained.func = kernel_func
            new_params = {
                "std": 0.05,
                "minimal_height": 0.04,
                "command_name": "object_pose",
                "robot_cfg": SceneEntityCfg("robot", joint_names=UR3E_ARM_JOINT_NAMES),
                "object_cfg": SceneEntityCfg("object"),
            }
            new_params.update(kernel_default_kwargs)
            self.rewards.object_goal_tracking_fine_grained.params = new_params

    _Cfg.__name__ = f"UR3eHandELift{_KERNEL_DISPLAY[kernel_name]}EnvCfg"
    _Cfg.__qualname__ = _Cfg.__name__
    return _Cfg


def register_ur3e_lift_variants() -> None:
    """Register Velgate-Bench-UR3e-HandE-Lift-{Tanh,Gaussian,VelocityGatedTanh}-v0."""
    for kernel_name, kernel_display in _KERNEL_DISPLAY.items():
        gym_id = _GYM_ID_TEMPLATE.format(kernel_display=kernel_display)
        if gym_id in gym.registry:
            continue
        cfg_cls = _make_ur3e_lift_cfg(kernel_name)
        globals()[cfg_cls.__name__] = cfg_cls
        gym.register(
            id=gym_id,
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            kwargs={
                "env_cfg_entry_point": f"{__name__}:{cfg_cls.__name__}",
                "skrl_cfg_entry_point": (
                    "isaac_robots.tasks.manager_based.lift.config.ur3e_hande.agents:skrl_ppo_cfg.yaml"
                ),
            },
            disable_env_checker=True,
        )
