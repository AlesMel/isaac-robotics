"""Gym registrations for the UR3e + Robotiq 2F-140 lift-cube task.

Duplicated from the ur3e_2f_85 package. Same arm/task, 2F-140 gripper.
"""

import gymnasium as gym

from . import joint_pos_env_cfg
from . import agents  # noqa: F401  # for cfg entry-point references below

##
# Joint-position control (RL training)
##
gym.register(
    id="Isaac-Lift-Cube-UR3e-2F140-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3e2F140CubeLiftEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-UR3e-2F140-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3e2F140CubeLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
