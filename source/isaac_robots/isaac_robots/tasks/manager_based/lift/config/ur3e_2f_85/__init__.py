"""Gym registrations for the UR3e + Robotiq 2F-85 lift-cube task.
 
Place this package at e.g.
    .../manipulation/lift/config/ur3e_2f85/__init__.py
and make sure the parent `lift/config/__init__.py` imports it (or that this
package is imported somewhere at startup) so the registrations run.
"""
 
import gymnasium as gym
 
from . import ik_rel_env_cfg, joint_pos_env_cfg
from . import agents  # noqa: F401  # for skrl_cfg_entry_point references below

##
# Joint-position control (RL training)
##
gym.register(
    id="Isaac-Lift-Cube-UR3e-2F85-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3e2F85CubeLiftEnvCfg",
        # Add your RL runner cfg entry point, e.g.:
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
 
gym.register(
    id="Isaac-Lift-Cube-UR3e-2F85-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR3e2F85CubeLiftEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
 
##
# Relative-IK control (teleop / data collection)
##
gym.register(
    id="Isaac-Lift-Cube-UR3e-2F85-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR3e2F85CubeLiftEnvCfg_IK_Rel",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)