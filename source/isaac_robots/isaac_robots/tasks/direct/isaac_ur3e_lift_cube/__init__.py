# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Robots-UR3e-Lift-Cube-Direct-v0",
    entry_point=f"{__name__}.ur3e_lift_cube_env:UR3eLiftCubeDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3e_lift_cube_env_cfg:UR3eLiftCubeEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Robots-UR3e-Lift-Cube-HandE-Direct-v0",
    entry_point=f"{__name__}.ur3e_lift_cube_env:UR3eLiftCubeDirectEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur3e_lift_cube_env_cfg:UR3eLiftCubeHandEEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
