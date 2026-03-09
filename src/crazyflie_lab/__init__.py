from __future__ import annotations

import gymnasium as gym

from crazyflie_lab.envs.crazyflie_env import CrazyflieDirectEnv, CrazyflieEnvCfg


ENV_ID = "Isaac-Crazyflie-Direct-v0"


def register_env() -> None:
    if ENV_ID in gym.registry:
        return

    gym.register(
        id=ENV_ID,
        entry_point="crazyflie_lab.envs:CrazyflieDirectEnv",
        kwargs={"cfg": CrazyflieEnvCfg()},
        disable_env_checker=True,
    )


register_env()

__all__ = ["CrazyflieDirectEnv", "CrazyflieEnvCfg", "ENV_ID", "register_env"]
