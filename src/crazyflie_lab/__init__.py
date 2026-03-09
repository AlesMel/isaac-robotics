from __future__ import annotations

import gymnasium as gym


ENV_ID = "Isaac-Crazyflie-Direct-v0"


def register_env() -> None:
    if ENV_ID in gym.registry:
        return

    gym.register(
        id=ENV_ID,
        entry_point="crazyflie_lab.envs:CrazyflieDirectEnv",
        disable_env_checker=True,
    )


__all__ = ["ENV_ID", "register_env"]
