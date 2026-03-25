import gymnasium as gym

from . import agents

_ENTRY = f"{__name__}.isaac_racing_env:RacingDirectEnv"
_SKRL = f"{agents.__name__}:skrl_ppo_cfg.yaml"

gym.register(
    id="Isaac-Robots-Racing-Direct-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_racing_env_cfg:RacingEnvCfg",
        "skrl_cfg_entry_point": _SKRL,
    },
)

gym.register(
    id="Isaac-Robots-Racing-Direct-Hard-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_racing_env_cfg:RacingEnvCfgHard",
        "skrl_cfg_entry_point": _SKRL,
    },
)

gym.register(
    id="Isaac-Robots-Racing-Direct-VeryHard-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_racing_env_cfg:RacingEnvCfgVeryHard",
        "skrl_cfg_entry_point": _SKRL,
    },
)
