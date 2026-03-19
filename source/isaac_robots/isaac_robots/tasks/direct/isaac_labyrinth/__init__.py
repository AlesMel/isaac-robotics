import gymnasium as gym

from . import agents

_ENTRY = f"{__name__}.isaac_labyrinth_env:LabyrinthDirectEnv"
_SKRL = f"{agents.__name__}:skrl_ppo_cfg.yaml"
_SKRL_SAC = f"{agents.__name__}:skrl_sac_cfg.yaml"

gym.register(
    id="Isaac-Robots-Labyrinth-Direct-v0",
    entry_point=_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.isaac_labyrinth_env_cfg:LabyrinthEnvCfg",
        "skrl_cfg_entry_point": _SKRL,
        "skrl_sac_cfg_entry_point": _SKRL_SAC,
    },
)
