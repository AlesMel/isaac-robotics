from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict as DictSpace
from gymnasium.spaces.utils import flatdim
from gymnasium.wrappers import FlattenObservation
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer

from crazyflie_lab import ENV_ID, register_env
from crazyflie_lab.config import SensorSelectionCfg
from crazyflie_lab.envs import CrazyflieEnvCfg


class GaussianPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_units=(256, 256)):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, min_log_std=-5.0, max_log_std=2.0)
        self.net = mlp(flatdim(observation_space), hidden_units, action_space.shape[0])
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space.shape[0]))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_units=(256, 256), clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = mlp(flatdim(observation_space), hidden_units, action_space.shape[0], squash=True)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_units=(256, 256)):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        input_dim = flatdim(observation_space) + action_space.shape[0]
        self.net = mlp(input_dim, hidden_units, 1)

    def compute(self, inputs, role):
        critic_input = torch.cat([inputs["states"], inputs["taken_actions"]], dim=-1)
        return self.net(critic_input), {}


class ValueModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, hidden_units=(256, 256)):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = mlp(flatdim(observation_space), hidden_units, 1)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def mlp(input_dim: int, hidden_units: Tuple[int, ...], output_dim: int, squash: bool = False) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_units:
        layers += [nn.Linear(last_dim, hidden_dim), nn.ELU()]
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if squash:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def build_sensor_cfg(args: argparse.Namespace) -> SensorSelectionCfg:
    return SensorSelectionCfg(
        enable_lidar=args.enable_lidar,
        enable_camera=args.enable_camera,
        lidar_channels=args.lidar_channels,
        lidar_horizontal_rays=args.lidar_horizontal_rays,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )


def compute_modality_dims(space) -> Dict[str, int]:
    if isinstance(space, DictSpace):
        dims: Dict[str, int] = {}
        for key, subspace in space.spaces.items():
            if isinstance(subspace, DictSpace):
                nested_dims = compute_modality_dims(subspace)
                for nested_key, value in nested_dims.items():
                    dims[f"{key}.{nested_key}"] = value
            else:
                dims[key] = flatdim(subspace)
        return dims
    if isinstance(space, Box):
        return {"policy": flatdim(space)}
    raise TypeError(f"Unsupported observation space: {space}")


def wrap_for_skrl(env: gym.Env):
    try:
        return wrap_env(env, wrapper="isaaclab")
    except Exception:
        return wrap_env(env, wrapper="gymnasium")


def make_env(sensor_cfg: SensorSelectionCfg, num_envs: int) -> gym.Env:
    register_env()
    cfg = CrazyflieEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sensor_selection = sensor_cfg
    cfg.__post_init__()
    env = gym.make(ENV_ID, cfg=cfg)
    return FlattenObservation(env)


def make_ppo_agent(env, memory, device):
    models = {
        "policy": GaussianPolicy(env.observation_space, env.action_space, device),
        "value": ValueModel(env.observation_space, env.action_space, device),
    }
    cfg = deepcopy(PPO_DEFAULT_CONFIG)
    cfg["rollouts"] = getattr(memory, "memory_size", 1024)
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": flatdim(env.observation_space), "device": device}
    return PPO(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def make_sac_agent(env, memory, device):
    models = {
        "policy": GaussianPolicy(env.observation_space, env.action_space, device),
        "critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
    }
    cfg = deepcopy(SAC_DEFAULT_CONFIG)
    cfg["batch_size"] = 256
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": flatdim(env.observation_space), "device": device}
    return SAC(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def make_td3_agent(env, memory, device):
    models = {
        "policy": DeterministicActor(env.observation_space, env.action_space, device),
        "target_policy": DeterministicActor(env.observation_space, env.action_space, device),
        "critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
    }
    cfg = deepcopy(TD3_DEFAULT_CONFIG)
    cfg["batch_size"] = 256
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": flatdim(env.observation_space), "device": device}
    return TD3(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Crazyflie Isaac Lab agents with optional sensors")
    parser.add_argument("--algorithm", choices=("ppo", "sac", "td3"), default="ppo")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--memory-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--lidar-channels", type=int, default=16)
    parser.add_argument("--lidar-horizontal-rays", type=int, default=180)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--camera-height", type=int, default=84)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensor_cfg = build_sensor_cfg(args)
    env = make_env(sensor_cfg, num_envs=args.num_envs)
    modality_dims = compute_modality_dims(env.unwrapped.observation_space)
    flat_obs_dim = flatdim(env.observation_space)
    print(f"Active observation modalities: {modality_dims}")
    print(f"Flattened policy input dimension: {flat_obs_dim}")

    wrapped_env = wrap_for_skrl(env)
    memory = RandomMemory(memory_size=args.memory_size, num_envs=args.num_envs, device=args.device)

    if args.algorithm == "ppo":
        agent = make_ppo_agent(wrapped_env, memory, args.device)
    elif args.algorithm == "sac":
        agent = make_sac_agent(wrapped_env, memory, args.device)
    else:
        agent = make_td3_agent(wrapped_env, memory, args.device)

    trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": True}, env=wrapped_env, agents=agent)
    trainer.train()


if __name__ == "__main__":
    main()
