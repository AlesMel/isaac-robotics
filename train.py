from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Dict as DictSpace
from gymnasium.spaces.utils import flatdim
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer


def bootstrap_isaaclab(cli_args: argparse.Namespace):
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Lab is not available in this Python environment. Run this script with the Isaac Lab/Isaac Sim Python launcher."
        ) from exc

    app_launcher = AppLauncher(cli_args)
    return app_launcher.app


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


def build_sensor_cfg(args: argparse.Namespace, sensor_selection_cls):
    return sensor_selection_cls(
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


def make_env(sensor_cfg, num_envs: int, sim_device: str) -> gym.Env:
    from crazyflie_lab import ENV_ID, register_env
    from crazyflie_lab.envs import CrazyflieEnvCfg

    register_env()
    cfg = CrazyflieEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.sensor_selection = sensor_cfg
    cfg.sim.device = sim_device
    if str(sim_device).startswith("cpu"):
        cfg.sim.use_fabric = False
    cfg.__post_init__()
    return gym.make(ENV_ID, cfg=cfg)


def configure_experiment_logging(cfg: dict, args: argparse.Namespace) -> dict:
    def _parse_interval(value: str) -> int | str:
        return value if value == "auto" else int(value)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["write_interval"] = _parse_interval(args.tensorboard_write_interval)
    cfg["experiment"]["checkpoint_interval"] = _parse_interval(args.checkpoint_interval)
    cfg["experiment"]["directory"] = str(Path(args.log_dir))
    cfg["experiment"]["experiment_name"] = args.run_name or f"{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return cfg


def maybe_launch_tensorboard(args: argparse.Namespace, log_dir: Path) -> subprocess.Popen | None:
    if not args.tensorboard:
        return None

    command = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(log_dir),
        "--port",
        str(args.tensorboard_port),
        "--host",
        args.tensorboard_host,
    ]
    try:
        process = subprocess.Popen(command)
    except Exception as exc:
        print(f"[train] Failed to launch TensorBoard automatically: {exc}", flush=True)
        return None

    print(
        f"[train] TensorBoard started at http://{args.tensorboard_host}:{args.tensorboard_port} for {log_dir}",
        flush=True,
    )
    return process


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
    return cfg
 

def make_ppo_agent_from_cfg(env, memory, device, cfg):
    models = {
        "policy": GaussianPolicy(env.observation_space, env.action_space, device),
        "value": ValueModel(env.observation_space, env.action_space, device),
    }
    return PPO(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def make_sac_agent(env, memory, device):
    cfg = deepcopy(SAC_DEFAULT_CONFIG)
    cfg["batch_size"] = 256
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": flatdim(env.observation_space), "device": device}
    return cfg


def make_sac_agent_from_cfg(env, memory, device, cfg):
    models = {
        "policy": GaussianPolicy(env.observation_space, env.action_space, device),
        "critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
    }
    return SAC(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def make_td3_agent(env, memory, device):
    cfg = deepcopy(TD3_DEFAULT_CONFIG)
    cfg["batch_size"] = 256
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": flatdim(env.observation_space), "device": device}
    return cfg


def make_td3_agent_from_cfg(env, memory, device, cfg):
    models = {
        "policy": DeterministicActor(env.observation_space, env.action_space, device),
        "target_policy": DeterministicActor(env.observation_space, env.action_space, device),
        "critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_1": DeterministicCritic(env.observation_space, env.action_space, device),
        "target_critic_2": DeterministicCritic(env.observation_space, env.action_space, device),
    }
    return TD3(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space, action_space=env.action_space, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Crazyflie Isaac Lab agents with optional sensors")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    except ImportError:
        pass
    parser.add_argument("--algorithm", choices=("ppo", "sac", "td3"), default="ppo")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--memory-size", type=int, default=1024)
    parser.add_argument("--rl-device", type=str, default=None)
    parser.add_argument("--enable-lidar", action="store_true")
    parser.add_argument("--enable-camera", action="store_true")
    parser.add_argument("--lidar-channels", type=int, default=16)
    parser.add_argument("--lidar-horizontal-rays", type=int, default=180)
    parser.add_argument("--camera-width", type=int, default=84)
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--tensorboard", action="store_true", help="Launch TensorBoard alongside training")
    parser.add_argument("--tensorboard-host", type=str, default="127.0.0.1")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--tensorboard-write-interval", type=str, default="auto")
    parser.add_argument("--checkpoint-interval", type=str, default="auto")
    parser.add_argument("--log-dir", type=str, default="runs/torch/crazyflie")
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def _device_flag_was_provided(argv: list[str]) -> bool:
    return any(arg == "--device" or arg.startswith("--device=") for arg in argv)


def main() -> None:
    args = parse_args()
    if _device_flag_was_provided(sys.argv[1:]):
        sim_device = getattr(args, "device", "cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        sim_device = "cpu"
    rl_device = args.rl_device if args.rl_device is not None else sim_device
    print("[train] Bootstrapping Isaac Lab...", flush=True)
    simulation_app = bootstrap_isaaclab(args)
    tensorboard_process = None

    try:
        from crazyflie_lab.config import SensorSelectionCfg

        print("[train] Building sensor configuration...", flush=True)
        sensor_cfg = build_sensor_cfg(args, SensorSelectionCfg)
        print(f"[train] Simulation device: {sim_device}", flush=True)
        print(f"[train] RL device: {rl_device}", flush=True)
        if not _device_flag_was_provided(sys.argv[1:]):
            print("[train] No --device provided; defaulting simulation to CPU for safer startup.", flush=True)
        print(f"[train] Creating environment with {args.num_envs} env(s)...", flush=True)
        env = make_env(sensor_cfg, num_envs=args.num_envs, sim_device=sim_device)
        modality_dims = compute_modality_dims(env.unwrapped.observation_space)
        flat_obs_dim = sum(modality_dims.values())
        print(f"[train] Active observation modalities: {modality_dims}", flush=True)
        print(f"[train] Flattened policy input dimension: {flat_obs_dim}", flush=True)

        print("[train] Wrapping environment for skrl...", flush=True)
        wrapped_env = wrap_for_skrl(env)
        print(f"[train] Allocating replay/rollout memory on {rl_device}...", flush=True)
        memory = RandomMemory(memory_size=args.memory_size, num_envs=args.num_envs, device=rl_device)

        print(f"[train] Creating {args.algorithm.upper()} agent...", flush=True)
        log_dir = Path(args.log_dir)
        if args.algorithm == "ppo":
            agent_cfg = configure_experiment_logging(make_ppo_agent(wrapped_env, memory, rl_device), args)
            agent = make_ppo_agent_from_cfg(wrapped_env, memory, rl_device, agent_cfg)
        elif args.algorithm == "sac":
            agent_cfg = configure_experiment_logging(make_sac_agent(wrapped_env, memory, rl_device), args)
            agent = make_sac_agent_from_cfg(wrapped_env, memory, rl_device, agent_cfg)
        else:
            agent_cfg = configure_experiment_logging(make_td3_agent(wrapped_env, memory, rl_device), args)
            agent = make_td3_agent_from_cfg(wrapped_env, memory, rl_device, agent_cfg)

        print(f"[train] skrl logs/checkpoints directory: {Path(args.log_dir)}", flush=True)
        tensorboard_process = maybe_launch_tensorboard(args, log_dir)

        print(f"[train] Starting training for {args.timesteps} timesteps...", flush=True)
        trainer = SequentialTrainer(cfg={"timesteps": args.timesteps, "headless": getattr(args, "headless", True)}, env=wrapped_env, agents=agent)
        trainer.train()
        print("[train] Training finished.", flush=True)
    except Exception:
        print("[train] Training failed with an exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        if tensorboard_process is not None and tensorboard_process.poll() is None:
            tensorboard_process.terminate()
        print("[train] Closing simulation app.", flush=True)
        simulation_app.close()


if __name__ == "__main__":
    main()
