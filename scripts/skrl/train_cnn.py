"""DEBUG ONLY: Minimal skrl PPO + camera — for isolating the 'orthonormal' warning.

WARNING: This script uses flat MLP models (no CNN) despite enabling the camera.
The full camera observation (proprio + H*W*frame_stack) is fed as a flat vector
into a simple MLP, which will NOT learn meaningful visual features. This script
exists solely to debug Isaac Sim camera-related warnings.

For actual CNN training, use:
    python scripts/labyrinth/train_curriculum.py --challenge corridor --num_envs 4096

If the warning still appears here, the issue is SkrlVecEnvWrapper.
If not, add back features one-by-one to find the culprit:
  1. state_space (asymmetric actor-critic)
  2. RunningStandardScaler
  3. CNN models
  4. KLAdaptiveLR

Usage:
    python scripts/skrl/train_cnn.py --num_envs 64 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal skrl PPO camera test.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Robots-Labyrinth-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
from datetime import datetime
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# ── Trivial MLP models (no CNN, no asymmetric) ──────────────────────────────

class SimplePolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-20.0, max_log_std=2.0)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class SimpleValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, **kwargs):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Enable 64x64 camera
    from isaac_robots.tasks.direct.isaac_labyrinth.cfg import CRAZYFLIE_AI_CAMERA_64_CFG
    env_cfg.camera = CRAZYFLIE_AI_CAMERA_64_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body/ai_camera"
    )
    env_cfg.observation_space = 12 + env_cfg.frame_stack * 64 * 64

    # Force render every sim step when recording video
    if args_cli.video:
        env_cfg.sim.render_interval = 1

    # Create env + skrl wrapper
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_cnn"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)
    device = env.device

    print(f"[INFO] obs={env.observation_space.shape}, act={env.action_space.shape}, "
          f"num_envs={args_cli.num_envs}")

    # Simple symmetric models (critic sees same obs as actor)
    models = {
        "policy": SimplePolicy(env.observation_space, env.action_space, device),
        "value": SimpleValue(env.observation_space, env.action_space, device),
    }

    memory = RandomMemory(memory_size=48, num_envs=env.num_envs, device=device)

    # Bare-minimum PPO — no preprocessors, no scheduler
    cfg = PPO_CFG(
        rollouts=48,
        learning_epochs=8,
        mini_batches=4,
        discount_factor=0.99,
        lambda_=0.95,
        learning_rate=3e-4,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.01,
        value_loss_scale=1.0,
        experiment={
            "directory": log_root_path,
            "experiment_name": log_dir,
            "write_interval": "auto",
            "checkpoint_interval": "auto",
        },
    )

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    print("[INFO] Starting minimal skrl PPO — watch for 'orthonormal' warnings...")
    trainer = SequentialTrainer(
        cfg={"timesteps": args_cli.max_iterations * 48, "close_environment_at_exit": False},
        env=env,
        agents=agent,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
