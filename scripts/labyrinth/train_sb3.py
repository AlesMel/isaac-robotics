"""Train the Isaac Labyrinth task with SB3 PPO (camera-based).

Minimal SB3 training script to verify that the camera sensor works
independently of skrl.  If the "matrix may not be orthonormal" warning
disappears here, the issue is skrl-specific.

Usage:
    python scripts/labyrinth/train_sb3.py --num_envs 64 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Isaac Labyrinth with SB3 PPO (camera test).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default="Isaac-Robots-Labyrinth-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401

from isaaclab_tasks.utils.hydra import hydra_task_config

# ── Constants (must match cnn_mlp_model.py) ──────────────────────────────────
PROPRIO_DIM = 12
FRAME_STACK = 4
CAM_H = 64
CAM_W = 64


# ── SB3 feature extractor ───────────────────────────────────────────────────

class CnnProprioExtractor(BaseFeaturesExtractor):
    """NatureCNN on stacked frames + proprio passthrough for SB3."""

    def __init__(self, observation_space, features_dim=524):
        # features_dim is overridden below; pass a placeholder to super
        super().__init__(observation_space, features_dim=1)

        self.cnn = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, FRAME_STACK, CAM_H, CAM_W)).shape[1]

        self.img_proj = nn.Sequential(
            nn.Linear(cnn_out, 512),
            nn.ReLU(),
        )
        self._features_dim = 512 + PROPRIO_DIM

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio = obs[:, :PROPRIO_DIM]
        img = obs[:, PROPRIO_DIM:].view(-1, FRAME_STACK, CAM_H, CAM_W)
        img_feat = self.img_proj(self.cnn(img))
        return torch.cat([img_feat, proprio], dim=-1)


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
    env_cfg.observation_space = PROPRIO_DIM + FRAME_STACK * CAM_H * CAM_W

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Sb3VecEnvWrapper(env)

    # Logging
    log_dir = os.path.join(
        "logs", "sb3", datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_cnn",
    )
    print(f"[INFO] Logging to: {log_dir}")
    print(f"[INFO] obs_dim={env.observation_space.shape}, act_dim={env.action_space.shape}")
    print(f"[INFO] num_envs={args_cli.num_envs}")

    # PPO with CNN feature extractor
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=CnnProprioExtractor,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
        learning_rate=3e-4,
        n_steps=48,
        batch_size=48 * args_cli.num_envs // 4,
        n_epochs=8,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=2.0,
        max_grad_norm=1.0,
        seed=args_cli.seed,
        tensorboard_log=log_dir,
        verbose=1,
        device="cuda",
    )

    print("[INFO] Starting SB3 PPO training — watch for 'orthonormal' warnings...")
    model.learn(total_timesteps=args_cli.max_iterations * 48 * args_cli.num_envs)
    model.save(os.path.join(log_dir, "final_model"))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
