"""Play a distillation-trained camera CNN student policy in the racing environment.

The checkpoint must have been produced by train_distill_racing.py (NatureCnnPolicy student).
It cannot be played with play_racing.py because that script rebuilds the flat-MLP
architecture from the YAML config, which won't match the CNN student's weights.

Usage:
    python scripts/skrl/play_distill_racing.py \\
        --checkpoint logs/skrl/racing_distill/<run>/checkpoints/best_agent.pt \\
        --task Isaac-Robots-Racing-Direct-v0 --num_envs 4
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a distillation-trained CNN student (racing).")
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to the student best_agent.pt produced by train_distill_racing.py.")
parser.add_argument("--task", type=str, default="Isaac-Robots-Racing-Direct-v0",
                    help="Racing task ID (difficulty variant).")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--real-time", action="store_true", default=False,
                    help="Sleep between steps to match sim dt.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler
from isaaclab.utils.dict import print_dict
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import isaac_robots.tasks  # noqa: F401

from isaac_robots.tasks.direct.isaac_labyrinth.agents.cnn_mlp_model import (
    NatureCnnPolicy,
    PROPRIO_DIM,
    FRAME_STACK,
    CAM_H,
    CAM_W,
)
from isaac_robots.tasks.direct.isaac_racing.cfg import CRAZYFLIE_AI_CAMERA_64_CFG


def main():
    device_str = args_cli.device or "cuda"
    device = torch.device(device_str)

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(args_cli.task, device=device_str, num_envs=args_cli.num_envs)
    env_cfg.camera = CRAZYFLIE_AI_CAMERA_64_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body/ai_camera"
    )
    env_cfg.observation_space = PROPRIO_DIM + FRAME_STACK * CAM_H * CAM_W  # 16396
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        log_dir = os.path.dirname(os.path.dirname(os.path.abspath(args_cli.checkpoint)))
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # ── Student policy ────────────────────────────────────────────────────────
    policy = NatureCnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    ).to(device)

    preprocessor = RunningStandardScaler(
        size=env.observation_space, device=device
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint_path = os.path.abspath(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[INFO] Checkpoint keys: {list(ckpt.keys())}")

    policy.load_state_dict(ckpt["policy"])
    if "state_preprocessor" in ckpt:
        preprocessor.load_state_dict(ckpt["state_preprocessor"])
    else:
        print("[WARN] No 'state_preprocessor' in checkpoint — obs will not be normalized.")

    policy.eval()
    preprocessor.eval()

    # ── Eval loop ─────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs_pre = preprocessor(obs, train=False)
            actions, _, extra = policy.act({"states": obs_pre}, role="policy")
            actions = extra.get("mean_actions", actions)
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
