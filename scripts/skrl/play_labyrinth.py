"""Play/evaluate a checkpoint trained with train_labyrinth.py.

Drop-in replacement for scripts/skrl/play.py that manually instantiates
CnnMlpSharedModel instead of going through SKRL's Runner (which only supports
its own hardcoded mixin names and cannot load custom Python classes).

Usage:
    python scripts/skrl/play_labyrinth.py \\
        --task Isaac-Labyrinth-Direct-v0 \\
        --checkpoint logs/skrl/labyrinth_direct/<run>/checkpoints/best_agent.pt \\
        --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a labyrinth checkpoint with CNN+MLP PPO.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Labyrinth-Direct-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # camera is always used in this script
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import torch

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401

from isaac_robots.tasks.direct.isaac_labyrinth.agents.cnn_mlp_model import CnnMlpSharedModel


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    # ── Environment ──────────────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 10000)
    env_cfg.seed = seed

    # ── Enable camera (must match training) ───────────────────────────────────
    # NOTE: __post_init__ already ran (via hydra_task_config) with camera=None, so
    # cfg.observation_space is stale. Update it here before gym.make reads it.
    from isaac_robots.tasks.direct.isaac_labyrinth.cfg import CRAZYFLIE_AI_CAMERA_CFG
    env_cfg.camera = CRAZYFLIE_AI_CAMERA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body/ai_camera"
    )
    env_cfg.observation_space += env_cfg.camera.width * env_cfg.camera.height

    # ── Checkpoint path ──────────────────────────────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=".*_ppo_torch_cnn", other_dirs=["checkpoints"]
        )
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # ── Create environment ───────────────────────────────────────────────────
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env)
    device = env.device

    # ── Model ──
    shared_model = CnnMlpSharedModel(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": shared_model, "value": shared_model}

    obs_dim = env.observation_space.shape[0]
    mode = "CNN+MLP (camera)" if obs_dim > 18 else "MLP only (no camera)"
    print(f"[INFO] Model mode: {mode}  (obs_dim={obs_dim})")

    # ── Agent (minimal config for eval — no training needed) ─────────────────
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = 'auto'      # no TensorBoard during play
    cfg["experiment"]["checkpoint_interval"] = 'auto'  # no checkpoints during play
    cfg['trainer']['timesteps'] = 1_000_000
    cfg['trainer']['environment_log'] = "log"

    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=device)
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    agent.load(resume_path)
    agent.set_running_mode("eval")

    # ── Eval loop ────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
