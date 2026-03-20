"""Train the Isaac Labyrinth task with asymmetric actor-critic.

Drop-in replacement for scripts/skrl/train.py that bypasses SKRL's YAML
model builder and directly instantiates NatureCnnPolicy + MlpCritic.

Asymmetric architecture:
  - Actor:  NatureCNN on 64×64 stacked frames + proprio(12) → Gaussian policy
  - Critic: MLP on privileged state (goal, vel, geodesic, obstacle dist) → value

Usage (same flags as train.py):
    python scripts/skrl/train_labyrinth.py \\
        --task Isaac-Labyrinth-Direct-v0  \\
        --num_envs 512                    \\  # 512 recommended when camera is ON
        --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Isaac Labyrinth with CNN+MLP PPO.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=1000)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-Labyrinth-Direct-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # camera is always used in this script
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
from datetime import datetime

import gymnasium as gym

import skrl
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_robots.tasks  # noqa: F401

from isaac_robots.tasks.direct.isaac_labyrinth.agents.cnn_mlp_model import NatureCnnPolicy, MlpCritic


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    # ── Environment overrides ────────────────────────────────────────────────
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    seed = args_cli.seed if args_cli.seed is not None else agent_cfg.get("seed", 42)
    if seed == -1:
        seed = random.randint(0, 10000)
    agent_cfg["seed"] = seed
    env_cfg.seed = seed

    # ── Logging ──────────────────────────────────────────────────────────────
    exp_dir = agent_cfg["agent"]["experiment"]["directory"]
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", exp_dir))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch_cnn"
    exp_name = agent_cfg["agent"]["experiment"].get("experiment_name", "")
    if exp_name:
        log_dir += f"_{exp_name}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {log_dir}")

    # ── Max iterations ───────────────────────────────────────────────────────
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        )

    # ── Enable camera (64×64, stacked frames) ──────────────────────────────────
    # NOTE: __post_init__ already ran (via hydra_task_config) with camera=None, so
    # cfg.observation_space is stale. Update it here before gym.make reads it.
    from isaac_robots.tasks.direct.isaac_labyrinth.cfg import CRAZYFLIE_AI_CAMERA_64_CFG
    env_cfg.camera = CRAZYFLIE_AI_CAMERA_64_CFG.replace(
        prim_path="/World/envs/env_.*/Robot/body/ai_camera"
    )
    # Actor obs: proprio(12) + stacked frames
    env_cfg.observation_space = 12 + env_cfg.frame_stack * env_cfg.camera.height * env_cfg.camera.width

    # ── Create environment ───────────────────────────────────────────────────
    env_cfg.log_dir = log_dir
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
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

    # Log env/agent config
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ── Models (asymmetric actor-critic) ────────────────────────────────────
    # Actor: NatureCNN on observation_space (proprio + stacked frames)
    policy_model = NatureCnnPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        clip_log_std=True,
        min_log_std=-20.0,
        max_log_std=2.0,
        initial_log_std=float(
            agent_cfg["models"]["policy"].get("initial_log_std", 0.0)
        ),
    )
    # Critic: MLP on state_space (privileged state)
    value_model = MlpCritic(
        observation_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )
    models = {"policy": policy_model, "value": value_model}

    obs_dim = env.observation_space.shape[0]
    state_dim = env.state_space.shape[0] if env.state_space is not None else 0
    print(f"[INFO] Asymmetric actor-critic  (obs_dim={obs_dim}, state_dim={state_dim})")

    # ── Memory ───────────────────────────────────────────────────────────────
    rollouts = agent_cfg["agent"].get("rollouts", 48)
    memory = RandomMemory(
        memory_size=rollouts,  # one slot per rollout step per env
        num_envs=env.num_envs,
        device=device,
    )

    # ── PPO config from YAML ─────────────────────────────────────────────────
    a = agent_cfg["agent"]
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = a.get("rollouts", 48)
    cfg["learning_epochs"] = a.get("learning_epochs", 8)
    cfg["mini_batches"] = a.get("mini_batches", 8)
    cfg["discount_factor"] = a.get("discount_factor", 0.995)
    cfg["lambda"] = a.get("lambda", 0.95)
    cfg["learning_rate"] = a.get("learning_rate", 3e-4)
    cfg["grad_norm_clip"] = a.get("grad_norm_clip", 1.0)
    cfg["ratio_clip"] = a.get("ratio_clip", 0.2)
    cfg["value_clip"] = a.get("value_clip", 0.2)
    cfg["clip_predicted_values"] = a.get("clip_predicted_values", True)
    cfg["entropy_loss_scale"] = a.get("entropy_loss_scale", 0.01)
    cfg["value_loss_scale"] = a.get("value_loss_scale", 2.0)
    cfg["kl_threshold"] = a.get("kl_threshold", 0.0)
    cfg["rewards_shaper"] = lambda r, *_: r * a.get("rewards_shaper_scale", 0.1)
    cfg["time_limit_bootstrap"] = a.get("time_limit_bootstrap", True)
    cfg["random_timesteps"] = a.get("random_timesteps", 0)
    cfg["learning_starts"] = a.get("learning_starts", 0)
    # Learning rate scheduler
    cfg["learning_rate_scheduler"] = KLAdaptiveLR
    kl_kwargs = a.get("learning_rate_scheduler_kwargs", {}) or {}
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": kl_kwargs.get("kl_threshold", 0.008)}
    # Preprocessors: don't normalize actor obs (images are [0,1], proprio is moderate range).
    # Normalize critic's privileged state and value targets.
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.state_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # Experiment
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = log_dir
    write_interval = a["experiment"].get("write_interval", "auto")
    checkpoint_interval = a["experiment"].get("checkpoint_interval", "auto")
    cfg["experiment"]["write_interval"] = write_interval if write_interval != "auto" else 800
    cfg["experiment"]["checkpoint_interval"] = checkpoint_interval if checkpoint_interval != "auto" else 8000

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        state_space=env.state_space,
        device=device,
    )

    # Resume from checkpoint if provided
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer_cfg = {
        "timesteps": agent_cfg["trainer"].get("timesteps", 1_000_000),
        "environment_info": agent_cfg["trainer"].get("environment_info", "log"),
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)
    trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
