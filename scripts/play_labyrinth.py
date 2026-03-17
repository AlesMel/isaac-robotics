"""Play a trained labyrinth policy using the same renderer path as view_labyrinth.py.

Workaround for RTX 5080 (Blackwell) crash in the standard play.py.

Usage:
    python scripts/play_labyrinth.py --task Isaac-Robots-Corridor-Direct-v0 --checkpoint <path>
    python scripts/play_labyrinth.py --task Isaac-Robots-Corridor-Direct-v0  # auto-find latest
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained labyrinth policy.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to agent.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from skrl.utils.runner.torch import Runner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaac_robots.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    dt = env.unwrapped.step_dt

    # Load agent config from gym spec
    spec = gym.spec(args_cli.task)
    agent_cfg_path = spec.kwargs.get("skrl_cfg_entry_point")

    # Resolve checkpoint
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        # Auto-find latest checkpoint
        from skrl.utils.runner.torch import Runner as _R
        import yaml

        module_name, file_name = agent_cfg_path.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(module_name)
        cfg_path = os.path.join(os.path.dirname(mod.__file__), file_name)
        with open(cfg_path, "r") as f:
            experiment_cfg = yaml.safe_load(f)

        log_root = os.path.join(
            "logs", "skrl",
            experiment_cfg["agent"]["experiment"]["directory"],
        )
        log_root = os.path.abspath(log_root)
        print(f"[INFO] Searching for checkpoints in: {log_root}")
        resume_path = get_checkpoint_path(log_root, other_dirs=["checkpoints"])

    print(f"[INFO] Loading checkpoint: {resume_path}")

    # Build runner + load weights
    module_name, file_name = agent_cfg_path.rsplit(":", 1)
    import importlib, yaml
    mod = importlib.import_module(module_name)
    cfg_path = os.path.join(os.path.dirname(mod.__file__), file_name)
    with open(cfg_path, "r") as f:
        experiment_cfg = yaml.safe_load(f)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    wrapped_env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(wrapped_env, experiment_cfg)
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Play loop
    obs, _ = wrapped_env.reset()
    print("[INFO] Playing. Close the window to quit.")
    while simulation_app.is_running():
        start = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = wrapped_env.step(actions)

        sleep_time = dt - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
