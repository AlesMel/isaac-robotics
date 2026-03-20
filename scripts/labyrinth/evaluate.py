"""Evaluate a trained labyrinth policy across challenges and difficulty levels.

Runs N episodes per (challenge, difficulty) pair and reports a summary table.

Usage:
    python scripts/labyrinth/evaluate.py --checkpoint /path/to/best_agent.pt
    python scripts/labyrinth/evaluate.py --checkpoint /path/to/best_agent.pt --challenges corridor gate_slalom
    python scripts/labyrinth/evaluate.py --checkpoint /path/to/best_agent.pt --difficulties 0.3 0.5 0.7 --csv results.csv
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate labyrinth policy across challenges and difficulties.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained agent checkpoint (.pt).")
parser.add_argument(
    "--challenges", type=str, nargs="+",
    default=["corridor", "gate_slalom", "pillar_forest", "vertical_layers", "room_maze"],
    help="Challenge types to evaluate.",
)
parser.add_argument(
    "--difficulties", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
    help="Difficulty levels to evaluate.",
)
parser.add_argument("--episodes", type=int, default=100, help="Episodes per (challenge, difficulty) pair.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments for evaluation.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path.")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import itertools
import os

import gymnasium as gym
import torch

import skrl
from skrl.utils.runner.torch import Runner

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

from common import CHALLENGE_TO_TASK


def evaluate_single(task_id: str, difficulty: float, checkpoint: str, num_envs: int,
                    episodes: int, seed: int) -> dict:
    """Run evaluation for one (task, difficulty) pair and return aggregated metrics."""
    # Load env + agent configs via hydra
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=num_envs)
    env_cfg.labyrinth.difficulty = difficulty
    env_cfg.labyrinth.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed

    env = gym.make(task_id, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # Load agent config from gym spec
    spec = gym.spec(task_id)
    agent_cfg_path = spec.kwargs.get("skrl_cfg_entry_point", "")
    if agent_cfg_path:
        from isaaclab_tasks.utils import load_cfg_from_registry
        agent_cfg = load_cfg_from_registry(task_id, "skrl_cfg_entry_point")
    else:
        raise RuntimeError(f"No skrl_cfg_entry_point for {task_id}")

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, agent_cfg)
    runner.agent.load(os.path.abspath(checkpoint))
    runner.agent.set_running_mode("eval")

    # Collect metrics
    collected_episodes = 0
    metrics_accum = {
        "success_rate": 0.0,
        "rings_completed": 0.0,
        "episode_length": 0.0,
        "collision_count": 0,
        "timeout_count": 0,
        "total_reward": 0.0,
    }

    obs, _ = env.reset()
    episode_rewards = torch.zeros(num_envs, device=env.device)
    max_steps = int(episodes * 1500 / num_envs) + 500  # generous step budget

    for step in range(max_steps):
        if collected_episodes >= episodes:
            break

        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, infos = env.step(actions)

        episode_rewards += rewards.squeeze()
        dones = (terminated | truncated).squeeze()

        if dones.any():
            done_count = dones.sum().item()
            collected_episodes += done_count
            episode_rewards[dones] = 0.0

            # Extract logged metrics from env extras
            log = getattr(env.unwrapped, "extras", {}).get("log", {})
            if log:
                metrics_accum["success_rate"] += log.get("Metrics/success_rate", 0.0) * done_count
                metrics_accum["rings_completed"] += log.get("Metrics/rings_completed", 0.0) * done_count
                metrics_accum["episode_length"] += log.get("Metrics/episode_length", 0.0) * done_count
                metrics_accum["collision_count"] += log.get("Episode_Termination/collision", 0)
                metrics_accum["timeout_count"] += log.get("Episode_Termination/time_out", 0)

    env.close()

    n = max(collected_episodes, 1)
    return {
        "success_rate": metrics_accum["success_rate"] / n,
        "rings_completed": metrics_accum["rings_completed"] / n,
        "episode_length": metrics_accum["episode_length"] / n,
        "collision_rate": metrics_accum["collision_count"] / n,
        "timeout_rate": metrics_accum["timeout_count"] / n,
        "episodes": collected_episodes,
    }


def main():
    results = {}

    for challenge, difficulty in itertools.product(args_cli.challenges, sorted(args_cli.difficulties)):
        task_id = CHALLENGE_TO_TASK[challenge]
        print(f"\n--- Evaluating {challenge} @ difficulty={difficulty} ---")
        metrics = evaluate_single(
            task_id, difficulty, args_cli.checkpoint,
            args_cli.num_envs, args_cli.episodes, args_cli.seed,
        )
        results[(challenge, difficulty)] = metrics
        print(f"  success={metrics['success_rate']:.2%}  rings={metrics['rings_completed']:.1f}  "
              f"len={metrics['episode_length']:.0f}  collision={metrics['collision_rate']:.2%}  "
              f"timeout={metrics['timeout_rate']:.2%}  ({metrics['episodes']} eps)")

    # Print summary table
    print(f"\n{'='*90}")
    print(f"{'Challenge':<18} {'Diff':>5} {'Success':>8} {'Rings':>6} {'Length':>7} {'Collide':>8} {'Timeout':>8}")
    print(f"{'-'*90}")
    for (challenge, difficulty), m in results.items():
        print(f"{challenge:<18} {difficulty:>5.1f} {m['success_rate']:>7.1%} {m['rings_completed']:>6.1f} "
              f"{m['episode_length']:>7.0f} {m['collision_rate']:>7.1%} {m['timeout_rate']:>7.1%}")
    print(f"{'='*90}")

    # Optional CSV export
    if args_cli.csv:
        with open(args_cli.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["challenge", "difficulty", "success_rate", "rings_completed",
                             "episode_length", "collision_rate", "timeout_rate", "episodes"])
            for (challenge, difficulty), m in results.items():
                writer.writerow([challenge, difficulty, f"{m['success_rate']:.4f}",
                                 f"{m['rings_completed']:.2f}", f"{m['episode_length']:.1f}",
                                 f"{m['collision_rate']:.4f}", f"{m['timeout_rate']:.4f}",
                                 m["episodes"]])
        print(f"\nResults saved to {args_cli.csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
