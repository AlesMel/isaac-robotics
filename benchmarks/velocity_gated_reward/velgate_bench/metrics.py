# Copyright (c) 2026, Ales Melichar
# SPDX-License-Identifier: BSD-3-Clause

"""Eval-rollout metrics for the velocity-gated reward benchmark.

The TB scalars from training give us peak/final reward and per-term value
trajectories, but the **goal-hold quality** (joint velocity at goal, EE
position variance during hold, etc.) requires deterministic eval rollouts
on a trained checkpoint. This module provides that infrastructure.

Usage (called from sweep.py or as standalone)::

    python -m velgate_bench.eval --task <gym_id> --checkpoint <path.pt> \\
        --num_envs 64 --num_episodes 100 --out <output.json>

Output is a JSON file with per-episode and per-run aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This module must be importable without Isaac Sim. The actual eval entry
# point launches Isaac Sim and runs rollouts; analyse-only consumers
# (analyze.py) only import the dataclass + I/O helpers.


# -----------------------------------------------------------------------------
# Metric definitions (pure dataclass, no Isaac Sim dependency)
# -----------------------------------------------------------------------------


def aggregate_metrics(per_episode: list[dict]) -> dict:
    """Aggregate per-episode metrics into run-level statistics.

    Returns dict with mean / std / median / q25 / q75 for each metric, plus
    the per-episode arrays for downstream analysis.
    """
    import numpy as np

    keys = per_episode[0].keys() if per_episode else []
    out = {"per_episode": per_episode, "n_episodes": len(per_episode)}
    for k in keys:
        vals = np.asarray([ep[k] for ep in per_episode], dtype=float)
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        out[f"{k}_median"] = float(np.median(vals))
        out[f"{k}_q25"] = float(np.quantile(vals, 0.25))
        out[f"{k}_q75"] = float(np.quantile(vals, 0.75))
    return out


# -----------------------------------------------------------------------------
# Eval entry point (only runs when this module is invoked directly with
# Isaac Sim available; not imported at analyze-time)
# -----------------------------------------------------------------------------


def run_eval(
    task: str,
    checkpoint: str,
    num_envs: int = 64,
    num_episodes: int = 100,
    out: str | None = None,
    seed: int = 0,
    arm_joint_pattern: str = ".*",
) -> dict:
    """Run deterministic eval rollouts and compute hold metrics.

    Returns the aggregated metrics dict. Also writes JSON to ``out`` if given.

    Metrics computed per episode (over the final ``hold_window_s`` seconds):

    - ``peak_reward``: peak episode reward seen in this rollout
    - ``final_reward``: episode total reward
    - ``success``: 1 if cube within ``success_eps`` of goal at episode end
    - ``hold_joint_vel_l2``: RMS arm joint velocity during hold window
    - ``hold_ee_z_std``: std of EE z-coordinate during hold window
    - ``hold_obj_goal_dist_mean``: mean cube-goal distance during hold window
    - ``hold_obj_goal_dist_std``: std of cube-goal distance during hold window
    - ``time_at_goal_s``: cumulative seconds cube was within ``goal_eps`` of target
    """
    from isaaclab.app import AppLauncher  # noqa: F401 - guard import

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app  # noqa: F841

    try:
        return _run_eval_impl(
            task=task,
            checkpoint=checkpoint,
            num_envs=num_envs,
            num_episodes=num_episodes,
            out=out,
            seed=seed,
            arm_joint_pattern=arm_joint_pattern,
        )
    finally:
        simulation_app.close()


def _run_eval_impl(
    task: str,
    checkpoint: str,
    num_envs: int,
    num_episodes: int,
    out: str | None,
    seed: int,
    arm_joint_pattern: str,
) -> dict:
    """Inner eval body (after Isaac Sim is launched)."""
    import re

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab.utils.math import combine_frame_transforms
    from isaaclab_tasks.utils import parse_env_cfg
    from skrl.utils.runner.torch import Runner

    # Trigger gym registration of this benchmark's variants
    import velgate_bench.envs  # noqa: F401

    env_cfg = parse_env_cfg(task, device="cuda", num_envs=num_envs)
    env = gym.make(task, cfg=env_cfg, render_mode=None)

    # Load policy via the skrl Runner (mirrors play.py logic)
    runner_cfg_path = gym.spec(task).kwargs["skrl_cfg_entry_point"]
    from isaaclab_tasks.utils import load_cfg_from_registry

    runner_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    runner = Runner(env, runner_cfg)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")

    # Resolve arm joint ids for velocity metric (regex against joint names)
    robot = env.unwrapped.scene["robot"]
    arm_re = re.compile(arm_joint_pattern)
    arm_joint_ids = [i for i, n in enumerate(robot.joint_names) if arm_re.match(n)]

    # Eval loop: collect per-episode metrics across `num_episodes` total
    hold_window_s = 2.0
    decimation = env_cfg.decimation
    sim_dt = env_cfg.sim.dt
    step_dt = decimation * sim_dt
    hold_steps = int(hold_window_s / step_dt)
    max_steps = int(env_cfg.episode_length_s / step_dt)
    success_eps = 0.03  # m, cube within 3cm of goal at end = success
    goal_eps = 0.05  # m, "at goal" means within 5cm

    per_episode_buf: list[list[dict]] = [[] for _ in range(num_envs)]
    obs_traj: list[torch.Tensor] = []
    completed: list[dict] = []

    obs, _ = env.reset(seed=seed)

    while len(completed) < num_episodes:
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
        obs, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated

        # Per-step instrumentation (vectorised across envs)
        with torch.inference_mode():
            scene = env.unwrapped.scene
            robot_data = scene["robot"].data
            obj_data = scene["object"].data
            ee_frame = scene["ee_frame"]
            command = env.unwrapped.command_manager.get_command("object_pose")
            des_pos_b = command[:, :3]
            des_pos_w, _ = combine_frame_transforms(
                robot_data.root_pos_w, robot_data.root_quat_w, des_pos_b
            )
            distance = torch.norm(des_pos_w - obj_data.root_pos_w, dim=1)
            joint_vels = robot_data.joint_vel[:, arm_joint_ids]
            joint_vel_mag = torch.norm(joint_vels, dim=1)
            ee_z = ee_frame.data.target_pos_w[..., 0, 2]

            for env_id in range(num_envs):
                per_episode_buf[env_id].append(
                    {
                        "reward": float(rewards[env_id]),
                        "distance": float(distance[env_id]),
                        "joint_vel_mag": float(joint_vel_mag[env_id]),
                        "ee_z": float(ee_z[env_id]),
                    }
                )

        # Episode boundaries: aggregate, then clear that env's buffer
        for env_id in range(num_envs):
            if not done[env_id]:
                continue
            traj = per_episode_buf[env_id]
            if not traj:
                continue
            ep_metrics = _episode_metrics(
                traj=traj,
                hold_steps=hold_steps,
                success_eps=success_eps,
                goal_eps=goal_eps,
                step_dt=step_dt,
            )
            completed.append(ep_metrics)
            per_episode_buf[env_id] = []
            if len(completed) >= num_episodes:
                break

    env.close()

    agg = aggregate_metrics(completed[:num_episodes])
    agg["task"] = task
    agg["checkpoint"] = str(checkpoint)
    agg["seed"] = seed

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(agg, indent=2))

    return agg


def _episode_metrics(
    traj: list[dict],
    hold_steps: int,
    success_eps: float,
    goal_eps: float,
    step_dt: float,
) -> dict:
    """Compute the per-episode summary metrics from a step-by-step trajectory."""
    import numpy as np

    rewards = np.asarray([s["reward"] for s in traj], dtype=float)
    distances = np.asarray([s["distance"] for s in traj], dtype=float)
    joint_vels = np.asarray([s["joint_vel_mag"] for s in traj], dtype=float)
    ee_zs = np.asarray([s["ee_z"] for s in traj], dtype=float)

    hold = traj[-min(hold_steps, len(traj)):]
    hold_d = np.asarray([s["distance"] for s in hold], dtype=float)
    hold_jv = np.asarray([s["joint_vel_mag"] for s in hold], dtype=float)
    hold_z = np.asarray([s["ee_z"] for s in hold], dtype=float)

    success = float(distances[-1] < success_eps)
    time_at_goal = float(np.sum(distances < goal_eps) * step_dt)

    return {
        "peak_reward": float(np.max(rewards)),
        "final_reward": float(np.sum(rewards)),
        "success": success,
        "hold_joint_vel_l2": float(np.sqrt(np.mean(hold_jv**2))),
        "hold_ee_z_std": float(np.std(hold_z)),
        "hold_obj_goal_dist_mean": float(np.mean(hold_d)),
        "hold_obj_goal_dist_std": float(np.std(hold_d)),
        "time_at_goal_s": time_at_goal,
        "episode_len_steps": len(traj),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic eval rollouts for velgate benchmark")
    parser.add_argument("--task", required=True, help="Gym ID of the env to evaluate")
    parser.add_argument("--checkpoint", required=True, help="Path to skrl agent checkpoint")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arm_joint_pattern", default=".*",
                        help="Regex to filter arm joints for joint_vel metric (default: all joints)")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args(argv)
    run_eval(
        task=args.task,
        checkpoint=args.checkpoint,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        out=args.out,
        seed=args.seed,
        arm_joint_pattern=args.arm_joint_pattern,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
