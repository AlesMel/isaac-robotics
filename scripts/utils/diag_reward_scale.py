# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose reward-scale / NaN blow-up for the UR3e-2F85 lift task.

Drives the env with actions of *escalating* magnitude (sigma = 0.5, 1, 2, 4, 8)
to mimic a policy whose std runs away. At every step it records:
  * raw (unweighted) value of each reward term, via the reward manager
  * the post-curriculum WEIGHTED penalty (action_rate, joint_vel) at weight -0.1
  * whether obs / reward / joint_vel / object state contain any non-finite value

Writes a summary to /tmp/reward_scale_result.txt. Headless, no rendering.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-UR3e-2F85-v0")
parser.add_argument("--steps_per_sigma", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import isaac_robots.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def finite_report(name, t):
    bad = ~torch.isfinite(t)
    if bad.any():
        return f"NON-FINITE in {name}: {int(bad.sum())} elems, sample max|.|={t[torch.isfinite(t)].abs().max().item():.4g}"
    return None


def main():
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=64)
    env = gym.make(args_cli.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    obj = base.scene["object"]
    rm = base.reward_manager

    lines = [f"task={args_cli.task}  num_envs={base.num_envs}  dt(step)={base.step_dt}"]
    lines.append(f"active reward terms: {rm.active_terms}")
    lines.append("")

    n_act = base.action_manager.total_action_dim
    first_nonfinite = None

    for sigma in (0.5, 1.0, 2.0, 4.0, 8.0):
        env.reset()
        prev = torch.zeros((base.num_envs, n_act), device=base.device)
        # accumulators for raw term values + key scalars
        term_max = {}
        ar_max = jv_max = spd_max = 0.0
        for t in range(args_cli.steps_per_sigma):
            act = sigma * torch.randn((base.num_envs, n_act), device=base.device)
            obs, rew, term, trunc, info = env.step(act)

            # --- non-finite hunt ---
            policy_obs = obs["policy"] if isinstance(obs, dict) else obs
            for nm, tns in (("obs", policy_obs), ("reward", rew),
                            ("joint_vel", robot.data.joint_vel),
                            ("object_pos", obj.data.root_pos_w),
                            ("object_vel", obj.data.root_lin_vel_w)):
                msg = finite_report(nm, tns)
                if msg and first_nonfinite is None:
                    first_nonfinite = f"sigma={sigma} step={t}: {msg}"

            # --- raw per-term reward magnitudes (unweighted), if the manager exposes them ---
            step_terms = getattr(rm, "_step_reward", None)
            if isinstance(step_terms, dict):
                for name, val in step_terms.items():
                    term_max[name] = max(term_max.get(name, 0.0), val.abs().max().item())

            # --- recompute the two penalties directly (raw, unweighted) ---
            ar = torch.sum(torch.square(act - prev), dim=1)
            jv = torch.sum(torch.square(robot.data.joint_vel), dim=1)
            spd = torch.norm(obj.data.root_lin_vel_w, dim=-1)
            ar_max = max(ar_max, ar.max().item())
            jv_max = max(jv_max, jv.max().item())
            spd_max = max(spd_max, spd.max().item())
            prev = act

        lines.append(f"--- sigma={sigma} ---")
        lines.append(f"  max action_rate_l2 (raw)   = {ar_max:.4g}   -> weighted@-0.1 = {-0.1*ar_max:.4g} / step")
        lines.append(f"  max joint_vel_l2  (raw)    = {jv_max:.4g}   -> weighted@-0.1 = {-0.1*jv_max:.4g} / step")
        lines.append(f"  max object speed (m/s)     = {spd_max:.4g}")
        if term_max:
            lines.append(f"  per-term max|raw|: " + ", ".join(f"{k}={v:.3g}" for k, v in term_max.items()))
        lines.append("")

    lines.append("========== FIRST NON-FINITE ==========")
    lines.append(first_nonfinite if first_nonfinite else "(none observed in this run)")

    with open("/tmp/reward_scale_result.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
